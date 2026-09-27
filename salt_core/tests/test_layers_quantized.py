"""`ConvLayer.forward_quantized`/`FCLayer.forward_quantized`(整數版,膜電位量化
模擬)的驗證:

1. **基本正確性**:單一事件、`Δt=0`(identity,跳過衰減)的最簡單案例,
   `v_final` 應該就是輸入的整數權重碼本身——純驗證佇列建構、權重 gather、
   dtype 轉型這條管線本身接對了。
2. **接線正確性**:`FCLayer.forward_quantized` 內部的每一塊(`build_fc_queue`、
   `apply_decay_table_int`、`chunk_scan.run_layer_forward_int`)都各自有單元
   測試,這裡驗證組裝起來的結果跟直接呼叫這些元件完全一致。
3. **conv catch-up**:真 tap 之後的 catch-up 衰減要真的套用到 `v_final`。
4. **跟訓練容量設定無關**:整數版每步一筆事件,不受 `ConvLayer.max_steps` 影響。
5. **多層串接**:`run_network_quantized` 跟手動逐層呼叫結果一致。
6. **`f_a` 越粗,結果確實不同**(不是接了一個沒作用的參數)。
7. **traced 版**:`v_steps` 最後一欄要等於 `forward_quantized` 的 `v_final`。
8. **入口檢查與容量診斷**:`q` 不是整數 dtype 要 raise;佇列/輸出容量出界要
   在 `LayerDiagInt` 回報。
9. **讀出、不 fire、溢位模式**:`run_network_quantized` 的讀出已經逐神經元
   乘回 $s_c$;`v_th_int=None` 的層不 fire;每層的 `overflow_mode` 有生效。
"""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from salt_core.chunk_scan import run_layer_forward_int
from salt_core.connectivity.fc import build_fc_queue
from salt_core.layer_chain import EventStream
from salt_core.layers import (ConvLayer, FCLayer, QuantizedLayerParams, dequantize_v_final,
                              run_network_quantized, run_network_quantized_traced)
from salt_core.quantize import apply_decay_table_int, build_decay_table_int


def _fc_stream():
    event_times = jnp.array([1.0, 2.0, 4.0])
    event_source_idx = jnp.array([0, 1, 0])
    return EventStream(event_times=event_times, event_source_idx=event_source_idx,
                       event_gain=jnp.ones_like(event_times),
                       n_real_events=jnp.array(3))


def _fc_layer(tau=4.0):
    return FCLayer(name="out", n_in=2, n_out=2, init_k=5.0, tau=tau, v_th=1.0, chunk_size=1)


def _params(q, *, tau, f_a, f_V, i_V, v_th_int, scale=1.0, overflow_mode="wrap"):
    return QuantizedLayerParams(q=q, decay_table_int=build_decay_table_int(f_a, tau),
                                v_th_int=v_th_int, scale=jnp.asarray(scale), f_a=f_a, f_V=f_V,
                                i_V=i_V, overflow_mode=overflow_mode)


def _conv_3x3_setup():
    """3x3 輸入、3x3 kernel、單一輸出神經元,9 筆事件依序打在 9 個 synapse 上。"""
    tau = 4.0
    conv = ConvLayer(name="conv", ic=1, h_in=3, w_in=3, oc=1, k=3, s=1, p=0,
                     init_k=5.0, tau=tau, v_th=1.0, chunk_size=1, L=9, max_out_spikes=9)
    n = 9
    in_stream = EventStream(event_times=jnp.arange(1.0, n + 1.0),
                            event_source_idx=jnp.arange(n),  # ic=1 時 flat index = y*w_in+x
                            event_gain=jnp.ones((n,)), n_real_events=jnp.array(n))
    return conv, in_stream


def test_fc_forward_quantized_single_identity_event_matches_hand_computation():
    """單一事件,event_times=[0.0] => Δt=0(is_identity=True,跳過衰減),
    v0=0,所以 v_final 應該就是這筆事件的整數權重碼本身,不管 f_a/tau/
    decay_table 是什麼——這是最簡單、零模糊地帶的正確性檢查。"""
    tau = 4.0
    layer = FCLayer(name="out", n_in=1, n_out=2, init_k=5.0, tau=tau, v_th=1.0, chunk_size=1)
    params = _params(jnp.array([[7], [3]]), tau=tau, f_a=4, f_V=0, i_V=16,
                     v_th_int=jnp.array([100, 100]))
    in_stream = EventStream(event_times=jnp.array([0.0]), event_source_idx=jnp.array([0]),
                            event_gain=jnp.ones((1,)), n_real_events=jnp.array(1))

    _out, result, _diag = layer.forward_quantized(params, in_stream)

    assert list(np.asarray(result.v_final)) == [7, 3]
    assert not bool(result.spike_mask.any())


def test_fc_forward_quantized_matches_manual_assembly_of_trusted_primitives():
    """`FCLayer.forward_quantized` 組裝起來的結果,跟直接呼叫
    `build_fc_queue`/`apply_decay_table_int`/`run_layer_forward_int` 完全一致
    ——多筆事件、真的會查表衰減的案例(不是上面那個 identity 特例)。"""
    tau = 4.0
    layer = _fc_layer(tau)
    params = _params(jnp.array([[5, 3], [2, 1]]), tau=tau, f_a=4, f_V=0, i_V=16,
                     v_th_int=jnp.array([8, 8]))
    in_stream = _fc_stream()

    _out_q, result_q, _diag = layer.forward_quantized(params, in_stream)

    queue = build_fc_queue(in_stream.event_times, in_stream.event_source_idx, params.q, tau,
                           event_gain=in_stream.event_gain,
                           n_real_events=in_stream.n_real_events)
    a_int, is_identity = apply_decay_table_int(queue.delta_t, params.decay_table_int)
    q_int = queue.maps.b.astype(jnp.int32)
    expected = run_layer_forward_int(a_int, is_identity, q_int, params.v_th_int,
                                     f_a=params.f_a, f_V=params.f_V, i_V=params.i_V)

    assert np.array_equal(np.asarray(result_q.v_final), np.asarray(expected.v_final))
    assert np.array_equal(np.asarray(result_q.spike_mask), np.asarray(expected.spike_mask))
    # 這組數字手算過(neuron0 在第三筆事件 fire、neuron1 全程不 fire),
    # 當一個具體數字的迴歸鎖定,不是只比對「兩條路一致」這個性質。
    assert bool(result_q.spike_mask[0].any()), "neuron0 應該 fire 過一次"
    assert not bool(result_q.spike_mask[1].any()), "neuron1 不該 fire"
    assert int(result_q.v_final[0]) == 0, "neuron0 fire 後硬重置成 0"
    assert int(result_q.v_final[1]) == 4


def test_fc_forward_quantized_coarser_f_a_changes_result():
    """`f_a` 越粗(查表精度越低),結果應該確實不同——用跟上面同一組多事件
    案例,只換 `f_a`。"""
    tau = 4.0
    layer = _fc_layer(tau)
    q = jnp.array([[5, 3], [2, 1]])
    v_th_int = jnp.array([1000, 1000])  # 夠大,只看軌跡終值,不管 fire
    in_stream = _fc_stream()

    fine = layer.forward_quantized(
        _params(q, tau=tau, f_a=14, f_V=0, i_V=16, v_th_int=v_th_int), in_stream)[1]
    coarse = layer.forward_quantized(
        _params(q, tau=tau, f_a=1, f_V=0, i_V=16, v_th_int=v_th_int), in_stream)[1]

    assert not np.array_equal(np.asarray(fine.v_final), np.asarray(coarse.v_final)), \
        "f_a=1(幾乎不查表)vs f_a=14(高精度)應該算出不同的 v_final"


def test_conv_forward_quantized_applies_catchup_decay():
    """conv 1x1 kernel、1x2 輸入,兩顆輸出神經元各自只看自己的像素。事件:
    t=2 打在神經元 0、t=5 打在神經元 1。tau=16、f_a=4、f_V=4、q=5:

    - 神經元 0:真 tap(Δt=2)=> 5*2^4=80;catch-up Δt=5-2=3,查表碼 13,
      13*80/16=65。catch-up 被跳過的話會停在 80。
    - 神經元 1:真 tap 就是全域最後一筆,catch-up Δt=0,維持 80。"""
    tau = 16.0
    conv = ConvLayer(name="c", ic=1, h_in=1, w_in=2, oc=1, k=1, s=1, p=0,
                     init_k=5.0, tau=tau, v_th=1.0, chunk_size=1, L=2, max_out_spikes=2)
    params = _params(jnp.array([[[[5]]]]), tau=tau, f_a=4, f_V=4, i_V=16,
                     v_th_int=jnp.array(10 ** 5))
    in_stream = EventStream(event_times=jnp.array([2.0, 5.0]),
                            event_source_idx=jnp.array([0, 1]),
                            event_gain=jnp.ones((2,)), n_real_events=jnp.array(2))

    _out, result, _diag = conv.forward_quantized(params, in_stream)

    assert list(np.asarray(result.v_final)) == [65, 80]


def test_conv_forward_quantized_ignores_training_max_steps():
    """`ConvLayer.max_steps` 是照訓練時的 `chunk_size` 校準出來的;整數版每步
    一筆事件,掃描長度是佇列長度 `L`,不能受它影響。兩個只有 `max_steps`
    不同的層算出來的結果要完全一樣。"""
    base, in_stream = _conv_3x3_setup()
    q = jnp.round(base.init_weight(jax.random.PRNGKey(0)) * 20).astype(jnp.int32)
    params = _params(q, tau=base.tau, f_a=8, f_V=2, i_V=16,
                     v_th_int=jnp.array([1000]))  # 只看 v_final 軌跡,不管 fire

    _out_small, result_small, _ = dataclasses.replace(base, max_steps=2).forward_quantized(
        params, in_stream)
    _out_large, result_large, _ = dataclasses.replace(base, max_steps=9).forward_quantized(
        params, in_stream)

    assert np.array_equal(np.asarray(result_small.v_final), np.asarray(result_large.v_final)), \
        "forward_quantized 的結果不該受 self.max_steps 影響"


def test_run_network_quantized_chains_conv_into_fc_matches_manual_chaining():
    """conv 接 FC 兩層,`run_network_quantized` 跟手動逐層呼叫、手動把輸出流
    接手,結果要完全一致。conv 的輸出流有 pad 事件(假時間 1e12),FC 的
    佇列要把它們的 Δt 遮成 0,不然查表入口會 raise。"""
    conv, input_stream = _conv_3x3_setup()
    fc = FCLayer(name="out", n_in=conv.n_neurons, n_out=2, init_k=5.0, tau=conv.tau,
                 v_th=1.0, chunk_size=1)
    layers = [conv, fc]

    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    quant_params = [
        _params(jnp.round(conv.init_weight(k1) * 20).astype(jnp.int32), tau=conv.tau,
                f_a=8, f_V=2, i_V=16, v_th_int=jnp.array([1000])),
        _params(jnp.round(fc.init_weight(k2) * 20).astype(jnp.int32), tau=fc.tau,
                f_a=8, f_V=2, i_V=16, v_th_int=jnp.array([1000, 1000])),
    ]

    readout, diags = run_network_quantized(layers, input_stream, quant_params)

    mid_stream, _mid_result, _ = conv.forward_quantized(quant_params[0], input_stream)
    _out_stream, expected, _ = fc.forward_quantized(quant_params[1], mid_stream)

    assert np.array_equal(np.asarray(readout.v_final),
                          np.asarray(dequantize_v_final(expected, quant_params[1]).v_final))
    assert np.array_equal(np.asarray(readout.spike_mask), np.asarray(expected.spike_mask))
    assert len(diags) == 2


def test_fc_forward_quantized_traced_last_column_matches_forward_quantized_v_final():
    """`v_steps` 最後一欄要等於 `forward_quantized` 的 `v_final`,不然溢位驗證
    拿 `v_steps` 算出的峰值會跟正常 forward 對不上。"""
    tau = 4.0
    layer = _fc_layer(tau)
    params = _params(jnp.array([[5, 3], [2, 1]]), tau=tau, f_a=4, f_V=0, i_V=16,
                     v_th_int=jnp.array([1000, 1000]))
    in_stream = _fc_stream()

    _out, result, _diag = layer.forward_quantized(params, in_stream)
    _out_t, result_t, v_steps, _diag_t = layer.forward_quantized_traced(params, in_stream)

    assert np.array_equal(np.asarray(v_steps[:, -1]), np.asarray(result.v_final)), \
        "v_steps 最後一欄應該等於 forward_quantized 的 v_final"
    assert np.array_equal(np.asarray(result_t.v_final), np.asarray(result.v_final))
    assert np.array_equal(np.asarray(result_t.spike_mask), np.asarray(result.spike_mask))


def test_run_network_quantized_traced_returns_result_v_steps_diag_per_layer():
    conv, input_stream = _conv_3x3_setup()
    q = jnp.round(conv.init_weight(jax.random.PRNGKey(0)) * 20).astype(jnp.int32)
    params = _params(q, tau=conv.tau, f_a=8, f_V=2, i_V=16, v_th_int=jnp.array([1000]))

    traces = run_network_quantized_traced([conv], input_stream, [params])
    result, v_steps, diag = traces[0]

    assert v_steps.shape == (conv.n_neurons, conv.L)
    assert int(v_steps[0, -1]) == int(result.v_final[0])
    assert not bool(diag.queue_truncated) and not bool(diag.output_truncated)


def test_forward_quantized_rejects_non_integer_weight_codes():
    """傳成乘回 scale 的浮點權重時,轉整數碼會全部變成 0 還照樣跑完,要直接拒絕。"""
    layer = _fc_layer()
    params = _params(jnp.array([[0.01, -0.02], [0.03, 0.0]]), tau=4.0, f_a=4, f_V=0, i_V=16,
                     v_th_int=jnp.array([100, 100]))
    with pytest.raises(ValueError):
        layer.forward_quantized(params, _fc_stream())


def test_conv_forward_quantized_reports_queue_truncation():
    """神經元需要 9 欄的佇列,L=4 裝不下,後面的輸入事件被截掉,要回報。"""
    conv, in_stream = _conv_3x3_setup()
    small_L = dataclasses.replace(conv, L=4)
    q = jnp.ones(conv.weight_shape, dtype=jnp.int32)
    params = _params(q, tau=conv.tau, f_a=8, f_V=2, i_V=16, v_th_int=jnp.array([1000]))

    _out, _result, diag = small_L.forward_quantized(params, in_stream)
    _out, _result, diag_ok = conv.forward_quantized(params, in_stream)

    assert bool(diag.queue_truncated)
    assert not bool(diag_ok.queue_truncated)


def test_conv_forward_quantized_reports_output_truncation():
    """門檻低到每筆事件都 fire(9 個 spike),輸出容量只有 2,要回報。"""
    conv, in_stream = _conv_3x3_setup()
    q = jnp.ones(conv.weight_shape, dtype=jnp.int32)
    params = _params(q, tau=conv.tau, f_a=8, f_V=0, i_V=16, v_th_int=jnp.array([1]))

    _out, result, diag = dataclasses.replace(conv, max_out_spikes=2).forward_quantized(
        params, in_stream)
    _out, _result, diag_ok = conv.forward_quantized(params, in_stream)

    assert int(result.spike_mask.sum()) == 9
    assert bool(diag.output_truncated)
    assert not bool(diag_ok.output_truncated)


def test_dequantize_v_final_multiplies_back_per_neuron_scale():
    """暫存器值 [100, 90]、f_V=1、s_c=[1.0, 2.0]:物理尺度 V=[50, 90]。
    整數值直接比大小會選 neuron0,乘回各自的 s_c 之後才是 neuron1。"""
    tau = 4.0
    layer = FCLayer(name="out", n_in=1, n_out=2, init_k=5.0, tau=tau, v_th=1e9, chunk_size=1)
    params = _params(jnp.array([[50], [45]]), tau=tau, f_a=4, f_V=1, i_V=16,
                     v_th_int=None, scale=[1.0, 2.0])
    in_stream = EventStream(event_times=jnp.array([0.0]), event_source_idx=jnp.array([0]),
                            event_gain=jnp.ones((1,)), n_real_events=jnp.array(1))

    _out, result, _diag = layer.forward_quantized(params, in_stream)
    readout, _diags = run_network_quantized([layer], in_stream, [params])

    assert list(np.asarray(result.v_final)) == [100, 90]
    assert np.allclose(np.asarray(readout.v_final), [50.0, 90.0])
    assert int(jnp.argmax(result.v_final)) == 0 and int(jnp.argmax(readout.v_final)) == 1


def test_forward_quantized_without_threshold_never_fires():
    """v_th_int=None 的層不 fire,v_final 就是整條佇列的累積值。"""
    tau = 4.0
    layer = _fc_layer(tau)
    with_threshold = _params(jnp.array([[5, 3], [2, 1]]), tau=tau, f_a=4, f_V=0, i_V=16,
                             v_th_int=jnp.array([8, 8]))
    no_threshold = with_threshold._replace(v_th_int=None)

    _out, fired, _ = layer.forward_quantized(with_threshold, _fc_stream())
    _out, silent, _ = layer.forward_quantized(no_threshold, _fc_stream())

    assert bool(fired.spike_mask.any()), "對照組:有門檻時 neuron0 會 fire"
    assert not bool(silent.spike_mask.any())


def test_forward_quantized_uses_layer_overflow_mode():
    """單一 identity 事件 q=9,暫存器 4 位元(範圍 [-8,7]):繞回是 -7,
    飽和是 7。溢位模式是逐層設定的。"""
    tau = 4.0
    layer = FCLayer(name="out", n_in=1, n_out=1, init_k=5.0, tau=tau, v_th=1e9, chunk_size=1)
    in_stream = EventStream(event_times=jnp.array([0.0]), event_source_idx=jnp.array([0]),
                            event_gain=jnp.ones((1,)), n_real_events=jnp.array(1))
    wrap = _params(jnp.array([[9]]), tau=tau, f_a=4, f_V=0, i_V=4, v_th_int=None)
    saturate = wrap._replace(overflow_mode="saturate")

    assert int(layer.forward_quantized(wrap, in_stream)[1].v_final[0]) == -7
    assert int(layer.forward_quantized(saturate, in_stream)[1].v_final[0]) == 7
