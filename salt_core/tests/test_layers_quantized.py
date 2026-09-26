"""`ConvLayer.forward_quantized`/`FCLayer.forward_quantized`(整數版,見
docs/問題紀錄.md 第十七~十九節)的驗證,分四類:

1. **基本正確性**:單一事件、`Δt=0`(identity,跳過衰減)的最簡單案例,
   `v_final` 應該就是輸入的整數權重碼本身——不牽涉衰減/捨入的任何模糊地帶,
   純驗證佇列建構、權重 gather、dtype 轉型這條管線本身接對了。
2. **接線正確性(跟已經各自測過的信任元件比對)**:`FCLayer.forward_quantized`
   內部做的事——`build_fc_queue` 取權重、`fc_delta_t` 算 Δt、
   `apply_decay_table_int` 查表、`chunk_scan.run_layer_forward_int` 跑遞迴——
   每一塊都已經有自己的單元測試,這裡驗證組裝起來的結果跟直接呼叫這些元件
   完全一致,不是重新手算一次遞迴數學。
3. **`ConvLayer.max_steps` 陷阱**:跟浮點版同一個問題,`forward_quantized`
   強制 `chunk_size=1` 之後不能沿用訓練時用 `chunk_size` 校準出來的
   `self.max_steps`,要改用 `self.L`。
4. **多層串接**:`run_network_quantized` 逐層呼叫 `forward_quantized`、把
   上一層輸出流餵給下一層這件事本身要對——驗證跟手動把 conv 接 FC 兩層分開
   呼叫、手動把輸出流接手,兩者結果一致。
5. **`f_a` 越粗,結果確實不同**(不是接了一個沒作用的參數)。
6. **`forward_quantized_traced`**:`v_steps` 最後一欄要等於 `forward_quantized`
   的 `v_final`。
"""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from salt_core.chunk_scan import run_layer_forward_int
from salt_core.connectivity.fc import build_fc_queue, fc_delta_t
from salt_core.layer_chain import EventStream
from salt_core.layers import ConvLayer, FCLayer, run_network_quantized
from salt_core.quantize import apply_decay_table_int, build_decay_table_int

TOL = 1e-4


def _fc_stream():
    event_times = jnp.array([1.0, 2.0, 4.0])
    event_source_idx = jnp.array([0, 1, 0])
    return EventStream(event_times=event_times, event_source_idx=event_source_idx,
                       event_gain=jnp.ones_like(event_times),
                       n_real_events=jnp.array(3))


def test_fc_forward_quantized_single_identity_event_matches_hand_computation():
    """單一事件,event_times=[0.0] => Δt=0(is_identity=True,跳過衰減),
    v0=0,所以 v_final 應該就是這筆事件的整數權重碼本身,不管 f_a/tau/
    decay_table 是什麼——這是最簡單、零模糊地帶的正確性檢查。"""
    tau = 4.0
    layer = FCLayer(name="out", n_in=1, n_out=2, init_k=5.0, tau=tau, v_th=1.0, chunk_size=1)
    q = jnp.array([[7], [3]])
    in_stream = EventStream(event_times=jnp.array([0.0]), event_source_idx=jnp.array([0]),
                            event_gain=jnp.ones((1,)), n_real_events=jnp.array(1))
    decay_table_int = build_decay_table_int(f_a=4, tau=tau)
    v_th_int = jnp.array([100, 100])

    _out, result = layer.forward_quantized(
        q, in_stream, decay_table_int=decay_table_int, v_th_int=v_th_int,
        f_a=4, f_V=0, i_V=16)

    assert list(np.asarray(result.v_final)) == [7, 3]
    assert not bool(result.spike_mask.any())


def test_fc_forward_quantized_matches_manual_assembly_of_trusted_primitives():
    """`FCLayer.forward_quantized` 內部組裝的每一塊(`build_fc_queue`/
    `fc_delta_t`/`apply_decay_table_int`/`run_layer_forward_int`)都各自有
    單元測試,這裡驗證組裝起來的結果跟直接呼叫這些元件完全一致——多筆事件、
    真的會查表衰減的案例(不是上面那個 identity 特例)。"""
    tau = 4.0
    layer = FCLayer(name="out", n_in=2, n_out=2, init_k=5.0, tau=tau, v_th=1.0, chunk_size=1)
    q = jnp.array([[5, 3], [2, 1]])
    in_stream = _fc_stream()
    f_a, f_V, i_V = 4, 0, 16
    decay_table_int = build_decay_table_int(f_a, tau)
    v_th_int = jnp.array([8, 8])

    _out_q, result_q = layer.forward_quantized(
        q, in_stream, decay_table_int=decay_table_int, v_th_int=v_th_int,
        f_a=f_a, f_V=f_V, i_V=i_V)

    maps = build_fc_queue(in_stream.event_times, in_stream.event_source_idx, q, tau,
                          event_gain=in_stream.event_gain,
                          n_real_events=in_stream.n_real_events)
    delta_t = fc_delta_t(in_stream.event_times, n_out_neurons=2)
    a_int, is_identity = apply_decay_table_int(delta_t, decay_table_int)
    q_int = jnp.round(maps.b).astype(jnp.int32)
    expected = run_layer_forward_int(a_int, is_identity, q_int, v_th_int, max_steps=3,
                                     n_real_events=in_stream.n_real_events,
                                     f_a=f_a, f_V=f_V, i_V=i_V)

    assert np.array_equal(np.asarray(result_q.v_final), np.asarray(expected.v_final))
    assert np.array_equal(np.asarray(result_q.spike_mask), np.asarray(expected.spike_mask))
    # 這組數字手算過(neuron0 在第三筆事件 fire、neuron1 全程不 fire),
    # 跟 test_process_event_int/test_chunk_scan 系列的手算風格一致,順便當
    # 一個具體數字的迴歸鎖定,不是只比對「兩條路一致」這個性質。
    assert bool(result_q.spike_mask[0].any()), "neuron0 應該 fire 過一次"
    assert not bool(result_q.spike_mask[1].any()), "neuron1 不該 fire"
    assert int(result_q.v_final[0]) == 0, "neuron0 fire 後硬重置成 0"
    assert int(result_q.v_final[1]) == 4


def test_fc_forward_quantized_coarser_f_a_changes_result():
    """`f_a` 越粗(查表精度越低),結果應該確實不同,不是接了一個沒作用的
    參數——用跟上面同一組多事件案例,只換 `f_a`。"""
    tau = 4.0
    layer = FCLayer(name="out", n_in=2, n_out=2, init_k=5.0, tau=tau, v_th=1.0, chunk_size=1)
    q = jnp.array([[5, 3], [2, 1]])
    in_stream = _fc_stream()
    v_th_int = jnp.array([1000, 1000])  # 夠大,只看軌跡終值,不管 fire

    fine = layer.forward_quantized(
        q, in_stream, decay_table_int=build_decay_table_int(f_a=14, tau=tau),
        v_th_int=v_th_int, f_a=14, f_V=0, i_V=16)[1]
    coarse = layer.forward_quantized(
        q, in_stream, decay_table_int=build_decay_table_int(f_a=1, tau=tau),
        v_th_int=v_th_int, f_a=1, f_V=0, i_V=16)[1]

    assert not np.array_equal(np.asarray(fine.v_final), np.asarray(coarse.v_final)), \
        "f_a=1(幾乎不查表)vs f_a=14(高精度)應該算出不同的 v_final"


def test_conv_forward_quantized_ignores_max_steps_uses_L():
    """`ConvLayer.max_steps` 是照訓練時的 `chunk_size` 校準出來的,
    `forward_quantized` 強制 `chunk_size=1` 之後步數需求會變大,不能沿用它。
    用兩個只有 `max_steps` 不同的 `ConvLayer`(一個故意設得很小)驗證算出來
    的結果完全一樣——代表真的是用 `self.L`,不是 `self.max_steps`。"""
    ic, h_in, w_in, oc, k, s, p = 1, 3, 3, 1, 3, 1, 0
    tau = 4.0
    base = ConvLayer(name="c", ic=ic, h_in=h_in, w_in=w_in, oc=oc, k=k, s=s, p=p,
                     init_k=5.0, tau=tau, v_th=1.0, chunk_size=1, L=9)
    q = jnp.round(base.init_weight(jax.random.PRNGKey(0)) * 20).astype(jnp.int32)

    n = 9  # 3x3 感受野,剛好 9 個輸入 synapse
    event_times = jnp.arange(1.0, n + 1.0)
    event_source_idx = jnp.arange(n)  # ic=1 時 flat index = y*w_in+x,剛好是 0..8
    in_stream = EventStream(event_times=event_times, event_source_idx=event_source_idx,
                            event_gain=jnp.ones((n,)), n_real_events=jnp.array(n))

    decay_table_int = build_decay_table_int(f_a=8, tau=tau)
    v_th_int = jnp.array([1000])  # 只看 v_final 軌跡,不管 fire

    small = dataclasses.replace(base, max_steps=2)
    large = dataclasses.replace(base, max_steps=9)

    _out_small, result_small = small.forward_quantized(
        q, in_stream, decay_table_int=decay_table_int, v_th_int=v_th_int,
        f_a=8, f_V=2, i_V=16)
    _out_large, result_large = large.forward_quantized(
        q, in_stream, decay_table_int=decay_table_int, v_th_int=v_th_int,
        f_a=8, f_V=2, i_V=16)

    assert np.array_equal(np.asarray(result_small.v_final), np.asarray(result_large.v_final)), \
        "forward_quantized 的結果不該受 self.max_steps 影響"


def test_run_network_quantized_chains_conv_into_fc_matches_manual_chaining():
    """`run_network_quantized`(notebook 之後會直接呼叫的多層串接函式)獨立
    測一次:conv 接 FC 兩層,驗證跟手動把兩層分開呼叫、手動把輸出流接手,
    結果完全一致——單層各自的 `forward_quantized` 測過沒問題,不代表
    `run_network_quantized` 的 for 迴圈/zip 接線本身也一定沒事。"""
    ic, h_in, w_in, oc, k, s, p = 1, 3, 3, 1, 3, 1, 0
    tau = 4.0
    conv = ConvLayer(name="conv", ic=ic, h_in=h_in, w_in=w_in, oc=oc, k=k, s=s, p=p,
                     init_k=5.0, tau=tau, v_th=1.0, chunk_size=1, L=9, max_out_spikes=9)
    fc = FCLayer(name="out", n_in=conv.n_neurons, n_out=2, init_k=5.0, tau=tau, v_th=1.0,
                chunk_size=1)
    layers = [conv, fc]

    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    q_conv = jnp.round(conv.init_weight(k1) * 20).astype(jnp.int32)
    q_fc = jnp.round(fc.init_weight(k2) * 20).astype(jnp.int32)
    weights = [q_conv, q_fc]

    n = 9
    event_times = jnp.arange(1.0, n + 1.0)
    event_source_idx = jnp.arange(n)
    input_stream = EventStream(event_times=event_times, event_source_idx=event_source_idx,
                               event_gain=jnp.ones((n,)), n_real_events=jnp.array(n))

    decay_tables_int = [build_decay_table_int(f_a=8, tau=conv.tau),
                        build_decay_table_int(f_a=8, tau=fc.tau)]
    v_th_int = [jnp.array([1000]), jnp.array([1000, 1000])]
    f_a = [8, 8]
    f_V = [2, 2]
    i_V = [16, 16]

    result = run_network_quantized(
        layers, input_stream, weights, decay_tables_int=decay_tables_int,
        v_th_int=v_th_int, f_a=f_a, f_V=f_V, i_V=i_V)

    # 手動串接同一組運算,當獨立 oracle
    mid_stream, _mid_result = conv.forward_quantized(
        q_conv, input_stream, decay_table_int=decay_tables_int[0], v_th_int=v_th_int[0],
        f_a=f_a[0], f_V=f_V[0], i_V=i_V[0])
    _out_stream, expected = fc.forward_quantized(
        q_fc, mid_stream, decay_table_int=decay_tables_int[1], v_th_int=v_th_int[1],
        f_a=f_a[1], f_V=f_V[1], i_V=i_V[1])

    assert np.array_equal(np.asarray(result.v_final), np.asarray(expected.v_final))
    assert np.array_equal(np.asarray(result.spike_mask), np.asarray(expected.spike_mask))


def test_fc_forward_quantized_traced_last_column_matches_forward_quantized_v_final():
    """`v_steps` 最後一欄要等於 `forward_quantized` 的 `v_final`——跟既有
    `forward`/`forward_traced` 的保證是同一個性質,量化整數版也要維持,不然
    溢位驗證拿 `v_steps` 算出的峰值會跟正常 forward 對不上。"""
    tau = 4.0
    layer = FCLayer(name="out", n_in=2, n_out=2, init_k=5.0, tau=tau, v_th=1.0, chunk_size=1)
    q = jnp.array([[5, 3], [2, 1]])
    in_stream = _fc_stream()
    decay_table_int = build_decay_table_int(f_a=4, tau=tau)
    v_th_int = jnp.array([1000, 1000])

    _out, result = layer.forward_quantized(
        q, in_stream, decay_table_int=decay_table_int, v_th_int=v_th_int,
        f_a=4, f_V=0, i_V=16)
    _out_t, result_t, v_steps = layer.forward_quantized_traced(
        q, in_stream, decay_table_int=decay_table_int, v_th_int=v_th_int,
        f_a=4, f_V=0, i_V=16)

    assert np.array_equal(np.asarray(v_steps[:, -1]), np.asarray(result.v_final)), \
        "v_steps 最後一欄應該等於 forward_quantized 的 v_final"
    assert np.array_equal(np.asarray(result_t.v_final), np.asarray(result.v_final))
    assert np.array_equal(np.asarray(result_t.spike_mask), np.asarray(result.spike_mask))


TESTS = [
    test_fc_forward_quantized_single_identity_event_matches_hand_computation,
    test_fc_forward_quantized_matches_manual_assembly_of_trusted_primitives,
    test_fc_forward_quantized_coarser_f_a_changes_result,
    test_conv_forward_quantized_ignores_max_steps_uses_L,
    test_run_network_quantized_chains_conv_into_fc_matches_manual_chaining,
    test_fc_forward_quantized_traced_last_column_matches_forward_quantized_v_final,
]


if __name__ == "__main__":
    for test in TESTS:
        test()
        print(f"PASS: {test.__name__}")
    print(f"\n全部 {len(TESTS)} 項測試通過")
