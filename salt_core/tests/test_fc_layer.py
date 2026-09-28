"""FCLayer 當會 fire 的隱藏層:容量(max_out_spikes、max_extra_steps)出界要被 fits 抓到,
給夠時結果正確;conv -> FC -> FC 串接跟逐層呼叫 primitive 一致。"""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from salt_core.connectivity.fc import build_fc_structure, fc_float_values
from salt_core.float.scan import run_layer_forward
from salt_core.layers import FCLayer
from salt_core.network import Network, RawEvents, run_network
from salt_core.quant.backend import QuantBackend, QuantizedLayerParams
from salt_core.quant.codes import build_decay_table_int
from salt_core.stream import extract_output_events_fc
from salt_core.tests._small_network import INPUT_SHAPE, init_params, raw_batch, small_layers

# 一顆神經元、兩個來源:來源 0 權重 1.0(單獨一筆就 fire),來源 1 權重 0.1。
# 10 筆事件、時間 0..9 ms,來源 0 在第 0、5 筆,其餘是來源 1:在第 0、5 筆各 fire 一次。
_W = jnp.array([[1.0, 0.1]])
_SOURCE = jnp.array([0, 1, 1, 1, 1, 0, 1, 1, 1, 1])


def _two_fire_case(**fc_fields):
    """(network, raw):輸入網格 (2, 1, 1),來源編號 = channel。"""
    layer = FCLayer(name="fc", n_in=2, n_out=1, init_k=1.0, tau=16.0, v_th=1.0, **fc_fields)
    raw = RawEvents(event_times=jnp.arange(10.0), x=jnp.zeros(10, dtype=jnp.int32),
                    y=jnp.zeros(10, dtype=jnp.int32), c=_SOURCE, n_real_events=jnp.array(10))
    return Network((2, 1, 1), [layer]), raw


def test_too_few_extra_steps_is_caught_by_fits():
    """chunk_size=5 時實際要 3 步(每步吃 1、5、4 筆),基本步數只有 ceil(10/5)=2。
    上界公式:b>0 有 10 筆、能量 floor(2.8/1)=2,m*=2,上界 2 + ceil(8/5)=4,需求 4 - 2 = 2。"""
    network, raw = _two_fire_case(chunk_size=5, max_extra_steps=0)
    reference, _ = _two_fire_case(chunk_size=1)
    out = network.apply((_W,), raw)
    assert not bool(out.fits)
    assert int(out.diags[0].needed["max_extra_steps"]) == 2
    # 第 6~9 筆沒處理到,v_final 跟正確答案不同
    assert not np.allclose(np.asarray(out.last.v_final),
                           np.asarray(reference.apply((_W,), raw).last.v_final))


def test_enough_extra_steps_processes_every_event():
    """旋鈕給需求值時,每筆事件都有處理到:結果跟 chunk_size=1(需求一定是 0)相同。"""
    network, raw = _two_fire_case(chunk_size=5, max_extra_steps=2)
    reference, _ = _two_fire_case(chunk_size=1)
    out = network.apply((_W,), raw)
    ref = reference.apply((_W,), raw)
    assert bool(out.fits) and bool(ref.fits)
    assert int(out.diags[0].spike_count) == int(ref.diags[0].spike_count) == 2
    np.testing.assert_allclose(np.asarray(out.last.v_final), np.asarray(ref.last.v_final),
                               atol=1e-6)


def test_output_spike_overflow_is_caught_by_fits():
    """兩次 fire、輸出容量 1:需求是真 spike 數 2,放不下;容量 2 放得下。"""
    network, raw = _two_fire_case(max_out_spikes=1)
    out = network.apply((_W,), raw)
    assert not bool(out.fits)
    assert int(out.diags[0].needed["max_out_spikes"]) == 2
    roomy, _ = _two_fire_case(max_out_spikes=2)
    assert bool(roomy.apply((_W,), raw).fits)


def test_quant_output_spike_overflow_is_caught_by_fits():
    """量化版:整數門檻 1,每筆正權重事件都 fire,10 次 fire 放不下輸出容量 1。"""
    network, raw = _two_fire_case(max_out_spikes=1)
    params = QuantizedLayerParams(q=jnp.array([[20, 2]], dtype=jnp.int32),
                                  decay_table_int=build_decay_table_int(8, 16.0),
                                  v_th_int=jnp.array(1), scale=jnp.asarray(1.0),
                                  f_a=8, f_V=2, i_V=16)
    out = network.apply((params,), raw, backend=QuantBackend())
    assert not bool(out.fits)
    assert int(out.diags[0].needed["max_out_spikes"]) == int(out.diags[0].spike_count) == 10


def _chain():
    """conv(2x8x8 -> 4x8x8) -> FC 隱藏層(會 fire,chunk_size=4) -> FC 輸出層(不 fire)。
    隱藏層的額外步數給輸入流長度,一定夠。"""
    conv1 = small_layers()[0]
    hidden = FCLayer(name="hidden", n_in=conv1.n_neurons, n_out=8, init_k=40.0, v_th=1.0,
                     chunk_size=4, max_out_spikes=4096, max_extra_steps=conv1.max_out_spikes)
    out = FCLayer(name="out", n_in=8, n_out=3, init_k=5.0, v_th=1e9, chunk_size=16,
                  max_out_spikes=1)
    return Network(INPUT_SHAPE, [conv1, hidden, out])


def _primitive_fc(layer: FCLayer, w, stream):
    """逐層呼叫 primitive:佇列結構 + 數值段 + 掃描整條佇列 + 抽輸出流。"""
    structure = build_fc_structure(stream.event_times, stream.event_source_idx,
                                   stream.n_real_events)
    maps = fc_float_values(structure, w, layer.tau, stream.event_gain)
    result = run_layer_forward(maps, layer.v_th, chunk_size=layer.chunk_size,
                               max_steps=maps.a.shape[1], alpha=layer.alpha,
                               n_real_events=jnp.broadcast_to(stream.n_real_events,
                                                              (layer.n_out,)))
    out_stream = extract_output_events_fc(result.spike_mask, result.spike_event_idx,
                                          result.s_spike, stream.event_times,
                                          max_total_spikes=layer.max_out_spikes)
    return result, out_stream


def test_conv_fc_fc_chain_matches_primitive_layer_by_layer():
    network = _chain()
    weights = init_params(list(network.layers), seed=3)
    raw = jax.tree_util.tree_map(lambda a: a[0], RawEvents(*raw_batch(seed=4)))
    out = run_network(network.layers, weights, network.input_stream(raw))
    assert bool(out.fits)

    stream = network.layers[0].forward(weights[0], network.input_stream(raw)).stream
    hidden_result, hidden_stream = _primitive_fc(network.layers[1], weights[1], stream)
    out_result, _ = _primitive_fc(network.layers[2], weights[2], hidden_stream)

    n_hidden_spikes = int(jnp.sum(hidden_result.spike_mask))
    assert n_hidden_spikes > 0, "隱藏層要真的 fire,這個測試才有意義"
    assert int(out.diags[1].spike_count) == n_hidden_spikes
    np.testing.assert_allclose(np.asarray(out.results[1].v_final),
                               np.asarray(hidden_result.v_final), atol=1e-5)
    np.testing.assert_allclose(np.asarray(out.last.v_final), np.asarray(out_result.v_final),
                               atol=1e-5)


def test_fc_capacity_matches_fields_and_with_capacity_replaces_them():
    layer = FCLayer(name="fc", n_in=4, n_out=2, init_k=1.0, max_out_spikes=7, max_extra_steps=3)
    assert dict(layer.capacity) == {"max_out_spikes": 7, "max_extra_steps": 3}
    grown = layer.with_capacity(layer.capacity.replace(max_extra_steps=9))
    assert grown == dataclasses.replace(layer, max_extra_steps=9)
