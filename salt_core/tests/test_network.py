"""salt_core/network.py:InputEvents.checked 的時間檢查、Network 的輸入流、第一層是 FC、
apply_batched 跟逐筆 apply 一致、fits 旗標。連接檢查在 test_layer_connections.py。"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from salt_core.layers import FCLayer
from salt_core.network import InputEvents, Network, run_network
from salt_core.quant.backend import QuantBackend
from salt_core.quant.params import QuantizedLayerParams
from salt_core.quant.codes import build_decay_table_int
from salt_core.tests._small_network import (GENEROUS, INPUT_SHAPE, init_params, raw_batch,
                                            small_layers)


def _raw(times, n_real):
    return (jnp.array(times, dtype=jnp.float32), jnp.zeros(len(times), dtype=jnp.int32),
            jnp.array(n_real))


def test_checked_accepts_valid_times_and_ignores_pad():
    """pad 位置的時間(這裡是 1e12)不檢查。"""
    raw = InputEvents.checked(*_raw([0.0, 3.0, 3.0, 1e12], n_real=3))
    assert isinstance(raw, InputEvents)


@pytest.mark.parametrize("times, message", [
    ([1.0, 2.5, 4.0], "整數"),
    ([-1.0, 2.0, 4.0], r"\[0, 2\^31\)"),
    ([1.0, 2.0, 2.0 ** 31], r"\[0, 2\^31\)"),
    ([1.0, 5.0, 4.0], "不遞減"),
])
def test_checked_rejects_invalid_real_times(times, message):
    with pytest.raises(ValueError, match=message):
        InputEvents.checked(*_raw(times, n_real=3))


def test_checked_accepts_batch():
    """一批時每筆各自只看自己的真事件:第 2 筆第 3 格是 pad,時間倒退也不算。"""
    times = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]])
    InputEvents.checked(times, jnp.zeros((2, 3), dtype=jnp.int32), jnp.array([3, 2]))


def test_input_stream_copies_source_idx():
    network = Network((2, 3, 4), [FCLayer(name="fc", n_in=24, n_out=2, init_k=1.0)])
    raw = InputEvents(event_times=jnp.array([1.0, 2.0]), source_idx=jnp.array([21, 3]),
                      n_real_events=jnp.array(2))
    stream = network.input_stream(raw)
    assert stream.event_source_idx.tolist() == [21, 3]
    assert stream.event_gain.tolist() == [1.0, 1.0]
    assert stream.n_real_events.dtype == jnp.int32


@pytest.mark.parametrize("input_shape", [(2, 3, 4), (24,)])
def test_first_layer_fc_runs(input_shape):
    """input_shape 不限維度:空間形狀跟攤平形狀都接得上 n_in=24 的 FC。"""
    network = Network(input_shape, [FCLayer(name="fc", n_in=24, n_out=2, init_k=1.0)])
    weights = network.init(jax.random.PRNGKey(0))
    raw = InputEvents(event_times=jnp.array([1.0, 2.0]), source_idx=jnp.array([21, 3]),
                      n_real_events=jnp.array(2))
    out = network.apply(weights, raw)
    expected = run_network(network.layers, weights, network.input_stream(raw))
    assert out.last.v_final.shape == (2,)
    np.testing.assert_array_equal(np.asarray(out.last.v_final), np.asarray(expected.last.v_final))


def _quant_params(layers, weights):
    """每層權重乘 20 取整當整數碼;最後一層不 fire。"""
    params = []
    for i, (layer, w) in enumerate(zip(layers, weights)):
        v_th_int = None if i == len(layers) - 1 else jnp.array(1000)
        params.append(QuantizedLayerParams(
            q=jnp.round(w * 20).astype(jnp.int32), decay_table_int=build_decay_table_int(8, layer.tau),
            v_th_int=v_th_int, scale=jnp.asarray(1.0), f_a=8, f_V=2, i_V=16))
    return params


@pytest.mark.parametrize("backend_name", ["float", "quant"])
def test_apply_batched_matches_per_sample_apply(backend_name):
    network = Network(INPUT_SHAPE, small_layers())
    weights = init_params(list(network.layers), seed=3)
    et, source_idx, nr = raw_batch(seed=4, n_samples=3)
    batch = InputEvents(jnp.floor(et), source_idx, nr)  # 整數 backend 要整數毫秒
    if backend_name == "float":
        params, kwargs = weights, {}
    else:
        params, kwargs = _quant_params(network.layers, weights), {"backend": QuantBackend()}

    batched = network.apply_batched(params, batch, **kwargs)
    for i in range(3):
        single = network.apply(params, jax.tree_util.tree_map(lambda a: a[i], batch), **kwargs)
        for got, want in zip(batched.results, single.results):
            np.testing.assert_array_equal(np.asarray(got.spike_mask[i]),
                                          np.asarray(want.spike_mask))
            np.testing.assert_array_equal(np.asarray(got.v_final[i]), np.asarray(want.v_final))


def test_apply_batched_accepts_numpy_weights_as_jit_constants():
    """從 npz 讀進來的權重是 numpy 陣列;當 jit 常數時,FC 用 traced index 取權重不能失敗。"""
    network = Network(INPUT_SHAPE, small_layers())
    weights = tuple(np.asarray(w) for w in init_params(list(network.layers), seed=3))
    batch = InputEvents(*raw_batch(seed=4, n_samples=2))

    got = jax.jit(lambda raw: network.apply_batched(weights, raw).last.v_final)(batch)
    want = network.apply_batched(tuple(jnp.asarray(w) for w in weights), batch).last.v_final
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def _fits_of(capacity: dict) -> np.ndarray:
    """6 筆合成樣本在這組容量下的 fits。"""
    network = Network(INPUT_SHAPE, small_layers(capacity))
    weights = init_params(list(network.layers), seed=3)
    return np.asarray(network.apply_batched(weights, InputEvents(*raw_batch(seed=4))).fits)


def test_fits_true_when_capacity_is_generous():
    assert _fits_of(GENEROUS).tolist() == [True] * 6


def test_fits_per_sample_when_one_layer_is_too_small():
    """conv2 每筆需要的輸出 spike 數是 62、63、41、26、36、49;上限 50 時前兩筆放不下。"""
    capacity = {**GENEROUS, "conv2": {**GENEROUS["conv2"], "max_out_spikes": 50}}
    assert _fits_of(capacity).tolist() == [False, False, True, True, True, True]


def test_fits_single_sample_is_scalar():
    network = Network(INPUT_SHAPE, small_layers())
    weights = init_params(list(network.layers), seed=3)
    raw = jax.tree_util.tree_map(lambda a: a[0], InputEvents(*raw_batch(seed=4)))
    fits = network.apply(weights, raw).fits
    assert fits.shape == () and bool(fits)
