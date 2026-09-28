"""salt_core 逐步軌跡監測(docs/監測規格.md §6)的測試。

四層:

- `run_layer_forward_traced` vs `run_layer_forward`:同參數下 `LayerForwardResult`
  五欄逐位元相同(共用 scan 內核),`v_steps` 的最後一欄膜電位 == `v_final`。
- `resolve_ms_*`:掃描步指標 -> 真實毫秒的還原,手算小例子 + 空轉步 = nan。
- `run_network(..., trace=True)`:結果跟不帶軌跡時一致、`LayerForwardTrace` 形狀對、
  `stop_gradient` 生效。
- `summarize_trace_scalars`:純歸約函式,手算小例子 + 非有限值計數。

`LayerForwardTrace` 2026-09-13 拔掉 `s_value` 欄位(這個架構下 forward 數值
恆等於 `spike_mask`,而且整條 trace 本身已經 `stop_gradient`,留著沒有實際
資訊量,見 `salt_core/trace.py` 開頭說明);`summarize_trace`/`pack_key`/
`unpack_key`/`layer_names` 隨同一批改動一併移除(唯一呼叫端 `example/
trace_probe.py` 已刪除)。
"""
import jax
import jax.numpy as jnp
import numpy as np

from salt_core.float.scan import run_layer_forward, run_layer_forward_traced
from salt_core.connectivity.fc import build_fc_structure, fc_float_values
from salt_core.layers import ConvLayer, FCLayer
from salt_core.network import Network, RawEvents, run_network
from salt_core.trace import (LayerForwardTrace, resolve_ms_conv,
                                resolve_ms_fc, summarize_trace_scalars)

TOL = 1e-6

_C1 = dict(ic=2, h_in=34, w_in=34, oc=8, k=3, s=2, p=1)
_C2 = dict(ic=8, h_in=17, w_in=17, oc=16, k=3, s=2, p=1)


def _layers():
    conv1 = ConvLayer(name="conv1", **_C1, max_queue_len=185, max_out_spikes=4000, init_k=5.0)
    conv2 = ConvLayer(name="conv2", **_C2, max_queue_len=400, max_out_spikes=12000, init_k=5.0)
    out = FCLayer(name="out", n_in=conv2.n_neurons, n_out=10, v_th=1e9, chunk_size=64, init_k=5.0)
    return [conv1, conv2, out]


def _params(layers, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), len(layers))
    return tuple(layer.init_weight(k) for layer, k in zip(layers, keys))


def _raw_batch(key, n_samples, max_len, h_in, w_in, ic):
    ks = jax.random.split(key, n_samples * 4)
    et = jnp.zeros((n_samples, max_len), dtype=jnp.float32)
    xs = jnp.zeros((n_samples, max_len), dtype=jnp.int32)
    ys = jnp.zeros((n_samples, max_len), dtype=jnp.int32)
    cs = jnp.zeros((n_samples, max_len), dtype=jnp.int32)
    nr = []
    for i in range(n_samples):
        kt, kx, ky, kc = ks[4 * i:4 * i + 4]
        n = max_len - (i % 3)
        t = jnp.sort(jax.random.uniform(kt, (n,), minval=1.0, maxval=30.0))
        et = et.at[i, :n].set(t)
        et = et.at[i, n:].set(t[-1])
        xs = xs.at[i, :n].set(jax.random.randint(kx, (n,), 0, w_in))
        ys = ys.at[i, :n].set(jax.random.randint(ky, (n,), 0, h_in))
        cs = cs.at[i, :n].set(jax.random.randint(kc, (n,), 0, ic))
        nr.append(n)
    return et, xs, ys, cs, jnp.array(nr, dtype=jnp.int32)


def _stream0(batch, layer0):
    raw = RawEvents(*(v[0] for v in batch))
    return Network(layer0.input_shape, [layer0]).input_stream(raw)


# ============================================================================
# A. run_layer_forward_traced == run_layer_forward(+軌跡)
# ============================================================================

def _toy_fc_maps(n_out, n_events, seed):
    k = jax.random.PRNGKey(seed)
    src = jnp.arange(n_events, dtype=jnp.int32) % 3
    times = jnp.sort(jax.random.uniform(k, (n_events,), minval=1.0, maxval=40.0))
    w = jax.random.uniform(jax.random.PRNGKey(seed + 1), (n_out, 3), minval=-1.0, maxval=1.0)
    maps = fc_float_values(build_fc_structure(times, src, n_events), w, 8.0, jnp.ones((n_events,)))
    return maps, times


def test_traced_forward_result_bit_identical():
    for chunk_size, v_th in [(1, 1.0), (1, 1e9), (4, 1.0), (4, 1e9)]:
        maps, _ = _toy_fc_maps(n_out=5, n_events=24, seed=chunk_size + int(v_th))
        max_steps = -(-maps.a.shape[1] // chunk_size)
        base = run_layer_forward(maps, v_th, chunk_size=chunk_size, max_steps=max_steps,
                                  n_real_events=maps.a.shape[1])
        traced, v_steps, pointer = run_layer_forward_traced(
            maps, v_th, chunk_size=chunk_size, max_steps=max_steps,
            n_real_events=maps.a.shape[1])
        for name in base._fields:
            np.testing.assert_array_equal(
                np.asarray(getattr(base, name)), np.asarray(getattr(traced, name)),
                err_msg=f"{name} 不一致(chunk_size={chunk_size}, v_th={v_th})")
        assert v_steps.shape == (5, max_steps)
        assert pointer.shape == (5, max_steps)
        np.testing.assert_allclose(np.asarray(v_steps[:, -1]),
                                    np.asarray(base.v_final), atol=TOL)


def test_traced_pointer_monotone_and_starts_at_zero():
    maps, _ = _toy_fc_maps(n_out=4, n_events=20, seed=7)
    _, _v_steps, pointer = run_layer_forward_traced(
        maps, v_th=1e9, chunk_size=1, max_steps=20, n_real_events=20)
    ptr = np.asarray(pointer)
    assert np.all(ptr[:, 0] == 0)
    assert np.all(np.diff(ptr, axis=1) >= 0), "pointer 不能倒退"
    # 純積分器 chunk_size=1:每步消化一筆 -> pointer 就是步序號
    np.testing.assert_array_equal(ptr, np.tile(np.arange(20), (4, 1)))


# ============================================================================
# B. resolve_ms_*
# ============================================================================

def test_resolve_ms_fc_hand():
    event_times = jnp.array([10.0, 11.0, 13.0, 20.0, 25.0])
    pointer = jnp.array([[0, 1, 2, 3, 4],
                         [0, 2, 4, 5, 6]])          # 第 2 列後兩步越界(n_real=5)
    n_real = jnp.array([5, 5], dtype=jnp.int32)
    ms = np.asarray(resolve_ms_fc(pointer, n_real, event_times))
    np.testing.assert_allclose(ms[0], [10.0, 11.0, 13.0, 20.0, 25.0])
    np.testing.assert_allclose(ms[1][:3], [10.0, 13.0, 25.0])
    assert np.isnan(ms[1][3]) and np.isnan(ms[1][4]), "pointer>=n_real 應為 nan"


def test_resolve_ms_conv_hand():
    event_times = jnp.array([5.0, 7.0, 8.0, 12.0])
    # 神經元 0 佇列局部欄 -> 全域事件 [1, 3, 哨兵];神經元 1 -> [0, 哨兵, 哨兵]
    local_to_global_j = jnp.array([[1, 3, 4],
                                   [0, 4, 4]], dtype=jnp.int32)   # 哨兵 = n_events = 4
    n_real = jnp.array([2, 1], dtype=jnp.int32)
    pointer = jnp.array([[0, 1, 2, 2],
                         [0, 1, 2, 2]], dtype=jnp.int32)
    ms = np.asarray(resolve_ms_conv(pointer, local_to_global_j, n_real, event_times))
    np.testing.assert_allclose(ms[0][:2], [7.0, 12.0])          # 事件 1, 3
    assert np.isnan(ms[0][2]) and np.isnan(ms[0][3])
    np.testing.assert_allclose(ms[1][0], 5.0)                   # 事件 0
    assert np.all(np.isnan(ms[1][1:]))


# ============================================================================
# C. layer.forward / run_network 帶 trace=True
# ============================================================================


def test_traced_event_ms_within_input_range_or_nan():
    layers = _layers()
    params = _params(layers)
    batch = _raw_batch(jax.random.PRNGKey(2), 3, 20, 34, 34, 2)
    stream = _stream0(batch, layers[0])
    trace = layers[0].forward(params[0], stream, trace=True).trace
    ms = np.asarray(trace.event_ms)
    real = np.asarray(stream.event_times)[:int(stream.n_real_events)]
    finite = ms[np.isfinite(ms)]
    assert finite.min() >= real.min() - TOL
    assert finite.max() <= real.max() + TOL
    assert np.isnan(ms).any(), "max_queue_len=185 遠大於事件數,應該有空轉步 = nan"


def test_run_network_trace_shape_and_alignment():
    layers = _layers()
    params = _params(layers)
    stream = _stream0(_raw_batch(jax.random.PRNGKey(3), 3, 20, 34, 34, 2), layers[0])
    traces = run_network(layers, params, stream, trace=True).traces
    assert len(traces) == len(layers)
    for layer, t in zip(layers, traces):
        assert isinstance(t, LayerForwardTrace)
        assert t.v_steps.shape[0] == layer.n_neurons


def test_run_network_trace_stops_gradient():
    layers = _layers()
    params = _params(layers)
    stream = _stream0(_raw_batch(jax.random.PRNGKey(4), 2, 16, 34, 34, 2), layers[0])

    def loss(ps):
        traces = run_network(layers, ps, stream, trace=True).traces
        return sum(jnp.nansum(t.v_steps) for t in traces)

    grads = jax.grad(loss)(params)
    for g in grads:
        np.testing.assert_array_equal(np.asarray(g), np.zeros_like(np.asarray(g)))


def test_run_network_trace_matches_run_network_without_trace():
    """帶軌跡的最後一層軌跡,跟不帶軌跡的最後一層結果一致。"""
    layers = _layers()
    params = _params(layers)
    stream = _stream0(_raw_batch(jax.random.PRNGKey(5), 3, 20, 34, 34, 2), layers[0])

    result = run_network(layers, params, stream).last
    traces = run_network(layers, params, stream, trace=True).traces
    np.testing.assert_array_equal(np.asarray(traces[-1].spike_mask),
                                   np.asarray(result.spike_mask))
    np.testing.assert_allclose(np.asarray(traces[-1].v_steps[:, -1]),
                                np.asarray(result.v_final), atol=TOL)


# ============================================================================
# D. summarize_trace_scalars
# ============================================================================

def _toy_trace():
    spike_mask = jnp.array([[True, False, True], [False, False, False]])
    v_steps = jnp.array([[0.1, 0.2, 0.9], [0.05, 0.05, 0.05]], dtype=jnp.float32)
    event_ms = jnp.array([[1.0, 2.0, jnp.nan], [jnp.nan, jnp.nan, jnp.nan]],
                        dtype=jnp.float32)
    return LayerForwardTrace(spike_mask=spike_mask, v_steps=v_steps, event_ms=event_ms)


def test_summarize_trace_scalars_hand():
    out = summarize_trace_scalars(_toy_trace())
    assert out["n"] == 2 and out["steps"] == 3
    assert out["total_spikes"] == 2
    np.testing.assert_array_equal(out["fired"], [0])          # 只有神經元 0 有 fire
    np.testing.assert_allclose(out["idle_frac"], 4 / 6, atol=TOL)   # 6 格裡 4 個 nan
    np.testing.assert_allclose(out["v_range"], (0.05, 0.9), atol=TOL)
    assert out["nonfinite_v"] == 0


def test_summarize_trace_scalars_nonfinite_counts():
    bad_v = jnp.array([[0.1, jnp.inf, 0.9], [0.05, 0.05, jnp.nan]], dtype=jnp.float32)
    trace = _toy_trace()._replace(v_steps=bad_v)
    out = summarize_trace_scalars(trace)
    assert out["nonfinite_v"] == 2          # 一個 inf + 一個 nan
