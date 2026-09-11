"""example/trace_probe.py 的單元測試。

- `due()` / `_full_due()` 的 gating。
- `run()` 寫 summary.npz:key / shape 對、跨呼叫累積、逐神經元摘要 == 手動
  `run_network_traced` + 縮減 + 對 K 平均。
- `full_every` 觸發 full_epoch_XXX.npz,shape `(S, n, max_steps)`。
- 出界重練退回同一個 epoch 號 -> 覆寫該列不新增。
"""
import os

import jax
import jax.numpy as jnp
import numpy as np

from salt_core.layers import (ConvLayer, FCLayer, raw_events_to_stream,
                               run_network_traced)
from example.trace_probe import TraceProbe

_C1 = dict(ic=2, h_in=34, w_in=34, oc=4, k=3, s=2, p=1)
_C2 = dict(ic=4, h_in=17, w_in=17, oc=8, k=3, s=2, p=1)


def _layers():
    conv1 = ConvLayer(name="conv1", **_C1, L=120, max_out_spikes=4000, init_k=5.0)
    conv2 = ConvLayer(name="conv2", **_C2, L=200, max_out_spikes=9000, init_k=5.0)
    out = FCLayer(name="out", n_in=conv2.n_neurons, n_out=10, chunk_size=64, init_k=5.0)
    return [conv1, conv2, out]


def _params(layers, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), len(layers))
    return tuple(layer.init_weight(k) for layer, k in zip(layers, keys))


def _raw_batch(key, n_samples, n_ev=300, h=34, w=34, ic=2):
    ks = jax.random.split(key, n_samples * 4)
    et = np.zeros((n_samples, n_ev), np.float32)
    xs = np.zeros((n_samples, n_ev), np.int32)
    ys = np.zeros((n_samples, n_ev), np.int32)
    cs = np.zeros((n_samples, n_ev), np.int32)
    nr = np.full((n_samples,), n_ev, np.int32)
    for i in range(n_samples):
        kt, kx, ky, kc = ks[4 * i:4 * i + 4]
        et[i] = np.sort(np.asarray(jax.random.uniform(kt, (n_ev,), minval=1.0, maxval=30.0)))
        xs[i] = np.asarray(jax.random.randint(kx, (n_ev,), 0, w))
        ys[i] = np.asarray(jax.random.randint(ky, (n_ev,), 0, h))
        cs[i] = np.asarray(jax.random.randint(kc, (n_ev,), 0, ic))
    return (et, xs, ys, cs, nr)


# ---------------------------------------------------------------------------

def test_due_gating():
    probe = TraceProbe("/tmp/_unused", _raw_batch(jax.random.PRNGKey(0), 2),
                        every=2, total_epochs=10)
    assert [e for e in range(10) if probe.due(e)] == [0, 2, 4, 6, 8, 9]  # 末 epoch 一定跑
    off = TraceProbe("/tmp/_unused", _raw_batch(jax.random.PRNGKey(0), 2),
                      every=0, total_epochs=10)
    assert not any(off.due(e) for e in range(10))


def test_full_due():
    probe = TraceProbe("/tmp/_unused", _raw_batch(jax.random.PRNGKey(0), 2),
                        every=1, total_epochs=10, full_every=3)
    assert [e for e in range(10) if probe._full_due(e)] == [0, 3, 6, 9]


def test_summary_written_and_matches_manual(tmp_path):
    layers, params = _layers(), _params(_layers())
    batch = _raw_batch(jax.random.PRNGKey(1), 3)
    probe = TraceProbe(str(tmp_path), batch, every=1, total_epochs=2)

    probe.run(layers, params, epoch=0)
    s = np.load(tmp_path / "summary.npz")

    assert list(s["epochs"]) == [0]
    for layer in layers:
        for key in ("spike_count", "s_value_sum", "v_final", "idle_frac"):
            arr = s[f"{layer.name}__{key}"]
            assert arr.shape == (1, layer.n_neurons)

    # 手動重算:逐樣本 run_network_traced -> 縮減 -> 對 3 筆平均
    h_in, w_in = layers[0].h_in, layers[0].w_in
    man = {layer.name: {k: np.zeros(layer.n_neurons) for k in
                        ("spike_count", "s_value_sum", "v_final", "idle_frac")}
           for layer in layers}
    for i in range(3):
        stream = raw_events_to_stream(*(jnp.asarray(v[i]) for v in batch), h_in, w_in)
        traces = run_network_traced(layers, stream, params)
        for layer, tr in zip(layers, traces):
            d = man[layer.name]
            d["spike_count"] += np.asarray(tr.spike_mask).sum(axis=1)
            d["s_value_sum"] += np.asarray(tr.s_value).sum(axis=1)
            d["v_final"] += np.asarray(tr.v_steps)[:, -1]
            d["idle_frac"] += np.isnan(np.asarray(tr.event_ms)).mean(axis=1)
    for layer in layers:
        for key in ("spike_count", "s_value_sum", "v_final", "idle_frac"):
            np.testing.assert_allclose(s[f"{layer.name}__{key}"][0],
                                        man[layer.name][key] / 3, rtol=1e-4, atol=1e-4)


def test_summary_accumulates_across_epochs(tmp_path):
    layers, params = _layers(), _params(_layers())
    probe = TraceProbe(str(tmp_path), _raw_batch(jax.random.PRNGKey(2), 2),
                        every=1, total_epochs=3)
    probe.run(layers, params, 0)
    probe.run(layers, params, 1)
    probe.run(layers, params, 2)
    s = np.load(tmp_path / "summary.npz")
    assert list(s["epochs"]) == [0, 1, 2]
    assert s["conv1__spike_count"].shape == (3, layers[0].n_neurons)


def test_rerecord_same_epoch_overwrites(tmp_path):
    layers, params = _layers(), _params(_layers())
    probe = TraceProbe(str(tmp_path), _raw_batch(jax.random.PRNGKey(3), 2),
                        every=1, total_epochs=3)
    probe.run(layers, params, 0)
    probe.run(layers, params, 1)
    probe.run(layers, params, 0)          # 出界重練:epoch 0 又跑一次
    s = np.load(tmp_path / "summary.npz")
    assert list(s["epochs"]) == [0, 1]    # 沒有第三列
    assert s["conv1__v_final"].shape == (2, layers[0].n_neurons)


def test_full_dump(tmp_path):
    layers, params = _layers(), _params(_layers())
    probe = TraceProbe(str(tmp_path), _raw_batch(jax.random.PRNGKey(4), 4),
                        every=1, total_epochs=4, full_every=2, full_samples=2)
    probe.run(layers, params, 0)          # full_due
    probe.run(layers, params, 1)          # 不是 full_due
    assert (tmp_path / "full_epoch_000.npz").is_file()
    assert not (tmp_path / "full_epoch_001.npz").is_file()

    f = np.load(tmp_path / "full_epoch_000.npz")
    for layer in layers:
        sm = f[f"{layer.name}__spike_mask"]
        assert sm.shape[0] == 2 and sm.shape[1] == layer.n_neurons
        assert sm.dtype == np.bool_
        for field in ("s_value", "v_steps", "event_ms"):
            assert f[f"{layer.name}__{field}"].shape == sm.shape


TESTS = [
    test_due_gating,
    test_full_due,
    test_summary_written_and_matches_manual,
    test_summary_accumulates_across_epochs,
    test_rerecord_same_epoch_overwrites,
    test_full_dump,
]

if __name__ == "__main__":
    import pathlib
    import tempfile

    for t in TESTS:
        if "tmp_path" in t.__code__.co_varnames:
            with tempfile.TemporaryDirectory() as d:
                t(pathlib.Path(d))
        else:
            t()
        print(f"PASS: {t.__name__}")
    print(f"\n全部 {len(TESTS)} 項通過")
