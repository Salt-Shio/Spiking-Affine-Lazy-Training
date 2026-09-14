"""salt_core/dormant.py 的測試。

分兩層:

- `dormant_score`(純歸約 `(n,)` -> dormant 統計):用合成活動量陣列,把
  「歸約公式對不對」跟「forward 對不對」(別處已驗證)分開測。
- `dormant_report`(逐層跑 forward + 分 chunk 累加):用小規模合成資料,確認
  接線 —— 只收 conv 層、輸出格式、分 chunk == 單一大批、跟手動歸約一致。
"""
import jax
import jax.numpy as jnp
import numpy as np

from salt_core.dormant import dormant_report, dormant_score
from salt_core.layers import ConvLayer, FCLayer, raw_events_to_stream

TOL = 1e-5

_C1 = dict(ic=2, h_in=34, w_in=34, oc=8, k=3, s=2, p=1)
_C2 = dict(ic=8, h_in=17, w_in=17, oc=16, k=3, s=2, p=1)


def _layers():
    conv1 = ConvLayer(name="conv1", **_C1, L=185, max_out_spikes=4000, init_k=5.0)
    conv2 = ConvLayer(name="conv2", **_C2, L=400, max_out_spikes=12000, init_k=5.0)
    out = FCLayer(name="out", n_in=conv2.n_neurons, n_out=10, chunk_size=512, init_k=5.0)
    return [conv1, conv2, out]


def _params(layers, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), len(layers))
    return tuple(layer.init_weight(k) for layer, k in zip(layers, keys))


def _synthetic_raw_batch(key, n_samples, max_len, h_in, w_in, ic):
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


# ============================================================================
# A. dormant_score:純歸約
# ============================================================================

def test_dormant_score_uniform_activity_zero_dormant():
    """全部一樣活躍 -> score 恆 1、dormant 比例 0。"""
    stats = dormant_score(np.full(100, 0.37), tau=0.1)
    assert stats["dormant_frac"] == 0.0
    np.testing.assert_allclose(stats["score"], 1.0, atol=TOL)


def test_dormant_score_bimodal_matches_fraction():
    """20% 飽和(~1.0)+ 80% 近零(~0.02):近零那批 score < 0.1 -> dormant ~0.8。"""
    act = np.concatenate([np.full(20, 1.0), np.full(80, 0.02)])
    stats = dormant_score(act, tau=0.1)
    assert abs(stats["dormant_frac"] - 0.8) < TOL


def test_dormant_score_all_zero_is_fully_dormant():
    stats = dormant_score(np.zeros(50), tau=0.1)
    assert stats["dormant_frac"] == 1.0
    np.testing.assert_array_equal(stats["score"], np.zeros(50))


def test_dormant_score_tau_is_inclusive_and_monotone():
    act = np.concatenate([np.full(10, 1.0), np.full(90, 0.05)])
    loose = dormant_score(act, tau=0.4)["dormant_frac"]
    tight = dormant_score(act, tau=0.1)["dormant_frac"]
    assert loose == 0.9
    assert tight == 0.0
    assert loose >= tight


# ============================================================================
# B. dormant_report:接線
# ============================================================================

def test_dormant_report_only_conv_layers_and_valid_shape():
    layers = _layers()
    params = _params(layers)
    batch = _synthetic_raw_batch(jax.random.PRNGKey(1), n_samples=6, max_len=20,
                                  h_in=34, w_in=34, ic=2)
    report = dormant_report(layers, params, batch, chunk=4)

    assert set(report) == {"conv1", "conv2"}, "FC 輸出層不該出現"
    for r in report.values():
        assert 0.0 <= r["dormant_frac"] <= 1.0


def test_dormant_report_chunking_is_invariant():
    """分 chunk 累加 == 一次全批。"""
    layers = _layers()
    params = _params(layers, seed=2)
    batch = _synthetic_raw_batch(jax.random.PRNGKey(3), n_samples=6, max_len=20,
                                  h_in=34, w_in=34, ic=2)
    one = dormant_report(layers, params, batch, chunk=6)
    many = dormant_report(layers, params, batch, chunk=2)
    for name in one:
        assert abs(one[name]["dormant_frac"] - many[name]["dormant_frac"]) < TOL


def test_dormant_report_matches_manual_reduction():
    """dormant_report 的數字 == 手動 vmap forward + sum(spike_mask) + dormant_score。"""
    layers = _layers()
    params = _params(layers, seed=4)
    n = 6
    batch = _synthetic_raw_batch(jax.random.PRNGKey(5), n_samples=n, max_len=20,
                                  h_in=34, w_in=34, ic=2)
    et, x, y, c, nr = batch
    conv1 = layers[0]

    def one(e, xx, yy, cc, rr):
        s = raw_events_to_stream(e, xx, yy, cc, rr, conv1.h_in, conv1.w_in)
        _out, result, _diag = conv1.forward(params[0], s)
        return jnp.sum(result.spike_mask, axis=1)

    per_sample = jax.vmap(one)(et, x, y, c, nr)          # (n, n_neurons)
    activity = np.asarray(jnp.mean(per_sample, axis=0))
    expect = dormant_score(activity, tau=0.1)
    got = dormant_report(layers, params, batch, chunk=4)["conv1"]
    assert abs(got["dormant_frac"] - expect["dormant_frac"]) < TOL


def test_dormant_report_s_value_activity_runs():
    layers = _layers()
    params = _params(layers, seed=6)
    batch = _synthetic_raw_batch(jax.random.PRNGKey(7), n_samples=4, max_len=18,
                                  h_in=34, w_in=34, ic=2)
    report = dormant_report(layers, params, batch, activity="s_value", chunk=2)
    assert set(report) == {"conv1", "conv2"}
    for r in report.values():
        assert 0.0 <= r["dormant_frac"] <= 1.0


def test_dormant_report_rejects_bad_activity():
    layers = _layers()
    params = _params(layers)
    batch = _synthetic_raw_batch(jax.random.PRNGKey(8), n_samples=2, max_len=12,
                                  h_in=34, w_in=34, ic=2)
    try:
        dormant_report(layers, params, batch, activity="spikes")
    except ValueError:
        pass
    else:
        raise AssertionError("activity 打錯字應該 raise ValueError")


TESTS = [
    test_dormant_score_uniform_activity_zero_dormant,
    test_dormant_score_bimodal_matches_fraction,
    test_dormant_score_all_zero_is_fully_dormant,
    test_dormant_score_tau_is_inclusive_and_monotone,
    test_dormant_report_only_conv_layers_and_valid_shape,
    test_dormant_report_chunking_is_invariant,
    test_dormant_report_matches_manual_reduction,
    test_dormant_report_s_value_activity_runs,
    test_dormant_report_rejects_bad_activity,
]


if __name__ == "__main__":
    for test in TESTS:
        test()
        print(f"PASS: {test.__name__}")
    print(f"\n全部 {len(TESTS)} 項測試通過")
