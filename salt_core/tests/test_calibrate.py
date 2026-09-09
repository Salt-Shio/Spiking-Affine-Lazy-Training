"""salt_core/calibrate.py 的測試。

分兩層:

- `calibrate_init_scale`(通用 bracket + 幾何二分):用假造的解析 measure 函式
  (不跑真的 SNN forward),把「演算法對不對」跟「SNN forward 對不對」
  (別處已驗證)分開測。
- `calibrate_network`(跨層前向順序解 init_k):用小規模合成資料,確認接線——
  `init_k is None` 的層被填上、FC 層跳過 / 沒 calibration_measure 會報錯、
  `ConvLayer.calibration_measure` 產出的 measure 對 init_k 單調。
"""
import math
from dataclasses import replace

import jax
import jax.numpy as jnp

from salt_core.calibrate import CalibrationError, calibrate_init_scale, calibrate_network
from salt_core.layers import ConvLayer, FCLayer, raw_events_to_stream, uniform_init

BAND = (0.20, 0.50)

# 規格書 conv 幾何(直接寫成 literal,不 import src——salt_core 測試不依賴 src)。
# h_out / w_out 不寫:是 ConvLayer 從 h_in/k/s/p 算的 property。
_C1 = dict(ic=2, h_in=34, w_in=34, oc=8, k=3, s=2, p=1)
_C2 = dict(ic=8, h_in=17, w_in=17, oc=16, k=3, s=2, p=1)


def _layers(init_k1=None, init_k2=None, init_k_fc=5.0):
    # 動力學欄位(tau=16 / v_th=1.0 / alpha=2.0 / chunk_size=1;FC 的 v_th=1e9)
    # 都吃 ConvLayer / FCLayer 的預設,這裡只寫跟預設不同的:L / max_out_spikes
    # (校準跑真 forward,佇列要夠長不被截)、FC 的 chunk_size。
    conv1 = ConvLayer(name="conv1", **_C1, L=185, max_out_spikes=4000, init_k=init_k1)
    conv2 = ConvLayer(name="conv2", **_C2, L=400, max_out_spikes=12000, init_k=init_k2)
    out = FCLayer(name="out", n_in=conv2.n_neurons, n_out=10,
                   chunk_size=512, init_k=init_k_fc)
    return [conv1, conv2, out]


# ============================================================================
# A. calibrate_init_scale:單調遞增函式,驗證能找到落帶內的 init_k
# ============================================================================

def test_bracket_start_already_in_band():
    res = calibrate_init_scale(lambda k: k, lambda w: 0.35, BAND)
    assert res.init_k == 1.0
    assert res.converged_reason == "band"
    assert len(res.curve) == 1, "已經落帶內不該多量測"


def test_needs_bracket_doubling_then_bisection():
    res = calibrate_init_scale(lambda k: k, lambda w: min(1.0, w / 50.0), BAND)
    assert BAND[0] <= res.measured <= BAND[1]
    assert res.converged_reason == "band"
    assert len(res.curve) > 1


def test_needs_bracket_halving_then_bisection():
    # min(1.0, 5*k^3):單調遞增,k=1 時 =1.0(遠高於帶),要不斷減半。三次方
    # 讓連續兩步減半的比值(8x)大於帶寬比(2.5x),保證跳過整個帶、逼出 bisection。
    res = calibrate_init_scale(lambda k: k, lambda w: min(1.0, 5.0 * w ** 3), BAND)
    assert BAND[0] <= res.measured <= BAND[1]
    assert res.converged_reason == "band"
    assert len(res.curve) > 2


def test_direct_hit_during_bracket_walk():
    # k=1(0.05)->2(0.10)->4(0.20=band_lo,落帶內)
    res = calibrate_init_scale(lambda k: k, lambda w: min(1.0, w / 20.0), BAND)
    assert res.init_k == 4.0
    assert res.converged_reason == "band"
    assert len(res.curve) == 3


# ============================================================================
# B. calibrate_init_scale:掃不到要 raise,不靜默妥協
# ============================================================================

def test_raises_when_band_unreachable_within_bounds():
    try:
        calibrate_init_scale(lambda k: k, lambda w: 0.05, BAND, name="dead", hi=1e3)
        assert False, "應該 raise CalibrationError"
    except CalibrationError as e:
        assert "dead" in str(e)


def test_raises_when_bracket_max_iter_exhausted_before_reaching_bound():
    try:
        calibrate_init_scale(lambda k: k, lambda w: min(1.0, w / 50.0), BAND,
                              bracket_factor=1.01, bracket_max_iter=5, hi=1e6)
        assert False, "應該 raise CalibrationError"
    except CalibrationError:
        pass


# ============================================================================
# C. calibrate_init_scale:寬度收斂(次要條件)——measure 卡在帶邊界震盪
# ============================================================================

def test_width_convergence_when_measure_oscillates_at_band_edge():
    # 階梯函式:k<10 -> 0.1(<band_lo)、k>=10 -> 0.9(>band_hi),帶裡永遠沒有解。
    res = calibrate_init_scale(lambda k: k, lambda w: 0.1 if w < 10.0 else 0.9, BAND,
                                bisect_rel_tol=0.01)
    assert res.converged_reason == "width"
    assert math.isfinite(res.init_k)
    assert res.measured in (0.1, 0.9)


# ============================================================================
# D. calibrate_network:跨層前向順序解 init_k
# ============================================================================

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


def test_calibrate_network_fills_missing_init_k_in_forward_order():
    """conv1/conv2 沒填 init_k -> calibrate_network 兩層都填上具體值;FC 有填
    -> 原封不動。用寬帶 (0,1) 讓掃描在起點就收斂,只驗證接線(不是演算法)。"""
    layers = _layers(init_k1=None, init_k2=None, init_k_fc=5.0)
    assert layers[0].init_k is None and layers[1].init_k is None

    batch = _synthetic_raw_batch(jax.random.PRNGKey(0), n_samples=4, max_len=24,
                                  h_in=34, w_in=34, ic=2)
    resolved = calibrate_network(layers, batch, key=jax.random.PRNGKey(1),
                                  band=(0.0, 1.0), measure_chunk=2)

    assert resolved[0].init_k is not None and resolved[0].init_k > 0
    assert resolved[1].init_k is not None and resolved[1].init_k > 0
    assert resolved[2].init_k == 5.0, "FC 有填的 init_k 不該被動"
    for f in ("name", "oc", "h_out", "L", "max_out_spikes", "L_grow_factor"):
        assert getattr(resolved[0], f) == getattr(layers[0], f)


def test_calibrate_network_raises_when_a_non_searchable_layer_lacks_init_k():
    """FC 層沒有 calibration_measure(v_th 純積分器沒 firing-rate 準則)。
    如果它的 init_k 留 None,calibrate_network 要明確報錯,不是默默跳過。"""
    layers = _layers(init_k1=8.0, init_k2=64.0, init_k_fc=None)

    batch = _synthetic_raw_batch(jax.random.PRNGKey(2), n_samples=3, max_len=20,
                                  h_in=34, w_in=34, ic=2)
    try:
        calibrate_network(layers, batch, key=jax.random.PRNGKey(3), band=(0.0, 1.0),
                           measure_chunk=2)
        assert False, "應該 raise CalibrationError"
    except CalibrationError as e:
        assert "out" in str(e)


def test_conv_layer_calibration_measure_monotonic_in_init_k():
    """ConvLayer.calibration_measure 產出的 measure:大 init_k 的 firing rate
    不小於小 init_k(定性,不是精確數學)。"""
    conv1 = _layers()[0]
    batch = _synthetic_raw_batch(jax.random.PRNGKey(4), n_samples=4, max_len=24,
                                  h_in=34, w_in=34, ic=2)
    stream_batch = jax.vmap(raw_events_to_stream, in_axes=(0, 0, 0, 0, 0, None, None))(
        *batch, conv1.h_in, conv1.w_in)

    measure = conv1.calibration_measure(stream_batch, chunk=2)
    key = jax.random.PRNGKey(5)
    small = measure(uniform_init(key, conv1.weight_shape, conv1.fan_in, 0.01))
    large = measure(uniform_init(key, conv1.weight_shape, conv1.fan_in, 100.0))
    assert 0.0 <= small <= large <= 1.0, f"small={small} large={large}"


TESTS = [
    test_bracket_start_already_in_band,
    test_needs_bracket_doubling_then_bisection,
    test_needs_bracket_halving_then_bisection,
    test_direct_hit_during_bracket_walk,
    test_raises_when_band_unreachable_within_bounds,
    test_raises_when_bracket_max_iter_exhausted_before_reaching_bound,
    test_width_convergence_when_measure_oscillates_at_band_edge,
    test_calibrate_network_fills_missing_init_k_in_forward_order,
    test_calibrate_network_raises_when_a_non_searchable_layer_lacks_init_k,
    test_conv_layer_calibration_measure_monotonic_in_init_k,
]


if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n全部 {len(TESTS)} 項測試通過")
