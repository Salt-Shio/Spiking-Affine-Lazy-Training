"""salt_core/quantize.py 的測試。推導見 docs/math/權重量化推導.md。

分三層:

- `fake_quantize_tensor`(線性量化 primitive):round-trip 誤差有界、bits 越多
  誤差越小、per-channel threshold 正確對到指定 axis、全零 channel 不除以零、
  `bits<2` 擋掉。
- `quantization_error`:跟手動算的統計對得上,恆等輸入回傳 mse=0/sqnr=inf。
- `quantize_params`:接線正確(shape/對齊 layers 不變),`clip_percentile=100`
  等同 `fake_quantize_tensor` 沒給 `threshold` 時的預設 max-abs 行為。
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from salt_core.layers import ConvLayer, FCLayer
from salt_core.quantize import (apply_decay_table_int, build_decay_table,
                                build_decay_table_int, delta_t_max, fake_quantize_tensor,
                                iv_from_measurement, iv_layer, iv_positive_lower_bound,
                                quantization_error, quantize_params, quantize_to_int,
                                round_half_away_from_zero, round_shift, v_th_to_int,
                                waste, wide_mul_shift, wrap_to_bits)

TOL = 1e-5

_C1 = dict(ic=2, h_in=34, w_in=34, oc=8, k=3, s=2, p=1)


def _layers():
    conv1 = ConvLayer(name="conv1", **_C1, L=185, max_out_spikes=4000, init_k=5.0)
    out = FCLayer(name="out", n_in=conv1.n_neurons, n_out=10, chunk_size=512, init_k=5.0)
    return [conv1, out]


def _params(layers, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), len(layers))
    return tuple(layer.init_weight(k) for layer, k in zip(layers, keys))


# ============================================================================
# A. fake_quantize_tensor
# ============================================================================

def test_fake_quantize_tensor_all_zero_input_maps_to_zero():
    """全零輸入:threshold 全零 fallback 成 1.0,不除以零,量化結果仍是 0。"""
    x = jnp.zeros((4, 4))
    x_hat, scale = fake_quantize_tensor(x, bits=8)
    assert np.allclose(np.asarray(x_hat), 0.0)
    assert float(scale) == pytest.approx(1.0 / (2 ** 7 - 1))


def test_fake_quantize_tensor_roundtrip_error_bounded_by_half_step():
    """round-to-nearest,誤差上界是半個量化步長(沒有值超出 threshold 時)。"""
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, (1000,), minval=-3.0, maxval=3.0)
    x_hat, scale = fake_quantize_tensor(x, bits=8)
    err = np.abs(np.asarray(x - x_hat))
    assert np.all(err <= float(scale) / 2 + TOL)


def test_fake_quantize_tensor_more_bits_lower_mse():
    """同一份資料,bits 越多量化噪聲(mse)應該越小(推導文件步驟 2)。"""
    key = jax.random.PRNGKey(1)
    x = jax.random.uniform(key, (2000,), minval=-5.0, maxval=5.0)
    mses = []
    for bits in (2, 4, 6, 8):
        x_hat, _ = fake_quantize_tensor(x, bits=bits)
        mses.append(quantization_error(x, x_hat)["mse"])
    assert all(a > b for a, b in zip(mses, mses[1:]))


def test_fake_quantize_tensor_clips_outlier_to_threshold():
    """超過 threshold 的值被夾到 threshold,不是照原值量化。"""
    x = jnp.array([0.0, 1.0, 100.0])
    x_hat, scale = fake_quantize_tensor(x, bits=8, threshold=1.0)
    assert float(x_hat[2]) == pytest.approx(1.0, abs=float(scale))
    assert float(x_hat[0]) == pytest.approx(0.0, abs=TOL)


def test_fake_quantize_tensor_per_channel_axis0_independent_threshold():
    """`axis=0` 時每個 channel(row)各自的 threshold 只看自己那一行。"""
    x = jnp.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    x_hat, scale = fake_quantize_tensor(x, bits=8, axis=0)
    assert scale.shape == (2, 1)
    assert float(scale[0, 0]) == pytest.approx(3.0 / (2 ** 7 - 1), rel=1e-6)
    assert float(scale[1, 0]) == pytest.approx(30.0 / (2 ** 7 - 1), rel=1e-6)


def test_fake_quantize_tensor_all_zero_channel_no_nan():
    """per-channel 量化下,某個 channel 全零不該讓那個 channel 出現 NaN。"""
    x = jnp.array([[0.0, 0.0], [1.0, 2.0]])
    x_hat, _scale = fake_quantize_tensor(x, bits=8, axis=0)
    assert not np.any(np.isnan(np.asarray(x_hat)))
    assert np.allclose(np.asarray(x_hat[0]), 0.0)


def test_fake_quantize_tensor_rejects_bits_below_2():
    with pytest.raises(ValueError):
        fake_quantize_tensor(jnp.ones((3,)), bits=1)


def test_fake_quantize_tensor_mode_truncate_rounds_toward_zero():
    """`mode="truncate"` 對稱量化下等於向零捨去,跟預設的 `"round"` 在
    非整數中點的地方要算出不同結果(膜電位量化的捨入規則 r(·) 兩種都要能選,
    見 docs/math/膜電位量化推導.md)。threshold 取 127 讓 scale 精確等於 1.0,
    量化前的值直接就是量化碼,方便手算對答案。"""
    x = jnp.array([0.6, -0.6, 1.4, -1.4])
    x_round, scale = fake_quantize_tensor(x, bits=8, threshold=127.0, mode="round")
    x_trunc, _ = fake_quantize_tensor(x, bits=8, threshold=127.0, mode="truncate")

    assert float(scale) == pytest.approx(1.0)
    assert np.allclose(np.asarray(x_round), [1.0, -1.0, 1.0, -1.0])
    assert np.allclose(np.asarray(x_trunc), [0.0, 0.0, 1.0, -1.0])


def test_fake_quantize_tensor_rejects_invalid_mode():
    with pytest.raises(ValueError):
        fake_quantize_tensor(jnp.ones((3,)), bits=8, mode="ceil")


def test_fake_quantize_tensor_round_mode_ties_away_from_zero_not_to_even():
    """第十八節定案:FPGA 是逢五進一,`mode="round"` 不能再是 jnp.round 的
    逢五取偶。threshold=127 讓 scale=1.0,2.5/-2.5 是中點,逢五取偶會給
    2.0/-2.0,這裡要的是 3.0/-3.0。"""
    x = jnp.array([2.5, -2.5])
    x_hat, scale = fake_quantize_tensor(x, bits=8, threshold=127.0, mode="round")
    assert float(scale) == pytest.approx(1.0)
    assert np.allclose(np.asarray(x_hat), [3.0, -3.0])


# ============================================================================
# B. quantization_error
# ============================================================================

def test_quantization_error_identical_arrays_zero_mse_inf_sqnr():
    x = jnp.array([1.0, -2.0, 3.0])
    stats = quantization_error(x, x)
    assert stats["mse"] == 0.0
    assert stats["max_abs_err"] == 0.0
    assert stats["sqnr_db"] == float("inf")


def test_quantization_error_matches_manual_computation():
    x = jnp.array([1.0, -1.0, 2.0, -2.0])
    x_hat = jnp.array([1.5, -0.5, 2.5, -1.5])
    stats = quantization_error(x, x_hat)
    err = np.asarray(x - x_hat)
    assert stats["mse"] == pytest.approx(float(np.mean(err ** 2)), rel=1e-6)
    assert stats["max_abs_err"] == pytest.approx(float(np.max(np.abs(err))), rel=1e-6)


# ============================================================================
# C. quantize_params
# ============================================================================

def test_quantize_params_preserves_shapes_and_alignment():
    layers = _layers()
    params = _params(layers)
    q = quantize_params(layers, params, bits=8)
    assert len(q) == len(layers)
    for w, w_q in zip(params, q):
        assert w_q.shape == w.shape


def test_quantize_params_clip_percentile_100_equals_max_abs_default():
    """`clip_percentile=100` 應該跟 `fake_quantize_tensor` 沒給 `threshold`
    時的預設(max-abs)行為一致——100th percentile 精確等於最大值。"""
    layers = _layers()
    params = _params(layers)
    q_pct = quantize_params(layers, params, bits=8, per_channel=True, clip_percentile=100.0)
    q_direct = tuple(fake_quantize_tensor(w, bits=8, axis=0)[0] for w in params)
    for a, b in zip(q_pct, q_direct):
        assert np.allclose(np.asarray(a), np.asarray(b), atol=TOL)


def test_quantize_params_lower_percentile_increases_error():
    """門檻夾得比 100th percentile 窄,uniform 初始化的權重下截斷誤差主導,
    整體 mse 不該系統性變小(推導文件步驟 2 的 trade-off 方向性檢查)。"""
    layers = _layers()
    params = _params(layers)
    q_full = quantize_params(layers, params, bits=8, clip_percentile=100.0)
    q_clip = quantize_params(layers, params, bits=8, clip_percentile=50.0)
    err_full = sum(quantization_error(w, w_hat)["mse"] for w, w_hat in zip(params, q_full))
    err_clip = sum(quantization_error(w, w_hat)["mse"] for w, w_hat in zip(params, q_clip))
    assert err_clip >= err_full - TOL


# ============================================================================
# D. 膜電位量化:delta_t_max / build_decay_table(推導見
#    docs/math/膜電位量化推導.md;整數版查表 build_decay_table_int/
#    apply_decay_table_int 的測試在下面 G 節)
# ============================================================================

def test_delta_t_max_boundary_property():
    """delta_t_max 是「量化後仍非零」的最大 Δt:自己這格要 >= eps(半個最小
    刻度),下一格要 < eps——不驗證某個硬編碼的數字,直接驗證這條邊界定義
    本身,對兩組不同的 (f_a, tau) 都要成立。"""
    for f_a, tau in [(8, 16.0), (4, 4.0), (12, 16.0)]:
        d = delta_t_max(f_a, tau)
        eps = 2.0 ** -(f_a + 1)
        base = 1.0 - 1.0 / tau
        assert base ** d >= eps, f"f_a={f_a},tau={tau}: 邊界格自己應該還 >= eps"
        assert base ** (d + 1) < eps, f"f_a={f_a},tau={tau}: 再往後一格應該 < eps"


def test_build_decay_table_matches_hand_computed_values():
    """f_a=4, tau=4.0(base=0.75)手算前三項:0.75、0.5625 剛好落在 1/16 網格
    上,量化前後不變;0.75**3=0.421875 沒有剛好落在網格上,0.421875*16=6.75
    四捨五入到 7,量化後是 7/16=0.4375。"""
    table = build_decay_table(f_a=4, tau=4.0)
    assert table.shape == (delta_t_max(4, 4.0),)
    assert float(table[0]) == pytest.approx(0.75, abs=TOL)
    assert float(table[1]) == pytest.approx(0.5625, abs=TOL)
    assert float(table[2]) == pytest.approx(0.4375, abs=TOL)


def test_build_decay_table_monotonic_and_in_range():
    table = np.asarray(build_decay_table(f_a=8, tau=16.0))
    assert np.all(table >= 0.0) and np.all(table < 1.0)
    assert np.all(np.diff(table) <= 0.0), "a_k 隨 Δt 增大應該單調不增"


# ============================================================================
# E. 膜電位量化:i_V 公式(推導見 docs/math/膜電位量化推導.md「i_V 怎麼選」節)
# ============================================================================

def test_iv_positive_lower_bound_matches_hand_computation():
    # v_th_tilde=1.0, b=8: max_q=127, log2(128)=7, +1=8
    assert iv_positive_lower_bound(v_th_tilde=1.0, b=8) == 8
    # v_th_tilde=1.0, b=6: max_q=31, log2(32)=5, +1=6
    assert iv_positive_lower_bound(v_th_tilde=1.0, b=6) == 6


def test_iv_from_measurement_matches_hand_computation():
    # M=2.0, T_c=1.0, b=8: levels=127, log2(2*127)=log2(254)≈7.988, ceil=8, +1=9
    assert iv_from_measurement(M=2.0, T_c=1.0, b=8) == 9


def test_iv_from_measurement_rejects_non_positive_measurement():
    with pytest.raises(ValueError):
        iv_from_measurement(M=0.0, T_c=1.0, b=8)


def test_iv_layer_and_waste():
    per_channel = [3, 5, 4]
    assert iv_layer(per_channel) == 5
    assert waste(iv_layer(per_channel), per_channel) == [2, 0, 1]


# ============================================================================
# F. 膜電位量化 第二輪:整數尺度運算 primitive(見 docs/問題紀錄.md 第十七~
#    十九節)。round_half_away_from_zero / round_shift / wrap_to_bits
# ============================================================================

def test_round_half_away_from_zero_ties_go_away_from_zero_not_to_even():
    """跟 jnp.round 的逢五取偶對照:2.5/-2.5/1.5/-1.5 這幾個中點,`jnp.round`
    會給 2/-2/2/-2(取偶),這裡要的逢五進一是 3/-3/2/-2。"""
    x = jnp.array([2.5, -2.5, 1.5, -1.5])
    result = np.asarray(round_half_away_from_zero(x))
    assert np.allclose(result, [3.0, -3.0, 2.0, -2.0])
    # 對照組:確認 jnp.round 在這組數字上真的是取偶,不是這裡誤判
    assert np.allclose(np.asarray(jnp.round(x)), [2.0, -2.0, 2.0, -2.0])


def test_round_half_away_from_zero_non_tie_values():
    x = jnp.array([0.4, 0.6, -0.4, -0.6, 0.0])
    result = np.asarray(round_half_away_from_zero(x))
    assert np.allclose(result, [0.0, 1.0, 0.0, -1.0, 0.0])


def test_round_shift_round_mode_matches_hand_computation():
    """shift_bits=2(除以 4):6/4=1.5 是中點,逢五進一是 2(-6→-2);
    5/4=1.25 非中點捨去是 1;7/4=1.75 非中點進位是 2。"""
    x = jnp.array([6, -6, 5, 7])
    result = np.asarray(round_shift(x, shift_bits=2, mode="round"))
    assert np.array_equal(result, [2, -2, 1, 2])


def test_round_shift_truncate_mode_rounds_toward_zero():
    x = jnp.array([7, -7, 5, -5])
    result = np.asarray(round_shift(x, shift_bits=2, mode="truncate"))
    assert np.array_equal(result, [1, -1, 1, -1])


def test_round_shift_zero_shift_bits_is_identity():
    x = jnp.array([3, -3, 0])
    assert np.array_equal(np.asarray(round_shift(x, shift_bits=0, mode="round")), [3, -3, 0])
    assert np.array_equal(np.asarray(round_shift(x, shift_bits=0, mode="truncate")), [3, -3, 0])


def test_round_shift_rejects_invalid_mode():
    with pytest.raises(ValueError):
        round_shift(jnp.array([1]), shift_bits=2, mode="ceil")


def test_round_shift_decay_step_never_grows_magnitude():
    """對應 docstring 裡的證明:只要 a_int < 2^shift_bits(衰減嚴格小於 1),
    捨入後的量值不會超過捨入前的量值,亂數測試這個性質,不挑手算案例。"""
    v_key, a_key = jax.random.split(jax.random.PRNGKey(0))
    shift_bits = 6
    v = jax.random.randint(v_key, (2000,), minval=-100000, maxval=100000)
    a_int = jax.random.randint(a_key, (2000,), minval=0, maxval=2 ** shift_bits)
    decayed_wide = a_int * v
    result = round_shift(decayed_wide, shift_bits=shift_bits, mode="round")
    assert np.all(np.abs(np.asarray(result)) <= np.abs(np.asarray(v)))


# ---- wrap_to_bits ----

def test_wrap_to_bits_in_range_values_unchanged_no_overflow():
    total_bits = 4  # 有號範圍 [-8, 7]
    x = jnp.array([7, -8, 0, 3])
    wrapped, overflowed = wrap_to_bits(x, total_bits)
    assert np.array_equal(np.asarray(wrapped), np.asarray(x))
    assert not np.any(np.asarray(overflowed))


def test_wrap_to_bits_positive_overflow_wraps_to_negative():
    """total_bits=4,範圍 [-8,7]。9 超出正向邊界一格,兩補數繞回去是
    9-16=-7(手算:9=0b1001,當成 4 位元有號數,最高位是符號位,值是
    9-16=-7)。"""
    wrapped, overflowed = wrap_to_bits(jnp.array([9]), total_bits=4)
    assert int(wrapped[0]) == -7
    assert bool(overflowed[0])


def test_wrap_to_bits_negative_overflow_wraps_to_positive():
    """-9 超出負向邊界一格,mod 16 繞回去是 -9+16=7。"""
    wrapped, overflowed = wrap_to_bits(jnp.array([-9]), total_bits=4)
    assert int(wrapped[0]) == 7
    assert bool(overflowed[0])


def test_wrap_to_bits_boundary_values_exactly_at_edge_no_overflow():
    total_bits = 4
    wrapped, overflowed = wrap_to_bits(jnp.array([7, -8]), total_bits)
    assert np.array_equal(np.asarray(wrapped), [7, -8])
    assert not np.any(np.asarray(overflowed))


# ============================================================================
# G. 膜電位量化 第二輪:quantize_to_int / build_decay_table_int /
#    apply_decay_table_int / v_th_to_int(見 docs/問題紀錄.md 第十七節)
# ============================================================================

def test_quantize_to_int_q_times_scale_equals_fake_quantize_tensor():
    """`quantize_to_int` 跟 `fake_quantize_tensor` 現在共用同一套底層邏輯,
    q*scale 應該精確等於 fake_quantize_tensor 吐出來的 x_hat。"""
    key = jax.random.PRNGKey(2)
    x = jax.random.uniform(key, (200,), minval=-5.0, maxval=5.0)
    x_hat, scale_a = fake_quantize_tensor(x, bits=8)
    q, scale_b = quantize_to_int(x, bits=8)
    assert float(scale_a) == pytest.approx(float(scale_b))
    assert np.allclose(np.asarray(q) * np.asarray(scale_b), np.asarray(x_hat), atol=TOL)


def test_build_decay_table_int_matches_build_decay_table_scaled_back():
    """整數表除以 2^f_a 應該精確等於原本的浮點表(同一組數字,只是還沒
    乘回 2^-f_a)。"""
    f_a, tau = 4, 4.0
    table_float = np.asarray(build_decay_table(f_a, tau))
    table_int = np.asarray(build_decay_table_int(f_a, tau))
    assert table_int.dtype == np.int32
    assert np.allclose(table_int / (2 ** f_a), table_float, atol=TOL)
    # 手算三項(跟 test_build_decay_table_matches_hand_computed_values 對應):
    # 0.75*16=12、0.5625*16=9、0.4375*16=7
    assert list(table_int[:3]) == [12, 9, 7]


def test_apply_decay_table_int_three_regimes():
    tau = 4.0
    table_int = build_decay_table_int(f_a=4, tau=tau)
    delta_t = jnp.array([0, 1, 2, 3, 60])  # 60 遠超過這組 (f_a,tau) 的表深度
    a_int, is_identity = apply_decay_table_int(delta_t, table_int)
    assert list(np.asarray(is_identity)) == [True, False, False, False, False]
    # Δt=0 那格是 identity,a_int 的值沒有意義,不檢查;其餘三格查表值(12/9/7,
    # 跟 test_build_decay_table_int_matches_build_decay_table_scaled_back 手算
    # 的前三項一致)+ 超出表深度那格是 0
    assert list(np.asarray(a_int)[1:]) == [12, 9, 7, 0]


def test_v_th_to_int_matches_hand_computation():
    # 1.95/0.3=6.5,*2^2=26.0(不是中點,浮點雜訊不影響取整結果)
    assert int(v_th_to_int(v_th=1.95, s_c=0.3, f_V=2)) == 26


def test_v_th_to_int_ties_away_from_zero():
    # v_th_tilde = 2.5/1.0 = 2.5,f_V=0 時 *2^0=2.5,剛好是中點,逢五進一是 3
    assert int(v_th_to_int(v_th=2.5, s_c=1.0, f_V=0)) == 3


# ============================================================================
# H. wide_mul_shift(見 docs/問題紀錄.md 第十七節下方討論:round_shift(a*v,..)
#    直接算 a*v 這個乘積,shift_bits 跟 v 的位元數加起來常常超過 int32,
#    這裡拆成高低兩半分開乘,兩個位元寬度限制各自獨立、不綁在一起)
# ============================================================================

def test_wide_mul_shift_matches_naive_round_shift_when_product_fits_int32():
    """乘積本身沒有寬到會溢位時,結果要跟直接算 round_shift(a*v,shift) 一樣——
    a_int=12,v_int=40,shift=4:480/16=30,整除沒有捨入。"""
    naive = round_shift(jnp.array(12 * 40), shift_bits=4, mode="round")
    wide = wide_mul_shift(jnp.array(12), jnp.array(40), shift_bits=4, mode="round")
    assert int(wide) == int(naive) == 30


def test_wide_mul_shift_negative_v_tie_must_take_abs_before_splitting():
    """迴歸測試:一開始的錯誤設計是直接對帶符號的 v_int 做 hi=v>>shift_bits
    (有號右移)去拆高低位,手算發現在負數 + 捨入卡中點時會算錯。
    a_int=1, v_int=-8, shift_bits=4:正確答案是 round_shift(1*(-8),4)=
    round_shift(-8,4)=-1(-8/16=-0.5,逢五進一離零方向到 -1)。如果直接對
    -8 做有號右移拆分(hi=-8>>4=-1, lo=8),再用「H+round_shift(L)」組合會
    算成 -1+round_shift(8,4)=-1+1=0——錯誤答案。這裡驗證 wide_mul_shift
    (正確版:先取絕對值再拆)給出 -1,不是 0。"""
    result = wide_mul_shift(jnp.array(1), jnp.array(-8), shift_bits=4, mode="round")
    assert int(result) == -1


def test_wide_mul_shift_truncate_mode_matches_round_shift_on_small_values():
    naive = round_shift(jnp.array(1 * 7), shift_bits=2, mode="truncate")
    wide = wide_mul_shift(jnp.array(1), jnp.array(7), shift_bits=2, mode="truncate")
    assert int(wide) == int(naive) == 1


def test_wide_mul_shift_zero_a_int_gives_zero():
    assert int(wide_mul_shift(jnp.array(0), jnp.array(12345), shift_bits=8, mode="round")) == 0


def test_wide_mul_shift_rejects_invalid_mode():
    with pytest.raises(ValueError):
        wide_mul_shift(jnp.array(1), jnp.array(1), shift_bits=2, mode="ceil")


def test_wide_mul_shift_matches_exact_python_int_arithmetic_random_wide_values():
    """亂數交叉驗證:用 Python 原生大整數(不受 int32/int64 限制,這裡只是
    測試本身要算 oracle,不是量化模擬要用的路徑,可以放心用)當基準,對照
    wide_mul_shift 在乘積本身就會溢位 int32 的寬參數範圍下(shift_bits=15,
    v_int 接近 i_V+f_V=30 位元)還是不是算對。"""
    a_key, v_key = jax.random.split(jax.random.PRNGKey(3))
    shift_bits = 15
    a_vals = jax.random.randint(a_key, (500,), minval=0, maxval=2 ** shift_bits)
    v_vals = jax.random.randint(v_key, (500,), minval=-(2 ** 29), maxval=2 ** 29)

    for mode in ("round", "truncate"):
        result = np.asarray(wide_mul_shift(a_vals, v_vals, shift_bits=shift_bits, mode=mode))
        for a, v, got in zip(np.asarray(a_vals), np.asarray(v_vals), result):
            a, v = int(a), int(v)
            exact = a * v  # Python 原生大整數乘法,不受 32 位元限制
            if mode == "round":
                expected = (1 if exact >= 0 else -1) * ((abs(exact) + (1 << (shift_bits - 1))) >> shift_bits)
            else:
                expected = (1 if exact >= 0 else -1) * (abs(exact) >> shift_bits)
            assert int(got) == expected, f"a={a},v={v},mode={mode}: got {got}, expected {expected}"
