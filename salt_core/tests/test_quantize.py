"""salt_core/quant/codes.py、salt_core/quant/ptq.py 的測試。推導見 docs/math/權重量化推導.md、
docs/math/膜電位量化推導.md。

- A~C 權重量化:`fake_quantize_tensor`/`quantize_to_int`(round-trip 誤差有界、
  bits 越多誤差越小、per-channel threshold、全零 channel、`bits<2` 擋掉)、
  `quantization_error`、`quantize_params`。
- D 衰減查表:`delta_t_max`/`build_decay_table_int`/`apply_decay_table_int`。
- E $i_V$ 公式:`iv_from_measurement`/`iv_layer`/`waste`。
- F 離線整數換算:`round_half_away_from_zero`/`v_th_to_int`。

逐事件遞迴用的定點數運算(`round_shift` 等)在 test_fixed_point.py。
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from salt_core.layers import ConvLayer, FCLayer
from salt_core.quant.codes import (apply_decay_table_int, build_decay_table_int, delta_t_max,
                                   iv_from_measurement, iv_layer, max_weight_code,
                                   quantize_to_int, round_half_away_from_zero, v_th_to_int,
                                   waste)
from salt_core.quant.ptq import fake_quantize_tensor, quantization_error, quantize_params

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


def test_fake_quantize_tensor_ties_away_from_zero_not_to_even():
    """第十八節:權重碼的離線捨入是卡在正中間時往離零方向,不是 jnp.round 的
    逢五取偶。threshold=127 讓 scale=1.0,2.5/-2.5 是中點,逢五取偶會給
    2.0/-2.0,這裡要的是 3.0/-3.0。"""
    x = jnp.array([2.5, -2.5])
    x_hat, scale = fake_quantize_tensor(x, bits=8, threshold=127.0)
    assert float(scale) == pytest.approx(1.0)
    assert np.allclose(np.asarray(x_hat), [3.0, -3.0])


def test_quantize_to_int_returns_int32_codes_matching_fake_quantize_tensor():
    """`q` 是 int32 整數碼,`q*scale` 精確等於 `fake_quantize_tensor` 的 x_hat。"""
    key = jax.random.PRNGKey(2)
    x = jax.random.uniform(key, (200,), minval=-5.0, maxval=5.0)
    x_hat, scale_a = fake_quantize_tensor(x, bits=8)
    q, scale_b = quantize_to_int(x, bits=8)
    assert q.dtype == jnp.int32
    assert int(jnp.max(jnp.abs(q))) <= max_weight_code(8)
    assert float(scale_a) == pytest.approx(float(scale_b))
    assert np.allclose(np.asarray(q) * np.asarray(scale_b), np.asarray(x_hat), atol=TOL)


def test_max_weight_code():
    assert max_weight_code(8) == 127
    assert max_weight_code(4) == 7


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
    q = quantize_params(params, bits=8)
    assert len(q) == len(layers)
    for w, w_q in zip(params, q):
        assert w_q.shape == w.shape


def test_quantize_params_clip_percentile_100_equals_max_abs_default():
    """`clip_percentile=100` 應該跟 `fake_quantize_tensor` 沒給 `threshold`
    時的預設(max-abs)行為一致——100th percentile 精確等於最大值。"""
    layers = _layers()
    params = _params(layers)
    q_pct = quantize_params(params, bits=8, per_channel=True, clip_percentile=100.0)
    q_direct = tuple(fake_quantize_tensor(w, bits=8, axis=0)[0] for w in params)
    for a, b in zip(q_pct, q_direct):
        assert np.allclose(np.asarray(a), np.asarray(b), atol=TOL)


def test_quantize_params_lower_percentile_increases_error():
    """門檻夾得比 100th percentile 窄,uniform 初始化的權重下截斷誤差主導,
    整體 mse 不該系統性變小(推導文件步驟 2 的 trade-off 方向性檢查)。"""
    layers = _layers()
    params = _params(layers)
    q_full = quantize_params(params, bits=8, clip_percentile=100.0)
    q_clip = quantize_params(params, bits=8, clip_percentile=50.0)
    err_full = sum(quantization_error(w, w_hat)["mse"] for w, w_hat in zip(params, q_full))
    err_clip = sum(quantization_error(w, w_hat)["mse"] for w, w_hat in zip(params, q_clip))
    assert err_clip >= err_full - TOL


# ============================================================================
# D. 衰減查表:delta_t_max / build_decay_table_int / apply_decay_table_int
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


def test_build_decay_table_int_matches_hand_computed_values():
    """f_a=4, tau=4.0(base=0.75)手算前三項:0.75*16=12、0.5625*16=9 剛好是
    整數;0.75**3*16=6.75 捨入到 7。"""
    table = np.asarray(build_decay_table_int(f_a=4, tau=4.0))
    assert table.dtype == np.int32
    assert table.shape == (delta_t_max(4, 4.0),)
    assert list(table[:3]) == [12, 9, 7]


def test_build_decay_table_int_caps_at_max_code():
    """a 很接近 1 時,a*2^f_a 捨入會進到 2^f_a,要夾在 Q0.f_a 的最大碼
    2^f_a-1。tau=64, f_a=4:(63/64)*16=15.75,捨入是 16,夾成 15。"""
    table = np.asarray(build_decay_table_int(f_a=4, tau=64.0))
    assert int(table[0]) == 15


def test_build_decay_table_int_monotonic_and_in_range():
    f_a = 8
    table = np.asarray(build_decay_table_int(f_a=f_a, tau=16.0))
    assert np.all(table >= 0) and np.all(table <= 2 ** f_a - 1)
    assert np.all(np.diff(table) <= 0), "a_k 隨 Δt 增大應該單調不增"


def test_apply_decay_table_int_three_regimes():
    """浮點 Δt 輸入(佇列建構給的就是浮點):Δt=0 是 identity,1~3 查表
    (12/9/7,見上面手算),60 遠超過表深度回傳 0。"""
    table_int = build_decay_table_int(f_a=4, tau=4.0)
    delta_t = jnp.array([0.0, 1.0, 2.0, 3.0, 60.0])
    a_int, is_identity = apply_decay_table_int(delta_t, table_int)
    assert list(np.asarray(is_identity)) == [True, False, False, False, False]
    # Δt=0 那格是 identity,a_int 的值沒有意義,不檢查
    assert list(np.asarray(a_int)[1:]) == [12, 9, 7, 0]


def test_apply_decay_table_int_rejects_non_integer_delta_t():
    """時間不是整數毫秒時,轉型會默默截掉小數,要直接拒絕。"""
    table_int = build_decay_table_int(f_a=4, tau=4.0)
    with pytest.raises(ValueError):
        apply_decay_table_int(jnp.array([1.0, 2.5]), table_int)


def test_apply_decay_table_int_rejects_out_of_int32_range_delta_t():
    """沒遮好的 pad 事件時間(1e12)在 float32 裡是整數,但轉 int32 會溢位;
    負的 Δt 代表時間倒退。兩種都要拒絕。"""
    table_int = build_decay_table_int(f_a=4, tau=4.0)
    with pytest.raises(ValueError):
        apply_decay_table_int(jnp.array([1.0, 1e12]), table_int)
    with pytest.raises(ValueError):
        apply_decay_table_int(jnp.array([1.0, -1.0]), table_int)


# ============================================================================
# E. i_V 公式(推導見 docs/math/膜電位量化推導.md「通用量測與公式」節)
# ============================================================================

def test_iv_from_measurement_matches_hand_computation():
    # M=2.0, T_c=1.0, b=8: x=2*127=254, floor(log2(254))=7, +2=9
    assert iv_from_measurement(M=2.0, T_c=1.0, b=8) == 9


def test_iv_from_measurement_power_of_two_leaves_room_for_positive_value():
    """x 剛好是 2 的次方時,有號格式正向碰不到 2^(i_V-1) 本身,要多給一位。
    M=4, T_c=127, b=8:x=4,i_V=floor(2)+2=4,範圍 [-8, 8) 放得下 +4
    (i_V=3 的範圍 [-4, 4) 放不下)。"""
    i_V = iv_from_measurement(M=4.0, T_c=127.0, b=8)
    assert i_V == 4
    assert 2 ** (i_V - 1) > 4


def test_iv_from_measurement_rejects_non_positive_measurement():
    with pytest.raises(ValueError):
        iv_from_measurement(M=0.0, T_c=1.0, b=8)


def test_iv_layer_and_waste():
    per_channel = [3, 5, 4]
    assert iv_layer(per_channel) == 5
    assert waste(iv_layer(per_channel), per_channel) == [2, 0, 1]


# ============================================================================
# F. 離線整數換算:round_half_away_from_zero / v_th_to_int
# ============================================================================


def test_round_half_away_from_zero_non_tie_values():
    x = jnp.array([0.4, 0.6, -0.4, -0.6, 0.0])
    result = np.asarray(round_half_away_from_zero(x))
    assert np.allclose(result, [0.0, 1.0, 0.0, -1.0, 0.0])


def test_v_th_to_int_matches_hand_computation():
    # 1.95/0.3=6.5,*2^2=26.0(不是中點,浮點雜訊不影響取整結果)
    assert int(v_th_to_int(v_th=1.95, s_c=0.3, f_V=2, i_V=8)) == 26


def test_v_th_to_int_ties_away_from_zero():
    # v_th_tilde = 2.5/1.0 = 2.5,f_V=0 時 *2^0=2.5,剛好是中點,往離零方向是 3
    assert int(v_th_to_int(v_th=2.5, s_c=1.0, f_V=0, i_V=8)) == 3


def test_v_th_to_int_rejects_threshold_outside_register_range():
    """輸出層 v_th=1e9 換算出來遠超過暫存器範圍,硬體比較器放不下,要直接
    拒絕,不能靠轉型飽和剛好擋住。邊界:i_V=4, f_V=0 範圍 [-8,7],7 可以、8 不行。"""
    with pytest.raises(ValueError):
        v_th_to_int(v_th=1e9, s_c=0.03, f_V=10, i_V=13)
    assert int(v_th_to_int(v_th=7.0, s_c=1.0, f_V=0, i_V=4)) == 7
    with pytest.raises(ValueError):
        v_th_to_int(v_th=8.0, s_c=1.0, f_V=0, i_V=4)
