"""salt_core/quant/codes.py 的測試:離線整數碼。推導見 docs/math/權重量化推導.md、
docs/math/膜電位量化推導.md。

- A 權重整數碼:quantize_to_int、max_weight_code。
- B 衰減查表:delta_t_max/build_decay_table_int/apply_decay_table_int。
- C i_V 公式:iv_from_measurement/iv_layer/waste。
- D 離線整數換算:round_half_away_from_zero/v_th_to_int。

逐事件遞迴用的定點數運算(round_shift 等)在 test_fixed_point.py。
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from salt_core.quant.codes import (apply_decay_table_int, build_decay_table_int, delta_t_max,
                                   iv_from_measurement, iv_layer, max_weight_code,
                                   quantize_to_int, round_half_away_from_zero, v_th_to_int,
                                   waste)
from salt_core.quant.ptq import fake_quantize_tensor

TOL = 1e-5

# ============================================================================
# A. 權重整數碼:quantize_to_int / max_weight_code
# ============================================================================

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
# B. 衰減查表:delta_t_max / build_decay_table_int / apply_decay_table_int
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
# C. i_V 公式(推導見 docs/math/膜電位量化推導.md「通用量測與公式」節)
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
# D. 離線整數換算:round_half_away_from_zero / v_th_to_int
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
