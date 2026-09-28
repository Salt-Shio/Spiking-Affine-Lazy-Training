"""salt_core/quant/fixed_point.py(定點數運算電路的整數模擬)的測試。

捨入慣例是直接對兩補數有號數運算(docs/問題紀錄.md 第十八節):
round 是先加半格再算術右移,truncate 是直接算術右移(floor)。
溢位模式有繞回跟飽和兩種(docs/math/膜電位量化推導.md「溢位政策」節)。
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from salt_core.quant.fixed_point import (MAX_REGISTER_BITS, MAX_SHIFT_BITS, OverflowMode, RoundMode,
                                   fit_to_bits, round_shift, wide_mul_shift)


# ============================================================================
# round_shift
# ============================================================================

def test_round_shift_round_mode_adds_half_then_arithmetic_shift():
    """shift_bits=1,[5,-5,3,-3] 除以 2 是 [2.5,-2.5,1.5,-1.5]:加半格再算術
    右移,卡在正中間時往正無窮 => [3,-2,2,-1]。"""
    x = jnp.array([5, -5, 3, -3])
    assert list(np.asarray(round_shift(x, shift_bits=1, round_mode="round"))) == [3, -2, 2, -1]


def test_round_shift_round_mode_non_tie_values():
    """shift_bits=2(除以 4):5/4=1.25 → 1、7/4=1.75 → 2、-5/4=-1.25 → -1、
    -7/4=-1.75 → -2。"""
    x = jnp.array([5, 7, -5, -7])
    assert list(np.asarray(round_shift(x, shift_bits=2, round_mode="round"))) == [1, 2, -1, -2]


def test_round_shift_truncate_mode_is_floor():
    """直接算術右移等於 floor(往負無窮),負數跟往零捨去差 1:
    [5,-5,3,-3]>>1 = [2,-3,1,-2]。"""
    x = jnp.array([5, -5, 3, -3])
    assert list(np.asarray(round_shift(x, shift_bits=1, round_mode="truncate"))) == [2, -3, 1, -2]


def test_round_shift_zero_shift_bits_is_identity():
    x = jnp.array([3, -3, 0])
    for mode in RoundMode:
        assert list(np.asarray(round_shift(x, shift_bits=0, round_mode=mode))) == [3, -3, 0]


def test_round_shift_rejects_invalid_round_mode():
    with pytest.raises(ValueError):
        round_shift(jnp.array([1]), shift_bits=2, round_mode="ceil")


def test_round_shift_decay_step_never_grows_magnitude():
    """只要 a_int < 2^shift_bits(衰減嚴格小於 1),捨入後的量值不會超過
    v 本身的量值——兩種捨入模式都一樣,亂數測試這個性質。"""
    v_key, a_key = jax.random.split(jax.random.PRNGKey(0))
    shift_bits = 6
    v = jax.random.randint(v_key, (2000,), minval=-100000, maxval=100000)
    a_int = jax.random.randint(a_key, (2000,), minval=0, maxval=2 ** shift_bits)
    for mode in RoundMode:
        result = round_shift(a_int * v, shift_bits=shift_bits, round_mode=mode)
        assert np.all(np.abs(np.asarray(result)) <= np.abs(np.asarray(v))), mode


# ============================================================================
# wide_mul_shift
# ============================================================================

def test_wide_mul_shift_matches_round_shift_when_product_fits_int32():
    """乘積本身放得進 int32 時,結果要跟直接算 round_shift(a*v) 一樣——
    a_int=12,v_int=40,shift=4:480/16=30,整除沒有捨入。"""
    naive = round_shift(jnp.array(12 * 40), shift_bits=4, round_mode="round")
    wide = wide_mul_shift(jnp.array(12), jnp.array(40), shift_bits=4, round_mode="round")
    assert int(wide) == int(naive) == 30


def test_wide_mul_shift_negative_tie_uses_twos_complement_convention():
    """有號 v_int 直接算術右移拆 hi/lo,結果跟整個乘積一次捨入相同。
    a_int=1, v_int=-8, shift_bits=4:-8/16=-0.5,round 往正無窮是 0
    ((-8+8)>>4=0),truncate 是 floor(-0.5)=-1。"""
    a, v = jnp.array(1), jnp.array(-8)
    assert int(wide_mul_shift(a, v, shift_bits=4, round_mode="round")) == 0
    assert int(wide_mul_shift(a, v, shift_bits=4, round_mode="truncate")) == -1


def test_wide_mul_shift_zero_shift_bits_is_plain_product():
    assert int(wide_mul_shift(jnp.array(3), jnp.array(-7), shift_bits=0, round_mode="round")) == -21


def test_wide_mul_shift_zero_a_int_gives_zero():
    assert int(wide_mul_shift(jnp.array(0), jnp.array(12345), shift_bits=8,
                              round_mode="round")) == 0


def test_wide_mul_shift_rejects_shift_bits_above_limit():
    with pytest.raises(ValueError):
        wide_mul_shift(jnp.array(1), jnp.array(1), shift_bits=MAX_SHIFT_BITS + 1,
                       round_mode="round")


def test_wide_mul_shift_matches_exact_python_int_arithmetic_random_wide_values():
    """亂數交叉驗證:用 Python 原生大整數(不受 int32 限制)算 oracle,對照
    乘積本身會溢位 int32 的寬參數範圍(shift_bits=15,v_int 接近 30 位元)。
    Python 的 >> 對負數也是算術右移,oracle 直接照兩補數慣例寫。"""
    a_key, v_key = jax.random.split(jax.random.PRNGKey(3))
    shift_bits = MAX_SHIFT_BITS
    a_vals = jax.random.randint(a_key, (500,), minval=0, maxval=2 ** shift_bits)
    v_vals = jax.random.randint(v_key, (500,), minval=-(2 ** 29), maxval=2 ** 29)

    for mode in RoundMode:
        result = np.asarray(wide_mul_shift(a_vals, v_vals, shift_bits=shift_bits, round_mode=mode))
        for a, v, got in zip(np.asarray(a_vals), np.asarray(v_vals), result):
            exact = int(a) * int(v)
            if mode is RoundMode.ROUND:
                expected = (exact + (1 << (shift_bits - 1))) >> shift_bits
            else:
                expected = exact >> shift_bits
            assert int(got) == expected, f"a={a},v={v},mode={mode}: got {got}, expected {expected}"


# ============================================================================
# fit_to_bits
# ============================================================================

def test_fit_to_bits_in_range_values_unchanged_no_overflow():
    x = jnp.array([7, -8, 0, 3])  # total_bits=4,有號範圍 [-8, 7]
    for mode in OverflowMode:
        fitted, overflowed = fit_to_bits(x, total_bits=4, overflow_mode=mode)
        assert np.array_equal(np.asarray(fitted), np.asarray(x)), mode
        assert not np.any(np.asarray(overflowed)), mode


def test_fit_to_bits_wrap_positive_overflow_wraps_to_negative():
    """total_bits=4,範圍 [-8,7]。9 超出正向邊界一格,兩補數繞回去是
    9-16=-7(9=0b1001,當成 4 位元有號數,最高位是符號位)。"""
    fitted, overflowed = fit_to_bits(jnp.array([9]), total_bits=4, overflow_mode="wrap")
    assert int(fitted[0]) == -7
    assert bool(overflowed[0])


def test_fit_to_bits_wrap_negative_overflow_wraps_to_positive():
    """-9 超出負向邊界一格,mod 16 繞回去是 -9+16=7。"""
    fitted, overflowed = fit_to_bits(jnp.array([-9]), total_bits=4, overflow_mode="wrap")
    assert int(fitted[0]) == 7
    assert bool(overflowed[0])


def test_fit_to_bits_saturate_clamps_to_range_edges():
    """飽和:9 夾到 7、-9 夾到 -8,正負號不會翻;overflowed 的意義跟繞回一樣。"""
    fitted, overflowed = fit_to_bits(jnp.array([9, -9, 100]), total_bits=4,
                                     overflow_mode="saturate")
    assert list(np.asarray(fitted)) == [7, -8, 7]
    assert np.all(np.asarray(overflowed))


def test_fit_to_bits_rejects_invalid_overflow_mode():
    with pytest.raises(ValueError):
        fit_to_bits(jnp.array([0]), total_bits=4, overflow_mode="clip")


def test_fit_to_bits_rejects_total_bits_above_limit():
    with pytest.raises(ValueError):
        fit_to_bits(jnp.array([0]), total_bits=MAX_REGISTER_BITS + 1, overflow_mode="wrap")
