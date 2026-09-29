"""定點數運算電路的整數模擬:乘完之後的移位捨入、拆高低兩半的寬乘法、寫回暫存器時的溢位處理。

對應 FPGA 上膜電位暫存器單步更新用到的電路,硬體運算式見 docs/math/膜電位量化推導.md。
全程 int32,理由見 docs/問題紀錄.md「決策:整數模擬全程 int32,不開 jax_enable_x64」。

捨入規則直接對兩補數有號數運算,不拆正負號:
- ROUND:先加半格再算術右移,卡在正中間時往正無窮(2.5 -> 3、-2.5 -> -2)。
- TRUNCATE:直接算術右移,等於往負無窮(-1.5 -> -2)。
"""
import enum

import jax.numpy as jnp

# 低位乘積 a_int*lo 最多 2*shift_bits 位元,加上半格要放得進 int32 的 31 位元量值。
MAX_SHIFT_BITS = 15
# 暫存器寬度上限:寫回之前的真實值(暫存器值加上一筆權重貢獻)要放得進 int32。
MAX_REGISTER_BITS = 30


class RoundMode(str, enum.Enum):
    """乘法後捨掉多餘小數位元的規則,見模組說明。也可以直接傳字串 "round"、"truncate"。"""
    ROUND = "round"
    TRUNCATE = "truncate"


class OverflowMode(str, enum.Enum):
    """值超出暫存器範圍時的處理,見 fit_to_bits。也可以直接傳字串 "wrap"、"saturate"。"""
    WRAP = "wrap"          # 兩補數繞回(mod 2^total_bits),加法器天生的行為
    SATURATE = "saturate"  # 夾在範圍的最大/最小值,硬體要多一組溢位偵測加選擇器


def round_shift(x_int: jnp.ndarray, shift_bits: int,
                round_mode: RoundMode | str) -> jnp.ndarray:
    """有號整數除以 2 ** shift_bits 再捨入:定點乘法器算完之後捨掉多出來的小數位元。

    round_mode 的規則見模組說明,不合法時 raise ValueError。shift_bits=0 原樣回傳。
    """
    round_mode = RoundMode(round_mode)
    x_int = jnp.asarray(x_int)
    if shift_bits == 0:
        return x_int
    if round_mode is RoundMode.ROUND:
        x_int = x_int + (1 << (shift_bits - 1))
    return x_int >> shift_bits


def wide_mul_shift(a_int: jnp.ndarray, v_int: jnp.ndarray, shift_bits: int,
                   round_mode: RoundMode | str) -> jnp.ndarray:
    """等於 round_shift(a_int * v_int, shift_bits, round_mode),但不在 int32 裡乘出寬乘積。

    v_int 拆成 hi * 2 ** shift_bits + lo,只有低位乘積要捨入;為什麼跟一次捨入逐位元相等見
    docs/math/膜電位量化推導.md「乘法後的捨入」。
    a_int: 0 <= a_int < 2 ** shift_bits,衰減係數的整數碼。
    v_int: 有號暫存器值,寬度不超過 MAX_REGISTER_BITS。
    shift_bits 超過 MAX_SHIFT_BITS 時 raise ValueError。
    """
    if shift_bits > MAX_SHIFT_BITS:
        raise ValueError(
            f"shift_bits={shift_bits} 超過 {MAX_SHIFT_BITS},低位乘積"
            f"(2*shift_bits 位元)會溢位 int32(見模組說明)。")
    a_int = jnp.asarray(a_int)
    v_int = jnp.asarray(v_int)
    hi = v_int >> shift_bits
    lo = v_int & ((1 << shift_bits) - 1)
    return a_int * hi + round_shift(a_int * lo, shift_bits, round_mode)


def fit_to_bits(x_int: jnp.ndarray, total_bits: int, overflow_mode: OverflowMode | str
                ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """把整數寫回 total_bits 位元的有號暫存器,範圍 [-2 ** (total_bits-1), 2 ** (total_bits-1) - 1]。

    超出範圍時:WRAP 是兩補數繞回(值會翻號),SATURATE 夾在最大或最小值。
    x_int: 寫回之前的真實值,呼叫端要保證它本身沒有溢位 int32。
    回傳 (fitted, overflowed):fitted 是寫回之後的值;overflowed 是逐元素布林,真實值超出範圍時
        是 True,兩種模式意義一樣。
    overflow_mode 不合法、或 total_bits 超過 MAX_REGISTER_BITS 時 raise ValueError。
    """
    overflow_mode = OverflowMode(overflow_mode)
    if total_bits > MAX_REGISTER_BITS:
        raise ValueError(
            f"total_bits={total_bits} 超過 {MAX_REGISTER_BITS},暫存器值加上"
            "權重貢獻之後可能溢位 int32(見模組說明)。")
    x_int = jnp.asarray(x_int)
    half = 1 << (total_bits - 1)
    overflowed = (x_int < -half) | (x_int >= half)
    if overflow_mode is OverflowMode.SATURATE:
        return jnp.clip(x_int, -half, half - 1), overflowed
    masked = x_int & ((1 << total_bits) - 1)
    wrapped = jnp.where(masked >= half, masked - (1 << total_bits), masked)
    return wrapped, overflowed
