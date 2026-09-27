"""定點數運算電路的整數模擬:乘完之後的移位捨入、拆高低兩半的寬乘法、
寫回暫存器時的溢位處理(繞回或飽和)。對應 FPGA 上膜電位暫存器 $\\tilde V$ 單步更新用到的
電路(硬體運算式見 docs/math/膜電位量化推導.md「乘法後的捨入」節),
唯一的正式呼叫端是 `quant.scan.process_event`。

**全程留在 int32,不開 `jax_enable_x64`**:這是全域設定,開了會讓
`jax.lax.scan`/`jnp.argmax` 等地方的預設整數 dtype 變成 int64,訓練熱路徑
的既有測試會大量失敗。所以寬乘積不能直接乘出來(見 `wide_mul_shift`),
每個 primitive 自己檢查輸入位元寬度在 int32 容得下的範圍內。專案其他地方
提到這個限制時都指回這裡。

捨入慣例直接對兩補數有號數運算,不拆正負號,跟硬體電路的做法一致
(docs/問題紀錄.md 第十八節):

- `RoundMode.ROUND`:先加半格,再算術右移。卡在正中間時往正無窮
  (2.5→3、-2.5→-2)。
- `RoundMode.TRUNCATE`:直接算術右移,等於 floor(往負無窮,-1.5→-2)。

溢位模式見 `OverflowMode`,預設繞回;定案理由見 docs/math/膜電位量化推導.md
「溢位政策」節。
"""
import enum

import jax.numpy as jnp

# 低位乘積 a_int*lo 最多 2*shift_bits 位元,加上半格要放得進 int32 的 31 位元量值。
MAX_SHIFT_BITS = 15
# 暫存器寬度上限:寫回之前的真實值(暫存器值加上一筆權重貢獻)要放得進 int32。
MAX_REGISTER_BITS = 30


class RoundMode(str, enum.Enum):
    """乘法後捨掉多餘小數位元的規則,見模組說明。繼承 `str`,呼叫端直接傳
    `"round"`/`"truncate"` 字串也可以。"""
    ROUND = "round"
    TRUNCATE = "truncate"


class OverflowMode(str, enum.Enum):
    """值超出暫存器範圍時的處理,見 `fit_to_bits`。繼承 `str`,呼叫端直接傳
    `"wrap"`/`"saturate"` 字串也可以。"""
    WRAP = "wrap"          # 兩補數繞回(mod 2^total_bits),加法器天生的行為
    SATURATE = "saturate"  # 夾在範圍的最大/最小值,硬體要多一組溢位偵測加選擇器


def round_shift(x_int: jnp.ndarray, shift_bits: int,
                round_mode: RoundMode | str) -> jnp.ndarray:
    """有號整數除以 $2^{\\text{shift\\_bits}}$ 再捨入,對應定點乘法器算完之後
    把多出來的小數位元捨掉那一步。`round_mode` 的兩種規則見模組說明;
    不合法的值直接 raise `ValueError`。`shift_bits=0` 是恆等。
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
    """算 `round_shift(a_int * v_int, shift_bits, round_mode)`,但不在 int32
    裡真的乘出 `a_int * v_int` 這個寬乘積。`v_int` 拆成
    `hi * 2^shift_bits + lo`(`hi` 是算術右移、`lo` 是低位遮罩,恆非負),
    高位乘積不用捨入,只有低位乘積要捨。這樣拆為什麼跟整個乘積一次捨入
    逐位元相等,見 docs/math/膜電位量化推導.md「乘法後的捨入」節。

    輸入條件:`0 <= a_int < 2^shift_bits`(衰減係數的 Q0.shift_bits 整數碼);
    `v_int` 是有號暫存器值,寬度不超過 `MAX_REGISTER_BITS`。
    限制:`shift_bits <= MAX_SHIFT_BITS`,超過就 raise `ValueError`。
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


def fit_to_bits(x_int: jnp.ndarray, total_bits: int, overflow_mode: OverflowMode | str):
    """把整數寫回 `total_bits` 位元的有號暫存器,範圍
    `[-2^(total_bits-1), 2^(total_bits-1)-1]`,超出時照 `overflow_mode` 處理:

    - `OverflowMode.WRAP`:兩補數繞回(mod $2^{\\text{total\\_bits}}$),值會翻號。
    - `OverflowMode.SATURATE`:夾在範圍的最大/最小值。

    `x_int` 是還沒寫回之前的真實值,呼叫端要保證它本身沒有先溢位 int32。
    不合法的 `overflow_mode` 直接 raise `ValueError`。
    限制:`total_bits <= MAX_REGISTER_BITS`,超過就 raise `ValueError`。

    回傳 `(fitted, overflowed)`:`fitted` 是寫回之後的值(沒溢位時等於
    `x_int`);`overflowed` 是逐元素布林陣列,`True` 代表真實值超出範圍,
    兩種模式意義一樣。
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
