"""離線算好、存進 FPGA 的整數常數:權重碼、衰減查表、整數門檻、i_V 位元數公式。

推導見 docs/math/權重量化推導.md、docs/math/膜電位量化推導.md。
浮點數換成整數碼一律用 round_half_away_from_zero;逐事件的整數遞迴在 quant/scan.py。
"""
import math

import jax
import jax.numpy as jnp


def max_weight_code(bits: int) -> int:
    """bits 位元對稱量化的最大整數碼 2 ** (bits-1) - 1,正負對稱、不用 -2 ** (bits-1)。"""
    return 2 ** (bits - 1) - 1


def round_half_away_from_zero(x: jnp.ndarray) -> jnp.ndarray:
    """四捨五入,卡在正中間時往離零的方向(2.5 -> 3、-2.5 -> -3),不是 jnp.round 的逢五取偶。

    只用在離線換整數碼;逐事件遞迴裡的捨入是 fixed_point.round_shift。
    """
    x = jnp.asarray(x)
    return jnp.sign(x) * jnp.floor(jnp.abs(x) + 0.5)


# ============================================================================
# 權重碼
# ============================================================================

def quantize_to_int(x: jnp.ndarray, bits: int, *, axis: int | None = None,
                    threshold: jnp.ndarray | float | None = None
                    ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """對稱線性量化。回傳 (q, scale):q 是 int32 整數碼,scale 是量化步長,q * scale 是量化再反量化的值。

    bits: 整數碼範圍 -max_weight_code(bits) ~ max_weight_code(bits)。小於 2 時 raise ValueError。
    axis: None 時整個張量共用一個門檻;給軸時該軸每個位置各一個門檻。
    threshold: clip 門檻,None 時是 max(|x|)(照 axis 取)。門檻 <= 0(全零 channel)時當成 1,
        那個 channel 的 q 全是 0。
    """
    if bits < 2:
        raise ValueError(f"bits 必須 >= 2,給的是 {bits!r}")
    x = jnp.asarray(x)

    if threshold is None:
        if axis is None:
            threshold = jnp.max(jnp.abs(x))
        else:
            reduce_axes = tuple(i for i in range(x.ndim) if i != axis)
            threshold = jnp.max(jnp.abs(x), axis=reduce_axes, keepdims=True)
    threshold = jnp.asarray(threshold, dtype=x.dtype)
    threshold = jnp.where(threshold > 0, threshold, jnp.ones_like(threshold))

    scale = threshold / max_weight_code(bits)
    x_clipped = jnp.clip(x, -threshold, threshold)
    q = round_half_away_from_zero(x_clipped / scale).astype(jnp.int32)
    return q, scale


# ============================================================================
# 膜電位量化的離線常數:a_k 衰減查表、整數門檻
# ============================================================================

def delta_t_max(f_a: int, tau: float) -> int:
    """衰減查表 a = (1 - 1/tau) ** dt 要開多深:a 捨入到 f_a 個小數位元後不是 0 的最大 dt。

    也就是 a >= 2 ** -(f_a + 1)。dt 更大時 a 是 0,不用存。表從 dt=1 開始,dt=0(a=1)不在表裡。
    推導見 docs/math/膜電位量化推導.md「a_k:查表,符號 f_a」。
    """
    eps = 2.0 ** -(f_a + 1)
    log_base = math.log(1.0 - 1.0 / tau)  # < 0(tau > 1 時)
    return max(int(math.floor(math.log(eps) / log_base)), 0)


def build_decay_table_int(f_a: int, tau: float) -> jnp.ndarray:
    """衰減的整數查表,形狀 (delta_t_max(f_a, tau),),int32。

    table[i] 是 dt = i + 1 的碼,衰減值 = table[i] / 2 ** f_a。碼值是 a * 2 ** f_a 捨入後夾在
    2 ** f_a - 1 以下(只有小數位元,存不下 1.0);dt=0 由 apply_decay_table_int 的 is_identity 處理。
    """
    n = delta_t_max(f_a, tau)
    delta_t = jnp.arange(1, n + 1, dtype=jnp.float32)
    a_exact = (1.0 - 1.0 / tau) ** delta_t
    code = round_half_away_from_zero(a_exact * 2 ** f_a)
    return jnp.minimum(code, 2 ** f_a - 1).astype(jnp.int32)


def apply_decay_table_int(delta_t: jnp.ndarray, table_int: jnp.ndarray
                          ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """用 dt 當 index 查整數衰減表。

    delta_t: 浮點陣列,值是整數毫秒。不是整數、或不在 [0, 2 ** 31) 時 raise ValueError;
        jit 裡(traced 值)沒辦法 raise,只做轉型,呼叫端要在 jit 外檢查過(InputEvents.checked)。
    回傳 (a_int, is_identity),形狀同 delta_t:dt=0 時 is_identity=True、a_int 沒有意義;
        dt 超過表深度時 a_int=0。
    """
    delta_t = jnp.asarray(delta_t)
    if not isinstance(delta_t, jax.core.Tracer):
        _check_delta_t_is_int32_index(delta_t)
    delta_t = delta_t.astype(jnp.int32)
    n = table_int.shape[0]
    idx = jnp.clip(delta_t - 1, 0, n - 1)
    gathered = table_int[idx]
    a_int = jnp.where(delta_t > n, jnp.zeros_like(gathered), gathered)
    is_identity = delta_t == 0
    return a_int, is_identity


def _check_delta_t_is_int32_index(delta_t: jnp.ndarray) -> None:
    """apply_decay_table_int 的入口檢查。"""
    if not bool(jnp.all(delta_t == jnp.floor(delta_t))):
        bad = delta_t[delta_t != jnp.floor(delta_t)]
        raise ValueError(f"Δt 必須是整數毫秒,有非整數值,例如 {bad[:5].tolist()}")
    out_of_range = (delta_t < 0) | (delta_t >= 2.0 ** 31)
    if bool(jnp.any(out_of_range)):
        raise ValueError(
            f"Δt 必須在 [0, 2^31) 裡,有超出範圍的值,例如 {delta_t[out_of_range][:5].tolist()}")


def v_th_to_int(v_th: jnp.ndarray | float, s_c: jnp.ndarray | float, f_V: int,
                i_V: int) -> jnp.ndarray:
    """物理尺度的門檻換成整數門檻 round_half_away_from_zero(v_th / s_c * 2 ** f_V)。

    除以 s_c 只在這裡做一次,整數遞迴不會看到 s_c;理由見 docs/問題紀錄.md
    「決策:膜電位量化的 a、b、v_th 在整數尺度算,不在物理尺度」。
    整數門檻超出 (i_V, f_V) 暫存器的範圍時 raise ValueError;不 fire 的層不要呼叫,改傳 v_th_int=None。
    """
    v_th_tilde = jnp.asarray(v_th) / jnp.asarray(s_c)
    code = round_half_away_from_zero(v_th_tilde * (2 ** f_V))
    half = 2 ** (i_V + f_V - 1)
    out_of_range = (code < -half) | (code > half - 1)
    if bool(jnp.any(out_of_range)):
        raise ValueError(
            f"整數門檻超出 i_V={i_V}、f_V={f_V} 暫存器的範圍 [{-half}, {half - 1}],"
            f"例如 {jnp.atleast_1d(code)[jnp.atleast_1d(out_of_range)][:5].tolist()}。"
            "這層不 fire 的話用 v_th_int=None。")
    return code.astype(jnp.int32)


# ============================================================================
# i_V 位元數公式
# ============================================================================

def iv_from_measurement(M: float, T_c: float, b: int) -> int:
    """一個 channel 的 i_V = floor(log2(x)) + 2,x = M * (2 ** (b-1) - 1) / T_c。

    M: 這個 channel 的膜電位單邊最大量值 max(V_max, |V_min|),用未量化的浮點權重量的。
    T_c: 權重量化的 clip 門檻。
    保證 2 ** (i_V - 1) > x。推導見 docs/math/膜電位量化推導.md「通用量測與公式(所有層)」。
    M <= 0 時 raise ValueError:沒有活動的 channel 要在呼叫前篩掉。
    """
    if M <= 0:
        raise ValueError(f"M 必須 > 0,給的是 {M!r}(死 channel 應該先篩掉,不要呼叫這個函式)")
    x = M * max_weight_code(b) / T_c
    return math.floor(math.log2(x)) + 2


def iv_layer(iv_per_channel: list[int]) -> int:
    """整層共用的 i_V:取所有 channel 裡最大的。"""
    return max(iv_per_channel)


def waste(iv_layer_value: int, iv_per_channel: list[int]) -> list[int]:
    """整層共用 i_V 時每個 channel 多付的位元數 = iv_layer_value - 那個 channel 的 i_V。

    怎麼用這個數決定整層共用還是逐 channel 見 docs/math/膜電位量化推導.md「通用量測與公式(所有層)」。
    """
    return [iv_layer_value - v for v in iv_per_channel]
