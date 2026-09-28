"""量化用的整數碼:權重碼、衰減整數查表、整數門檻、i_V 位元數公式。都是離線算好、
直接存進 FPGA 的常數,硬體不會執行這一步。推導見 docs/math/權重量化推導.md、
docs/math/膜電位量化推導.md。

浮點數換成整數碼一律用 round_half_away_from_zero。逐事件的整數遞迴在
quant.scan,用到的定點數運算電路在 quant.fixed_point。
"""
import math

import jax
import jax.numpy as jnp


def max_weight_code(bits: int) -> int:
    """`bits` 位元對稱量化的最大整數碼 $2^{b-1}-1$(犧牲一個編碼點換嚴格
    對稱,見權重量化推導文件步驟 1)。"""
    return 2 ** (bits - 1) - 1


def round_half_away_from_zero(x: jnp.ndarray) -> jnp.ndarray:
    """四捨五入,卡在正中間時往離零的方向(2.5→3、-2.5→-3),不是
    `jnp.round` 預設的逢五取偶(2.5→2)。只用在離線、一次性把浮點數換成
    整數碼的場合;逐事件遞迴裡的移位捨入見 `quant.fixed_point.round_shift`。"""
    x = jnp.asarray(x)
    return jnp.sign(x) * jnp.floor(jnp.abs(x) + 0.5)


# ============================================================================
# 權重碼
# ============================================================================

def quantize_to_int(x: jnp.ndarray, bits: int, *, axis: int | None = None,
                    threshold: jnp.ndarray | float | None = None):
    """對稱線性量化,回傳 `(q, scale)`:`q` 是 int32 整數碼,`scale` 是量化
    步長 `Δ`(浮點)。`q * scale` 就是量化再反量化的值(見
    `quant.ptq.fake_quantize_tensor`)。

    `bits`:位元寬度,整數碼範圍 `{-max_weight_code(bits), ..., max_weight_code(bits)}`。
    `bits < 2` 無意義(至少要有正負兩格)。

    `axis`:`None` 是 per-tensor(整個 `x` 共用一個 threshold);給一個軸
    索引就是沿該軸 per-channel(該軸每個位置各自的 threshold,對其餘軸取
    `max(|x|)` 或用呼叫端傳入的 `threshold`)。

    `threshold`:clip 門檻 `T`。`None` 時預設 `max(|x|)`(沿 `axis` 之外的軸
    reduce,`axis=None` 時整個 tensor reduce);呼叫端也可以自己算好
    (例如某個 percentile)傳進來,對應推導文件步驟 2 的網格搜尋。

    全零的 channel(`threshold<=0`)會被夾成 `threshold=1.0` 避免除以零,
    這種 channel 的 `q` 恆為 0(輸入本來就全零)。捨入用
    `round_half_away_from_zero`。
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
    """$a_k=(1-1/\\tau)^{\\Delta t}$ 查表要開多深(推導文件「$a_k$:查表」節)。

    只算 $\\Delta t \\ge 1$ 的部分($\\Delta t=0$ 是 $a=1.0$,Q0.$f_a$ 存不下,
    不在表裡,見 `build_decay_table_int`)。回傳最大的 $\\Delta t$,使得
    $(1-1/\\tau)^{\\Delta t}$ 捨入到 $f_a$ 個小數位元後仍不是 0(即
    $\\ge 2^{-(f_a+1)}$,半個最小刻度);超過這個 $\\Delta t$ 的表項全部是 0,
    不用存(`apply_decay_table_int` 直接回傳 0)。
    """
    eps = 2.0 ** -(f_a + 1)
    log_base = math.log(1.0 - 1.0 / tau)  # < 0(tau > 1 時)
    return max(int(math.floor(math.log(eps) / log_base)), 0)


def build_decay_table_int(f_a: int, tau: float) -> jnp.ndarray:
    """$a_k$ 的整數查表,shape `(delta_t_max(f_a, tau),)`,int32。`table[i]`
    是 $\\Delta t=i+1$ 的 Q0.$f_a$ 碼,實際衰減值是 `table[i] / 2^f_a`。

    碼值是 $a\\cdot2^{f_a}$ 捨入後夾在 $2^{f_a}-1$ 以下:Q0.$f_a$ 無符號、沒有
    整數位元,最大只能表示 $1-2^{-f_a}$。$\\Delta t=0$($a=1.0$)存不下,不在表
    裡,由 `apply_decay_table_int` 的 `is_identity` 旗標處理。
    """
    n = delta_t_max(f_a, tau)
    delta_t = jnp.arange(1, n + 1, dtype=jnp.float32)
    a_exact = (1.0 - 1.0 / tau) ** delta_t
    code = round_half_away_from_zero(a_exact * 2 ** f_a)
    return jnp.minimum(code, 2 ** f_a - 1).astype(jnp.int32)


def apply_decay_table_int(delta_t: jnp.ndarray, table_int: jnp.ndarray):
    """用 Δt 當 index 查整數衰減表,shape 跟 `delta_t` 一樣。

    `delta_t` 是佇列建構算出來的浮點 Δt(整數毫秒)。轉成 int32 之前先檢查
    兩件事,不符合就 raise `ValueError`:

    - 每個值都是整數:時間不是整數毫秒時,轉型會默默截掉小數。
    - 每個值都在 `[0, 2^31)` 裡:負的 Δt 代表時間倒退;太大的值轉 int32 會
      溢位(例如沒遮好的 pad 事件時間 `stream._PAD_TIME`)。

    **這兩個檢查只在 `delta_t` 是具體陣列時才會跑。** 在 `jax.jit` 裡面
    `delta_t` 是 traced 值,沒辦法 raise,檢查會直接跳過、只做轉型。之後如果
    把量化 forward 包進 jit 加速,這層保護就消失了,要另外檢查。

    回傳 `(a_int, is_identity)`:
    - `Δt=0`:`is_identity=True`,`a_int` 沒有意義,呼叫端要整個跳過衰減
      (Q0.$f_a$ 存不下 1.0,見 `build_decay_table_int`)。
    - `1<=Δt<=表深度`:查表。
    - `Δt>表深度`:`a_int=0`(衰減到格式存不下)。
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
    """`apply_decay_table_int` 入口檢查,見該函式說明。"""
    if not bool(jnp.all(delta_t == jnp.floor(delta_t))):
        bad = delta_t[delta_t != jnp.floor(delta_t)]
        raise ValueError(f"Δt 必須是整數毫秒,有非整數值,例如 {bad[:5].tolist()}")
    out_of_range = (delta_t < 0) | (delta_t >= 2.0 ** 31)
    if bool(jnp.any(out_of_range)):
        raise ValueError(
            f"Δt 必須在 [0, 2^31) 裡,有超出範圍的值,例如 {delta_t[out_of_range][:5].tolist()}")


def v_th_to_int(v_th: jnp.ndarray | float, s_c: jnp.ndarray | float, f_V: int,
                i_V: int) -> jnp.ndarray:
    """把物理尺度的門檻 $v_{th}$ 換算成整數遞迴要比較的整數門檻
    $\\mathrm{round}(\\tilde v_{th}\\cdot2^{f_V})$,$\\tilde v_{th}=v_{th}/s_c$,
    捨入卡在正中間時往離零方向(`round_half_away_from_zero`)。

    除以 $s_c$ 只在這裡做一次,整數遞迴全程只跟這個整數門檻比大小,
    不會看到 $s_c$(docs/問題紀錄.md 第十七節)。

    整數門檻超出 $(i_V,f_V)$ 暫存器的範圍時 raise `ValueError`:硬體比較器
    放不下這個門檻。不 fire 的層(例如膜電位回歸的輸出層,$v_{th}$ 設超大)
    不要呼叫這個函式,整數版 forward 直接傳 `v_th_int=None`。
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
    """逐 channel $i_V(b)$(推導文件「通用量測與公式」節):

    $$i_V^{(l,c)}(b) = \\lfloor \\log_2 x \\rfloor + 2,\\qquad
    x = M^{(l,c)} \\cdot \\frac{2^{b-1}-1}{T_c}$$

    $x$ 是換算成整數權重單位之後的最大量值。$(i_V,f_V)$ 有號格式正向碰不到
    $2^{i_V-1}$ 本身,這個式子保證 $2^{i_V-1}>x$ 嚴格成立,$x$ 剛好是 2 的
    次方時也一樣。

    `M`:這個 channel 實測的單邊最大量值 $\\max(V_{\\max}, |V_{\\min}|)$,用
    **原始、未量化的浮點權重**跑出來的(不依賴 `b`,只跑一次,見推導文件
    「整體規劃」)。`T_c`:這個候選 `b` 底下,權重量化算出來的 clip 門檻。
    `M` 必須 > 0(完全沒有活動的 channel 應該在呼叫前就篩掉,不是餵 0 進來
    讓這裡算出沒意義的負無限大)。
    """
    if M <= 0:
        raise ValueError(f"M 必須 > 0,給的是 {M!r}(死 channel 應該先篩掉,不要呼叫這個函式)")
    x = M * max_weight_code(b) / T_c
    return math.floor(math.log2(x)) + 2


def iv_layer(iv_per_channel: list[int]) -> int:
    """逐 layer 共用的 $i_V$:取這層所有 channel 裡最吃緊的那個(推導文件
    「兩種粒度怎麼比較」節)。"""
    return max(iv_per_channel)


def waste(iv_layer_value: int, iv_per_channel: list[int]) -> list[int]:
    """每個 channel 被迫多付出的位元數(推導文件同節):$\\text{waste}^{(c)} =
    i_V^{(l)} - i_V^{(l,c)} \\ge 0$。大部分是 0/1 代表逐 layer 共用幾乎不浪費;
    少數 channel 遠大於其他 channel 才值得逐 channel 分開存。"""
    return [iv_layer_value - v for v in iv_per_channel]
