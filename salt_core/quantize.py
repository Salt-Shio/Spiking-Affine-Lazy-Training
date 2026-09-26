"""量化 primitive,FPGA 部署三部曲(權重/剪枝/膜電位)第 1、3 步共用同一個檔案。

**第 1 步(權重)**:對稱、無 zero-point 的線性量化——`fake_quantize_tensor`/
`quantization_error`/`quantize_params`。完整推導見 docs/math/權重量化推導.md;
方法決策見 docs/規格書.md「FPGA 部署:權重量化」。

**第 3 步(膜電位)**:$a_k$ 衰減查表(`delta_t_max`/`build_decay_table`/
`build_decay_table_int`/`apply_decay_table_int`)、$i_V$ 位元數公式
(`iv_positive_lower_bound`/`iv_from_measurement`/`iv_layer`/`waste`)、整數
尺度運算 primitive(`round_half_away_from_zero`/`round_shift`/
`wide_mul_shift`/`wrap_to_bits`/`v_th_to_int`,見 docs/問題紀錄.md
第十七~十九節)。這幾個函式建表/建門檻/做單步整數運算,不跑遞迴——遞迴
掃描本身在 `salt_core/chunk_scan.py`(浮點路 `run_layer_forward` 的
`round_step`/`round_mode`;整數路 `run_layer_forward_int`)。完整推導見
docs/math/膜電位量化推導.md、docs/問題紀錄.md 第十七~十九節。

放 salt_core:都是純數值運算,不是這個專案專屬(`quantize_params` 額外需要知道
`salt_core.layers` 的權重 axis 慣例:ConvLayer/FCLayer 的 `weight_shape` 都是
axis 0 = 輸出 channel/neuron,但不碰 data/example),跟 `salt_core/dormant.py`
同一個放置判準。

第 1 步目前只服務 PTQ(post-training,不重訓)。QAT 需要的
straight-through estimator(`jax.custom_vjp`)還沒實作,見權重量化推導文件
步驟 4。
"""
import math

import jax.numpy as jnp


def quantize_to_int(x: jnp.ndarray, bits: int, *, axis: int | None = None,
                    threshold: jnp.ndarray | float | None = None,
                    mode: str = "round"):
    """對稱線性量化,只回傳整數碼 `q` 跟 `scale`,不乘回去反量化。

    `fake_quantize_tensor` 拿到的 `x_hat=q*scale` 是「乘回物理尺度的浮點數」,
    給不需要知道整數碼是什麼的呼叫端用(量測 PTQ 準確率);這個函式給的是
    `q` 本身——`AffineMap.b` 要餵這個乾淨整數進去,不是 `q*s_c`(見
    docs/問題紀錄.md 第十七節,`s_c` 不能在整數版遞迴裡出現)。

    參數意義跟 `fake_quantize_tensor` 完全一樣(`bits`/`axis`/`threshold`/
    `mode`),這裡不重複講。`mode="round"` 用 `round_half_away_from_zero`
    (逢五進一),不是 `jnp.round` 的逢五取偶(見 docs/問題紀錄.md 第十八節,
    這裡定案成跟 FPGA 電路一致的慣例)。
    """
    if bits < 2:
        raise ValueError(f"bits 必須 >= 2,給的是 {bits!r}")
    if mode not in ("round", "truncate"):
        raise ValueError(f"mode 必須是 'round' 或 'truncate',給的是 {mode!r}")
    x = jnp.asarray(x)
    levels = 2 ** (bits - 1) - 1

    if threshold is None:
        if axis is None:
            threshold = jnp.max(jnp.abs(x))
        else:
            reduce_axes = tuple(i for i in range(x.ndim) if i != axis)
            threshold = jnp.max(jnp.abs(x), axis=reduce_axes, keepdims=True)
    threshold = jnp.asarray(threshold, dtype=x.dtype)
    threshold = jnp.where(threshold > 0, threshold, jnp.ones_like(threshold))

    scale = threshold / levels
    x_clipped = jnp.clip(x, -threshold, threshold)
    x_scaled = x_clipped / scale
    q = round_half_away_from_zero(x_scaled) if mode == "round" else jnp.trunc(x_scaled)
    return q, scale


def fake_quantize_tensor(x: jnp.ndarray, bits: int, *, axis: int | None = None,
                         threshold: jnp.ndarray | float | None = None,
                         mode: str = "round"):
    """對稱線性量化再反量化(模擬量化誤差),不改變 shape。

    `bits`:位元寬度,整數編碼範圍 `{-(2**(bits-1)-1), ..., 2**(bits-1)-1}`
    (犧牲一個編碼點換嚴格對稱,見推導文件步驟 1)。`bits < 2` 無意義
    (至少要有正負兩格)。

    `axis`:`None` 是 per-tensor(整個 `x` 共用一個 threshold);給一個軸
    索引就是沿該軸 per-channel(該軸每個位置各自的 threshold,對其餘軸取
    `max(|x|)` 或用呼叫端傳入的 `threshold`)。

    `threshold`:clip 門檻 `T`。`None` 時預設 `max(|x|)`(沿 `axis` 之外的軸
    reduce,`axis=None` 時整個 tensor reduce);呼叫端也可以自己算好
    (例如某個 percentile)傳進來,對應推導文件步驟 2 的網格搜尋。

    `mode`:量化到整數碼這一步要「四捨五入」(`"round"`,預設,權重量化
    目前唯一用的模式,逢五進一,見 `quantize_to_int`)還是「直接砍掉多餘
    小數位元」(`"truncate"`,對稱量化下等於向零捨去)。膜電位量化的捨入
    規則 $r(\\cdot)$(見 docs/math/膜電位量化推導.md)兩種都要能選,才能掃出
    哪種對準確率/硬體成本比較划算,所以在這裡加一個選項,不是另外重寫一次
    量化邏輯。

    回傳 `(x_hat, scale)`:`x_hat` 是量化再反量化的結果(浮點,可以直接餵給
    `salt_core.layers` 的 forward 驗證 PTQ 準確率);`scale` 是實際用的量化
    步長 `Δ`。全零的 channel(`threshold<=0`)會被夾成 `threshold=1.0`
    避免除以零,`x_hat` 對這種 channel 恆為 0(輸入本來就全零)。整數碼 `q`
    本身(不乘回 `scale`)要用 `quantize_to_int`。
    """
    q, scale = quantize_to_int(x, bits, axis=axis, threshold=threshold, mode=mode)
    return q * scale, scale


def quantization_error(x: jnp.ndarray, x_hat: jnp.ndarray) -> dict:
    """量化前後的誤差統計,純歸約,不管量化怎麼做的。

    回傳 `{"mse": float, "max_abs_err": float, "sqnr_db": float}`。
    `sqnr_db`(訊號功率對量化噪聲功率的比值,dB)是推導文件步驟 2 拿來比較
    不同 clip threshold/bit width 的主要指標;`x`、`x_hat` 全等時 `mse=0`,
    `sqnr_db` 回傳 `inf`。
    """
    x = jnp.asarray(x)
    x_hat = jnp.asarray(x_hat)
    err = x - x_hat
    mse = float(jnp.mean(err ** 2))
    max_abs_err = float(jnp.max(jnp.abs(err)))
    signal_power = float(jnp.mean(x ** 2))
    sqnr_db = float("inf") if mse <= 0.0 else 10.0 * float(jnp.log10(signal_power / mse))
    return {"mse": mse, "max_abs_err": max_abs_err, "sqnr_db": sqnr_db}


def _percentile_abs_threshold(x: jnp.ndarray, percentile: float, axis: int | None):
    """`|x|` 的 percentile 當 clip threshold。`axis=None` 對整個 tensor 取;
    給定 `axis` 時,對其餘所有軸攤平後在該軸的每個位置各自取,回傳形狀跟
    `fake_quantize_tensor` 的 `threshold` 參數(keepdims 廣播用)相容。
    `percentile=100.0` 精確等於 `max(|x|)`,跟 `fake_quantize_tensor` 沒給
    `threshold` 時的預設行為一致。
    """
    abs_x = jnp.abs(x)
    if axis is None:
        return jnp.percentile(abs_x, percentile)
    moved = jnp.moveaxis(abs_x, axis, 0)
    flat = moved.reshape(moved.shape[0], -1)
    thresh = jnp.percentile(flat, percentile, axis=1)
    shape = [1] * abs_x.ndim
    shape[axis] = -1
    return thresh.reshape(shape)


def quantize_params(layers: list, params: tuple, *, bits: int, per_channel: bool = True,
                    clip_percentile: float = 100.0) -> tuple:
    """對齊 `layers` 的整份權重(每層一個陣列的 tuple)套用 PTQ。

    `per_channel=True` 時用每層 `weight_shape` 的 axis 0(`ConvLayer`/
    `FCLayer` 都是輸出 channel/neuron)當量化軸,`per_channel=False` 是
    per-tensor,兩者的取捨見推導文件步驟 3。`clip_percentile` 是推導文件
    步驟 2 網格搜尋的旋鈕:`100.0`(預設)等於 max-abs,呼叫端掃一組候選值
    (例如 100/99.9/99/95)搭配 `quantization_error` 或直接接
    `example.utils.make_evaluate` 量測 PTQ 準確率,取最好的組合。

    只換權重數值,不動 `layers`(容量、幾何、chunk_size 都不變)——回傳的
    `params` 可以直接餵給既有的 `run_network`/`make_evaluate`。
    """
    out = []
    for layer, w in zip(layers, params):
        axis = 0 if per_channel else None
        threshold = _percentile_abs_threshold(w, clip_percentile, axis=axis)
        w_hat, _scale = fake_quantize_tensor(w, bits, axis=axis, threshold=threshold)
        out.append(w_hat)
    return tuple(out)


# ============================================================================
# 膜電位量化(FPGA 部署三部曲第 3 步),見 docs/math/膜電位量化推導.md。
# ============================================================================

def delta_t_max(f_a: int, tau: float) -> int:
    """$a_k=(1-1/\\tau)^{\\Delta t}$ 查表要開多深(推導文件「$a_k$:查表」節)。

    $\\Delta t=0$ 恆等於 $a=1.0$(沒有經過任何衰減),不透過表(Q0.$f_a$ 格式
    值域不含 1,見 `build_decay_table`);這裡只算 $\\Delta t \\ge 1$ 的部分。
    回傳最大的 $\\Delta t$,使得 $(1-1/\\tau)^{\\Delta t}$ 捨入到 $f_a$ 個小數
    位元後仍不是 0(即 $\\ge 2^{-(f_a+1)}$,半個最小刻度);超過這個 $\\Delta t$
    的表項全部是 0,不用存(`apply_decay_table_int` 直接回傳 0,不用查表)。
    """
    eps = 2.0 ** -(f_a + 1)
    log_base = math.log(1.0 - 1.0 / tau)  # < 0(tau > 1 時)
    return max(int(math.floor(math.log(eps) / log_base)), 0)


def build_decay_table(f_a: int, tau: float) -> jnp.ndarray:
    """建 $a_k$ 查表,shape `(delta_t_max(f_a, tau),)`——`table[i]` 是
    $\\Delta t=i+1$(索引從 $\\Delta t=1$ 開始,見 `delta_t_max`)的量化值。

    量化本身直接重用 `fake_quantize_tensor`:Q0.$f_a$(無符號、無整數位元,
    值域 $[0, 1-2^{-f_a}]$)等於 `bits=f_a+1` 的對稱量化器,把 threshold 設成
    這個格式能表示的最大值 $(2^{f_a}-1)\\cdot 2^{-f_a}$,算出來的 `scale` 就
    精確等於 $2^{-f_a}$——表項全部是正數,`fake_quantize_tensor` 允許負值的
    對稱範圍在這裡沒有用到,不影響結果。
    """
    n = delta_t_max(f_a, tau)
    delta_t = jnp.arange(1, n + 1, dtype=jnp.float32)
    a_exact = (1.0 - 1.0 / tau) ** delta_t
    threshold = (2.0 ** f_a - 1) * 2.0 ** -f_a
    a_q, _scale = fake_quantize_tensor(a_exact, bits=f_a + 1, threshold=threshold)
    return a_q


def build_decay_table_int(f_a: int, tau: float) -> jnp.ndarray:
    """`build_decay_table` 的整數版:回傳整數碼(`0 ~ 2^f_a-1`),不是乘完
    `2^-f_a` 的小數——餵給整數版遞迴的表要跟 `s_c` 完全無關(見
    docs/問題紀錄.md 第十七節),`table[i]/2^f_a` 才是實際衰減值。

    shape/索引慣例跟 `build_decay_table`一致(`table[i]` 對應
    $\\Delta t=i+1$)。`Δt=0`(恆等於 1.0)不在這張表裡——Q0.$f_a$ 整數格式
    最大只到 `2^f_a-1`(代表 `(2^f_a-1)/2^f_a<1`),塞不下剛好等於 1 的值,
    這個特例由 `apply_decay_table_int` 用另一個旗標處理,不是查表查出來的。
    """
    n = delta_t_max(f_a, tau)
    delta_t = jnp.arange(1, n + 1, dtype=jnp.float32)
    a_exact = (1.0 - 1.0 / tau) ** delta_t
    threshold = (2.0 ** f_a - 1) * 2.0 ** -f_a
    a_q_int, _scale = quantize_to_int(a_exact, bits=f_a + 1, threshold=threshold)
    return a_q_int.astype(jnp.int32)


def apply_decay_table_int(delta_t: jnp.ndarray, table_int: jnp.ndarray):
    """整數版查表:直接吃真正的整數 Δt(佇列建構那一步本來就算過,見
    docs/問題紀錄.md 第十九節),不是從 `a` 反推 `log(a)/log(1-1/tau)` 再
    查表——反推在 $\\Delta t$ 大到 float32 精度逼近 denormal 時會失真(連續
    幾個不同的 Δt 被捨進同一個浮點值),直接拿整數 Δt 當 index 完全不會有
    這個問題。

    回傳 `(a_int, is_identity)`:
    - `is_identity=True`(`Δt=0`):`a_int` 這個位置的值沒有意義,呼叫端要
      整個跳過衰減這一步(乘法、捨位都不做,衰減後的值直接等於原本的
      $\\tilde V_{k-1}$)——原因見 `build_decay_table_int`。
    - `is_identity=False`:`1<=Δt<=table深度` 查表;`Δt` 更大代表衰減到量化
      格式存不下,回傳整數 `0`(這個可以正常走乘法/捨位,結果本來就是 0,
      不需要額外特判)。
    """
    n = table_int.shape[0]
    idx = jnp.clip(delta_t - 1, 0, n - 1)
    gathered = table_int[idx]
    a_int = jnp.where(delta_t > n, jnp.zeros_like(gathered), gathered)
    is_identity = delta_t == 0
    return a_int, is_identity


def v_th_to_int(v_th: jnp.ndarray | float, s_c: jnp.ndarray | float, f_V: int) -> jnp.ndarray:
    """把物理尺度的門檻 $v_{th}$ 換算成整數版遞迴要比較的門檻
    $\\lfloor \\tilde v_{th} \\cdot 2^{f_V} \\rceil$(逢五進一,`⌈⌋` 代表這個
    捨入慣例),$\\tilde v_{th}=v_{th}/s_c$。

    這個除以 `s_c` 只做這一次(建門檻,不是逐事件遞迴裡的運算),換算完之後
    整數版遞迴全程只跟這個整數門檻比大小,不會再看到 `s_c`——這正是第十七節
    要求的:`s_c` 只能在邊界出現一次,不能在迴圈裡反覆出現。
    """
    v_th_tilde = jnp.asarray(v_th) / jnp.asarray(s_c)
    return round_half_away_from_zero(v_th_tilde * (2 ** f_V)).astype(jnp.int32)


def iv_positive_lower_bound(v_th_tilde: float, b: int) -> int:
    """$i_V$ 正向(fire 方向)硬性下限(推導文件「conv1/conv2 的正向、負向界」節):

    $$i_V \\ge \\lceil \\log_2(\\tilde v_{th} + \\max(q_k)) \\rceil + 1$$

    `v_th_tilde`:$\\tilde v_{th}=v_{th}/s_c$(呼叫端自己換算好傳進來,這個
    函式不知道、也不需要知道 $s_c$ 怎麼來的)。`b`:候選權重 bit width,
    $\\max(q_k)=2^{b-1}-1$。低於這個值正向一定溢位,是硬性下限,不是建議值。
    """
    max_q = 2 ** (b - 1) - 1
    return math.ceil(math.log2(v_th_tilde + max_q)) + 1


def iv_from_measurement(M: float, T_c: float, b: int) -> int:
    """逐 channel $i_V(b)$(推導文件「通用量測與公式」節):

    $$i_V^{(l,c)}(b) = \\left\\lceil \\log_2\\!\\left(M^{(l,c)} \\cdot \\frac{2^{b-1}-1}{T_c}\\right) \\right\\rceil + 1$$

    `M`:這個 channel 實測的單邊最大量值 $\\max(V_{\\max}, |V_{\\min}|)$,用
    **原始、未量化的浮點權重**跑出來的(不依賴 `b`,只跑一次,見推導文件
    「整體規劃」)。`T_c`:這個候選 `b` 底下,權重量化算出來的 clip 門檻。
    `M` 必須 > 0(完全沒有活動的 channel 應該在呼叫前就篩掉,不是餵 0 進來
    讓這裡算出沒意義的負無限大)。
    """
    if M <= 0:
        raise ValueError(f"M 必須 > 0,給的是 {M!r}(死 channel 應該先篩掉,不要呼叫這個函式)")
    levels = 2 ** (b - 1) - 1
    return math.ceil(math.log2(M * levels / T_c)) + 1


def iv_layer(iv_per_channel: list[int]) -> int:
    """逐 layer 共用的 $i_V$:取這層所有 channel 裡最吃緊的那個(推導文件
    「兩種粒度怎麼比較」節)。"""
    return max(iv_per_channel)


def waste(iv_layer_value: int, iv_per_channel: list[int]) -> list[int]:
    """每個 channel 被迫多付出的位元數(推導文件同節):$\\text{waste}^{(c)} =
    i_V^{(l)} - i_V^{(l,c)} \\ge 0$。大部分是 0/1 代表逐 layer 共用幾乎不浪費;
    少數 channel 遠大於其他 channel 才值得逐 channel 分開存。"""
    return [iv_layer_value - v for v in iv_per_channel]


# ============================================================================
# 膜電位量化 第二輪:整數尺度運算 primitive(見 docs/問題紀錄.md 第十七~
# 十九節)。舊的浮點路(`build_decay_table`/反推 Δt 查表、`layers.py` 餵給
# 遞迴的 a/b/v_th 乘回 `s_c` 的物理尺度)已經整個換掉,不是並存的兩條路——
# `s_c` 曾經出現在兩個各自獨立算出來的地方(權重貢獻、捨入格距),浮點數除法
# 不保證兩邊剛好互相消掉,跟硬體「暫存器裡從頭到尾只有整數,從來沒有 `s_c`
# 這種東西」不是同一個算法。這裡的函式讓整數版遞迴(`core.process_event_int`)
# 完全不需要碰 `s_c`,實際遞迴/掃描見 `core.py`/`chunk_scan.py`,接線見
# `layers.py` 的 `forward_quantized`。
# ============================================================================

def round_half_away_from_zero(x: jnp.ndarray) -> jnp.ndarray:
    """四捨五入,卡在正中間時「往離零的方向」進位(2.5→3、-2.5→-3),不是
    `jnp.round` 預設的逢五取偶(2.5→2)。只用在「浮點數換算成整數,只做一次」
    的場合(例如把 $v_{th}$ 換算成整數門檻),不是逐事件遞迴要用的位移運算,
    見 `round_shift`。"""
    x = jnp.asarray(x)
    return jnp.sign(x) * jnp.floor(jnp.abs(x) + 0.5)


def round_shift(x_int: jnp.ndarray, shift_bits: int, mode: str = "round") -> jnp.ndarray:
    """整數版「除以 $2^{\\text{shift\\_bits}}$ 再捨入」,對應硬體定點乘法器算完
    之後、要把多出來的小數位元捨掉那一步(例如 $a_k\\tilde V_{k-1}$ 這個乘積
    多了 $f_a$ 個小數位元,要捨回 $\\tilde V$ 原本的格式)。

    刻意不用 Python/numpy 的 `>>` 直接對負數右移:右移對負數是「向負無窮捨去」
    (floor division),不是這裡要的逢五進一或向零捨去,所以先取絕對值、算完
    再套回符號。`mode="round"` 用 `round_half_away_from_zero` 同一個逢五進一
    慣例(半格距 `1 << (shift_bits-1)` 先加上去再右移,等於 `floor(|x|/2^n+0.5)`
    的整數版);`mode="truncate"` 是直接右移(向零捨去)。

    `shift_bits=0` 是恆等(沒有多餘小數位元要捨),兩種模式都直接回傳原值。

    這一步不檢查、也不需要檢查溢位:$|a_k|<1$(嚴格小於,$\\Delta t=0$/超出表
    深度的兩種特例在呼叫端各自處理,不會呼叫這個函式),衰減只會讓量值變小,
    逢五進一最多把結果拉回到跟 `x_int` 除以 `2^shift_bits` 前的量值一樣大,
    不可能算出比原本儲存的 $\\tilde V_{k-1}$ 更大的值,所以這一步不可能讓
    暫存器寬度不夠——真正需要檢查溢位的地方是加上 $q_k$ 之後,見 `wrap_to_bits`。
    """
    if mode not in ("round", "truncate"):
        raise ValueError(f"mode 必須是 'round' 或 'truncate',給的是 {mode!r}")
    x_int = jnp.asarray(x_int)
    if shift_bits == 0:
        return x_int
    sign = jnp.sign(x_int)
    abs_x = jnp.abs(x_int)
    if mode == "round":
        half = 1 << (shift_bits - 1)
        shifted = (abs_x + half) >> shift_bits
    else:
        shifted = abs_x >> shift_bits
    return sign * shifted


def wide_mul_shift(a_int: jnp.ndarray, v_int: jnp.ndarray, shift_bits: int,
                   mode: str = "round") -> jnp.ndarray:
    """算 `round_shift(a_int * v_int, shift_bits, mode)`,但不真的算出
    `a_int*v_int` 這個寬乘積——這個專案不能開 `jax_enable_x64`(見
    `core.process_event_int` 文件,全域設定會讓訓練熱路徑一堆地方的預設整數
    dtype 跟著變),`a_int` 的位元數(`shift_bits`)跟 `v_int` 的位元數
    (`i_V+f_V`)加起來常常超過 int32 能安全相乘的範圍,直接算 `a_int*v_int`
    會在這個容器裡先溢位。

    做法是硬體乘法器內部本來就會做的事:把 `v_int` 拆成高低兩半分開乘,
    不是為了繞過限制硬湊出來的技巧。

    ```
    sign_v = sign(v_int); abs_v = |v_int|
    hi = abs_v >> shift_bits              # 非負,abs_v = hi*2^shift_bits + lo 精確成立
    lo = abs_v & (2^shift_bits - 1)        # 非負,lo < 2^shift_bits
    H  = a_int * hi                        # H < abs_v(見下方證明),跟 v_int 同一個位元數量級
    L  = a_int * lo                        # L < 2^(2*shift_bits),shift_bits<=15 時 int32 夠用
    結果 = sign_v * (H + round_shift(L, shift_bits, mode))
    ```

    `H<abs_v` 的證明:`a_int<2^shift_bits`,`hi<=abs_v/2^shift_bits`(floor),
    兩式相乘 `H=a_int*hi<abs_v`——這代表只要 `v_int` 本身能安全放進 int32,
    `H` 就一定也放得進去,不需要對 `i_V+f_V` 另外設一個更嚴的上限;真正對
    `shift_bits`(也就是 `f_a`)的限制只剩 `2*shift_bits` 要放得進 int32
    (`shift_bits<=15` 左右)。

    這裡刻意用 `round_shift(L, ...)` 算低位那一半的捨入,不是重寫一次逢五
    進一的邏輯——`L` 保證非負(`a_int`、`lo` 都非負),`round_shift` 對非負
    輸入本來就是對的答案,直接重用。

    `H*2^shift_bits + L` 的移位可以精確拆成 `H + shift(L)` 這件事,是因為
    `H*2^shift_bits` 本身就是 `2^shift_bits` 的整數倍,右移不會產生進位到
    `H` 這一側——這對任意大小、正負的 `H` 都成立,前提是 `L` 本身非負且
    不需要再帶符號。**這正是為什麼要先對 `v_int` 取絕對值再拆,不能直接對
    `v_int` 本身做有號右移去拆 hi/lo**:如果直接拆帶符號的 `v_int`,低位
    `lo` 的符號會跟著 `hi` 的符號綁在一起,逢五進一這個慣例是看「整個數字」
    的正負號決定進位方向,不是看拆出來的 high part 的正負號,兩者混在一起
    會在捨入卡在中點時算出錯的答案(手算反例:`a_int=1,v_int=-8,shift_bits=4`
    直接對 `v_int` 做有號拆分會算出 `0`,正確答案是 `-1`,見對應測試)。
    """
    if mode not in ("round", "truncate"):
        raise ValueError(f"mode 必須是 'round' 或 'truncate',給的是 {mode!r}")
    a_int = jnp.asarray(a_int)
    v_int = jnp.asarray(v_int)
    if shift_bits == 0:
        return a_int * v_int
    sign_v = jnp.sign(v_int)
    abs_v = jnp.abs(v_int)
    hi = abs_v >> shift_bits
    lo = abs_v & ((1 << shift_bits) - 1)
    big = a_int * hi
    small = a_int * lo
    shifted_mag = big + round_shift(small, shift_bits, mode)
    return sign_v * shifted_mag


def wrap_to_bits(x_int: jnp.ndarray, total_bits: int):
    """把整數繞回兩補數 `total_bits` 位元的有號表示範圍
    `[-2^(total_bits-1), 2^(total_bits-1)-1]`,對應硬體暫存器寬度不夠時真的
    會發生的溢位行為(mod $2^{\\text{total\\_bits}}$ 運算,不是夾住/飽和)。

    呼叫端要先算出「沒有繞回去之前的真實值」再傳進來,這個真實值本身就是
    拿來跟合法範圍比對、判斷有沒有溢位的基準——只要呼叫端算真實值那一步
    本身不會先溢位所在的整數容器,這個比對就準。(這個專案的整數路徑全程
    留在 `int32`,不能開 `jax_enable_x64` 換成 `int64`——會讓訓練熱路徑一堆
    地方的預設整數 dtype 跟著變,見 `core.process_event_int` 文件跟
    docs/問題紀錄.md 第十七節;呼叫端要自己確保 `total_bits` 夠小、乘法/
    加法不會先撞到 int32 的 32 位元邊界。)

    回傳 `(wrapped, overflowed)`:`wrapped` 是繞回 `total_bits` 位元後的值
    (沒溢位時就等於 `x_int` 本身);`overflowed` 是逐元素布林陣列,`True`
    代表這個位置的真實值本來就超出 `total_bits` 位元能表示的範圍。
    """
    x_int = jnp.asarray(x_int)
    mask = (1 << total_bits) - 1
    half = 1 << (total_bits - 1)
    masked = x_int & mask
    wrapped = jnp.where(masked >= half, masked - (1 << total_bits), masked)
    overflowed = (x_int < -half) | (x_int >= half)
    return wrapped, overflowed
