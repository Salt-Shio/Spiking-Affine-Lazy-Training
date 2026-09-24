"""權重的線性(均勻)量化 primitive,FPGA 部署三部曲(權重/剪枝/膜電位)第 1 步。

完整推導見 docs/math/權重量化推導.md;方法決策見 docs/規格書.md「FPGA 部署:
權重量化」。這裡只放對稱、無 zero-point 的線性量化——公式跟取捨理由都在推導
文件裡,這裡不重複。

放 salt_core:量化 primitive(`fake_quantize_tensor`/`quantization_error`)
是純數值運算,不是這個專案專屬;`quantize_params` 需要知道
`salt_core.layers` 的權重 axis 慣例(ConvLayer/FCLayer 的 `weight_shape`
都是 axis 0 = 輸出 channel/neuron),但不碰 data/example,跟
`salt_core/dormant.py` 同一個放置判準。

目前只服務 PTQ(post-training,不重訓)。QAT 需要的
straight-through estimator(`jax.custom_vjp`)還沒實作,見推導文件步驟 4。
"""
import jax.numpy as jnp


def fake_quantize_tensor(x: jnp.ndarray, bits: int, *, axis: int | None = None,
                         threshold: jnp.ndarray | float | None = None):
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

    回傳 `(x_hat, scale)`:`x_hat` 是量化再反量化的結果(浮點,可以直接餵給
    `salt_core.layers` 的 forward 驗證 PTQ 準確率);`scale` 是實際用的量化
    步長 `Δ`。全零的 channel(`threshold<=0`)會被夾成 `threshold=1.0`
    避免除以零,`x_hat` 對這種 channel 恆為 0(輸入本來就全零)。
    """
    if bits < 2:
        raise ValueError(f"bits 必須 >= 2,給的是 {bits!r}")
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
    q = jnp.round(x_clipped / scale)
    x_hat = q * scale
    return x_hat, scale


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
