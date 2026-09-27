"""權重的訓練後量化(PTQ):對稱、無 zero-point 的線性量化,量化再反量化模擬誤差。
推導見 docs/math/權重量化推導.md;方法決策見 docs/規格書.md「FPGA 部署:權重量化」。

QAT 需要的 straight-through estimator 還沒實作,見權重量化推導文件步驟 4。
"""
import jax.numpy as jnp

from salt_core.quant.codes import quantize_to_int


def fake_quantize_tensor(x: jnp.ndarray, bits: int, *, axis: int | None = None,
                         threshold: jnp.ndarray | float | None = None):
    """對稱線性量化再反量化(模擬量化誤差),不改變 shape。參數見
    `quantize_to_int`。

    回傳 `(x_hat, scale)`:`x_hat = q * scale` 是乘回物理尺度的浮點數,
    可以直接餵給 `salt_core.layers` 的 forward 驗證 PTQ 準確率。整數碼 `q`
    本身要用 `quantize_to_int`。
    """
    q, scale = quantize_to_int(x, bits, axis=axis, threshold=threshold)
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


def percentile_abs_threshold(x: jnp.ndarray, percentile: float, axis: int | None):
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


def quantize_params(params: tuple, *, bits: int, per_channel: bool = True,
                    clip_percentile: float = 100.0) -> tuple:
    """整份權重(每層一個陣列的 tuple)套用 PTQ,回傳同形狀的 fake-quantized 權重。

    per_channel: True 用 axis 0(輸出 channel/neuron)當量化軸,False 是 per-tensor。
    clip_percentile: 截斷門檻取 |w| 的第幾百分位,100 等於 max-abs。
    取捨見 docs/math/權重量化推導.md。
    """
    out = []
    for w in params:
        axis = 0 if per_channel else None
        threshold = percentile_abs_threshold(w, clip_percentile, axis=axis)
        w_hat, _scale = fake_quantize_tensor(w, bits, axis=axis, threshold=threshold)
        out.append(w_hat)
    return tuple(out)
