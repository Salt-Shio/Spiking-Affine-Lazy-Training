"""權重的訓練後量化(PTQ):對稱、無 zero-point 的線性量化,量化再反量化模擬誤差。

推導見 docs/math/權重量化推導.md;方法決策見 docs/規格書.md「FPGA 部署:權重量化」。
"""
from collections.abc import Sequence
from typing import TypedDict

import jax.numpy as jnp

from salt_core.quant.codes import quantize_to_int


def fake_quantize_tensor(x: jnp.ndarray, bits: int, *, axis: int | None = None,
                         threshold: jnp.ndarray | float | None = None
                         ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """對稱線性量化再反量化(模擬量化誤差),不改變形狀。參數同 quantize_to_int。

    回傳 (x_hat, scale):x_hat = q * scale,是乘回物理尺度的浮點權重,可以直接給層的
    forward 用。要整數碼 q 本身用 quantize_to_int。
    """
    q, scale = quantize_to_int(x, bits, axis=axis, threshold=threshold)
    return q * scale, scale


class QuantizationError(TypedDict):
    """quantization_error 的回傳。"""
    mse: float
    max_abs_err: float
    sqnr_db: float


def quantization_error(x: jnp.ndarray, x_hat: jnp.ndarray) -> QuantizationError:
    """量化前後的誤差統計。

    sqnr_db 是訊號功率對
    量化噪聲功率的比值(dB);x 跟 x_hat 全等時 mse=0、sqnr_db=inf。
    """
    x = jnp.asarray(x)
    x_hat = jnp.asarray(x_hat)
    err = x - x_hat
    mse = float(jnp.mean(err ** 2))
    max_abs_err = float(jnp.max(jnp.abs(err)))
    signal_power = float(jnp.mean(x ** 2))
    sqnr_db = float("inf") if mse <= 0.0 else 10.0 * float(jnp.log10(signal_power / mse))
    return {"mse": mse, "max_abs_err": max_abs_err, "sqnr_db": sqnr_db}


def percentile_abs_threshold(x: jnp.ndarray, percentile: float, axis: int | None
                             ) -> jnp.ndarray:
    """|x| 的第 percentile 百分位,當 clip 門檻。

    axis=None 對整個張量取一個值;給 axis 時在 axis 的每個位置各取一個,形狀保留其餘軸為 1,
    可以直接當 fake_quantize_tensor 的 threshold。percentile=100 等於 max(|x|)。
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


def quantize_params(params: Sequence[jnp.ndarray], *, bits: int, per_channel: bool = True,
                    clip_percentile: float = 100.0) -> tuple[jnp.ndarray, ...]:
    """整份權重(每層一個陣列的 tuple)套用 PTQ,回傳同形狀的量化再反量化權重。

    per_channel: True 時每個輸出 channel(axis 0)各自量化,False 時整個張量一起。
    clip_percentile: 截斷門檻取 |w| 的第幾百分位,100 等於 max(|w|)。
    """
    out = []
    for w in params:
        axis = 0 if per_channel else None
        threshold = percentile_abs_threshold(w, clip_percentile, axis=axis)
        w_hat, _scale = fake_quantize_tensor(w, bits, axis=axis, threshold=threshold)
        out.append(w_hat)
    return tuple(out)
