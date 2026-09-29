"""浮點模型 + 量化規格 -> 每層量化版 forward 的參數(QuantizedLayerParams)。

公式見 docs/math/權重量化推導.md、docs/math/膜電位量化推導.md。
"""
from collections.abc import Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from salt_core.layers.base import Layer
from salt_core.quant.params import QuantizedLayerParams
from salt_core.quant.codes import (build_decay_table_int, iv_from_measurement, iv_layer,
                                   max_weight_code, quantize_to_int, v_th_to_int)
from salt_core.quant.fixed_point import OverflowMode
from salt_core.quant.ptq import percentile_abs_threshold


class LayerQuantSpec(NamedTuple):
    """一層的量化規格。"""
    bits: int                  # 權重位元寬度 b
    f_a: int                   # 衰減碼小數位元
    f_V: int                   # 暫存器小數位元
    per_channel: bool = True   # False 時整層共用一個 s_c
    fires: bool = True         # False 時不算門檻(例如膜電位回歸的輸出層)
    clip_percentile: float = 100.0  # 截斷門檻取 |w| 的第幾百分位,100 等於 max-abs
    overflow_mode: OverflowMode | str = OverflowMode.WRAP


def weight_codes(w: jax.Array, spec: LayerQuantSpec) -> tuple[jax.Array, jax.Array]:
    """權重換成整數碼。

    w: 權重,第 0 軸是輸出 channel。
    回傳 (q, scale):q 是 int32 整數碼,形狀同 w;scale 是逐 channel 的 s_c,
        形狀 (w.shape[0],),per_channel=False 時每個 channel 同一個值。
    """
    axis = 0 if spec.per_channel else None
    threshold = percentile_abs_threshold(w, spec.clip_percentile, axis=axis)
    q, scale = quantize_to_int(w, bits=spec.bits, axis=axis, threshold=threshold)
    return q, jnp.broadcast_to(jnp.asarray(scale).reshape(-1), (w.shape[0],))


def iv_per_channel(v_abs_max: npt.ArrayLike, scale: npt.ArrayLike, bits: int) -> list[int]:
    """逐 channel 的 i_V。

    v_abs_max: (n_channels,) 實測的膜電位單邊最大量值 M。
    scale: (n_channels,) 逐 channel 的 s_c(weight_codes 的回傳)。
    長度不一樣時 raise ValueError;M <= 0 時 iv_from_measurement raise ValueError。
    """
    v_abs_max = np.asarray(v_abs_max)
    clip_threshold = np.asarray(scale) * max_weight_code(bits)
    if v_abs_max.shape != clip_threshold.shape:
        raise ValueError(f"v_abs_max 形狀 {v_abs_max.shape} 跟 scale 形狀 "
                         f"{clip_threshold.shape} 不一樣")
    return [iv_from_measurement(float(m), float(t), bits)
            for m, t in zip(v_abs_max, clip_threshold)]


def build_quantized_params(layers: Sequence[Layer], float_params: Sequence[jax.Array],
                           specs: Sequence[LayerQuantSpec], v_abs_max: Sequence[np.ndarray]
                           ) -> list[QuantizedLayerParams]:
    """每層的 QuantizedLayerParams,直接當 run_network 在 QuantBackend 下的 weights。

    float_params: 對齊 layers 的浮點權重。
    specs: 對齊 layers 的 LayerQuantSpec。
    v_abs_max: 對齊 layers,每層 (n_channels,) 的實測 M(見 quant.calibrate)。
    i_V 取整層最吃緊的 channel。layers、float_params、specs、v_abs_max 長度不一樣時
    raise ValueError。
    """
    lengths = {len(layers), len(float_params), len(specs), len(v_abs_max)}
    if len(lengths) != 1:
        raise ValueError(f"layers、float_params、specs、v_abs_max 長度要一樣,拿到 "
                         f"{len(layers)}、{len(float_params)}、{len(specs)}、{len(v_abs_max)}")
    out = []
    for layer, w, spec, layer_v_abs_max in zip(layers, float_params, specs, v_abs_max):
        q, scale_per_channel = weight_codes(w, spec)
        i_V = iv_layer(iv_per_channel(layer_v_abs_max, scale_per_channel, spec.bits))
        scale_per_neuron = layer.broadcast_channels(scale_per_channel)
        v_th_int = (v_th_to_int(layer.v_th, scale_per_neuron, spec.f_V, i_V)
                    if spec.fires else None)
        out.append(QuantizedLayerParams(
            q=q, decay_table_int=build_decay_table_int(spec.f_a, layer.tau), v_th_int=v_th_int,
            scale=scale_per_neuron, f_a=spec.f_a, f_V=spec.f_V, i_V=i_V,
            overflow_mode=spec.overflow_mode))
    return out
