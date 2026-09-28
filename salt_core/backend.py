"""backend 介面:一層 forward 裡「算數值段 + 掃描」這一段怎麼做。層負責建結構、抽輸出流、
診斷,中間交給 backend。浮點版在 salt_core/float/backend.py,整數版在
salt_core/quant/backend.py。

backend 要提供 scan(layer, structure, params, event_gain, *, trace) -> ScanOutput 跟
readout(result, params)。backend 只透過層提供的方法讀佇列:float_values、
gather_weight_codes、neuron_delta_t、neuron_n_real、scan_steps,不 import layers。
"""
from typing import NamedTuple

import jax


class ScanOutput(NamedTuple):
    """backend.scan 的回傳。"""
    result: NamedTuple            # 浮點是 LayerForwardResult,整數是 QuantLayerResult
    spike_gain: jax.Array         # (n_neurons, 步數),下一層的 event_gain
    steps_needed: jax.Array       # int32 純量,這筆樣本需要的掃描步數;整數版不用步數上限,是 0
    v_steps: jax.Array | None     # (n_neurons, 步數) 每步結束的膜電位,trace=True 才有
    pointer_steps: jax.Array | None  # (n_neurons, 步數) 每步開始時的佇列欄位,trace=True 才有
