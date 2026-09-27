"""backend:一層 forward 裡「算數值段 + 掃描」這一段怎麼做。層負責建結構、抽輸出流、
診斷,中間交給 backend。浮點版在這裡,整數版在 salt_core/quant/backend.py。

backend 只透過層提供的方法讀佇列:float_values、neuron_delta_t、neuron_n_real、
scan_steps,不 import layers。
"""
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from salt_core.chunk_scan import run_layer_forward, run_layer_forward_traced
from salt_core.core import spike_step_upper_bound


class ScanOutput(NamedTuple):
    """backend.scan 的回傳。"""
    result: NamedTuple            # 浮點是 LayerForwardResult,整數是 LayerForwardResultInt
    spike_gain: jax.Array         # (n_neurons, 步數),下一層的 event_gain
    steps_needed: jax.Array       # int32 純量,這筆樣本需要的掃描步數;整數版不用步數上限,是 0
    v_steps: jax.Array | None     # (n_neurons, 步數) 每步結束的膜電位,trace=True 才有
    pointer_steps: jax.Array | None  # (n_neurons, 步數) 每步開始時的佇列欄位,trace=True 才有


@dataclass(frozen=True)
class FloatBackend:
    """浮點 backend:params 是浮點權重,surrogate 掃描。訓練跟浮點推論用。"""

    def scan(self, layer, structure, w: jax.Array, event_gain: jax.Array | None, *,
             trace: bool) -> ScanOutput:
        """一層的數值段 + 掃描。structure 是層自己的佇列結構,w 是這層的浮點權重。"""
        maps = layer.float_values(structure, w, event_gain)
        scan_kwargs = dict(chunk_size=layer.chunk_size, max_steps=layer.scan_steps(structure),
                           alpha=layer.alpha, n_real_events=layer.neuron_n_real(structure))
        if trace:
            result, v_steps, pointer_steps = run_layer_forward_traced(
                maps, layer.v_th, **scan_kwargs)
        else:
            result = run_layer_forward(maps, layer.v_th, **scan_kwargs)
            v_steps = pointer_steps = None
        steps_needed = jnp.max(spike_step_upper_bound(maps.b, layer.v_th, layer.chunk_size))
        return ScanOutput(result=result, spike_gain=result.s_spike, steps_needed=steps_needed,
                          v_steps=v_steps, pointer_steps=pointer_steps)

    def readout(self, result, params):
        """最後一層結果換成解碼器要的尺度。浮點版已經是物理尺度,原樣回傳。"""
        return result


FLOAT = FloatBackend()
