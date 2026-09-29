"""浮點 backend:浮點數值段 + surrogate 掃描。介面見 salt_core/backend.py。"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from salt_core.backend import QueueStructure, ScanLayer, ScanOutput
from salt_core.float.affine import extra_steps_upper_bound
from salt_core.float.scan import FloatLayerResult, run_layer, run_layer_traced


@dataclass(frozen=True)
class FloatBackend:
    """浮點 backend:params 是浮點權重,surrogate 掃描。訓練跟浮點推論用。"""

    def scan(self, layer: ScanLayer, structure: QueueStructure, w: jax.Array,
             event_gain: jax.Array | None, *, trace: bool) -> ScanOutput:
        """一層的數值段 + 掃描。structure 是層自己的佇列結構,w 是這層的浮點權重。"""
        maps = layer.float_values(structure, w, event_gain)
        scan_kwargs = dict(chunk_size=layer.chunk_size, max_steps=layer.scan_steps(structure),
                           alpha=layer.alpha, n_real_events=layer.neuron_n_real(structure))
        if trace:
            result, v_steps, pointer_steps = run_layer_traced(
                maps, layer.v_th, **scan_kwargs)
        else:
            result = run_layer(maps, layer.v_th, **scan_kwargs)
            v_steps = pointer_steps = None
        extra_steps_needed = jnp.max(extra_steps_upper_bound(maps.b, layer.v_th, layer.chunk_size))
        return ScanOutput(result=result, spike_gain=result.s_spike,
                          extra_steps_needed=extra_steps_needed,
                          v_steps=v_steps, pointer_steps=pointer_steps)

    def readout(self, result: FloatLayerResult, params: jax.Array) -> FloatLayerResult:
        """最後一層結果換成解碼器要的尺度。浮點版已經是物理尺度,原樣回傳。"""
        return result


FLOAT = FloatBackend()
