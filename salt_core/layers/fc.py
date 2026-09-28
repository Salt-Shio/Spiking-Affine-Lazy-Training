"""FCLayer:全連接層。"""
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp

from salt_core.capacity import Capacity
from salt_core.connectivity.fc import (FCQueueStructure, build_fc_structure, fc_float_values,
                                        fc_weight_codes)
from salt_core.float.affine import AffineMap, base_scan_steps
from salt_core.float.backend import FLOAT
from salt_core.layers.base import LayerOutput, _check_leading_axis, _layer_diag, uniform_init
from salt_core.stream import EventStream, extract_output_events_fc
from salt_core.trace import LayerForwardTrace, resolve_ms_fc


@dataclass(frozen=True)
class FCLayer:
    """一個密集版 FC 層,可以當隱藏層或輸出層。輸出層要不要 fire 是門檻設定,
    怎麼讀成預測是解碼器的事。

    佇列是整條輸入流,長度 = 上一層的輸出容量,不是自己的欄位。掃描步數 =
    ceil(輸入流長度 / chunk_size) + max_extra_steps;上一層容量變時前一項跟著變,
    旋鈕只記因為 fire 要多跑的步數(不 fire 時 0 就夠)。
    """
    name: str
    n_in: int
    n_out: int
    init_k: float
    # 神經元動力學 —— 有預設,跟 ConvLayer 一樣。
    tau: float = 16.0
    v_th: float = 1.0
    alpha: float = 2.0
    chunk_size: int = 1
    # 容量 —— 有預設,出界會自己長大。
    max_out_spikes: int = 8192
    max_extra_steps: int = 0

    @property
    def n_neurons(self) -> int:
        return self.n_out

    @property
    def input_shape(self) -> tuple:
        return (self.n_in,)

    @property
    def output_shape(self) -> tuple:
        return (self.n_out,)

    @property
    def fan_in(self) -> int:
        return self.n_in

    @property
    def weight_shape(self) -> tuple:
        return (self.n_out, self.n_in)

    @property
    def capacity(self) -> Capacity:
        return Capacity(max_out_spikes=self.max_out_spikes, max_extra_steps=self.max_extra_steps)

    def with_capacity(self, capacity: Capacity) -> "FCLayer":
        """換成 capacity 的容量值,其他欄位不變。"""
        return replace(self, **capacity)

    def init_weight(self, key: jax.Array) -> jax.Array:
        return uniform_init(key, self.weight_shape, self.fan_in, self.init_k)

    def unflatten_neurons(self, values):
        """同 ConvLayer.unflatten_neurons。每顆神經元自成一個 channel,
        回傳 (n_out, 1, 1, ...)。"""
        _check_leading_axis(values, self.n_out, "n_out")
        return values.reshape(self.n_out, 1, 1, *values.shape[1:])

    def broadcast_channels(self, values):
        """同 ConvLayer.broadcast_channels。每顆神經元自成一個 channel,原樣回傳。"""
        _check_leading_axis(values, self.n_out, "n_out")
        return values

    def forward(self, params, in_stream: EventStream, *, backend=FLOAT,
                trace: bool = False) -> LayerOutput:
        """同 ConvLayer.forward。params 在浮點 backend 是權重 (n_out, n_in)。"""
        structure = self.build_structure(in_stream)
        scan = backend.scan(self, structure, params, in_stream.event_gain, trace=trace)
        out_stream = extract_output_events_fc(
            scan.result.spike_mask, scan.result.spike_event_idx, scan.spike_gain,
            in_stream.event_times, max_total_spikes=self.max_out_spikes)
        diag = _layer_diag(scan.result.spike_mask, self.n_neurons, in_stream.n_real_events,
                           needed={"max_out_spikes": out_stream.n_real_events,
                                   "max_extra_steps": scan.extra_steps_needed})
        layer_trace = None
        if trace:
            event_ms = resolve_ms_fc(scan.pointer_steps, self.neuron_n_real(structure),
                                     in_stream.event_times)
            layer_trace = LayerForwardTrace(spike_mask=scan.result.spike_mask,
                                            v_steps=scan.v_steps, event_ms=event_ms)
        return LayerOutput(stream=out_stream, result=scan.result, diag=diag, trace=layer_trace)

    def build_structure(self, in_stream: EventStream) -> FCQueueStructure:
        """這層的佇列結構段。"""
        return build_fc_structure(in_stream.event_times, in_stream.event_source_idx,
                                  in_stream.n_real_events)

    def float_values(self, structure: FCQueueStructure, w: jax.Array,
                     event_gain: jax.Array | None) -> AffineMap:
        """浮點數值段,a、b 形狀 (n_out, 輸入流長度)。"""
        return fc_float_values(structure, w, self.tau, event_gain)

    def gather_weight_codes(self, structure: FCQueueStructure, q: jax.Array) -> jax.Array:
        """整數數值段,int32,(n_out, 輸入流長度)。"""
        return fc_weight_codes(structure, q)

    def neuron_delta_t(self, structure: FCQueueStructure) -> jax.Array:
        """逐神經元的 Δt,(n_out, 輸入流長度),每顆神經元都一樣。"""
        return jnp.broadcast_to(structure.delta_t[None, :],
                                (self.n_out, structure.delta_t.shape[0]))

    def neuron_n_real(self, structure: FCQueueStructure) -> jax.Array:
        """逐神經元的真事件數,(n_out,),每顆神經元都一樣。"""
        return jnp.broadcast_to(structure.n_real_events, (self.n_out,))

    def scan_steps(self, structure: FCQueueStructure) -> int:
        """浮點掃描的步數:ceil(輸入流長度 / chunk_size) + max_extra_steps。"""
        return base_scan_steps(structure.delta_t.shape[0], self.chunk_size) + self.max_extra_steps

    def with_chunk_size(self, chunk_size: int) -> "FCLayer":
        """換 chunk_size,max_extra_steps 不變。

        chunk_size=1 時需求一定是 0,一定夠;換成其他值可能不夠,由 forward 的 fits 偵測
        (輸入流長度在建構時不知道,算不出一定夠的值)。
        """
        return replace(self, chunk_size=chunk_size)
