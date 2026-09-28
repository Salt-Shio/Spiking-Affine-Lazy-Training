"""ConvLayer:conv 層。"""
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp

from salt_core.capacity import Capacity
from salt_core.connectivity.conv import (ConvQueueStructure, build_conv_structure,
                                          conv_float_values, conv_weight_codes, tile_channels,
                                          unravel_conv_source)
from salt_core.float.affine import AffineMap, base_scan_steps, safe_extra_steps
from salt_core.float.backend import FLOAT
from salt_core.layers.base import LayerOutput, _check_leading_axis, _layer_diag, uniform_init
from salt_core.stream import EventStream, extract_output_events_conv
from salt_core.trace import LayerForwardTrace, resolve_ms_conv


@dataclass(frozen=True)
class ConvLayer:
    """一個壓縮版 conv 層。靜態欄位分五組:輸入面幾何 / 這層幾何 / init_k /
    神經元動力學 / 容量。

    輸入面幾何(`ic` / `h_in` / `w_in`)= 上一層的輸出:`ic` 要等於上一層的
    `oc`,`h_in`/`w_in` 要等於上一層的 `h_out`/`w_out`——組層 list 的時候
    Python 層級檢查一次(就是 PyTorch 要你自己對齊 channel 的那個檢查)。
    第一層的「上一層」是虛擬輸入網格 `(ic, h_in, w_in)`,由呼叫端把原始事件
    ravel 成扁平編號餵進來(見 `salt_core.network.Network.input_stream`)。

    輸出面尺寸 `h_out` / `w_out` **不是欄位**,是從 `h_in` / `k` / `s` / `p`
    算的 property(floor 模式、無 dilation:`(h_in + 2p - k)//s + 1`)——沒有
    人在任何地方填它,存成欄位只會多一個可能跟其他欄位對不上的數字。
    """
    name: str
    # 輸入面幾何(= 上一層輸出)—— 必填,沒有通用預設
    ic: int
    h_in: int
    w_in: int
    # 這層幾何 —— 必填(h_out / w_out 是 property,不在這裡)
    oc: int
    k: int
    s: int
    p: int
    # 初始權重尺度 —— 必填,不校準,委定值見 docs/問題紀錄.md §12(firing-rate
    # 目標帶準則廢棄,固定 init_k=5.0)。
    init_k: float
    # 神經元動力學(逐層)—— 有預設,是「起點」,config 要覆蓋就覆蓋。
    tau: float = 16.0
    v_th: float = 1.0
    alpha: float = 2.0
    chunk_size: int = 1
    # 容量 —— 有預設。max_queue_len / max_out_spikes 的值不重要(出界會自己長大),預設只求
    # 「不要太小、少幾次開頭重編譯」。
    max_queue_len: int = 128
    max_out_spikes: int = 8192
    # 掃描步數 = ceil(max_queue_len / chunk_size) + max_extra_steps,後者是因為 fire 要多跑的步數
    # (見 docs/math/掃描步數上界推導.md)。None 時設成一定夠的值(總步數 = max_queue_len)。
    max_extra_steps: int | None = None

    def __post_init__(self) -> None:
        if self.max_extra_steps is None:
            object.__setattr__(self, "max_extra_steps",
                               safe_extra_steps(self.max_queue_len, self.chunk_size))

    @property
    def h_out(self) -> int:
        return (self.h_in + 2 * self.p - self.k) // self.s + 1

    @property
    def w_out(self) -> int:
        return (self.w_in + 2 * self.p - self.k) // self.s + 1

    @property
    def n_neurons(self) -> int:
        return self.oc * self.h_out * self.w_out

    @property
    def input_shape(self) -> tuple:
        return (self.ic, self.h_in, self.w_in)

    @property
    def output_shape(self) -> tuple:
        return (self.oc, self.h_out, self.w_out)

    @property
    def fan_in(self) -> int:
        return self.ic * self.k * self.k

    @property
    def weight_shape(self) -> tuple:
        return (self.oc, self.ic, self.k, self.k)

    @property
    def capacity(self) -> Capacity:
        return Capacity(max_queue_len=self.max_queue_len, max_out_spikes=self.max_out_spikes,
                        max_extra_steps=self.max_extra_steps)

    def with_capacity(self, capacity: Capacity) -> "ConvLayer":
        """換成 capacity 的容量值,其他欄位不變。"""
        return replace(self, **capacity)

    def init_weight(self, key: jax.Array) -> jax.Array:
        return uniform_init(key, self.weight_shape, self.fan_in, self.init_k)

    def unflatten_neurons(self, values):
        """逐神經元的值還原成 (oc, h_out, w_out, ...)。

        values: 第 0 軸長度 n_neurons,神經元編號 = c*h_out*w_out + y*w_out + x;
            其餘軸原樣保留。numpy、jax 陣列都可以,回傳同一種。
        第 0 軸長度不對時 raise ValueError。
        """
        _check_leading_axis(values, self.n_neurons, "n_neurons")
        return values.reshape(self.oc, self.h_out, self.w_out, *values.shape[1:])

    def broadcast_channels(self, values):
        """逐 channel 的值展開成逐神經元,同一個 channel 的 h_out*w_out 顆神經元同一個值。

        values: 第 0 軸長度 oc,其餘軸原樣保留。回傳第 0 軸長度 n_neurons,
            numpy、jax 陣列都可以,回傳同一種。
        第 0 軸長度不對時 raise ValueError。
        """
        _check_leading_axis(values, self.oc, "oc")
        return values.repeat(self.h_out * self.w_out, axis=0)

    def forward(self, params, in_stream: EventStream, *, backend=FLOAT,
                trace: bool = False) -> LayerOutput:
        """建壓縮佇列 -> backend 算數值段跟掃描 -> 抽輸出流、算診斷。

        params: 浮點 backend 是權重 (oc, ic, k, k);整數 backend 是 QuantizedLayerParams。
        trace: True 時 LayerOutput.trace 帶逐步軌跡。
        """
        structure = self.build_structure(in_stream)
        scan = backend.scan(self, structure, params, in_stream.event_gain, trace=trace)
        local_to_global_j = tile_channels(structure.local_to_global_j, self.oc)
        out_stream = extract_output_events_conv(
            scan.result.spike_mask, scan.result.spike_event_idx, scan.spike_gain,
            in_stream.event_times, local_to_global_j, max_total_spikes=self.max_out_spikes)
        diag = _layer_diag(scan.result.spike_mask, self.n_neurons, in_stream.n_real_events,
                           needed={"max_queue_len": jnp.max(structure.n_real_events),
                                   "max_out_spikes": out_stream.n_real_events,
                                   "max_extra_steps": scan.extra_steps_needed})
        layer_trace = None
        if trace:
            event_ms = resolve_ms_conv(scan.pointer_steps, local_to_global_j,
                                       self.neuron_n_real(structure),
                                       in_stream.event_times)
            layer_trace = LayerForwardTrace(spike_mask=scan.result.spike_mask,
                                            v_steps=scan.v_steps, event_ms=event_ms)
        return LayerOutput(stream=out_stream, result=scan.result, diag=diag, trace=layer_trace)

    def build_structure(self, in_stream: EventStream) -> ConvQueueStructure:
        """這層的佇列結構段。扁平來源編號用這層的輸入面尺寸還原成 (x, y, c)。"""
        x, y, c = unravel_conv_source(in_stream.event_source_idx, self.h_in, self.w_in)
        return build_conv_structure(
            in_stream.event_times, x, y, c, self.k, self.s, self.p, self.h_out, self.w_out,
            self.max_queue_len, in_stream.n_real_events)

    def float_values(self, structure: ConvQueueStructure, w: jax.Array,
                     event_gain: jax.Array | None) -> AffineMap:
        """浮點數值段,a、b 形狀 (n_neurons, max_queue_len)。"""
        return conv_float_values(structure, w, self.tau, event_gain)

    def gather_weight_codes(self, structure: ConvQueueStructure, q: jax.Array) -> jax.Array:
        """整數數值段,int32,(n_neurons, max_queue_len)。"""
        return conv_weight_codes(structure, q)

    def neuron_delta_t(self, structure: ConvQueueStructure) -> jax.Array:
        """逐神經元的 Δt,(n_neurons, max_queue_len)。"""
        return tile_channels(structure.delta_t, self.oc)

    def neuron_n_real(self, structure: ConvQueueStructure) -> jax.Array:
        """逐神經元的真 tap 數,(n_neurons,)。"""
        return tile_channels(structure.n_real_events, self.oc)

    def scan_steps(self, structure: ConvQueueStructure) -> int:
        """浮點掃描的步數:ceil(max_queue_len / chunk_size) + max_extra_steps。"""
        return base_scan_steps(self.max_queue_len, self.chunk_size) + self.max_extra_steps

    def with_chunk_size(self, chunk_size: int) -> "ConvLayer":
        """換 chunk_size,max_extra_steps 設成一定夠的值(總步數 = max_queue_len)。

        訓練時 max_extra_steps 是照舊 chunk_size 的需求縮小過的,換 chunk_size 後可能不夠;
        max_queue_len 步一定夠,因為每一步至少處理一筆事件。
        """
        return replace(self, chunk_size=chunk_size,
                       max_extra_steps=safe_extra_steps(self.max_queue_len, chunk_size))
