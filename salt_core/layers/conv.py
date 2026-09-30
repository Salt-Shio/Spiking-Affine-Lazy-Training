"""ConvLayer:conv 層;ConvSpec、conv:給 Network.build 的描述。"""
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

from salt_core.backend import Backend, LayerParams
from salt_core.capacity import Capacity
from salt_core.connectivity.conv import (ConvQueueStructure, build_conv_structure,
                                          conv_float_values, conv_weight_codes, tile_channels,
                                          unravel_conv_source)
from salt_core.float.affine import AffineMap, base_scan_steps, safe_extra_steps
from salt_core.float.backend import FLOAT
from salt_core.layers.base import (ArrayT, LayerOutput, _check_leading_axis, _layer_diag,
                                   uniform_init)
from salt_core.stream import EventStream, extract_output_events_conv
from salt_core.trace import LayerForwardTrace, resolve_ms_conv


@dataclass(frozen=True)
class ConvLayer:
    """conv 層。每顆神經元的佇列只放落在它感受野裡的事件。

    輸入面幾何(ic、h_in、w_in)要等於上一層的輸出形狀,Network 建構時檢查;第一層對的是
    網路的輸入形狀。h_out、w_out 由 h_in、w_in、k、s、p 算:(h_in + 2p - k) // s + 1。
    h_out 或 w_out 小於 1 時 raise ValueError。
    """
    name: str
    # 輸入面幾何
    ic: int
    h_in: int
    w_in: int
    # 這層幾何
    oc: int
    k: int
    s: int
    p: int
    # 初始權重尺度,選值理由見 docs/math/初始權重尺度推導.md
    init_k: float
    # 神經元動力學
    tau: float = 16.0
    v_th: float = 1.0
    alpha: float = 2.0
    chunk_size: int = 1
    # 容量。訓練時出界會放大,預設值只影響開頭要重編譯幾次。
    # max_queue_len 是每個位置的佇列欄數,真事件之後要留 1 欄給 catch-up。
    max_queue_len: int = 128
    max_out_spikes: int = 8192
    # 掃描步數 = ceil(max_queue_len / chunk_size) + max_extra_steps,見 docs/math/掃描步數上界推導.md。
    # None 時設成一定夠的值(總步數 = max_queue_len)。
    max_extra_steps: int | None = None

    def __post_init__(self) -> None:
        if self.h_out < 1 or self.w_out < 1:
            raise ValueError(
                f"{self.name} 幾何退化:輸入 {self.h_in}x{self.w_in}、k={self.k} s={self.s} "
                f"p={self.p} -> 輸出 {self.h_out}x{self.w_out}")
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
    def input_shape(self) -> tuple[int, ...]:
        return (self.ic, self.h_in, self.w_in)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (self.oc, self.h_out, self.w_out)

    @property
    def fan_in(self) -> int:
        return self.ic * self.k * self.k

    @property
    def weight_shape(self) -> tuple[int, ...]:
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

    def unflatten_neurons(self, values: ArrayT) -> ArrayT:
        """逐神經元的值還原成 (oc, h_out, w_out, ...)。

        values: 第 0 軸長度 n_neurons,神經元編號 = c*h_out*w_out + y*w_out + x;
            其餘軸原樣保留。numpy、jax 陣列都可以,回傳同一種。
        第 0 軸長度不對時 raise ValueError。
        """
        _check_leading_axis(values, self.n_neurons, "n_neurons")
        return values.reshape(self.oc, self.h_out, self.w_out, *values.shape[1:])

    def broadcast_channels(self, values: ArrayT) -> ArrayT:
        """逐 channel 的值展開成逐神經元,同一個 channel 的 h_out*w_out 顆神經元同一個值。

        values: 第 0 軸長度 oc,其餘軸原樣保留。回傳第 0 軸長度 n_neurons,
            numpy、jax 陣列都可以,回傳同一種。
        第 0 軸長度不對時 raise ValueError。
        """
        _check_leading_axis(values, self.oc, "oc")
        return values.repeat(self.h_out * self.w_out, axis=0)

    def forward(self, params: LayerParams, in_stream: EventStream, *, backend: Backend = FLOAT,
                trace: bool = False) -> LayerOutput:
        """建佇列 -> backend 算數值段跟掃描 -> 輸出事件流、診斷。

        params: 浮點 backend 是權重 (oc, ic, k, k);整數 backend 是 QuantizedLayerParams。
        trace: True 時 LayerOutput.trace 帶逐步軌跡。
        """
        structure = self.build_structure(in_stream)
        scan = backend.scan(self, structure, params, in_stream.event_gain, trace=trace)
        local_to_global_j = tile_channels(structure.local_to_global_j, self.oc)
        out_stream = extract_output_events_conv(
            scan.result.spike_mask, scan.result.spike_event_idx, scan.spike_gain,
            in_stream.event_times, local_to_global_j, max_total_spikes=self.max_out_spikes)
        # 佇列需求多 1 欄給 catch-up,理由見 docs/問題紀錄.md「決策:conv 佇列需求算進 catch-up 欄」
        diag = _layer_diag(scan.result.spike_mask, self.n_neurons, in_stream.n_real_events,
                           needed={"max_queue_len": jnp.max(structure.n_real_events) + 1,
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


@dataclass(frozen=True)
class ConvSpec:
    """ConvLayer 少了輸入面幾何(ic、h_in、w_in)的描述,Network.build 接上前一層時補。

    options: ConvLayer 其餘欄位(init_k、tau、容量等)。
    """
    oc: int
    k: int
    s: int
    p: int
    name: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
    name_prefix: ClassVar[str] = "conv"

    def build(self, input_shape: tuple[int, ...], name: str) -> ConvLayer:
        """接在輸出形狀 input_shape = (C, H, W) 後面的 ConvLayer。
        input_shape 不是三維、或幾何退化時 raise ValueError;options 有 ConvLayer 不認得的欄位時
        raise TypeError。"""
        if len(input_shape) != 3:
            raise ValueError(f"{name}:conv 要空間輸入 (C, H, W),前一層的輸出是攤平的 {input_shape}")
        ic, h_in, w_in = input_shape
        return ConvLayer(name=name, ic=ic, h_in=h_in, w_in=w_in, oc=self.oc, k=self.k, s=self.s,
                         p=self.p, **self.options)


def conv(oc: int, k: int, s: int, p: int, *, name: str | None = None, **options: Any) -> ConvSpec:
    """ConvSpec 的簡寫,例如 conv(8, k=3, s=2, p=1, init_k=5.0)。"""
    return ConvSpec(oc=oc, k=k, s=s, p=p, name=name, options=options)
