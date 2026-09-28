"""層物件:把「建佇列 → 跑一層 → 吐標準事件流」這條鏈收進一個自足的東西。

跟前三層(神經元模擬器 / 佇列建構器 / 標準事件流)的關係:

- 神經元行為(`float.scan.run_layer_forward`)、佇列建構(`connectivity/`)、
  標準事件流(`layer_chain.EventStream` + `extract_output_events*`)都不動,
  這個檔案只是把它們按「一種 layer 型別」串起來,對外只露兩個約定:
  **讀一條 `EventStream`、吐一份 `LayerOutput`(輸出流、結果、診斷、軌跡)**。
  數值段跟掃描交給 backend(`salt_core.backend`、`salt_core.quant.backend`)。
- 壓縮版的內部記帳(`local_to_global_j` 查表)留在 `ConvLayer.forward` 裡自己
  清掉,呼叫端看不到。conv 的「扁平神經元編號 → (x,y,c)」也在 `ConvLayer`
  內用自己的 `h_in`/`w_in` 還原,不外洩成呼叫端的一步。

**靜態 vs 會被微分的切分**(JAX 函數式):

- 層物件本身 = frozen dataclass,只有 Python 純量欄位(幾何 / 門檻 / chunk /
  容量 / init 尺度),可雜湊 → 能當 `jax.jit` 靜態參數 / 被閉包捕捉。
  **不含任何 JAX 陣列**,不是 pytree,不會被 trace。
- 權重 = 一個 pytree(一層一份陣列),由 `run_network` 的 `weights` 參數獨立
  傳遞,是唯一餵給 `jax.grad` 的東西。
- 容量旋鈕動態放大縮小 = `with_capacity` 換出一個新的靜態層物件(觸發一次
  重編譯),公式見 `salt_core.capacity`。
"""
from dataclasses import dataclass, replace
from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp

from salt_core.float.backend import FLOAT
from salt_core.capacity import Capacity, LayerDiag
from salt_core.connectivity.conv import (ConvQueueStructure, build_conv_structure,
                                          conv_float_values, conv_weight_codes, tile_channels,
                                          unravel_conv_source)
from salt_core.connectivity.fc import (FCQueueStructure, build_fc_structure, fc_float_values,
                                        fc_weight_codes)
from salt_core.float.affine import AffineMap, base_scan_steps, safe_extra_steps
from salt_core.stream import (EventStream, extract_output_events,
                                    extract_output_events_compressed)
from salt_core.trace import (LayerForwardTrace, resolve_ms_compressed,
                                resolve_ms_dense)


def uniform_init(key: jax.Array, shape: tuple, fan_in: int, init_k: float) -> jax.Array:
    """單一權重張量的 uniform 初始化,`limit = init_k / sqrt(fan_in)`。這是通用的
    權重初始化 primitive(標準 U(-1/sqrt(fan_in), 1/sqrt(fan_in)) 尺度,乘上
    可調的 init_k),層的 `init_weight` 跟找 k 的掃描都用它——同一個 key 只換
    init_k,firing rate 的變化才只來自 init_k 本身。"""
    limit = init_k / jnp.sqrt(float(fan_in))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)


class Layer(Protocol):
    """一個 layer 的對外約定(純文件用途,`run_network` 跟 backend 靠 duck typing)。
    `run_network` 跟 backend 只需要底下這幾樣,不管是 conv 還是 FC。"""
    name: str
    input_shape: tuple    # 吃空間輸入時是 (channel, 高, 寬),吃攤平輸入時是 (n,)
    output_shape: tuple   # 同上,這層輸出的形狀
    capacity: Capacity | None  # 容量旋鈕;沒有容量的層是 None。有容量的層另外提供 with_capacity
    v_th: float           # 以下三個給浮點 backend 的掃描用
    alpha: float
    chunk_size: int

    def init_weight(self, key: jax.Array) -> jax.Array:
        """這一層的權重張量(形狀 / fan_in / init_k 都是層自己的知識)。"""
        ...

    def unflatten_neurons(self, values):
        """逐神經元的值還原成 (channel, h, w, ...),其餘軸原樣保留。"""
        ...

    def broadcast_channels(self, values):
        """逐 channel 的值展開成逐神經元 (n_neurons, ...)。"""
        ...

    def forward(self, params, in_stream: EventStream, *, backend=FLOAT,
                trace: bool = False) -> "LayerOutput":
        """讀一條標準事件流 + 這層參數(backend 決定是浮點權重還是量化參數),
        吐 LayerOutput。trace=True 時多帶逐步軌跡。"""
        ...

    # 以下給 backend 用:佇列的數值段跟掃描設定。structure 是這層自己的佇列結構。
    def float_values(self, structure, w: jax.Array, event_gain: jax.Array | None) -> AffineMap:
        """浮點數值段,a、b 形狀 (n_neurons, 佇列長度)。"""
        ...

    def gather_weight_codes(self, structure, q: jax.Array) -> jax.Array:
        """整數數值段:每欄的整數權重碼,int32,(n_neurons, 佇列長度),非真事件是 0。"""
        ...

    def neuron_delta_t(self, structure) -> jax.Array:
        """逐神經元的 Δt,(n_neurons, 佇列長度)。"""
        ...

    def neuron_n_real(self, structure) -> jax.Array:
        """逐神經元的真事件數,(n_neurons,)。"""
        ...

    def scan_steps(self, structure) -> int:
        """浮點掃描的步數上限。"""
        ...

    def with_chunk_size(self, chunk_size: int) -> "Layer":
        """換 chunk_size,其他跟著要改的欄位(例如 max_extra_steps)由層自己改對。"""
        ...


class LayerOutput(NamedTuple):
    """一層 forward 的輸出。"""
    stream: EventStream                  # 給下一層的輸出事件流
    result: NamedTuple                   # 浮點是 LayerForwardResult,整數是 QuantLayerResult
    diag: LayerDiag
    trace: LayerForwardTrace | None      # trace=True 才有


def _layer_diag(spike_mask: jax.Array, n_neurons: int, n_real_in: jax.Array,
                needed: dict) -> LayerDiag:
    """一層的 LayerDiag。firing_rate = spike 數 / (n_neurons * max(輸入真事件數, 1))。"""
    spike_count = jnp.sum(spike_mask)
    return LayerDiag(spike_count=spike_count,
                     firing_rate=spike_count / (n_neurons * jnp.maximum(n_real_in, 1)),
                     needed=needed)


def _check_leading_axis(values, expected: int, what: str) -> None:
    """unflatten_neurons / broadcast_channels 的入口檢查。"""
    if values.ndim == 0 or values.shape[0] != expected:
        raise ValueError(f"第 0 軸長度要等於 {what}={expected},拿到的形狀是 {values.shape}")


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
        out_stream = extract_output_events_compressed(
            scan.result.spike_mask, scan.result.spike_event_idx, scan.spike_gain,
            in_stream.event_times, local_to_global_j, max_total_spikes=self.max_out_spikes)
        diag = _layer_diag(scan.result.spike_mask, self.n_neurons, in_stream.n_real_events,
                           needed={"max_queue_len": jnp.max(structure.n_real_events),
                                   "max_out_spikes": out_stream.n_real_events,
                                   "max_extra_steps": scan.extra_steps_needed})
        layer_trace = None
        if trace:
            event_ms = resolve_ms_compressed(scan.pointer_steps, local_to_global_j,
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
        out_stream = extract_output_events(
            scan.result.spike_mask, scan.result.spike_event_idx, scan.spike_gain,
            in_stream.event_times, max_total_spikes=self.max_out_spikes)
        diag = _layer_diag(scan.result.spike_mask, self.n_neurons, in_stream.n_real_events,
                           needed={"max_out_spikes": out_stream.n_real_events,
                                   "max_extra_steps": scan.extra_steps_needed})
        layer_trace = None
        if trace:
            event_ms = resolve_ms_dense(scan.pointer_steps, self.neuron_n_real(structure),
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
