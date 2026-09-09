"""層物件:把「建佇列 → 跑一層 → 吐標準事件流」這條鏈收進一個自足的東西。

跟前三層(神經元模擬器 / 佇列建構器 / 標準事件流)的關係:

- 神經元行為(`chunk_scan.run_layer_forward`)、佇列建構(`connectivity/`)、
  標準事件流(`layer_chain.EventStream` + `extract_output_events*`)都不動,
  這個檔案只是把它們按「一種 layer 型別」串起來,對外只露兩個約定:
  **讀一條 `EventStream`、吐一條 `EventStream` + 一份 `LayerDiag`**。
- 壓縮版的內部記帳(`local_to_global_j` 查表)留在 `ConvLayer.forward` 裡自己
  清掉,呼叫端看不到。conv 的「扁平神經元編號 → (x,y,c)」也在 `ConvLayer`
  內用自己的 `h_in`/`w_in` 還原,不外洩成呼叫端的一步。

**靜態 vs 會被微分的切分**(JAX 函數式):

- 層物件本身 = frozen dataclass,只有 Python 純量欄位(幾何 / 門檻 / chunk /
  容量 / init 尺度 / 放大倍率),可雜湊 → 能當 `jax.jit` 靜態參數 / 被閉包捕捉。
  **不含任何 JAX 陣列**,不是 pytree,不會被 trace。
- 權重 = 一個 pytree(一層一份陣列),由 `run_network` 的 `weights` 參數獨立
  傳遞,是唯一餵給 `jax.grad` 的東西。
- 容量旋鈕(`L` / `max_out_spikes`)動態放大 = `dataclasses.replace` 出一個
  新的靜態層物件(觸發一次重編譯),見 `grown_to_fit`。
"""
import math
from dataclasses import dataclass, replace
from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp

from salt_core.chunk_scan import LayerForwardResult, run_layer_forward
from salt_core.connectivity.conv import (build_conv_queue_compressed,
                                          conv_layer_receptive_field_firing_rate,
                                          unravel_conv_source)
from salt_core.connectivity.fc import build_fc_queue
from salt_core.layer_chain import (EventStream, extract_output_events,
                                    extract_output_events_compressed)


def uniform_init(key: jax.Array, shape: tuple, fan_in: int, init_k: float) -> jax.Array:
    """單一權重張量的 uniform 初始化,`limit = init_k / sqrt(fan_in)`。這是通用的
    權重初始化 primitive(標準 U(-1/sqrt(fan_in), 1/sqrt(fan_in)) 尺度,乘上
    可調的 init_k),層的 `init_weight` 跟找 k 的掃描都用它——同一個 key 只換
    init_k,firing rate 的變化才只來自 init_k 本身。"""
    limit = init_k / jnp.sqrt(float(fan_in))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)


class LayerDiag(NamedTuple):
    """每一層跑完一次 forward 都吐同樣這一份固定紀錄(取代原本「事後往
    diagnostics dict 塞欄位」的做法)。全部是純量。

    - `spike_count` / `firing_rate`:輕量哨兵指標,`spike_count /
      (n_neurons * max(n_real_in, 1))`——**不是**找 k 用的感受野正規化版本
      (那個要多算一次感受野幾何,見 connectivity/conv.py)。
    - `max_real_queue`:這層每顆神經元「真正需要的壓縮佇列長度」的最大值
      (`build_conv_queue_compressed` 算出的 `n_real_events`,沒被截到 L)。
      **L 出界偵測訊號**。FC 層沒有壓縮佇列,固定 0。
    - `n_out_spikes`:這層真正吐幾筆 spike(完整 spike_mask 的 sum,沒被
      `max_out_spikes` 截斷)。**輸出上界出界偵測訊號**。
    """
    spike_count: jax.Array
    firing_rate: jax.Array
    max_real_queue: jax.Array
    n_out_spikes: jax.Array


class Layer(Protocol):
    """一個 layer 的對外約定(純文件用途,`run_network` 靠 duck typing)。
    `run_network` 只需要底下這幾樣,不管是 conv 還是 FC。"""
    name: str

    def init_weight(self, key: jax.Array) -> jax.Array:
        """這一層的權重張量(形狀 / fan_in / init_k 都是層自己的知識)。"""
        ...

    def forward(self, w: jax.Array,
                in_stream: EventStream) -> tuple[EventStream, LayerForwardResult, LayerDiag]:
        """讀一條標準事件流 + 這層權重,吐 (標準輸出事件流, 原始 forward 結果,
        固定診斷)。原始結果留給最後一層的解碼器(step 5)用。"""
        ...

    def grown_to_fit(self, diag: LayerDiag) -> "Layer":
        """給定這層剛跑完的診斷(概念上是 host 端具體數值,不是 traced),
        沒出界回自己、出界回一個容量放大過的新層物件。"""
        ...


def _grow(observed: int, current: int, factor: float) -> int:
    """容量放大:放大到蓋過觀測值,再上浮 factor 倍留餘裕。對齊原
    train_conv_compressed.py 的 `int(math.ceil(max(observed, current) * factor))`。"""
    return int(math.ceil(max(int(observed), int(current)) * factor))


@dataclass(frozen=True)
class ConvLayer:
    """一個壓縮版 conv 層。靜態欄位分四組:輸入面幾何 / 這層幾何 / 神經元動力學 /
    容量 + init + 放大倍率。

    輸入面幾何(`ic` / `h_in` / `w_in`)= 上一層的輸出:`ic` 要等於上一層的
    `oc`,`h_in`/`w_in` 要等於上一層的 `h_out`/`w_out`——組層 list 的時候
    Python 層級檢查一次(就是 PyTorch 要你自己對齊 channel 的那個檢查)。
    第一層的「上一層」是虛擬輸入網格 `(ic, h_in, w_in)`,由呼叫端把原始事件
    ravel 成扁平編號餵進來(見 `raw_events_to_stream`)。

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
    # 神經元動力學(逐層)—— 有預設,是「起點」,config 要覆蓋就覆蓋。
    tau: float = 16.0
    v_th: float = 1.0
    alpha: float = 2.0
    chunk_size: int = 1
    # 容量 + init + 放大倍率 —— 有預設。L / max_out_spikes 的值不重要(出界會
    # 自己長大),預設只求「不要太小、少幾次開頭重編譯」。
    L: int = 128
    max_out_spikes: int = 8192
    init_k: float | None = None   # None = 開訓前用 calibration_measure 現算(見 salt_core.calibrate)
    L_grow_factor: float = 1.5
    out_grow_factor: float = 1.5

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
    def fan_in(self) -> int:
        return self.ic * self.k * self.k

    @property
    def weight_shape(self) -> tuple:
        return (self.oc, self.ic, self.k, self.k)

    def init_weight(self, key: jax.Array) -> jax.Array:
        return uniform_init(key, self.weight_shape, self.fan_in, self.init_k)

    def forward(self, w: jax.Array,
                in_stream: EventStream) -> tuple[EventStream, LayerForwardResult, LayerDiag]:
        # 扁平來源編號 -> (x,y,c),用這層自己的輸入面尺寸。第一層吃 ravel 過的
        # 原始事件,ravel↔unravel 對合法座標((0..w_in-1, 0..h_in-1, 0..ic-1),
        # pad 也是 (0,0,0))是嚴格逆運算,不改數值。
        x, y, c = unravel_conv_source(in_stream.event_source_idx, self.h_in, self.w_in)
        cq = build_conv_queue_compressed(
            in_stream.event_times, x, y, c, w, self.tau,
            self.s, self.p, self.h_out, self.w_out, self.L,
            event_gain=in_stream.event_gain, n_real_events=in_stream.n_real_events)
        result = run_layer_forward(
            cq.maps, self.v_th, chunk_size=self.chunk_size, max_steps=self.L,
            alpha=self.alpha, n_real_events=cq.n_real_events)
        out_stream = extract_output_events_compressed(
            result.spike_mask, result.spike_event_idx, result.s_spike,
            in_stream.event_times, cq.local_to_global_j,
            max_total_spikes=self.max_out_spikes)
        spike_count = jnp.sum(result.spike_mask)
        diag = LayerDiag(
            spike_count=spike_count,
            firing_rate=spike_count / (self.n_neurons * jnp.maximum(in_stream.n_real_events, 1)),
            max_real_queue=jnp.max(cq.n_real_events),
            n_out_spikes=out_stream.n_real_events)
        return out_stream, result, diag

    def calibration_measure(self, calib_stream_batch: EventStream, chunk: int = 16):
        """回傳一個 `measure(weight) -> 純量`:對一批校準輸入流跑這層 forward,
        算感受野正規化的 firing rate(每顆神經元 spike 數 / 自己的感受野事件數,
        只對感受野事件數 > 0 的神經元取平均),再對整批樣本取平均。分批 vmap
        避免整批一次建構壓縮佇列 OOM。`salt_core.calibrate` 找 init_k 時用這個
        當 measure——準則(要不要 firing rate、目標帶多少)是呼叫端的決定。"""
        n = calib_stream_batch.event_times.shape[0]

        def measure(w: jax.Array) -> float:
            def one(s: EventStream):
                _out, result, _diag = self.forward(w, s)
                x, y, _c = unravel_conv_source(s.event_source_idx, self.h_in, self.w_in)
                return conv_layer_receptive_field_firing_rate(
                    result.spike_mask, x, y, self.s, self.p, self.h_out, self.w_out,
                    self.k, self.oc, s.n_real_events)

            total = 0.0
            for lo in range(0, n, chunk):
                sub = type(calib_stream_batch)(*(f[lo:min(lo + chunk, n)]
                                                  for f in calib_stream_batch))
                total += float(jnp.sum(jax.vmap(one)(sub)))
            return total / n

        return measure

    def grown_to_fit(self, diag: LayerDiag) -> "ConvLayer":
        new_L = (_grow(diag.max_real_queue, self.L, self.L_grow_factor)
                 if int(diag.max_real_queue) > self.L else self.L)
        new_out = (_grow(diag.n_out_spikes, self.max_out_spikes, self.out_grow_factor)
                   if int(diag.n_out_spikes) > self.max_out_spikes else self.max_out_spikes)
        if new_L == self.L and new_out == self.max_out_spikes:
            return self
        return replace(self, L=new_L, max_out_spikes=new_out)


@dataclass(frozen=True)
class FCLayer:
    """一個密集版 FC 層。目前只當輸出層用(`v_th` 設超大 → 純積分器,永遠不
    fire),但不是特殊型別——它就是一個普通 layer,怎麼把它的活動解讀成預測
    是解碼器的事(step 5)。

    要積分的「真實事件數」預算 = 上一層宣告的輸出容量(輸入流的固定長度
    `in_stream.event_times.shape[0]`),不是自己的欄位:上一層 `max_out_spikes`
    長大、重編譯時,這層拿到的輸入流變長,scan 步數自動跟著變多。實際
    `lax.scan` 步數 = ceil(輸入流長度 / chunk_size)。
    """
    name: str
    n_in: int
    n_out: int
    init_k: float               # 必填:FC 不 fire,沒有 firing-rate 準則可校準
    # 神經元動力學 —— 有預設。v_th 預設 1e9:FC 目前只當輸出層(純積分器),
    # 配錯有 decoder.validate 擋;真的當隱藏層用再明填正常門檻。
    tau: float = 16.0
    v_th: float = 1e9
    alpha: float = 2.0
    chunk_size: int = 1

    @property
    def n_neurons(self) -> int:
        return self.n_out

    @property
    def fan_in(self) -> int:
        return self.n_in

    @property
    def weight_shape(self) -> tuple:
        return (self.n_out, self.n_in)

    def init_weight(self, key: jax.Array) -> jax.Array:
        return uniform_init(key, self.weight_shape, self.fan_in, self.init_k)

    def forward(self, w: jax.Array,
                in_stream: EventStream) -> tuple[EventStream, LayerForwardResult, LayerDiag]:
        maps = build_fc_queue(
            in_stream.event_times, in_stream.event_source_idx, w, self.tau,
            event_gain=in_stream.event_gain, n_real_events=in_stream.n_real_events)
        # 積分預算 = 上一層宣告的輸出容量(輸入流固定長度),不是自己的欄位。
        scan_steps = -(-in_stream.event_times.shape[0] // self.chunk_size)  # ceil div
        result = run_layer_forward(
            maps, self.v_th, chunk_size=self.chunk_size, max_steps=scan_steps,
            alpha=self.alpha, n_real_events=in_stream.n_real_events)
        out_stream = extract_output_events(
            result.spike_mask, result.spike_event_idx, result.s_spike,
            in_stream.event_times, max_total_spikes=self.n_out)
        spike_count = jnp.sum(result.spike_mask)
        diag = LayerDiag(
            spike_count=spike_count,
            firing_rate=spike_count / (self.n_neurons * jnp.maximum(in_stream.n_real_events, 1)),
            max_real_queue=jnp.zeros((), dtype=jnp.int32),
            n_out_spikes=out_stream.n_real_events)
        return out_stream, result, diag

    def grown_to_fit(self, diag: LayerDiag) -> "FCLayer":
        # 輸出層沒有自己的容量旋鈕:積分預算來自上一層的輸出容量(輸入流長度),
        # 上一層長大、重編譯時這層自動拿到更長的輸入流、更多 scan 步數。
        return self


def raw_events_to_stream(event_times: jax.Array, x: jax.Array, y: jax.Array,
                          c: jax.Array, n_real_events: jax.Array,
                          h_in: int, w_in: int) -> EventStream:
    """把資料端原生的 (event_times, x, y, c, n_real_events) 包成一條標準事件流,
    餵給第一層。`event_source_idx` ravel 進虛擬輸入網格 `(C_in, h_in, w_in)`,
    第一層再用同一組 `h_in`/`w_in` 還原——來回是整數運算、成本可忽略,換到
    「每個 conv 層的 forward 長得一模一樣,沒有第一層特例」。`event_gain` 全 1
    (原始輸入沒有上游可微分增益)。"""
    source_idx = c * (h_in * w_in) + y * w_in + x
    return EventStream(
        event_times=event_times,
        event_source_idx=source_idx.astype(jnp.int32),
        event_gain=jnp.ones_like(jnp.asarray(event_times, dtype=jnp.float32)),
        n_real_events=jnp.asarray(n_real_events, dtype=jnp.int32))


def run_network(layers, input_stream: EventStream, weights):
    """一列 layer 串起來:把輸入流餵進第一層,每層的輸出流轉給下一層,收集
    每層的 `LayerDiag`。回傳 (最後一層的原始 forward 結果, 每層診斷 list)。

    `layers`:一列符合 `Layer` 約定的物件(靜態,`jax.jit` 下當閉包捕捉或
    static 參數)。`weights`:對齊 `layers` 的權重序列(pytree,`jax.grad` 的
    對象)。`input_stream`:`EventStream`。

    加 pooling / residual / 新 layer 型別 = 照 `Layer` 約定寫一個新層物件,
    這支函式一行不用改。
    """
    stream = input_stream
    result = None
    diags = []
    for layer, w in zip(layers, weights):
        stream, result, diag = layer.forward(w, stream)
        diags.append(diag)
    return result, diags
