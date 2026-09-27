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

from salt_core.chunk_scan import (LayerForwardResult, LayerForwardResultInt,
                                   run_layer_forward, run_layer_forward_int,
                                   run_layer_forward_int_traced, run_layer_forward_traced)
from salt_core.core import spike_step_upper_bound
from salt_core.connectivity.conv import (CompressedConvQueue, build_conv_queue_compressed,
                                          unravel_conv_source)
from salt_core.connectivity.fc import FCQueue, build_fc_queue
from salt_core.fixed_point import OverflowMode, RoundMode
from salt_core.layer_chain import (EventStream, extract_output_events,
                                    extract_output_events_compressed)
from salt_core.monitor import (LayerForwardTrace, resolve_ms_compressed,
                                resolve_ms_dense)
from salt_core.quantize import apply_decay_table_int


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
    - `min_steps_needed`:用當下權重算出的掃描步數上界(見 `core.py` 的
      `spike_step_upper_bound`,推導見 docs/math/掃描步數上界推導.md),對這層
      所有神經元取 `max`。**`max_steps` 出界偵測訊號**:超過目前
      `ConvLayer.max_steps` 就代表這個 batch 用當下權重算出來的上界已經不夠
      安全,要照 `L`/`max_out_spikes` 一樣的方式放大。
    """
    spike_count: jax.Array
    firing_rate: jax.Array
    max_real_queue: jax.Array
    n_out_spikes: jax.Array
    min_steps_needed: jax.Array


class Layer(Protocol):
    """一個 layer 的對外約定(純文件用途,`run_network` 靠 duck typing)。
    `run_network` 只需要底下這幾樣,不管是 conv 還是 FC。"""
    name: str
    input_shape: tuple    # 吃空間輸入時是 (channel, 高, 寬),吃攤平輸入時是 (n,)
    output_shape: tuple   # 同上,這層輸出的形狀

    def init_weight(self, key: jax.Array) -> jax.Array:
        """這一層的權重張量(形狀 / fan_in / init_k 都是層自己的知識)。"""
        ...

    def forward(self, w: jax.Array,
                in_stream: EventStream) -> tuple[EventStream, LayerForwardResult, LayerDiag]:
        """讀一條標準事件流 + 這層權重,吐 (標準輸出事件流, 原始 forward 結果,
        固定診斷)。原始結果留給最後一層的解碼器(step 5)用。"""
        ...

    def forward_traced(self, w: jax.Array,
                       in_stream: EventStream) -> tuple[EventStream, LayerForwardTrace]:
        """跟 `forward` 一樣跑一層,但吐 (輸出事件流, `LayerForwardTrace` 逐步軌跡)。
        給 `run_network_traced` 的週期性 debug probe 用,不進訓練熱路徑。"""
        ...

    def grown_to_fit(self, diag: LayerDiag) -> "Layer":
        """給定這層剛跑完的診斷(概念上是 host 端具體數值,不是 traced),
        沒出界回自己、出界回一個容量放大過的新層物件。"""
        ...

    def shrink_max_steps(self, observed: int) -> "Layer":
        """給一個成功跑完的 epoch 裡、所有真實 batch 觀察到的 `max_steps`
        需求最大值,決定要不要縮、縮多少,回一個新層物件;沒有可縮欄位的層
        (例如 FC)原樣傳回自己。跟 `grown_to_fit` 是獨立的路:只在
        `example/train_conv_compressed.py` 的 epoch 成功跑完之後呼叫,吃的是
        整個 epoch 累積的觀察值,不是單一 batch 的 `LayerDiag`。"""
        ...

    def shrink_max_out_spikes(self, observed: int) -> "Layer":
        """跟 `shrink_max_steps` 同一套規則,縮的旋鈕換成 `max_out_spikes`,
        `observed` 是這個 epoch 裡所有真實 batch 觀察到的 `LayerDiag.
        n_out_spikes` 最大值(真實觀察值,不是理論上界)。沒有可縮欄位的層
        (例如 FC)原樣傳回自己。"""
        ...


def _grow(observed: int, current: int, factor: float) -> int:
    """容量放大:放大到蓋過觀測值,再上浮 factor 倍留餘裕。對齊原
    train_conv_compressed.py 的 `int(math.ceil(max(observed, current) * factor))`。"""
    return int(math.ceil(max(int(observed), int(current)) * factor))


class QuantizedLayerParams(NamedTuple):
    """一層整數版 forward(`forward_quantized`)要的量化設定,推導見
    docs/math/膜電位量化推導.md。"""
    q: jax.Array                   # 整數權重碼(`quantize.quantize_to_int` 的 q),整數 dtype,
                                   # shape 同 `layer.weight_shape`
    decay_table_int: jax.Array     # `quantize.build_decay_table_int(f_a, layer.tau)`
    v_th_int: jax.Array | None     # `quantize.v_th_to_int` 的整數門檻,純量或 (n_neurons,);
                                   # None 代表這層不 fire(例如膜電位回歸的輸出層)
    scale: jax.Array               # 逐神經元的權重量化步長 s_c,純量或 (n_neurons,);
                                   # 只在讀出換回物理尺度時用(`dequantize_v_final`)
    f_a: int                       # 衰減碼小數位元
    f_V: int                       # 暫存器小數位元
    i_V: int                       # 暫存器整數位元(含符號位)
    overflow_mode: OverflowMode | str = OverflowMode.WRAP  # 暫存器溢位處理,見 fixed_point


class LayerDiagInt(NamedTuple):
    """整數版 forward 的容量診斷,全部是 bool 純量。容量(`L`/
    `max_out_spikes`)是照浮點模型的放電量校準的,量化之後放電量會變;超過
    容量時多出來的輸入事件或輸出 spike 會被默默截掉,這組結果就不能用。"""
    queue_truncated: jax.Array   # 壓縮佇列需要的長度超過 L(FC 沒有壓縮佇列,固定 False)
    output_truncated: jax.Array  # 這層真正吐出的 spike 數超過輸出容量


def _run_quantized_scan(queue: CompressedConvQueue | FCQueue, params: QuantizedLayerParams,
                        round_mode: RoundMode | str, trace: bool):
    """Conv/FC 整數版 forward 共用的「查表 → 取權重碼 → 整數掃描」。回傳
    `(result, v_steps)`,`v_steps` 只有 `trace=True` 時是陣列,否則 `None`。"""
    a_int, is_identity = apply_decay_table_int(queue.delta_t, params.decay_table_int)
    # 佇列建構把權重碼帶成 float32(AffineMap.b)。q 是整數 dtype(入口檢查過),
    # |q| < 2^24 時 float32 表示是精確的,直接轉回 int32 不會改值。
    q_int = queue.maps.b.astype(jnp.int32)
    scan_args = (a_int, is_identity, q_int, params.v_th_int)
    scan_kwargs = dict(f_a=params.f_a, f_V=params.f_V, i_V=params.i_V, round_mode=round_mode,
                       overflow_mode=params.overflow_mode)
    if trace:
        return run_layer_forward_int_traced(*scan_args, **scan_kwargs)
    return run_layer_forward_int(*scan_args, **scan_kwargs), None


def dequantize_v_final(result: LayerForwardResultInt,
                       params: QuantizedLayerParams) -> LayerForwardResultInt:
    """把整數版結果的 `v_final` 從暫存器值換回物理尺度
    $V=\\tilde V\\cdot s_c = v_{int}\\cdot2^{-f_V}\\cdot s_c$(逐神經元),
    其他欄位不變。

    膜電位回歸的讀出要用換過的值:權重逐輸出 channel 量化時,每顆輸出
    神經元的 $s_c$ 不同,整數暫存器值直接比大小會選錯類別。整層共用一個
    $s_c$ 時乘同一個正數不改變 argmax,所以同一段讀出對兩種量化粒度都對。
    """
    v_physical = (result.v_final.astype(jnp.float32) * 2.0 ** -params.f_V
                  * jnp.asarray(params.scale, dtype=jnp.float32))
    return result._replace(v_final=v_physical)


def _check_weight_codes(q: jax.Array) -> None:
    """整數版 forward 的入口檢查:`q` 必須是整數 dtype。傳成乘回 scale 的
    浮點權重(量值大約 0.01)時,轉成整數碼會全部變成 0,forward 照樣跑完,
    不會有任何錯誤訊息。"""
    dtype = jnp.asarray(q).dtype
    if not jnp.issubdtype(dtype, jnp.integer):
        raise ValueError(
            f"q 必須是整數權重碼(quantize.quantize_to_int 的 q),拿到的 dtype 是 {dtype}")


def _constant_spike_gain(spike_mask: jax.Array) -> jax.Array:
    """整數版沒有 surrogate gradient,跨層的 event_gain 直接用常數 1。下一層
    gather 出整數權重碼之後乘上這個 1,值不變。"""
    return jnp.ones_like(spike_mask, dtype=jnp.float32)


@dataclass(frozen=True)
class ConvLayer:
    """一個壓縮版 conv 層。靜態欄位分五組:輸入面幾何 / 這層幾何 / init_k /
    神經元動力學 / 容量 + 放大倍率。

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
    # 初始權重尺度 —— 必填,不校準,委定值見 docs/問題紀錄.md §12(firing-rate
    # 目標帶準則廢棄,固定 init_k=5.0)。
    init_k: float
    # 神經元動力學(逐層)—— 有預設,是「起點」,config 要覆蓋就覆蓋。
    tau: float = 16.0
    v_th: float = 1.0
    alpha: float = 2.0
    chunk_size: int = 1
    # 容量 + 放大倍率 —— 有預設。L / max_out_spikes 的值不重要(出界會
    # 自己長大),預設只求「不要太小、少幾次開頭重編譯」。
    L: int = 128
    max_out_spikes: int = 8192
    L_grow_factor: float = 1.5
    out_grow_factor: float = 1.5
    # max_out_spikes 縮小門檻(比照 max_steps_shrink_threshold):候選值要掉到
    # 現在 max_out_spikes 的這個比例以下才值得縮。長大/縮小共用同一個
    # out_grow_factor 公式(ceil(觀察值 * factor)),真實需求沒變時兩次算出來的
    # 目標值相等,天然防震盪,道理跟 max_steps_grow_factor 一樣。跟 max_steps
    # 縮小的差別:這裡吃的是真實觀察值 n_out_spikes,不是任何理論上界(見
    # docs/問題紀錄.md 第十四節,為什麼 max_out_spikes 不能用 m*)。
    out_shrink_threshold: float = 0.5
    # 跟 L 脫鉤的掃描步數上界(見 docs/math/掃描步數上界推導.md)。`None`
    # (預設)代表「沒特別設起始猜測」,`__post_init__` 落到 `self.L`,對齊這個
    # 欄位存在之前的行為(safe fallback,永遠夠用)——這是給**沒有經過**
    # `example/train_conv_compressed.py` 動態放大迴圈的呼叫端(例如
    # 直接用 `build_network` 的分析腳本)用的安全預設,
    # 不會因為這個欄位的新增而默默截斷掃描、算出錯的結果。訓練腳本要用小
    # 起始值讓它自己長(跟 `L`/`max_out_spikes` 同一種「config 給起始猜測」
    # 的用法),config 就直接填這個欄位,不要靠這個 fallback。
    max_steps: int | None = None
    # 長大跟縮小共用同一個倍率:長大時補到 `ceil(觀察值 * factor)`(留餘裕,
    # 不是補精確值);縮小時候選值也用同一個公式算(`ceil(觀察值 * factor)`),
    # 保證「真實需求沒變 → 兩次算出來的目標值相等 → 不會縮」,不需要另外
    # 調參數搭配才能防震盪(見 docs/規格書.md「conv 層 max_steps」的推導)。
    max_steps_grow_factor: float = 1.5
    # 縮小門檻:候選值要掉到現在 max_steps 的這個比例以下才值得縮(付一次
    # 重編譯的代價換空間)。不影響防震盪(那是上面 factor 共用的效果),純粹是
    #「值不值得縮」的效率取捨。
    max_steps_shrink_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.max_steps is None:
            object.__setattr__(self, "max_steps", self.L)

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

    def init_weight(self, key: jax.Array) -> jax.Array:
        return uniform_init(key, self.weight_shape, self.fan_in, self.init_k)

    def _run_forward(self, w: jax.Array, in_stream: EventStream, *, trace: bool):
        """建壓縮佇列 + 跑一層 + 抽輸出流。`forward` / `forward_traced` 共用。
        `trace=False` 時 graph 跟舊 `forward` body 逐位元相同(`trace` 是 Python
        端靜態 bool,分支在 trace 期被消掉)。回傳 `(out_stream, result, cq,
        v_steps, pointer_steps)`;後兩個只有 `trace=True` 時是陣列,否則 `None`。"""
        # 扁平來源編號 -> (x,y,c),用這層自己的輸入面尺寸。第一層吃 ravel 過的
        # 原始事件,ravel↔unravel 對合法座標((0..w_in-1, 0..h_in-1, 0..ic-1),
        # pad 也是 (0,0,0))是嚴格逆運算,不改數值。
        x, y, c = unravel_conv_source(in_stream.event_source_idx, self.h_in, self.w_in)
        cq = build_conv_queue_compressed(
            in_stream.event_times, x, y, c, w, self.tau,
            self.s, self.p, self.h_out, self.w_out, self.L,
            event_gain=in_stream.event_gain, n_real_events=in_stream.n_real_events)
        if trace:
            result, v_steps, pointer_steps = run_layer_forward_traced(
                cq.maps, self.v_th, chunk_size=self.chunk_size, max_steps=self.max_steps,
                alpha=self.alpha, n_real_events=cq.n_real_events)
        else:
            result = run_layer_forward(
                cq.maps, self.v_th, chunk_size=self.chunk_size, max_steps=self.max_steps,
                alpha=self.alpha, n_real_events=cq.n_real_events)
            v_steps = pointer_steps = None
        out_stream = extract_output_events_compressed(
            result.spike_mask, result.spike_event_idx, result.s_spike,
            in_stream.event_times, cq.local_to_global_j,
            max_total_spikes=self.max_out_spikes)
        return out_stream, result, cq, v_steps, pointer_steps

    def forward(self, w: jax.Array,
                in_stream: EventStream) -> tuple[EventStream, LayerForwardResult, LayerDiag]:
        out_stream, result, cq, _, _ = self._run_forward(w, in_stream, trace=False)
        spike_count = jnp.sum(result.spike_mask)
        diag = LayerDiag(
            spike_count=spike_count,
            firing_rate=spike_count / (self.n_neurons * jnp.maximum(in_stream.n_real_events, 1)),
            max_real_queue=jnp.max(cq.n_real_events),
            n_out_spikes=out_stream.n_real_events,
            min_steps_needed=jnp.max(spike_step_upper_bound(
                cq.maps.b, self.v_th, self.chunk_size)))
        return out_stream, result, diag

    def forward_traced(self, w: jax.Array,
                       in_stream: EventStream) -> tuple[EventStream, LayerForwardTrace]:
        """跟 `forward` 一樣跑一層,但吐 `LayerForwardTrace`(逐步軌跡)取代
        `(LayerForwardResult, LayerDiag)`。給 `run_network_traced` 用。"""
        out_stream, result, cq, v_steps, pointer_steps = self._run_forward(
            w, in_stream, trace=True)
        event_ms = resolve_ms_compressed(
            pointer_steps, cq.local_to_global_j, cq.n_real_events, in_stream.event_times)
        trace = LayerForwardTrace(spike_mask=result.spike_mask,
                                   v_steps=v_steps, event_ms=event_ms)
        return out_stream, trace

    def _run_forward_quantized(self, params: QuantizedLayerParams, in_stream: EventStream, *,
                               round_mode: RoundMode | str, trace: bool):
        """`forward_quantized`/`forward_quantized_traced` 共用。回傳
        `(out_stream, result, v_steps, diag)`,`v_steps` 只有 `trace=True` 時
        是陣列,否則 `None`。"""
        _check_weight_codes(params.q)
        x, y, c = unravel_conv_source(in_stream.event_source_idx, self.h_in, self.w_in)
        cq = build_conv_queue_compressed(
            in_stream.event_times, x, y, c, params.q, self.tau,
            self.s, self.p, self.h_out, self.w_out, self.L,
            event_gain=in_stream.event_gain, n_real_events=in_stream.n_real_events)
        result, v_steps = _run_quantized_scan(cq, params, round_mode, trace)
        out_stream = extract_output_events_compressed(
            result.spike_mask, result.spike_event_idx, _constant_spike_gain(result.spike_mask),
            in_stream.event_times, cq.local_to_global_j,
            max_total_spikes=self.max_out_spikes)
        diag = LayerDiagInt(queue_truncated=jnp.max(cq.n_real_events) > self.L,
                            output_truncated=out_stream.n_real_events > self.max_out_spikes)
        return out_stream, result, v_steps, diag

    def forward_quantized(self, params: QuantizedLayerParams, in_stream: EventStream, *,
                          round_mode: RoundMode | str = RoundMode.ROUND
                          ) -> tuple[EventStream, LayerForwardResultInt, LayerDiagInt]:
        """整數版 forward(膜電位量化模擬,見 docs/math/膜電位量化推導.md):
        全程整數運算,沒有 $s_c$、沒有浮點除法,模擬 FPGA 逐事件更新暫存器。
        跟 `forward` 不共用程式碼,不影響訓練路徑。

        `params`:這層的 `QuantizedLayerParams`;`q` 不是整數 dtype 時 raise
        `ValueError`。`round_mode`:乘法後的捨入規則(`fixed_point.RoundMode`)。
        一步處理一筆事件(chunk_size 恆為 1),掃描長度就是壓縮佇列長度
        `self.L`,跟訓練用的 `self.chunk_size`/`self.max_steps` 無關。

        回傳 `(out_stream, result, diag)`:`result` 是 `LayerForwardResultInt`;
        `diag` 是 `LayerDiagInt`,任一旗標為 True 時這組結果不能用。
        """
        out_stream, result, _v_steps, diag = self._run_forward_quantized(
            params, in_stream, round_mode=round_mode, trace=False)
        return out_stream, result, diag

    def forward_quantized_traced(self, params: QuantizedLayerParams, in_stream: EventStream, *,
                                 round_mode: RoundMode | str = RoundMode.ROUND
                                 ) -> tuple[EventStream, LayerForwardResultInt, jax.Array,
                                            LayerDiagInt]:
        """跟 `forward_quantized` 一樣,額外回傳每步(寫回之後)的暫存器值
        `v_steps`,shape `(n_neurons, L)`。回傳 `(out_stream, result, v_steps, diag)`。

        溢位驗證要看逐步值,不能只看 `v_final`:神經元可能在某一步衝到最高點,
        之後因為衰減或後面的事件又掉下來,只看最終值會漏掉中間真正的峰值。
        不回傳 `LayerForwardTrace`:這條路沒有梯度,不需要 `event_ms` 時間戳
        反查;`result.overflowed` 已經逐步疊好。
        """
        return self._run_forward_quantized(params, in_stream, round_mode=round_mode, trace=True)

    def grown_to_fit(self, diag: LayerDiag) -> "ConvLayer":
        new_L = (_grow(diag.max_real_queue, self.L, self.L_grow_factor) # 這裡算完必定 >= self.L
                 if int(diag.max_real_queue) > self.L else self.L)
        new_out_spikes = (_grow(diag.n_out_spikes, self.max_out_spikes, self.out_grow_factor)
                   if int(diag.n_out_spikes) > self.max_out_spikes else self.max_out_spikes)
        # max_steps 的安全網:diag.min_steps_needed 是用「這個 batch 的實際權重」
        # 算出來、保證夠用的步數上界(見 docs/math/掃描步數上界推導.md),超過
        # 現在的 max_steps 就補到 ceil(min_steps_needed * max_steps_grow_factor)
        # ——留跟 L/max_out_spikes 同樣精神的餘裕,也讓長大跟縮小(見
        # shrink_max_steps)用同一個公式,兩者目標值才可能相等而不互相震盪。
        # 但 L 這次如果也跟著長大,這批的 min_steps_needed 是在「舊、不夠大」
        # 的佇列上算出來的,沒看到長大後才會出現的額外真實事件,不能信——退回
        # 全保守值(= 新 L),下一批或下次 epoch 重估再用長大後的真實佇列重新
        # 估出更緊的值。
        if new_L != self.L: # 目前 new_L 嚴格 > self.L
            new_max_steps = new_L
        elif int(diag.min_steps_needed) > self.max_steps:
            new_max_steps = _grow(diag.min_steps_needed, 0, self.max_steps_grow_factor)
        else:
            new_max_steps = self.max_steps
        if new_L == self.L and new_out_spikes == self.max_out_spikes and new_max_steps == self.max_steps:
            return self
        return replace(self, L=new_L, max_out_spikes=new_out_spikes, max_steps=new_max_steps)
        # L: 事件佇列
        # max_out_spikes: 作為輸入事件量的上界
        # max_steps: affine map 的 b 估算出來的上界，作為輸出事件上界

    def shrink_max_steps(self, observed: int) -> "ConvLayer":
        """`max_steps` 的選擇性縮小路徑,給 `train_conv_compressed.py` 在一個
        **成功跑完的 epoch** 之後呼叫(見 docs/規格書.md「conv 層 max_steps」)。
        跟 `grown_to_fit` 是兩條獨立的路:這裡只縮不長。

        `observed`:這個 epoch 裡,所有真實 batch 的 `LayerDiag.min_steps_needed`
        取過的最大值——不是探測批,是這個 epoch 真正跑過的訓練資料。

        候選值用跟 `grown_to_fit` **同一個公式**算(`ceil(observed *
        max_steps_grow_factor)`),不是 `observed` 本身:真實需求沒變時,兩次
        算出來的目標值會相等,天然不會縮,不用另外湊參數防震盪。候選值還要
        掉到現在 `max_steps` 的 `max_steps_shrink_threshold` 比例以下才真的
        縮(值不值得付一次重編譯的效率門檻,不影響防震盪)。都沒過就原樣傳回
        (同一個物件,不觸發重編譯)。"""
        candidate = _grow(observed, 0, self.max_steps_grow_factor)
        if candidate >= self.max_steps * self.max_steps_shrink_threshold:
            return self
        return replace(self, max_steps=candidate)

    def shrink_max_out_spikes(self, observed: int) -> "ConvLayer":
        """`max_out_spikes` 的選擇性縮小路徑,規則跟 `shrink_max_steps` 逐項對應
        (見 docs/規格書.md「conv 層 max_steps」)。跟 `max_steps` 唯一的差別:
        這裡吃的是真實觀察值(`LayerDiag.n_out_spikes`),不是任何理論上界——
        `max_out_spikes` 出界是「真實資料裝不下」的被動事實,不像 `max_steps`
        非得靠證明過的上界不可(見 docs/問題紀錄.md 第十四節)。

        `observed`:這個 epoch 裡,所有真實 batch 的 `LayerDiag.n_out_spikes`
        取過的最大值——不是探測批,是這個 epoch 真正跑過的訓練資料。

        候選值用跟 `grown_to_fit` 同一個公式算(`ceil(observed *
        out_grow_factor)`),真實需求沒變時兩次算出來的目標值會相等,天然不會
        縮,不用另外湊參數防震盪。候選值還要掉到現在 `max_out_spikes` 的
        `out_shrink_threshold` 比例以下才真的縮。都沒過就原樣傳回(同一個
        物件,不觸發重編譯)。"""
        candidate = _grow(observed, 0, self.out_grow_factor)
        if candidate >= self.max_out_spikes * self.out_shrink_threshold:
            return self
        return replace(self, max_out_spikes=candidate)


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

    def init_weight(self, key: jax.Array) -> jax.Array:
        return uniform_init(key, self.weight_shape, self.fan_in, self.init_k)

    def _run_forward(self, w: jax.Array, in_stream: EventStream, *, trace: bool):
        """建密集佇列 + 跑一層 + 抽輸出流。`forward` / `forward_traced` 共用。
        `trace=False` 時 graph 跟舊 `forward` body 逐位元相同。回傳
        `(out_stream, result, v_steps, pointer_steps)`;後兩個只有 `trace=True`
        時是陣列,否則 `None`。"""
        maps = build_fc_queue(
            in_stream.event_times, in_stream.event_source_idx, w, self.tau,
            event_gain=in_stream.event_gain, n_real_events=in_stream.n_real_events).maps
        # 積分預算 = 上一層宣告的輸出容量(輸入流固定長度),不是自己的欄位。
        scan_steps = -(-in_stream.event_times.shape[0] // self.chunk_size)  # ceil div
        if trace:
            result, v_steps, pointer_steps = run_layer_forward_traced(
                maps, self.v_th, chunk_size=self.chunk_size, max_steps=scan_steps,
                alpha=self.alpha, n_real_events=in_stream.n_real_events)
        else:
            result = run_layer_forward(
                maps, self.v_th, chunk_size=self.chunk_size, max_steps=scan_steps,
                alpha=self.alpha, n_real_events=in_stream.n_real_events)
            v_steps = pointer_steps = None
        out_stream = extract_output_events(
            result.spike_mask, result.spike_event_idx, result.s_spike,
            in_stream.event_times, max_total_spikes=self.n_out)
        return out_stream, result, maps, v_steps, pointer_steps

    def forward(self, w: jax.Array,
                in_stream: EventStream) -> tuple[EventStream, LayerForwardResult, LayerDiag]:
        out_stream, result, maps, _, _ = self._run_forward(w, in_stream, trace=False)
        spike_count = jnp.sum(result.spike_mask)
        diag = LayerDiag(
            spike_count=spike_count,
            firing_rate=spike_count / (self.n_neurons * jnp.maximum(in_stream.n_real_events, 1)),
            max_real_queue=jnp.zeros((), dtype=jnp.int32),
            n_out_spikes=out_stream.n_real_events,
            min_steps_needed=jnp.max(spike_step_upper_bound(
                maps.b, self.v_th, self.chunk_size)))
        return out_stream, result, diag

    def forward_traced(self, w: jax.Array,
                       in_stream: EventStream) -> tuple[EventStream, LayerForwardTrace]:
        """跟 `forward` 一樣跑一層,但吐 `LayerForwardTrace`(逐步軌跡)。密集
        佇列的 `pointer` 直接是全域事件 index,`resolve_ms_dense` 不必查表。"""
        out_stream, result, _maps, v_steps, pointer_steps = self._run_forward(
            w, in_stream, trace=True)
        n_real = jnp.broadcast_to(
            jnp.asarray(in_stream.n_real_events, jnp.int32), (self.n_out,))
        event_ms = resolve_ms_dense(pointer_steps, n_real, in_stream.event_times)
        trace = LayerForwardTrace(spike_mask=result.spike_mask,
                                   v_steps=v_steps, event_ms=event_ms)
        return out_stream, trace

    def _run_forward_quantized(self, params: QuantizedLayerParams, in_stream: EventStream, *,
                               round_mode: RoundMode | str, trace: bool):
        """跟 `ConvLayer._run_forward_quantized` 同一個組裝方式。回傳
        `(out_stream, result, v_steps, diag)`。"""
        _check_weight_codes(params.q)
        queue = build_fc_queue(
            in_stream.event_times, in_stream.event_source_idx, params.q, self.tau,
            event_gain=in_stream.event_gain, n_real_events=in_stream.n_real_events)
        result, v_steps = _run_quantized_scan(queue, params, round_mode, trace)
        out_stream = extract_output_events(
            result.spike_mask, result.spike_event_idx, _constant_spike_gain(result.spike_mask),
            in_stream.event_times, max_total_spikes=self.n_out)
        diag = LayerDiagInt(queue_truncated=jnp.zeros((), dtype=bool),
                            output_truncated=out_stream.n_real_events > self.n_out)
        return out_stream, result, v_steps, diag

    def forward_quantized(self, params: QuantizedLayerParams, in_stream: EventStream, *,
                          round_mode: RoundMode | str = RoundMode.ROUND
                          ) -> tuple[EventStream, LayerForwardResultInt, LayerDiagInt]:
        """同 `ConvLayer.forward_quantized`;掃描長度是輸入流長度
        (`in_stream.event_times.shape[0]`)。"""
        out_stream, result, _v_steps, diag = self._run_forward_quantized(
            params, in_stream, round_mode=round_mode, trace=False)
        return out_stream, result, diag

    def forward_quantized_traced(self, params: QuantizedLayerParams, in_stream: EventStream, *,
                                 round_mode: RoundMode | str = RoundMode.ROUND
                                 ) -> tuple[EventStream, LayerForwardResultInt, jax.Array,
                                            LayerDiagInt]:
        """同 `ConvLayer.forward_quantized_traced`。"""
        return self._run_forward_quantized(params, in_stream, round_mode=round_mode, trace=True)

    def grown_to_fit(self, diag: LayerDiag) -> "FCLayer":
        # 輸出層沒有自己的容量旋鈕:積分預算來自上一層的輸出容量(輸入流長度),
        # 上一層長大、重編譯時這層自動拿到更長的輸入流、更多 scan 步數。
        return self

    def shrink_max_steps(self, observed: int) -> "FCLayer":
        # 沒有獨立的 max_steps 欄位(積分步數是上一層輸出容量現算的,見
        # _run_forward 的 scan_steps),沒東西可以縮。
        return self

    def shrink_max_out_spikes(self, observed: int) -> "FCLayer":
        # 沒有獨立的 max_out_spikes 欄位(輸出容量固定是 n_out),沒東西可以縮。
        return self


def check_layer_connections(layers) -> None:
    """檢查相鄰兩層接得起來。

    下一層吃空間輸入(input_shape 有三維)時,上一層的 output_shape 要完全相同;
    吃攤平輸入(input_shape 只有一維)時,元素總數要相同。
    接不上時 raise ValueError,訊息寫出是哪兩層、各自的形狀。
    """
    for prev, nxt in zip(layers, layers[1:]):
        out_shape, in_shape = tuple(prev.output_shape), tuple(nxt.input_shape)
        if len(in_shape) == 1:
            connected = math.prod(out_shape) == in_shape[0]
        else:
            connected = out_shape == in_shape
        if connected:
            continue
        if len(out_shape) < len(in_shape):
            raise ValueError(f"{prev.name} 的輸出 {out_shape} 沒有空間形狀,"
                             f"不能接吃空間輸入的 {nxt.name}(輸入 {in_shape})")
        raise ValueError(f"{prev.name} 的輸出 {out_shape} 接不上 {nxt.name} 的輸入 {in_shape}")


def max_over_batch(diag: LayerDiag) -> LayerDiag:
    """一個 batch 的逐筆 LayerDiag(每個欄位 shape (B,))每個欄位取最大值。

    給容量判斷用:只要有一筆超過容量,這個 batch 就放不下。
    """
    return LayerDiag(*(jnp.max(field) for field in diag))


def grown_to_fit_batch(layers: list, batch_diags: list) -> list:
    """一列層放不放得下一個 batch;放不下的層換成放大過的版本。

    batch_diags: 對齊 layers 的逐筆 LayerDiag(每個欄位 shape (B,))。
    每層看 batch 裡最需要容量的那一筆,放大公式是各層的 grown_to_fit。
    全部放得下時回傳傳進來的同一個 list 物件。
    """
    grown = [layer.grown_to_fit(max_over_batch(diag))
             for layer, diag in zip(layers, batch_diags)]
    if all(new is old for new, old in zip(grown, layers)):
        return layers
    return grown


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
    check_layer_connections(layers)
    stream = input_stream
    result = None
    diags = []
    for layer, w in zip(layers, weights):
        stream, result, diag = layer.forward(w, stream)
        diags.append(diag)
    return result, diags


def run_network_traced(layers, input_stream: EventStream, weights):
    """`run_network` 的 forward-only 姊妹:逐層跑 `forward_traced`,每層收一份
    `LayerForwardTrace`(逐步軌跡),`stop_gradient` 後回傳 list。

    定位是週期性 debug probe(見 docs/監測規格.md §6):shape 比 `run_network`
    大一截、另編一個函式,不織進 `train_step`、不參與 grad。呼叫端拿它 dump
    `.npz` 看自訂 decoder / 動力學。
    """
    check_layer_connections(layers)
    stream = input_stream
    traces = []
    for layer, w in zip(layers, weights):
        stream, trace = layer.forward_traced(w, stream)
        traces.append(trace)
    return jax.tree_util.tree_map(jax.lax.stop_gradient, traces)


def run_network_quantized(layers, input_stream: EventStream,
                          quant_params: list[QuantizedLayerParams], *,
                          round_mode: RoundMode | str = RoundMode.ROUND):
    """`run_network` 的整數版姊妹(膜電位量化模擬):逐層跑
    `forward_quantized`。定位是事後分析,不進訓練路徑。

    `quant_params`:對齊 `layers` 的 `QuantizedLayerParams` list,每層各自的
    權重碼、查表、門檻、位元寬度,呼叫端照 docs/math/膜電位量化推導.md 的
    公式算好傳進來。`round_mode` 是整個網路共用的單一設計選擇。

    回傳 `(readout, diags)`:
    - `readout`:最後一層的 `LayerForwardResultInt`,`v_final` 已經用
      `dequantize_v_final` 換回物理尺度,直接餵解碼器。要看暫存器裡的
      整數值,用 `run_network_quantized_traced`。
    - `diags`:每層的 `LayerDiagInt`,任何一個旗標為 True 時這組結果不能用。
    """
    check_layer_connections(layers)
    stream = input_stream
    result = None
    diags = []
    for layer, params in zip(layers, quant_params):
        stream, result, diag = layer.forward_quantized(params, stream, round_mode=round_mode)
        diags.append(diag)
    return dequantize_v_final(result, quant_params[-1]), diags


def run_network_quantized_traced(layers, input_stream: EventStream,
                                 quant_params: list[QuantizedLayerParams], *,
                                 round_mode: RoundMode | str = RoundMode.ROUND):
    """`run_network_quantized` 的逐步軌跡版本:逐層跑 `forward_quantized_traced`,
    回傳對齊 `layers` 的 list,每層一份 `(result, v_steps, diag)`,
    `result.v_final`/`v_steps` 都是暫存器裡的整數值(沒有換回物理尺度)。
    為什麼溢位驗證要逐步值,見 `ConvLayer.forward_quantized_traced`。參數同
    `run_network_quantized`。"""
    check_layer_connections(layers)
    stream = input_stream
    traces = []
    for layer, params in zip(layers, quant_params):
        stream, result, v_steps, diag = layer.forward_quantized_traced(
            params, stream, round_mode=round_mode)
        traces.append((result, v_steps, diag))
    return traces
