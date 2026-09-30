"""網路容器:原始事件型別 RawEvents、一列層加上輸入網格的 Network、唯一的 run_network。

Network 本身是靜態的(frozen dataclass,不含 JAX 陣列),可以當 jax.jit 的閉包;
權重另外傳,是 jax.grad 的對象。
"""
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from salt_core.backend import Backend, LayerParams, LayerResult
from salt_core.capacity import LayerDiag
from salt_core.float.backend import FLOAT
from salt_core.layers.base import Layer, LayerOutput
from salt_core.stream import EventStream
from salt_core.trace import LayerForwardTrace

_MAX_EVENT_TIME = 2 ** 31  # 整數掃描把 Δt 轉成 int32 當查表 index


class RawEvents(NamedTuple):
    """資料端原生的事件,可以直接 jit、vmap。單筆時各欄位 (n_events,),n_real_events 是純量;
    一批時多一個 batch 軸。前 n_real_events 筆是真事件,其餘是 pad。"""
    event_times: jax.Array   # 整數毫秒,真事件部分不遞減
    x: jax.Array
    y: jax.Array
    c: jax.Array
    n_real_events: jax.Array

    @staticmethod
    def checked(event_times: jax.Array | np.ndarray, x: jax.Array | np.ndarray,
                y: jax.Array | np.ndarray, c: jax.Array | np.ndarray,
                n_real_events: jax.Array | np.ndarray | int) -> "RawEvents":
        """在 host 端檢查真事件的時間,通過才包成 RawEvents。

        jit 裡沒辦法 raise,整數版 forward 的 Δt 檢查會被跳過,所以在事件進來時檢查一次。
        真事件時間不是整數、不在 [0, 2^31)、或遞減時 raise ValueError。
        """
        times = np.asarray(event_times)
        n_real = np.asarray(n_real_events)
        is_real = np.arange(times.shape[-1]) < n_real[..., None]
        real_times = times[is_real]
        if not np.all(real_times == np.floor(real_times)):
            raise ValueError("事件時間必須是整數毫秒")
        if np.any(real_times < 0) or np.any(real_times >= _MAX_EVENT_TIME):
            raise ValueError(f"事件時間必須在 [0, 2^31) 裡,拿到的範圍是 "
                             f"[{real_times.min()}, {real_times.max()}]")
        both_real = is_real[..., 1:]
        if np.any(np.diff(times, axis=-1)[both_real] < 0):
            raise ValueError("真事件時間必須不遞減")
        return RawEvents(event_times=event_times, x=x, y=y, c=c, n_real_events=n_real_events)


class NetworkOutput(NamedTuple):
    """run_network 的輸出。results、diags、traces 每層一個,對齊 layers。"""
    results: tuple[LayerResult, ...]
    diags: tuple[LayerDiag, ...]
    traces: tuple[LayerForwardTrace, ...] | None  # trace=True 才有,已經 stop_gradient
    fits: jax.Array       # bool:每個有容量的層都放得下;False 時結果可能被截斷,不可信

    @property
    def last(self) -> LayerResult:
        """最後一層的結果,給解碼器。"""
        return self.results[-1]


def _check_connected(out_shape: tuple[int, ...], in_shape: tuple[int, ...], out_name: str,
                     in_name: str) -> None:
    """out_shape 接得上 in_shape:吃空間輸入時形狀要完全相同,吃攤平輸入時元素總數要相同。
    接不上時 raise ValueError。"""
    out_shape, in_shape = tuple(out_shape), tuple(in_shape)
    if len(in_shape) == 1:
        connected = math.prod(out_shape) == in_shape[0]
    else:
        connected = out_shape == in_shape
    if connected:
        return
    if len(out_shape) < len(in_shape):
        raise ValueError(f"{out_name} 的輸出 {out_shape} 沒有空間形狀,"
                         f"不能接吃空間輸入的 {in_name}(輸入 {in_shape})")
    raise ValueError(f"{out_name} 的輸出 {out_shape} 接不上 {in_name} 的輸入 {in_shape}")


def check_layer_connections(layers: Sequence[Layer]) -> None:
    """檢查相鄰兩層接得起來,規則見 _check_connected。接不上時 raise ValueError。"""
    for prev, nxt in zip(layers, layers[1:]):
        _check_connected(prev.output_shape, nxt.input_shape, prev.name, nxt.name)


def run_network(layers: Sequence[Layer], weights: Sequence[LayerParams],
                input_stream: EventStream, *, backend: Backend = FLOAT,
                trace: bool = False) -> NetworkOutput:
    """一列 layer 串起來跑:每層的輸出流餵給下一層。

    layers: 一列符合 Layer 約定的物件(靜態,jax.jit 下當閉包捕捉)。
    weights: 對齊 layers 的每層參數;浮點 backend 是權重,整數 backend 是
        QuantizedLayerParams。
    trace: True 時收每層逐步軌跡。
    回傳的 fits:每個有容量的層都放得下這筆輸入。
    相鄰層接不上時 raise ValueError。
    """
    check_layer_connections(layers)
    stream = input_stream
    outputs: list[LayerOutput] = []
    for layer, params in zip(layers, weights):
        output = layer.forward(params, stream, backend=backend, trace=trace)
        outputs.append(output)
        stream = output.stream
    traces = None
    if trace:
        traces = jax.tree_util.tree_map(jax.lax.stop_gradient,
                                        tuple(output.trace for output in outputs))
    fits = jnp.array(True)
    for layer, output in zip(layers, outputs):
        if layer.capacity is not None:
            fits = fits & layer.capacity.fits(output.diag)
    return NetworkOutput(results=tuple(output.result for output in outputs),
                         diags=tuple(output.diag for output in outputs), traces=traces,
                         fits=fits)


@dataclass(frozen=True)
class Network:
    """原始事件網格 input_shape = (C, H, W) 加上一列層。

    建構時檢查 input_shape 接得上第一層、相鄰層接得上,接不上時 raise ValueError。
    """
    input_shape: tuple[int, int, int]
    layers: tuple[Layer, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_shape", tuple(int(v) for v in self.input_shape))
        object.__setattr__(self, "layers", tuple(self.layers))
        if len(self.input_shape) != 3:
            raise ValueError(f"input_shape 要是 (C, H, W),拿到 {self.input_shape}")
        if not self.layers:
            raise ValueError("layers 不能是空的")
        _check_connected(self.input_shape, self.layers[0].input_shape, "輸入",
                         self.layers[0].name)
        check_layer_connections(self.layers)

    def init(self, key: jax.Array) -> tuple[jax.Array, ...]:
        """每層各自初始化的權重,對齊 layers。"""
        keys = jax.random.split(key, len(self.layers))
        return tuple(layer.init_weight(k) for layer, k in zip(self.layers, keys))

    def input_stream(self, raw: RawEvents) -> EventStream:
        """單筆原始事件包成第一層的輸入流:來源編號 = c*H*W + y*W + x,event_gain 全 1。"""
        _c, h, w = self.input_shape
        source_idx = raw.c * (h * w) + raw.y * w + raw.x
        return EventStream(
            event_times=raw.event_times,
            event_source_idx=jnp.asarray(source_idx, dtype=jnp.int32),
            event_gain=jnp.ones_like(jnp.asarray(raw.event_times, dtype=jnp.float32)),
            n_real_events=jnp.asarray(raw.n_real_events, dtype=jnp.int32))

    def apply(self, weights: Sequence[LayerParams], raw: RawEvents, *, backend: Backend = FLOAT,
              trace: bool = False) -> NetworkOutput:
        """單筆 forward,見 run_network。"""
        return run_network(self.layers, weights, self.input_stream(raw), backend=backend,
                           trace=trace)

    def apply_batched(self, weights: Sequence[LayerParams], raw_batch: RawEvents, *,
                      backend: Backend = FLOAT, trace: bool = False) -> NetworkOutput:
        """一批 forward:對 raw_batch 的 batch 軸 vmap,權重共用。輸出每個陣列多一個 batch 軸。"""
        return jax.vmap(lambda raw: self.apply(weights, raw, backend=backend, trace=trace))(
            raw_batch)

    def replace_layers(self, layers: Sequence[Layer]) -> "Network":
        """換一列層(例如容量放大後),input_shape 不變。"""
        return replace(self, layers=tuple(layers))
