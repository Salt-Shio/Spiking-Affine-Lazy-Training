"""層的共用部分:Layer 約定、LayerOutput、權重初始化、診斷、形狀檢查。"""
from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp

from salt_core.capacity import Capacity, LayerDiag
from salt_core.float.affine import AffineMap
from salt_core.float.backend import FLOAT
from salt_core.stream import EventStream
from salt_core.trace import LayerForwardTrace


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
