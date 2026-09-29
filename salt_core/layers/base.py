"""層的共用部分:Layer 約定、LayerOutput、權重初始化、診斷、形狀檢查。"""
from typing import NamedTuple, Protocol, TypeVar

import jax
import jax.numpy as jnp
import numpy as np

from salt_core.backend import Backend, LayerParams, LayerResult, ScanLayer
from salt_core.capacity import Capacity, LayerDiag
from salt_core.float.backend import FLOAT
from salt_core.stream import EventStream
from salt_core.trace import LayerForwardTrace

# numpy、jax 陣列都收,回傳同一種。
ArrayT = TypeVar("ArrayT", np.ndarray, jax.Array)


def uniform_init(key: jax.Array, shape: tuple[int, ...], fan_in: int, init_k: float) -> jax.Array:
    """權重張量的 uniform 初始化:U(-limit, limit),limit = init_k / sqrt(fan_in)。"""
    limit = init_k / jnp.sqrt(float(fan_in))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)


class Layer(ScanLayer, Protocol):
    """層的約定:run_network 用到這些,backend 用到的在 ScanLayer。只當文件用,實際靠 duck typing。"""
    name: str
    input_shape: tuple[int, ...]   # 吃空間輸入時是 (channel, 高, 寬),吃攤平輸入時是 (n,)
    output_shape: tuple[int, ...]  # 同上,這層輸出的形狀
    capacity: Capacity | None  # 容量旋鈕;沒有容量的層是 None。有容量的層另外提供 with_capacity
    tau: float            # 衰減時間常數,整數版換算衰減查表用

    def init_weight(self, key: jax.Array) -> jax.Array:
        """這一層的初始權重。"""
        ...

    def unflatten_neurons(self, values: ArrayT) -> ArrayT:
        """逐神經元的值還原成 (channel, h, w, ...),其餘軸原樣保留。"""
        ...

    def broadcast_channels(self, values: ArrayT) -> ArrayT:
        """逐 channel 的值展開成逐神經元 (n_neurons, ...)。"""
        ...

    def forward(self, params: LayerParams, in_stream: EventStream, *, backend: Backend = FLOAT,
                trace: bool = False) -> "LayerOutput":
        """輸入事件流 -> LayerOutput。params 的型別由 backend 決定(浮點權重或量化參數);
        trace=True 時多帶逐步軌跡。"""
        ...

    def with_chunk_size(self, chunk_size: int) -> "Layer":
        """換 chunk_size,其他跟著要改的欄位(例如 max_extra_steps)由層自己改對。"""
        ...


class LayerOutput(NamedTuple):
    """一層 forward 的輸出。"""
    stream: EventStream                  # 給下一層的輸出事件流
    result: LayerResult
    diag: LayerDiag
    trace: LayerForwardTrace | None      # trace=True 才有


def _layer_diag(spike_mask: jax.Array, n_neurons: int, n_real_in: jax.Array,
                needed: dict[str, jax.Array]) -> LayerDiag:
    """一層的 LayerDiag。firing_rate = spike 數 / (n_neurons * max(輸入真事件數, 1))。"""
    spike_count = jnp.sum(spike_mask)
    return LayerDiag(spike_count=spike_count,
                     firing_rate=spike_count / (n_neurons * jnp.maximum(n_real_in, 1)),
                     needed=needed)


def _check_leading_axis(values: np.ndarray | jax.Array, expected: int, what: str) -> None:
    """unflatten_neurons / broadcast_channels 的入口檢查。"""
    if values.ndim == 0 or values.shape[0] != expected:
        raise ValueError(f"第 0 軸長度要等於 {what}={expected},拿到的形狀是 {values.shape}")
