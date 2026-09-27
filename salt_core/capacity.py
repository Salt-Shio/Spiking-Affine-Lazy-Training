"""容量:層的容量旋鈕、每層 forward 的診斷、跨 batch 合併、放不放得下。"""
from collections.abc import Mapping
from typing import NamedTuple

import jax
import jax.numpy as jnp


class LayerDiag(NamedTuple):
    """一層 forward 的診斷。

    spike_count: 這層 fire 的次數。
    firing_rate: spike_count / (n_neurons * max(輸入真事件數, 1))。
    needed: 旋鈕名 -> 這筆樣本需要的容量,key 跟層的 capacity 一樣;沒有容量的層是空 dict。
    """
    spike_count: jax.Array
    firing_rate: jax.Array
    needed: dict


def reduce_over_batch(diag: LayerDiag) -> LayerDiag:
    """一個 batch 的逐筆診斷(每個值 shape (B,))合成一份。

    needed 取最大:只要一筆樣本裝不下,這個 batch 就裝不下。spike_count、firing_rate 取平均。
    """
    return LayerDiag(spike_count=jnp.mean(diag.spike_count),
                     firing_rate=jnp.mean(diag.firing_rate),
                     needed={knob: jnp.max(value) for knob, value in diag.needed.items()})


class Capacity(Mapping):
    """旋鈕名 -> 容量值,不可變。"""

    def __init__(self, **knobs: int):
        self._knobs = {knob: int(value) for knob, value in knobs.items()}

    def __getitem__(self, knob: str) -> int:
        return self._knobs[knob]

    def __iter__(self):
        return iter(self._knobs)

    def __len__(self) -> int:
        return len(self._knobs)

    def __repr__(self) -> str:
        return f"Capacity({', '.join(f'{k}={v}' for k, v in self._knobs.items())})"

    def replace(self, **knobs: int) -> "Capacity":
        """換掉部分旋鈕的值。旋鈕名不在這份容量裡時 raise KeyError。"""
        unknown = set(knobs) - set(self._knobs)
        if unknown:
            raise KeyError(f"沒有這些旋鈕:{sorted(unknown)},有的是 {list(self._knobs)}")
        return Capacity(**{**self._knobs, **knobs})

    def fits(self, diag: LayerDiag) -> jax.Array:
        """每個旋鈕都放得下 diag.needed。diag 是一個 batch 的逐筆診斷時,回傳每筆一個值。"""
        return jnp.all(jnp.stack([diag.needed[knob] <= value
                                  for knob, value in self._knobs.items()]), axis=0)
