"""容量:層的容量旋鈕、每層 forward 的診斷、跨 batch 合併、放不放得下、放大縮小公式。"""
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from salt_core.float.affine import safe_extra_steps


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


def _grow(observed: int, factor: float) -> int:
    """放大公式:觀察值上浮 factor 倍留餘裕。"""
    return int(math.ceil(int(observed) * factor))


@dataclass(frozen=True)
class GrowthPolicy:
    """一層容量的放大縮小公式。倍率跟門檻的理由見 docs/規格書.md「掃描步數旋鈕 max_extra_steps」。

    放大、縮小共用同一個倍率:需求沒變時兩邊算出的目標值相等,不會來回震盪。
    縮小門檻:候選值要掉到現值乘這個比例以下才縮,值得付一次重編譯。
    """
    max_queue_len_grow_factor: float = 1.5
    out_grow_factor: float = 1.5
    max_extra_steps_grow_factor: float = 1.5
    out_shrink_threshold: float = 0.5
    max_extra_steps_shrink_threshold: float = 0.5

    def _grow_factor(self, knob: str) -> float:
        return {"max_queue_len": self.max_queue_len_grow_factor, "max_out_spikes": self.out_grow_factor,
                "max_extra_steps": self.max_extra_steps_grow_factor}[knob]

    def _shrink_threshold(self, knob: str) -> float | None:
        return {"max_out_spikes": self.out_shrink_threshold,
                "max_extra_steps": self.max_extra_steps_shrink_threshold}.get(knob)

    def grown(self, capacity: Capacity, needed: dict, chunk_size: int) -> Capacity:
        """needed 超過容量的旋鈕放大到 ceil(needed * 倍率),其他不變。

        needed: 旋鈕名 -> 需求量(純量)。chunk_size: 這層的 chunk_size。
        max_queue_len 放大時 max_extra_steps 直接設成一定夠的值(總步數 = 新的 max_queue_len):
        這批的步數需求是在裝不下的佇列上算的,不可信;每一步至少處理一筆事件。
        """
        new = {knob: _grow(needed[knob], self._grow_factor(knob))
                     if int(needed[knob]) > value else value
               for knob, value in capacity.items()}
        if ("max_queue_len" in new and "max_extra_steps" in new
                and new["max_queue_len"] != capacity["max_queue_len"]):
            new["max_extra_steps"] = safe_extra_steps(new["max_queue_len"], chunk_size)
        return Capacity(**new)

    def shrunk(self, capacity: Capacity, observed: dict) -> Capacity:
        """用一整個 epoch 的最大需求決定要不要縮。

        observed: 旋鈕名 -> 這個 epoch 所有 batch 的最大需求。
        候選值 ceil(observed * 倍率) 掉到現值 * 縮小門檻以下才縮;沒有縮小門檻的旋鈕(max_queue_len)不縮。
        候選值至少是 1:整層一個 epoch 都沒 fire 時觀察值是 0,容量 0 會讓下一層拿到
        長度 0 的輸入流,建佇列時直接出錯。
        """
        new = dict(capacity)
        for knob, value in capacity.items():
            threshold = self._shrink_threshold(knob)
            if threshold is None:
                continue
            candidate = max(_grow(observed[knob], self._grow_factor(knob)), 1)
            if candidate < value * threshold:
                new[knob] = candidate
        return Capacity(**new)


def _replace_capacity(layers: list, new_capacities: list) -> list:
    """容量有變的層換成新容量;全部沒變時回傳傳進來的同一個 list 物件(不觸發重編譯)。"""
    replaced = [layer if capacity is None or capacity == layer.capacity
                else layer.with_capacity(capacity)
                for layer, capacity in zip(layers, new_capacities)]
    if all(new is old for new, old in zip(replaced, layers)):
        return layers
    return replaced


def grown_to_fit(layers: list, policies: dict, diags: list) -> list:
    """放不下的層換成放大過的版本。

    policies: 層名 -> GrowthPolicy,有容量的層都要有。
    diags: 對齊 layers 的 LayerDiag,已經合併成一份(needed 是純量)。
    全部放得下時回傳傳進來的同一個 list 物件。
    """
    return _replace_capacity(layers, [
        None if layer.capacity is None
        else policies[layer.name].grown(layer.capacity, diag.needed, layer.chunk_size)
        for layer, diag in zip(layers, diags)])


def grown_to_fit_batch(layers: list, policies: dict, batch_diags: list) -> list:
    """同 grown_to_fit,batch_diags 是逐筆診斷(每個值 shape (B,)),每層看最需要容量的那一筆。"""
    return grown_to_fit(layers, policies, [reduce_over_batch(diag) for diag in batch_diags])


def shrunk_to_observed(layers: list, policies: dict, observed: dict) -> list:
    """用一整個 epoch 的最大需求縮小容量。

    observed: 層名 -> 旋鈕名 -> 最大需求,有容量的層都要有。
    都沒縮時回傳傳進來的同一個 list 物件。
    """
    return _replace_capacity(layers, [
        None if layer.capacity is None
        else policies[layer.name].shrunk(layer.capacity, observed[layer.name])
        for layer in layers])
