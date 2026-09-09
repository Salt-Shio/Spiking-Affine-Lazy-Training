"""輸出編碼的解碼器:把最後一層的 `LayerForwardResult` 讀成「拿去比對 label
的分數張量」。

跟前四層(神經元模擬器 / 佇列建構器 / 標準事件流 / 組裝器)的關係:

- 解碼器**不是網路的一部分**,是網路輸出到任務之間的接縫。`run_network` /
  層物件不因為換編碼而要改。
- 真正承重的介面是 `LayerForwardResult` 本身(`chunk_scan.py`):`v_final` /
  `s_value` / `spike_mask` 三個量、加上「pad 步對 `s_value` 貢獻 0」的保證,
  已經涵蓋任何 readout 會想要的東西。想要別的編碼,寫一個滿足 `Decoder`
  協定的新物件,或乾脆在自己的 loss_fn 裡直接讀 `LayerForwardResult`——
  兩條路都開著,解碼器不是唯一出口。
- 這裡只提供三個標準編碼(膜電位回歸 / 頻率 / 群體)+ 一個門檻配對檢查
  (`validate`),把三個坑封起來:
    1. 要 `s_value` 不是 `s_spike`(用錯會把「不套閘」設計在 readout 上破壞掉)。
    2. pad 步不能漏梯度(primitive 已處理,呼叫端不必知道)。
    3. 最後一層 `v_th` 設超大(純積分)還是正常(放電)要跟編碼配。

靜態 vs 會被微分:解碼器是 frozen dataclass,只有 Python 純量欄位,不含
JAX 陣列、沒有可學參數,可雜湊 → 能被 `jax.jit` 當閉包捕捉 / `jax.vmap`。
"""
from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp

from salt_core.chunk_scan import LayerForwardResult


class Decoder(Protocol):
    """一個解碼器的對外約定(純文件用途,呼叫端靠 duck typing)。"""

    def decode(self, result: LayerForwardResult) -> tuple[jax.Array, dict]:
        """讀最後一層的 forward 結果,吐 `(scores, metrics)`。單樣本(批次由
        呼叫端 vmap);`scores` 形狀 `(類別,)`,同時給 loss 和 argmax 用;
        `metrics` 是編碼特定的純量 dict,可為空。"""
        ...

    def validate(self, last_layer) -> None:
        """檢查最後一層的門檻設定跟這個編碼配不配,不配就 raise `ValueError`
        (擋掉「配錯 → 靜默算垃圾」)。只讀 `last_layer.v_th` 之類的靜態欄位。"""
        ...


@dataclass(frozen=True)
class MembraneRegressionDecoder:
    """膜電位回歸:直接拿最後一層消化完整條佇列後的膜電位當分數。要求最後一
    層近乎不放電(`v_th` 設超大),`v_final` 才是「全部加權事件的純仿射積分」。
    """
    min_out_v_th: float = 1e6

    def decode(self, result: LayerForwardResult) -> tuple[jax.Array, dict]:
        return result.v_final, {}

    def validate(self, last_layer) -> None:
        if last_layer.v_th < self.min_out_v_th:
            raise ValueError(
                f"膜電位回歸要求輸出層近乎不 fire(v_th >= {self.min_out_v_th:g}),"
                f"但最後一層 {getattr(last_layer, 'name', '?')} 的 "
                f"v_th={last_layer.v_th:g}")


@dataclass(frozen=True)
class RateDecoder:
    """頻率編碼:每顆輸出神經元自己的 spike 活動量當分數。分數用可微的
    `s_value` 沿時間軸加總,硬 spike 次數(`sum(spike_mask)`)只放 `metrics`
    當監看值(部署行為),不進梯度路徑。要求最後一層照常放電(`v_th` 正常)。
    """
    max_out_v_th: float = 1e3

    def decode(self, result: LayerForwardResult) -> tuple[jax.Array, dict]:
        soft = jnp.sum(result.s_value, axis=1)
        hard = jnp.sum(result.spike_mask, axis=1).astype(jnp.float32)
        metrics = {"hard_count_mean": jnp.mean(hard),
                   "hard_count_max": jnp.max(hard)}
        return soft, metrics

    def validate(self, last_layer) -> None:
        if last_layer.v_th > self.max_out_v_th:
            raise ValueError(
                f"頻率編碼要求輸出層照常放電(v_th <= {self.max_out_v_th:g}),"
                f"但最後一層 {getattr(last_layer, 'name', '?')} 的 "
                f"v_th={last_layer.v_th:g} 過大、形同純積分器")


@dataclass(frozen=True)
class PopulationDecoder:
    """群體編碼:輸出神經元**連續等分**成 `n_classes` 組、每組 `group_size`
    顆,一個類別的分數 = 該組神經元的 `s_value` 加總再相加(神經元 `[0,
    group_size)` → 類 0,依此類推)。要求最後一層照常放電。
    """
    n_classes: int
    group_size: int
    max_out_v_th: float = 1e3

    def decode(self, result: LayerForwardResult) -> tuple[jax.Array, dict]:
        per_neuron = jnp.sum(result.s_value, axis=1)  # (n_classes * group_size,)
        scores = per_neuron.reshape(self.n_classes, self.group_size).sum(axis=1)
        hard = jnp.sum(result.spike_mask, axis=1).astype(jnp.float32)
        metrics = {"hard_count_mean": jnp.mean(hard),
                   "hard_count_max": jnp.max(hard)}
        return scores, metrics

    def validate(self, last_layer) -> None:
        if last_layer.v_th > self.max_out_v_th:
            raise ValueError(
                f"群體編碼要求輸出層照常放電(v_th <= {self.max_out_v_th:g}),"
                f"但最後一層 {getattr(last_layer, 'name', '?')} 的 "
                f"v_th={last_layer.v_th:g} 過大、形同純積分器")
        n_out = getattr(last_layer, "n_out", None)
        expected = self.n_classes * self.group_size
        if n_out is not None and n_out != expected:
            raise ValueError(
                f"群體編碼:最後一層 n_out={n_out} != "
                f"n_classes*group_size={self.n_classes}*{self.group_size}={expected}")
