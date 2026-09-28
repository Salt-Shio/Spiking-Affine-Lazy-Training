"""輸出解碼器:最後一層的 LayerForwardResult -> 跟 label 比對的分數。

三種標準編碼:膜電位回歸讀 v_final,頻率、群體讀 s_value(定義見 float/scan.py)。
validate 檢查最後一層的門檻跟編碼配不配。要別的編碼可以寫一個符合 Decoder 的新物件,
或在 loss 裡直接讀 LayerForwardResult。
解碼器是 frozen dataclass,沒有可學參數,可以被 jax.jit 閉包捕捉。
"""
from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp

from salt_core.float.scan import LayerForwardResult


class Decoder(Protocol):
    """解碼器的約定。只當文件用,實際靠 duck typing。"""

    def decode(self, result: LayerForwardResult) -> tuple[jax.Array, dict]:
        """單筆樣本最後一層的結果 -> (scores, metrics)。批次由呼叫端 vmap。

        scores: (類別數,),loss 跟 argmax 都用它。metrics: 這種編碼的監看純量,可以是空 dict。
        """
        ...

    def validate(self, last_layer) -> None:
        """最後一層的門檻跟這個編碼不配時 raise ValueError。只讀層的靜態欄位。"""
        ...


@dataclass(frozen=True)
class MembraneRegressionDecoder:
    """膜電位回歸:最後一層消化完整條佇列後的膜電位 v_final 當分數。

    最後一層要近乎不 fire(v_th >= min_out_v_th),v_final 才是所有加權事件的積分。
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
    """頻率編碼:每顆輸出神經元的 s_value 沿時間加總當分數。

    硬 spike 次數只放 metrics 當監看值,不進梯度。最後一層要照常 fire(v_th <= max_out_v_th)。
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
    """群體編碼:輸出神經元依序分成 n_classes 組、每組 group_size 顆,一組的 s_value 總和是
    那個類別的分數(神經元 0 ~ group_size-1 是類別 0,依此類推)。

    最後一層要照常 fire,n_out 要等於 n_classes * group_size。
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
