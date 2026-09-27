"""量每層逐 channel 的膜電位範圍,給 quant.convert 算 i_V。

逐樣本跑 run_network(..., trace=True)、挑哪些樣本,由呼叫端決定。
"""
import numpy as np


def v_range_per_channel(layers: list, traces: list) -> list[tuple[np.ndarray, np.ndarray]]:
    """一筆樣本每層逐 channel 的膜電位最大、最小值。

    traces: run_network(..., trace=True).traces,對齊 layers。
    回傳 list,每層一個 (v_max, v_min),形狀都是 (n_channels,)。
    任一層 chunk_size 不是 1 時 raise ValueError:軌跡只記每個 chunk 結束時的值,
    chunk 中間的峰值會漏掉。
    """
    not_one = [layer.name for layer in layers if layer.chunk_size != 1]
    if not_one:
        raise ValueError(f"量膜電位範圍要 chunk_size=1,這些層不是:{not_one}")
    ranges = []
    for layer, trace in zip(layers, traces):
        v_steps = np.asarray(trace.v_steps)
        v_max = layer.unflatten_neurons(v_steps.max(axis=1)).max(axis=(1, 2))
        v_min = layer.unflatten_neurons(v_steps.min(axis=1)).min(axis=(1, 2))
        ranges.append((v_max, v_min))
    return ranges


def merge_v_ranges(per_sample: list) -> list[tuple[np.ndarray, np.ndarray]]:
    """多筆樣本的 v_range_per_channel 合併:最大取最大、最小取最小。

    per_sample: 每筆樣本一份 v_range_per_channel 的回傳,至少一筆,否則 raise ValueError。
    """
    if not per_sample:
        raise ValueError("per_sample 至少要一筆")
    n_layers = len(per_sample[0])
    merged = []
    for i in range(n_layers):
        v_max = np.max([sample[i][0] for sample in per_sample], axis=0)
        v_min = np.min([sample[i][1] for sample in per_sample], axis=0)
        merged.append((v_max, v_min))
    return merged


def v_abs_max_per_channel(v_ranges: list) -> list[np.ndarray]:
    """每層逐 channel 的單邊最大量值 M = max(v_max, |v_min|)。"""
    return [np.maximum(v_max, np.abs(v_min)) for v_max, v_min in v_ranges]
