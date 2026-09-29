"""逐步軌跡:LayerForwardTrace、掃描步指標換成真實毫秒、軌跡摘要。

run_network(..., trace=True) 收集,不進訓練熱路徑;浮點、整數 backend 共用。
欄位的取捨見 docs/監測規格.md「LayerForwardTrace / run_network(trace=True)(逐步軌跡,已實作)」。
"""
from typing import NamedTuple, TypedDict

import jax
import jax.numpy as jnp
import numpy as np


class LayerForwardTrace(NamedTuple):
    """一層的逐步軌跡。形狀都是 (n, max_steps),n 是這層的神經元數。

    spike_mask: 同 forward 結果的 spike_mask。
    v_steps: 每步結束(套過 reset)的膜電位,最後一欄等於 v_final。整數 backend 是暫存器值。
    event_ms: 每步處理的第一筆事件的真實毫秒;空轉步是 nan。chunk_size=1 時每步剛好一筆事件。
    """
    spike_mask: jax.Array   # (n, max_steps) bool
    v_steps: jax.Array      # (n, max_steps) float
    event_ms: jax.Array     # (n, max_steps) float


def resolve_ms_fc(pointer: jax.Array, n_real_per_neuron: jax.Array,
                  event_times: jax.Array) -> jax.Array:
    """FC 層:每步處理的事件換成真實毫秒。

    pointer: (n, max_steps) 每步開始時的佇列欄位;FC 的欄位就是全域事件 index。
    n_real_per_neuron: (n,) 逐神經元的真事件數。
    event_times: (n_events,) 這層的輸入事件時間。
    回傳 (n, max_steps);空轉步(pointer >= 真事件數)是 nan。
    """
    n_events = event_times.shape[0]
    ms = jnp.asarray(event_times)[jnp.clip(pointer, 0, n_events - 1)]
    idle = pointer >= n_real_per_neuron[:, None]
    return jnp.where(idle, jnp.nan, ms)


def resolve_ms_conv(pointer: jax.Array, local_to_global_j: jax.Array,
                    n_real_per_neuron: jax.Array,
                    event_times: jax.Array) -> jax.Array:
    """conv 層:每步處理的事件換成真實毫秒。

    pointer: (n, max_steps) 每步開始時的佇列欄位,是這顆神經元自己佇列的局部欄。
    local_to_global_j: (n, max_queue_len) 局部欄 -> 全域事件 index,空欄是 n_events。
    其餘參數跟回傳同 resolve_ms_fc。
    """
    n_events = event_times.shape[0]
    n_cols = local_to_global_j.shape[1]
    global_idx = jnp.take_along_axis(
        local_to_global_j, jnp.clip(pointer, 0, n_cols - 1), axis=1)  # (n, max_steps)
    ms = jnp.asarray(event_times)[jnp.minimum(global_idx, n_events - 1)]
    idle = pointer >= n_real_per_neuron[:, None]
    return jnp.where(idle, jnp.nan, ms)


class TraceSummary(TypedDict):
    """summarize_trace_scalars 的回傳。"""
    n: int                           # 神經元數
    steps: int                       # 掃描步數
    total_spikes: int                # 總 spike 數
    fired: np.ndarray                # 有 fire 過的神經元 index
    idle_frac: float                 # 空轉步比例
    v_range: tuple[float, float]     # v_steps 的 (min, max)
    nonfinite_v: int                 # v_steps 裡非有限值的個數


def summarize_trace_scalars(trace: LayerForwardTrace) -> TraceSummary:
    """一筆樣本一層的軌跡 -> 整層的純量摘要,給人讀。欄位見 TraceSummary。"""
    sm = np.asarray(trace.spike_mask)
    vs = np.asarray(trace.v_steps)
    ms = np.asarray(trace.event_ms)
    n, steps = sm.shape
    fired = np.where(sm.sum(axis=1) > 0)[0]
    return {
        "n": n, "steps": steps, "total_spikes": int(sm.sum()), "fired": fired,
        "idle_frac": float(np.isnan(ms).mean()),
        "v_range": (float(np.nanmin(vs)), float(np.nanmax(vs))),
        "nonfinite_v": int(np.sum(~np.isfinite(vs))),
    }
