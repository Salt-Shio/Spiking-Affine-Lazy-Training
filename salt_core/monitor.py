"""逐步軌跡監測(週期性 debug probe,見 docs/監測規格.md §6)。

`LayerForwardTrace` = 一層 traced forward 吐的 `(n, max_steps)` 陣列包,跟
`LayerForwardResult`(readout 契約)/ `LayerDiag`(容量哨兵)同一套 `Layer...`
命名,是給人工 debug 自訂 decoder / 動力學用的紀錄。`layers.run_network(...,
trace=True)` 收集,`stop_gradient` 後回傳;浮點、整數 backend 都用這個型別。

這裡除了結構定義,還有「掃描步指標 -> 真實毫秒」的還原(`resolve_ms_*`)、
一個對 `LayerForwardTrace` 的純歸約函式(`summarize_trace_scalars`)——只吃
這個型別,不知道呼叫端是誰,跟 `dormant.py` 的 `dormant_score` 同一個放置
理由(見 docs/監測規格.md §4.1:「salt_core 放通用 helper」)。

實際跑 traced forward 在 `salt_core.chunk_scan.run_layer_forward_traced` /
`salt_core.layers`。

**2026-09-13 拔掉的東西**:`s_value` 欄位、`summarize_trace`、`pack_key`/
`unpack_key`/`layer_names`。理由:(1) `s_value` 在這個架構下 forward 數值
恆等於 `spike_mask`(一個 chunk 裡最多一筆事件跨過門檻,`chunk_scan.py` 的
`n_valid_in_chunk` 邏輯保證),而 `LayerForwardTrace` 本身又是
`stop_gradient` 後才回傳,連它在別處才有意義的「可微分」也用不上——留著
純粹是重複資訊。(2) 這四樣東西唯一的呼叫端是已移除的
`example/trace_probe.py`(多層攤平存 `summary.npz`),拔掉 `s_value` 之後
`summarize_trace` 也跟著沒有存在理由。見 docs/監測規格.md §7。
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class LayerForwardTrace(NamedTuple):
    """一層 traced forward 的逐步軌跡。全部 `(n, max_steps)`,n = 該層神經元數。

    - `spike_mask`:轉抄 `LayerForwardResult` 的同名欄位。
    - `v_steps`:每步 chunk 結束(套過 soft reset)的膜電位 = 完整膜電位軌跡;
      最後一欄 = `LayerForwardResult.v_final`。整數 backend 是暫存器的整數值。
    - `event_ms`:每步消化的(首)事件真實毫秒;空轉步(pointer 已越過該神經元
      的真實事件數)= `nan`。`chunk_size=1` 時每步剛好對到一筆事件;
      `chunk_size>1` 時是「這步從哪個時刻開始」。
    """
    spike_mask: jax.Array   # (n, max_steps) bool
    v_steps: jax.Array      # (n, max_steps) float
    event_ms: jax.Array     # (n, max_steps) float


def resolve_ms_dense(pointer: jax.Array, n_real_per_neuron: jax.Array,
                      event_times: jax.Array) -> jax.Array:
    """FC 佇列(`build_fc_structure`):`pointer[i, k]` 直接是全域事件 index。

    pointer / n_real_per_neuron 形狀 `(n,)` 對齊;event_times `(n_events,)`。
    回傳 `(n, max_steps)`,空轉步(`pointer >= n_real`)= `nan`。
    """
    n_events = event_times.shape[0]
    ms = jnp.asarray(event_times)[jnp.clip(pointer, 0, n_events - 1)]
    idle = pointer >= n_real_per_neuron[:, None]
    return jnp.where(idle, jnp.nan, ms)


def resolve_ms_compressed(pointer: jax.Array, local_to_global_j: jax.Array,
                           n_real_per_neuron: jax.Array,
                           event_times: jax.Array) -> jax.Array:
    """壓縮 conv 佇列(`build_conv_structure`):`pointer[i, k]` 是這顆
    神經元壓縮佇列的局部欄,要先查 `local_to_global_j[i, col]` 得全域事件 index。

    `local_to_global_j` `(n, L)`;空欄的哨兵值 = `n_events`(見 connectivity/conv.py)。
    回傳 `(n, max_steps)`,空轉步 = `nan`。
    """
    n_events = event_times.shape[0]
    n_cols = local_to_global_j.shape[1]
    global_idx = jnp.take_along_axis(
        local_to_global_j, jnp.clip(pointer, 0, n_cols - 1), axis=1)  # (n, max_steps)
    ms = jnp.asarray(event_times)[jnp.minimum(global_idx, n_events - 1)]
    idle = pointer >= n_real_per_neuron[:, None]
    return jnp.where(idle, jnp.nan, ms)


def summarize_trace_scalars(trace: LayerForwardTrace) -> dict:
    """一份 `LayerForwardTrace`(單一樣本)-> 整層純量摘要 dict,給人工讀逐步
    軌跡用(`example/replay_epoch.py` 的 CLI 用它印報表)。

      n / steps       形狀
      total_spikes    整層總 spike 數
      fired           有 fire 過的神經元 index,`(k,)` 陣列
      idle_frac       空轉步比例(對整層平均)
      v_range         `(v_steps 的 min, max)`
      nonfinite_v     `v_steps` 裡非有限值的個數
    """
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
