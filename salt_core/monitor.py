"""逐步軌跡監測(週期性 debug probe,見 docs/監測規格.md §6)。

`LayerForwardTrace` = 一層 traced forward 吐的 `(n, max_steps)` 陣列包,跟
`LayerForwardResult`(readout 契約)/ `LayerDiag`(容量哨兵)同一套 `Layer...`
命名,是給人工 debug 自訂 decoder / 動力學用的紀錄。**不進訓練熱路徑** ——
`layers.run_network_traced` 是 forward-only、`stop_gradient` 後回傳的獨立函式。

這裡只有結構定義 + 「掃描步指標 -> 真實毫秒」的還原(純函式)。實際跑 traced
forward 在 `salt_core.chunk_scan.run_layer_forward_traced` / `salt_core.layers`。
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp


class LayerForwardTrace(NamedTuple):
    """一層 traced forward 的逐步軌跡。全部 `(n, max_steps)`,n = 該層神經元數。

    - `spike_mask` / `s_value`:轉抄 `LayerForwardResult` 的同名欄位,讓一份
      `LayerForwardTrace` 自成完整的 `.npz` dump,不必另外帶 `LayerForwardResult`。
    - `v_steps`:每步 chunk 結束(套過 soft reset)的膜電位 = 完整膜電位軌跡;
      最後一欄 = `LayerForwardResult.v_final`。
    - `event_ms`:每步消化的(首)事件真實毫秒;空轉步(pointer 已越過該神經元
      的真實事件數)= `nan`。`chunk_size=1` 時每步剛好對到一筆事件;
      `chunk_size>1` 時是「這步從哪個時刻開始」。
    """
    spike_mask: jax.Array   # (n, max_steps) bool
    s_value: jax.Array      # (n, max_steps) float
    v_steps: jax.Array      # (n, max_steps) float
    event_ms: jax.Array     # (n, max_steps) float


def resolve_ms_dense(pointer: jax.Array, n_real_per_neuron: jax.Array,
                      event_times: jax.Array) -> jax.Array:
    """密集佇列(`build_fc_queue`):`pointer[i, k]` 直接是全域事件 index。

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
    """壓縮 conv 佇列(`build_conv_queue_compressed`):`pointer[i, k]` 是這顆
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
