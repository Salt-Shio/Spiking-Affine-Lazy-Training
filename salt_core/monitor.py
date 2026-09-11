"""逐步軌跡監測(週期性 debug probe,見 docs/監測規格.md §6)。

`LayerForwardTrace` = 一層 traced forward 吐的 `(n, max_steps)` 陣列包,跟
`LayerForwardResult`(readout 契約)/ `LayerDiag`(容量哨兵)同一套 `Layer...`
命名,是給人工 debug 自訂 decoder / 動力學用的紀錄。**不進訓練熱路徑** ——
`layers.run_network_traced` 是 forward-only、`stop_gradient` 後回傳的獨立函式。

這裡除了結構定義 + 「掃描步指標 -> 真實毫秒」的還原,還有:

- 兩個對 `LayerForwardTrace` 的純歸約函式(`summarize_trace` /
  `summarize_trace_scalars`)——只吃這個型別,不知道呼叫端是訓練期探測還是
  離線讀 `.npz` 分析,跟 `dormant.py` 的 `dormant_score` 同一個放置理由(見
  docs/監測規格.md §4.1:「salt_core 放通用 helper」)。
- `pack_key`/`unpack_key`/`layer_names`:「多層 x 多欄」攤平成扁平字串 key 的
  命名慣例,一樣只認 salt_core 自己的型別(一列 `LayerForwardTrace` 按層名對齊),
  不知道呼叫端要拿去存 npz 還是別的格式。

實際跑 traced forward 在 `salt_core.chunk_scan.run_layer_forward_traced` /
`salt_core.layers`。
"""
from typing import Iterable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

_SEP = "__"


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


def summarize_trace(trace: LayerForwardTrace) -> dict:
    """一份 `LayerForwardTrace`(單一樣本,四欄皆 `(n, max_steps)`)-> 逐神經元
    `(n,)` 摘要 dict。訓練期週期性探測(`example/trace_probe.py`)逐樣本呼叫、
    對探測批平均,寫進 `summary.npz`;四個 key 的意義見 docs/監測規格.md §2/§7.1。

      spike_count  該樣本這顆神經元的總 spike 數
      s_value_sum  Σ_t s_value(離門檻多近的連續量累積)
      v_final      最終膜電位(= v_steps 最後一欄)
      idle_frac    空轉步(event_ms = nan)比例
    """
    return {
        "spike_count": trace.spike_mask.sum(axis=1).astype(jnp.float32),
        "s_value_sum": trace.s_value.sum(axis=1),
        "v_final": trace.v_steps[:, -1],
        "idle_frac": jnp.isnan(trace.event_ms).mean(axis=1),
    }


def summarize_trace_scalars(trace: LayerForwardTrace) -> dict:
    """一份 `LayerForwardTrace`(單一樣本)-> 整層純量摘要 dict,給人工讀
    `full_epoch_XXX.npz` 用(`example/inspect_traces.py`)。

      n / steps                   形狀
      total_spikes                整層總 spike 數
      fired                       有 fire 過的神經元 index,`(k,)` 陣列
      idle_frac                   空轉步比例(對整層平均)
      v_range                     `(v_steps 的 min, max)`
      nonfinite_v / nonfinite_s   `v_steps` / `s_value` 裡非有限值的個數
    """
    sm = np.asarray(trace.spike_mask)
    sv = np.asarray(trace.s_value)
    vs = np.asarray(trace.v_steps)
    ms = np.asarray(trace.event_ms)
    n, steps = sm.shape
    fired = np.where(sm.sum(axis=1) > 0)[0]
    return {
        "n": n, "steps": steps, "total_spikes": int(sm.sum()), "fired": fired,
        "idle_frac": float(np.isnan(ms).mean()),
        "v_range": (float(np.nanmin(vs)), float(np.nanmax(vs))),
        "nonfinite_v": int(np.sum(~np.isfinite(vs))),
        "nonfinite_s": int(np.sum(~np.isfinite(sv))),
    }


def pack_key(layer_name: str, field: str) -> str:
    """`(層名, 欄名)` -> 扁平字串 key,例如 `("conv1", "spike_count")` ->
    `"conv1__spike_count"`。給任何要把「多層 x 多欄」的資料存成單層 dict(例如
    `np.savez`)的呼叫端共用——訓練期探測(`example/trace_probe.py`)寫、離線
    分析(`example/inspect_traces.py`)讀,是這個命名慣例唯一的實作,不必各自
    重刻一次 `f"{a}__{b}"`。"""
    return f"{layer_name}{_SEP}{field}"


def unpack_key(key: str) -> tuple[str, str]:
    """`pack_key` 的反函式:扁平字串 key -> `(層名, 欄名)`。層名本身不能含 `__`。"""
    name, field = key.split(_SEP, 1)
    return name, field


def layer_names(keys: Iterable[str]) -> list[str]:
    """從一堆 `pack_key` 產生的 key 依首次出現順序取出不重複的層名。略過沒有
    `__` 的 key(例如 `summary.npz` 裡的 `epochs`)。"""
    names: list[str] = []
    for key in keys:
        if _SEP not in key:
            continue
        name, _field = unpack_key(key)
        if name not in names:
            names.append(name)
    return names
