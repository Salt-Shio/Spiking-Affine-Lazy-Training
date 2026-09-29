"""層與層之間的事件流:EventStream,跟把一層的 spike 轉成 EventStream 的函式。

每種層用自己的函式轉:FC 的 spike_event_idx 已經是全域事件 index;conv 的是這顆神經元
佇列的局部欄,要先查 local_to_global_j。之後的排序、補 pad、打包共用 _pack_stream。
event_gain 帶 s_spike 的理由見 docs/問題紀錄.md「洞見:離散索引 gather 不會把梯度帶回「決定索引值的來源」」。
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

# pad 事件的時間。不用 inf:inf - inf 是 NaN,jnp.where 的 backward 會把沒選到那邊的 NaN 帶進梯度。
_PAD_TIME = 1e12


class EventStream(NamedTuple):
    """層與層之間的事件流。前 n_real_events 筆是真事件,其餘是 pad。"""
    event_times: jax.Array       # (max_total_spikes,) 遞增;pad 位置是 _PAD_TIME
    event_source_idx: jax.Array  # (max_total_spikes,) 送出這筆事件的神經元
    event_gain: jax.Array        # (max_total_spikes,) 這筆事件的 s_spike,下一層乘進權重
    n_real_events: jax.Array     # int 純量(JAX 陣列)


def _pack_stream(neuron_idx: jax.Array, global_event_idx: jax.Array,
                  raw_s_spike: jax.Array, input_event_times: jax.Array,
                  n_real_events: jax.Array, max_total_spikes: int) -> EventStream:
    """候選 spike 的 (來源神經元, 全域事件 index, s_spike) -> 排序、補 pad、打包成 EventStream。

    排序鍵是全域事件 index:輸入事件已按時間排序,index 的順序就是時間順序,同一個時間戳也
    保留先後;同一筆事件讓幾顆神經元同時 fire 時,照神經元 index 由小到大(lexsort 是穩定排序)。
    回傳長度固定 max_total_spikes 的 EventStream。
    """
    n_events = input_event_times.shape[0]
    real_mask = jnp.arange(max_total_spikes) < n_real_events

    # pad 的 global_event_idx 可能是 n_events,先夾回範圍內再取時間;取到的值會被下面換成 _PAD_TIME
    safe_global_idx = jnp.minimum(global_event_idx, n_events - 1)
    raw_times = jnp.asarray(input_event_times)[safe_global_idx]

    order = jnp.lexsort((jnp.where(real_mask, global_event_idx, n_events),))

    out_times = jnp.where(real_mask, raw_times, _PAD_TIME)[order]
    return EventStream(event_times=out_times,
                        event_source_idx=neuron_idx[order],
                        event_gain=raw_s_spike[order],
                        n_real_events=n_real_events)


def _select_spikes(spike_mask: jax.Array, spike_event_idx: jax.Array,
                    s_spike: jax.Array, max_total_spikes: int | None):
    """(n, max_steps) 的 spike 格點攤平成固定長度的候選清單。

    回傳 (neuron_idx, queue_col, raw_s_spike, n_real_events, max_total_spikes);queue_col 是 spike
    在來源佇列的第幾欄,由呼叫端解讀。
    """
    n_source_neurons, max_steps = spike_mask.shape
    if max_total_spikes is None:
        max_total_spikes = n_source_neurons * max_steps
    neuron_idx, step_idx = jnp.nonzero(spike_mask, size=max_total_spikes, fill_value=0)
    n_real_events = jnp.sum(spike_mask)
    queue_col = spike_event_idx[neuron_idx, step_idx]
    raw_s_spike = s_spike[neuron_idx, step_idx]
    return neuron_idx, queue_col, raw_s_spike, n_real_events, max_total_spikes


def extract_output_events_fc(spike_mask: jax.Array, spike_event_idx: jax.Array,
                             s_spike: jax.Array, event_times: jax.Array,
                             max_total_spikes: int | None = None) -> EventStream:
    """FC 層的 spike -> EventStream。spike_event_idx 就是全域事件 index。

    spike_mask: (n, max_steps) bool。
    spike_event_idx: (n, max_steps) int。
    s_spike: (n, max_steps) forward 結果的 s_spike(不是 s_value)。
    event_times: (n_events,) 這層的輸入事件時間。
    max_total_spikes: 輸出長度(靜態)。預設 n * max_steps,一定裝得下。
    """
    neuron_idx, global_event_idx, raw_s_spike, n_real_events, max_out = _select_spikes(
        spike_mask, spike_event_idx, s_spike, max_total_spikes)
    return _pack_stream(neuron_idx, global_event_idx, raw_s_spike, event_times,
                         n_real_events, max_out)


def extract_output_events_conv(spike_mask: jax.Array, spike_event_idx: jax.Array,
                               s_spike: jax.Array, event_times: jax.Array,
                               local_to_global_j: jax.Array,
                               max_total_spikes: int | None = None) -> EventStream:
    """conv 層的 spike -> EventStream。spike_event_idx 是這顆神經元佇列的局部欄,先查回全域事件 index。

    local_to_global_j: (n, max_queue_len) int,局部欄 -> 全域事件 index(ConvQueueStructure.local_to_global_j
        用 tile_channels 展開到每個 channel)。
    其餘參數同 extract_output_events_fc。
    """
    neuron_idx, local_col, raw_s_spike, n_real_events, max_out = _select_spikes(
        spike_mask, spike_event_idx, s_spike, max_total_spikes)
    max_queue_len = local_to_global_j.shape[1]
    safe_local_col = jnp.minimum(local_col, max_queue_len - 1)
    global_event_idx = local_to_global_j[neuron_idx, safe_local_col]
    return _pack_stream(neuron_idx, global_event_idx, raw_s_spike, event_times,
                         n_real_events, max_out)
