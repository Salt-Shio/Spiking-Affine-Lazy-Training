"""把一層的輸出 spike 轉成「標準事件流」給下一層。這是層與層之間唯一的
共用約定:一串「什麼時候、哪顆神經元、多強」(event_times / event_source_idx /
event_gain),外加真事件數(n_real_events)。對應 docs/math/全連接forward訓練範例.md
第 6 節「多層怎麼疊」。

每種 layer 自己負責「怎麼從自己的 forward 結果吐出這條標準流」——不是一個
認識所有 layer 配對的萬能轉接器:

- `extract_output_events`:FC / 密集 conv。`spike_event_idx` 本身就是全域事件
  index,直接用。
- `extract_output_events_compressed`:壓縮 conv。`spike_event_idx` 是「這顆
  神經元壓縮佇列的第幾欄」,要先用 `local_to_global_j` 查回全域事件 index。
  這個查表是壓縮 conv 的內部細節,收在這個函式裡,不外洩成別人的參數。

兩者算出「(神經元, 全域事件 index, s_spike)」三元組之後,共用 `_pack_stream`
做排序 + 補 pad + 打包,那一步不分連接方式。

**為什麼要帶 s_spike 當下一層的 event_gain**(純 JAX、不跳出計算圖):下一層
`build_fc_queue` 用 event_source_idx(離散索引)從 W 查權重,索引操作本身對
「被索引的 W」有梯度,但不會讓上一層的權重出現在算式裡——不管上一層的權重
是多少,查出來的都只是 W 的某個 column,這條路徑上 loss 對上一層權重的梯度
恆為 0。修法:額外帶一個 s_spike(這一層每個 spike 事件,自己 spike 那一刻的
可微分強度,forward 精確等於 1)當下一層的 event_gain,乘進下一層的權重裡:
數值不變,但 s_spike 是 atan_spike 算出來的、對上一層權重有梯度,讓這條路徑
重新接通——跟 core.py 的 soft reset (1-s)*v 是同一個技巧。
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

# 補 pad 用的時間值:固定的有限大數,不用 inf。inf 會在兩個 pad 位置的
# jnp.diff 裡算出 inf-inf=NaN,而且 jnp.where 的 backward pass 會把「沒被選到
# 的分支」也算一次梯度再乘上 0——NaN*0 還是 NaN,會把梯度污染到不該污染的
# 地方(這是 JAX where/select 常見的坑)。用有限大數就不會有這個問題:
# (1-1/tau)**大數 在浮點數下乾淨地下溢成 0.0,不會變成 NaN。
_PAD_TIME = 1e12


class EventStream(NamedTuple):
    """層與層之間的標準事件流。欄位名跟 build_fc_queue 的參數一一對齊
    (event_times / event_source_idx / event_gain / n_real_events),下一層可以
    直接 `build_fc_queue(**stream._asdict(), W=..., tau=...)` 展開。

    **注意**:這裡的 event_times 是這一層吐出、= 下一層輸入的事件時間,跟
    `extract_output_events` 的輸入參數 event_times(這一層自己的輸入事件時間)
    同名但不是同一個陣列,差一層,而且是輸入經 gather + 排序 + 補 pad 後的子集。
    """
    event_times: jax.Array       # (max_total_spikes,) 已排序(遞增),前 n_real_events 筆真、後面補 _PAD_TIME
    event_source_idx: jax.Array  # (max_total_spikes,) 這層神經元自己的 index,前 n_real_events 筆有意義
    event_gain: jax.Array        # (max_total_spikes,) 對應每筆事件的 s_spike,前 n_real_events 筆有意義
    n_real_events: jax.Array     # 純量 int:真正的 spike 數量(jnp.sum 算出來,不是 Python int)


def _pack_stream(neuron_idx: jax.Array, global_event_idx: jax.Array,
                  raw_s_spike: jax.Array, input_event_times: jax.Array,
                  n_real_events: jax.Array, max_total_spikes: int) -> EventStream:
    """共用打包:給定每筆候選 spike 的「(來源神經元, 全域事件 index, s_spike)」,
    排序 + 補 pad + 打包成 EventStream。不分連接方式——FC / 密集 conv / 壓縮 conv
    的差別只在「怎麼算出 global_event_idx」,那一步在各自的 emit 函式裡做完了。

    排序鍵直接用整數 global_event_idx(假事件蓋成 n_events,保證比任何合法 index
    都大):全域事件列表本身已依時間排序,index 越大時間不會越早,所以按 index
    排 == 按時間排;量化到 ms 造成「不同事件、同一時間戳」時,index 仍是唯一且
    反映真實先後,不需要額外的 tie-break key。同一個全域事件觸發多顆神經元同時
    fire(物理同時刻),靠 `jnp.lexsort` 的穩定排序保留 `jnp.nonzero` 掃描順序
    (神經元 index 由小到大)。

    回傳 EventStream(長度固定 max_total_spikes):
      - event_times:排序後的輸出時間,前 n_real_events 筆真、後面補 _PAD_TIME。
      - event_source_idx:對應的來源神經元 index,前 n_real_events 筆有意義。
      - event_gain:對應的 s_spike,前 n_real_events 筆有意義。
      - n_real_events:原封轉出。下一層要把它分別傳進 build_fc_queue 跟
        run_layer_forward,讓 pad 事件被強制當 identity 映射、也不被算進 s_value
        ——分別解決「數值安全」跟「梯度正確」,缺一不可。
    """
    n_events = input_event_times.shape[0]
    real_mask = jnp.arange(max_total_spikes) < n_real_events

    # gather 輸出時間前把 index 明確夾進 [0, n_events):假事件的 global_event_idx
    # 可能是 sentinel(=n_events),不夾的話 JAX 預設 clip 會夾到「陣列最後一格」
    # ——有 pad 事件時那格可能是超大假時間,catch-up 會算出天文數字 Δt。
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
    """兩個 emit 函式共用的第一步:把 (n_source_neurons, max_steps) 的 spike
    格點攤平成固定長度的候選清單。回傳
    (neuron_idx, queue_col, raw_s_spike, n_real_events, max_total_spikes)。
    `queue_col` 是 spike 落在來源佇列的哪一欄——密集版就是全域事件 index,
    壓縮版是局部欄位,由呼叫端各自解讀。"""
    n_source_neurons, max_steps = spike_mask.shape
    if max_total_spikes is None:
        max_total_spikes = n_source_neurons * max_steps
    neuron_idx, step_idx = jnp.nonzero(spike_mask, size=max_total_spikes, fill_value=0)
    n_real_events = jnp.sum(spike_mask)
    queue_col = spike_event_idx[neuron_idx, step_idx]
    raw_s_spike = s_spike[neuron_idx, step_idx]
    return neuron_idx, queue_col, raw_s_spike, n_real_events, max_total_spikes


def extract_output_events(spike_mask: jax.Array, spike_event_idx: jax.Array,
                           s_spike: jax.Array, event_times: jax.Array,
                           max_total_spikes: int | None = None) -> EventStream:
    """FC / 密集 conv 的 emit:`spike_event_idx` 本身就是全域事件 index,
    直接打包。

    spike_mask: (n_source_neurons, max_steps) bool。
    spike_event_idx: (n_source_neurons, max_steps) int,全域事件 index。
    s_spike: (n_source_neurons, max_steps),spike 那一刻的可微分強度
      (chunk_scan.run_layer_forward 的同名欄位,**不是** s_value)。
    event_times: (n_total_events,) 這層自己的輸入事件時間。
    max_total_spikes: 輸出陣列固定長度上限(JAX 要靜態 shape)。預設
      n_source_neurons * max_steps,恆安全(每顆神經元每步最多一個 spike)。
    """
    neuron_idx, global_event_idx, raw_s_spike, n_real_events, max_out = _select_spikes(
        spike_mask, spike_event_idx, s_spike, max_total_spikes)
    return _pack_stream(neuron_idx, global_event_idx, raw_s_spike, event_times,
                         n_real_events, max_out)


def extract_output_events_compressed(spike_mask: jax.Array, spike_event_idx: jax.Array,
                                      s_spike: jax.Array, event_times: jax.Array,
                                      local_to_global_j: jax.Array,
                                      max_total_spikes: int | None = None) -> EventStream:
    """壓縮 conv 的 emit:`spike_event_idx` 是「這顆神經元壓縮佇列的第幾欄」,
    先用 `local_to_global_j` 查回全域事件 index,再走跟密集版完全一樣的打包。
    對應 docs/math/conv事件佇列壓縮版推導.md 第 5.3 節。

    local_to_global_j: (n_source_neurons, L) int,(神經元, 局部欄) -> 全域事件
      index,就是 connectivity/conv.py `build_conv_queue_compressed` 回傳的
      CompressedConvQueue.local_to_global_j,呼叫端原封傳進來。查表前把局部欄
      index 明確夾進 [0, L),不依賴 JAX gather 對越界 index 的預設行為。
    其餘參數同 `extract_output_events`。
    """
    neuron_idx, local_col, raw_s_spike, n_real_events, max_out = _select_spikes(
        spike_mask, spike_event_idx, s_spike, max_total_spikes)
    L = local_to_global_j.shape[1]
    safe_local_col = jnp.minimum(local_col, L - 1)
    global_event_idx = local_to_global_j[neuron_idx, safe_local_col]
    return _pack_stream(neuron_idx, global_event_idx, raw_s_spike, event_times,
                         n_real_events, max_out)
