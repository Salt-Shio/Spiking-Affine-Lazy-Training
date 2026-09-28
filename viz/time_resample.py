"""逐神經元、逐步的事件時間 event_ms 攤成等距的真實時間軸,給動畫用。

輸入要是 chunk_size=1 跑出來的軌跡:每個非空轉步剛好一筆事件,同一顆神經元的 event_ms 嚴格遞增。
- resample_pulse(離散量,例如 spike):每幀只看自己時間窗裡有沒有事件,沒有就填背景值。
- resample_decay(連續量,例如膜電位):從上一筆事件的值用 decay = 1 - 1/tau 衰減到這一幀。
- sliding_windows:從整段結果逐幀切一小塊窗口,神經元多、時間長時才看得清楚。
"""
import numpy as np


def build_frame_grid(t_start: float, t_end: float, dt: float) -> np.ndarray:
    """[t_start, t_end] 之間間隔 dt 的等距時間格(毫秒),最後一格 <= t_end。"""
    if dt <= 0:
        raise ValueError(f"dt 必須 > 0,收到 {dt}")
    if t_end < t_start:
        raise ValueError(f"t_end({t_end}) 不能小於 t_start({t_start})")
    n_frames = int(np.floor((t_end - t_start) / dt + 1e-9)) + 1
    return t_start + np.arange(n_frames) * dt


def pad_events_by_neuron(neuron_idx: np.ndarray, event_ms: np.ndarray, n_neurons: int) -> np.ndarray:
    """扁平的事件串(每筆一個 neuron_idx、event_ms)按神經元分組、組內照時間排序,
    重排成 (n_neurons, max_steps),不夠長的補 nan。形狀同 LayerForwardTrace.event_ms。"""
    neuron_idx = np.asarray(neuron_idx).astype(np.int64)
    event_ms = np.asarray(event_ms, dtype=np.float64)
    order = np.lexsort((event_ms, neuron_idx))  # 先照神經元,同神經元內照時間
    sorted_idx = neuron_idx[order]
    sorted_ms = event_ms[order]
    counts = np.bincount(neuron_idx, minlength=n_neurons)
    max_steps = int(counts.max()) if counts.size > 0 else 0
    out = np.full((n_neurons, max_steps), np.nan, dtype=np.float64)
    if max_steps == 0:
        return out
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    within_neuron_pos = np.arange(sorted_idx.shape[0]) - starts[sorted_idx]
    out[sorted_idx, within_neuron_pos] = sorted_ms
    return out


def _last_event_before(event_ms_row: np.ndarray, values_row: np.ndarray,
                        frame_ms: np.ndarray, before_first):
    """單一神經元:每一幀找最後一筆 <= frame_ms 的事件,回傳 (t, value, 有沒有找到)。
    找不到時 t=0、value=before_first。"""
    valid = ~np.isnan(event_ms_row)
    valid_ms = event_ms_row[valid]
    valid_vals = values_row[valid]
    has_event = np.zeros(frame_ms.shape, dtype=bool)
    t_last = np.zeros(frame_ms.shape, dtype=np.float64)
    v_last = np.full(frame_ms.shape, before_first)
    if valid_ms.size > 0:
        idx = np.searchsorted(valid_ms, frame_ms, side="right") - 1
        has_event = idx >= 0
        idx_clipped = np.clip(idx, 0, valid_ms.size - 1)
        t_last = np.where(has_event, valid_ms[idx_clipped], 0.0)
        v_last = np.where(has_event, valid_vals[idx_clipped], before_first)
    return t_last, v_last, has_event


def resample_pulse(event_ms: np.ndarray, values: np.ndarray, frame_ms: np.ndarray,
                    dt: float, background) -> np.ndarray:
    """離散量的重取樣。event_ms、values: (n, steps);frame_ms: (frames,),build_frame_grid 產生。

    回傳 (n, frames):第 f 幀只看 [frame_ms[f], frame_ms[f] + dt) 裡的事件,有就填值(多筆取最晚的),
    沒有就填 background,不沿用前面的事件。
    """
    event_ms = np.asarray(event_ms)
    values = np.asarray(values)
    frame_ms = np.asarray(frame_ms)
    n, n_frames = event_ms.shape[0], frame_ms.shape[0]
    out = np.full((n, n_frames), background, dtype=values.dtype)
    if n_frames == 0:
        return out
    for i in range(n):
        valid = ~np.isnan(event_ms[i])
        valid_ms = event_ms[i][valid]
        valid_vals = values[i][valid]
        if valid_ms.size == 0:
            continue
        bucket = np.floor((valid_ms - frame_ms[0]) / dt).astype(int)
        in_range = (bucket >= 0) & (bucket < n_frames)
        out[i, bucket[in_range]] = valid_vals[in_range]
    return out


def sliding_windows(array: np.ndarray, center_indices, half_width: int, pad_value) -> np.ndarray:
    """array: (rows, n_frames)。對每個 center_indices 切出欄 [center - half_width, center + half_width]。

    超出邊界的欄填 pad_value,不夾、不循環。回傳 (len(center_indices), rows, 2 * half_width + 1)。
    """
    array = np.asarray(array)
    center_indices = np.asarray(center_indices)
    rows, n_frames = array.shape
    width = 2 * half_width + 1
    out = np.full((center_indices.shape[0], rows, width), pad_value,
                  dtype=np.result_type(array, pad_value))
    for i, center in enumerate(center_indices):
        lo, hi = center - half_width, center + half_width + 1
        src_lo, src_hi = max(lo, 0), min(hi, n_frames)
        if src_lo >= src_hi:
            continue
        dst_lo = src_lo - lo
        out[i, :, dst_lo:dst_lo + (src_hi - src_lo)] = array[:, src_lo:src_hi]
    return out


def resample_decay(event_ms: np.ndarray, values: np.ndarray, frame_ms: np.ndarray,
                    tau: float, before_first: float = 0.0) -> np.ndarray:
    """連續量的重取樣:找最後一筆 <= frame_ms 的 (t_last, v_last),值是
    v_last * (1 - 1/tau) ** (frame_ms - t_last)。早於第一筆事件時從 t=0、v=before_first(預設 0)開始衰減。
    """
    event_ms = np.asarray(event_ms)
    values = np.asarray(values)
    frame_ms = np.asarray(frame_ms)
    decay = 1.0 - 1.0 / tau
    n = event_ms.shape[0]
    out = np.empty((n, frame_ms.shape[0]), dtype=np.float64)
    for i in range(n):
        t_last, v_last, _ = _last_event_before(event_ms[i], values[i], frame_ms, before_first)
        out[i] = v_last * decay ** (frame_ms - t_last)
    return out
