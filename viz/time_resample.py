"""`event_ms`(逐神經元、逐步的真實事件時間戳)攤成一條等距的真實時間軸。

`example/replay_epoch.py` 保證 `chunk_size=1`,所以每個非 idle 步剛好對應
一筆真實事件——同一顆神經元的 `event_ms`(忽略 `nan` 的 idle 尾巴)嚴格遞增。
這裡不用再處理「一步吃到 chunk 裡好幾筆事件中的哪一筆」的模糊地帶。

離散量(`spike_mask`)用 `resample_pulse` 做瞬時顯示:spike 是一個時間點上
發生的瞬時事件,事件跟事件之間這個 neuron 根本沒有被計算過(不是「還在
fire」,是單純沒發生任何事),所以某個 frame 有沒有落到真實事件就是有/沒有,
不會沿用前一個 frame 的值。連續量(`v_steps`)用 `resample_decay` 做衰減插值
(用該層的 `tau` 算 `decay = 1 - 1/tau`,frame 落在兩筆事件中間時從上一筆事件
的值往前衰減——電位是真的連續存在、持續衰減的物理量,「沿用前一筆事件的值
再衰減」才有意義,這點跟 spike 不一樣)。兩者都不知道 `spike_mask`/`v_steps`
這些字眼,只吃 `(event_ms, values)` 這組通用資料。
"""
import numpy as np


def build_frame_grid(t_start: float, t_end: float, dt: float) -> np.ndarray:
    """`[t_start, t_end]` 之間、間隔 `dt` 的等距時間格(單位跟 `event_ms` 一樣
    是毫秒)。`t_end` 落在格子之間時,最後一格 <= `t_end`,不會超出去。"""
    if dt <= 0:
        raise ValueError(f"dt 必須 > 0,收到 {dt}")
    if t_end < t_start:
        raise ValueError(f"t_end({t_end}) 不能小於 t_start({t_start})")
    n_frames = int(np.floor((t_end - t_start) / dt + 1e-9)) + 1
    return t_start + np.arange(n_frames) * dt


def _last_event_before(event_ms_row: np.ndarray, values_row: np.ndarray,
                        frame_ms: np.ndarray, before_first):
    """單一神經元:對每個 frame 找「最後一筆 `<= frame_ms` 的真實事件」,回傳
    `(該事件的 t, 該事件的 value, 該 frame 是否真的找到事件)`。找不到(frame
    早於第一筆事件,或這顆神經元整條軌跡都沒有真實事件)時,`t=0`(LIF 的
    起始時刻)、`value=before_first`。"""
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
    """離散量的瞬時顯示重取樣。`event_ms`/`values` 形狀 `(n, steps)`,
    `frame_ms` 形狀 `(frames,)`(`build_frame_grid` 產生的等距格,間距就是
    `dt`)。回傳 `(n, frames)`:每個 frame 只看自己的時間窗
    `[frame_ms[f], frame_ms[f]+dt)` 裡有沒有真實事件——有,填那筆事件的值
    (窗內不只一筆就用最晚的一筆);沒有,填 `background`。**不會**像
    `resample_decay` 那樣往前找更早的事件——事件跟事件之間沒有「持續狀態」
    這種東西可以沿用。"""
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


def resample_decay(event_ms: np.ndarray, values: np.ndarray, frame_ms: np.ndarray,
                    tau: float, before_first: float = 0.0) -> np.ndarray:
    """連續量的衰減插值重取樣。先找「最後一筆 `<= frame_ms` 的 `(t_last,
    v_last)`」,再用 `decay = 1 - 1/tau` 算 `v_last * decay**(frame_ms -
    t_last)`(對齊 `salt_core/core.py` 的仿射衰減)。早於第一筆事件的 frame
    從 `t=0, v=before_first` 開始衰減——`before_first` 預設 0,對齊神經元真正
    的起始電位(不是拿 `nan` 隨便填)。"""
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
