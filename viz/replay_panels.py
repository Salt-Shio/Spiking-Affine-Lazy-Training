"""輸入、conv 層、FC 層的動畫 panel,給 viz.channel_grid.ChannelGridAnimation 用。

這個檔依賴 salt_core(讀 ConvLayer、FCLayer、LayerForwardTrace 的欄位),viz 的其他模組不依賴。
panel 建構時就把整段 frame_ms 算完存起來,frame(t) 只取第 t 幀。
"""
import numpy as np

from salt_core.layers import ConvLayer, FCLayer
from salt_core.trace import LayerForwardTrace
from viz.time_resample import pad_events_by_neuron, resample_decay, resample_pulse, sliding_windows


def _resample_quantity(layer, quantity: str, trace: LayerForwardTrace,
                        frame_ms: np.ndarray, dt_ms: float):
    """spike_mask 用 resample_pulse、v_steps 用 resample_decay 重取樣,conv、FC 共用。
    回傳 (resampled, is_discrete),resampled 形狀 (n_neurons, n_frames)。"""
    is_discrete = quantity == "spike_mask"
    event_ms = np.asarray(trace.event_ms)
    values = np.asarray(trace.spike_mask if is_discrete else trace.v_steps)
    if is_discrete:
        resampled = resample_pulse(event_ms, values, frame_ms, dt=dt_ms, background=False)
    else:
        resampled = resample_decay(event_ms, values, frame_ms, tau=layer.tau, before_first=0.0)
    return resampled, is_discrete


class InputChannelPanel:
    """原始輸入某一個 channel 的逐幀空間快照,(h_in, w_in) 圖,永遠是離散量(這一刻有沒有事件)。

    樣本是 (event_times, x, y, c, n_real_events),格式同 example/replay_epoch.py 的 load_train_sample;
    定址同 example/utils.py 的 grid_input_events:c*(h*w) + y*w + x。不跑 forward。
    """

    def __init__(self, channel: int, oc: int, h_in: int, w_in: int,
                 event_times: np.ndarray, x: np.ndarray, y: np.ndarray, c: np.ndarray,
                 n_real_events: int, frame_ms: np.ndarray, dt_ms: float,
                 label_prefix: str = ""):
        if not (0 <= channel < oc):
            raise ValueError(f"input channel 要在 [0, {oc}) 之間,收到 {channel}")

        n_real = int(n_real_events)
        event_times = np.asarray(event_times)[:n_real]
        x, y, c = np.asarray(x)[:n_real], np.asarray(y)[:n_real], np.asarray(c)[:n_real]
        neuron_idx = c * (h_in * w_in) + y * w_in + x

        padded_ms = pad_events_by_neuron(neuron_idx, event_times, oc * h_in * w_in)
        values = np.ones(padded_ms.shape, dtype=bool)
        resampled = resample_pulse(padded_ms, values, frame_ms, dt=dt_ms, background=False)
        self._frames = np.stack([
            resampled[:, t].reshape(oc, h_in, w_in)[channel]
            for t in range(resampled.shape[1])
        ]).astype(np.float64)

        self.discrete = True
        self.title = f"{label_prefix}input / ch{channel}"
        self.extent = None
        self.xlabel = None
        self.ylabel = None
        self.n_frames = self._frames.shape[0]
        self.value_range = None

    def frame(self, t: int) -> np.ndarray:
        return self._frames[t]


class ConvChannelPanel:
    """conv 層某一個 channel 的逐幀空間快照,(h_out, w_out) 圖。像素位置沒有座標意義,不設 extent、標籤。"""

    def __init__(self, layer: ConvLayer, channel: int, quantity: str,
                 trace: LayerForwardTrace, frame_ms: np.ndarray, dt_ms: float,
                 label_prefix: str = ""):
        if not (0 <= channel < layer.oc):
            raise ValueError(f"{layer.name} channel 要在 [0, {layer.oc}) 之間,收到 {channel}")

        resampled, self.discrete = _resample_quantity(layer, quantity, trace, frame_ms, dt_ms)
        self._frames = np.stack([
            layer.unflatten_neurons(resampled[:, t])[channel]
            for t in range(resampled.shape[1])
        ]).astype(np.float64)

        self.title = f"{label_prefix}{layer.name} / {quantity} / ch{channel}"
        self.extent = None
        self.xlabel = None
        self.ylabel = None
        self.n_frames = self._frames.shape[0]
        self.value_range = None if self.discrete else (
            float(np.nanmin(self._frames)), float(np.nanmax(self._frames)))

    def frame(self, t: int) -> np.ndarray:
        return self._frames[t]


class FCWindowPanel:
    """FC 層一段神經元範圍的滑動時間窗口(sliding_windows)。座標軸是神經元 index 跟相對播放時間的 ms 偏移。"""

    def __init__(self, layer: FCLayer, neuron_lo: int, neuron_hi: int, quantity: str,
                 window_ms: float, trace: LayerForwardTrace, frame_ms: np.ndarray,
                 dt_ms: float, label_prefix: str = ""):
        if not (0 <= neuron_lo < neuron_hi <= layer.n_neurons):
            raise ValueError(f"{layer.name} neuron 範圍要滿足 0 <= lo < hi <= "
                             f"{layer.n_neurons},收到 [{neuron_lo}, {neuron_hi})")

        resampled, self.discrete = _resample_quantity(layer, quantity, trace, frame_ms, dt_ms)
        selected = resampled[neuron_lo:neuron_hi].astype(np.float64)
        half_width = max(1, round(window_ms / dt_ms))
        self._frames = sliding_windows(selected, center_indices=np.arange(frame_ms.shape[0]),
                                       half_width=half_width, pad_value=np.nan)

        window_span = half_width * dt_ms
        self.title = f"{label_prefix}{layer.name} / {quantity} / neuron [{neuron_lo},{neuron_hi})"
        self.extent = (-window_span, window_span, neuron_hi, neuron_lo)
        self.xlabel = "ms(相對目前播放時間)"
        self.ylabel = "neuron index"
        self.n_frames = self._frames.shape[0]
        self.value_range = None if self.discrete else (
            float(np.nanmin(self._frames)), float(np.nanmax(self._frames)))

    def frame(self, t: int) -> np.ndarray:
        return self._frames[t]
