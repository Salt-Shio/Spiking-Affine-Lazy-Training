"""輸入/conv/FC 各自「知道怎麼呈現自己」的 panel 物件,給
`viz.channel_grid.ChannelGridAnimation` 用。這裡才知道 conv/FC/`spike_mask`/
`v_steps` 這些字眼——`viz.channel_grid` 完全不知道,只認得 panel 物件的共同
介面(`viz.channel_grid.AnimatedPanel`:`title`/`discrete`/`extent`/
`xlabel`/`ylabel`/`value_range`/`n_frames` + `frame(t)`)。這個介面不帶任何
位置資訊,panel 只負責描述「要畫什麼」;要跟誰排在同一排、排在第幾排,是
呼叫端呼叫 `viz.channel_grid.ChannelGridAnimation.add_row` 時才決定的事,
不在這裡。

這個檔案依賴 `salt_core`(認識 `ConvLayer`/`FCLayer`/`LayerForwardTrace` 的
實際欄位),是 `viz` 套件裡專門知道 `salt_core` 這個特定核心的轉接層——
`viz.channel_grid`/`viz.time_resample` 本身不依賴 `salt_core`,不會因為這個
檔案存在而被拉進 conv/FC 的語意。放在 `viz/` 而不是 `example/`,是因為
`ConvChannelPanel`/`FCWindowPanel` 是任何用 `salt_core` 的人都會需要的可重用
轉接層,不是「怎麼呼叫 salt_core」的一次性範例。

`ConvChannelPanel`/`FCWindowPanel` 建構的時候就把整段 `frame_ms` 的內容算完
存起來,`frame(t)` 只是單純的 index,不是每次呼叫都重算——跟
`ChannelGridAnimation` 逐 frame `set_data` 的播放模型對齊。
"""
import numpy as np

from salt_core.layers import ConvLayer, FCLayer
from salt_core.monitor import LayerForwardTrace
from viz.channel_grid import unflatten_channels
from viz.time_resample import pad_events_by_neuron, resample_decay, resample_pulse, sliding_windows


def _resample_quantity(layer, quantity: str, trace: LayerForwardTrace,
                        frame_ms: np.ndarray, dt_ms: float):
    """離散量(`spike_mask`)瞬時顯示、連續量(`v_steps`)衰減插值,conv/FC
    共用同一套規則,見 `viz/time_resample.py`。回傳 `(resampled, is_discrete)`,
    `resampled` 形狀 `(n_neurons, n_frames)`。"""
    is_discrete = quantity == "spike_mask"
    event_ms = np.asarray(trace.event_ms)
    values = np.asarray(trace.spike_mask if is_discrete else trace.v_steps)
    if is_discrete:
        resampled = resample_pulse(event_ms, values, frame_ms, dt=dt_ms, background=False)
    else:
        resampled = resample_decay(event_ms, values, frame_ms, tau=layer.tau, before_first=0.0)
    return resampled, is_discrete


class InputChannelPanel:
    """原始輸入(dataset 樣本本身,`(event_times, x, y, c, n_real_events)`,
    跟 `example/replay_epoch.py` 的 `_load_train_sample` 回傳的格式一致)某一
    個 channel 的逐 frame 空間快照,跟 `ConvChannelPanel` 同構(攤成
    `(h_in, w_in)` 圖、channel-major 定址跟 `salt_core.layers.
    raw_events_to_stream` 用的 `c*(h*w)+y*w+x` 公式一致)。不用重跑
    forward——這就是 dataset 的原始樣本;也沒有 `v_steps`:輸入不是神經元,
    沒有膜電位,只有離散的「這個像素這一刻有沒有收到事件」,永遠是離散量,
    不像 `ConvChannelPanel`/`FCWindowPanel` 有 `quantity` 可選。"""

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
            unflatten_channels(resampled[:, t], oc, h_in, w_in)[channel]
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
    """conv 層某一個 channel 的逐 frame 空間快照,攤成 `(h_out, w_out)` 圖
    (`unflatten_channels`)。conv 的 pixel 位置沒有真實座標意義,不設
    `extent`/標籤。"""

    def __init__(self, layer: ConvLayer, channel: int, quantity: str,
                 trace: LayerForwardTrace, frame_ms: np.ndarray, dt_ms: float,
                 label_prefix: str = ""):
        if not (0 <= channel < layer.oc):
            raise ValueError(f"{layer.name} channel 要在 [0, {layer.oc}) 之間,收到 {channel}")

        resampled, self.discrete = _resample_quantity(layer, quantity, trace, frame_ms, dt_ms)
        self._frames = np.stack([
            unflatten_channels(resampled[:, t], layer.oc, layer.h_out, layer.w_out)[channel]
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
    """FC 層一段 neuron 範圍的滑動時間窗口——FC 沒有空間結構,neuron index
    本身就是座標,不像 conv 要經過 `unflatten_channels` 轉換;範圍/窗口寬度
    都要可控,不能把整段軌跡硬塞成一張圖(見 `viz/time_resample.py` 的
    `sliding_windows`)。座標軸是真實座標(neuron index / 相對播放時間的 ms
    偏移)。"""

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
