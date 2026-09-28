"""viz/replay_panels.py 的 InputChannelPanel 測試(合成事件,不需要訓練
run)。跟 ConvChannelPanel/FCWindowPanel 的測試(test_replay_panels.py)
分開放,因為 InputChannelPanel 不需要 salt_core 的層/trace,只吃原始
事件陣列——不用拖一次真實訓練當 fixture。"""
import numpy as np

from viz.replay_panels import InputChannelPanel
from viz.time_resample import build_frame_grid


def test_input_panel_places_event_at_correct_pixel_and_frame():
    # oc=2, h=3, w=4。一筆事件在 channel 0、(x=1, y=2)、t=2。
    event_times = np.array([2.0])
    x = np.array([1])
    y = np.array([2])
    c = np.array([0])
    frame_ms = build_frame_grid(0.0, 4.0, 1.0)

    panel = InputChannelPanel(channel=0, oc=2, h_in=3, w_in=4,
                               event_times=event_times, x=x, y=y, c=c,
                               n_real_events=1, frame_ms=frame_ms, dt_ms=1.0)

    frame_at_event = panel.frame(2)
    assert frame_at_event.shape == (3, 4)
    assert frame_at_event[2, 1] == 1.0
    assert np.array_equal(np.delete(frame_at_event.flatten(),
                                     2 * 4 + 1), np.zeros(3 * 4 - 1))
    # 事件發生前後的 frame,這個像素不該亮(離散量,不沿用前一筆事件)。
    assert panel.frame(0)[2, 1] == 0.0
    assert panel.frame(3)[2, 1] == 0.0


def test_input_panel_ignores_padding_beyond_n_real_events():
    # 陣列裡有 2 筆,但 n_real_events=1——第 2 筆是 padding,不該被畫出來。
    event_times = np.array([2.0, 999.0])
    x = np.array([1, 3])
    y = np.array([2, 2])
    c = np.array([0, 1])
    frame_ms = build_frame_grid(0.0, 4.0, 1.0)

    panel = InputChannelPanel(channel=1, oc=2, h_in=3, w_in=4,
                               event_times=event_times, x=x, y=y, c=c,
                               n_real_events=1, frame_ms=frame_ms, dt_ms=1.0)

    # channel=1 那一筆是 padding,被 n_real_events 切掉了,全部 frame 都該是 0。
    for t in range(panel.n_frames):
        assert not panel.frame(t).any()


def test_input_panel_is_always_discrete_with_no_value_range():
    event_times = np.array([1.0])
    frame_ms = build_frame_grid(0.0, 2.0, 1.0)

    panel = InputChannelPanel(channel=0, oc=1, h_in=2, w_in=2,
                               event_times=event_times, x=np.array([0]), y=np.array([0]),
                               c=np.array([0]), n_real_events=1, frame_ms=frame_ms, dt_ms=1.0)

    assert panel.discrete is True
    assert panel.value_range is None
    assert panel.extent is None
    assert panel.xlabel is None and panel.ylabel is None


def test_input_panel_invalid_channel_raises():
    frame_ms = build_frame_grid(0.0, 1.0, 1.0)
    try:
        InputChannelPanel(channel=2, oc=2, h_in=2, w_in=2,
                           event_times=np.array([]), x=np.array([]), y=np.array([]),
                           c=np.array([]), n_real_events=0, frame_ms=frame_ms, dt_ms=1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("預期 channel 超出 [0, oc) 要拋 ValueError")


def test_input_panel_no_events_is_all_background():
    frame_ms = build_frame_grid(0.0, 2.0, 1.0)

    panel = InputChannelPanel(channel=0, oc=1, h_in=2, w_in=2,
                               event_times=np.array([]), x=np.array([]), y=np.array([]),
                               c=np.array([]), n_real_events=0, frame_ms=frame_ms, dt_ms=1.0)

    for t in range(panel.n_frames):
        assert not panel.frame(t).any()
