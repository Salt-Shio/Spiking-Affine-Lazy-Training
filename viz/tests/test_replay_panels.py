"""`viz/replay_panels.py` 的 ConvChannelPanel、FCWindowPanel 測試:用手工建的層跟
逐步軌跡,驗證 frame(t) 的形狀、值跟層的幾何,以及 AnimatedPanel 介面
(viz/channel_grid.py)要求的屬性都對得上。
"""
import numpy as np

from salt_core.layers import ConvLayer, FCLayer
from salt_core.monitor import LayerForwardTrace
from viz.replay_panels import ConvChannelPanel, FCWindowPanel
from viz.time_resample import build_frame_grid

_CONV1 = ConvLayer(name="conv1", ic=2, h_in=8, w_in=8, oc=2, k=3, s=2, p=1, init_k=5.0)
_FC_OUT = FCLayer(name="out", n_in=_CONV1.n_neurons, n_out=10, init_k=5.0)


def _trace(n_neurons: int, seed: int) -> LayerForwardTrace:
    """5 步的軌跡:每顆神經元在 1、3、5、8 ms 各消化一筆事件,最後一步空轉(event_ms 是 nan)。"""
    rng = np.random.default_rng(seed)
    event_ms = np.tile(np.array([1.0, 3.0, 5.0, 8.0, np.nan]), (n_neurons, 1))
    v_steps = rng.normal(0.0, 1.0, size=(n_neurons, 5))
    return LayerForwardTrace(spike_mask=v_steps > 1.0, v_steps=v_steps, event_ms=event_ms)


_CONV1_TRACE = _trace(_CONV1.n_neurons, seed=0)
_FC_TRACE = _trace(_FC_OUT.n_neurons, seed=1)

_DT_MS = 1.0
_FRAME_MS = build_frame_grid(0.0, float(np.nanmax(_CONV1_TRACE.event_ms)), _DT_MS)


def test_conv_panel_frame_shape_matches_layer_geometry():
    panel = ConvChannelPanel(_CONV1, channel=0, quantity="v_steps",
                              trace=_CONV1_TRACE, frame_ms=_FRAME_MS, dt_ms=_DT_MS)

    assert panel.n_frames == _FRAME_MS.shape[0]
    assert panel.frame(0).shape == (_CONV1.h_out, _CONV1.w_out)
    assert panel.frame(panel.n_frames - 1).shape == (_CONV1.h_out, _CONV1.w_out)


def test_conv_panel_spike_mask_is_discrete_with_no_value_range():
    panel = ConvChannelPanel(_CONV1, channel=0, quantity="spike_mask",
                              trace=_CONV1_TRACE, frame_ms=_FRAME_MS, dt_ms=_DT_MS)

    assert panel.discrete is True
    assert panel.value_range is None
    assert panel.extent is None
    assert panel.xlabel is None and panel.ylabel is None


def test_conv_panel_v_steps_is_continuous_with_value_range():
    panel = ConvChannelPanel(_CONV1, channel=0, quantity="v_steps",
                              trace=_CONV1_TRACE, frame_ms=_FRAME_MS, dt_ms=_DT_MS)

    assert panel.discrete is False
    assert panel.value_range is not None
    lo, hi = panel.value_range
    assert lo <= hi


def test_conv_panel_invalid_channel_raises():
    try:
        ConvChannelPanel(_CONV1, channel=_CONV1.oc, quantity="v_steps",
                          trace=_CONV1_TRACE, frame_ms=_FRAME_MS, dt_ms=_DT_MS)
    except ValueError:
        pass
    else:
        raise AssertionError("預期 channel 超出 [0, oc) 要拋 ValueError")


def test_fc_panel_frame_shape_matches_neuron_range_and_window():
    window_ms = 5.0
    lo, hi = 0, 5
    panel = FCWindowPanel(_FC_OUT, neuron_lo=lo, neuron_hi=hi, quantity="v_steps",
                           window_ms=window_ms, trace=_FC_TRACE, frame_ms=_FRAME_MS, dt_ms=_DT_MS)

    half_width = round(window_ms / _DT_MS)
    assert panel.n_frames == _FRAME_MS.shape[0]
    assert panel.frame(0).shape == (hi - lo, 2 * half_width + 1)


def test_fc_panel_extent_and_labels_use_real_coordinates():
    window_ms = 5.0
    lo, hi = 0, 5
    panel = FCWindowPanel(_FC_OUT, neuron_lo=lo, neuron_hi=hi, quantity="v_steps",
                           window_ms=window_ms, trace=_FC_TRACE, frame_ms=_FRAME_MS, dt_ms=_DT_MS)

    assert panel.extent == (-window_ms, window_ms, hi, lo)
    assert panel.xlabel == "ms(相對目前播放時間)"
    assert panel.ylabel == "neuron index"


def test_fc_panel_invalid_neuron_range_raises():
    try:
        FCWindowPanel(_FC_OUT, neuron_lo=5, neuron_hi=5, quantity="v_steps",
                      window_ms=5.0, trace=_FC_TRACE, frame_ms=_FRAME_MS, dt_ms=_DT_MS)
    except ValueError:
        pass
    else:
        raise AssertionError("預期 lo == hi 要拋 ValueError")


def test_fc_panel_early_frames_pad_missing_history_with_nan():
    # 播放剛開始(frame 0),窗口左半邊理論上要延伸到負的時間,還沒有真實
    # 歷史可用,應該是 NaN,不是隨便一個數字。
    window_ms = 5.0
    panel = FCWindowPanel(_FC_OUT, neuron_lo=0, neuron_hi=5, quantity="v_steps",
                           window_ms=window_ms, trace=_FC_TRACE, frame_ms=_FRAME_MS, dt_ms=_DT_MS)

    first_frame = panel.frame(0)
    half_width = round(window_ms / _DT_MS)
    assert np.isnan(first_frame[:, :half_width]).all()
