"""`viz/replay_panels.py` 的測試:小規模真實訓練(內嵌 config,開
`weight_snapshot_every`),驗證 `ConvChannelPanel`/`FCWindowPanel` 算出來的
`frame(t)` 形狀/值跟層的幾何、`AnimatedPanel` 介面(`viz/channel_grid.py`)
要求的屬性都對得上。整個模組只訓練一次,測試共用同一個 `exp_dir`。

被測的模組本身只依賴 `salt_core`/`viz`,不依賴 `example/`;這裡的測試依賴
`example/` 只是為了借它的訓練/replay 機制產生真實的層幾何跟 trace 當 fixture
用,不代表被測模組本身跟 `example/` 有關係。
"""
import os
import shutil

import numpy as np
import yaml

from example.paths import EXPERIMENTS_DIR
from example.replay_epoch import load_train_sample, load_epoch_weights, replay_sample
from example.train_conv_compressed import train
from example.utils import load_run_record
from salt_core.layers import ConvLayer, FCLayer
from viz.replay_panels import ConvChannelPanel, FCWindowPanel
from viz.time_resample import build_frame_grid

_TEST_TEMP = os.path.join(EXPERIMENTS_DIR, "TEST_TEMP_replay_panels")
shutil.rmtree(_TEST_TEMP, ignore_errors=True)
os.makedirs(_TEST_TEMP, exist_ok=True)

_CFG = {
    "run_name": "replay_panels_smoke",
    "model": {
        "decoder": "membrane_regression",
        "input_shape": [2, 34, 34],
        "layers": [
            {"type": "conv", "oc": 8, "k": 3, "s": 2, "p": 1, "tau": 16.0, "v_th": 1.0,
             "alpha": 2.0, "chunk_size": 1, "L": 185, "max_out_spikes": 800, "init_k": 8.0,
             "L_grow_factor": 1.5, "out_grow_factor": 1.5},
            {"type": "conv", "oc": 16, "k": 3, "s": 2, "p": 1, "tau": 16.0, "v_th": 1.0,
             "alpha": 2.0, "chunk_size": 1, "L": 32, "max_out_spikes": 600, "init_k": 64.0,
             "L_grow_factor": 1.5, "out_grow_factor": 1.5},
            {"type": "fc", "name": "out", "n_out": 10, "tau": 16.0, "v_th": 1.0e9,
             "alpha": 2.0, "chunk_size": 512, "init_k": 5.0},
        ],
    },
    "data": {"max_events": 2000, "train_size": 16, "val_size": 8, "seed_train": 0, "seed_val": 0},
    "train": {"lr": 1.0e-2, "epochs": 1, "batch_size": 4, "seed": 42, "weight_snapshot_every": 1},
}
_CFG_PATH = os.path.join(_TEST_TEMP, "replay_panels_smoke.yaml")
with open(_CFG_PATH, "w", encoding="utf-8") as _f:
    yaml.safe_dump(_CFG, _f, allow_unicode=True)

_EXP_DIR, _NET, _PARAMS, _TRAIN_SPLIT, _VAL_SPLIT, _RUN_RECORD = train(_CFG_PATH, exp_root=_TEST_TEMP)
_LAYERS, _ = load_epoch_weights(_EXP_DIR, 0)
_RUN_RECORD = load_run_record(_EXP_DIR)
_CONV1 = next(l for l in _LAYERS if isinstance(l, ConvLayer) and l.name == "conv1")
_FC_OUT = next(l for l in _LAYERS if isinstance(l, FCLayer))

_SAMPLE = load_train_sample(_RUN_RECORD, 0)
_TRACES = replay_sample(_EXP_DIR, 0, *_SAMPLE)
_CONV1_TRACE = _TRACES[_LAYERS.index(_CONV1)]
_FC_TRACE = _TRACES[_LAYERS.index(_FC_OUT)]

_DT_MS = 1.0
_FRAME_MS = build_frame_grid(0.0, float(np.nanmax(np.asarray(_CONV1_TRACE.event_ms))), _DT_MS)


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
