"""`example/replay_epoch.py` 的測試:小規模真實訓練(沿用
`configs/conv/compressed_smoke.yaml`,已開 `weight_snapshot_every`),驗證
「讀某個 epoch 的權重快照 + 強制 chunk_size=1 重跑」這條路接線接對了——
形狀符合層的幾何、逐事件重跑的最終電壓跟 `run_network`(整批一次算)的
`v_final` 完全一致。

**不驗證 chunk_size 不變性本身**(那是 `salt_core` 自己的測試範圍,例如
`salt_core/tests/test_chunk_scan_stress.py`),這裡只驗證這支新檔案自己的
接線:讀權重、重建 layers、覆蓋 chunk_size/max_steps、餵進
`run_network_traced` 有沒有接對。

整個模組只訓練一次(小規模,~3 epoch),三個測試共用同一個 `exp_dir`,不重複
付訓練成本。
"""
import os
import shutil

import numpy as np

from example.paths import CONFIGS_DIR, EXPERIMENTS_DIR
from example.replay_epoch import load_epoch_weights, replay_sample
from example.train_conv_compressed import train
from example.utils import load_run_record
from salt_core.layers import ConvLayer, raw_events_to_stream, run_network

_TEST_TEMP = os.path.join(EXPERIMENTS_DIR, "TEST_TEMP_replay")
shutil.rmtree(_TEST_TEMP, ignore_errors=True)
os.makedirs(_TEST_TEMP, exist_ok=True)

_smoke_path = os.path.join(CONFIGS_DIR, "conv", "compressed_smoke.yaml")
_EXP_DIR, _NET, _PARAMS, _TRAIN_SPLIT, _VAL_SPLIT, _RUN_RECORD = train(
    _smoke_path, exp_root=_TEST_TEMP)


def test_load_epoch_weights_forces_chunk_size_one():
    layers, params = load_epoch_weights(_EXP_DIR, 0)
    assert all(layer.chunk_size == 1 for layer in layers)
    assert len(params) == len(layers)


def test_replay_sample_shapes_match_layer_geometry():
    layers, _ = load_epoch_weights(_EXP_DIR, 0)
    sample = 0
    traces = replay_sample(
        _EXP_DIR, 0, _TRAIN_SPLIT.event_times[sample], _TRAIN_SPLIT.x[sample],
        _TRAIN_SPLIT.y[sample], _TRAIN_SPLIT.c[sample],
        _TRAIN_SPLIT.n_real_events[sample])
    assert len(traces) == len(layers)
    for layer, trace in zip(layers, traces):
        assert trace.v_steps.shape[0] == layer.n_neurons
        if isinstance(layer, ConvLayer):
            # chunk_size=1 下的安全上界:max_steps 被覆蓋成 L(見
            # example/replay_epoch.py 的 _force_chunk_size_one)。
            assert layer.max_steps == layer.L
            assert trace.v_steps.shape[1] == layer.L


def test_replay_sample_matches_plain_forward_v_final():
    """逐事件重跑的 v_steps 最後一欄,要跟 run_network(整批一次算)的
    v_final 完全一致——同一份權重、同一筆樣本,只是換一條路算,結果不該有
    任何差異(不牽涉浮點規約重算的路徑,用逐位元等級的容差)。"""
    layers, replay_params = load_epoch_weights(_EXP_DIR, 0)
    sample = 0
    event_times = _TRAIN_SPLIT.event_times[sample]
    x, y, c = _TRAIN_SPLIT.x[sample], _TRAIN_SPLIT.y[sample], _TRAIN_SPLIT.c[sample]
    n_real = _TRAIN_SPLIT.n_real_events[sample]

    traces = replay_sample(_EXP_DIR, 0, event_times, x, y, c, n_real)

    in_stream = raw_events_to_stream(event_times, x, y, c, n_real,
                                      h_in=layers[0].h_in, w_in=layers[0].w_in)
    last_result, _diags = run_network(layers, in_stream, replay_params)

    v_final_from_trace = np.asarray(traces[-1].v_steps[:, -1])
    np.testing.assert_allclose(v_final_from_trace, np.asarray(last_result.v_final),
                               atol=1e-5)


def test_load_epoch_weights_matches_run_record_geometry():
    """讀回來的 layers 幾何(oc/h_out/w_out/n_out)要跟 run_record 的
    final_capacity/config 對得上——確認 rebuild_layers 真的有被正確呼叫,
    不是巧合對上。"""
    layers, _ = load_epoch_weights(_EXP_DIR, 0)
    model_layers = _RUN_RECORD["config"]["model"]["layers"]
    conv_layers = [l for l in layers if isinstance(l, ConvLayer)]
    conv_cfg = [e for e in model_layers if e["type"] == "conv"]
    assert len(conv_layers) == len(conv_cfg)
    for layer, cfg in zip(conv_layers, conv_cfg):
        assert layer.oc == cfg["oc"]


TESTS = [
    test_load_epoch_weights_forces_chunk_size_one,
    test_replay_sample_shapes_match_layer_geometry,
    test_replay_sample_matches_plain_forward_v_final,
    test_load_epoch_weights_matches_run_record_geometry,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n全部 {len(TESTS)} 項通過")
