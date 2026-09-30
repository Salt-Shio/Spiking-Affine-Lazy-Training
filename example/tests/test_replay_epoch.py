"""example/replay_epoch.py 的測試:用共用的參考訓練(conftest.py 的 reference_run)重跑 epoch 0。"""
import csv
import os

import numpy as np

from example.metrics_log import KNOB_COLUMNS
from example.replay_epoch import load_epoch_weights, load_train_sample, replay_sample
from example.utils import (TRAIN_DIRNAME, WEIGHTS_DIRNAME, grid_input_events, load_run_record,
                           weight_snapshot_path)
from salt_core.io import load_weights

# chunk_size 不同,浮點加總的順序就不同。這組資料 FC 吃約 2.2 萬筆事件,
# chunk_size=1 跟 512 的 v_final 實測相對差約 6e-6,兩者離 float64 逐事件遞迴都在 6e-6 以內。
RTOL = 5e-5


def test_load_epoch_weights_forces_chunk_size_one(reference_run):
    exp_dir = reference_run.exp_dir
    network, params = load_epoch_weights(exp_dir, 0)
    assert [layer.chunk_size for layer in network.layers] == [1] * len(network.layers)
    assert len(params) == len(network.layers)


def test_snapshot_carries_capacity_of_its_epoch(reference_run):
    """每個 epoch 的權重快照帶的容量,等於 metrics.csv 那個 epoch 記的容量。"""
    exp_dir = reference_run.exp_dir
    with open(os.path.join(exp_dir, TRAIN_DIRNAME, "metrics.csv"), newline="",
              encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        epoch = int(row["epoch"])
        network, _ = load_weights(weight_snapshot_path(os.path.join(exp_dir, WEIGHTS_DIRNAME),
                                                       epoch))
        for layer in network.layers:
            if layer.capacity is None:
                continue
            expected = {knob: int(row[f"{layer.name}_{KNOB_COLUMNS[knob].capacity}"])
                        for knob in layer.capacity}
            assert dict(layer.capacity) == expected, f"epoch {epoch} {layer.name}"


def test_replay_matches_training_forward(reference_run):
    """replay(chunk_size=1)每層軌跡形狀對得上層的神經元數;最後一層軌跡的最後一步,
    要跟用訓練時原本 chunk_size 跑出來的 v_final 在浮點捨入誤差內一致。"""
    exp_dir = reference_run.exp_dir
    run_record = load_run_record(exp_dir)
    sample = load_train_sample(run_record, 0)
    replay_network, params = load_epoch_weights(exp_dir, 0)
    replay_layers = replay_network.layers

    traces = replay_sample(exp_dir, 0, *sample)

    assert len(traces) == len(replay_layers)
    for layer, trace in zip(replay_layers, traces):
        assert trace.v_steps.shape[0] == layer.n_neurons
        assert trace.spike_mask.shape == trace.v_steps.shape == trace.event_ms.shape

    training_network, _ = load_weights(
        weight_snapshot_path(os.path.join(exp_dir, WEIGHTS_DIRNAME), 0))
    result = training_network.apply(
        params, grid_input_events(*sample, training_network.input_shape)).last
    np.testing.assert_allclose(np.asarray(traces[-1].v_steps[:, -1]),
                               np.asarray(result.v_final), rtol=RTOL)
