"""example/replay_epoch.py 的測試:用共用的參考訓練(conftest.py 的 reference_run)重跑 epoch 0。"""
import numpy as np

from example.replay_epoch import load_epoch_weights, load_train_sample, replay_sample
from example.utils import load_run_record, rebuild_layers
from salt_core.layers import raw_events_to_stream, run_network

# chunk_size 不同,浮點加總的順序就不同。這組資料 FC 吃約 2.2 萬筆事件,
# chunk_size=1 跟 512 的 v_final 實測相對差約 6e-6,兩者離 float64 逐事件遞迴都在 6e-6 以內。
RTOL = 5e-5


def test_load_epoch_weights_forces_chunk_size_one(reference_run):
    (exp_dir, *_), _stdout = reference_run
    layers, params = load_epoch_weights(exp_dir, 0)
    assert [layer.chunk_size for layer in layers] == [1] * len(layers)
    assert len(params) == len(layers)


def test_replay_matches_training_forward(reference_run):
    """replay(chunk_size=1)每層軌跡形狀對得上層的神經元數;最後一層軌跡的最後一步,
    要跟用訓練時原本 chunk_size 跑出來的 v_final 在浮點捨入誤差內一致。"""
    (exp_dir, *_), _stdout = reference_run
    run_record = load_run_record(exp_dir)
    sample = load_train_sample(run_record, 0)
    replay_layers, params = load_epoch_weights(exp_dir, 0)

    traces = replay_sample(exp_dir, 0, *sample)

    assert len(traces) == len(replay_layers)
    for layer, trace in zip(replay_layers, traces):
        assert trace.v_steps.shape[0] == layer.n_neurons
        assert trace.spike_mask.shape == trace.v_steps.shape == trace.event_ms.shape

    train_layers = rebuild_layers(run_record)
    first = train_layers[0]
    in_stream = raw_events_to_stream(*sample, h_in=first.h_in, w_in=first.w_in)
    result = run_network(train_layers, params, in_stream).last
    np.testing.assert_allclose(np.asarray(traces[-1].v_steps[:, -1]),
                               np.asarray(result.v_final), rtol=RTOL)
