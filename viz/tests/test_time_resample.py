"""viz/time_resample.py 的單元測試(合成陣列,不需要訓練 run)。"""
import numpy as np

from viz.time_resample import (build_frame_grid, pad_events_by_neuron, resample_decay,
                               resample_pulse, sliding_windows)


def test_build_frame_grid_exact_multiple():
    grid = build_frame_grid(0, 10, 2)
    assert np.array_equal(grid, [0, 2, 4, 6, 8, 10])


def test_build_frame_grid_last_frame_not_exceeding_t_end():
    grid = build_frame_grid(0, 9, 2)
    assert np.array_equal(grid, [0, 2, 4, 6, 8])


def test_build_frame_grid_nonpositive_dt_raises():
    try:
        build_frame_grid(0, 10, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("預期 dt<=0 要拋 ValueError")


def test_build_frame_grid_t_end_before_t_start_raises():
    try:
        build_frame_grid(10, 0, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("預期 t_end < t_start 要拋 ValueError")


def _pulse_scenario():
    # neuron 0:真實事件在 t=1,3,5,值 True/False/True(chunk_size=1 保證嚴格遞增)
    # neuron 1:只有一筆真實事件在 t=2,值 True,之後 idle(nan 尾巴)
    event_ms = np.array([[1.0, 3.0, 5.0],
                          [2.0, np.nan, np.nan]])
    values = np.array([[True, False, True],
                        [True, False, False]])
    frame_ms = np.array([0, 1, 2, 3, 4, 5, 6], dtype=float)
    return event_ms, values, frame_ms


def test_resample_pulse_only_lights_up_the_frame_the_event_falls_in():
    event_ms, values, frame_ms = _pulse_scenario()

    out = resample_pulse(event_ms, values, frame_ms, dt=1.0, background=False)

    # 事件跟事件之間(t=2,4,6 這些沒有真實事件的 frame)要是 background,
    # 不能沿用前一筆事件的值(這是跟舊的 resample_hold 最關鍵的行為差異)。
    assert np.array_equal(out[0], [False, True, False, False, False, True, False])
    assert np.array_equal(out[1], [False, False, True, False, False, False, False])


def test_resample_pulse_neuron_with_no_real_events_is_all_background():
    event_ms = np.array([[np.nan, np.nan]])
    values = np.array([[True, False]])
    frame_ms = np.array([0.0, 5.0, 10.0])

    out = resample_pulse(event_ms, values, frame_ms, dt=5.0, background=False)

    assert np.array_equal(out[0], [False, False, False])


def test_resample_pulse_two_events_in_same_bucket_last_one_wins():
    # t=0.2 跟 t=0.7 都落在 [0,1) 這個時間窗裡,較晚的(t=0.7,值 False)應該蓋過
    # 較早的(t=0.2,值 True)。
    event_ms = np.array([[0.2, 0.7]])
    values = np.array([[True, False]])
    frame_ms = np.array([0.0, 1.0])

    out = resample_pulse(event_ms, values, frame_ms, dt=1.0, background=False)

    assert np.array_equal(out[0], [False, False])


def test_resample_decay_matches_hand_computed_values():
    # 單一神經元,一筆真實事件在 t=2、值 v=4.0,tau=10 -> decay=0.9。
    event_ms = np.array([[2.0, np.nan, np.nan]])
    values = np.array([[4.0, np.nan, np.nan]])
    frame_ms = np.array([0.0, 2.0, 3.0, 4.0])

    out = resample_decay(event_ms, values, frame_ms, tau=10.0, before_first=0.0)

    expected = [0.0, 4.0, 4.0 * 0.9, 4.0 * 0.9 ** 2]
    assert np.allclose(out[0], expected)


def test_resample_decay_no_real_events_decays_before_first_from_t_zero():
    event_ms = np.array([[np.nan, np.nan]])
    values = np.array([[np.nan, np.nan]])
    frame_ms = np.array([0.0, 1.0, 2.0])
    decay = 1.0 - 1.0 / 5.0

    out = resample_decay(event_ms, values, frame_ms, tau=5.0, before_first=1.0)

    assert np.allclose(out[0], 1.0 * decay ** frame_ms)


def test_pad_events_by_neuron_groups_and_sorts_per_neuron():
    # 扁平事件(不按神經元分好、也不按時間排):neuron 1 在 t=5, neuron 0 在
    # t=3, neuron 0 在 t=1, neuron 1 在 t=2。neuron 0 應該變成 [1,3]、
    # neuron 1 應該變成 [2,5](組內按時間遞增,不是照原始出現順序)。
    neuron_idx = np.array([1, 0, 0, 1])
    event_ms = np.array([5.0, 3.0, 1.0, 2.0])

    out = pad_events_by_neuron(neuron_idx, event_ms, n_neurons=2)

    assert out.shape == (2, 2)
    assert np.array_equal(out[0], [1.0, 3.0])
    assert np.array_equal(out[1], [2.0, 5.0])


def test_pad_events_by_neuron_pads_shorter_neurons_with_nan():
    # neuron 0 有 2 筆事件,neuron 1 只有 1 筆——neuron 1 該用 nan 補到跟
    # neuron 0 一樣長,不能悄悄補 0 或別的數字。
    neuron_idx = np.array([0, 0, 1])
    event_ms = np.array([1.0, 2.0, 3.0])

    out = pad_events_by_neuron(neuron_idx, event_ms, n_neurons=2)

    assert np.array_equal(out[0], [1.0, 2.0])
    assert out[1, 0] == 3.0
    assert np.isnan(out[1, 1])


def test_pad_events_by_neuron_neuron_with_no_events_is_all_nan():
    neuron_idx = np.array([0])
    event_ms = np.array([1.0])

    out = pad_events_by_neuron(neuron_idx, event_ms, n_neurons=3)

    assert np.isnan(out[1]).all()
    assert np.isnan(out[2]).all()


def test_pad_events_by_neuron_no_events_at_all_returns_empty_columns():
    out = pad_events_by_neuron(np.array([], dtype=int), np.array([]), n_neurons=2)

    assert out.shape == (2, 0)


def test_sliding_windows_fully_inside_bounds():
    # 2 列,每列 10 欄,值 0..9(row0)/10..19(row1)。
    array = np.array([np.arange(10), np.arange(10, 20)])

    out = sliding_windows(array, center_indices=[5], half_width=2, pad_value=np.nan)

    assert out.shape == (1, 2, 5)
    assert np.array_equal(out[0], array[:, 3:8])


def test_sliding_windows_left_boundary_pads():
    array = np.array([np.arange(10), np.arange(10, 20)])

    out = sliding_windows(array, center_indices=[1], half_width=2, pad_value=np.nan)

    # center=1, half_width=2 -> 理論窗口是欄 [-1..3],左邊那格(index -1)不存在。
    assert np.isnan(out[0, :, 0]).all()
    assert np.array_equal(out[0, :, 1:], array[:, 0:4])


def test_sliding_windows_right_boundary_pads():
    array = np.array([np.arange(10), np.arange(10, 20)])

    out = sliding_windows(array, center_indices=[9], half_width=2, pad_value=np.nan)

    # center=9(最後一欄), 理論窗口是欄 [7..11],右邊 10、11 不存在。
    assert np.array_equal(out[0, :, 0:3], array[:, 7:10])
    assert np.isnan(out[0, :, 3:]).all()


def test_sliding_windows_multiple_centers_stacked():
    array = np.array([np.arange(10)])

    out = sliding_windows(array, center_indices=[2, 5], half_width=1, pad_value=-1)

    assert out.shape == (2, 1, 3)
    assert np.array_equal(out[0, 0], [1, 2, 3])
    assert np.array_equal(out[1, 0], [4, 5, 6])


def test_sliding_windows_bool_array_with_nan_pad_promotes_dtype():
    array = np.array([[True, False, True]])

    out = sliding_windows(array, center_indices=[0], half_width=1, pad_value=np.nan)

    assert np.isnan(out[0, 0, 0])
    assert out[0, 0, 1] == 1.0
    assert out[0, 0, 2] == 0.0
