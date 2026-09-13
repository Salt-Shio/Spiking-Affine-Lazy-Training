"""viz/time_resample.py 的單元測試(合成陣列,不需要訓練 run)。"""
import numpy as np

from viz.time_resample import build_frame_grid, resample_decay, resample_pulse


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


TESTS = [
    test_build_frame_grid_exact_multiple,
    test_build_frame_grid_last_frame_not_exceeding_t_end,
    test_build_frame_grid_nonpositive_dt_raises,
    test_build_frame_grid_t_end_before_t_start_raises,
    test_resample_pulse_only_lights_up_the_frame_the_event_falls_in,
    test_resample_pulse_neuron_with_no_real_events_is_all_background,
    test_resample_pulse_two_events_in_same_bucket_last_one_wins,
    test_resample_decay_matches_hand_computed_values,
    test_resample_decay_no_real_events_decays_before_first_from_t_zero,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n全部 {len(TESTS)} 項通過")
