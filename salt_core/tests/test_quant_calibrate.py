"""quant.calibrate:軌跡 -> 逐 channel 的膜電位範圍。"""
import dataclasses

import numpy as np
import pytest

from salt_core.layers import ConvLayer, FCLayer
from salt_core.monitor import LayerForwardTrace
from salt_core.quant.calibrate import merge_v_ranges, v_abs_max_per_channel, v_range_per_channel

# conv:2 channel x 1 x 2 = 4 顆神經元;FC:3 顆
CONV = ConvLayer(name="conv", ic=1, h_in=1, w_in=2, oc=2, k=1, s=1, p=0, init_k=5.0,
                 chunk_size=1, max_queue_len=3)
FC = FCLayer(name="out", n_in=4, n_out=3, init_k=5.0)


def _trace(v_steps):
    v_steps = np.asarray(v_steps, dtype=np.float32)
    return LayerForwardTrace(spike_mask=np.zeros(v_steps.shape, dtype=bool), v_steps=v_steps,
                             event_ms=np.zeros(v_steps.shape))


# 每列一顆神經元、每欄一步;前兩列是 channel 0,後兩列是 channel 1
CONV_STEPS = [[0.1, 0.4, 0.2],
              [-0.3, 0.0, 0.0],
              [0.5, 0.9, -0.1],
              [0.2, 0.2, 0.2]]
FC_STEPS = [[1.0, 2.0, 3.0],
            [-4.0, -1.0, 0.0],
            [0.0, 0.0, 0.0]]


def test_v_range_per_channel_matches_hand_computation():
    ranges = v_range_per_channel([CONV, FC], [_trace(CONV_STEPS), _trace(FC_STEPS)])

    (conv_max, conv_min), (fc_max, fc_min) = ranges
    # channel 0 = 前兩列:max 0.4、min -0.3;channel 1 = 後兩列:max 0.9、min -0.1
    assert np.allclose(conv_max, [0.4, 0.9]) and np.allclose(conv_min, [-0.3, -0.1])
    assert np.allclose(fc_max, [3.0, 0.0, 0.0]) and np.allclose(fc_min, [1.0, -4.0, 0.0])


def test_v_range_per_channel_rejects_chunk_size_above_one():
    conv = dataclasses.replace(CONV, chunk_size=2)
    with pytest.raises(ValueError):
        v_range_per_channel([conv, FC], [_trace(CONV_STEPS), _trace(FC_STEPS)])


def test_merge_takes_max_of_max_and_min_of_min():
    sample_a = [(np.array([1.0, 2.0]), np.array([-1.0, 0.5]))]
    sample_b = [(np.array([3.0, 1.0]), np.array([0.0, -2.0]))]

    [(v_max, v_min)] = merge_v_ranges([sample_a, sample_b])

    assert np.array_equal(v_max, [3.0, 2.0])
    assert np.array_equal(v_min, [-1.0, -2.0])


def test_merge_rejects_empty_input():
    with pytest.raises(ValueError):
        merge_v_ranges([])


def test_v_abs_max_is_larger_of_max_and_abs_min():
    [m] = v_abs_max_per_channel([(np.array([3.0, 1.0]), np.array([-1.0, -2.0]))])
    assert np.array_equal(m, [3.0, 2.0])
