"""example/utils.py 的 grid_input_events:網格座標攤平成 source_idx、座標越界 raise。"""
import numpy as np
import pytest

from example.utils import grid_input_events


def test_flattens_row_major():
    """(C,H,W)=(2,3,4):(x,y,c)=(1,2,1) -> 1*12 + 2*4 + 1 = 21;(3,0,0) -> 3。"""
    raw = grid_input_events(np.array([1.0, 2.0]), x=np.array([1, 3]), y=np.array([2, 0]),
                            c=np.array([1, 0]), n_real_events=2, grid_shape=(2, 3, 4))
    assert raw.source_idx.tolist() == [21, 3]


def test_flattens_batch():
    raw = grid_input_events(np.array([[1.0], [2.0]]), x=np.array([[1], [3]]),
                            y=np.array([[2], [0]]), c=np.array([[1], [0]]),
                            n_real_events=np.array([1, 1]), grid_shape=(2, 3, 4))
    assert raw.source_idx.tolist() == [[21], [3]]


def test_rejects_coordinate_outside_grid():
    """x=4 超出 W=4。"""
    with pytest.raises(ValueError):
        grid_input_events(np.array([1.0]), x=np.array([4]), y=np.array([0]), c=np.array([0]),
                          n_real_events=1, grid_shape=(2, 3, 4))
