"""salt_core/fpga_export.py:hex 行格式、權重寬字打包、門檻逐 channel 化、範圍檢查、整層寫檔。"""
import numpy as np
import pytest

from salt_core.fpga_export import (channel_thresholds, conv_weight_lines, decay_lines, hex_line,
                                   threshold_lines, write_conv_layer_mem)
from salt_core.layers import ConvLayer
from salt_core.quant.params import QuantizedLayerParams


def test_hex_line_fixed_digits_left_padded():
    # 7 位元 -> ceil(7/4) = 2 個字元;9 位元 -> 3 個字元
    assert hex_line(5, 7) == "05"
    assert hex_line(0x1fd, 9) == "1fd"
    assert hex_line(0, 1) == "0"


@pytest.mark.parametrize("value, width", [(128, 7), (-1, 7), (0, 0)])
def test_hex_line_out_of_range_raises(value, width):
    with pytest.raises(ValueError):
        hex_line(value, width)


def test_weight_word_tap_order_lowest_bit_first():
    # K=2, b=4,tap0..3 = 3, -1, 0, -7
    # 4 位元二補數:3 -> 0011, -1 -> 1111, 0 -> 0000, -7 -> 1001
    # tap0 在最低位,由高到低是 tap3 tap2 tap1 tap0 = 1001 0000 1111 0011 = 90f3
    q = np.array([3, -1, 0, -7]).reshape(1, 1, 2, 2)
    assert conv_weight_lines(q, 4) == ["90f3"]


def test_weight_address_is_out_channel_times_in_channels_plus_in_channel():
    # K=1、C=3(不是 2 的次方),q[o_c, c] = 10*o_c + c,一行一個碼,照 o_c*3 + c 排
    q = (10 * np.arange(2)[:, None] + np.arange(3)[None, :]).reshape(2, 3, 1, 1)
    assert conv_weight_lines(q, 7) == ["00", "01", "02", "0a", "0b", "0c"]


def test_weight_word_width_not_multiple_of_four():
    # K=3, b=7,寬字 63 位元 -> 16 個字元,最高位元是補的 0
    # 全部 -1:63 個 1 = 0x7fffffffffffffff
    assert conv_weight_lines(np.full((1, 1, 3, 3), -1), 7) == ["7fffffffffffffff"]
    # 只有 tap8 = -63:7 位元二補數 1000001 = 0x41,放在第 8*7 = 56 位元起 -> 0x41 << 56
    q = np.zeros((1, 1, 3, 3), dtype=np.int32)
    q[0, 0, 2, 2] = -63
    assert conv_weight_lines(q, 7) == ["4100000000000000"]


def test_weight_most_negative_code_fits():
    # 7 位元二補數下限 -64 -> 1000000 = 40
    assert conv_weight_lines(np.full((1, 1, 1, 1), -64), 7) == ["40"]


@pytest.mark.parametrize("code", [64, -65])
def test_weight_out_of_range_raises(code):
    with pytest.raises(ValueError):
        conv_weight_lines(np.full((1, 1, 1, 1), code), 7)


@pytest.mark.parametrize("shape", [(1, 1, 3), (1, 1, 3, 2)])
def test_weight_bad_shape_raises(shape):
    with pytest.raises(ValueError):
        conv_weight_lines(np.zeros(shape, dtype=np.int32), 7)


def test_channel_thresholds_from_per_neuron():
    # 2 個 output channel,每個 3 顆神經元
    assert channel_thresholds(np.array([5, 5, 5, -3, -3, -3]), 2).tolist() == [5, -3]
    assert channel_thresholds(np.array(7), 3).tolist() == [7, 7, 7]


def test_channel_thresholds_not_uniform_raises():
    with pytest.raises(ValueError, match=r"\[1\]"):
        channel_thresholds(np.array([5, 5, 5, -3, -2, -3]), 2)


def test_channel_thresholds_length_not_divisible_raises():
    with pytest.raises(ValueError):
        channel_thresholds(np.array([5, 5, 5]), 2)


def test_threshold_lines_twos_complement():
    # 9 位元:5 -> 005;-3 -> 512 - 3 = 509 = 1fd
    assert threshold_lines([5, -3], 9) == ["005", "1fd"]


@pytest.mark.parametrize("value", [256, -257])
def test_threshold_out_of_range_raises(value):
    with pytest.raises(ValueError):
        threshold_lines([value], 9)


def test_decay_lines_unsigned():
    assert decay_lines([60, 56, 1], 6) == ["3c", "38", "01"]


@pytest.mark.parametrize("value", [64, -1])
def test_decay_out_of_range_raises(value):
    with pytest.raises(ValueError):
        decay_lines([value], 6)


# 2 個輸入 channel、5x5 -> k=3 s=2 p=1 -> 3x3 輸出,3 個 output channel
LAYER = ConvLayer(name="conv1", ic=2, h_in=5, w_in=5, oc=3, k=3, s=2, p=1, init_k=5.0)


def _layer_params(**changes) -> QuantizedLayerParams:
    rng = np.random.default_rng(0)
    params = QuantizedLayerParams(
        q=rng.integers(-63, 64, size=(3, 2, 3, 3)),
        decay_table_int=np.array([60, 56, 53]),
        v_th_int=np.repeat(np.array([24, 35, 63]), 3 * 3),
        scale=np.ones(3 * 3 * 3, dtype=np.float32),
        f_a=6, f_V=0, i_V=9)
    return params._replace(**changes)


def test_write_conv_layer_mem_three_files(tmp_path):
    params = _layer_params()
    paths = write_conv_layer_mem(tmp_path, LAYER, params, 7)

    assert paths == {kind: tmp_path / f"conv1_{kind}.mem" for kind in ("weight", "threshold", "decay")}
    assert paths["weight"].read_text().splitlines() == conv_weight_lines(params.q, 7)
    assert paths["threshold"].read_text() == "018\n023\n03f\n"
    assert paths["decay"].read_text() == "3c\n38\n35\n"


@pytest.mark.parametrize("changes", [
    {"q": np.zeros((3, 2, 2, 2), dtype=np.int32)},     # k 對不上
    {"v_th_int": None},                                 # 不 fire
    {"v_th_int": np.full(3 * 2 * 2, 24)},               # 神經元數對不上
])
def test_write_conv_layer_mem_bad_params_raises_without_writing(tmp_path, changes):
    with pytest.raises(ValueError):
        write_conv_layer_mem(tmp_path, LAYER, _layer_params(**changes), 7)
    assert list(tmp_path.iterdir()) == []
