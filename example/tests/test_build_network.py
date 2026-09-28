"""build_network:layer entry 之間的形狀往下傳。"""
import pytest

from example.models.conv_net import build_network

_CONV = {"type": "conv", "oc": 4, "k": 3, "s": 2, "p": 1, "init_k": 5.0}


def test_fc_after_fc_takes_previous_n_out_as_n_in():
    network = build_network({"input_shape": [2, 8, 8], "layers": [
        _CONV, {"type": "fc", "n_out": 8, "init_k": 5.0}, {"type": "fc", "n_out": 10, "init_k": 5.0}]})
    conv, fc1, fc2 = network.layers
    assert fc1.n_in == conv.n_neurons == 4 * 4 * 4
    assert fc2.n_in == fc1.n_out == 8


def test_conv_after_fc_is_rejected():
    with pytest.raises(ValueError, match="conv 要空間輸入"):
        build_network({"input_shape": [2, 8, 8], "layers": [
            {"type": "fc", "n_out": 8, "init_k": 5.0}, _CONV]})
