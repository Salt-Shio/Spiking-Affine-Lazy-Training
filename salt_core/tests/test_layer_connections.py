"""salt_core/layers.py 的 check_layer_connections:相鄰兩層接不上時 raise。"""
from dataclasses import replace

import pytest

from salt_core.dormant import dormant_report
from salt_core.layers import (ConvLayer, FCLayer, check_layer_connections, run_network,
                              run_network_quantized, run_network_quantized_traced,
                              run_network_traced)

# 2x8x8 -> 4x8x8 -> 4x4x4(攤平 64)-> FC 10
CONV1 = ConvLayer(name="conv1", ic=2, h_in=8, w_in=8, oc=4, k=3, s=1, p=1, init_k=1.0)
CONV2 = ConvLayer(name="conv2", ic=4, h_in=8, w_in=8, oc=4, k=3, s=2, p=1, init_k=1.0)
FC1 = FCLayer(name="fc1", n_in=64, n_out=10, init_k=1.0)
FC2 = FCLayer(name="fc2", n_in=10, n_out=3, init_k=1.0)


def test_connected_chain_passes():
    check_layer_connections([CONV1, CONV2, FC1, FC2])
    check_layer_connections([FC1])


def test_conv_channel_mismatch_raises():
    with pytest.raises(ValueError, match="conv1.*conv2"):
        check_layer_connections([CONV1, replace(CONV2, ic=3)])


def test_conv_spatial_mismatch_raises():
    with pytest.raises(ValueError, match="conv1.*conv2"):
        check_layer_connections([CONV1, replace(CONV2, h_in=9)])


def test_conv_to_fc_size_mismatch_raises():
    with pytest.raises(ValueError, match="conv2.*fc1"):
        check_layer_connections([CONV1, CONV2, replace(FC1, n_in=63)])


def test_fc_to_fc_size_mismatch_raises():
    with pytest.raises(ValueError, match="fc1.*fc2"):
        check_layer_connections([FC1, replace(FC2, n_in=11)])


def test_conv_after_fc_raises():
    fc = FCLayer(name="fc", n_in=4, n_out=2 * 8 * 8, init_k=1.0)
    with pytest.raises(ValueError, match="沒有空間形狀"):
        check_layer_connections([fc, CONV1])


@pytest.mark.parametrize("run", [run_network, run_network_traced, run_network_quantized,
                                 run_network_quantized_traced])
def test_network_entry_points_check_connections(run):
    # 檢查在碰到輸入之前就 raise,輸入跟權重用不到
    with pytest.raises(ValueError, match="conv1.*conv2"):
        run([CONV1, replace(CONV2, ic=3)], None, None)


def test_dormant_report_checks_connections():
    with pytest.raises(ValueError, match="conv1.*conv2"):
        dormant_report([CONV1, replace(CONV2, ic=3)], None, None, None)
