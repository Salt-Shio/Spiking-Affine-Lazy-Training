"""salt_core/network.py 的 Network.build:層描述接形狀、預設命名、接不上時 raise;ConvLayer 幾何退化。"""
import pytest

from salt_core.layers import ConvLayer, FCLayer, conv, fc
from salt_core.network import Network


def test_build_fills_input_side_from_previous_layer():
    """2x8x8 -> conv(s=1) 4x8x8 -> conv(s=2) 4x4x4 -> FC n_in = 64 -> FC n_in = 10。"""
    network = Network.build((2, 8, 8), [conv(4, k=3, s=1, p=1, init_k=1.0),
                                        conv(4, k=3, s=2, p=1, init_k=1.0, tau=8.0),
                                        fc(10, init_k=1.0), fc(3, init_k=1.0)])
    assert network.layers == (
        ConvLayer(name="conv1", ic=2, h_in=8, w_in=8, oc=4, k=3, s=1, p=1, init_k=1.0),
        ConvLayer(name="conv2", ic=4, h_in=8, w_in=8, oc=4, k=3, s=2, p=1, init_k=1.0, tau=8.0),
        FCLayer(name="fc1", n_in=64, n_out=10, init_k=1.0),
        FCLayer(name="fc2", n_in=10, n_out=3, init_k=1.0))


def test_build_numbers_unnamed_layers_per_prefix():
    """有名字的層也佔一個編號:conv、fc(name="out")、fc -> conv1、out、fc2。"""
    network = Network.build((2, 8, 8), [conv(4, k=3, s=2, p=1, init_k=1.0),
                                        fc(8, name="out", init_k=1.0), fc(3, init_k=1.0)])
    assert [layer.name for layer in network.layers] == ["conv1", "out", "fc2"]


def test_build_accepts_flat_input_shape():
    network = Network.build([24], [fc(2, init_k=1.0)])
    assert network.input_shape == (24,)
    assert network.layers[0].n_in == 24


def test_conv_after_fc_is_rejected():
    with pytest.raises(ValueError, match="conv 要空間輸入"):
        Network.build((2, 8, 8), [fc(8, init_k=1.0), conv(4, k=3, s=2, p=1, init_k=1.0)])


def test_empty_specs_is_rejected():
    with pytest.raises(ValueError, match="layers 不能是空的"):
        Network.build((2, 8, 8), [])


def test_degenerate_conv_geometry_is_rejected():
    """輸入 2x2、k=5、p=0:h_out = (2 - 5) // 1 + 1 = -2。"""
    with pytest.raises(ValueError, match="幾何退化"):
        ConvLayer(name="conv1", ic=1, h_in=2, w_in=2, oc=1, k=5, s=1, p=0, init_k=1.0)
