"""example/models/conv_net.py:layer entry 的分組、layer_defaults 合併、形狀往下傳、growth 組。"""
import pytest

from example.models.conv_net import build_growth_policies, build_network
from salt_core.capacity import GrowthPolicy

_CONV = {"type": "conv", "oc": 4, "k": 3, "s": 2, "p": 1}
_DEFAULTS = {"training": {"init_k": 5.0}}


def _model(layers: list, defaults: dict = _DEFAULTS) -> dict:
    return {"input_shape": [2, 8, 8], "layer_defaults": defaults, "layers": layers}


def test_fc_after_fc_takes_previous_n_out_as_n_in():
    network = build_network(_model([_CONV, {"type": "fc", "n_out": 8},
                                    {"type": "fc", "n_out": 10}]))
    conv, fc1, fc2 = network.layers
    assert fc1.n_in == conv.n_neurons == 4 * 4 * 4
    assert fc2.n_in == fc1.n_out == 8


def test_layer_group_overrides_defaults_key_by_key():
    """defaults 的 neuron 有 tau、v_th;層只蓋 v_th,tau 沿用 defaults。yaml 的字串數字要轉型。"""
    defaults = {"neuron": {"tau": 8.0, "v_th": 1.0}, "training": {"init_k": 5.0},
                "capacity": {"chunk_size": 4}}
    out = {"type": "fc", "name": "out", "n_out": 10, "neuron": {"v_th": "1.0e9"},
           "capacity": {"chunk_size": 32}}
    conv, fc = build_network(_model([_CONV, out], defaults)).layers
    assert (conv.tau, conv.v_th, conv.chunk_size) == (8.0, 1.0, 4)
    assert (fc.tau, fc.v_th, fc.chunk_size) == (8.0, 1.0e9, 32)


def test_growth_group_goes_to_policy_not_layer():
    defaults = {"training": {"init_k": 5.0}, "growth": {"out_grow_factor": 2.0}}
    conv = {**_CONV, "growth": {"max_queue_len_grow_factor": 3.0}}
    model = _model([conv], defaults)
    network = build_network(model)
    assert build_growth_policies(model, list(network.layers)) == {
        "conv1": GrowthPolicy(max_queue_len_grow_factor=3.0, out_grow_factor=2.0)}


@pytest.mark.parametrize("layer, message", [
    ({**_CONV, "tau": 16.0}, r"不認得的 key:\['tau'\]"),
    ({**_CONV, "capacity": {"tau": 16.0}}, r"capacity 有不認得的 key:\['tau'\]"),
    ({"type": "conv", "oc": 4, "k": 3, "s": 2}, r"conv 少了 \['p'\]"),
    ({"type": "pool"}, "未知的 type 'pool'"),
])
def test_bad_layer_entry_is_rejected(layer, message):
    with pytest.raises(ValueError, match=message):
        build_network(_model([layer]))


def test_unknown_group_in_defaults_is_rejected():
    with pytest.raises(ValueError, match=r"layer_defaults 只能放.*\['optim'\]"):
        build_network(_model([_CONV], {**_DEFAULTS, "optim": {"lr": 1.0}}))


def test_conv_after_fc_is_rejected():
    with pytest.raises(ValueError, match="conv 要空間輸入"):
        build_network(_model([{"type": "fc", "n_out": 8}, _CONV]))
