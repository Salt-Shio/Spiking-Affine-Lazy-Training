"""salt_core/io.py:網路描述 dict 來回、權重連同網路描述的存讀、格式不對時 raise。"""
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from salt_core.io import (NETWORK_KEY, load_weights, network_from_dict, network_to_dict,
                          save_weights)
from salt_core.layers import ConvLayer, FCLayer
from salt_core.network import Network

# 容量、chunk_size、動力學都用非預設值,確認每個欄位都有存到
NETWORK = Network(input_shape=(2, 8, 8), layers=(
    ConvLayer(name="conv1", ic=2, h_in=8, w_in=8, oc=3, k=3, s=2, p=1, init_k=5.0,
              tau=12.0, v_th=0.8, alpha=3.0, chunk_size=4,
              max_queue_len=17, max_out_spikes=40, max_steps=9),
    FCLayer(name="out", n_in=3 * 4 * 4, n_out=5, init_k=2.0, tau=20.0, chunk_size=8)))


def test_dict_roundtrip():
    description = network_to_dict(NETWORK)
    assert description["format"] == 1
    assert [entry["type"] for entry in description["layers"]] == ["conv", "fc"]
    # 可以直接存成 json
    assert network_from_dict(json.loads(json.dumps(description))) == NETWORK


def test_weights_roundtrip_bit_exact_as_jax_arrays(tmp_path):
    weights = NETWORK.init(jax.random.PRNGKey(0))
    path = tmp_path / "w.npz"
    save_weights(path, NETWORK, weights)
    network, loaded = load_weights(path)
    assert network == NETWORK
    for saved, read in zip(weights, loaded):
        assert isinstance(read, jax.Array)
        assert jnp.array_equal(saved, read)


def test_npz_keys_are_layer_names_plus_network(tmp_path):
    path = tmp_path / "w.npz"
    save_weights(path, NETWORK, NETWORK.init(jax.random.PRNGKey(0)))
    with np.load(path) as data:
        assert sorted(data.files) == sorted(["conv1", "out", NETWORK_KEY])


def test_unknown_format_raises():
    description = network_to_dict(NETWORK)
    description["format"] = 2
    with pytest.raises(ValueError, match="format"):
        network_from_dict(description)


def test_unknown_layer_type_raises():
    description = network_to_dict(NETWORK)
    description["layers"][1]["type"] = "pool"
    with pytest.raises(ValueError, match="pool"):
        network_from_dict(description)


def test_layer_named_like_network_key_raises(tmp_path):
    fc = FCLayer(name=NETWORK_KEY, n_in=2 * 8 * 8, n_out=5, init_k=2.0)
    network = Network(input_shape=(2, 8, 8), layers=(fc,))
    with pytest.raises(ValueError, match=NETWORK_KEY):
        save_weights(tmp_path / "w.npz", network, network.init(jax.random.PRNGKey(0)))


def test_file_without_network_raises(tmp_path):
    """只有權重、沒有網路描述的舊格式檔。"""
    path = tmp_path / "old.npz"
    np.savez(path, out=np.zeros((5, 128), dtype=np.float32))
    with pytest.raises(ValueError, match=NETWORK_KEY):
        load_weights(path)
