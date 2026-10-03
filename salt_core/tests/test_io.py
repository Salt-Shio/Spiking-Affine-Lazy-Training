"""salt_core/io.py:網路描述 dict 來回、權重(浮點、量化)連同網路描述的存讀、格式不對時 raise。"""
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from salt_core.io import (NETWORK_KEY, QUANT_KEY, load_quantized, load_weights,
                          network_from_dict, network_to_dict, save_quantized, save_weights)
from salt_core.layers import ConvLayer, FCLayer
from salt_core.network import InputEvents, Network
from salt_core.quant.backend import QuantBackend
from salt_core.quant.convert import LayerQuantSpec, build_quantized_params
from salt_core.quant.fixed_point import RoundMode

# 容量、chunk_size、動力學都用非預設值,確認每個欄位都有存到
NETWORK = Network(input_shape=(2, 8, 8), layers=(
    ConvLayer(name="conv1", ic=2, h_in=8, w_in=8, oc=3, k=3, s=2, p=1, init_k=5.0,
              tau=12.0, v_th=0.8, alpha=3.0, chunk_size=4,
              max_queue_len=17, max_out_spikes=40, max_extra_steps=9),
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


# ============================================================================
# 量化模型
# ============================================================================

METADATA = {"spec": {"bits": 4}, "source": "unit-test", "v_abs_max": [[2.0, 2.0, 2.0], [2.0] * 5]}


def _quant_params():
    """NETWORK 的量化參數:conv1 會 fire,out 不 fire(v_th_int 是 None)。M 全部 2.0。"""
    spec = LayerQuantSpec(bits=4, f_a=8, f_V=6)
    v_abs_max = [np.full(3, 2.0), np.full(5, 2.0)]
    return build_quantized_params(NETWORK.layers, NETWORK.init(jax.random.PRNGKey(0)),
                                  [spec, spec._replace(fires=False)], v_abs_max)


def _input_events():
    """2x8x8 網格上 12 筆事件,時間 1..12 ms,座標照編號輪流。"""
    source_idx = jnp.arange(12, dtype=jnp.int32) * 11 % (2 * 8 * 8)
    return InputEvents.checked(jnp.arange(1, 13, dtype=jnp.float32), source_idx, 12)


def test_quantized_roundtrip_every_field_equal(tmp_path):
    params = _quant_params()
    path = tmp_path / "q.npz"
    save_quantized(path, NETWORK, params, "truncate", METADATA)
    model = load_quantized(path)

    assert model.network == NETWORK
    assert model.round_mode is RoundMode.TRUNCATE
    assert model.metadata == METADATA
    for saved, read in zip(params, model.params):
        for field in ("q", "decay_table_int", "scale"):
            assert getattr(read, field).dtype == getattr(saved, field).dtype
            assert jnp.array_equal(getattr(saved, field), getattr(read, field))
        assert (read.f_a, read.f_V, read.i_V, read.overflow_mode) == \
            (saved.f_a, saved.f_V, saved.i_V, saved.overflow_mode)
    assert jnp.array_equal(params[0].v_th_int, model.params[0].v_th_int)
    assert model.params[1].v_th_int is None


def test_quantized_roundtrip_forward_bit_exact(tmp_path):
    params = _quant_params()
    path = tmp_path / "q.npz"
    save_quantized(path, NETWORK, params, "round", METADATA)
    model = load_quantized(path)
    raw = _input_events()

    before = NETWORK.apply(params, raw, backend=QuantBackend(round_mode="round"))
    after = model.network.apply(model.params, raw, backend=QuantBackend(model.round_mode))
    for r_before, r_after in zip(before.results, after.results):
        assert jnp.array_equal(r_before.v_final, r_after.v_final)
        assert jnp.array_equal(r_before.spike_mask, r_after.spike_mask)


def test_quantized_npz_keys(tmp_path):
    path = tmp_path / "q.npz"
    save_quantized(path, NETWORK, _quant_params(), "round", METADATA)
    with np.load(path) as data:
        assert sorted(data.files) == sorted([
            NETWORK_KEY, QUANT_KEY,
            "conv1/q", "conv1/decay_table_int", "conv1/v_th_int", "conv1/scale",
            "out/q", "out/decay_table_int", "out/scale"])


def test_quantized_param_count_mismatch_raises(tmp_path):
    with pytest.raises(ValueError, match="量化參數"):
        save_quantized(tmp_path / "q.npz", NETWORK, _quant_params()[:1], "round", METADATA)


def test_load_quantized_from_float_weights_file_raises(tmp_path):
    path = tmp_path / "w.npz"
    save_weights(path, NETWORK, NETWORK.init(jax.random.PRNGKey(0)))
    with pytest.raises(ValueError, match=QUANT_KEY):
        load_quantized(path)
