"""quant.convert:浮點權重 + 量化規格 -> QuantizedLayerParams。"""
import jax.numpy as jnp
import numpy as np
import pytest

from salt_core.layer_chain import EventStream
from salt_core.layers import ConvLayer, FCLayer, run_network_quantized
from salt_core.quant.convert import (LayerQuantSpec, build_quantized_params, iv_per_channel,
                                     weight_codes)
from salt_core.quantize import build_decay_table_int

# conv:2 channel、每 channel 2x2 = 4 顆神經元;FC:8 -> 2
CONV = ConvLayer(name="conv", ic=2, h_in=2, w_in=2, oc=2, k=1, s=1, p=0, init_k=5.0,
                 tau=16.0, v_th=1.0, chunk_size=1, L=8, max_out_spikes=64)
FC = FCLayer(name="out", n_in=8, n_out=2, init_k=5.0, tau=16.0)
# channel 0 = [0.375, 0.125],channel 1 = [-1.0, 0.25]
CONV_W = jnp.array([0.375, 0.125, -1.0, 0.25]).reshape(2, 2, 1, 1)
FC_W = jnp.linspace(-0.5, 0.3, 16).reshape(2, 8)


def test_weight_codes_per_channel_matches_hand_computation():
    # bits=4 -> 最大碼 7
    # channel 0: s_c = 0.375/7,q = [7, 0.125/s_c = 2.33 -> 2]
    # channel 1: s_c = 1/7,q = [-7, 0.25/s_c = 1.75 -> 2]
    q, scale = weight_codes(CONV_W, LayerQuantSpec(bits=4, f_a=8, f_V=4))

    assert q.dtype == jnp.int32
    assert np.array_equal(np.asarray(q).reshape(2, 2), [[7, 2], [-7, 2]])
    assert np.allclose(np.asarray(scale), [0.375 / 7, 1 / 7])


def test_weight_codes_per_tensor_shares_one_scale():
    # 整層 max|w| = 1.0 -> s_c = 1/7,channel 0: [2.625 -> 3, 0.875 -> 1]
    q, scale = weight_codes(CONV_W, LayerQuantSpec(bits=4, f_a=8, f_V=4, per_channel=False))

    assert np.array_equal(np.asarray(q).reshape(2, 2), [[3, 1], [-7, 2]])
    assert np.allclose(np.asarray(scale), [1 / 7, 1 / 7])


def test_weight_codes_clip_percentile_lowers_threshold():
    # channel 1 的 |w| = [1.0, 0.25],第 50 百分位(線性內插)= 0.625
    _q, scale = weight_codes(CONV_W, LayerQuantSpec(bits=4, f_a=8, f_V=4, clip_percentile=50.0))

    assert float(scale[1]) == pytest.approx(0.625 / 7)


def test_iv_per_channel_matches_hand_computation():
    # i_V = floor(log2(M * 7 / T_c)) + 2,T_c = s_c * 7
    # channel 0: T_c = 0.375, x = 2.0 * 7 / 0.375 = 37.3 -> 5 + 2 = 7
    # channel 1: T_c = 1.0,   x = 0.5 * 7 = 3.5       -> 1 + 2 = 3
    assert iv_per_channel(np.array([2.0, 0.5]), jnp.array([0.375 / 7, 1 / 7]), bits=4) == [7, 3]


def test_iv_per_channel_length_mismatch_raises():
    with pytest.raises(ValueError):
        iv_per_channel(np.array([2.0, 0.5, 1.0]), jnp.array([0.1, 0.2]), bits=4)


def _build(fc_spec):
    specs = [LayerQuantSpec(bits=4, f_a=8, f_V=4), fc_spec]
    v_abs_max = [np.array([2.0, 0.5]), np.array([3.0, 1.0])]
    return build_quantized_params([CONV, FC], (CONV_W, FC_W), specs, v_abs_max)


def test_build_quantized_params_conv_layer():
    conv_params, _ = _build(LayerQuantSpec(bits=4, f_a=8, f_V=4, fires=False))

    # i_V 取兩個 channel 裡大的那個:max(7, 3)
    assert conv_params.i_V == 7
    # 每個 channel 的 s_c 給它底下 4 顆神經元
    assert np.allclose(np.asarray(conv_params.scale), [0.375 / 7] * 4 + [1 / 7] * 4)
    # 門檻 round(v_th / s_c * 2^f_V):channel 0: 1 / (0.375/7) * 16 = 298.67 -> 299
    #                               channel 1: 1 / (1/7) * 16 = 112
    assert np.array_equal(np.asarray(conv_params.v_th_int), [299] * 4 + [112] * 4)
    assert np.array_equal(np.asarray(conv_params.decay_table_int),
                          np.asarray(build_decay_table_int(8, 16.0)))
    assert (conv_params.f_a, conv_params.f_V) == (8, 4)


def test_build_quantized_params_non_firing_per_tensor_layer():
    _, fc_params = _build(LayerQuantSpec(bits=4, f_a=8, f_V=4, per_channel=False, fires=False))

    assert fc_params.v_th_int is None
    scale = np.asarray(fc_params.scale)
    assert scale.shape == (2,) and scale[0] == scale[1]


def test_build_quantized_params_length_mismatch_raises():
    with pytest.raises(ValueError):
        build_quantized_params([CONV, FC], (CONV_W, FC_W), [LayerQuantSpec(4, 8, 4)],
                               [np.array([2.0, 0.5]), np.array([3.0, 1.0])])


def test_build_quantized_params_output_runs_through_quantized_network():
    params = _build(LayerQuantSpec(bits=4, f_a=8, f_V=4, fires=False))
    n = 4
    in_stream = EventStream(event_times=jnp.arange(1.0, n + 1.0),
                            event_source_idx=jnp.arange(n),
                            event_gain=jnp.ones((n,)), n_real_events=jnp.array(n))

    readout, diags = run_network_quantized([CONV, FC], in_stream, params)

    assert readout.v_final.shape == (2,)
    assert not any(bool(d.queue_truncated) or bool(d.output_truncated) for d in diags)
