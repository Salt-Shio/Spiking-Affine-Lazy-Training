"""salt_core/layers.py 的 max_over_batch、grown_to_fit_batch。"""
import jax.numpy as jnp

from salt_core.layers import ConvLayer, FCLayer, LayerDiag, grown_to_fit_batch, max_over_batch

CONV = ConvLayer(name="conv", ic=1, h_in=4, w_in=4, oc=1, k=3, s=1, p=1, init_k=1.0,
                 L=10, max_out_spikes=20, max_steps=10)
FC = FCLayer(name="out", n_in=16, n_out=2, init_k=1.0)


def _diag(max_real_queue, n_out_spikes, min_steps_needed) -> LayerDiag:
    """三筆樣本的逐筆診斷。"""
    zeros = jnp.zeros(3)
    return LayerDiag(spike_count=zeros, firing_rate=zeros,
                     max_real_queue=jnp.array(max_real_queue),
                     n_out_spikes=jnp.array(n_out_spikes),
                     min_steps_needed=jnp.array(min_steps_needed))


def test_max_over_batch_takes_each_field_max():
    reduced = max_over_batch(_diag([3, 9, 5], [1, 2, 7], [4, 4, 6]))
    assert int(reduced.max_real_queue) == 9
    assert int(reduced.n_out_spikes) == 7
    assert int(reduced.min_steps_needed) == 6


def test_fits_returns_same_list_object():
    layers = [CONV, FC]
    fits = [_diag([10, 3, 3], [20, 1, 1], [10, 2, 2]), _diag([0, 0, 0], [0, 0, 0], [1, 1, 1])]
    assert grown_to_fit_batch(layers, fits) is layers


def test_one_overflowing_sample_grows_layer():
    # 只有第二筆超過 L=10;放大公式 ceil(max(11, 10) * 1.5) = 17,L 放大時 max_steps 跟著變成新 L
    layers = [CONV, FC]
    diags = [_diag([2, 11, 2], [1, 1, 1], [1, 1, 1]), _diag([0, 0, 0], [0, 0, 0], [1, 1, 1])]
    grown = grown_to_fit_batch(layers, diags)
    assert grown is not layers
    assert (grown[0].L, grown[0].max_out_spikes, grown[0].max_steps) == (17, 20, 17)
    assert grown[1] is FC
