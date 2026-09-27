"""salt_core/capacity.py 的 Capacity、reduce_over_batch,跟 layers.py 的 grown_to_fit_batch。"""
import jax.numpy as jnp
import numpy as np
import pytest

from salt_core.capacity import Capacity, LayerDiag, reduce_over_batch
from salt_core.layers import ConvLayer, FCLayer, grown_to_fit_batch

CONV = ConvLayer(name="conv", ic=1, h_in=4, w_in=4, oc=1, k=3, s=1, p=1, init_k=1.0,
                 L=10, max_out_spikes=20, max_steps=10)
FC = FCLayer(name="out", n_in=16, n_out=2, init_k=1.0)


def _diag(queue, out_spikes, steps, spike_count=(0, 0, 0)) -> LayerDiag:
    """三筆樣本的逐筆診斷。"""
    return LayerDiag(spike_count=jnp.array(spike_count, dtype=jnp.float32),
                     firing_rate=jnp.zeros(3),
                     needed={"L": jnp.array(queue), "max_out_spikes": jnp.array(out_spikes),
                             "max_steps": jnp.array(steps)})


def _fc_diag() -> LayerDiag:
    return LayerDiag(spike_count=jnp.zeros(3), firing_rate=jnp.zeros(3), needed={})


def test_reduce_over_batch_takes_max_of_needed_and_mean_of_stats():
    reduced = reduce_over_batch(_diag([3, 9, 5], [1, 2, 7], [4, 4, 6], spike_count=[100, 200, 300]))
    assert {k: int(v) for k, v in reduced.needed.items()} == {"L": 9, "max_out_spikes": 7,
                                                              "max_steps": 6}
    assert float(reduced.spike_count) == 200.0


def test_conv_capacity_matches_fields_and_with_capacity_replaces_them():
    assert dict(CONV.capacity) == {"L": 10, "max_out_spikes": 20, "max_steps": 10}
    grown = CONV.with_capacity(CONV.capacity.replace(L=30))
    assert (grown.L, grown.max_out_spikes, grown.max_steps) == (30, 20, 10)


def test_fc_has_no_capacity():
    assert FC.capacity is None


def test_capacity_replace_rejects_unknown_knob():
    with pytest.raises(KeyError):
        CONV.capacity.replace(queue=5)


def test_fits_checks_every_knob_per_sample():
    # 第二筆 L 需要 11 > 10,第三筆 max_steps 需要 12 > 10
    fits = CONV.capacity.fits(_diag([10, 11, 3], [20, 1, 1], [10, 2, 12]))
    assert np.array_equal(np.asarray(fits), [True, False, False])


def test_fits_returns_same_list_object():
    layers = [CONV, FC]
    diags = [_diag([10, 3, 3], [20, 1, 1], [10, 2, 2]), _fc_diag()]
    assert grown_to_fit_batch(layers, diags) is layers


def test_one_overflowing_sample_grows_layer():
    # 只有第二筆超過 L=10;放大公式 ceil(max(11, 10) * 1.5) = 17,L 放大時 max_steps 跟著變成新 L
    layers = [CONV, FC]
    diags = [_diag([2, 11, 2], [1, 1, 1], [1, 1, 1]), _fc_diag()]
    grown = grown_to_fit_batch(layers, diags)
    assert grown is not layers
    assert (grown[0].L, grown[0].max_out_spikes, grown[0].max_steps) == (17, 20, 17)
    assert grown[1] is FC
