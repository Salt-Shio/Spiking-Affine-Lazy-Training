"""層的 unflatten_neurons / broadcast_channels:神經元攤平順序跟 forward 一致。"""
import jax.numpy as jnp
import numpy as np
import pytest

from salt_core.stream import EventStream
from salt_core.layers import ConvLayer, FCLayer


def _conv():
    # 輸出 3 channel x 2 x 3 = 18 顆神經元
    return ConvLayer(name="conv", ic=1, h_in=2, w_in=3, oc=3, k=1, s=1, p=0,
                     init_k=5.0, v_th=1.0, chunk_size=1, max_queue_len=8, max_out_spikes=64)


def _fc():
    return FCLayer(name="out", n_in=4, n_out=5, init_k=5.0)


def test_conv_forward_spikes_land_in_the_only_active_channel():
    # 權重只有 channel 1 非零,每個輸入像素一筆事件,channel 1 每顆神經元都 fire 一次
    conv = _conv()
    w = jnp.zeros(conv.weight_shape).at[1].set(2.0)
    n = conv.h_in * conv.w_in
    in_stream = EventStream(event_times=jnp.arange(1.0, n + 1.0),
                            event_source_idx=jnp.arange(n),
                            event_gain=jnp.ones((n,)), n_real_events=jnp.array(n))

    result = conv.forward(w, in_stream).result
    spikes = np.asarray(result.spike_mask).sum(axis=1)

    grid = conv.unflatten_neurons(spikes)
    assert grid.shape == (3, 2, 3)
    assert np.array_equal(grid[1], np.ones((2, 3)))
    assert grid[0].sum() == 0 and grid[2].sum() == 0


def test_conv_unflatten_keeps_trailing_axes_and_follows_channel_major_order():
    # 神經元編號 = c*h*w + y*w + x,h*w = 6
    conv = _conv()
    x = np.arange(18 * 2).reshape(18, 2)

    grid = conv.unflatten_neurons(x)

    assert grid.shape == (3, 2, 3, 2)
    assert np.array_equal(grid[1, 0, 2], x[1 * 6 + 0 * 3 + 2])


def test_conv_broadcast_channels_is_inverse_of_unflatten():
    conv = _conv()
    per_channel = jnp.array([0.5, 1.5, 2.5])

    per_neuron = conv.broadcast_channels(per_channel)

    assert per_neuron.shape == (18,)
    grid = conv.unflatten_neurons(per_neuron)
    assert np.array_equal(np.asarray(grid.max(axis=(1, 2))), np.asarray(per_channel))
    assert np.array_equal(np.asarray(grid.min(axis=(1, 2))), np.asarray(per_channel))


def test_fc_each_neuron_is_its_own_channel():
    fc = _fc()
    x = np.arange(5.0)

    assert fc.unflatten_neurons(x).shape == (5, 1, 1)
    assert np.array_equal(fc.unflatten_neurons(x).max(axis=(1, 2)), x)
    assert np.array_equal(fc.broadcast_channels(x), x)


@pytest.mark.parametrize("layer, method, bad_length", [
    (_conv(), "unflatten_neurons", 17),
    (_conv(), "broadcast_channels", 18),
    (_fc(), "unflatten_neurons", 4),
    (_fc(), "broadcast_channels", 4),
])
def test_wrong_leading_axis_length_raises(layer, method, bad_length):
    with pytest.raises(ValueError):
        getattr(layer, method)(np.zeros(bad_length))
