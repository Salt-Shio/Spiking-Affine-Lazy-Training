"""salt_core/capacity.py:Capacity、reduce_over_batch、GrowthPolicy、層清單的放大縮小。"""
import jax.numpy as jnp
import numpy as np
import pytest

from salt_core.capacity import (Capacity, GrowthPolicy, LayerDiag, grown_to_fit, grown_to_fit_batch,
                                reduce_over_batch, shrunk_to_observed)
from salt_core.float.affine import extra_steps_upper_bound
from salt_core.layers import ConvLayer, FCLayer

CONV = ConvLayer(name="conv", ic=1, h_in=4, w_in=4, oc=1, k=3, s=1, p=1, init_k=1.0, chunk_size=4,
                 max_queue_len=10, max_out_spikes=20, max_extra_steps=10)
FC = FCLayer(name="out", n_in=16, n_out=2, init_k=1.0)
POLICIES = {"conv": GrowthPolicy(), "out": GrowthPolicy()}


def _diag(queue, out_spikes, steps, spike_count=(0, 0, 0)) -> LayerDiag:
    """三筆樣本的逐筆診斷。"""
    return LayerDiag(spike_count=jnp.array(spike_count, dtype=jnp.float32),
                     firing_rate=jnp.zeros(3),
                     needed={"max_queue_len": jnp.array(queue), "max_out_spikes": jnp.array(out_spikes),
                             "max_extra_steps": jnp.array(steps)})


def _fc_diag(out_spikes=(0, 0, 0), extra_steps=(0, 0, 0)) -> LayerDiag:
    return LayerDiag(spike_count=jnp.zeros(3), firing_rate=jnp.zeros(3),
                     needed={"max_out_spikes": jnp.array(out_spikes),
                             "max_extra_steps": jnp.array(extra_steps)})


def test_reduce_over_batch_takes_max_of_needed_and_mean_of_stats():
    reduced = reduce_over_batch(_diag([3, 9, 5], [1, 2, 7], [4, 4, 6], spike_count=[100, 200, 300]))
    assert {k: int(v) for k, v in reduced.needed.items()} == {"max_queue_len": 9, "max_out_spikes": 7,
                                                              "max_extra_steps": 6}
    assert float(reduced.spike_count) == 200.0


def test_conv_capacity_matches_fields_and_with_capacity_replaces_them():
    assert dict(CONV.capacity) == {"max_queue_len": 10, "max_out_spikes": 20, "max_extra_steps": 10}
    grown = CONV.with_capacity(CONV.capacity.replace(max_queue_len=30))
    assert (grown.max_queue_len, grown.max_out_spikes, grown.max_extra_steps) == (30, 20, 10)


def test_fc_capacity_defaults():
    assert dict(FC.capacity) == {"max_out_spikes": 8192, "max_extra_steps": 0}


def test_capacity_replace_rejects_unknown_knob():
    with pytest.raises(KeyError):
        CONV.capacity.replace(queue=5)


def test_fits_checks_every_knob_per_sample():
    # 第二筆 max_queue_len 需要 11 > 10,第三筆 max_extra_steps 需要 12 > 10
    fits = CONV.capacity.fits(_diag([10, 11, 3], [20, 1, 1], [10, 2, 12]))
    assert np.array_equal(np.asarray(fits), [True, False, False])


def test_fits_returns_same_list_object():
    layers = [CONV, FC]
    diags = [_diag([10, 3, 3], [20, 1, 1], [10, 2, 2]), _fc_diag()]
    assert grown_to_fit_batch(layers, POLICIES, diags) is layers


def test_one_overflowing_sample_grows_layer():
    # 只有第二筆超過 max_queue_len=10;放大公式 ceil(max(11, 10) * 1.5) = 17,max_queue_len 放大時
    # max_extra_steps 跟著變成一定夠的值 17 - ceil(17 / 4) = 12
    layers = [CONV, FC]
    diags = [_diag([2, 11, 2], [1, 1, 1], [1, 1, 1]), _fc_diag()]
    grown = grown_to_fit_batch(layers, POLICIES, diags)
    assert grown is not layers
    assert (grown[0].max_queue_len, grown[0].max_out_spikes, grown[0].max_extra_steps) == (17, 20, 12)
    assert grown[1] is FC


CONV2 = ConvLayer(name="conv2", ic=1, h_in=4, w_in=4, oc=1, k=3, s=1, p=1, init_k=1.0, chunk_size=4,
                  max_queue_len=10, max_out_spikes=20, max_extra_steps=10)
TWO_CONV_POLICIES = {"conv": GrowthPolicy(), "conv2": GrowthPolicy(), "out": GrowthPolicy()}


def test_only_second_layer_overflowing_grows_only_second_layer():
    # conv 全部放得下;conv2 第二筆 max_queue_len 需要 15 > 10:ceil(15 * 1.5) = 23,
    # max_extra_steps 跟著變 23 - ceil(23 / 4) = 17
    layers = [CONV, CONV2, FC]
    diags = [_diag([2, 3, 2], [1, 1, 1], [1, 1, 1]), _diag([2, 15, 2], [1, 1, 1], [1, 1, 1]),
             _fc_diag()]
    grown = grown_to_fit_batch(layers, TWO_CONV_POLICIES, diags)
    assert grown[0] is CONV
    assert dict(grown[1].capacity) == {"max_queue_len": 23, "max_out_spikes": 20, "max_extra_steps": 17}
    assert grown[2] is FC


def test_two_layers_overflowing_grow_together_in_one_call():
    # conv max_queue_len 需要 12 > 10:ceil(12 * 1.5) = 18,max_extra_steps 跟著變 18 - ceil(18 / 4) = 13;
    # conv2 max_out_spikes 需要 25 > 20:ceil(25 * 1.5) = 38
    layers = [CONV, CONV2, FC]
    diags = [_diag([12, 3, 2], [1, 1, 1], [1, 1, 1]), _diag([2, 3, 2], [1, 25, 1], [1, 1, 1]),
             _fc_diag()]
    grown = grown_to_fit_batch(layers, TWO_CONV_POLICIES, diags)
    assert dict(grown[0].capacity) == {"max_queue_len": 18, "max_out_spikes": 20, "max_extra_steps": 13}
    assert dict(grown[1].capacity) == {"max_queue_len": 10, "max_out_spikes": 38, "max_extra_steps": 10}
    assert grown[2] is FC


# ============================================================================
# GrowthPolicy
# ============================================================================

def _needed(queue, out_spikes, steps) -> dict:
    return {"max_queue_len": queue, "max_out_spikes": out_spikes, "max_extra_steps": steps}


def test_grown_returns_same_capacity_when_everything_fits():
    capacity = Capacity(max_queue_len=100, max_out_spikes=2000, max_extra_steps=100)
    assert GrowthPolicy().grown(capacity, _needed(50, 10, 100), chunk_size=4) == capacity


def test_grown_queue_overflow_resets_extra_steps_to_safe_value():
    # 只有 max_queue_len 出界:ceil(777 * 1.5) = 1166;這批的步數需求是在裝不下的佇列上算的,
    # 不可信,max_extra_steps 直接設成一定夠的值 1166 - ceil(1166 / 4) = 874(總步數 = 1166)
    capacity = Capacity(max_queue_len=100, max_out_spikes=2000, max_extra_steps=100)
    grown = GrowthPolicy().grown(capacity, _needed(777, 10, 0), chunk_size=4)
    assert dict(grown) == {"max_queue_len": 1166, "max_out_spikes": 2000, "max_extra_steps": 874}


def test_grown_only_extra_steps_overflow_bumps_only_extra_steps():
    # max_extra_steps 需求 107 > 100:ceil(107 * 2.0) = 214,其他不動
    capacity = Capacity(max_queue_len=100, max_out_spikes=2000, max_extra_steps=100)
    grown = GrowthPolicy(max_extra_steps_grow_factor=2.0).grown(capacity, _needed(50, 10, 107),
                                                                chunk_size=4)
    assert dict(grown) == {"max_queue_len": 100, "max_out_spikes": 2000, "max_extra_steps": 214}


def test_shrunk_needs_candidate_below_threshold():
    # max_out_spikes:觀察 600 -> 候選 ceil(600 * 1.5) = 900 < 2000 * 0.5 = 1000,縮
    # max_extra_steps:觀察 40 -> 候選 60,不低於 100 * 0.5 = 50,不縮
    # max_queue_len 沒有縮小門檻,不縮
    capacity = Capacity(max_queue_len=100, max_out_spikes=2000, max_extra_steps=100)
    shrunk = GrowthPolicy().shrunk(capacity, _needed(10, 600, 40))
    assert dict(shrunk) == {"max_queue_len": 100, "max_out_spikes": 900, "max_extra_steps": 100}


def test_shrunk_never_goes_below_one():
    # 整層沒 fire:觀察 0 -> ceil(0 * 1.5) = 0,夾到 1(容量 0 會讓下一層拿到長度 0 的輸入流)
    capacity = Capacity(max_queue_len=100, max_out_spikes=2000, max_extra_steps=100)
    shrunk = GrowthPolicy().shrunk(capacity, _needed(0, 0, 40))
    assert shrunk["max_out_spikes"] == 1


def test_grown_to_fit_changes_only_capacity_fields():
    """放大後的層除了容量,其他欄位都跟原本一樣。"""
    diag = LayerDiag(spike_count=jnp.array(0), firing_rate=jnp.array(0.0),
                     needed=_needed(jnp.array(11), jnp.array(1), jnp.array(1)))
    [grown] = grown_to_fit([CONV], POLICIES, [diag])
    assert grown is not CONV
    assert grown == CONV.with_capacity(grown.capacity)
    assert grown.max_queue_len == 17


def test_shrunk_to_observed_returns_same_list_when_nothing_shrinks():
    layers = [CONV, FC]
    observed = {"conv": _needed(10, 20, 10), "out": {"max_out_spikes": 8192, "max_extra_steps": 0}}
    assert shrunk_to_observed(layers, POLICIES, observed) is layers


def test_with_chunk_size_resets_conv_extra_steps_to_safe_value():
    # max_extra_steps=3 是照 chunk_size=4 縮過的值;換成 2 之後設成 30 - ceil(30 / 2) = 15,
    # 總步數 15 + 15 = max_queue_len
    conv = ConvLayer(name="conv", ic=1, h_in=4, w_in=4, oc=1, k=3, s=1, p=1, init_k=1.0,
                     chunk_size=4, max_queue_len=30, max_out_spikes=20, max_extra_steps=3)
    two = conv.with_chunk_size(2)
    assert (two.chunk_size, two.max_extra_steps, two.max_queue_len) == (2, 15, 30)
    assert two.scan_steps(None) == 30
    assert FC.with_chunk_size(3) == FCLayer(name="out", n_in=16, n_out=2, init_k=1.0, chunk_size=3)


# ============================================================================
# 額外步數:掃描步數 = ceil(佇列長度 / chunk_size) + max_extra_steps
# ============================================================================

def test_extra_steps_upper_bound_hand_example():
    # 佇列長度 10、chunk_size=5、v_th=1:四筆 b=0.6 -> m=4,能量上界 floor(2.4)=2,m*=2;
    # 步數上界 2 + ceil(8/5) = 4,基本步數 ceil(10/5) = 2,額外步數 2
    b = jnp.array([[0.6, 0.6, 0.6, 0.6, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    assert extra_steps_upper_bound(b, 1.0, 5).tolist() == [2]


def test_extra_steps_upper_bound_is_zero_without_fire_or_with_chunk_size_one():
    b = jnp.array([[0.6, 0.6, 0.6, 0.6, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    assert extra_steps_upper_bound(b, 1e9, 5).tolist() == [0]
    assert extra_steps_upper_bound(b, 1.0, 1).tolist() == [0]


def test_conv_default_extra_steps_scans_whole_queue():
    # 沒填 max_extra_steps:30 - ceil(30 / 4) = 22,總步數 8 + 22 = max_queue_len
    conv = ConvLayer(name="conv", ic=1, h_in=4, w_in=4, oc=1, k=3, s=1, p=1, init_k=1.0,
                     chunk_size=4, max_queue_len=30)
    assert (conv.max_extra_steps, conv.scan_steps(None)) == (22, 30)


def test_fc_extra_steps_overflow_grows_only_fc():
    # FC max_extra_steps 需要 3 > 0:ceil(3 * 1.5) = 5;沒有佇列,沒有特例
    layers = [CONV, FC]
    diags = [_diag([2, 3, 2], [1, 1, 1], [1, 1, 1]), _fc_diag(extra_steps=[0, 3, 1])]
    grown = grown_to_fit_batch(layers, POLICIES, diags)
    assert grown[0] is CONV
    assert dict(grown[1].capacity) == {"max_out_spikes": 8192, "max_extra_steps": 5}
