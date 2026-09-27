"""三個標準解碼器的驗證。

解碼器只讀 `LayerForwardResult`(chunk_scan.py 的公開契約),把它讀成分數張量:

  - 膜電位回歸:直接回 `v_final`
  - 頻率:每顆神經元 `s_value` 沿時間軸加總(可微),硬 count 放 metrics
  - 群體:神經元連續等分成組,每組 `s_value` 加總相加

`validate` 擋「最後一層門檻設定跟編碼不配」。
"""
import jax
import jax.numpy as jnp
import pytest

from salt_core.chunk_scan import LayerForwardResult
from salt_core.decoder import (MembraneRegressionDecoder, PopulationDecoder,
                                RateDecoder)
from salt_core.layers import FCLayer

TOL = 1e-6


def _hand_result(n_neurons: int = 3, n_steps: int = 4) -> LayerForwardResult:
    """手構一個 `LayerForwardResult`:數值挑成好心算。`s_value` / `spike_mask`
    / `v_final` 直接指定,不跑 forward。"""
    s_value = jnp.array([[0.1, 0.2, 0.0, 0.0],     # 加總 0.3
                         [0.5, 0.5, 0.5, 0.0],     # 加總 1.5
                         [0.0, 0.0, 0.0, 0.0]])    # 加總 0.0
    spike_mask = jnp.array([[True, False, False, False],   # 1 次
                            [True, True, False, False],    # 2 次
                            [False, False, False, False]]) # 0 次
    v_final = jnp.array([1.25, -0.5, 3.0])
    return LayerForwardResult(
        spike_mask=spike_mask,
        spike_event_idx=jnp.zeros((n_neurons, n_steps), dtype=jnp.int32),
        s_spike=jnp.zeros((n_neurons, n_steps)),
        s_value=s_value,
        v_final=v_final)


def _fc(v_th: float, n_out: int = 3) -> FCLayer:
    # validate() 只讀 .v_th / .n_out,其餘吃 FCLayer 預設。
    return FCLayer(name="out", n_in=4, n_out=n_out, v_th=v_th, init_k=1.0)


def test_membrane_regression_returns_v_final():
    scores, metrics = MembraneRegressionDecoder().decode(_hand_result())
    assert jnp.allclose(scores, jnp.array([1.25, -0.5, 3.0]), atol=TOL)
    assert metrics == {}


def test_rate_returns_s_value_sum_and_hard_count_metric():
    scores, metrics = RateDecoder().decode(_hand_result())
    assert jnp.allclose(scores, jnp.array([0.3, 1.5, 0.0]), atol=TOL)
    # 硬 count 每顆 1 / 2 / 0 -> mean 1.0、max 2.0
    assert float(metrics["hard_count_mean"]) == pytest.approx(1.0)
    assert float(metrics["hard_count_max"]) == pytest.approx(2.0)


def test_population_groups_are_contiguous_equal_partitions():
    # 6 神經元 -> 2 類 x 3 顆
    s_value = jnp.array([[1.0, 0.0], [0.5, 0.5], [0.0, 0.0],   # 類0: 1.0 + 1.0 + 0.0 = 2.0
                         [0.0, 0.0], [2.0, 0.0], [0.5, 0.5]])  # 類1: 0.0 + 2.0 + 1.0 = 3.0
    r = LayerForwardResult(
        spike_mask=jnp.zeros((6, 2), dtype=bool),
        spike_event_idx=jnp.zeros((6, 2), dtype=jnp.int32),
        s_spike=jnp.zeros((6, 2)),
        s_value=s_value,
        v_final=jnp.zeros(6))
    scores, _ = PopulationDecoder(n_classes=2, group_size=3).decode(r)
    assert jnp.allclose(scores, jnp.array([2.0, 3.0]), atol=TOL)


def test_validate_membrane_requires_non_firing_output_layer():
    MembraneRegressionDecoder().validate(_fc(v_th=1e9))       # 不 raise
    with pytest.raises(ValueError):
        MembraneRegressionDecoder().validate(_fc(v_th=1.0))


def test_validate_rate_requires_firing_output_layer():
    RateDecoder().validate(_fc(v_th=1.0))                     # 不 raise
    with pytest.raises(ValueError):
        RateDecoder().validate(_fc(v_th=1e9))


def test_validate_population_checks_group_layout():
    PopulationDecoder(n_classes=2, group_size=3).validate(_fc(v_th=1.0, n_out=6))
    with pytest.raises(ValueError):
        # n_out=3 != 2*3
        PopulationDecoder(n_classes=2, group_size=3).validate(_fc(v_th=1.0, n_out=3))


def test_decode_is_vmappable():
    r = _hand_result()
    batch = jax.tree_util.tree_map(lambda a: jnp.stack([a, a]), r)
    scores, metrics = jax.vmap(RateDecoder().decode)(batch)
    assert scores.shape == (2, 3)
    assert jnp.allclose(scores[0], jnp.array([0.3, 1.5, 0.0]), atol=TOL)
    assert metrics["hard_count_mean"].shape == (2,)
