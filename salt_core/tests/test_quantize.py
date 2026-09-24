"""salt_core/quantize.py 的測試。推導見 docs/math/權重量化推導.md。

**這份測試在寫出來的當下沒有跑過**(這個 session 的環境沒有裝 JAX、也沒有
GPU),下一次有能跑的環境時要先執行這份測試確認邏輯正確,不能直接信任。

分三層:

- `fake_quantize_tensor`(線性量化 primitive):round-trip 誤差有界、bits 越多
  誤差越小、per-channel threshold 正確對到指定 axis、全零 channel 不除以零、
  `bits<2` 擋掉。
- `quantization_error`:跟手動算的統計對得上,恆等輸入回傳 mse=0/sqnr=inf。
- `quantize_params`:接線正確(shape/對齊 layers 不變),`clip_percentile=100`
  等同 `fake_quantize_tensor` 沒給 `threshold` 時的預設 max-abs 行為。
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from salt_core.layers import ConvLayer, FCLayer
from salt_core.quantize import (fake_quantize_tensor, quantization_error,
                                quantize_params)

TOL = 1e-5

_C1 = dict(ic=2, h_in=34, w_in=34, oc=8, k=3, s=2, p=1)


def _layers():
    conv1 = ConvLayer(name="conv1", **_C1, L=185, max_out_spikes=4000, init_k=5.0)
    out = FCLayer(name="out", n_in=conv1.n_neurons, n_out=10, chunk_size=512, init_k=5.0)
    return [conv1, out]


def _params(layers, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), len(layers))
    return tuple(layer.init_weight(k) for layer, k in zip(layers, keys))


# ============================================================================
# A. fake_quantize_tensor
# ============================================================================

def test_fake_quantize_tensor_all_zero_input_maps_to_zero():
    """全零輸入:threshold 全零 fallback 成 1.0,不除以零,量化結果仍是 0。"""
    x = jnp.zeros((4, 4))
    x_hat, scale = fake_quantize_tensor(x, bits=8)
    assert np.allclose(np.asarray(x_hat), 0.0)
    assert float(scale) == pytest.approx(1.0 / (2 ** 7 - 1))


def test_fake_quantize_tensor_roundtrip_error_bounded_by_half_step():
    """round-to-nearest,誤差上界是半個量化步長(沒有值超出 threshold 時)。"""
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, (1000,), minval=-3.0, maxval=3.0)
    x_hat, scale = fake_quantize_tensor(x, bits=8)
    err = np.abs(np.asarray(x - x_hat))
    assert np.all(err <= float(scale) / 2 + TOL)


def test_fake_quantize_tensor_more_bits_lower_mse():
    """同一份資料,bits 越多量化噪聲(mse)應該越小(推導文件步驟 2)。"""
    key = jax.random.PRNGKey(1)
    x = jax.random.uniform(key, (2000,), minval=-5.0, maxval=5.0)
    mses = []
    for bits in (2, 4, 6, 8):
        x_hat, _ = fake_quantize_tensor(x, bits=bits)
        mses.append(quantization_error(x, x_hat)["mse"])
    assert all(a > b for a, b in zip(mses, mses[1:]))


def test_fake_quantize_tensor_clips_outlier_to_threshold():
    """超過 threshold 的值被夾到 threshold,不是照原值量化。"""
    x = jnp.array([0.0, 1.0, 100.0])
    x_hat, scale = fake_quantize_tensor(x, bits=8, threshold=1.0)
    assert float(x_hat[2]) == pytest.approx(1.0, abs=float(scale))
    assert float(x_hat[0]) == pytest.approx(0.0, abs=TOL)


def test_fake_quantize_tensor_per_channel_axis0_independent_threshold():
    """`axis=0` 時每個 channel(row)各自的 threshold 只看自己那一行。"""
    x = jnp.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    x_hat, scale = fake_quantize_tensor(x, bits=8, axis=0)
    assert scale.shape == (2, 1)
    assert float(scale[0, 0]) == pytest.approx(3.0 / (2 ** 7 - 1), rel=1e-6)
    assert float(scale[1, 0]) == pytest.approx(30.0 / (2 ** 7 - 1), rel=1e-6)


def test_fake_quantize_tensor_all_zero_channel_no_nan():
    """per-channel 量化下,某個 channel 全零不該讓那個 channel 出現 NaN。"""
    x = jnp.array([[0.0, 0.0], [1.0, 2.0]])
    x_hat, _scale = fake_quantize_tensor(x, bits=8, axis=0)
    assert not np.any(np.isnan(np.asarray(x_hat)))
    assert np.allclose(np.asarray(x_hat[0]), 0.0)


def test_fake_quantize_tensor_rejects_bits_below_2():
    with pytest.raises(ValueError):
        fake_quantize_tensor(jnp.ones((3,)), bits=1)


# ============================================================================
# B. quantization_error
# ============================================================================

def test_quantization_error_identical_arrays_zero_mse_inf_sqnr():
    x = jnp.array([1.0, -2.0, 3.0])
    stats = quantization_error(x, x)
    assert stats["mse"] == 0.0
    assert stats["max_abs_err"] == 0.0
    assert stats["sqnr_db"] == float("inf")


def test_quantization_error_matches_manual_computation():
    x = jnp.array([1.0, -1.0, 2.0, -2.0])
    x_hat = jnp.array([1.5, -0.5, 2.5, -1.5])
    stats = quantization_error(x, x_hat)
    err = np.asarray(x - x_hat)
    assert stats["mse"] == pytest.approx(float(np.mean(err ** 2)), rel=1e-6)
    assert stats["max_abs_err"] == pytest.approx(float(np.max(np.abs(err))), rel=1e-6)


# ============================================================================
# C. quantize_params
# ============================================================================

def test_quantize_params_preserves_shapes_and_alignment():
    layers = _layers()
    params = _params(layers)
    q = quantize_params(layers, params, bits=8)
    assert len(q) == len(layers)
    for w, w_q in zip(params, q):
        assert w_q.shape == w.shape


def test_quantize_params_clip_percentile_100_equals_max_abs_default():
    """`clip_percentile=100` 應該跟 `fake_quantize_tensor` 沒給 `threshold`
    時的預設(max-abs)行為一致——100th percentile 精確等於最大值。"""
    layers = _layers()
    params = _params(layers)
    q_pct = quantize_params(layers, params, bits=8, per_channel=True, clip_percentile=100.0)
    q_direct = tuple(fake_quantize_tensor(w, bits=8, axis=0)[0] for w in params)
    for a, b in zip(q_pct, q_direct):
        assert np.allclose(np.asarray(a), np.asarray(b), atol=TOL)


def test_quantize_params_lower_percentile_increases_error():
    """門檻夾得比 100th percentile 窄,uniform 初始化的權重下截斷誤差主導,
    整體 mse 不該系統性變小(推導文件步驟 2 的 trade-off 方向性檢查)。"""
    layers = _layers()
    params = _params(layers)
    q_full = quantize_params(layers, params, bits=8, clip_percentile=100.0)
    q_clip = quantize_params(layers, params, bits=8, clip_percentile=50.0)
    err_full = sum(quantization_error(w, w_hat)["mse"] for w, w_hat in zip(params, q_full))
    err_clip = sum(quantization_error(w, w_hat)["mse"] for w, w_hat in zip(params, q_clip))
    assert err_clip >= err_full - TOL
