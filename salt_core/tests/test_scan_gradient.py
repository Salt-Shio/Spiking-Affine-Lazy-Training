"""surrogate gradient 接進 run_layer_forward 的 chunk 化掃描之後梯度還是對的。

例子同 test_surrogate.py(tau=4,v_th=1.0,N=[0,1,4],w=[0.6,0.6,0.9],梯度 [0.898341, 0.680819,
0.910170]),包成 n=3(三個來源各發一次事件)、m=1 的 FC 佇列。loss = sum(s_value):每筆真事件的 s
剛好加一次,跟切成幾個 chunk 無關,所以梯度要等於 test_surrogate.py 的序列版。
chunk_size=1(每步一筆)跟 chunk_size=3(整條進同一個 chunk,fire 前還有事件)各跑一次。
"""

import jax
import jax.numpy as jnp

from salt_core.float.scan import run_layer_forward
from salt_core.connectivity.fc import build_fc_structure, fc_float_values

TOL = 1e-4

# test_surrogate.py::test_gradient_flows_through_fire_reset 手算的梯度
EXPECTED_GRAD = [0.898341, 0.680819, 0.910170]


def assert_allclose(actual, expected, msg, tol=TOL):
    actual = float(actual)
    expected = float(expected)
    assert abs(actual - expected) < tol, f"{msg}: got {actual}, expected {expected}"


def _loss(W, event_times, event_source_idx, tau, v_th, chunk_size, alpha):
    maps = fc_float_values(build_fc_structure(event_times, event_source_idx, event_times.shape[0]),
                           W, tau, None)
    n_real_events = event_times.shape[0]
    _, _, _, s_value, _ = run_layer_forward(maps, v_th, chunk_size=chunk_size,
                                            max_steps=n_real_events, alpha=alpha,
                                            n_real_events=n_real_events)
    # s_value 已經排除 pad、空轉步,直接整個加總
    return jnp.sum(s_value)


def _check_grad_matches_reference(chunk_size):
    tau = 4.0
    v_th = 1.0
    alpha = 2.0

    # 事件時間 [0,1,5] 對應 N=[0,1,4];W[0]=[0.6,0.6,0.9] 同 test_surrogate.py 的 w
    event_times = jnp.array([0.0, 1.0, 5.0])
    event_source_idx = jnp.array([0, 1, 2])
    W = jnp.array([[0.6, 0.6, 0.9]])

    grad_fn = jax.grad(_loss)
    grad_W = grad_fn(W, event_times, event_source_idx, tau, v_th, chunk_size, alpha)

    assert grad_W.shape == (1, 3)
    for i, expected in enumerate(EXPECTED_GRAD):
        assert_allclose(grad_W[0, i], expected,
                         f"chunk_size={chunk_size}: d(loss)/dW[0,{i}]")


def test_gradient_matches_sequential_reference_chunk_size_1():
    _check_grad_matches_reference(chunk_size=1)


def test_gradient_matches_sequential_reference_chunk_size_full():
    """chunk_size=3:整條進同一個 chunk,在 event1 fire、event0 在 fire 之前;梯度跟 chunk_size=1 一致。"""
    _check_grad_matches_reference(chunk_size=3)


# docs/math/不套閘與soft-reset梯度推導.md「用具體數字驗算」手算的 v_final 梯度(同一組例子,
# g_1 = -slope_1 * x_1 ≈ -1.024711):
#   d(v_final)/dw1 = g_1         ≈ -1.024711(fire 事件自己)
#   d(v_final)/dw0 = g_1 * a_1   ≈ -0.768533(fire 之前的事件)
#   d(v_final)/dw2 = 0          (fire 之後的事件沒被讀到)
# 這是 fire 那一步結束時的 v_final,所以下面的 max_steps 剛好停在 fire 那一步,不處理 event2。
EXPECTED_GRAD_V_FINAL = [-0.768533, -1.024711, 0.0]


def _loss_v_final(W, event_times, event_source_idx, tau, v_th, chunk_size, max_steps, alpha):
    """loss 直接是 v_final(膜電位回歸),不經過 s_value。"""
    maps = fc_float_values(build_fc_structure(event_times, event_source_idx, event_times.shape[0]),
                           W, tau, None)
    _, _, _, _, v_final = run_layer_forward(maps, v_th, chunk_size=chunk_size,
                                            max_steps=max_steps, alpha=alpha,
                                            n_real_events=maps.a.shape[1])
    return v_final[0]


def _check_v_final_grad_matches_reference(chunk_size, max_steps):
    tau = 4.0
    v_th = 1.0
    alpha = 2.0

    event_times = jnp.array([0.0, 1.0, 5.0])
    event_source_idx = jnp.array([0, 1, 2])
    W = jnp.array([[0.6, 0.6, 0.9]])

    grad_fn = jax.grad(_loss_v_final)
    grad_W = grad_fn(W, event_times, event_source_idx, tau, v_th, chunk_size, max_steps, alpha)

    assert grad_W.shape == (1, 3)
    for i, expected in enumerate(EXPECTED_GRAD_V_FINAL):
        assert_allclose(grad_W[0, i], expected,
                         f"chunk_size={chunk_size},max_steps={max_steps}: d(v_final)/dW[0,{i}]")


def test_v_final_regression_gradient_single_chunk():
    """chunk_size=3、max_steps=1:三筆事件進同一個 chunk,一步就停,event1 fire,event2 沒被讀到。"""
    _check_v_final_grad_matches_reference(chunk_size=3, max_steps=1)


def test_v_final_regression_gradient_two_small_chunks():
    """chunk_size=1、max_steps=2:event0、event1 各一步,event1 fire 後停;梯度跟上一個測試一樣。"""
    _check_v_final_grad_matches_reference(chunk_size=1, max_steps=2)
