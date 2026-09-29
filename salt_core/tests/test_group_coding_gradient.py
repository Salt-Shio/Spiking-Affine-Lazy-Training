"""跨神經元線性組合的 loss 的梯度:L = S1 - S2,S 是每顆神經元 s_value 的總和。

設定同 docs/math/全連接forward訓練範例.md「具體例子:n=2,m=2」(也同 test_fc_forward.py):
tau=4,v_th=1.0,上游事件 (t=1,a1)(t=2,a2)(t=4,a1),W=[[0.6,0.5],[0.3,0.2]]。
b1 在 t=4(事件 2)fire 一次,b2 不 fire。
同一個權重被同一顆神經元的兩筆事件共用(b1 的事件 0、2 都來自 a1,共用 w11),梯度要把兩筆的
貢獻加起來。手算結果見下面的常數,tol 1e-3 涵蓋手算的捨入。
"""

import jax
import jax.numpy as jnp

from salt_core.float.scan import run_layer
from salt_core.connectivity.fc import build_fc_structure, fc_float_values

TOL = 1e-3

# 手算的梯度:
#   dL/dw11 = dS1/dw11(event0、event2 都用 w11,兩項相加) ≈ 2.326497
#   dL/dw21 = dS1/dw21(只有 event1 用 w21)               ≈ 1.453337
#   dL/dw12 = -dS2/dw12(event0、event2 都用 w12)         ≈ -0.806479
#   dL/dw22 = -dS2/dw22(只有 event1 用 w22)              ≈ -0.416229
EXPECTED_GRAD_W = jnp.array([[2.326497, 1.453337],
                             [-0.806479, -0.416229]])


def assert_allclose(actual, expected, msg, tol=TOL):
    actual = float(actual)
    expected = float(expected)
    assert abs(actual - expected) < tol, f"{msg}: got {actual}, expected {expected}"


def _loss(W, event_times, event_source_idx, tau, v_th, chunk_size, alpha):
    maps = fc_float_values(build_fc_structure(event_times, event_source_idx, event_times.shape[0]),
                           W, tau, None)
    n_real_events = event_times.shape[0]
    _, _, _, s_value, _ = run_layer(maps, v_th, chunk_size=chunk_size,
                                    max_steps=n_real_events, alpha=alpha,
                                    n_real_events=n_real_events)
    S = jnp.sum(s_value, axis=1)  # 每顆神經元自己的 spike 總和,shape (m,)
    return S[0] - S[1]


def _check_grad_matches_reference(chunk_size):
    tau = 4.0
    v_th = 1.0
    alpha = 2.0

    event_times = jnp.array([1.0, 2.0, 4.0])
    event_source_idx = jnp.array([0, 1, 0])
    W = jnp.array([[0.6, 0.5],
                   [0.3, 0.2]])

    grad_fn = jax.grad(_loss)
    grad_W = grad_fn(W, event_times, event_source_idx, tau, v_th, chunk_size, alpha)

    assert grad_W.shape == (2, 2)
    for i in range(2):
        for j in range(2):
            assert_allclose(grad_W[i, j], EXPECTED_GRAD_W[i, j],
                             f"chunk_size={chunk_size}: d(L)/dW[{i},{j}]")


def test_group_coding_gradient_chunk_size_1():
    _check_grad_matches_reference(chunk_size=1)


def test_group_coding_gradient_chunk_size_full():
    """chunk_size=3:梯度跟 chunk_size=1 一致,兩筆事件共用同一個權重在 chunk 裡也算對。"""
    _check_grad_matches_reference(chunk_size=3)
