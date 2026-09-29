"""event_gain 補的是真的梯度缺口:下一層用離散索引查權重,對上一層權重的梯度恆為 0,
傳上一層的 s_spike 當 event_gain 才接得回來。理由見 docs/問題紀錄.md
「洞見:離散索引 gather 不會把梯度帶回「決定索引值的來源」」。

例子:layer1 一顆神經元 p,三個各發一次事件的來源(同 test_scan_gradient.py:tau=4,v_th=1.0,
event_times=[0,1,5],W1=[[0.6,0.6,0.9]]),p 在事件 1(t=1)fire。layer2 一顆神經元 q,只接 p,
W2=[[0.5]]。loss 是 q 的 v_final(layer2 不 fire)。

手算:
  x_0 = 0.6(N=0,a=1)
  x_1 = 0.75*0.6+0.6 = 1.05 >= v_th,spike,s_spike = atan_spike(0.05) forward=1
  atan_spike backward 在 x=0.05,alpha=2 處的斜率:
    a=alpha/2=1, ax=pi*1*0.05≈0.157080, slope=1/(1+ax^2)≈0.975920
  v_final,q = a_2*0 + s_spike*W2[q,p] = s_spike*0.5(a_2 這項因為 v0=0 恆為 0,
    跟 layer2 的 N 算成多少無關)
  d(s_spike)/dw1 = slope*dx1/dw1 = slope*1 ≈ 0.975920(w1 是直接項)
  d(s_spike)/dw0 = slope*dx1/dw0 = slope*a_1 = slope*0.75 ≈ 0.731940
    (w0 只透過 x0->x1 的鏈式間接影響)
  d(s_spike)/dw2 = 0(事件2在 max_steps=2 就停了,從來沒被讀取過,不是遮罩掉)
  d(v_final,q)/dW1 = 0.5 * d(s_spike)/dW1 ≈ [0.365970, 0.487960, 0.0]
"""

import jax
import jax.numpy as jnp

from salt_core.float.scan import run_layer
from salt_core.connectivity.fc import build_fc_structure, fc_float_values
from salt_core.stream import extract_output_events_fc

TOL = 1e-3

EXPECTED_GRAD_W1 = jnp.array([[0.365970, 0.487960, 0.0]])


def assert_allclose(actual, expected, msg, tol=TOL):
    actual = float(actual)
    expected = float(expected)
    assert abs(actual - expected) < tol, f"{msg}: got {actual}, expected {expected}"


def _two_layer_v_final_q(W1, event_times, event_source_idx, tau, v_th, alpha, W2,
                          use_gain):
    """layer1(1 顆神經元 p,3 個來源)接 layer2(1 顆神經元 q,只接 p)。
    max_steps=2:layer1 處理完事件 0、事件 1(fire)就停,事件 2 沒被讀到。
    use_gain=False 時不傳 event_gain,當對照組。"""
    maps1 = fc_float_values(build_fc_structure(event_times, event_source_idx, event_times.shape[0]),
                            W1, tau, None)
    spike_mask, spike_event_idx, s_spike, _, _ = run_layer(
        maps1, v_th, chunk_size=1, max_steps=2, alpha=alpha,
        n_real_events=maps1.a.shape[1])

    times2, src2, gain2, n_real_events2 = extract_output_events_fc(
        spike_mask, spike_event_idx, s_spike, event_times)

    maps2 = fc_float_values(build_fc_structure(times2, src2, n_real_events2),
                            W2, tau, (gain2 if use_gain else None))
    _, _, _, _, v_final2 = run_layer(
        maps2, v_th, chunk_size=1, max_steps=maps2.a.shape[1], alpha=alpha,
        n_real_events=n_real_events2)
    return v_final2[0]


def _setup():
    tau = 4.0
    v_th = 1.0
    alpha = 2.0
    event_times = jnp.array([0.0, 1.0, 5.0])
    event_source_idx = jnp.array([0, 1, 2])
    W1 = jnp.array([[0.6, 0.6, 0.9]])
    W2 = jnp.array([[0.5]])
    return W1, event_times, event_source_idx, tau, v_th, alpha, W2


def test_forward_value_matches_hand_calc():
    """forward 數值對:有沒有 event_gain 都一樣,因為 s_spike forward 等於 1。"""
    W1, event_times, event_source_idx, tau, v_th, alpha, W2 = _setup()
    v_gain = _two_layer_v_final_q(W1, event_times, event_source_idx, tau, v_th,
                                   alpha, W2, use_gain=True)
    v_no_gain = _two_layer_v_final_q(W1, event_times, event_source_idx, tau, v_th,
                                      alpha, W2, use_gain=False)
    assert_allclose(v_gain, 0.5, "v_final,q 手算應該是 0.5")
    assert_allclose(v_no_gain, 0.5, "沒有 event_gain 時 forward 數值應該完全相同")


def test_cross_layer_gradient_matches_hand_calc():
    """有 event_gain 時 d(v_final_q)/dW1 不是 0,而且對上手算的三個分量。"""
    W1, event_times, event_source_idx, tau, v_th, alpha, W2 = _setup()

    grad_fn = jax.grad(lambda W1: _two_layer_v_final_q(
        W1, event_times, event_source_idx, tau, v_th, alpha, W2, use_gain=True))
    grad_W1 = grad_fn(W1)

    assert grad_W1.shape == (1, 3)
    for j in range(3):
        assert_allclose(grad_W1[0, j], EXPECTED_GRAD_W1[0, j],
                         f"d(v_final,q)/dW1[0,{j}]")


def test_without_event_gain_gradient_is_zero():
    """對照組:不傳 event_gain 時梯度恆為 0,計算圖裡沒有這條邊。"""
    W1, event_times, event_source_idx, tau, v_th, alpha, W2 = _setup()

    grad_fn = jax.grad(lambda W1: _two_layer_v_final_q(
        W1, event_times, event_source_idx, tau, v_th, alpha, W2, use_gain=False))
    grad_W1 = grad_fn(W1)

    assert grad_W1.shape == (1, 3)
    for j in range(3):
        assert_allclose(grad_W1[0, j], 0.0,
                         f"沒有 event_gain 時 d(v_final,q)/dW1[0,{j}] 應該恆為 0")
