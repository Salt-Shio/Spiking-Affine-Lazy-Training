"""atan_spike 的 forward、backward,跟接上一個序列版單狀態遞迴之後,梯度能穿過 fire/reset。

chunk 化掃描的梯度在 test_scan_gradient.py。
"""

import jax
import jax.numpy as jnp

from salt_core.float.surrogate import atan_spike, atan_smooth

TOL = 1e-6


def assert_allclose(actual, expected, msg, tol=TOL):
    actual = float(actual)
    expected = float(expected)
    assert abs(actual - expected) < tol, f"{msg}: got {actual}, expected {expected}"


def test_forward_matches_exact_heaviside():
    alpha = 2.0
    for x in [-5.0, -0.5, -1e-6, 0.0, 1e-6, 0.5, 5.0]:
        expected = 1.0 if x >= 0 else 0.0
        got = float(atan_spike(jnp.asarray(x), alpha))
        assert_allclose(got, expected, f"atan_spike forward at x={x}")


def test_backward_matches_smooth_primitive_derivative():
    """custom_vjp 的 backward 要等於 atan_smooth 用 autodiff 算的導數。
    硬階梯函數的真實梯度幾乎處處是 0,不能拿來比。"""
    for alpha in [1.0, 2.0, 4.0]:
        for x in [-3.0, -1.0, -0.1, 0.0, 0.1, 1.0, 3.0]:
            x_arr = jnp.asarray(x)
            got = float(jax.grad(atan_spike, argnums=0)(x_arr, alpha))
            expected = float(jax.grad(atan_smooth, argnums=0)(x_arr, alpha))
            assert_allclose(got, expected, f"backward mismatch at x={x}, alpha={alpha}")


def _sequential_lif_with_surrogate(w_arr, n_ms_arr, tau, v_th, alpha):
    """序列版單狀態遞迴(lax.scan,不分 chunk),fire 判斷用 atan_spike。

    跟 process_chunk 同一個語意(不套閘):只有 fire 時才 soft reset,V = (1-s)*h,forward 等於 0;
    沒 fire 時 h 直接往下傳,不乘 (1-s)。理由見 docs/問題紀錄.md「決策:不套閘 + soft reset,不是硬 reset」。
    """
    def step(v, inputs):
        n_ms, w = inputs
        a = (1.0 - 1.0 / tau) ** n_ms
        h = v * a + w
        s = atan_spike(h - v_th, alpha)
        fired = jax.lax.stop_gradient(s) >= 0.5
        v_after_fire = (1.0 - s) * h
        v_new = jnp.where(fired, v_after_fire, h)
        return v_new, (h, s)

    v_final, (h_seq, s_seq) = jax.lax.scan(step, jnp.asarray(0.0), (n_ms_arr, w_arr))
    return v_final, h_seq, s_seq


def test_gradient_flows_through_fire_reset():
    """手算例子:tau=4,v_th=1.0,N=[0,1,4],w=[0.6,0.6,0.9],在第 2 筆事件(index 1)fire 一次,最終 V=0.9。

    先確認換成 atan_spike 之後 forward(h 序列、fire 位置、最終電壓)跟硬判斷一樣。
    再對 loss = sum(s) 求三個權重的梯度,跟手算的鏈式法則比(不套閘:沒 fire 的 v0_post、v2_post
    直接是 h,不含 s 的修正):
      s2 只有直接項(最後一筆,沒有下游):
        d s2/d w2 = slope(h2 - v_th) = slope(-0.1) ≈ 0.910170
      s1 的直接項 slope(0.05) ≈ 0.975919;fire 之後(s1=1)經過 soft reset 把帶負號的間接梯度傳給
        h2 再傳給 s2,兩項相加 d(loss)/d w1 ≈ 0.680819。
      s0 沒有直接項(不 fire),間接路徑 v0_post=h0 -> h1 -> s1(fire,soft reset 的負修正)-> h2 -> s2,
        d(loss)/d w0 ≈ 0.898341。每步都乘 (1-s) 的套閘版會是 0.779554,只有這一項受套不套閘影響。
    """
    tau = 4.0
    v_th = 1.0
    alpha = 2.0
    n_ms_arr = jnp.array([0.0, 1.0, 4.0])
    w_arr = jnp.array([0.6, 0.6, 0.9])

    v_final, h_seq, s_seq = _sequential_lif_with_surrogate(w_arr, n_ms_arr, tau, v_th, alpha)
    assert_allclose(v_final, 0.9, "forward 電壓應該跟硬判斷版一致", tol=1e-4)
    assert_allclose(h_seq[0], 0.6, "h0")
    assert_allclose(h_seq[1], 1.05, "h1", tol=1e-4)
    assert_allclose(h_seq[2], 0.9, "h2", tol=1e-4)
    assert list(map(float, s_seq)) == [0.0, 1.0, 0.0], "應該只在第 2 筆事件 fire"

    def loss_fn(w):
        _, _, s_seq = _sequential_lif_with_surrogate(w, n_ms_arr, tau, v_th, alpha)
        return jnp.sum(s_seq)

    grad = jax.grad(loss_fn)(w_arr)
    grad = [float(g) for g in grad]

    for g in grad:
        assert jnp.isfinite(jnp.asarray(g)), f"梯度必須是有限值: {grad}"
        # slope = (alpha/2)/(1+(pi/2*alpha*x)^2) 對任何有限 x 都 > 0,乘出來的梯度不會是 0
        assert g != 0.0, f"梯度不該剛好是 0(surrogate slope 處處嚴格 > 0): {grad}"

    assert_allclose(grad[0], 0.898341, "d(loss)/dw0(手算鏈式法則核對)", tol=1e-4)
    assert_allclose(grad[1], 0.680819, "d(loss)/dw1(手算鏈式法則核對)", tol=1e-4)
    assert_allclose(grad[2], 0.910170, "d(loss)/dw2(手算鏈式法則核對)", tol=1e-4)
