"""驗證 surrogate.atan_spike 的 forward/backward 對不對,以及接上一個簡化的
序列版單狀態遞迴後,梯度真的能穿過 fire/reset 這個判斷點。

不測 float/scan.py 現有的 chunk 化/argmax 版本——「哪個事件是第一個 fire」
這個離散選擇本身怎麼處理梯度,是下一步才要處理的整合工作,見 docs/TODO.md
任務 5。這裡只驗證核心機制:單一事件的 surrogate 判斷,跟一個用它接起來的
簡化序列遞迴。
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
    """atan_spike 的 custom_vjp backward,要跟 atan_smooth(它宣稱對應的平滑
    原函數)用一般 autodiff 算出來的導數完全吻合——這是唯一能拿來核對
    「backward 公式刻得對不對」的客觀標準,因為硬階梯函數本身的真實梯度
    幾乎處處是 0,不能拿來當比較基準。"""
    for alpha in [1.0, 2.0, 4.0]:
        for x in [-3.0, -1.0, -0.1, 0.0, 0.1, 1.0, 3.0]:
            x_arr = jnp.asarray(x)
            got = float(jax.grad(atan_spike, argnums=0)(x_arr, alpha))
            expected = float(jax.grad(atan_smooth, argnums=0)(x_arr, alpha))
            assert_allclose(got, expected, f"backward mismatch at x={x}, alpha={alpha}")


def _sequential_lif_with_surrogate(w_arr, n_ms_arr, tau, v_th, alpha):
    """跟 process_chunk 一樣的單狀態遞迴,fire 判斷換成 atan_spike。純序列
    lax.scan,不是 float/scan.py 的 chunk 化版本,但要跟它用同一套「不套閘」
    語意:只有真的 fire 時才套用可微分的 soft reset(V_new=(1-s)*h,forward
    精確等於硬重置的 0),沒 fire 就讓 h 原始值直接往下傳,不對它套用任何
    surrogate 修正——不是每一步都無條件乘 (1-s)。

    這個選擇不是圖省事:「每一步都套閘」(spikingjelly 逐 tick 的寫法)會讓
    平行 chunk 化的 associative_scan 沒辦法用(套閘後每一步不再是仿射函數,
    見對話記錄的完整討論);「只在 fire 時套閘」則跟 Bullet Trains 的精神
    一致(修正只發生在真正的決策點),而且剛好等於 float/affine.py/float/scan.py
    現有實作已經在做的事,不需要額外的 custom_vjp。
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
    """TODO.md 手算過的例子(tau=4,v_th=1.0,N=[0,1,4],w=[0.6,0.6,0.9],
    在第 2 筆事件(0-based index 1)fire 一次,最終 V=0.9)。

    先確認換成 atan_spike 之後,forward 數字(h 序列、fire 位置、最終電壓)
    跟純硬判斷版完全一致——surrogate 不該改變前向行為。

    再拿 loss = sum(s) 對三個權重求梯度,用鏈式法則手算逐項核對(不是只看
    程式碼吐出什麼就信什麼)。注意這裡是「不套閘」語意(只有真的 fire 才
    套 soft reset,沒 fire 的 v0_post/v2_post 是原始值 h 直接往下傳,不含
    任何 s 的修正項):
      s2 只有直接項(它是最後一筆事件,沒有下游):
        d s2/d w2 = slope(h2-v_th) = slope(-0.1) ≈ 0.910170
      s1 的直接項是 slope(h1-v_th)=slope(0.05)≈0.975919,它 fire 之後
        (s1=1)透過 soft reset 把 v1_post 帶負的間接梯度傳給 h2,再傳給 s2,
        兩項相消後 d(loss)/d w1 ≈ 0.680819(這條路徑一定會經過 s1 自己的
        fire 分支,不管套不套閘都一樣,所以這個數字沒有變)。
      s0 沒有直接項(s0=0,不 fire)。它的間接路徑是:v0_post=h0(不套閘,
        直接傳,不含 s0 的修正)→ h1 → s1(這條會 fire,帶 soft reset 的
        負修正)→ h2 → s2。因為 v0_post 這一步沒有套閘、少了 spikingjelly
        式「每步都乘 (1-s)」會有的額外修正項,d(loss)/d w0 ≈ 0.898341,
        比套閘版本算出來的 0.779554 大——這個差異正是「套不套閘」這個選擇
        唯一會影響到的地方(w1、w2 的梯度鏈都沒經過任何一個「沒 fire」的
        閘,所以完全不受影響)。
    三個都用獨立算出來的鏈式法則數字核對,不是拿程式碼自己驗自己。
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
        # atan_backward 的 slope = (alpha/2)/(1+(pi/2*alpha*x)^2) 對任何有限 x
        # 都嚴格 > 0,所以鏈式法則乘出來的梯度不可能剛好等於 0——這點不用
        # 看數字多少,結構上就保證了。
        assert g != 0.0, f"梯度不該剛好是 0(surrogate slope 處處嚴格 > 0): {grad}"

    assert_allclose(grad[0], 0.898341, "d(loss)/dw0(手算鏈式法則核對)", tol=1e-4)
    assert_allclose(grad[1], 0.680819, "d(loss)/dw1(手算鏈式法則核對)", tol=1e-4)
    assert_allclose(grad[2], 0.910170, "d(loss)/dw2(手算鏈式法則核對)", tol=1e-4)
