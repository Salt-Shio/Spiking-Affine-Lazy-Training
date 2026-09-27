"""驗證 surrogate gradient 真的接進 chunk_scan.run_layer_forward 的
argmax-based pipeline,不是只在 core.process_chunk 單一 chunk 或
test_surrogate.py 的簡化序列遞迴裡才work。

作法:重用 test_surrogate.py 已經手算鏈式法則驗證過的例子(tau=4,v_th=1.0,
N=[0,1,4],w=[0.6,0.6,0.9],梯度 [0.898341, 0.680819, 0.910170]——「不套閘」
語意,只有真的 fire 才套 soft reset,見 test_surrogate.py 的完整說明),包裝
成一個 n=3(三個各自只發一次事件的來源神經元)、m=1 的 FC 佇列。loss 定義
成 sum(s_value):s_value 是「這個 chunk 步驟裡,把有效範圍內每筆真實事件
自己的 s 全部加起來」(見 chunk_scan.py run_layer_forward 的說明,不是只挑
一個代表值),所以不管切成幾個 chunk、fire 發生在哪一步,加總起來永遠等於
「每一筆事件自己的 s 都恰好算一次」——這正是 test_surrogate.py 序列版參考
在算的東西,梯度應該精確相等,不需要重新手算。

用 chunk_size=1(每個事件自己一步,沒有任何「一個 chunk 塞多筆事件」的情況)
跟 chunk_size=3(整條事件塞進單一 chunk,process_chunk 內部真的會用到 argmax
在多個候選事件裡挑第一個 fire,而且 fire 位置之前還有事件)分別跑一次,驗證
兩者都精確吻合,不只是驗證「argmax 離散選擇不會破壞梯度」,也驗證了「fire
位置之前的事件,梯度不會被 chunk 化的窗口悄悄漏算」。
"""

import jax
import jax.numpy as jnp

from salt_core.chunk_scan import run_layer_forward
from salt_core.connectivity.fc import build_fc_structure, fc_float_values

TOL = 1e-4

# test_surrogate.py::test_gradient_flows_through_fire_reset 手算鏈式法則核對過的梯度
# (「不套閘」語意:只有真的 fire 才套 soft reset)
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
    # s_value 本身已經是「有效範圍內每筆真實事件的 s 加總」(見 chunk_scan.py
    # run_layer_forward 的說明),包含空轉步驟自動貢獻 0 這件事,直接加總
    # 整個陣列即可,不需要呼叫端再另外處理。
    return jnp.sum(s_value)


def _check_grad_matches_reference(chunk_size):
    tau = 4.0
    v_th = 1.0
    alpha = 2.0

    # 事件時間 [0,1,5] 對應 N=[0,1,4](prepend=0),三個來源各自只發一次事件,
    # 權重 W[0]=[0.6,0.6,0.9] 剛好對上 test_surrogate.py 的 w=[0.6,0.6,0.9]
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
    """chunk_size=3(整條事件塞進單一 chunk),fire 發生在 event1,event0 是
    fire 位置之前的事件——驗證梯度跟 chunk_size=1 完全一致,證明 argmax 這個
    離散選擇沒有破壞梯度,而且 fire 位置之前的事件自己的 s 也確實被算進去,
    不會因為窗口切法不同就漏算或多算。"""
    _check_grad_matches_reference(chunk_size=3)


# docs/math/不套閘與soft-reset梯度推導.md 第 7 節手算過的 v_final 梯度(同一組
# tau=4,v_th=1.0,N=[0,1,4],w=[0.6,0.6,0.9] 例子,g_1=-slope_1*x_1≈-1.024711):
#   d(v_final)/dw1 = g_1                ≈ -1.024711(fire 事件自己)
#   d(v_final)/dw0 = g_1 * a_1          ≈ -0.768533(fire 之前的事件)
#   d(v_final)/dw2 = 0                  (fire 之後被丟棄的事件,計算圖裡
#                                         根本沒有引用過,不是遮罩湊出來的)
# 注意:這是「單一個 chunk 呼叫、fire 之後立刻停下來」的 v_final,不是
# run_layer_forward 把 event2 也接著處理完之後的最終電壓(event2 沒 fire,
# 會被接到 event1 fire 後的 v_final 上繼續往下算,變成另一個數字、另一組
# 梯度)——所以下面呼叫 run_layer_forward 時,max_steps 刻意設成「剛好停在
# fire 那一步」,不讓它繼續消化 event2,才會對上這裡的期望值。
EXPECTED_GRAD_V_FINAL = [-0.768533, -1.024711, 0.0]


def _loss_v_final(W, event_times, event_source_idx, tau, v_th, chunk_size, max_steps, alpha):
    """膜電位回歸型的 loss:直接對 v_final 求梯度,不透過任何 s_value——
    對照 sum(s_value) 那種頻率編碼型 loss,驗證 process_chunk 對外交付的
    兩種量(v_final、s_value)各自的梯度路徑都是對的,不是只驗過其中一種。"""
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
    """chunk_size=3、max_steps=1:三筆事件塞進同一個 chunk,一步處理完就停,
    fire 發生在 event1,event2 從頭到尾沒被讀取過(不是遮罩掉,是計算圖裡
    根本沒有這條邊)——直接對應推導文件第 7 節分析的那個單一 chunk 呼叫。"""
    _check_v_final_grad_matches_reference(chunk_size=3, max_steps=1)


def test_v_final_regression_gradient_two_small_chunks():
    """chunk_size=1、max_steps=2:event0、event1 各自一步,event1 fire 後
    scan 就停了(max_steps=2,不會有第三步去處理 event2)——跟上一個測試
    是同一個數學,只是切成兩個小 chunk 而不是一個大 chunk,梯度應該完全
    一致,驗證這個「在 fire 後立刻停止」的場景本身也不受 chunk_size 影響。"""
    _check_v_final_grad_matches_reference(chunk_size=1, max_steps=2)
