"""簡化版群體編碼(population coding)loss 的梯度驗證:loss 不是單一神經元
自己的 s 加總,是「跨神經元」的線性組合,例如 L=S_1-S_2(每顆神經元自己
sum(s_value)之後再做線性組合)——對應論文/實務常見的「用哪個神經元 fire
比較多次來判斷分類結果」這種頻率編碼,不是完整的 softmax cross-entropy
(那個手算量大很多,留給之後有需要再做)。

沿用 docs/math/全連接forward訓練範例.md 第 3 節的 n=2(a1,a2)、m=2(b1,b2) FC
結構(跟 test_fc_forward.py 完全一樣的設定):tau=4,v_th=1.0,上游事件
(t=1,a1)(t=2,a2)(t=4,a1),W=[[0.6,0.5],[0.3,0.2]]。b1 在 t=4(事件 index2)
fire 一次,b2 全程不 fire。

這個結構有一個之前測試都沒有覆蓋到的重點:**同一個權重被同一個神經元的
兩筆不同事件共用**(b1 的事件0、事件2 都來自 a1,共用 w11;b2 同理共用
w12)——因為全連接下,同一個來源神經元多次觸發時,權重矩陣裡的那個值不會
變,梯度要正確地把兩筆事件各自的貢獻加總。

手算鏈式法則逐項核對過(見程式碼裡的計算),跟 jax.grad 實際跑出來的結果
吻合(誤差在四捨五入範圍內),tol 用 1e-3 涵蓋手算累積的捨入誤差。
"""

import jax
import jax.numpy as jnp

from salt_core.float.scan import run_layer_forward
from salt_core.connectivity.fc import build_fc_structure, fc_float_values

TOL = 1e-3

# 手算鏈式法則逐項核對過的梯度(見本檔案 docstring 的結構說明):
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
    _, _, _, s_value, _ = run_layer_forward(maps, v_th, chunk_size=chunk_size,
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
    """chunk_size=3:同一個群體編碼 loss,驗證梯度跟 chunk_size=1 完全一致,
    包含「同一權重被同一神經元的兩筆事件共用」這件事在 chunk 化窗口下
    也不會算錯。"""
    _check_grad_matches_reference(chunk_size=3)
