"""驗證 chunk_scan.run_layer_forward 的梯度不受 chunk_size 影響,涵蓋比
test_chunk_scan_gradient.py 更多種 chunk_size、而且事件序列裡有連續兩次 fire
(不是只 fire 一次)。

沿用 test_chunk_scan_stress.py 的 7 事件、連續 fire 兩次(0-based index 2、5)
的例子,包裝成 n=7(七個各自只發一次事件的來源神經元)、m=1 的 FC 佇列。
用 chunk_size=1 的梯度當 oracle(不需要另外手算):chunk_size=1 時每個事件
自己獨占一個 scan 步驟,定義上就等於逐筆序列參考,可信度不需要靠額外驗算
建立。掃過 chunk_size ∈ {1,2,3,4,7} 之後,全部應該精確吻合 chunk_size=1
的結果——這是在驗證任意 chunk 切法(包含 fire 點卡在 chunk 邊界、chunk 中間,
以及整條事件塞進單一 chunk 內部發生兩次 fire)都不會讓梯度跑掉。
"""

import jax
import jax.numpy as jnp

from salt_core.chunk_scan import run_layer_forward
from salt_core.connectivity.fc import build_fc_queue

TOL = 1e-4

CHUNK_SIZES = [1, 2, 3, 4, 7]


def assert_allclose(actual, expected, msg, tol=TOL):
    actual = float(actual)
    expected = float(expected)
    assert abs(actual - expected) < tol, f"{msg}: got {actual}, expected {expected}"


def _loss(W, event_times, event_source_idx, tau, v_th, chunk_size, alpha):
    maps = build_fc_queue(event_times, event_source_idx, W, tau, n_real_events=event_times.shape[0]).maps
    n_real_events = event_times.shape[0]
    _, _, _, s_value, _ = run_layer_forward(maps, v_th, chunk_size=chunk_size,
                                            max_steps=n_real_events, alpha=alpha,
                                            n_real_events=n_real_events)
    return jnp.sum(s_value)


def test_gradient_invariant_across_chunk_sizes_with_two_fires():
    tau = 4.0
    v_th = 1.0
    alpha = 2.0

    # 對應 test_chunk_scan_stress.py 的 n_ms_list=[1]*7(事件間隔全部是 1ms,
    # 從 t=0 起算,累加成 event_times=[1..7]),w_list 完全相同,在 index 2、5
    # 各 fire 一次
    event_times = jnp.arange(1.0, 8.0)
    event_source_idx = jnp.arange(7)
    W = jnp.array([[0.5, 0.6, 0.3, 0.9, 0.2, 0.95, 0.1]])

    n_real_events = event_times.shape[0]
    grad_fn = jax.grad(_loss)

    oracle = grad_fn(W, event_times, event_source_idx, tau, v_th, 1, alpha)
    assert oracle.shape == (1, 7)
    assert all(jnp.isfinite(oracle).flatten().tolist())
    assert any(abs(float(g)) > 0.0 for g in oracle[0]), "oracle 梯度不該全部是 0"

    for chunk_size in CHUNK_SIZES:
        grad_W = grad_fn(W, event_times, event_source_idx, tau, v_th, chunk_size, alpha)
        assert grad_W.shape == (1, 7)
        for i in range(7):
            assert_allclose(grad_W[0, i], oracle[0, i],
                             f"chunk_size={chunk_size}: d(loss)/dW[0,{i}] 應該跟 chunk_size=1 一致")
