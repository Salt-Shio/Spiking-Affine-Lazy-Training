"""chunk_scan.run_layer_forward 在多次 fire、不同 chunk_size 下的壓力測試。

拿一個會連續 fire 兩次的 7 筆事件序列,跑過 chunk_size = 1,2,3,4,7(含事件總數,
整條佇列當一個 chunk)分別驗證,結果都要跟逐筆序列的 oracle 完全一致——這是在
驗證「fire 後從下一筆事件重新起跑」的邊界處理,在各種 chunk 切法(包含 fire 點
剛好落在 chunk 邊界、chunk 中間、以及整條事件塞進同一個 chunk 內部發生多次 fire)
下都不會算錯,比 test_fc_forward.py(只 fire 一次)跟
test_core.py::test_multi_chunk_worked_example(手動切兩個 chunk)更全面。
"""

import jax.numpy as jnp

from salt_core.chunk_scan import run_layer_forward
from salt_core.core import create_affine_maps

TOL = 1e-4


def _sequential_oracle(n_ms_list, w_list, tau, v_th):
    v = 0.0
    fires = []
    for i, (n, w) in enumerate(zip(n_ms_list, w_list)):
        a = (1.0 - 1.0 / tau) ** n
        h = v * a + w
        if h >= v_th:
            fires.append(i)
            v = 0.0
        else:
            v = h
    return fires, v


def test_multi_fire_matches_oracle_across_chunk_sizes():
    tau = 4.0
    v_th = 1.0
    n_ms_list = [1, 1, 1, 1, 1, 1, 1]
    w_list = [0.5, 0.6, 0.3, 0.9, 0.2, 0.95, 0.1]
    n_real_events = len(n_ms_list)

    expected_fires, expected_v = _sequential_oracle(n_ms_list, w_list, tau, v_th)
    assert expected_fires == [2, 5], expected_fires  # 確認這組假資料真的會連續 fire 兩次,不是退化案例

    maps = create_affine_maps(jnp.array([n_ms_list]), jnp.array([w_list]), tau)

    for chunk_size in [1, 2, 3, 4, 7]:
        spike_mask, spike_event_idx, _s_spike, _s_value, v_final = run_layer_forward(
            maps, v_th, chunk_size=chunk_size, max_steps=n_real_events,
            n_real_events=n_real_events)

        got_fires = sorted(
            int(spike_event_idx[0, i])
            for i in range(spike_mask.shape[1])
            if bool(spike_mask[0, i])
        )
        assert got_fires == expected_fires, (
            f"chunk_size={chunk_size}: got fires {got_fires}, expected {expected_fires}")
        assert abs(float(v_final[0]) - expected_v) < TOL, (
            f"chunk_size={chunk_size}: got v_final={float(v_final[0])}, expected {expected_v}")
