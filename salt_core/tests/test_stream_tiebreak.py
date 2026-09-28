"""extract_output_events_fc 在同一時間戳記有多筆輸出時的排序:依輸入佇列的位置(spike_event_idx),
不是只看時間。理由見 docs/問題紀錄.md「洞見:同一時間戳記排序,用複合鍵不能只用時間」。

反例:p0 被佇列裡較晚的事件 Y fire,p1 被較早的事件 X fire,X、Y 時間相同。只用時間排序時同分會
退化成 jnp.nonzero 的順序(先神經元 0 再神經元 1),輸出 [p0(Y), p1(X)],順序反了;正確是 [p1(X), p0(Y)]。

構造:
  event_times = [1.0, 1.0](索引 0 是 X,索引 1 是 Y)
  event_source_idx = [0, 0](單一來源,兩筆共用同一個權重)
  W = [[0.6], [1.2]]
    p0(w=0.6):0.6 < v_th,兩筆疊加(N=0 不衰減)1.2 >= v_th,在索引 1(Y)fire
    p1(w=1.2):第一筆就 fire,在索引 0(X)
  chunk_size=2、max_steps=1:一步處理完整條佇列、fire 後就停,p1 不會在第二筆再 fire。
"""

import jax.numpy as jnp

from salt_core.float.scan import run_layer_forward
from salt_core.connectivity.fc import build_fc_structure, fc_float_values
from salt_core.stream import extract_output_events_fc

TOL = 1e-4


def assert_allclose(actual, expected, msg):
    actual = float(actual)
    expected = float(expected)
    assert abs(actual - expected) < TOL, f"{msg}: got {actual}, expected {expected}"


def test_tiebreak_uses_spike_event_idx_not_neuron_order():
    tau = 4.0
    v_th = 1.0

    event_times = jnp.array([1.0, 1.0])   # 索引0=X, 索引1=Y,刻意同分
    event_source_idx = jnp.array([0, 0])
    W = jnp.array([[0.6],
                   [1.2]])

    maps = fc_float_values(build_fc_structure(event_times, event_source_idx, event_times.shape[0]),
                           W, tau, None)
    spike_mask, spike_event_idx, s_spike, _, _ = run_layer_forward(
        maps, v_th, chunk_size=2, max_steps=1, n_real_events=maps.a.shape[1])

    # 前提:p0 在索引 1(Y)fire、p1 在索引 0(X)fire
    assert bool(spike_mask[0, 0]), "p0 應該要 fire"
    assert bool(spike_mask[1, 0]), "p1 應該要 fire"
    assert int(spike_event_idx[0, 0]) == 1, "p0 應該是被事件索引1(Y)觸發"
    assert int(spike_event_idx[1, 0]) == 0, "p1 應該是被事件索引0(X)觸發"

    times, source_idx, gain, n_real_events = extract_output_events_fc(
        spike_mask, spike_event_idx, s_spike, event_times)

    assert int(n_real_events) == 2, n_real_events
    # 兩筆輸出事件時間戳記都是 1.0(本來就是同分)
    assert_allclose(times[0], 1.0, "第一筆輸出事件時間")
    assert_allclose(times[1], 1.0, "第二筆輸出事件時間")

    # X 的 spike_event_idx(0)比 Y(1)小,排前面
    assert int(source_idx[0]) == 1, (
        f"tie-break 應該讓被較早事件(X,index=0)觸發的 p1 排第一筆,"
        f"實際 source_idx={list(map(int, source_idx))}")
    assert int(source_idx[1]) == 0, (
        f"被較晚事件(Y,index=1)觸發的 p0 應該排第二筆,"
        f"實際 source_idx={list(map(int, source_idx))}")


def test_tiebreak_with_three_neurons_not_in_row_order():
    """三顆神經元同一個時間戳記各 fire 一次,觸發的事件索引跟神經元編號錯開,排序要照
    spike_event_idx 遞增(B, C, A),不是神經元編號(A, B, C)。

    構造(單一來源,三筆事件同時間,N=0 不衰減):
      A(row0,w=0.4):0.4 -> 0.8 -> 1.2,在索引 2 fire
      B(row1,w=1.2):1.2,在索引 0 fire
      C(row2,w=0.6):0.6 -> 1.2,在索引 1 fire
    chunk_size=3、max_steps=1:一步處理完、fire 後就停。
    """
    tau = 4.0
    v_th = 1.0

    event_times = jnp.array([1.0, 1.0, 1.0])
    event_source_idx = jnp.array([0, 0, 0])
    W = jnp.array([[0.4],
                   [1.2],
                   [0.6]])

    maps = fc_float_values(build_fc_structure(event_times, event_source_idx, event_times.shape[0]),
                           W, tau, None)
    spike_mask, spike_event_idx, s_spike, _, _ = run_layer_forward(
        maps, v_th, chunk_size=3, max_steps=1, n_real_events=maps.a.shape[1])

    # 前提:三顆神經元各自在預期的索引 fire
    assert bool(spike_mask[:, 0].all()), "A、B、C 都應該要 fire"
    assert int(spike_event_idx[0, 0]) == 2, "A 應該是被事件索引2觸發"
    assert int(spike_event_idx[1, 0]) == 0, "B 應該是被事件索引0觸發"
    assert int(spike_event_idx[2, 0]) == 1, "C 應該是被事件索引1觸發"

    times, source_idx, gain, n_real_events = extract_output_events_fc(
        spike_mask, spike_event_idx, s_spike, event_times)

    assert int(n_real_events) == 3, n_real_events
    for i in range(3):
        assert_allclose(times[i], 1.0, f"第{i}筆輸出事件時間")

    # 照 spike_event_idx 遞增:B(idx0)、C(idx1)、A(idx2)
    expected_order = [1, 2, 0]  # B, C, A 的神經元 index
    got_order = list(map(int, source_idx[:3]))
    assert got_order == expected_order, (
        f"tie-break 應該照 spike_event_idx 遞增排序,預期 {expected_order}(B,C,A),"
        f"實際 {got_order}")
