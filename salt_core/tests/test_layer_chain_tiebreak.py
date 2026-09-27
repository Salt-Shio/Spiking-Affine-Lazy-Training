"""驗證 layer_chain.extract_output_events 同一時間戳記多來源並列時的排序規則:
用 (times, spike_event_idx) 複合鍵排序,不是只用 times。

反例(源自跟另一個 agent 討論時驗證過的具體案例):$p_0$(layer1 的神經元0)
被佇列裡索引比較晚的事件 Y 觸發 fire,$p_1$(神經元1)被索引比較早的事件 X
觸發 fire,X、Y 兩筆事件的時間戳記量化後剛好相等(這裡直接用完全相同的
event_times 數值構造,不需要另外做量化)。

如果排序只用 times 當 key,同分會退化成 jnp.nonzero 攤平出來的順序——固定
是先掃神經元0、再掃神經元1(跟真實時間先後完全無關),所以會輸出 [p0(Y),
p1(X)],把真正先發生的 X 排到後面,順序是反的。

改用 (spike_event_idx, times) 當複合鍵(lexsort 最後一個 key 是主鍵)之後,
tie-break 依據變成「這筆事件在共用佇列裡的原始位置」——FC 無延遲,佇列本身
的排列順序天生就是真實時間先後順序(見 fc_queue.py、layer_chain.py 的說明),
X 的 index 比 Y 小,理當排在前面,輸出應該是 [p1(X), p0(Y)]。

具體構造(tau、v_th 對這個測試不重要,只要能精確控制哪個神經元在哪個事件
fire 即可):
  event_times = [1.0, 1.0](索引0=X,索引1=Y,刻意做成同一個時間戳記)
  event_source_idx = [0, 0](單一來源,兩筆事件共用同一個權重)
  W = [[0.6], [1.2]]
    p0(row0,w=0.6): 單筆事件不夠 fire(0.6<v_th),兩筆疊加(N=0,不衰減)
      0.6+0.6=1.2>=v_th,在事件索引1(=Y)才 fire
    p1(row1,w=1.2): 單筆事件就夠 fire(1.2>=v_th),在事件索引0(=X)立刻 fire
  用 chunk_size=2(兩筆事件塞進同一個 chunk)、max_steps=1,讓 process_chunk
  一次處理完整條佇列、fire 後立刻停,避免 p1 fire 之後在第二筆事件又重複
  fire 把測試複雜化。
"""

import jax.numpy as jnp

from salt_core.chunk_scan import run_layer_forward
from salt_core.connectivity.fc import build_fc_structure, fc_float_values
from salt_core.layer_chain import extract_output_events

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

    # 前置確認:p0 真的是被索引1(Y)觸發,p1 真的是被索引0(X)觸發——這是
    # 這個反例成立的前提,不是這次要驗證的重點,但要先確認前提沒有搭錯
    assert bool(spike_mask[0, 0]), "p0 應該要 fire"
    assert bool(spike_mask[1, 0]), "p1 應該要 fire"
    assert int(spike_event_idx[0, 0]) == 1, "p0 應該是被事件索引1(Y)觸發"
    assert int(spike_event_idx[1, 0]) == 0, "p1 應該是被事件索引0(X)觸發"

    times, source_idx, gain, n_real_events = extract_output_events(
        spike_mask, spike_event_idx, s_spike, event_times)

    assert int(n_real_events) == 2, n_real_events
    # 兩筆輸出事件時間戳記都是 1.0(本來就是同分)
    assert_allclose(times[0], 1.0, "第一筆輸出事件時間")
    assert_allclose(times[1], 1.0, "第二筆輸出事件時間")

    # 重點:排序結果應該是 [p1(X), p0(Y)],不是 [p0(Y), p1(X)]——
    # X 的 spike_event_idx(=0)比 Y(=1)小,真正先發生,tie-break 之後應該排前面
    assert int(source_idx[0]) == 1, (
        f"tie-break 應該讓被較早事件(X,index=0)觸發的 p1 排第一筆,"
        f"實際 source_idx={list(map(int, source_idx))}")
    assert int(source_idx[1]) == 0, (
        f"被較晚事件(Y,index=1)觸發的 p0 應該排第二筆,"
        f"實際 source_idx={list(map(int, source_idx))}")


def test_tiebreak_with_three_neurons_not_in_row_order():
    """把反例從 2 顆神經元擴大到 3 顆:三顆神經元同一個時間戳記各自 fire 一次,
    觸發它們的事件索引刻意跟神經元編號完全錯開(A 被最晚的索引2觸發、B 被
    最早的索引0觸發、C 被中間的索引1觸發),確認 lexsort 排序結果是照
    spike_event_idx 遞增(B,C,A),不是退化成神經元編號順序(A,B,C)。

    構造(單一來源,三筆事件同一時間戳記,N=0 不衰減,直接累加):
      A(row0,w=0.4): 0.4 -> 0.8 -> 1.2(在索引2 fire)
      B(row1,w=1.2): 1.2(在索引0 立刻 fire)
      C(row2,w=0.6): 0.6 -> 1.2(在索引1 fire)
    chunk_size=3、max_steps=1,process_chunk 一次處理完整條佇列、fire 後
    立刻停,避免任何一顆神經元 reset 後在同一個 chunk 內又重複 fire。
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

    # 前置確認:三顆神經元各自在預期的事件索引 fire
    assert bool(spike_mask[:, 0].all()), "A、B、C 都應該要 fire"
    assert int(spike_event_idx[0, 0]) == 2, "A 應該是被事件索引2觸發"
    assert int(spike_event_idx[1, 0]) == 0, "B 應該是被事件索引0觸發"
    assert int(spike_event_idx[2, 0]) == 1, "C 應該是被事件索引1觸發"

    times, source_idx, gain, n_real_events = extract_output_events(
        spike_mask, spike_event_idx, s_spike, event_times)

    assert int(n_real_events) == 3, n_real_events
    for i in range(3):
        assert_allclose(times[i], 1.0, f"第{i}筆輸出事件時間")

    # 重點:排序結果應該照 spike_event_idx 遞增排成 [B(idx0), C(idx1), A(idx2)],
    # 不是退化成神經元編號順序 [A(row0), B(row1), C(row2)]
    expected_order = [1, 2, 0]  # B, C, A 的神經元 index
    got_order = list(map(int, source_idx[:3]))
    assert got_order == expected_order, (
        f"tie-break 應該照 spike_event_idx 遞增排序,預期 {expected_order}(B,C,A),"
        f"實際 {got_order}")
