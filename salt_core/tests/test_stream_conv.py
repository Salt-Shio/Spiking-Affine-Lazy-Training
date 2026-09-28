"""extract_output_events_conv:局部欄查回全域事件 index、排序、打包。

1. 推導文件(docs/math/conv事件佇列壓縮版推導.md「三個函式的分工」)的手算例子。
2. 同一筆全域事件讓兩顆神經元同時 fire 時,照神經元 index 由小到大。
3. build_conv_structure + conv_float_values -> run_layer_forward -> extract_output_events_conv,
   跟參考實作(_reference.dense_conv_affine_map -> run_layer_forward -> extract_output_events_fc)比
   EventStream 四個欄位。event_gain 放錯位置不會讓 forward 跑掉,只會讓下一層的梯度算錯,所以四個都比。
"""

import jax.numpy as jnp

from salt_core.float.scan import run_layer_forward
from salt_core.connectivity.conv import build_conv_structure, conv_float_values, tile_channels
from salt_core.tests._reference import dense_conv_affine_map
from salt_core.stream import extract_output_events_fc, extract_output_events_conv

TOL = 1e-6


def assert_allclose(actual, expected, msg, tol=TOL):
    actual = float(actual)
    expected = float(expected)
    assert abs(actual - expected) < tol, f"{msg}: got {actual}, expected {expected}"


# ============================================================================
# 1. 推導文件的手算例子
# ============================================================================

def test_extract_output_events_conv_matches_worked_example():
    """推導文件的例子:事件時間 [1,2,3,5]、神經元 5/6/7、v_th=0.6。神經元 5 在局部欄 2 fire、神經元 7
    在局部欄 0 fire、神經元 6 不 fire;查表 神經元5->[j=0,1,3]、神經元6->[j=2,3,pad]、神經元7->[j=1,pad,pad]。
    預期:event_source_idx=[7,5,...]、event_times=[2,5,...]、n_real_events=2。
    直接合成 spike_mask、spike_event_idx、local_to_global_j,神經元 id 用 0,1,2 對應文件的 5,6,7。
    """
    # 神經元 id:0->文件神經元5, 1->文件神經元6, 2->文件神經元7
    # max_steps=1:只關心有沒有 fire、fire 在哪個局部欄
    spike_mask = jnp.array([[True], [False], [True]])
    spike_event_idx = jnp.array([[2], [0], [0]])  # 神經元0 局部欄位2、神經元2 局部欄位0
    s_spike = jnp.array([[0.9], [0.0], [0.8]])  # 隨便挑的強度
    event_times = jnp.array([1.0, 2.0, 3.0, 5.0])  # 這層自己的輸入事件時間(j=0..3)
    local_to_global_j = jnp.array([
        [0, 1, 3],  # 神經元0(文件神經元5)
        [2, 3, 4],  # 神經元1(文件神經元6),4=sentinel(=n_events,pad)
        [1, 4, 4],  # 神經元2(文件神經元7)
    ])

    result = extract_output_events_conv(spike_mask, spike_event_idx, s_spike, event_times,
                                        local_to_global_j, max_total_spikes=4)

    assert int(result.n_real_events) == 2, result.n_real_events
    # 排序後應該是 [神經元2(全域j=1,t=2), 神經元0(全域j=3,t=5)]
    assert list(map(int, result.event_source_idx[:2])) == [2, 0], \
        f"預期 [神經元2,神經元0],實際 {list(map(int, result.event_source_idx[:2]))}"
    assert_allclose(result.event_times[0], 2.0, "第一筆輸出時間")
    assert_allclose(result.event_times[1], 5.0, "第二筆輸出時間")
    assert_allclose(result.event_gain[0], 0.8, "第一筆輸出增益(神經元2自己的s_spike)")
    assert_allclose(result.event_gain[1], 0.9, "第二筆輸出增益(神經元0自己的s_spike)")
    # pad 位置的時間應該是保證超出真實時間的大數(_PAD_TIME),不是真實時間
    assert float(result.event_times[2]) > 1e6, "pad 位置應該是超大時間值,不是真實時間"
    assert float(result.event_times[3]) > 1e6, "pad 位置應該是超大時間值,不是真實時間"


# ============================================================================
# 2. 同一筆全域事件讓多顆神經元同時 fire
# ============================================================================

def test_tiebreak_stable_when_multiple_neurons_share_same_global_j():
    """神經元 0、2 的 spike 都對到全域 j=5(同一筆事件落在兩顆神經元的感受野裡)。
    順序要確定:照 jnp.nonzero 的掃描順序,神經元 id 小的在前。"""
    spike_mask = jnp.array([[True], [False], [True]])
    spike_event_idx = jnp.array([[0], [0], [0]])
    s_spike = jnp.array([[1.0], [0.0], [1.0]])
    event_times = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    # 神經元0、神經元2 的局部欄位0 都對應全域 j=5(同一個來源事件)
    local_to_global_j = jnp.array([
        [5],
        [6],  # 用不到(神經元1 不 fire)
        [5],
    ])

    result = extract_output_events_conv(spike_mask, spike_event_idx, s_spike, event_times,
                                        local_to_global_j, max_total_spikes=4)

    assert int(result.n_real_events) == 2, result.n_real_events
    assert list(map(int, result.event_source_idx[:2])) == [0, 2], \
        (f"j 相同時應保留原始掃描順序(神經元0在前),"
         f"實際 {list(map(int, result.event_source_idx[:2]))}")
    assert_allclose(result.event_times[0], 60.0, "兩筆都對應全域j=5的時間")
    assert_allclose(result.event_times[1], 60.0, "兩筆都對應全域j=5的時間")


# ============================================================================
# 3. 端到端:conv 整條 vs 參考實作整條,四個欄位全比
# ============================================================================

TAU = 4.0
K, S, P = 3, 2, 1
H_OUT = W_OUT = 3


def _make_weight(oc_offset=0.0):
    base = jnp.array([[ky * 3 + kx + 1 for kx in range(3)] for ky in range(3)],
                      dtype=jnp.float32)
    return (base + oc_offset)[None, None, :, :]


def _extracted_events_equal(a, b, tol=1e-4):
    n = int(a.n_real_events)
    assert n == int(b.n_real_events), (int(a.n_real_events), int(b.n_real_events))
    assert bool(jnp.allclose(a.event_times[:n], b.event_times[:n], atol=tol)), \
        (a.event_times[:n], b.event_times[:n])
    assert bool(jnp.array_equal(a.event_source_idx[:n], b.event_source_idx[:n])), \
        (a.event_source_idx[:n], b.event_source_idx[:n])
    assert bool(jnp.allclose(a.event_gain[:n], b.event_gain[:n], atol=tol)), \
        (a.event_gain[:n], b.event_gain[:n])


def _conv_queue(event_times, x, y, c, W, max_queue_len, n_real_events):
    """結構 + 浮點數值段。回傳 (maps, 逐神經元 n_real_events, 逐神經元 local_to_global_j)。"""
    oc = W.shape[0]
    structure = build_conv_structure(event_times, x, y, c, K, S, P, H_OUT, W_OUT,
                                     max_queue_len, n_real_events)
    return (conv_float_values(structure, W, TAU, None),
            tile_channels(structure.n_real_events, oc),
            tile_channels(structure.local_to_global_j, oc))


def test_end_to_end_matches_dense_all_four_fields():
    """conv 整條跟參考實作整條比 EventStream 四個欄位。場景會真的 fire(idx0 在局部欄 1、全域 j=2),
    s_spike 不是全 0 或全 1。"""
    W = _make_weight()
    event_times = jnp.array([1.0, 1.5, 2.0, 2.5, 5.0])
    x = jnp.array([1, 2, 1, 2, 0]); y = jnp.array([1, 2, 1, 2, 0]); c = jnp.array([0, 0, 0, 0, 0])
    v_th = 15.0
    max_queue_len = 5
    max_spikes = 9 * 5  # 一定裝得下

    # 參考實作整條
    maps_dense = dense_conv_affine_map(event_times, x, y, c, W, TAU, S, P, H_OUT, W_OUT)
    result_dense = run_layer_forward(maps_dense, v_th, chunk_size=5, max_steps=5, n_real_events=maps_dense.a.shape[1])
    ev_dense = extract_output_events_fc(result_dense.spike_mask, result_dense.spike_event_idx,
                                        result_dense.s_spike, event_times,
                                        max_total_spikes=max_spikes)

    # conv 整條
    maps, n_real_per_neuron, local_to_global_j = _conv_queue(
        event_times, x, y, c, W, max_queue_len, event_times.shape[0])
    result_conv = run_layer_forward(maps, v_th, chunk_size=5, max_steps=5,
                                           n_real_events=n_real_per_neuron)
    ev_conv = extract_output_events_conv(result_conv.spike_mask,
                                               result_conv.spike_event_idx,
                                               result_conv.s_spike, event_times,
                                               local_to_global_j,
                                               max_total_spikes=max_spikes)

    assert int(ev_dense.n_real_events) >= 1, "前置確認:這個場景至少要有一筆真的輸出事件"
    _extracted_events_equal(ev_dense, ev_conv)


def test_end_to_end_matches_dense_with_multiple_fires_and_pad_input():
    """輸入帶 pad 事件(n_real_events < 陣列長度),而且多顆神經元 fire,多筆輸出事件四個欄位都要對。"""
    W = _make_weight()
    # 前 5 筆是真事件,第 6 筆是 pad(座標偽裝合法、時間超大)
    event_times = jnp.array([1.0, 1.5, 2.0, 2.5, 5.0, 1e12])
    x = jnp.array([1, 2, 1, 2, 0, 0]); y = jnp.array([1, 2, 1, 2, 0, 0])
    c = jnp.array([0, 0, 0, 0, 0, 0])
    n_real_events = 5
    v_th = 5.0  # 調低一點,讓不只一顆神經元 fire
    max_queue_len = 5
    max_spikes = 9 * 5

    maps_dense = dense_conv_affine_map(event_times, x, y, c, W, TAU, S, P, H_OUT, W_OUT,
                                       n_real=n_real_events)
    result_dense = run_layer_forward(maps_dense, v_th, chunk_size=6, max_steps=6,
                                      n_real_events=n_real_events)
    ev_dense = extract_output_events_fc(result_dense.spike_mask, result_dense.spike_event_idx,
                                        result_dense.s_spike, event_times,
                                        max_total_spikes=max_spikes)

    maps, n_real_per_neuron, local_to_global_j = _conv_queue(
        event_times, x, y, c, W, max_queue_len, n_real_events)
    result_conv = run_layer_forward(maps, v_th, chunk_size=max_queue_len,
                                           max_steps=max_queue_len,
                                           n_real_events=n_real_per_neuron)
    ev_conv = extract_output_events_conv(result_conv.spike_mask,
                                               result_conv.spike_event_idx,
                                               result_conv.s_spike, event_times,
                                               local_to_global_j,
                                               max_total_spikes=max_spikes)

    assert int(ev_dense.n_real_events) >= 2, \
        f"前置確認:這個場景至少要有兩顆神經元 fire 才夠測排序,實際 {int(ev_dense.n_real_events)}"
    _extracted_events_equal(ev_dense, ev_conv)
