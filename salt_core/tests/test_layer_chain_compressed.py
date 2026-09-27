"""驗證 layer_chain.extract_output_events 的 local_to_global_j 查表(任務7
第二階段第3段)。對應 docs/math/conv事件佇列壓縮版推導.md 第 5.3 節。

分三類測試:
1. 直接照推導文件第 5.3 節的手算範例(延續第 2-4 節神經元 5/6/7 那組例子)
   構造 spike_mask/spike_event_idx/local_to_global_j,驗證查表+排序的結果
   逐位元對得上文件表格。
2. tie-break 穩定性:同一個全域 j 觸發兩顆不同神經元同時 fire,確認排序後
   維持 jnp.nonzero 原始掃描順序(神經元 index 小的排前面),不是隨機順序。
3. 端到端:build_conv_queue_compressed -> run_layer_forward ->
   extract_output_events_compressed 整條串起來,跟透明參考
   (_reference.dense_conv_affine_map -> run_layer_forward -> extract_output_events,
   不給 local_to_global_j)比對 ExtractedEvents **四個欄位全部**(event_times/
   event_source_idx/event_gain/n_real_events)——只比 v_final 或只比前後
   兩個欄位,查不到 event_gain(問題紀錄第四節的 s_spike 跨層梯度機制)被
   搬到錯位置的 bug,這種 bug 不會讓 forward 數值跑掉,只會讓下一層的梯度
   算錯。
"""

import jax
import jax.numpy as jnp

from salt_core.chunk_scan import run_layer_forward
from salt_core.connectivity.conv import build_conv_queue_compressed
from salt_core.tests._reference import dense_conv_affine_map
from salt_core.layer_chain import extract_output_events, extract_output_events_compressed

TOL = 1e-6


def assert_allclose(actual, expected, msg, tol=TOL):
    actual = float(actual)
    expected = float(expected)
    assert abs(actual - expected) < tol, f"{msg}: got {actual}, expected {expected}"


# ============================================================================
# 第一類:推導文件第 5.3 節手算範例
# ============================================================================

def test_extract_output_events_matches_worked_example_section_5_3():
    """延續文件第 2-4 節事件時間 [1,2,3,5]、神經元 5/6/7、v_th=0.6 那組例子。
    文件第 5.2 節算出的 spike 結果:神經元5 在局部欄位2 fire(v_final~=0)、
    神經元7 在局部欄位0 fire(v_final=0)、神經元6 不 fire。查表用第 5.1 節
    的對照表(神經元5->[j=0,1,3]、神經元6->[j=2,3,pad]、神經元7->[j=1,pad,pad])。

    第 5.3 節預期結果:排序後 event_source_idx=[7,5,...]、event_times=[2,5,...]、
    n_real_events=2。這裡只需要合成 spike_mask/spike_event_idx/
    local_to_global_j 三個陣列(不用真的跑 build_conv_queue_compressed,
    這個測試只驗證 extract_output_events 這一步本身),神經元 id 直接用
    0,1,2 對應文件的 5,6,7(哪個數字當 id 不影響查表/排序邏輯)。
    """
    # 神經元 id:0->文件神經元5, 1->文件神經元6, 2->文件神經元7
    # max_steps=1(每顆神經元最多一步就 fire 或不 fire,這裡只關心有沒有
    # fire、fire 在哪個局部欄位,不需要多步)
    spike_mask = jnp.array([[True], [False], [True]])
    spike_event_idx = jnp.array([[2], [0], [0]])  # 神經元0 局部欄位2、神經元2 局部欄位0
    s_spike = jnp.array([[0.9], [0.0], [0.8]])  # 假可微分強度示意值,不是重點
    event_times = jnp.array([1.0, 2.0, 3.0, 5.0])  # 這層自己的輸入事件時間(j=0..3)
    local_to_global_j = jnp.array([
        [0, 1, 3],  # 神經元0(文件神經元5)
        [2, 3, 4],  # 神經元1(文件神經元6),4=sentinel(=n_events,pad)
        [1, 4, 4],  # 神經元2(文件神經元7)
    ])

    result = extract_output_events_compressed(spike_mask, spike_event_idx, s_spike, event_times,
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
# 第二類:tie-break 穩定性——同一個全域 j 觸發多顆神經元同時 fire
# ============================================================================

def test_tiebreak_stable_when_multiple_neurons_share_same_global_j():
    """兩顆不同神經元(id=0,2)的 spike 都查表對應到同一個全域 j=5(物理上是
    同一個來源事件同時觸發兩個下游神經元,合法情況——例如同一個 conv1 輸出
    事件同時落在 conv2 兩個不同輸出神經元的感受野內)。順序不重要但需要
    確定,預期結果:保留 jnp.nonzero 掃描到的原始順序(神經元 id 小的在前),
    不是任意順序。"""
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

    result = extract_output_events_compressed(spike_mask, spike_event_idx, s_spike, event_times,
                                             local_to_global_j, max_total_spikes=4)

    assert int(result.n_real_events) == 2, result.n_real_events
    assert list(map(int, result.event_source_idx[:2])) == [0, 2], \
        (f"j 相同時應保留原始掃描順序(神經元0在前),"
         f"實際 {list(map(int, result.event_source_idx[:2]))}")
    assert_allclose(result.event_times[0], 60.0, "兩筆都對應全域j=5的時間")
    assert_allclose(result.event_times[1], 60.0, "兩筆都對應全域j=5的時間")


# ============================================================================
# 第三類:端到端,壓縮版整條串接 vs 密集版整條串接,四個欄位全比
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


def test_end_to_end_compressed_matches_dense_all_four_fields():
    """build_conv_queue_compressed -> run_layer_forward ->
    extract_output_events_compressed 整條串起來,跟密集版整條
    串起來,ExtractedEvents 四個欄位(event_times/event_source_idx/
    event_gain/n_real_events)全部比對,不是只比前後兩個。event_gain 對不對
    是這個測試存在的主要理由:forward v_final 對,不代表 event_gain 排對了
    位置,但下一層的梯度完全靠這個欄位排對。用跟階段2同一個「會真的 fire」
    的場景(idx0 在局部欄位1、全域j=2 fire),確保 s_spike 不是全 0/全 1 的
    退化情況。"""
    W = _make_weight()
    event_times = jnp.array([1.0, 1.5, 2.0, 2.5, 5.0])
    x = jnp.array([1, 2, 1, 2, 0]); y = jnp.array([1, 2, 1, 2, 0]); c = jnp.array([0, 0, 0, 0, 0])
    v_th = 15.0
    max_queue_len = 5
    max_spikes = 9 * 5  # 恆安全上界,這個測試不是在驗證 max_total_spikes 怎麼抓

    # 密集版整條串接
    maps_dense = dense_conv_affine_map(event_times, x, y, c, W, TAU, S, P, H_OUT, W_OUT)
    result_dense = run_layer_forward(maps_dense, v_th, chunk_size=5, max_steps=5, n_real_events=maps_dense.a.shape[1])
    ev_dense = extract_output_events(result_dense.spike_mask, result_dense.spike_event_idx,
                                      result_dense.s_spike, event_times,
                                      max_total_spikes=max_spikes)

    # 壓縮版整條串接
    cq = build_conv_queue_compressed(event_times, x, y, c, W, TAU, S, P, H_OUT, W_OUT,
                                      max_queue_len, n_real_events=event_times.shape[0])
    result_compressed = run_layer_forward(cq.maps, v_th, chunk_size=5, max_steps=5,
                                           n_real_events=cq.n_real_events)
    ev_compressed = extract_output_events_compressed(result_compressed.spike_mask,
                                           result_compressed.spike_event_idx,
                                           result_compressed.s_spike, event_times,
                                           cq.local_to_global_j,
                                           max_total_spikes=max_spikes)

    assert int(ev_dense.n_real_events) >= 1, "前置確認:這個場景至少要有一筆真的輸出事件"
    _extracted_events_equal(ev_dense, ev_compressed)


def test_end_to_end_compressed_matches_dense_with_multiple_fires_and_pad_input():
    """比上一個測試更完整的端到端案例:輸入端本身就帶 n_real_events(模擬
    上一層 extract_output_events 補的 pad 事件),而且場景會讓不只一顆神經元
    fire,四個欄位在多筆輸出事件上都要對得起來,不是只驗證單一筆。"""
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
    ev_dense = extract_output_events(result_dense.spike_mask, result_dense.spike_event_idx,
                                      result_dense.s_spike, event_times,
                                      max_total_spikes=max_spikes)

    cq = build_conv_queue_compressed(event_times, x, y, c, W, TAU, S, P, H_OUT, W_OUT,
                                      max_queue_len, n_real_events=n_real_events)
    result_compressed = run_layer_forward(cq.maps, v_th, chunk_size=max_queue_len,
                                           max_steps=max_queue_len,
                                           n_real_events=cq.n_real_events)
    ev_compressed = extract_output_events_compressed(result_compressed.spike_mask,
                                           result_compressed.spike_event_idx,
                                           result_compressed.s_spike, event_times,
                                           cq.local_to_global_j,
                                           max_total_spikes=max_spikes)

    assert int(ev_dense.n_real_events) >= 2, \
        f"前置確認:這個場景至少要有兩顆神經元 fire 才夠測排序,實際 {int(ev_dense.n_real_events)}"
    _extracted_events_equal(ev_dense, ev_compressed)
