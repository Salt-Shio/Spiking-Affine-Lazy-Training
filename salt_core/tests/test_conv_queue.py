"""conv 佇列建構(connectivity/conv.py),推導見 docs/math/conv事件佇列壓縮版推導.md。

1. _compress_candidates、_delta_t_three_regimes、conv_float_values 對推導文件的手算例子。
2. build_conv_structure + conv_float_values 跟參考實作(_reference.dense_conv_affine_map)比:
   run_layer 之後的 v_final、spike 細節、梯度。不 fire 時 v_final 對 W、gain 是線性的,
   梯度用中央差分對參考算;會 fire 的場景用參考的 autodiff 對照。
3. 佇列剛好裝滿真事件時沒有 catch-up 欄:ConvLayer 的需求要算進 catch-up,判成出界。
幾何變化跟跨層梯度在 test_conv_geometry.py。
"""

import jax
import jax.numpy as jnp
import numpy as np

from salt_core.float.scan import run_layer
from salt_core.connectivity.conv import (ConvQueueStructure, _compress_candidates,
                                          _delta_t_three_regimes, build_conv_structure,
                                          conv_float_values, conv_weight_codes, tile_channels)
from salt_core.layers import ConvLayer
from salt_core.network import InputEvents, Network
from salt_core.tests._reference import dense_conv_affine_map, finite_diff_grad

TOL = 1e-6

TAU = 4.0
K, S, P = 3, 2, 1
H_OUT = W_OUT = 3


def assert_allclose(actual, expected, msg, tol=TOL):
    actual = float(actual)
    expected = float(expected)
    assert abs(actual - expected) < tol, f"{msg}: got {actual}, expected {expected}"


def _make_weight(oc_offset=0.0):
    """W[0,0,ky,kx] = ky*3+kx+1(+oc_offset),形狀 (1,1,3,3)。"""
    base = jnp.array([[ky * 3 + kx + 1 for kx in range(3)] for ky in range(3)],
                      dtype=jnp.float32)
    return (base + oc_offset)[None, None, :, :]


# ============================================================================
# 1. 對推導文件的手算例子
# ============================================================================

def test_compress_candidates_matches_worked_example():
    """推導文件「演算法:排序 + 分段重置計數」的數值例子:4 筆事件、單軸 N=2,
    排序加分段計數之後,神經元 5、6、7 收到文件表格列的那幾欄。
    n_out_spatial 取 8,不合法候選標成 8(文件用 99 示意,>= n_out_spatial 都一樣)。
    """
    n_flat = jnp.array([5, 8, 5, 7, 6, 8, 5, 6], dtype=jnp.int32)
    j_flat = jnp.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=jnp.int32)
    n_out_spatial = 8
    max_queue_len = 3
    n_events = 4  # j 值域 [0,4)

    local_to_global_j, n_real = _compress_candidates(n_flat, j_flat, n_out_spatial, max_queue_len,
                                                     n_events)

    assert local_to_global_j.shape == (n_out_spatial, max_queue_len)
    assert n_real.shape == (n_out_spatial,)

    # 文件表格:神經元 5 -> [j=0,j=1,j=3],神經元 6 -> [j=2,j=3,pad],
    # 神經元 7 -> [j=1,pad,pad]。pad 格填 sentinel=n_events=4。
    assert list(map(int, local_to_global_j[5])) == [0, 1, 3]
    assert list(map(int, local_to_global_j[6])) == [2, 3, 4]
    assert list(map(int, local_to_global_j[7])) == [1, 4, 4]

    assert int(n_real[5]) == 3
    assert int(n_real[6]) == 2
    assert int(n_real[7]) == 1

    # 其餘神經元(0,1,2,3,4)完全沒出現在候選清單裡,應該是全 pad、n_real=0
    for n in (0, 1, 2, 3, 4):
        assert int(n_real[n]) == 0
        assert list(map(int, local_to_global_j[n])) == [4, 4, 4]


def _maps_from_worked_example(t_gathered, n_real, global_last_time, b_real):
    """手算例子直接給每格的 b,不經過座標幾何:權重 w[0,0,row,col] = b_real[row,col],
    每格的 kernel 位置指回自己那一格。回傳 (maps, delta_t)。"""
    rows, cols = b_real.shape
    delta_t = _delta_t_three_regimes(t_gathered, n_real, global_last_time)
    tap_ky, tap_kx = jnp.meshgrid(jnp.arange(rows), jnp.arange(cols), indexing="ij")
    structure = ConvQueueStructure(
        local_to_global_j=jnp.zeros((rows, cols), dtype=jnp.int32), n_real_events=n_real,
        delta_t=delta_t, tap_c=jnp.zeros((rows, cols), dtype=jnp.int32),
        tap_ky=tap_ky, tap_kx=tap_kx, n_input_events=jnp.asarray(1, dtype=jnp.int32))
    return conv_float_values(structure, b_real[None, None], TAU, None), delta_t


def test_delta_t_and_float_values_match_worked_example():
    """推導文件「局部時間差與仿射映射」「修正後驗證」的數值例子:事件時間 [1,2,3,5],tau=4
    (衰減底數 0.75)。神經元 7 的最後一筆事件 t=2 早於全域最後一筆 t=5,就是「補位不能直接補
    identity」的反例;神經元 6 的最後一筆剛好是全域最後一筆,catch-up 退化成 identity。
    """
    # 3 列對應神經元 5、6、7;這個函式只看每列自己的時間跟 n_real。pad 欄的時間填什麼都不影響結果
    t_gathered = jnp.array([
        [1.0, 2.0, 5.0],   # 神經元 5:j=0,1,3 -> t=1,2,5
        [3.0, 5.0, 5.0],   # 神經元 6:j=2,3 -> t=3,5;第三欄是 pad,填什麼都無所謂
        [2.0, 2.0, 2.0],   # 神經元 7:j=1 -> t=2;後兩欄是 pad
    ])
    n_real = jnp.array([3, 2, 1], dtype=jnp.int32)
    global_last_time = jnp.asarray(5.0)  # 全域最後一筆事件(j=3)的時間
    b_real = jnp.array([
        [0.2, 0.4, 0.5],
        [-0.3, 0.6, 0.0],
        [0.7, 0.0, 0.0],
    ])

    maps, delta_t = _maps_from_worked_example(t_gathered, n_real, global_last_time, b_real)

    # 神經元 5(索引 0):全部都是真 tap,沒有補位
    assert_allclose(maps.a[0, 0], 0.75, "神經元5 col0 a")
    assert_allclose(maps.a[0, 1], 0.75, "神經元5 col1 a")
    assert_allclose(maps.a[0, 2], 0.421875, "神經元5 col2 a")
    assert_allclose(maps.b[0, 0], 0.2, "神經元5 col0 b")
    assert_allclose(maps.b[0, 1], 0.4, "神經元5 col1 b")
    assert_allclose(maps.b[0, 2], 0.5, "神經元5 col2 b")
    assert list(delta_t[0]) == [1, 1, 3], "神經元5 三個真 tap 的 Δt(1,2,5 的相鄰差)"

    # 神經元 6(索引 1):col2 是 catch-up,最後一筆就是全域最後一筆,退化成 a=1、b=0
    assert_allclose(maps.a[1, 0], 0.421875, "神經元6 col0 a")
    assert_allclose(maps.a[1, 1], 0.5625, "神經元6 col1 a")
    assert_allclose(maps.a[1, 2], 1.0, "神經元6 catch-up 退化 a")
    assert_allclose(maps.b[1, 0], -0.3, "神經元6 col0 b")
    assert_allclose(maps.b[1, 1], 0.6, "神經元6 col1 b")
    assert_allclose(maps.b[1, 2], 0.0, "神經元6 catch-up b")
    assert list(delta_t[1]) == [3, 2, 0], "神經元6 catch-up Δt=5-5=0,跟 a=1 一致"

    # 神經元 7(索引 2):col1 是真正的 catch-up(Δt=5-2=3),col2 是純 identity
    assert_allclose(maps.a[2, 0], 0.5625, "神經元7 col0 a")
    assert_allclose(maps.a[2, 1], 0.421875, "神經元7 catch-up a(補位反例的關鍵格)")
    assert_allclose(maps.a[2, 2], 1.0, "神經元7 identity a")
    assert_allclose(maps.b[2, 0], 0.7, "神經元7 col0 b")
    assert_allclose(maps.b[2, 1], 0.0, "神經元7 catch-up b")
    assert_allclose(maps.b[2, 2], 0.0, "神經元7 identity b")
    assert list(delta_t[2]) == [2, 3, 0], "神經元7 catch-up Δt=5-2=3,identity Δt=0"


def test_catchup_identity_pitfall_matches_hand_derivation():
    """推導文件「原本的想法是錯的」的反例:補位直接補 identity 時神經元 7 會算成 0.7,
    正確答案是 0.2953125。"""
    t_gathered = jnp.array([[2.0, 2.0, 2.0]])  # 神經元 7,只有 j=1(t=2)
    n_real = jnp.array([1], dtype=jnp.int32)
    global_last_time = jnp.asarray(5.0)
    b_real = jnp.array([[0.7, 0.0, 0.0]])

    maps, _delta_t = _maps_from_worked_example(t_gathered, n_real, global_last_time, b_real)

    v0 = 0.0
    v1 = maps.a[0, 0] * v0 + maps.b[0, 0]
    v2 = maps.a[0, 1] * v1 + maps.b[0, 1]
    v3 = maps.a[0, 2] * v2 + maps.b[0, 2]

    assert_allclose(v1, 0.7, "x0")
    assert_allclose(v2, 0.2953125, "x1(正確答案,不是錯誤的 0.7)")
    assert_allclose(v3, 0.2953125, "x2")


# ============================================================================
# 2. build_conv_structure + conv_float_values 跟參考實作比
#    梯度:不 fire 時用中央差分對參考;會 fire 時用參考的 autodiff。

def _run_ref(event_times, x, y, c, W, v_th, max_steps, n_real_events=None, event_gain=None):
    """參考實作建密集 (a, b) -> run_layer。"""
    n_real_events = event_times.shape[0] if n_real_events is None else n_real_events
    maps = dense_conv_affine_map(event_times, x, y, c, W, TAU, S, P, H_OUT, W_OUT,
                                  gain=event_gain, n_real=n_real_events)
    return run_layer(maps, v_th, chunk_size=max_steps, max_steps=max_steps,
                     n_real_events=n_real_events)


def _run_conv(event_times, x, y, c, W, v_th, max_queue_len, max_steps,
              n_real_events=None, event_gain=None):
    """回傳 (result, structure);W 的 oc 都是 1 時,structure 的列就是神經元。"""
    n_real_events = event_times.shape[0] if n_real_events is None else n_real_events
    structure = build_conv_structure(event_times, x, y, c, K, S, P, H_OUT, W_OUT,
                                     max_queue_len, n_real_events)
    maps = conv_float_values(structure, W, TAU, event_gain)
    result = run_layer(maps, v_th, chunk_size=max_steps, max_steps=max_steps,
                       n_real_events=tile_channels(structure.n_real_events, W.shape[0]))
    return result, structure


def _ref_vfinal_all_affine(event_times, x, y, c, W, n_real_events=None, event_gain=None):
    """不 fire(v_th=1e9)時所有神經元的 v_final:numpy float64 直接折疊參考的 (a, b),
    不經過 run_layer,給中央差分用。"""
    maps = dense_conv_affine_map(event_times, x, y, c, W, TAU, S, P, H_OUT, W_OUT,
                                  gain=event_gain, n_real=n_real_events)
    a = np.asarray(maps.a, dtype=np.float64)
    b = np.asarray(maps.b, dtype=np.float64)
    v = np.zeros(a.shape[0])
    for j in range(a.shape[1]):
        v = a[:, j] * v + b[:, j]
    return v


def _grad_conv(event_times, x, y, c, v_th, max_queue_len, max_steps, neuron_idx=0,
               n_real_events=None, event_gain=None):
    def loss_fn(W):
        result, _ = _run_conv(event_times, x, y, c, W, v_th, max_queue_len, max_steps,
                              n_real_events, event_gain)
        return result.v_final[neuron_idx]
    return jax.grad(loss_fn)


def _fd_grad_ref_wrt_W(event_times, x, y, c, W, reduce, n_real_events=None, event_gain=None):
    """參考 forward(不 fire)對 W 的中央差分梯度。reduce(v_vec) -> 純量 決定 loss。

    不 fire 時 v_final 對每個 W 元素是線性的(tap 合不合法只看座標),中央差分沒有截斷誤差;
    eps 開 0.1 是為了壓低 float32 捨入被 1/eps 放大。"""
    return finite_diff_grad(
        lambda w: float(reduce(_ref_vfinal_all_affine(
            event_times, x, y, c, w, n_real_events, event_gain))),
        np.asarray(W), eps=0.1)


def _grad_ref_autodiff_firing(event_times, x, y, c, v_th, max_steps, neuron_idx=0):
    """會 fire 的場景:電壓在 fire 邊界不連續,中央差分不可靠,改用參考實作的 autodiff。"""
    def loss_fn(W):
        maps = dense_conv_affine_map(event_times, x, y, c, W, TAU, S, P, H_OUT, W_OUT)
        return run_layer(maps, v_th, chunk_size=max_steps,
                         max_steps=max_steps, n_real_events=maps.a.shape[1]).v_final[neuron_idx]
    return jax.grad(loss_fn)


_GRAD_TOL = 2e-4  # 有限差分(eps=1e-4、參考 forward numpy float64)的精度量級


def test_matches_dense_single_event():
    """單一事件、4 個候選都合法。"""
    W = _make_weight()
    event_times = jnp.array([1.0])
    x = jnp.array([1]); y = jnp.array([1]); c = jnp.array([0])

    ref = _run_ref(event_times, x, y, c, W, v_th=1e9, max_steps=1)
    result, structure = _run_conv(event_times, x, y, c, W, v_th=1e9,
                                  max_queue_len=1, max_steps=1)

    assert bool(jnp.allclose(ref.v_final, result.v_final, atol=TOL)), \
        (ref.v_final, result.v_final)

    # 手算過的期望值(idx0=9.0, idx1=7.0, idx3=3.0, idx4=1.0, 其餘 0)
    expected = jnp.zeros(9).at[jnp.array([0, 1, 3, 4])].set(jnp.array([9.0, 7.0, 3.0, 1.0]))
    assert bool(jnp.allclose(result.v_final, expected, atol=TOL)), result.v_final


def test_matches_dense_two_events_with_degenerate_catchup():
    """兩筆事件,邊界那筆只有 1/4 候選合法。max_queue_len 取 3(比需要的 2 多),逼出 catch-up 欄;
    每個神經元最後一筆都剛好是全域最後一筆,catch-up 退化成 dt=0,確認多出來的欄不影響結果。"""
    W = _make_weight()
    event_times = jnp.array([1.0, 2.0])
    x = jnp.array([1, 0]); y = jnp.array([1, 0]); c = jnp.array([0, 0])

    ref = _run_ref(event_times, x, y, c, W, v_th=1e9, max_steps=2)
    result, structure = _run_conv(event_times, x, y, c, W, v_th=1e9,
                                  max_queue_len=3, max_steps=3)

    assert bool(jnp.allclose(ref.v_final, result.v_final, atol=TOL)), \
        (ref.v_final, result.v_final)

    expected = jnp.zeros(9).at[jnp.array([0, 1, 3, 4])].set(
        jnp.array([11.75, 5.25, 2.25, 0.75]))
    assert bool(jnp.allclose(result.v_final, expected, atol=TOL)), result.v_final


def test_matches_dense_with_genuine_non_degenerate_catchup():
    """三筆事件,idx1 的最後一筆 t=1 早於全域最後一筆 t=5,catch-up 真的要衰減。

    事件 0、1 同上一個測試;事件 2 在 (2,2,t=5),只碰得到 idx4。"""
    W = _make_weight()
    event_times = jnp.array([1.0, 2.0, 5.0])
    x = jnp.array([1, 0, 2]); y = jnp.array([1, 0, 2]); c = jnp.array([0, 0, 0])

    ref = _run_ref(event_times, x, y, c, W, v_th=1e9, max_steps=3)
    result, structure = _run_conv(event_times, x, y, c, W, v_th=1e9,
                                  max_queue_len=3, max_steps=3)

    assert bool(jnp.allclose(ref.v_final, result.v_final, atol=TOL)), \
        (ref.v_final, result.v_final)

    # idx1 真的用到非退化 catch-up:col0 real(a=0.75,b=7)->x0=7;
    # catch-up Δt=5-1=4 -> a=0.75^4=0.31640625 -> x1=7*0.31640625=2.21484375;
    # 沒有第三欄真的可用(n_real=1,max_queue_len=3,col2 是 identity)-> x2 不變。
    assert_allclose(result.v_final[1], 2.21484375, "idx1 用到非退化 catch-up")


def test_matches_dense_with_pad_events():
    """第三筆是 pad 事件,座標 (0,0,0) 看起來合法。n_real_events=2,pad 不能被當成任何神經元的真 tap。"""
    W = _make_weight()
    event_times = jnp.array([1.0, 2.0, 1e12])  # 第三筆是 pad,時間刻意設超大
    x = jnp.array([1, 0, 0]); y = jnp.array([1, 0, 0]); c = jnp.array([0, 0, 0])
    n_real_events = 2

    ref = _run_ref(event_times, x, y, c, W, v_th=1e9, max_steps=3,
                        n_real_events=n_real_events)
    result, structure = _run_conv(event_times, x, y, c, W, v_th=1e9,
                                  max_queue_len=3, max_steps=3,
                                  n_real_events=n_real_events)

    assert bool(jnp.allclose(ref.v_final, result.v_final, atol=TOL)), \
        (ref.v_final, result.v_final)

    # idx0 真正該收到的合法 tap 是 2 筆(事件 0 在 (1,1) 貢獻 k=(2,2)、
    # 事件 1 在 (0,0) 貢獻 k=(1,1),兩者座標不同、都是真事件);pad 事件
    # (事件 2)座標也是 (0,0)、跟事件 1 一樣會落在 idx0 的候選裡,如果沒有
    # 被 n_real_events 正確排除,會被誤算成第 3 筆,變成 n_real=3。
    assert int(structure.n_real_events[0]) == 2, \
        f"idx0 應該只收到 2 筆真實 tap,pad 事件不該被算進去,拿到 {int(structure.n_real_events[0])}"
    assert list(map(int, structure.local_to_global_j[0][:2])) == [0, 1], \
        "idx0 的兩個真實 tap 應該是事件 0、事件 1,pad 事件(j=2)不該出現"


def test_neuron_with_zero_real_events_matches_dense_zero():
    """某個神經元完全沒有事件:單一事件只碰得到 4 個角落神經元,idx2 的 n_real 是 0、v_final 是 0,
    不能 crash 或 NaN。"""
    W = _make_weight()
    event_times = jnp.array([1.0])
    x = jnp.array([1]); y = jnp.array([1]); c = jnp.array([0])

    ref = _run_ref(event_times, x, y, c, W, v_th=1e9, max_steps=1)
    result, structure = _run_conv(event_times, x, y, c, W, v_th=1e9,
                                  max_queue_len=1, max_steps=1)

    assert int(structure.n_real_events[2]) == 0
    assert_allclose(result.v_final[2], 0.0, "沒有任何相關事件的神經元,v_final 應該是 0")
    assert_allclose(ref.v_final[2], 0.0, "參考實作同一顆神經元也是 0")
    assert not bool(jnp.any(jnp.isnan(result.v_final))), "不該出現 NaN"


def test_exactly_fills_max_queue_len_no_padding_needed():
    """真 tap 數剛好等於 max_queue_len,沒有 catch-up 欄:兩筆事件都落在 idx0,max_queue_len=2。"""
    W = _make_weight()
    event_times = jnp.array([1.0, 3.0])
    x = jnp.array([1, 1]); y = jnp.array([1, 1]); c = jnp.array([0, 0])  # 兩筆都在同一個像素

    ref = _run_ref(event_times, x, y, c, W, v_th=1e9, max_steps=2)
    result, structure = _run_conv(event_times, x, y, c, W, v_th=1e9,
                                  max_queue_len=2, max_steps=2)

    assert int(structure.n_real_events[0]) == 2, "idx0 應該收到剛好 2 筆真實事件,等於 max_queue_len"
    assert bool(jnp.allclose(ref.v_final, result.v_final, atol=TOL)), \
        (ref.v_final, result.v_final)


def test_oc_independent_candidacy_only_weight_differs():
    """候選篩選跟 oc 無關(推導文件「為什麼感受野跟 oc 無關」):OC=2 的兩個 channel 權重差很多,
    a、local_to_global_j、n_real_events 兩個 channel 相同,只有 b 不同。"""
    w0 = _make_weight(oc_offset=0.0)   # (1,1,3,3)
    w1 = _make_weight(oc_offset=100.0)  # 數值差很大,確保不是巧合相等
    W = jnp.concatenate([w0, w1], axis=0)  # (2,1,3,3)

    event_times = jnp.array([1.0, 2.0])
    x = jnp.array([1, 0]); y = jnp.array([1, 0]); c = jnp.array([0, 0])
    max_queue_len = 3
    n_out_spatial = H_OUT * W_OUT

    structure = build_conv_structure(event_times, x, y, c, K, S, P, H_OUT, W_OUT,
                                     max_queue_len, event_times.shape[0])
    maps = conv_float_values(structure, W, TAU, None)

    assert structure.local_to_global_j.shape == (n_out_spatial, max_queue_len), \
        "結構段沒有 oc 軸,local_to_global_j 不受 oc 影響"
    assert structure.n_real_events.shape == (n_out_spatial,), \
        "結構段沒有 oc 軸,n_real_events 不受 oc 影響"
    for spatial_idx in range(n_out_spatial):
        oc0_row = spatial_idx               # oc=0 的第 spatial_idx 個神經元
        oc1_row = n_out_spatial + spatial_idx  # oc=1 的同一個空間位置
        assert bool(jnp.allclose(maps.a[oc0_row], maps.a[oc1_row], atol=TOL)), \
            f"空間位置 {spatial_idx}:a(衰減)只跟時間差有關,不該受 oc 影響"

    # b 應該不同(權重確實不同,不是巧合)——至少有真 tap 的位置要能驗證這件事
    has_real_tap = structure.n_real_events > 0
    assert bool(jnp.any(has_real_tap)), "這個場景至少要有一個神經元收到真 tap 才能驗證 b 不同"
    real_spatial_idx = int(jnp.argmax(has_real_tap))
    b_oc0 = maps.b[real_spatial_idx, 0]
    b_oc1 = maps.b[n_out_spatial + real_spatial_idx, 0]
    assert abs(float(b_oc0) - float(b_oc1)) > 1.0, "不同 oc 的權重差很大,b 應該明顯不同"


def test_matches_dense_on_realistic_random_case():
    """隨機多事件,max_queue_len 等於事件數(不截斷):逐神經元 v_final 跟整個 W 的梯度都跟參考一致。
    跑 3 個 seed:排序跟 scatter 只在特定排列下出錯(很多事件打到同一顆、合法跟不合法候選混雜)。"""
    for seed in (0, 1, 2):
        key = jax.random.PRNGKey(seed)
        k_t, k_xy, k_w = jax.random.split(key, 3)

        n_events = 30
        event_times = jnp.sort(jax.random.randint(k_t, (n_events,), 0, 200).astype(jnp.float32))
        kx, ky_ = jax.random.split(k_xy)
        x = jax.random.randint(kx, (n_events,), 0, 5).astype(jnp.int32)
        y = jax.random.randint(ky_, (n_events,), 0, 5).astype(jnp.int32)
        c = jnp.zeros(n_events, dtype=jnp.int32)

        IC, OC = 1, 2
        W = jax.random.uniform(k_w, (OC, IC, K, K), minval=-1.0, maxval=1.0)

        ref = _run_ref(event_times, x, y, c, W, v_th=1e9, max_steps=n_events)
        result, structure = _run_conv(event_times, x, y, c, W, v_th=1e9,
                                      max_queue_len=n_events, max_steps=n_events)

        assert bool(jnp.allclose(ref.v_final, result.v_final, atol=1e-4)), \
            (seed, ref.v_final, result.v_final)

        grad_ref = _fd_grad_ref_wrt_W(event_times, x, y, c, W, reduce=lambda v: v[0])
        grad_conv = _grad_conv(event_times, x, y, c, v_th=1e9,
                               max_queue_len=n_events, max_steps=n_events)(W)
        assert bool(jnp.allclose(grad_ref, grad_conv, atol=_GRAD_TOL)), \
            (seed, grad_ref, grad_conv)


def test_gradient_matches_dense_pure_affine():
    """同 test_matches_dense_two_events_with_degenerate_catchup 的場景(不 fire),loss=v_final[0],
    比整個 W 的梯度陣列,不只挑幾個位置。"""
    event_times = jnp.array([1.0, 2.0])
    x = jnp.array([1, 0]); y = jnp.array([1, 0]); c = jnp.array([0, 0])

    grad_ref = _fd_grad_ref_wrt_W(event_times, x, y, c, _make_weight(), reduce=lambda v: v[0])
    grad_conv = _grad_conv(event_times, x, y, c, v_th=1e9,
                           max_queue_len=3, max_steps=3)(_make_weight())

    assert bool(jnp.allclose(grad_ref, grad_conv, atol=_GRAD_TOL)), \
        (grad_ref, grad_conv)


def test_dropped_candidate_gradient_matches_dense():
    """單一事件 (0,0,c=0,t=1):k=-1 的 3 個候選不合法,被丟進垃圾桶。唯一真 tap 的梯度是 1.0,
    只被不合法候選碰到的 W 位置梯度要精確是 0。"""
    event_times = jnp.array([1.0])
    x = jnp.array([0]); y = jnp.array([0]); c = jnp.array([0])

    grad_ref = _fd_grad_ref_wrt_W(event_times, x, y, c, _make_weight(), reduce=lambda v: v[0])
    grad_conv = _grad_conv(event_times, x, y, c, v_th=1e9,
                           max_queue_len=1, max_steps=1)(_make_weight())

    # 唯一真 tap W[0,0,1,1] 梯度 1.0;k=-1 會 wraparound 到的 3 個位置要精確是 0
    assert_allclose(grad_conv[0, 0, 1, 1], 1.0, "唯一真實 tap 的梯度")
    for ky, kx in [(1, 2), (2, 1), (2, 2)]:
        assert_allclose(grad_conv[0, 0, ky, kx], 0.0,
                         f"W[0,0,{ky},{kx}] 只被丟棄候選碰過,壓縮版梯度應該精確是 0")
    assert bool(jnp.allclose(grad_ref, grad_conv, atol=_GRAD_TOL)), \
        (grad_ref, grad_conv)


def test_gradient_matches_dense_with_genuine_catchup():
    """同 test_matches_dense_with_genuine_non_degenerate_catchup 的場景:catch-up 那格的 a 只看時間差,
    不能多出或擋掉 W 的梯度。"""
    event_times = jnp.array([1.0, 2.0, 5.0])
    x = jnp.array([1, 0, 2]); y = jnp.array([1, 0, 2]); c = jnp.array([0, 0, 0])

    grad_ref = _fd_grad_ref_wrt_W(event_times, x, y, c, _make_weight(), reduce=lambda v: v[1])
    grad_conv = _grad_conv(event_times, x, y, c, v_th=1e9, max_queue_len=3,
                           max_steps=3, neuron_idx=1)(_make_weight())

    assert bool(jnp.allclose(grad_ref, grad_conv, atol=_GRAD_TOL)), \
        (grad_ref, grad_conv)


def test_gradient_matches_dense_with_pad_events():
    """同 test_matches_dense_with_pad_events 的場景:pad 事件座標跟事件 1 相同,沒排除的話
    梯度會多一條路徑。"""
    event_times = jnp.array([1.0, 2.0, 1e12])
    x = jnp.array([1, 0, 0]); y = jnp.array([1, 0, 0]); c = jnp.array([0, 0, 0])
    n_real_events = 2

    grad_ref = _fd_grad_ref_wrt_W(event_times, x, y, c, _make_weight(), reduce=lambda v: v[0],
                                   n_real_events=n_real_events)
    grad_conv = _grad_conv(event_times, x, y, c, v_th=1e9, max_queue_len=3,
                           max_steps=3, n_real_events=n_real_events)(_make_weight())

    assert bool(jnp.allclose(grad_ref, grad_conv, atol=_GRAD_TOL)), \
        (grad_ref, grad_conv)


# ============================================================================
# 3. 會 fire 時的 spike 細節跟梯度

def test_matches_dense_spike_details_and_gradient_when_neuron_fires():
    """5 筆事件,idx0 在佇列中段 fire、reset,之後還有真 tap、catch-up、identity 欄。

    j=1、j=3 落在 (2,2),不影響 idx0,所以 idx0 的局部欄跟全域 j 不相等
    (local_to_global_j[0] = [0,2,4,...]),才測得到局部欄轉全域 j。
    比對:整個 spike_mask、整個 s_value(catch-up、identity 欄要排除)、fire 位置的 spike_event_idx
    轉成全域 j 之後跟參考相同、v_final[0] 對整個 W 的梯度。
    """
    W = _make_weight()
    event_times = jnp.array([1.0, 1.5, 2.0, 2.5, 5.0])
    x = jnp.array([1, 2, 1, 2, 0]); y = jnp.array([1, 2, 1, 2, 0]); c = jnp.array([0, 0, 0, 0, 0])
    v_th = 15.0
    max_queue_len = 5  # 留寬到等於全域事件數,這個測試的重點不是 max_queue_len 太小截斷

    ref = _run_ref(event_times, x, y, c, W, v_th, max_steps=5)
    result, structure = _run_conv(event_times, x, y, c, W, v_th, max_queue_len, max_steps=5)

    # 前置確認:idx0 真的如預期在局部欄位1(不是欄位0)fire,且局部欄位1
    # 對應的全域 j 是 2,不是 1——這是這個測試場景成立的前提。
    assert bool(result.spike_mask[0, 0]), "idx0 應該要 fire"
    assert int(result.spike_event_idx[0, 0]) == 1, "idx0 應該在局部欄位1(第二個真實 tap)fire"
    assert int(structure.local_to_global_j[0, 1]) == 2, "局部欄位1 應該對應全域 j=2,不是巧合等於 1"
    assert int(ref.spike_event_idx[0, 0]) == 2, "參考實作 idx0 應該在全域 j=2 fire"

    # 整個陣列都比,沒 fire 的欄(catch-up、identity)也要一致
    assert bool(jnp.array_equal(ref.spike_mask, result.spike_mask)), \
        (ref.spike_mask, result.spike_mask)
    assert bool(jnp.allclose(ref.s_value, result.s_value, atol=TOL)), \
        (ref.s_value, result.s_value)

    # spike_event_idx 只在 fire 的位置有意義;局部欄轉成全域 j 再跟參考比
    global_j_from_conv = jnp.take_along_axis(
        structure.local_to_global_j, result.spike_event_idx, axis=1)
    assert bool(jnp.all(jnp.where(
        result.spike_mask, global_j_from_conv == ref.spike_event_idx, True))), \
        (global_j_from_conv, ref.spike_event_idx, result.spike_mask)

    grad_ref = _grad_ref_autodiff_firing(event_times, x, y, c, v_th, max_steps=5,
                                          neuron_idx=0)(W)
    grad_conv = _grad_conv(event_times, x, y, c, v_th, max_queue_len, max_steps=5,
                           neuron_idx=0)(W)
    assert bool(jnp.allclose(grad_ref, grad_conv, atol=1e-4)), \
        (grad_ref, grad_conv)


# ============================================================================
# 4. event_gain(接在上一層後面時是上一層的 s_spike)

def test_matches_dense_with_event_gain():
    """隨機多事件,event_gain 不是全 1:v_final 逐神經元、整個 W 的梯度都跟參考一致(多個 seed)。"""
    for seed in (0, 1, 2):
        key = jax.random.PRNGKey(seed)
        k_t, k_xy, k_w, k_g = jax.random.split(key, 4)

        n_events = 30
        event_times = jnp.sort(jax.random.randint(k_t, (n_events,), 0, 200).astype(jnp.float32))
        kx, ky_ = jax.random.split(k_xy)
        x = jax.random.randint(kx, (n_events,), 0, 5).astype(jnp.int32)
        y = jax.random.randint(ky_, (n_events,), 0, 5).astype(jnp.int32)
        c = jnp.zeros(n_events, dtype=jnp.int32)

        IC, OC = 1, 2
        W = jax.random.uniform(k_w, (OC, IC, K, K), minval=-1.0, maxval=1.0)
        # 增益值域涵蓋 <1 跟 >1,不是退化的全 1
        event_gain = jax.random.uniform(k_g, (n_events,), minval=0.3, maxval=1.3)

        ref = _run_ref(event_times, x, y, c, W, v_th=1e9, max_steps=n_events,
                            event_gain=event_gain)
        result, _ = _run_conv(event_times, x, y, c, W, v_th=1e9,
                              max_queue_len=n_events, max_steps=n_events,
                              event_gain=event_gain)
        assert bool(jnp.allclose(ref.v_final, result.v_final, atol=1e-4)), \
            (seed, ref.v_final, result.v_final)

        grad_ref = _fd_grad_ref_wrt_W(event_times, x, y, c, W, reduce=lambda v: v[0],
                                       event_gain=event_gain)
        grad_conv = _grad_conv(event_times, x, y, c, v_th=1e9,
                               max_queue_len=n_events, max_steps=n_events,
                               event_gain=event_gain)(W)
        assert bool(jnp.allclose(grad_ref, grad_conv, atol=_GRAD_TOL)), \
            (seed, grad_ref, grad_conv)


def test_event_gain_gradient_matches_dense():
    """W 固定,對整個 event_gain 求導,要跟參考一致:跨層時梯度靠這條路傳回上一層權重。"""
    key = jax.random.PRNGKey(7)
    k_t, k_xy, k_w, k_g = jax.random.split(key, 4)
    n_events = 24
    event_times = jnp.sort(jax.random.randint(k_t, (n_events,), 0, 150).astype(jnp.float32))
    kx, ky_ = jax.random.split(k_xy)
    x = jax.random.randint(kx, (n_events,), 0, 5).astype(jnp.int32)
    y = jax.random.randint(ky_, (n_events,), 0, 5).astype(jnp.int32)
    c = jnp.zeros(n_events, dtype=jnp.int32)
    W = jax.random.uniform(k_w, (2, 1, K, K), minval=-1.0, maxval=1.0)
    event_gain0 = jax.random.uniform(k_g, (n_events,), minval=0.3, maxval=1.3)

    def conv_loss(g):
        result, _ = _run_conv(event_times, x, y, c, W, v_th=1e9,
                              max_queue_len=n_events, max_steps=n_events, event_gain=g)
        return jnp.sum(result.v_final)

    grad_ref = finite_diff_grad(
        lambda g: float(_ref_vfinal_all_affine(event_times, x, y, c, W, event_gain=g).sum()),
        np.asarray(event_gain0), eps=0.1)  # v_final 對 gain 也是線性,中央差分精確
    grad_conv = jax.grad(conv_loss)(event_gain0)
    assert bool(jnp.allclose(grad_ref, grad_conv, atol=_GRAD_TOL)), \
        (grad_ref, grad_conv)


def test_conv_weight_codes_are_int32_and_match_float_values_b():
    """整數數值段直接取權重碼,值要跟浮點數值段的 b 一樣(非真 tap 都是 0),dtype 是 int32。
    隨機事件、OC=2,max_queue_len 取 3 讓部分神經元有 catch-up/identity 欄、部分放不下。"""
    k_t, k_xy, k_q = jax.random.split(jax.random.PRNGKey(3), 3)
    n_events = 20
    event_times = jnp.sort(jax.random.randint(k_t, (n_events,), 0, 100).astype(jnp.float32))
    kx, ky_ = jax.random.split(k_xy)
    x = jax.random.randint(kx, (n_events,), 0, 5)
    y = jax.random.randint(ky_, (n_events,), 0, 5)
    c = jnp.zeros(n_events, dtype=jnp.int32)
    q = jax.random.randint(k_q, (2, 1, K, K), -7, 8).astype(jnp.int32)

    structure = build_conv_structure(event_times, x, y, c, K, S, P, H_OUT, W_OUT, 3, n_events)
    codes = conv_weight_codes(structure, q)

    assert codes.dtype == jnp.int32
    assert codes.shape == (2 * H_OUT * W_OUT, 3)
    assert jnp.array_equal(codes, conv_float_values(structure, q, TAU, None).b)


# ============================================================================
# 3. 佇列剛好裝滿真事件:需求算進 catch-up 欄
# ============================================================================

def _full_queue_case(max_queue_len):
    """輸入 1x4x4、k=3 s=2 p=1 -> 2x2,tau=4,權重全 0.1,不 fire。事件:t=1 像素 0、t=2 像素 3、
    t=3 像素 0、t=5 像素 15。位置 0 收 2 筆(最後一筆 t=3),全域最後一筆 T=5。"""
    layer = ConvLayer(name="conv", ic=1, h_in=4, w_in=4, oc=1, k=3, s=2, p=1, init_k=1.0,
                      tau=4.0, v_th=1e9, max_queue_len=max_queue_len, max_out_spikes=16)
    raw = InputEvents(jnp.array([1.0, 2.0, 3.0, 5.0]), jnp.array([0, 3, 0, 15]), jnp.array(4))
    return Network((1, 4, 4), [layer]).apply((jnp.full((1, 1, 3, 3), 0.1),), raw)


def test_queue_full_of_real_events_is_reported_as_overflow():
    """位置 0 的 2 筆真事件佔滿 2 欄,catch-up 沒位置放,需求是 2 + 1 = 3。"""
    out = _full_queue_case(max_queue_len=2)
    assert int(out.diags[0].needed["max_queue_len"]) == 3
    assert not bool(out.fits)


def test_queue_with_room_for_catchup_decays_to_global_last_time():
    """位置 0:t=1 時 V=0.1,t=3 時 V=0.1*0.75**2 + 0.1 = 0.15625,catch-up 衰減到 T=5:
    0.15625 * 0.75**2 = 0.087890625。"""
    out = _full_queue_case(max_queue_len=3)
    assert bool(out.fits)
    assert_allclose(out.last.v_final[0], 0.087890625, "位置 0 的 v_final")
