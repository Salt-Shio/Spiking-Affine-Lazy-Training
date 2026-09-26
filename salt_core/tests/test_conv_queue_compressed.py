"""驗證 connectivity/conv.py 的壓縮版佇列建構(任務 7 第二階段第 1 段)。
對應 docs/math/conv事件佇列壓縮版推導.md。

分兩類測試:
1. 直接對 `_compress_candidates`/`_affine_with_catchup` 這兩個內部函式,用
   推導文件第 2、3、4 節的手算範例當黃金測試——這兩個函式合起來是壓縮版
   跟「密集佈局」唯一的數值差異來源,獨立驗證過,信心比整條 pipeline 一次
   測完還扎實。
2. 對外部介面 `build_conv_queue_compressed`,用「壓縮版 vs 透明參考
   (`salt_core/tests/_reference.py` 的 `dense_conv_affine_map`——笨方法建密集
   (a,b),clip+where 而非 scatter,跟壓縮版不共用任何機制)run_layer_forward
   之後的 v_final 要一致」當主要驗證手段。梯度用有限差分驗參考 forward
   (純仿射 v_final 對 W/gain 恰好線性,中央差分數學上精確),完全繞開
   autodiff;只有真的 fire 的場景才退回 autodiff-vs-autodiff(參考仍可微)。
   幾何變化(各種 N/S/P、多 OC/IC)+ 跨層梯度在 test_conv_geometry.py。
"""

import jax
import jax.numpy as jnp
import numpy as np

from salt_core.chunk_scan import run_layer_forward
from salt_core.connectivity.conv import (_affine_with_catchup, _compress_candidates,
                                          build_conv_queue_compressed)
from salt_core.tests._reference import dense_conv_affine_map, finite_diff_grad

TOL = 1e-6

# 跟 test_conv_queue.py 共用同一組 fixture 數字,方便直接沿用那邊已經手算
# 過、對過帳的期望值。
TAU = 4.0
K, S, P = 3, 2, 1
H_OUT = W_OUT = 3


def assert_allclose(actual, expected, msg, tol=TOL):
    actual = float(actual)
    expected = float(expected)
    assert abs(actual - expected) < tol, f"{msg}: got {actual}, expected {expected}"


def _make_weight(oc_offset=0.0):
    """W[0,0,ky,kx] = ky*3+kx+1(+oc_offset),跟 test_conv_queue.py 同一個
    生成規則,shape (1,1,3,3)。"""
    base = jnp.array([[ky * 3 + kx + 1 for kx in range(3)] for ky in range(3)],
                      dtype=jnp.float32)
    return (base + oc_offset)[None, None, :, :]


# ============================================================================
# 第一類:直接測 _compress_candidates / _affine_with_catchup,用推導文件
# 第 2、3、4 節的手算範例當黃金測試。
# ============================================================================

def test_compress_candidates_matches_worked_example():
    """推導文件第 2 節的數值例子:4 筆事件、單軸 N=2,候選 (n,j) 清單
    排序+分段計數之後,神經元 5/6/7 應該收到文件表格列出的那幾欄。

    n_out_spatial 取 8(神經元 id 5,6,7 都落在 [0,8) 內),不合法候選標成
    sentinel=8(=n_out_spatial,文件裡用 99 示意,這裡改用剛好等於
    n_out_spatial 的值,呼叫慣例上兩者等價,只要落在 [n_out_spatial,∞)
    都會被 scatter 的 mode='drop' 丟掉)。
    """
    n_flat = jnp.array([5, 8, 5, 7, 6, 8, 5, 6], dtype=jnp.int32)
    j_flat = jnp.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=jnp.int32)
    n_out_spatial = 8
    L = 3
    n_events = 4  # j 值域 [0,4)

    local_to_global_j, n_real = _compress_candidates(n_flat, j_flat, n_out_spatial, L, n_events)

    assert local_to_global_j.shape == (n_out_spatial, L)
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


def test_affine_with_catchup_matches_worked_example():
    """推導文件第 3、4.4 節的數值例子:事件時間 [1,2,3,5],tau=4(衰減底數
    0.75)。神經元 7 的情況(最後一筆相關事件 j=1,t=2,不等於全域最後一筆
    j=3,t=5)是第 4 節證明「補位不能直接補 identity」的反例本身,神經元 6
    (最後一筆相關事件剛好是全域最後一筆)驗證 catch-up 退化成 identity 不
    會把已經算對的結果弄壞。
    """
    # 用 3 個 row 直接對應神經元 5/6/7(這個函式本身不在乎 n 的實際數值,
    # 只在乎每一 row 自己的子序列跟 n_real,所以不需要真的建 n_out_spatial=8
    # 的完整陣列)。t_gathered 在 pad 欄位(不影響結果)填任意值示意。
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

    maps, delta_t = _affine_with_catchup(t_gathered, n_real, global_last_time, TAU, b_real)

    # 神經元 5(索引 0):全部都是真 tap,沒有補位
    assert_allclose(maps.a[0, 0], 0.75, "神經元5 col0 a")
    assert_allclose(maps.a[0, 1], 0.75, "神經元5 col1 a")
    assert_allclose(maps.a[0, 2], 0.421875, "神經元5 col2 a")
    assert_allclose(maps.b[0, 0], 0.2, "神經元5 col0 b")
    assert_allclose(maps.b[0, 1], 0.4, "神經元5 col1 b")
    assert_allclose(maps.b[0, 2], 0.5, "神經元5 col2 b")
    assert list(delta_t[0]) == [1, 1, 3], "神經元5 三個真 tap 的整數 Δt(1,2,5 的相鄰差)"

    # 神經元 6(索引 1):col2 是 catch-up,但因為最後相關事件剛好等於全域
    # 最後一筆,退化成 a=1,b=0(第 4.4 節)
    assert_allclose(maps.a[1, 0], 0.421875, "神經元6 col0 a")
    assert_allclose(maps.a[1, 1], 0.5625, "神經元6 col1 a")
    assert_allclose(maps.a[1, 2], 1.0, "神經元6 catch-up 退化 a")
    assert_allclose(maps.b[1, 0], -0.3, "神經元6 col0 b")
    assert_allclose(maps.b[1, 1], 0.6, "神經元6 col1 b")
    assert_allclose(maps.b[1, 2], 0.0, "神經元6 catch-up b")
    assert list(delta_t[1]) == [3, 2, 0], "神經元6 catch-up Δt=5-5=0,跟 a=1 一致"

    # 神經元 7(索引 2):col1 是真正的 catch-up(Δt=5-2=3),col2 是純 identity
    assert_allclose(maps.a[2, 0], 0.5625, "神經元7 col0 a")
    assert_allclose(maps.a[2, 1], 0.421875, "神經元7 catch-up a(這是第4.1節反例證明過的關鍵格)")
    assert_allclose(maps.a[2, 2], 1.0, "神經元7 identity a")
    assert_allclose(maps.b[2, 0], 0.7, "神經元7 col0 b")
    assert_allclose(maps.b[2, 1], 0.0, "神經元7 catch-up b")
    assert_allclose(maps.b[2, 2], 0.0, "神經元7 identity b")
    assert list(delta_t[2]) == [2, 3, 0], "神經元7 catch-up Δt=5-2=3,identity Δt=0"


def test_affine_with_catchup_identity_pitfall_matches_hand_derivation():
    """第 4.1 節的反例本身:如果補位直接補 identity(不做 catch-up),神經元
    7 的最終電壓會算成 0.7,但正確答案(密集版)是 0.2953125——這個測試
    直接照文件的手算過程重算一次,確認 `_affine_with_catchup` 給的是正確
    答案,不是「補 identity」那個錯誤答案。"""
    t_gathered = jnp.array([[2.0, 2.0, 2.0]])  # 神經元 7,只有 j=1(t=2)
    n_real = jnp.array([1], dtype=jnp.int32)
    global_last_time = jnp.asarray(5.0)
    b_real = jnp.array([[0.7, 0.0, 0.0]])

    maps, _delta_t = _affine_with_catchup(t_gathered, n_real, global_last_time, TAU, b_real)

    v0 = 0.0
    v1 = maps.a[0, 0] * v0 + maps.b[0, 0]
    v2 = maps.a[0, 1] * v1 + maps.b[0, 1]
    v3 = maps.a[0, 2] * v2 + maps.b[0, 2]

    assert_allclose(v1, 0.7, "x0")
    assert_allclose(v2, 0.2953125, "x1(正確答案,不是錯誤的 0.7)")
    assert_allclose(v3, 0.2953125, "x2")


# ============================================================================
# 第二類:build_conv_queue_compressed 對外介面,跟**透明參考實作**
# (salt_core/tests/_reference.py,笨方法建密集 (a,b),不共用壓縮版任何機制)
# 的 v_final 一致性 + 邊界情況。梯度用有限差分驗參考 forward,繞開 autodiff。
# ============================================================================

def _run_ref(event_times, x, y, c, W, v_th, max_steps, n_real_events=None, event_gain=None):
    """透明參考:笨方法建密集 (a,b) -> 真正的 run_layer_forward。"""
    n_real_events = event_times.shape[0] if n_real_events is None else n_real_events
    maps = dense_conv_affine_map(event_times, x, y, c, W, TAU, S, P, H_OUT, W_OUT,
                                  gain=event_gain, n_real=n_real_events)
    return run_layer_forward(maps, v_th, chunk_size=max_steps, max_steps=max_steps,
                              n_real_events=n_real_events)


def _run_compressed(event_times, x, y, c, W, v_th, max_queue_len, max_steps,
                     n_real_events=None, event_gain=None):
    n_real_events = event_times.shape[0] if n_real_events is None else n_real_events
    cq = build_conv_queue_compressed(event_times, x, y, c, W, TAU, S, P, H_OUT, W_OUT,
                                      max_queue_len, event_gain=event_gain,
                                      n_real_events=n_real_events)
    result = run_layer_forward(cq.maps, v_th, chunk_size=max_steps, max_steps=max_steps,
                                n_real_events=cq.n_real_events)
    return result, cq


def _ref_vfinal_all_affine(event_times, x, y, c, W, n_real_events=None, event_gain=None):
    """純仿射(v_th=1e9,不 fire)時所有神經元的 v_final,用 numpy float64 直接
    折疊參考 (a,b),**不經過 run_layer_forward**——給 finite_diff_grad 當乾淨、
    高精度的梯度 oracle。"""
    maps = dense_conv_affine_map(event_times, x, y, c, W, TAU, S, P, H_OUT, W_OUT,
                                  gain=event_gain, n_real=n_real_events)
    a = np.asarray(maps.a, dtype=np.float64)
    b = np.asarray(maps.b, dtype=np.float64)
    v = np.zeros(a.shape[0])
    for j in range(a.shape[1]):
        v = a[:, j] * v + b[:, j]
    return v


def _grad_compressed(event_times, x, y, c, v_th, max_queue_len, max_steps, neuron_idx=0,
                      n_real_events=None, event_gain=None):
    def loss_fn(W):
        result, _ = _run_compressed(event_times, x, y, c, W, v_th, max_queue_len, max_steps,
                                     n_real_events, event_gain)
        return result.v_final[neuron_idx]
    return jax.grad(loss_fn)


def _fd_grad_ref_wrt_W(event_times, x, y, c, W, reduce, n_real_events=None, event_gain=None):
    """對參考 forward(純仿射)做 W 的有限差分梯度。`reduce(v_vec) -> 純量`
    決定 loss(取某顆神經元 or sum)。

    純仿射 v_final 對每個 W 元素**恰好是線性函式**(tap 合法性只看座標、不看
    W 值),所以中央差分在數學上精確,eps 開大(0.1)只是為了壓低 float32
    roundoff 被 1/eps 放大的量,不會有截斷誤差。"""
    return finite_diff_grad(
        lambda w: float(reduce(_ref_vfinal_all_affine(
            event_times, x, y, c, w, n_real_events, event_gain))),
        np.asarray(W), eps=0.1)


def _grad_ref_autodiff_firing(event_times, x, y, c, v_th, max_steps, neuron_idx=0):
    """真的 fire 的場景:有限差分不可靠(電壓在 fire 邊界不連續),退回
    autodiff——但走的是**透明參考** dense_conv_affine_map(仍可微、跟壓縮版
    不共用機制),不是壓縮版自己。"""
    def loss_fn(W):
        maps = dense_conv_affine_map(event_times, x, y, c, W, TAU, S, P, H_OUT, W_OUT)
        return run_layer_forward(maps, v_th, chunk_size=max_steps,
                                  max_steps=max_steps, n_real_events=maps.a.shape[1]).v_final[neuron_idx]
    return jax.grad(loss_fn)


_GRAD_TOL = 2e-4  # 有限差分(eps=1e-4、參考 forward numpy float64)的精度量級


def test_compressed_matches_dense_single_event():
    """單一事件、4 個候選都合法(跟 test_conv_queue.py 的
    test_single_event_all_four_candidates_valid 同一個場景)。"""
    W = _make_weight()
    event_times = jnp.array([1.0])
    x = jnp.array([1]); y = jnp.array([1]); c = jnp.array([0])

    ref = _run_ref(event_times, x, y, c, W, v_th=1e9, max_steps=1)
    result, cq = _run_compressed(event_times, x, y, c, W, v_th=1e9,
                                  max_queue_len=1, max_steps=1)

    assert bool(jnp.allclose(ref.v_final, result.v_final, atol=TOL)), \
        (ref.v_final, result.v_final)

    # 手算過的期望值(idx0=9.0, idx1=7.0, idx3=3.0, idx4=1.0, 其餘 0)
    expected = jnp.zeros(9).at[jnp.array([0, 1, 3, 4])].set(jnp.array([9.0, 7.0, 3.0, 1.0]))
    assert bool(jnp.allclose(result.v_final, expected, atol=TOL)), result.v_final


def test_compressed_matches_dense_two_events_with_degenerate_catchup():
    """兩筆事件、邊界事件只有 1/4 候選合法(跟 test_conv_queue.py 的
    test_two_events_boundary_event_has_fewer_valid_candidates 同一個場景)。
    max_queue_len 刻意取 3(比任何神經元真正需要的 2 還多),逼出 catch-up
    欄位——這個場景每個神經元最後一筆相關事件都剛好是全域最後一筆事件,
    catch-up 會退化成 identity(Δt=0),但仍然驗證了「多出來的欄位不會把
    已經算對的結果弄壞」。"""
    W = _make_weight()
    event_times = jnp.array([1.0, 2.0])
    x = jnp.array([1, 0]); y = jnp.array([1, 0]); c = jnp.array([0, 0])

    ref = _run_ref(event_times, x, y, c, W, v_th=1e9, max_steps=2)
    result, cq = _run_compressed(event_times, x, y, c, W, v_th=1e9,
                                  max_queue_len=3, max_steps=3)

    assert bool(jnp.allclose(ref.v_final, result.v_final, atol=TOL)), \
        (ref.v_final, result.v_final)

    expected = jnp.zeros(9).at[jnp.array([0, 1, 3, 4])].set(
        jnp.array([11.75, 5.25, 2.25, 0.75]))
    assert bool(jnp.allclose(result.v_final, expected, atol=TOL)), result.v_final


def test_compressed_matches_dense_with_genuine_non_degenerate_catchup():
    """三筆事件,刻意讓某個神經元的最後相關事件早於全域最後一筆事件,逼出
    真正非退化的 catch-up(不是上一個測試那種剛好等於 0 的退化情況)。

    事件 0,1 跟上一個測試相同;事件 2 在 (2,2,t=5),只碰得到 idx4(算法見
    docstring 內註解),不影響 idx1——idx1 的最後相關事件停在 t=1,但全域
    最後一筆事件變成 t=5,idx1 這一列會用到真正的 catch-up 衰減。"""
    W = _make_weight()
    event_times = jnp.array([1.0, 2.0, 5.0])
    x = jnp.array([1, 0, 2]); y = jnp.array([1, 0, 2]); c = jnp.array([0, 0, 0])

    ref = _run_ref(event_times, x, y, c, W, v_th=1e9, max_steps=3)
    result, cq = _run_compressed(event_times, x, y, c, W, v_th=1e9,
                                  max_queue_len=3, max_steps=3)

    assert bool(jnp.allclose(ref.v_final, result.v_final, atol=TOL)), \
        (ref.v_final, result.v_final)

    # idx1 真的用到非退化 catch-up:col0 real(a=0.75,b=7)->x0=7;
    # catch-up Δt=5-1=4 -> a=0.75^4=0.31640625 -> x1=7*0.31640625=2.21484375;
    # 沒有第三欄真的可用(n_real=1,L=3,col2 是 identity)-> x2 不變。
    assert_allclose(result.v_final[1], 2.21484375, "idx1 用到非退化 catch-up")


def test_compressed_matches_dense_with_pad_events():
    """加一筆 pad 事件(座標 (0,0,0),偽裝合法座標,密集版第 8.1 節的危險
    在壓縮版一樣要擋)。用 n_real_events=2 告訴兩個版本只有前兩筆事件是真的,
    第三筆(pad)不該被壓縮版當成任何神經元的真 tap。"""
    W = _make_weight()
    event_times = jnp.array([1.0, 2.0, 1e12])  # 第三筆是 pad,時間刻意設超大
    x = jnp.array([1, 0, 0]); y = jnp.array([1, 0, 0]); c = jnp.array([0, 0, 0])
    n_real_events = 2

    ref = _run_ref(event_times, x, y, c, W, v_th=1e9, max_steps=3,
                        n_real_events=n_real_events)
    result, cq = _run_compressed(event_times, x, y, c, W, v_th=1e9,
                                  max_queue_len=3, max_steps=3,
                                  n_real_events=n_real_events)

    assert bool(jnp.allclose(ref.v_final, result.v_final, atol=TOL)), \
        (ref.v_final, result.v_final)

    # idx0 真正該收到的合法 tap 是 2 筆(事件 0 在 (1,1) 貢獻 k=(2,2)、
    # 事件 1 在 (0,0) 貢獻 k=(1,1),兩者座標不同、都是真事件);pad 事件
    # (事件 2)座標也是 (0,0)、跟事件 1 一樣會落在 idx0 的候選裡,如果沒有
    # 被 n_real_events 正確排除,會被誤算成第 3 筆,變成 n_real=3。
    assert int(cq.n_real_events[0]) == 2, \
        f"idx0 應該只收到 2 筆真實 tap,pad 事件不該被算進去,拿到 {int(cq.n_real_events[0])}"
    assert list(map(int, cq.local_to_global_j[0][:2])) == [0, 1], \
        "idx0 的兩個真實 tap 應該是事件 0、事件 1,pad 事件(j=2)不該出現"


def test_compressed_neuron_with_zero_real_events_matches_dense_zero():
    """邊界情況:某個神經元完全沒有相關事件(第 1 階段測試重點之一)。單一
    事件只碰得到 4 個角落神經元,中心以外、四角以外的神經元(例如 idx2)
    完全沒有候選,n_real 應該是 0,v_final 應該是 0,不該 crash/NaN。"""
    W = _make_weight()
    event_times = jnp.array([1.0])
    x = jnp.array([1]); y = jnp.array([1]); c = jnp.array([0])

    ref = _run_ref(event_times, x, y, c, W, v_th=1e9, max_steps=1)
    result, cq = _run_compressed(event_times, x, y, c, W, v_th=1e9,
                                  max_queue_len=1, max_steps=1)

    assert int(cq.n_real_events[2]) == 0
    assert_allclose(result.v_final[2], 0.0, "沒有任何相關事件的神經元,v_final 應該是 0")
    assert_allclose(ref.v_final[2], 0.0, "密集版同一顆神經元也應該是 0(交叉確認)")
    assert not bool(jnp.any(jnp.isnan(result.v_final))), "不該出現 NaN"


def test_compressed_exactly_fills_max_queue_len_no_padding_needed():
    """邊界情況:某個神經元收到的真實 tap 數剛好等於 max_queue_len,完全
    沒有 catch-up/identity 欄位可補(第 1 階段測試重點之一)。用兩筆事件都
    落在 idx0,max_queue_len 剛好設成 2。"""
    W = _make_weight()
    event_times = jnp.array([1.0, 3.0])
    x = jnp.array([1, 1]); y = jnp.array([1, 1]); c = jnp.array([0, 0])  # 兩筆都在同一個像素

    ref = _run_ref(event_times, x, y, c, W, v_th=1e9, max_steps=2)
    result, cq = _run_compressed(event_times, x, y, c, W, v_th=1e9,
                                  max_queue_len=2, max_steps=2)

    assert int(cq.n_real_events[0]) == 2, "idx0 應該收到剛好 2 筆真實事件,等於 max_queue_len"
    assert bool(jnp.allclose(ref.v_final, result.v_final, atol=TOL)), \
        (ref.v_final, result.v_final)


def test_compressed_oc_independent_candidacy_only_weight_differs():
    """第 1 節:候選篩選/local_to_global_j/n_real_events 應該完全不受 oc
    影響,只有 b(權重)不同——用 OC=2、兩個 channel 權重值差很多,確認
    a、local_to_global_j、n_real_events 在兩個 oc 對應的 row 上完全一樣,
    只有 b 不同。"""
    w0 = _make_weight(oc_offset=0.0)   # (1,1,3,3)
    w1 = _make_weight(oc_offset=100.0)  # 數值差很大,確保不是巧合相等
    W = jnp.concatenate([w0, w1], axis=0)  # (2,1,3,3)

    event_times = jnp.array([1.0, 2.0])
    x = jnp.array([1, 0]); y = jnp.array([1, 0]); c = jnp.array([0, 0])
    max_queue_len = 3
    n_out_spatial = H_OUT * W_OUT

    cq = build_conv_queue_compressed(event_times, x, y, c, W, TAU, S, P, H_OUT, W_OUT,
                                      max_queue_len, n_real_events=event_times.shape[0])

    for spatial_idx in range(n_out_spatial):
        oc0_row = spatial_idx               # oc=0 的第 spatial_idx 個神經元
        oc1_row = n_out_spatial + spatial_idx  # oc=1 的同一個空間位置

        assert list(map(int, cq.local_to_global_j[oc0_row])) == \
            list(map(int, cq.local_to_global_j[oc1_row])), \
            f"空間位置 {spatial_idx}:local_to_global_j 不該受 oc 影響"
        assert int(cq.n_real_events[oc0_row]) == int(cq.n_real_events[oc1_row]), \
            f"空間位置 {spatial_idx}:n_real_events 不該受 oc 影響"
        assert bool(jnp.allclose(cq.maps.a[oc0_row], cq.maps.a[oc1_row], atol=TOL)), \
            f"空間位置 {spatial_idx}:a(衰減)只跟時間差有關,不該受 oc 影響"

    # b 應該不同(權重確實不同,不是巧合)——至少有真 tap 的位置要能驗證這件事
    has_real_tap = cq.n_real_events[:n_out_spatial] > 0
    assert bool(jnp.any(has_real_tap)), "這個場景至少要有一個神經元收到真 tap 才能驗證 b 不同"
    real_spatial_idx = int(jnp.argmax(has_real_tap))
    b_oc0 = cq.maps.b[real_spatial_idx, 0]
    b_oc1 = cq.maps.b[n_out_spatial + real_spatial_idx, 0]
    assert abs(float(b_oc0) - float(b_oc1)) > 1.0, "不同 oc 的權重差很大,b 應該明顯不同"


def test_compressed_matches_dense_on_realistic_random_case():
    """比手算例子更大規模的隨機案例(第 1 階段測試重點:「壓縮版 vs 密集版
    v_final 一致性(隨機案例)」)。多筆事件、隨機座標/權重,max_queue_len
    留寬(等於全域事件數,保證不會截斷),驗證壓縮版跟密集版逐神經元
    v_final 一致,以及整個 W 的梯度陣列一致(用 3 個不同 seed 各跑一次——
    壓縮版的排序/scatter 邏輯只在特定資料型態下才會出錯,例如很多事件剛好
    打到同一個神經元、或合法/不合法候選混雜的方式恰好是單一 seed 沒測到的
    排列,多換幾個 seed 比單一固定案例更有機會抓到這類問題)。"""
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
        result, cq = _run_compressed(event_times, x, y, c, W, v_th=1e9,
                                      max_queue_len=n_events, max_steps=n_events)

        assert bool(jnp.allclose(ref.v_final, result.v_final, atol=1e-4)), \
            (seed, ref.v_final, result.v_final)

        grad_ref = _fd_grad_ref_wrt_W(event_times, x, y, c, W, reduce=lambda v: v[0])
        grad_compressed = _grad_compressed(event_times, x, y, c, v_th=1e9,
                                            max_queue_len=n_events, max_steps=n_events)(W)
        assert bool(jnp.allclose(grad_ref, grad_compressed, atol=_GRAD_TOL)), \
            (seed, grad_ref, grad_compressed)


def test_compressed_gradient_matches_dense_pure_affine():
    """沿用 test_compressed_matches_dense_two_events_with_degenerate_catchup
    的兩事件邊界場景(v_th=1e9,不 fire,純仿射),loss=v_final[0]。比對
    整個 W 的梯度陣列(不是只挑密集版原本手算過的兩個位置)——壓縮版正確
    與否要看每個位置都對,不能只挑巧合對的位置。密集版本身已經被驗證過,
    直接拿它的梯度當答案,不用另外手推。"""
    event_times = jnp.array([1.0, 2.0])
    x = jnp.array([1, 0]); y = jnp.array([1, 0]); c = jnp.array([0, 0])

    grad_ref = _fd_grad_ref_wrt_W(event_times, x, y, c, _make_weight(), reduce=lambda v: v[0])
    grad_compressed = _grad_compressed(event_times, x, y, c, v_th=1e9,
                                        max_queue_len=3, max_steps=3)(_make_weight())

    assert bool(jnp.allclose(grad_ref, grad_compressed, atol=_GRAD_TOL)), \
        (grad_ref, grad_compressed)


def test_compressed_dropped_candidate_gradient_matches_dense():
    """沿用密集版 test_dropped_candidate_gradient_is_exactly_zero 的邊界事件
    場景(單一事件 (0,0,c=0,t=1),k=-1 wraparound,3 個候選會被判定不合法、
    丟進 _compress_candidates 的 sentinel/trash 那組)。這條路徑(候選被排序
    進假 id 那一段、local_rank 完全不會被用到)完全沒被前面幾個測試覆蓋過,
    是壓縮版新增程式碼裡最容易藏梯度洩漏的地方——密集版已經驗證過這個
    場景的正確梯度(唯一真實 tap 是 1.0,其餘精確是 0),直接拿來當答案。
    """
    event_times = jnp.array([1.0])
    x = jnp.array([0]); y = jnp.array([0]); c = jnp.array([0])

    grad_ref = _fd_grad_ref_wrt_W(event_times, x, y, c, _make_weight(), reduce=lambda v: v[0])
    grad_compressed = _grad_compressed(event_times, x, y, c, v_th=1e9,
                                        max_queue_len=1, max_steps=1)(_make_weight())

    # 唯一真實 tap W[0,0,1,1] 梯度 1.0;3 個只被丟棄候選(k=-1 wraparound)碰過的
    # 位置,壓縮版 autodiff 要精確給 0(mode='drop' 不留梯度)。
    assert_allclose(grad_compressed[0, 0, 1, 1], 1.0, "唯一真實 tap 的梯度")
    for ky, kx in [(1, 2), (2, 1), (2, 2)]:
        assert_allclose(grad_compressed[0, 0, ky, kx], 0.0,
                         f"W[0,0,{ky},{kx}] 只被丟棄候選碰過,壓縮版梯度應該精確是 0")
    assert bool(jnp.allclose(grad_ref, grad_compressed, atol=_GRAD_TOL)), \
        (grad_ref, grad_compressed)


def test_compressed_gradient_matches_dense_with_genuine_catchup():
    """沿用 test_compressed_matches_dense_with_genuine_non_degenerate_catchup
    的三事件場景(idx1 真正用到非退化 catch-up:a 只依賴時間差、不依賴 W)。
    確認 catch-up 那一格不會意外洩漏梯度給某個 W 位置、也不會意外阻斷本來
    該有的梯度——密集版同一個場景本來就會算出正確梯度,直接拿來比。"""
    event_times = jnp.array([1.0, 2.0, 5.0])
    x = jnp.array([1, 0, 2]); y = jnp.array([1, 0, 2]); c = jnp.array([0, 0, 0])

    grad_ref = _fd_grad_ref_wrt_W(event_times, x, y, c, _make_weight(), reduce=lambda v: v[1])
    grad_compressed = _grad_compressed(event_times, x, y, c, v_th=1e9, max_queue_len=3,
                                        max_steps=3, neuron_idx=1)(_make_weight())

    assert bool(jnp.allclose(grad_ref, grad_compressed, atol=_GRAD_TOL)), \
        (grad_ref, grad_compressed)


def test_compressed_gradient_matches_dense_with_pad_events():
    """沿用 test_compressed_matches_dense_with_pad_events 的 pad 事件場景
    (n_real_events)。確認這階段自己延伸補的「事件必須是真的」判斷,沒有讓
    假事件的座標意外洩漏梯度進 W(pad 事件座標 (0,0) 跟事件 1 相同,如果
    pad 過濾漏做,pad 事件會被當成第 3 個真 tap,梯度會多算進一條不該有的
    路徑)。"""
    event_times = jnp.array([1.0, 2.0, 1e12])
    x = jnp.array([1, 0, 0]); y = jnp.array([1, 0, 0]); c = jnp.array([0, 0, 0])
    n_real_events = 2

    grad_ref = _fd_grad_ref_wrt_W(event_times, x, y, c, _make_weight(), reduce=lambda v: v[0],
                                   n_real_events=n_real_events)
    grad_compressed = _grad_compressed(event_times, x, y, c, v_th=1e9, max_queue_len=3,
                                        max_steps=3, n_real_events=n_real_events)(_make_weight())

    assert bool(jnp.allclose(grad_ref, grad_compressed, atol=_GRAD_TOL)), \
        (grad_ref, grad_compressed)


# ============================================================================
# 第三類:任務7第二階段第2段(run_layer_forward 呼叫端把 n_real_events 從
# 純量換成 (n_out,) 陣列)。第一類/第二類測試已經用 cq.n_real_events(陣列)
# 餵過 run_layer_forward,但全部場景都用 v_th=1e9(純積分器,不會 fire),
# 只比對過 v_final——spike_mask/spike_event_idx/s_value 這三個欄位、以及
# 真的 fire、經過 atan_spike surrogate 那條梯度路徑,完全沒被驗證過。
# ============================================================================

def test_compressed_matches_dense_spike_details_and_gradient_when_neuron_fires():
    """5 筆事件,v_th 調到會讓 idx0 在佇列中段(不是第一筆)真的 fire、
    reset,之後還有更多欄位要繼續處理(真 tap、catch-up、identity 都有)。

    刻意在 idx0 的兩筆真實 tap(j=0,j=2)中間插入只跟其他神經元有關的事件
    (j=1,j=3 落在 (2,2),不影響 idx0),讓 idx0 的壓縮佇列局部欄位跟全域 j
    不是巧合地相等(local_to_global_j[0]=[0,2,4,...]:local col1 對應
    global j=2,不是 1)——這樣才能真的測到「局部欄位轉全域 j」這一步,不是
    局部欄位剛好等於全域 j 的退化情況。

    比對範圍刻意涵蓋 spike_mask 整個陣列(不只 idx0)、s_value 整個陣列
    (驗證 catch-up/identity 那幾欄的 s_value 正確排除、不會被誤算成有效
    tap)、以及把壓縮版 spike_event_idx(局部)透過 cq.local_to_global_j 轉成
    全域 j 之後,在真的有 fire 的位置要跟密集版 spike_event_idx(本來就是
    全域 j)完全一致。最後比對 v_final[0](idx0,真的 fire 過的神經元)對整個
    W 的梯度——第一階段全部梯度測試都是 v_th=1e9 不 fire 的純仿射路徑,這是
    壓縮版第一次驗證「真的 fire、經過 soft reset/atan_spike surrogate」那條
    梯度路徑。
    """
    W = _make_weight()
    event_times = jnp.array([1.0, 1.5, 2.0, 2.5, 5.0])
    x = jnp.array([1, 2, 1, 2, 0]); y = jnp.array([1, 2, 1, 2, 0]); c = jnp.array([0, 0, 0, 0, 0])
    v_th = 15.0
    max_queue_len = 5  # 留寬到等於全域事件數,這個測試的重點不是 L 太小截斷

    ref = _run_ref(event_times, x, y, c, W, v_th, max_steps=5)
    result, cq = _run_compressed(event_times, x, y, c, W, v_th, max_queue_len, max_steps=5)

    # 前置確認:idx0 真的如預期在局部欄位1(不是欄位0)fire,且局部欄位1
    # 對應的全域 j 是 2,不是 1——這是這個測試場景成立的前提。
    assert bool(result.spike_mask[0, 0]), "idx0 應該要 fire"
    assert int(result.spike_event_idx[0, 0]) == 1, "idx0 應該在局部欄位1(第二個真實 tap)fire"
    assert int(cq.local_to_global_j[0, 1]) == 2, "局部欄位1 應該對應全域 j=2,不是巧合等於 1"
    assert int(ref.spike_event_idx[0, 0]) == 2, "密集版 idx0 應該在全域 j=2 fire"

    # spike_mask、s_value 整個陣列(全部神經元、全部步數)逐位元比對——不只
    # 看有沒有 fire 的那個位置,也要看沒 fire 的位置(catch-up/identity 那幾
    # 欄)兩邊算出的 s_value 是否一致排除掉。
    assert bool(jnp.array_equal(ref.spike_mask, result.spike_mask)), \
        (ref.spike_mask, result.spike_mask)
    assert bool(jnp.allclose(ref.s_value, result.s_value, atol=TOL)), \
        (ref.s_value, result.s_value)

    # spike_event_idx 只在 spike_mask=True 的位置有意義(docstring 明講)。
    # 把壓縮版的局部欄位透過 cq.local_to_global_j 轉成全域 j,拿去跟密集版
    # (本來就是全域 j)比對,只比 spike_mask 為真的位置。
    global_j_from_compressed = jnp.take_along_axis(
        cq.local_to_global_j, result.spike_event_idx, axis=1)
    assert bool(jnp.all(jnp.where(
        result.spike_mask, global_j_from_compressed == ref.spike_event_idx, True))), \
        (global_j_from_compressed, ref.spike_event_idx, result.spike_mask)

    grad_ref = _grad_ref_autodiff_firing(event_times, x, y, c, v_th, max_steps=5,
                                          neuron_idx=0)(W)
    grad_compressed = _grad_compressed(event_times, x, y, c, v_th, max_queue_len, max_steps=5,
                                        neuron_idx=0)(W)
    assert bool(jnp.allclose(grad_ref, grad_compressed, atol=1e-4)), \
        (grad_ref, grad_compressed)


# ============================================================================
# 第四類:段 4 補上的 event_gain 支援(段 1 當時刻意排除,因為 conv1 直接吃
# 原始事件、沒有上游層)。ConvNetCompressed 的 conv2 靠 event_gain 把 conv1
# 的 s_spike 乘進來,重新接通對 w_conv1 的跨層梯度路徑。以「密集版
# build_conv_queue(event_gain=g) 已驗證過」當基準,比對壓縮版帶同一個 g。
# ============================================================================

def test_compressed_matches_dense_with_event_gain():
    """隨機多事件案例,帶一個非全 1 的 event_gain(每個全域事件一個增益)。
    比對壓縮版 vs 密集版:v_final 逐神經元一致、整個 W 的梯度一致(多個
    seed——event_gain 用 safe_j gather,跟座標/權重 gather 同一個索引,某些
    seed 的候選排列才會踩到邊界)。"""
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
        result, _ = _run_compressed(event_times, x, y, c, W, v_th=1e9,
                                     max_queue_len=n_events, max_steps=n_events,
                                     event_gain=event_gain)
        assert bool(jnp.allclose(ref.v_final, result.v_final, atol=1e-4)), \
            (seed, ref.v_final, result.v_final)

        grad_ref = _fd_grad_ref_wrt_W(event_times, x, y, c, W, reduce=lambda v: v[0],
                                       event_gain=event_gain)
        grad_compressed = _grad_compressed(event_times, x, y, c, v_th=1e9,
                                            max_queue_len=n_events, max_steps=n_events,
                                            event_gain=event_gain)(W)
        assert bool(jnp.allclose(grad_ref, grad_compressed, atol=_GRAD_TOL)), \
            (seed, grad_ref, grad_compressed)


def test_compressed_event_gain_gradient_matches_dense():
    """dL/d(event_gain) 一致——這是 event_gain 存在的理由:跨層時它是上一層
    的 s_spike,梯度要能穿過它傳回上一層權重。W 固定,對整個 event_gain 向量
    求導,壓縮版跟密集版必須給出同一個梯度。"""
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

    def compressed_loss(g):
        result, _ = _run_compressed(event_times, x, y, c, W, v_th=1e9,
                                     max_queue_len=n_events, max_steps=n_events, event_gain=g)
        return jnp.sum(result.v_final)

    grad_ref = finite_diff_grad(
        lambda g: float(_ref_vfinal_all_affine(event_times, x, y, c, W, event_gain=g).sum()),
        np.asarray(event_gain0), eps=0.1)  # v_final 對 gain 也是線性,中央差分精確
    grad_compressed = jax.grad(compressed_loss)(event_gain0)
    assert bool(jnp.allclose(grad_ref, grad_compressed, atol=_GRAD_TOL)), \
        (grad_ref, grad_compressed)


TESTS = [
    test_compress_candidates_matches_worked_example,
    test_affine_with_catchup_matches_worked_example,
    test_affine_with_catchup_identity_pitfall_matches_hand_derivation,
    test_compressed_matches_dense_single_event,
    test_compressed_matches_dense_two_events_with_degenerate_catchup,
    test_compressed_matches_dense_with_genuine_non_degenerate_catchup,
    test_compressed_matches_dense_with_pad_events,
    test_compressed_neuron_with_zero_real_events_matches_dense_zero,
    test_compressed_exactly_fills_max_queue_len_no_padding_needed,
    test_compressed_oc_independent_candidacy_only_weight_differs,
    test_compressed_matches_dense_on_realistic_random_case,
    test_compressed_gradient_matches_dense_pure_affine,
    test_compressed_dropped_candidate_gradient_matches_dense,
    test_compressed_gradient_matches_dense_with_genuine_catchup,
    test_compressed_gradient_matches_dense_with_pad_events,
    test_compressed_matches_dense_spike_details_and_gradient_when_neuron_fires,
    test_compressed_matches_dense_with_event_gain,
    test_compressed_event_gain_gradient_matches_dense,
]


if __name__ == "__main__":
    for test in TESTS:
        test()
        print(f"PASS: {test.__name__}")
    print(f"\n全部 {len(TESTS)} 項測試通過")
