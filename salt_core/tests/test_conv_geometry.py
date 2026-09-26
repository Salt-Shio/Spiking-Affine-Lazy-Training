"""conv 佇列建構的**幾何覆蓋** + **跨層梯度**測試,step 4d 從已刪除的
test_conv_queue.py(原本測密集版 build_conv_queue)搬過來、改成打壓縮版
`build_conv_queue_compressed`,對照組換成 `salt_core/tests/_reference.py` 的
透明 numpy 參考(`dense_conv_affine_map`)。

分三塊:
1. `unravel_conv_source` round-trip(純函式,不牽涉佇列建構器)。
2. 參考本身的錨定:拿手算 literal 對 `dense_conv_affine_map` 的 (a,b),確保
   後面「壓縮版 vs 參考」不是循環驗證。
3. 壓縮版 vs 參考:各種 N(=K,S 決定的單軸扇出)、S/P、多 OC/多 IC、
   放大規模;以及 conv->conv / conv->FC 的跨層梯度(atan surrogate 斜率手算)。
"""
import math

import jax
import jax.numpy as jnp

from salt_core.chunk_scan import run_layer_forward
from salt_core.connectivity.conv import build_conv_queue_compressed, unravel_conv_source
from salt_core.connectivity.fc import build_fc_queue
from salt_core.layer_chain import extract_output_events, extract_output_events_compressed
from salt_core.tests._reference import dense_conv_affine_map

TOL = 1e-5
TAU = 4.0
A = 0.75  # 1 - 1/tau


def _tol_close(a, b, tol=TOL):
    return bool(jnp.allclose(jnp.asarray(a), jnp.asarray(b), atol=tol))


def _run_ref(event_times, x, y, c, W, tau, S, P, H_out, W_out, v_th, max_steps,
              gain=None, n_real=None):
    n_real = event_times.shape[0] if n_real is None else n_real
    maps = dense_conv_affine_map(event_times, x, y, c, W, tau, S, P, H_out, W_out,
                                  gain=gain, n_real=n_real)
    return run_layer_forward(maps, v_th, chunk_size=max_steps, max_steps=max_steps,
                              n_real_events=n_real)


def _run_compressed(event_times, x, y, c, W, tau, S, P, H_out, W_out, L, v_th, max_steps,
                     gain=None, n_real=None):
    n_real = event_times.shape[0] if n_real is None else n_real
    cq = build_conv_queue_compressed(event_times, x, y, c, W, tau, S, P, H_out, W_out, L,
                                      event_gain=gain, n_real_events=n_real)
    return run_layer_forward(cq.maps, v_th, chunk_size=max_steps, max_steps=max_steps,
                              n_real_events=cq.n_real_events)


# ============================================================================
# 1. unravel_conv_source round-trip
# ============================================================================

def test_unravel_conv_source_roundtrip():
    """攤平公式的反運算:任意 (oc,oy,ox) 攤平成 flat id,再 unravel 回來要拿到
    原本的三元組。"""
    OC, H_in, W_in = 4, 3, 5
    oc = jnp.arange(OC)[:, None, None]
    oy = jnp.arange(H_in)[None, :, None]
    ox = jnp.arange(W_in)[None, None, :]
    oc_f, oy_f, ox_f = jnp.broadcast_arrays(oc, oy, ox)
    flat = (oc_f * H_in * W_in + oy_f * W_in + ox_f).ravel()

    x, y, c = unravel_conv_source(flat, H_in, W_in)
    assert bool(jnp.all(x == ox_f.ravel()))
    assert bool(jnp.all(y == oy_f.ravel()))
    assert bool(jnp.all(c == oc_f.ravel()))


# ============================================================================
# 2. 參考本身的錨定(手算 literal)
# ============================================================================

def test_reference_anchored_n1_one_to_one():
    """N=1(K=1,S=1,P=0):每個輸出位置只跟同座標的輸入一對一,沒有空間混合。
    事件 A (0,0,t=1)->o=(0,0)=idx0;事件 B (2,2,t=2)->o=(2,2)=idx8。
    手算(a=0.75):idx0 x2=0.75*5+0=3.75;idx8 x2=0.75*0+5=5.0;其餘 0。"""
    W = jnp.array([[[[5.0]]]])  # (OC=1,IC=1,K=1,K=1)
    et = jnp.array([1.0, 2.0]); x = jnp.array([0, 2]); y = jnp.array([0, 2]); c = jnp.array([0, 0])
    maps = dense_conv_affine_map(et, x, y, c, W, TAU, S=1, P=0, H_out=3, W_out=3)

    def v_after(idx):
        return A * (A * 0.0 + maps.b[idx, 0]) + maps.b[idx, 1]

    assert _tol_close(v_after(0), 3.75)
    assert _tol_close(v_after(8), 5.0)
    for idx in (1, 2, 3, 4, 5, 6, 7):
        assert _tol_close(v_after(idx), 0.0)


def test_reference_anchored_n3_multiple_candidates_per_axis():
    """N=3(K=5,S=2,P=2):單一事件在 (4,4),兩軸各 3 個候選都合法 -> 9 個候選
    落在不同神經元。W[0,0,ky,kx]=ky*5+kx+1。逐一手算 (o_y,k_y)x(o_x,k_x)。"""
    W = (jnp.arange(25, dtype=jnp.float32) + 1).reshape(1, 1, 5, 5)
    et = jnp.array([1.0]); x = jnp.array([4]); y = jnp.array([4]); c = jnp.array([0])
    H_out = W_out = 6
    maps = dense_conv_affine_map(et, x, y, c, W, TAU, S=2, P=2, H_out=H_out, W_out=W_out)

    expected = {}
    for oy, ky in [(1, 4), (2, 2), (3, 0)]:
        for ox, kx in [(1, 4), (2, 2), (3, 0)]:
            expected[oy * W_out + ox] = float(ky * 5 + kx + 1)
    for idx in range(H_out * W_out):
        assert _tol_close(maps.b[idx, 0], expected.get(idx, 0.0)), (idx, float(maps.b[idx, 0]))


# ============================================================================
# 3. 壓縮版 vs 參考:幾何變化
# ============================================================================

def _assert_compressed_matches_ref(et, x, y, c, W, S, P, H_out, W_out, L=None,
                                    v_th=1e9, gain=None, n_real=None):
    n_events = int(et.shape[0])
    L = n_events if L is None else L
    ref = _run_ref(et, x, y, c, W, TAU, S, P, H_out, W_out, v_th, n_events, gain, n_real)
    comp = _run_compressed(et, x, y, c, W, TAU, S, P, H_out, W_out, L, v_th, L, gain, n_real)
    assert _tol_close(ref.v_final, comp.v_final, tol=1e-4), (ref.v_final, comp.v_final)


def test_compressed_matches_ref_multi_output_channels():
    W0 = jnp.arange(1, 10, dtype=jnp.float32).reshape(1, 1, 3, 3)
    W1 = (100 + jnp.arange(1, 10, dtype=jnp.float32)).reshape(1, 1, 3, 3)
    W = jnp.concatenate([W0, W1], axis=0)  # (OC=2,IC=1,3,3)
    et = jnp.array([1.0, 2.0]); x = jnp.array([1, 0]); y = jnp.array([1, 0]); c = jnp.array([0, 0])
    _assert_compressed_matches_ref(et, x, y, c, W, S=2, P=1, H_out=3, W_out=3, L=3)


def test_compressed_matches_ref_multi_input_channels():
    W_ic0 = jnp.arange(1, 10, dtype=jnp.float32).reshape(1, 1, 3, 3)
    W_ic1 = (10 + jnp.arange(1, 10, dtype=jnp.float32)).reshape(1, 1, 3, 3)
    W = jnp.concatenate([W_ic0, W_ic1], axis=1)  # (OC=1,IC=2,3,3)
    et = jnp.array([1.0, 2.0]); x = jnp.array([1, 1]); y = jnp.array([1, 1]); c = jnp.array([0, 1])
    _assert_compressed_matches_ref(et, x, y, c, W, S=2, P=1, H_out=3, W_out=3, L=3)


def test_compressed_matches_ref_n1_stride1_pad0():
    W = jnp.array([[[[5.0]]]])
    et = jnp.array([1.0, 2.0]); x = jnp.array([0, 2]); y = jnp.array([0, 2]); c = jnp.array([0, 0])
    _assert_compressed_matches_ref(et, x, y, c, W, S=1, P=0, H_out=3, W_out=3, L=2)


def test_compressed_matches_ref_n3_k5_stride2_pad2():
    W = (jnp.arange(25, dtype=jnp.float32) + 1).reshape(1, 1, 5, 5)
    et = jnp.array([1.0, 3.0, 7.0]); x = jnp.array([4, 5, 2]); y = jnp.array([4, 1, 3])
    c = jnp.array([0, 0, 0])
    _assert_compressed_matches_ref(et, x, y, c, W, S=2, P=2, H_out=6, W_out=6, L=3)


def test_compressed_matches_ref_random_various_geometry():
    """幾組隨機事件 x 幾種 (K,S,P),壓縮版 v_final 要跟參考一致。"""
    for seed, (K, S, P, H_out, W_out) in enumerate([
        (3, 2, 1, 4, 4), (1, 1, 0, 5, 5), (5, 2, 2, 3, 3), (3, 1, 1, 6, 6),
    ]):
        key = jax.random.PRNGKey(seed)
        k_t, k_x, k_y, k_c, k_w = jax.random.split(key, 5)
        n_events = 20
        et = jnp.sort(jax.random.randint(k_t, (n_events,), 0, 100).astype(jnp.float32))
        H_in = H_out * S + K
        x = jax.random.randint(k_x, (n_events,), 0, H_in).astype(jnp.int32)
        y = jax.random.randint(k_y, (n_events,), 0, H_in).astype(jnp.int32)
        c = jax.random.randint(k_c, (n_events,), 0, 2).astype(jnp.int32)
        W = jax.random.uniform(k_w, (3, 2, K, K), minval=-1.0, maxval=1.0)
        _assert_compressed_matches_ref(et, x, y, c, W, S, P, H_out, W_out, L=n_events)


def test_compressed_realistic_scale_smoke():
    """conv1 實際規模:OC=8,IC=2,H_out=W_out=64,K=3,S=2,P=1。只確認跑得動、
    shape 對、沒有 NaN/Inf(手算不現實)。"""
    OC, IC, K, S, P, H_out, W_out = 8, 2, 3, 2, 1, 64, 64
    n_events = 200
    key = jax.random.PRNGKey(0)
    k_t, k_x, k_y, k_c, k_w = jax.random.split(key, 5)
    et = jnp.sort(jax.random.uniform(k_t, (n_events,), minval=0.0, maxval=100.0))
    x = jax.random.randint(k_x, (n_events,), 0, 130).astype(jnp.int32)
    y = jax.random.randint(k_y, (n_events,), 0, 130).astype(jnp.int32)
    c = jax.random.randint(k_c, (n_events,), 0, IC).astype(jnp.int32)
    W = jax.random.normal(k_w, (OC, IC, K, K))

    cq = build_conv_queue_compressed(et, x, y, c, W, TAU, S, P, H_out, W_out, n_events, n_real_events=et.shape[0])
    assert cq.maps.a.shape == (OC * H_out * W_out, n_events)
    assert bool(jnp.all(jnp.isfinite(cq.maps.a))) and bool(jnp.all(jnp.isfinite(cq.maps.b)))
    assert bool(jnp.any(cq.maps.b != 0.0)), "應該至少有一些合法 tap 寫進權重"
    result = run_layer_forward(cq.maps, v_th=1e9, chunk_size=n_events, max_steps=n_events,
                                n_real_events=cq.n_real_events)
    assert bool(jnp.all(jnp.isfinite(result.v_final)))


# ============================================================================
# 3b. 跨層梯度(atan surrogate 斜率手算)——原 test_conv_queue.py 的
#     conv->conv / conv->FC 手算梯度,改用壓縮版 + extract_output_events_compressed。
# ============================================================================

# 共用小場景:K=3,S=2,P=1,H_out=W_out=3;W1[0,0,ky,kx]=ky*3+kx+1(1..9)。
_K, _S, _P, _HW = 3, 2, 1, 3


def _w1():
    return jnp.arange(1, 10, dtype=jnp.float32).reshape(1, 1, 3, 3)


def _atan_slope(z, alpha=2.0):
    a = alpha / 2.0
    ax = math.pi * a * z
    return 1.0 / (1.0 + ax * ax)


def _conv1_fire_then_extract(W1):
    """conv1:單一事件 (1,1,c=0,t=1),v_th=8.5 -> 只有 flat id=0(b=9.0)fire,
    s_spike forward 精確 1.0,觸發時間 t=1。回傳 extract_output_events_compressed
    的結果。"""
    et = jnp.array([1.0]); x = jnp.array([1]); y = jnp.array([1]); c = jnp.array([0])
    cq = build_conv_queue_compressed(et, x, y, c, W1, TAU, _S, _P, _HW, _HW, 1, n_real_events=et.shape[0])
    r = run_layer_forward(cq.maps, v_th=8.5, chunk_size=1, max_steps=1,
                           n_real_events=cq.n_real_events)
    return extract_output_events_compressed(r.spike_mask, r.spike_event_idx, r.s_spike, et,
                                             cq.local_to_global_j, max_total_spikes=9)


def test_conv_to_conv_cross_layer_gradient_matches_hand_calc():
    """conv1 -> unravel -> conv2(帶 event_gain),loss = conv2 v_final[0]。
    手算:v1 = W1[0,0,2,2] = 9.0;slope = atan_slope(9.0-8.5);
    W2[0,0,1,1] = 14 -> dL/dW1[0,0,2,2] = 14 * slope。其餘位置精確 0。"""
    def fwd(W1):
        ev = _conv1_fire_then_extract(W1)
        x2, y2, c2 = unravel_conv_source(ev.event_source_idx, _HW, _HW)
        W2 = (10 + jnp.arange(9, dtype=jnp.float32)).reshape(1, 1, 3, 3)
        cq2 = build_conv_queue_compressed(ev.event_times, x2, y2, c2, W2, TAU, _S, _P,
                                           _HW, _HW, 1, event_gain=ev.event_gain,
                                           n_real_events=ev.n_real_events)
        r2 = run_layer_forward(cq2.maps, v_th=1e9, chunk_size=1, max_steps=1,
                                n_real_events=cq2.n_real_events)
        return r2.v_final[0]

    g = jax.grad(fwd)(_w1())
    expected = 14.0 * _atan_slope(9.0 - 8.5)
    assert _tol_close(g[0, 0, 2, 2], expected, tol=1e-3), (float(g[0, 0, 2, 2]), expected)
    mask = jnp.ones((1, 1, 3, 3), dtype=bool).at[0, 0, 2, 2].set(False)
    assert _tol_close(jnp.where(mask, g, 0.0), 0.0, tol=TOL), g


def test_conv_to_fc_cross_layer_gradient_matches_hand_calc():
    """conv1 fire 出來的扁平 id 直接餵 build_fc_queue(不 unravel),loss = FC
    v_final[0]。W_fc[:,0] = [3.0,-1.0] -> dL/dW1[0,0,2,2] = 3.0 * slope。"""
    def fwd(W1):
        ev = _conv1_fire_then_extract(W1)
        W_fc = jnp.zeros((2, 9), dtype=jnp.float32).at[:, 0].set(jnp.array([3.0, -1.0]))
        maps_fc = build_fc_queue(ev.event_times, ev.event_source_idx, W_fc, TAU,
                                  event_gain=ev.event_gain, n_real_events=ev.n_real_events).maps
        r_fc = run_layer_forward(maps_fc, v_th=1e9, chunk_size=maps_fc.a.shape[1],
                                  max_steps=maps_fc.a.shape[1], n_real_events=ev.n_real_events)
        return r_fc.v_final[0]

    g = jax.grad(fwd)(_w1())
    expected = 3.0 * _atan_slope(9.0 - 8.5)
    assert _tol_close(g[0, 0, 2, 2], expected, tol=1e-3), (float(g[0, 0, 2, 2]), expected)
    mask = jnp.ones((1, 1, 3, 3), dtype=bool).at[0, 0, 2, 2].set(False)
    assert _tol_close(jnp.where(mask, g, 0.0), 0.0, tol=TOL), g


def test_without_event_gain_cross_layer_gradient_is_exactly_zero():
    """conv1 -> conv2 但**不**把 event_gain 傳給 conv2:計算圖裡沒有這條邊,
    jax.grad 對 W1 精確是 0(不是算錯,是根本沒有梯度路徑)。"""
    def fwd_no_gain(W1):
        ev = _conv1_fire_then_extract(W1)
        x2, y2, c2 = unravel_conv_source(ev.event_source_idx, _HW, _HW)
        W2 = (10 + jnp.arange(9, dtype=jnp.float32)).reshape(1, 1, 3, 3)
        cq2 = build_conv_queue_compressed(ev.event_times, x2, y2, c2, W2, TAU, _S, _P,
                                           _HW, _HW, 1, n_real_events=ev.n_real_events)
        r2 = run_layer_forward(cq2.maps, v_th=1e9, chunk_size=1, max_steps=1,
                                n_real_events=cq2.n_real_events)
        return r2.v_final[0]

    g = jax.grad(fwd_no_gain)(_w1())
    assert _tol_close(g, 0.0, tol=TOL), g


TESTS = [
    test_unravel_conv_source_roundtrip,
    test_reference_anchored_n1_one_to_one,
    test_reference_anchored_n3_multiple_candidates_per_axis,
    test_compressed_matches_ref_multi_output_channels,
    test_compressed_matches_ref_multi_input_channels,
    test_compressed_matches_ref_n1_stride1_pad0,
    test_compressed_matches_ref_n3_k5_stride2_pad2,
    test_compressed_matches_ref_random_various_geometry,
    test_compressed_realistic_scale_smoke,
    test_conv_to_conv_cross_layer_gradient_matches_hand_calc,
    test_conv_to_fc_cross_layer_gradient_matches_hand_calc,
    test_without_event_gain_cross_layer_gradient_is_exactly_zero,
]


if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n全部 {len(TESTS)} 項測試通過")
