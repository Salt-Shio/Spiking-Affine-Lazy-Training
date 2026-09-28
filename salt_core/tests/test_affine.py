"""仿射合成跟一個 chunk 的 fire 偵測(float/affine.py)。

1. combine + associative_scan 合成的結果,等於逐筆套用 h = V*a + w。
2. process_chunk 的 fire 偵測跟 reset,對手算的例子。
事件的 (N, w) 都直接給,不經過佇列建構。
"""

import jax.numpy as jnp

from salt_core.float.affine import AffineMap, combine, create_affine_maps, process_chunk

TOL = 1e-6


def assert_allclose(actual, expected, msg):
    actual = float(actual)
    expected = float(expected)
    assert abs(actual - expected) < TOL, f"{msg}: got {actual}, expected {expected}"


def test_combine_matches_sequential_composition():
    """associative_scan 合成的前綴仿射映射,跟逐筆序列套用 h = V*a + w 要完全一致。"""
    tau = 16.0
    v0 = 0.37
    n_ms_list = [0, 2, 1, 5, 0, 3]
    w_list = [0.3, -0.1, 0.5, 0.2, 0.05, -0.2]

    maps = create_affine_maps(jnp.array(n_ms_list, dtype=jnp.float32),
                               jnp.array(w_list, dtype=jnp.float32), tau)

    result = process_chunk(v0, maps, v_th=1e9)  # v_th 設極大,強迫不觸發 spike/reset 分支
    v_sequence = result.v_sequence

    # 序列版 oracle:一筆一筆套用 h = V_old * a_i + w_i
    v_ref = v0
    expected = []
    for n_ms, w in zip(n_ms_list, w_list):
        a = (1.0 - 1.0 / tau) ** n_ms
        v_ref = v_ref * a + w
        expected.append(v_ref)

    for i, exp in enumerate(expected):
        assert_allclose(v_sequence[i], exp, f"v_sequence[{i}]")

    assert not bool(result.is_spiked), "v_th 設極大時不應該 spike"
    assert_allclose(result.v_final, expected[-1], "v_final(無 spike 應等於序列最後一個值)")


def test_chunk_no_fire():
    """整個 chunk 都沒有跨過門檻:is_spiked=False,spike_idx=chunk 長度(sentinel),
    v_final=最後一個值。"""
    tau = 4.0
    v_th = 1.0
    n_ms_list = [0, 3, 2]
    w_list = [0.2, 0.1, 0.3]

    maps = create_affine_maps(jnp.array(n_ms_list, dtype=jnp.float32),
                               jnp.array(w_list, dtype=jnp.float32), tau)
    result = process_chunk(0.0, maps, v_th)

    assert not bool(result.is_spiked)
    assert int(result.spike_idx) == len(n_ms_list)
    assert_allclose(result.v_final, result.v_sequence[-1], "v_final 應等於 v_sequence 最後一項")


def test_chunk_fire_at_first_event():
    """第一筆事件自己就跨過門檻:spike_idx=0,v_final 立刻 reset 成 0。"""
    tau = 4.0
    v_th = 1.0
    n_ms_list = [0, 1]
    w_list = [1.5, 0.4]  # 第一筆權重就直接超過門檻

    maps = create_affine_maps(jnp.array(n_ms_list, dtype=jnp.float32),
                               jnp.array(w_list, dtype=jnp.float32), tau)
    result = process_chunk(0.0, maps, v_th)

    assert bool(result.is_spiked)
    assert int(result.spike_idx) == 0
    assert_allclose(result.v_final, 0.0, "spike 後應硬重置成 0")


def test_multi_chunk_worked_example():
    """手算例子(同 docs/math/不套閘與soft-reset梯度推導.md):tau=4,v_th=1.0,事件 (m=0,w=0.6)(m=1,w=0.6)(m=5,w=0.9)。

    手算軌跡:
      event1(m=0,N=0): h=0*1+0.6=0.6,      不 spike, V=0.6
      event2(m=1,N=1): h=0.6*0.75+0.6=1.05, spike,     V<-0
      event3(m=5,N=4): h=0*0.75^4+0.9=0.9,  不 spike, V=0.9
    預期:只在 event2(0-based idx1)spike 一次,最終 V=0.9。

    N_i 只跟「跟上一筆事件的真實時間差」有關,跟 reset 有沒有發生無關,所以整條
    事件的仿射映射可以一次算好;chunk 只是決定「這段合成結果能相信到哪裡」,
    reset 之後從下一筆事件重新起跑,套用同一份 maps 裡對應的係數,不用重算。
    """
    tau = 4.0
    v_th = 1.0
    n_ms_list = [0, 1, 4]
    w_list = [0.6, 0.6, 0.9]

    maps = create_affine_maps(jnp.array(n_ms_list, dtype=jnp.float32),
                               jnp.array(w_list, dtype=jnp.float32), tau)

    # 第一個 chunk:把全部 3 筆事件當成一個 chunk,假設全程不 spike 去猜。
    first = process_chunk(0.0, maps, v_th)
    assert bool(first.is_spiked)
    assert int(first.spike_idx) == 1
    assert_allclose(first.v_sequence[0], 0.6, "event1 後(未 spike)的 V")
    assert_allclose(first.v_sequence[1], 1.05, "event2 觸發 spike 前的 h")
    assert_allclose(first.v_final, 0.0, "event2 spike 後硬重置")

    # 第二個 chunk:從 event3 開始,v0 用第一個 chunk reset 後的值,
    # 直接切用同一份 maps 裡 event3 自己的係數(index 2),不重算。
    remaining_maps = AffineMap(a=maps.a[2:], b=maps.b[2:])
    second = process_chunk(first.v_final, remaining_maps, v_th)
    assert not bool(second.is_spiked)
    assert_allclose(second.v_final, 0.9, "event3 後的最終 V")


def test_combine_associativity():
    """combine 本身要滿足結合律,associative_scan 的正確性建立在這個性質上。"""
    tau = 8.0
    m1 = create_affine_maps(jnp.array(1.0), jnp.array(0.4), tau)
    m2 = create_affine_maps(jnp.array(2.0), jnp.array(-0.3), tau)
    m3 = create_affine_maps(jnp.array(0.0), jnp.array(0.7), tau)

    left_assoc = combine(combine(m1, m2), m3)
    right_assoc = combine(m1, combine(m2, m3))

    assert_allclose(left_assoc.a, right_assoc.a, "combine 結合律: a")
    assert_allclose(left_assoc.b, right_assoc.b, "combine 結合律: b")
