"""單狀態仿射合成/chunk spike 偵測的驗證。

驗證方式:
1. combine + associative_scan 合成的結果,要跟逐筆序列套用 h = V*a + w 完全一致
   (驗證平行掃描沒有算錯,不是換一種方式重新定義正確性)。
2. process_chunk 的 spike 偵測/reset,對照 docs/math/單狀態仿射平行掃描推導.md
   跟 docs/TODO.md 裡手算過的具體數字例子逐項核對。
不含佇列建構,事件的 (N, w) 都是測試裡直接給定的假資料。
"""

import jax.numpy as jnp

from salt_core.core import (AffineMap, combine, create_affine_maps, process_chunk,
                            process_event_int)

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
    """docs/TODO.md 手算過的例子:tau=4,v_th=1.0,事件 (m=0,w=0.6)(m=1,w=0.6)(m=5,w=0.9)。

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


def test_process_chunk_round_step_rounds_the_decay_term_not_the_sum():
    """round_step(docs/math/膜電位量化推導.md 的 r(·):$\\tilde V_k=r(a_k\\tilde
    V_{k-1})+q_k$)只對「衰減後、還沒加這筆事件貢獻」的值捨入,`b`(這筆事件的
    貢獻)是之後才加上去的,不會被捨入影響——用 `v0=1.28`(非零,才有「衰減」
    可言)、`b=0.0`(隔離掉貢獻項,只看衰減項捨入)驗證:`a*v0=0.75*1.28=0.96`
    捨到最近的 0.1 格是 1.0。"""
    tau = 4.0
    v_th = 1e9  # 設極大,只看 v_sequence 有沒有被捨入,不管 spike
    maps = create_affine_maps(jnp.array([1.0]), jnp.array([0.0]), tau)  # n_ms=1 => a=0.75,b=0

    result = process_chunk(1.28, maps, v_th, round_step=0.1)

    assert_allclose(result.v_sequence[0], 1.0, "0.75*1.28=0.96 應捨入到最近的 0.1 格 = 1.0")


def test_process_chunk_round_step_can_trigger_spike_that_raw_value_would_not():
    """捨入要在門檻比較之前生效:捨入後才跨過門檻,必須真的 spike,
    不能只影響回報出來的 v_sequence 數值。跟上一個測試同一組數字
    (v0=1.28, a=0.75, b=0):原始衰減值 0.96 < v_th=1.0 不該 spike,
    捨入成 1.0 之後應該要 spike。"""
    tau = 4.0
    v_th = 1.0
    maps = create_affine_maps(jnp.array([1.0]), jnp.array([0.0]), tau)

    without_round = process_chunk(1.28, maps, v_th)
    assert not bool(without_round.is_spiked), "沒有 round_step 時,0.96 不該 spike"

    with_round = process_chunk(1.28, maps, v_th, round_step=0.1)  # 捨入後變成 1.0,跨過門檻
    assert bool(with_round.is_spiked), "round_step=0.1 把 0.96 捨入成 1.0,應該要 spike"
    assert_allclose(with_round.v_final, 0.0, "spike 後硬重置成 0")


def test_process_chunk_round_mode_truncate_rounds_toward_zero():
    """round_mode="truncate" 跟預設的 "round" 要算出不同結果——這是膜電位
    量化捨入規則 r(·) 兩種選項都要能選的那個開關,不是裝飾參數。同一組
    v0=1.28, a=0.75, b=0:四捨五入 0.96->1.0,向零捨去 0.96->0.9。"""
    tau = 4.0
    v_th = 1e9
    maps = create_affine_maps(jnp.array([1.0]), jnp.array([0.0]), tau)

    rounded = process_chunk(1.28, maps, v_th, round_step=0.1, round_mode="round")
    truncated = process_chunk(1.28, maps, v_th, round_step=0.1, round_mode="truncate")

    assert_allclose(rounded.v_sequence[0], 1.0, "0.96 四捨五入到 0.1 格 = 1.0")
    assert_allclose(truncated.v_sequence[0], 0.9, "0.96 向零捨去到 0.1 格 = 0.9")


def test_process_chunk_round_before_add_matters_for_truncate_across_zero():
    """迴歸測試:code review 抓到的問題——`truncate` 模式下「先捨衰減值、
    再加貢獻」(文件公式)跟「先加總、再捨一次」不是同一件事,一旦這筆事件
    的貢獻讓數值跨過 0 就會分岔。反例:衰減後的值 -0.6、這筆事件貢獻 +1、
    格距 1。

    文件公式(這個函式現在的做法):trunc(-0.6)=0,0+1=1.0。
    先加總再捨(修正前的做法):-0.6+1=0.4,trunc(0.4)=0.0——答案不一樣。

    round 模式沒有這個問題(round 建立在 floor 上,floor 對整數平移不分正負
    號都成立),這裡只測 truncate,對照上面已經驗證過 round 沒事。
    """
    tau = 4.0
    v_th = 1e9
    # v0=-0.6, a=1.0(n_ms=0 =>不衰減,直接讓「衰減後的值」就是 v0 本身,
    # 方便對上反例的手算數字),b=1.0(這筆事件的整數貢獻,格距=1 下就是 q_k=1)。
    maps = create_affine_maps(jnp.array([0.0]), jnp.array([1.0]), tau)

    result = process_chunk(-0.6, maps, v_th, round_step=1.0, round_mode="truncate")

    assert_allclose(result.v_sequence[0], 1.0,
                    "文件順序(先捨衰減值再加貢獻)trunc(-0.6)+1=1.0;"
                    "若先加總再捨會錯算成 trunc(-0.6+1)=trunc(0.4)=0.0")


def test_process_chunk_rejects_invalid_round_mode():
    tau = 4.0
    maps = create_affine_maps(jnp.array([0.0]), jnp.array([0.5]), tau)
    try:
        process_chunk(0.0, maps, 1.0, round_step=0.1, round_mode="ceil")
        raised = False
    except ValueError:
        raised = True
    assert raised, "無效的 round_mode 應該要 raise ValueError"


def test_process_event_int_matches_hand_computation_no_overflow_no_spike():
    """i_V=8,f_V=2(總位元 10,範圍 [-512,511]),f_a=4。a_int=12(對應
    build_decay_table_int(4,4.0) 的第一項,即 Δt=1 時 0.75 的整數碼),
    v0_int=40(=10.0 in 2^-2 格),q_int=5。

    decayed=round_shift(12*40,4,"round")=round_shift(480,4)=480/16=30
    (整除,沒有捨入)。raw=30+5*4=50,在 10 位元範圍內不溢位。
    v_th_int=100 > 50,不 fire。"""
    result = process_event_int(v0_int=40, a_int=12, is_identity=False, q_int=5,
                                v_th_int=100, f_a=4, f_V=2, i_V=8)
    assert int(result.v_final) == 50
    assert not bool(result.is_spiked)
    assert not bool(result.overflowed)


def test_process_event_int_spike_resets_to_zero():
    """跟上面同一組數字,只把 v_th_int 降到 40(<=50),應該 fire 並硬重置成 0。"""
    result = process_event_int(v0_int=40, a_int=12, is_identity=False, q_int=5,
                                v_th_int=40, f_a=4, f_V=2, i_V=8)
    assert bool(result.is_spiked)
    assert int(result.v_final) == 0


def test_process_event_int_identity_skips_decay():
    """is_identity=True(Δt=0)時完全跳過衰減,a_int 的值(這裡故意給一個不合理
    的數字 999)不該影響結果:decayed 應該就是 v0_int=17 本身。
    q_int=3,f_V=2 => 貢獻 3*4=12,raw=17+12=29。"""
    result = process_event_int(v0_int=17, a_int=999, is_identity=True, q_int=3,
                                v_th_int=100, f_a=4, f_V=2, i_V=8)
    assert int(result.v_final) == 29
    assert not bool(result.overflowed)


def test_process_event_int_round_vs_truncate_differ():
    """v0_int=7, a_int=1, f_a=2(shift_bits=2):7/4=1.75,round 進到 2、
    truncate 向零捨去到 1——對照 quantize.round_shift 自己的手算測試。
    q_int=0 隔離掉權重貢獻,只看捨入差異。"""
    rounded = process_event_int(v0_int=7, a_int=1, is_identity=False, q_int=0,
                                v_th_int=100, f_a=2, f_V=0, i_V=8, round_mode="round")
    truncated = process_event_int(v0_int=7, a_int=1, is_identity=False, q_int=0,
                                   v_th_int=100, f_a=2, f_V=0, i_V=8, round_mode="truncate")
    assert int(rounded.v_final) == 2
    assert int(truncated.v_final) == 1


def test_process_event_int_overflow_wraps_and_can_mask_a_true_spike():
    """i_V=4,f_V=0(總位元 4,範圍 [-8,7])。is_identity=True 跳過衰減,
    v0_int=7(已經是這個寬度能存的最大值)+ q_int=1 => 真實值是 8,超出範圍
    一格,兩補數繞回去是 8-16=-8(手算:8=0b1000 當成 4 位元有號數,
    值是 8-16=-8)。

    v_th_int=5:如果比較的是繞回去之前的真實值 8,理應 fire(8>=5);但硬體
    暫存器只留得住繞回去之後的 -8,fire 判斷讀到的是 -8,不會 fire——這是
    選錯 i_V 會讓 fire 判斷跟著出錯的具體例子,不是純理論疑慮。"""
    result = process_event_int(v0_int=7, a_int=999, is_identity=True, q_int=1,
                                v_th_int=5, f_a=4, f_V=0, i_V=4)
    assert bool(result.overflowed)
    assert int(result.v_final) == -8
    assert not bool(result.is_spiked), "繞回去之後的 -8 讀不到 fire,即使真實值 8 本來會 fire"


def test_process_event_int_rejects_f_a_too_wide_for_wide_mul_shift():
    """f_a>15 時,wide_mul_shift 內部低位乘積(2*f_a 位元)會先溢位 int32,
    要直接拒絕,不能悄悄算錯(這個專案不能開 jax_enable_x64,見函式文件)。"""
    try:
        process_event_int(v0_int=0, a_int=0, is_identity=True, q_int=0,
                          v_th_int=1, f_a=16, f_V=2, i_V=8)
        raised = False
    except ValueError:
        raised = True
    assert raised, "f_a 超過 15 應該要 raise ValueError"


def test_process_event_int_rejects_register_width_too_wide_for_int32():
    """i_V+f_V>30 時,暫存器總寬度容不進 int32 的安全範圍,要直接拒絕。
    i_V=20, f_V=15 加起來 35。"""
    try:
        process_event_int(v0_int=0, a_int=0, is_identity=True, q_int=0,
                          v_th_int=1, f_a=4, f_V=15, i_V=20)
        raised = False
    except ValueError:
        raised = True
    assert raised, "i_V+f_V 超過 30 應該要 raise ValueError"


def test_process_event_int_decoupled_bit_widths_allow_combo_that_old_guard_would_reject():
    """f_a=10, f_V=14, i_V=6(三者相加 30,剛好卡在原本「綁在一起」的舊上限
    邊界,但這組合實務上偏緊——這裡驗證拆開乘法之後,f_a 可以獨立开到 15、
    i_V+f_V 可以獨立开到接近 30,不必綁死三者總和,例如 f_a=15,f_V=14,
    i_V=16(總和 45,舊的「總和<=30」一定會拒絕,新的兩個獨立檢查都過)。"""
    result = process_event_int(v0_int=100, a_int=1, is_identity=False, q_int=1,
                                v_th_int=10 ** 8, f_a=15, f_V=14, i_V=16)
    assert not bool(result.overflowed)


def test_process_event_int_rejects_invalid_round_mode():
    try:
        process_event_int(v0_int=0, a_int=0, is_identity=True, q_int=0,
                          v_th_int=1, f_a=4, f_V=2, i_V=8, round_mode="ceil")
        raised = False
    except ValueError:
        raised = True
    assert raised, "無效的 round_mode 應該要 raise ValueError"


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


TESTS = [
    test_combine_matches_sequential_composition,
    test_chunk_no_fire,
    test_chunk_fire_at_first_event,
    test_multi_chunk_worked_example,
    test_process_chunk_round_step_rounds_the_decay_term_not_the_sum,
    test_process_chunk_round_step_can_trigger_spike_that_raw_value_would_not,
    test_process_chunk_round_mode_truncate_rounds_toward_zero,
    test_process_chunk_round_before_add_matters_for_truncate_across_zero,
    test_process_chunk_rejects_invalid_round_mode,
    test_process_event_int_matches_hand_computation_no_overflow_no_spike,
    test_process_event_int_spike_resets_to_zero,
    test_process_event_int_identity_skips_decay,
    test_process_event_int_round_vs_truncate_differ,
    test_process_event_int_overflow_wraps_and_can_mask_a_true_spike,
    test_process_event_int_rejects_f_a_too_wide_for_wide_mul_shift,
    test_process_event_int_rejects_register_width_too_wide_for_int32,
    test_process_event_int_decoupled_bit_widths_allow_combo_that_old_guard_would_reject,
    test_process_event_int_rejects_invalid_round_mode,
    test_combine_associativity,
]


if __name__ == "__main__":
    for test in TESTS:
        test()
        print(f"PASS: {test.__name__}")
    print(f"\n全部 {len(TESTS)} 項測試通過")
