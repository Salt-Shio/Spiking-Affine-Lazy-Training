"""整數版掃描的驗證:process_event(一筆事件的單步更新:衰減捨入、溢位繞回/飽和、
fire/reset)跟 run_layer/run_layer_traced(整條佇列)。
直接給 a_int、is_identity、q_int,不經過佇列建構,手算鏈式遞迴對答案。
"""

import jax.numpy as jnp

from salt_core.quant.scan import process_event, run_layer, run_layer_traced


def test_run_layer_two_events_chained_matches_hand_computation():
    """單一神經元,兩筆事件鏈式遞迴。i_V=8,f_V=2,f_a=4。

    事件 0:is_identity=True(跳過衰減),q=5,f_V=2 => 0+5*4=20,無溢位。
    事件 1:is_identity=False,a_int=12(對應 tau=4 時 Δt=1 的 0.75 查表值),
    v0=20:12*20=240,240/16=15(整除),decayed=15;q=3 => 15+3*4=27。"""
    a_int = jnp.array([[999, 12]])       # 事件0的 a_int 是垃圾值,is_identity=True 時不會被用到
    is_identity = jnp.array([[True, False]])
    q_int = jnp.array([[5, 3]])
    v_th_int = jnp.array([100])

    result = run_layer(a_int, is_identity, q_int, v_th_int, f_a=4, f_V=2, i_V=8)

    assert int(result.v_final[0]) == 27
    assert not bool(result.spike_mask[0, 0]) and not bool(result.spike_mask[0, 1])
    assert not bool(result.overflowed[0, 0]) and not bool(result.overflowed[0, 1])


def test_run_layer_applies_catchup_decay_after_last_real_tap():
    """每一欄都照實套用,conv 的 catch-up 欄(真 tap 之後的純衰減)不能被當
    pad 跳過。tau=16、f_a=4、f_V=4,單顆神經元:

    - 第 0 欄:真 tap,q=5,v0=0 => 5*2^4=80。
    - 第 1 欄:catch-up,Δt=3,a=(15/16)^3≈0.824 => 查表碼 13;q=0。
      13*80/16=65(整除)。
    - 第 2 欄:identity,維持 65。

    catch-up 被跳過的話 v_final 會停在 80。"""
    a_int = jnp.array([[14, 13, 0]])     # 第 0 欄 v0=0,a_int 不影響結果
    is_identity = jnp.array([[False, False, True]])
    q_int = jnp.array([[5, 0, 0]])
    v_th_int = jnp.array([10 ** 5])

    result = run_layer(a_int, is_identity, q_int, v_th_int, f_a=4, f_V=4, i_V=16)

    assert int(result.v_final[0]) == 65, "catch-up 欄的衰減要生效,不是停在 80"


def test_run_layer_scan_length_is_queue_length():
    """掃描長度等於佇列欄數,每欄一步,spike_event_idx 就是欄位索引。"""
    a_int = jnp.zeros((2, 3), dtype=jnp.int32)
    is_identity = jnp.ones((2, 3), dtype=bool)
    q_int = jnp.zeros((2, 3), dtype=jnp.int32)

    result = run_layer(a_int, is_identity, q_int, jnp.array(100), f_a=4, f_V=0, i_V=8)

    assert result.spike_mask.shape == (2, 3)
    assert result.overflowed.shape == (2, 3)
    assert list(result.spike_event_idx[1]) == [0, 1, 2]


def test_run_layer_per_neuron_v_th_fire_and_reset():
    """兩顆神經元收到完全一樣的事件,各自的 v_th_int 不同——門檻低的那顆該
    fire 並硬重置成 0,門檻高的那顆不該 fire。"""
    a_int = jnp.array([[999], [999]])
    is_identity = jnp.array([[True], [True]])
    q_int = jnp.array([[20], [20]])   # f_V=0 時直接貢獻 20
    v_th_int = jnp.array([15, 25])

    result = run_layer(a_int, is_identity, q_int, v_th_int, f_a=4, f_V=0, i_V=8)

    assert bool(result.spike_mask[0, 0]), "neuron0: 20>=15,應該 fire"
    assert int(result.v_final[0]) == 0, "fire 後硬重置成 0"
    assert not bool(result.spike_mask[1, 0]), "neuron1: 20<25,不該 fire"
    assert int(result.v_final[1]) == 20


def test_run_layer_overflow_flag_set_and_propagates_to_v_final():
    """i_V=4,f_V=0(總位元 4,範圍 [-8,7]),單一事件 q=9:0+9=9,超出範圍一格,
    繞回去是 9-16=-7,overflowed 應該是 True,v_final 反映繞回去之後的值。"""
    a_int = jnp.array([[999]])
    is_identity = jnp.array([[True]])
    q_int = jnp.array([[9]])
    v_th_int = jnp.array([100])

    result = run_layer(a_int, is_identity, q_int, v_th_int, f_a=4, f_V=0, i_V=4)

    assert bool(result.overflowed[0, 0])
    assert int(result.v_final[0]) == -7
    assert not bool(result.spike_mask[0, 0])


def test_run_layer_unfitted_extremes_include_value_before_reset_and_wrap():
    """v_unfitted_min/max 記的是寫回之前的真實值,fire 歸零、溢位繞回都看得到。
    f_V=0、i_V=6(範圍 [-32,31]),全部 identity(不衰減)。

    - 神經元 0,v_th=15:q = 20 => 真實值 20,fire 歸零;q = -3 => -3。
      寫回之後的值是 [0, -3],最大只看得到 0;真實值的最大是 20、最小是 -3。
    - 神經元 1,不會 fire(v_th=100):q = -30、-5 => -30、-35,-35 超出下限,
      繞回成 -35+64=29。真實值的最小是 -35,最大是初始值 0。"""
    a_int = jnp.full((2, 2), 999)
    is_identity = jnp.ones((2, 2), dtype=bool)
    q_int = jnp.array([[20, -3], [-30, -5]])
    v_th_int = jnp.array([15, 100])

    result = run_layer(a_int, is_identity, q_int, v_th_int, f_a=4, f_V=0, i_V=6)

    assert [int(v) for v in result.v_final] == [-3, 29]
    assert bool(result.overflowed[1, 1])
    assert [int(v) for v in result.v_unfitted_max] == [20, 0]
    assert [int(v) for v in result.v_unfitted_min] == [-3, -35]


def test_process_event_reports_value_before_wrap():
    """跟 test_run_layer_overflow_flag_set_and_propagates_to_v_final 同一組數字:
    真實值 9 繞回成 -7,v_unfitted 是 9。"""
    result = process_event(v0_int=0, a_int=999, is_identity=True, q_int=9,
                           v_th_int=100, f_a=4, f_V=0, i_V=4)
    assert int(result.v_final) == -7
    assert int(result.v_unfitted) == 9


def test_run_layer_traced_matches_untraced_and_last_v_step_is_v_final():
    a_int = jnp.array([[999, 12]])
    is_identity = jnp.array([[True, False]])
    q_int = jnp.array([[5, 3]])
    v_th_int = jnp.array([100])

    untraced = run_layer(a_int, is_identity, q_int, v_th_int, f_a=4, f_V=2, i_V=8)
    traced, v_steps = run_layer_traced(a_int, is_identity, q_int, v_th_int,
                                                    f_a=4, f_V=2, i_V=8)

    assert int(traced.v_final[0]) == int(untraced.v_final[0])
    assert bool(traced.spike_mask[0, 0]) == bool(untraced.spike_mask[0, 0])
    assert v_steps.shape == (1, 2)
    assert int(v_steps[0, 0]) == 20, "第一步套用完的膜電位"
    assert int(v_steps[0, -1]) == int(traced.v_final[0]), "最後一欄要等於 v_final"


def test_process_event_matches_hand_computation_no_overflow_no_spike():
    """i_V=8,f_V=2(總位元 10,範圍 [-512,511]),f_a=4。a_int=12(對應
    build_decay_table_int(4,4.0) 的第一項,即 Δt=1 時 0.75 的整數碼),
    v0_int=40(=10.0 in 2^-2 格),q_int=5。

    decayed=12*40/16=480/16=30(整除,沒有捨入)。繞回之前的值
    30+5*4=50,在 10 位元範圍內不溢位。
    v_th_int=100 > 50,不 fire。"""
    result = process_event(v0_int=40, a_int=12, is_identity=False, q_int=5,
                                v_th_int=100, f_a=4, f_V=2, i_V=8)
    assert int(result.v_final) == 50
    assert not bool(result.is_spiked)
    assert not bool(result.overflowed)


def test_process_event_spike_resets_to_zero():
    """跟上面同一組數字,只把 v_th_int 降到 40(<=50),應該 fire 並硬重置成 0。"""
    result = process_event(v0_int=40, a_int=12, is_identity=False, q_int=5,
                                v_th_int=40, f_a=4, f_V=2, i_V=8)
    assert bool(result.is_spiked)
    assert int(result.v_final) == 0


def test_process_event_identity_skips_decay():
    """is_identity=True(Δt=0)時完全跳過衰減,a_int 的值(這裡故意給一個不合理
    的數字 999)不該影響結果:decayed 應該就是 v0_int=17 本身。
    q_int=3,f_V=2 => 貢獻 3*4=12,17+12=29。"""
    result = process_event(v0_int=17, a_int=999, is_identity=True, q_int=3,
                                v_th_int=100, f_a=4, f_V=2, i_V=8)
    assert int(result.v_final) == 29
    assert not bool(result.overflowed)


def test_process_event_round_vs_truncate_differ():
    """v0_int=7, a_int=1, f_a=2:7/4=1.75,round 進到 2、truncate(算術右移,
    floor)捨到 1。q_int=0 隔離掉權重貢獻,只看捨入差異。"""
    rounded = process_event(v0_int=7, a_int=1, is_identity=False, q_int=0,
                                v_th_int=100, f_a=2, f_V=0, i_V=8, round_mode="round")
    truncated = process_event(v0_int=7, a_int=1, is_identity=False, q_int=0,
                                   v_th_int=100, f_a=2, f_V=0, i_V=8, round_mode="truncate")
    assert int(rounded.v_final) == 2
    assert int(truncated.v_final) == 1


def test_process_event_negative_tie_rounds_toward_positive_infinity():
    """兩補數捨入慣例:負數卡在正中間時 round 往正無窮、truncate 往負無窮。
    v0_int=-6, a_int=1, f_a=2:-6/4=-1.5,round 是 (-6+2)>>2=-1,
    truncate 是 -6>>2=-2。"""
    rounded = process_event(v0_int=-6, a_int=1, is_identity=False, q_int=0,
                                v_th_int=100, f_a=2, f_V=0, i_V=8, round_mode="round")
    truncated = process_event(v0_int=-6, a_int=1, is_identity=False, q_int=0,
                                   v_th_int=100, f_a=2, f_V=0, i_V=8, round_mode="truncate")
    assert int(rounded.v_final) == -1
    assert int(truncated.v_final) == -2


def test_process_event_overflow_wraps_and_can_mask_a_true_spike():
    """i_V=4,f_V=0(總位元 4,範圍 [-8,7])。is_identity=True 跳過衰減,
    v0_int=7(已經是這個寬度能存的最大值)+ q_int=1 => 真實值是 8,超出範圍
    一格,兩補數繞回去是 8-16=-8(手算:8=0b1000 當成 4 位元有號數,
    值是 8-16=-8)。

    v_th_int=5:如果比較的是繞回去之前的真實值 8,理應 fire(8>=5);但硬體
    暫存器只留得住繞回去之後的 -8,fire 判斷讀到的是 -8,不會 fire——這是
    選錯 i_V 會讓 fire 判斷跟著出錯的具體例子,不是純理論疑慮。"""
    result = process_event(v0_int=7, a_int=999, is_identity=True, q_int=1,
                                v_th_int=5, f_a=4, f_V=0, i_V=4)
    assert bool(result.overflowed)
    assert int(result.v_final) == -8
    assert not bool(result.is_spiked), "繞回去之後的 -8 讀不到 fire,即使真實值 8 本來會 fire"


def test_process_event_saturate_clamps_and_keeps_true_spike():
    """跟上一個測試同一組數字,改成飽和:真實值 8 夾到 7,不會翻號,
    7>=5 照樣 fire;overflowed 一樣回報 True。"""
    result = process_event(v0_int=7, a_int=999, is_identity=True, q_int=1,
                                v_th_int=5, f_a=4, f_V=0, i_V=4, overflow_mode="saturate")
    assert bool(result.overflowed)
    assert bool(result.is_spiked)
    assert int(result.v_final) == 0, "fire 後硬重置成 0"


def test_process_event_no_threshold_never_fires_or_resets():
    """v_th_int=None 代表這層不 fire:值再大也不 fire、不重置,一路累積。"""
    result = process_event(v0_int=100, a_int=999, is_identity=True, q_int=50,
                                v_th_int=None, f_a=4, f_V=0, i_V=16)
    assert not bool(result.is_spiked)
    assert int(result.v_final) == 150


def test_process_event_accepts_f_a_and_register_width_each_within_own_limit():
    """f_a 跟 i_V+f_V 各自在自己的上限內就接受,三者加總不受限:
    f_a=15、i_V+f_V=30,加總 45。"""
    result = process_event(v0_int=100, a_int=1, is_identity=False, q_int=1,
                                v_th_int=10 ** 8, f_a=15, f_V=14, i_V=16)
    assert not bool(result.overflowed)
