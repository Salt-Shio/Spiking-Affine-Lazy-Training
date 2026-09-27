"""`run_layer_forward_int`/`run_layer_forward_int_traced`(整數版掃描)的驗證。
直接給 `a_int`/`is_identity`/`q_int`,不經過佇列建構,手算鏈式遞迴對答案。
"""

import jax.numpy as jnp

from salt_core.chunk_scan import run_layer_forward_int, run_layer_forward_int_traced


def test_run_layer_forward_int_two_events_chained_matches_hand_computation():
    """單一神經元,兩筆事件鏈式遞迴。i_V=8,f_V=2,f_a=4。

    事件 0:is_identity=True(跳過衰減),q=5,f_V=2 => 0+5*4=20,無溢位。
    事件 1:is_identity=False,a_int=12(對應 tau=4 時 Δt=1 的 0.75 查表值),
    v0=20:12*20=240,240/16=15(整除),decayed=15;q=3 => 15+3*4=27。"""
    a_int = jnp.array([[999, 12]])       # 事件0的 a_int 是垃圾值,is_identity=True 時不會被用到
    is_identity = jnp.array([[True, False]])
    q_int = jnp.array([[5, 3]])
    v_th_int = jnp.array([100])

    result = run_layer_forward_int(a_int, is_identity, q_int, v_th_int, f_a=4, f_V=2, i_V=8)

    assert int(result.v_final[0]) == 27
    assert not bool(result.spike_mask[0, 0]) and not bool(result.spike_mask[0, 1])
    assert not bool(result.overflowed[0, 0]) and not bool(result.overflowed[0, 1])


def test_run_layer_forward_int_applies_catchup_decay_after_last_real_tap():
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

    result = run_layer_forward_int(a_int, is_identity, q_int, v_th_int, f_a=4, f_V=4, i_V=16)

    assert int(result.v_final[0]) == 65, "catch-up 欄的衰減要生效,不是停在 80"


def test_run_layer_forward_int_scan_length_is_queue_length():
    """掃描長度等於佇列欄數,每欄一步,`spike_event_idx` 就是欄位索引。"""
    a_int = jnp.zeros((2, 3), dtype=jnp.int32)
    is_identity = jnp.ones((2, 3), dtype=bool)
    q_int = jnp.zeros((2, 3), dtype=jnp.int32)

    result = run_layer_forward_int(a_int, is_identity, q_int, jnp.array(100), f_a=4, f_V=0, i_V=8)

    assert result.spike_mask.shape == (2, 3)
    assert result.overflowed.shape == (2, 3)
    assert list(result.spike_event_idx[1]) == [0, 1, 2]


def test_run_layer_forward_int_per_neuron_v_th_fire_and_reset():
    """兩顆神經元收到完全一樣的事件,各自的 v_th_int 不同——門檻低的那顆該
    fire 並硬重置成 0,門檻高的那顆不該 fire。"""
    a_int = jnp.array([[999], [999]])
    is_identity = jnp.array([[True], [True]])
    q_int = jnp.array([[20], [20]])   # f_V=0 時直接貢獻 20
    v_th_int = jnp.array([15, 25])

    result = run_layer_forward_int(a_int, is_identity, q_int, v_th_int, f_a=4, f_V=0, i_V=8)

    assert bool(result.spike_mask[0, 0]), "neuron0: 20>=15,應該 fire"
    assert int(result.v_final[0]) == 0, "fire 後硬重置成 0"
    assert not bool(result.spike_mask[1, 0]), "neuron1: 20<25,不該 fire"
    assert int(result.v_final[1]) == 20


def test_run_layer_forward_int_overflow_flag_set_and_propagates_to_v_final():
    """i_V=4,f_V=0(總位元 4,範圍 [-8,7]),單一事件 q=9:0+9=9,超出範圍一格,
    繞回去是 9-16=-7,overflowed 應該是 True,v_final 反映繞回去之後的值。"""
    a_int = jnp.array([[999]])
    is_identity = jnp.array([[True]])
    q_int = jnp.array([[9]])
    v_th_int = jnp.array([100])

    result = run_layer_forward_int(a_int, is_identity, q_int, v_th_int, f_a=4, f_V=0, i_V=4)

    assert bool(result.overflowed[0, 0])
    assert int(result.v_final[0]) == -7
    assert not bool(result.spike_mask[0, 0])


def test_run_layer_forward_int_no_threshold_accumulates_without_firing():
    """v_th_int=None(不 fire 的層):兩筆 identity 事件 q=20 一路累積到 40,不 fire。"""
    a_int = jnp.array([[0, 0]])
    is_identity = jnp.array([[True, True]])
    q_int = jnp.array([[20, 20]])

    result = run_layer_forward_int(a_int, is_identity, q_int, None, f_a=4, f_V=0, i_V=8)

    assert int(result.v_final[0]) == 40
    assert not bool(result.spike_mask.any())


def test_run_layer_forward_int_overflow_mode_reaches_register():
    """overflow_mode 要傳到每一步:跟上面溢位同一組數字(4 位元,q=9),
    飽和時夾到 7,繞回時是 -7。"""
    args = (jnp.array([[0]]), jnp.array([[True]]), jnp.array([[9]]), jnp.array([100]))
    wrapped = run_layer_forward_int(*args, f_a=4, f_V=0, i_V=4, overflow_mode="wrap")
    saturated = run_layer_forward_int(*args, f_a=4, f_V=0, i_V=4, overflow_mode="saturate")
    assert int(wrapped.v_final[0]) == -7
    assert int(saturated.v_final[0]) == 7
    assert bool(saturated.overflowed[0, 0])


def test_run_layer_forward_int_traced_matches_untraced_and_last_v_step_is_v_final():
    a_int = jnp.array([[999, 12]])
    is_identity = jnp.array([[True, False]])
    q_int = jnp.array([[5, 3]])
    v_th_int = jnp.array([100])

    untraced = run_layer_forward_int(a_int, is_identity, q_int, v_th_int, f_a=4, f_V=2, i_V=8)
    traced, v_steps = run_layer_forward_int_traced(a_int, is_identity, q_int, v_th_int,
                                                    f_a=4, f_V=2, i_V=8)

    assert int(traced.v_final[0]) == int(untraced.v_final[0])
    assert bool(traced.spike_mask[0, 0]) == bool(untraced.spike_mask[0, 0])
    assert v_steps.shape == (1, 2)
    assert int(v_steps[0, 0]) == 20, "第一步套用完的膜電位"
    assert int(v_steps[0, -1]) == int(traced.v_final[0]), "最後一欄要等於 v_final"
