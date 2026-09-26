"""`run_layer_forward` 新加的兩個能力,直接用 `core.create_affine_maps` 兜出
`AffineMap`(跟 `test_core.py` 同一種手法,不經過 fc/conv 佇列建構,只測
`chunk_scan.py` 這層自己的接線):

1. `v_th` 可以是逐神經元陣列,不再只能整層共用一個純量。
2. `round_step`/`round_mode` 會原樣穿透到 `core.process_chunk`,`chunk_scan.py`
   自己沒有另外做任何事——而且 `round_step` 一樣要能逐神經元各自不同(跟
   `v_th` 同一個理由:膜電位量化的捨入格距 s_c*2^-f_V 逐 channel 不同)。

見 docs/math/膜電位量化推導.md。
"""

import jax.numpy as jnp

from salt_core.chunk_scan import run_layer_forward, run_layer_forward_int, run_layer_forward_int_traced
from salt_core.core import AffineMap, create_affine_maps

TOL = 1e-6


def assert_allclose(actual, expected, msg):
    actual = float(actual)
    expected = float(expected)
    assert abs(actual - expected) < TOL, f"{msg}: got {actual}, expected {expected}"


def test_run_layer_forward_scalar_v_th_unchanged():
    """v_th 傳純量時行為要跟改動前完全一樣(回歸測試,對照
    test_fc_forward.py 同一個手算例子的單步版本)。"""
    tau = 4.0
    maps = create_affine_maps(jnp.array([[0.0]]), jnp.array([[1.5]]), tau)  # (n=1, L=1)
    result = run_layer_forward(maps, v_th=1.0, chunk_size=1, max_steps=1, n_real_events=1)

    assert bool(result.spike_mask[0, 0]), "b=1.5 >= v_th=1.0,應該 fire"
    assert_allclose(result.v_final[0], 0.0, "fire 後硬重置成 0")


def test_run_layer_forward_per_neuron_v_th():
    """兩顆神經元收到完全一樣的事件(a=1.0, b=1.5),但各自的 v_th 不同——
    v_th=1.0 的那顆該 fire,v_th=2.0 的那顆不該 fire。"""
    tau = 4.0
    a = jnp.array([[1.0], [1.0]])
    b = jnp.array([[1.5], [1.5]])
    maps = AffineMap(a=a, b=b)
    v_th_per_neuron = jnp.array([1.0, 2.0])

    result = run_layer_forward(maps, v_th_per_neuron, chunk_size=1, max_steps=1, n_real_events=1)

    assert bool(result.spike_mask[0, 0]), "neuron0: b=1.5 >= v_th=1.0,應該 fire"
    assert not bool(result.spike_mask[1, 0]), "neuron1: b=1.5 < v_th=2.0,不該 fire"
    assert_allclose(result.v_final[0], 0.0, "neuron0 fire 後硬重置成 0")
    assert_allclose(result.v_final[1], 1.5, "neuron1 沒 fire,維持原始值")


def test_run_layer_forward_round_step_threading():
    """round_step 原樣穿透到 process_chunk,而且是逐步(逐事件)生效,不是
    整條軌跡算完才捨一次——round_step 只捨「衰減項」(見
    test_core.py 的 test_process_chunk_round_step_rounds_the_decay_term_not_the_sum),
    所以要兩筆事件才測得出來:第一筆事件從 v0=0 出發,衰減項恆為 0,不受
    round_step 影響,先把狀態墊到 1.28;第二筆事件的衰減項
    a*1.28=0.75*1.28=0.96 才是真正被捨入的地方。v_th 設極大只看軌跡,
    不管 spike。"""
    tau = 4.0
    maps = create_affine_maps(jnp.array([[0.0, 1.0]]), jnp.array([[1.28, 0.0]]), tau)

    without_round = run_layer_forward(maps, v_th=1e9, chunk_size=1, max_steps=2, n_real_events=2)
    assert_allclose(without_round.v_final[0], 0.96, "沒有 round_step:0.75*1.28=0.96")

    with_round = run_layer_forward(maps, v_th=1e9, chunk_size=1, max_steps=2, n_real_events=2,
                                    round_step=0.1)
    assert_allclose(with_round.v_final[0], 1.0, "round_step=0.1 把 0.96 捨入成 1.0")


def test_run_layer_forward_per_neuron_round_step():
    """round_step 逐神經元各自不同,才是膜電位量化真正要用的樣子(每個 channel
    的捨入格距 s_c*2^-f_V 不一樣)。兩顆神經元收到完全一樣的兩筆事件(跟上一
    個測試同一組數字),但各自的捨入格距不同——格距 0.1 的那顆把 0.96 捨入
    成 1.0,格距 0.01 的那顆捨入後幾乎不變(仍是 0.96)。這是原本用單一
    Python 函式物件當 round_fn 時做不到、且沒被測出來的情況。"""
    tau = 4.0
    a = jnp.array([[1.0, 0.75], [1.0, 0.75]])
    b = jnp.array([[1.28, 0.0], [1.28, 0.0]])
    maps = AffineMap(a=a, b=b)
    round_step_per_neuron = jnp.array([0.1, 0.01])

    result = run_layer_forward(maps, v_th=1e9, chunk_size=1, max_steps=2, n_real_events=2,
                                round_step=round_step_per_neuron)

    assert_allclose(result.v_final[0], 1.0, "neuron0: 格距 0.1,0.96 捨入成 1.0")
    assert_allclose(result.v_final[1], 0.96, "neuron1: 格距 0.01,0.96 捨入後幾乎不變")


# ============================================================================
# run_layer_forward_int / run_layer_forward_int_traced(見 docs/問題紀錄.md
# 第十七節)。chunk_size 恆為 1,直接手算兩筆事件的鏈式遞迴對答案。
# ============================================================================

def test_run_layer_forward_int_two_events_chained_matches_hand_computation():
    """單一神經元,兩筆事件鏈式遞迴。i_V=8,f_V=2,f_a=4。

    事件 0:is_identity=True(跳過衰減),q=5,f_V=2 => raw=0+5*4=20,無溢位。
    事件 1:is_identity=False,a_int=12(對應 tau=4 時 Δt=1 的 0.75 查表值),
    v0=20:12*20=240,240/16=15(整除),decayed=15;q=3 => raw=15+3*4=27。"""
    a_int = jnp.array([[999, 12]])       # 事件0的 a_int 是垃圾值,is_identity=True 時不會被用到
    is_identity = jnp.array([[True, False]])
    q_int = jnp.array([[5, 3]])
    v_th_int = jnp.array([100])

    result = run_layer_forward_int(a_int, is_identity, q_int, v_th_int, max_steps=2,
                                    n_real_events=2, f_a=4, f_V=2, i_V=8)

    assert int(result.v_final[0]) == 27
    assert not bool(result.spike_mask[0, 0]) and not bool(result.spike_mask[0, 1])
    assert not bool(result.overflowed[0, 0]) and not bool(result.overflowed[0, 1])


def test_run_layer_forward_int_pad_positions_forced_to_identity_and_zero():
    """n_real_events=1:第二欄雖然存了誇張的垃圾值(a_int 很大、is_identity=
    False、q_int 很大),但因為超過 n_real_events,要被強制當 identity+q=0,
    v_final 應該停在第一筆事件算完的值(20),不受第二欄垃圾值影響。"""
    a_int = jnp.array([[999, 999999]])
    is_identity = jnp.array([[True, False]])
    q_int = jnp.array([[5, 999999]])
    v_th_int = jnp.array([100])

    result = run_layer_forward_int(a_int, is_identity, q_int, v_th_int, max_steps=2,
                                    n_real_events=1, f_a=4, f_V=2, i_V=8)

    assert int(result.v_final[0]) == 20, "pad 欄位不該貢獻任何變化"
    assert not bool(result.spike_mask[0, 1])


def test_run_layer_forward_int_per_neuron_v_th_fire_and_reset():
    """兩顆神經元收到完全一樣的事件,各自的 v_th_int 不同——門檻低的那顆該
    fire 並硬重置成 0,門檻高的那顆不該 fire。"""
    a_int = jnp.array([[999], [999]])
    is_identity = jnp.array([[True], [True]])
    q_int = jnp.array([[20], [20]])   # f_V=0 時直接貢獻 20
    v_th_int = jnp.array([15, 25])

    result = run_layer_forward_int(a_int, is_identity, q_int, v_th_int, max_steps=1,
                                    n_real_events=1, f_a=4, f_V=0, i_V=8)

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

    result = run_layer_forward_int(a_int, is_identity, q_int, v_th_int, max_steps=1,
                                    n_real_events=1, f_a=4, f_V=0, i_V=4)

    assert bool(result.overflowed[0, 0])
    assert int(result.v_final[0]) == -7
    assert not bool(result.spike_mask[0, 0])


def test_run_layer_forward_int_traced_matches_untraced_and_last_v_step_is_v_final():
    a_int = jnp.array([[999, 12]])
    is_identity = jnp.array([[True, False]])
    q_int = jnp.array([[5, 3]])
    v_th_int = jnp.array([100])

    untraced = run_layer_forward_int(a_int, is_identity, q_int, v_th_int, max_steps=2,
                                      n_real_events=2, f_a=4, f_V=2, i_V=8)
    traced, v_steps = run_layer_forward_int_traced(a_int, is_identity, q_int, v_th_int,
                                                    max_steps=2, n_real_events=2,
                                                    f_a=4, f_V=2, i_V=8)

    assert int(traced.v_final[0]) == int(untraced.v_final[0])
    assert bool(traced.spike_mask[0, 0]) == bool(untraced.spike_mask[0, 0])
    assert v_steps.shape == (1, 2)
    assert int(v_steps[0, 0]) == 20, "第一步套用完的膜電位"
    assert int(v_steps[0, -1]) == int(traced.v_final[0]), "最後一欄要等於 v_final"


TESTS = [
    test_run_layer_forward_scalar_v_th_unchanged,
    test_run_layer_forward_per_neuron_v_th,
    test_run_layer_forward_round_step_threading,
    test_run_layer_forward_per_neuron_round_step,
    test_run_layer_forward_int_two_events_chained_matches_hand_computation,
    test_run_layer_forward_int_pad_positions_forced_to_identity_and_zero,
    test_run_layer_forward_int_per_neuron_v_th_fire_and_reset,
    test_run_layer_forward_int_overflow_flag_set_and_propagates_to_v_final,
    test_run_layer_forward_int_traced_matches_untraced_and_last_v_step_is_v_final,
]


if __name__ == "__main__":
    for test in TESTS:
        test()
        print(f"PASS: {test.__name__}")
    print(f"\n全部 {len(TESTS)} 項測試通過")
