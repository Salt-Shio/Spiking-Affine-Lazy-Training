"""驗證 build_fc_structure + fc_float_values + chunk_scan.run_layer_forward 串起來,
數字對得上 docs/math/全連接forward訓練範例.md 第 3、4 節的手算例子:

  n=2 (a1,a2), m=2 (b1,b2), tau=4, v_th=1.0
  W = [[w11=0.6, w21=0.5],   (連到 b1)
       [w12=0.3, w22=0.2]]   (連到 b2)
  上游事件(全域,已排序): (t=1,a1) (t=2,a2) (t=4,a1)

  手算結果: b1 在 t=4(全域事件 index=2)fire,fire 後 V=0;
            b2 全程不 fire,最終 V=0.539。

用兩種 chunk_size(1 跟 3,事件總數)分別跑一次,確認結果跟 chunk_size 無關
——這是驗證「fire 後從下一筆事件重新起跑,不用重算係數」這個設計對不對的
關鍵測試,不是只測 forward 數字本身。
"""

import jax.numpy as jnp

from salt_core.chunk_scan import run_layer_forward
from salt_core.connectivity.fc import build_fc_structure, fc_float_values

TOL = 1e-4


def assert_allclose(actual, expected, msg):
    actual = float(actual)
    expected = float(expected)
    assert abs(actual - expected) < TOL, f"{msg}: got {actual}, expected {expected}"


def _run(chunk_size):
    tau = 4.0
    v_th = 1.0
    event_times = jnp.array([1.0, 2.0, 4.0])
    event_source_idx = jnp.array([0, 1, 0])
    W = jnp.array([[0.6, 0.5],
                   [0.3, 0.2]])

    maps = fc_float_values(build_fc_structure(event_times, event_source_idx, event_times.shape[0]),
                           W, tau, None)
    assert maps.a.shape == (2, 3)

    n_real_events = event_times.shape[0]
    return run_layer_forward(maps, v_th, chunk_size=chunk_size, max_steps=n_real_events,
                              n_real_events=n_real_events)


def _check(spike_mask, spike_event_idx, _s_spike, _s_value, v_final):
    # b1(row 0)恰好 fire 一次,在全域事件 index=2(t=4)
    b1_spikes = [i for i in range(spike_mask.shape[1]) if bool(spike_mask[0, i])]
    assert len(b1_spikes) == 1, f"b1 應該恰好 fire 一次,實際: {b1_spikes}"
    assert int(spike_event_idx[0, b1_spikes[0]]) == 2, "b1 fire 的事件 index 應該是 2(t=4)"

    # b2(row 1)全程不 fire
    assert not bool(spike_mask[1].any()), "b2 不應該 fire"

    assert_allclose(v_final[0], 0.0, "b1 fire 後應硬重置成 0")
    assert_allclose(v_final[1], 0.539, "b2 最終電壓")


def test_fc_forward_chunk_size_1():
    _check(*_run(chunk_size=1))


def test_fc_forward_chunk_size_full():
    _check(*_run(chunk_size=3))


def test_fc_structure_delta_t_matches_hand_computation():
    """跟上面同一組 event_times=[1,2,4]:Δt 是跟前一筆事件的差,第一筆跟 t=0
    比,[1-0, 2-1, 4-2]=[1,1,2]。兩顆輸出神經元的 a 都用這組 Δt 算。"""
    event_times = jnp.array([1.0, 2.0, 4.0])
    structure = build_fc_structure(event_times, jnp.array([0, 1, 0]), 3)
    maps = fc_float_values(structure, jnp.ones((2, 2)), 4.0, None)
    assert list(structure.delta_t) == [1, 1, 2]
    assert maps.a.shape == (2, 3)
    assert jnp.allclose(maps.a, 0.75 ** structure.delta_t[None, :])


def test_fc_structure_pad_positions_are_identity_with_zero_delta_t():
    """n_real_events=2:第三筆是 pad 事件,時間是多層串接用的假時間 1e12。
    pad 位置的 Δt 要是 0(不是 1e12-2 這種天文數字)、a=1、b=0。"""
    event_times = jnp.array([1.0, 2.0, 1e12])
    structure = build_fc_structure(event_times, jnp.array([0, 1, 0]), 2)
    maps = fc_float_values(structure, jnp.array([[0.5, 0.7]]), 4.0, None)
    assert list(structure.delta_t) == [1, 1, 0]
    assert float(maps.a[0, 2]) == 1.0
    assert float(maps.b[0, 2]) == 0.0
