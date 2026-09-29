"""整數掃描:一次一筆事件更新整數膜電位暫存器,模擬 FPGA 的逐事件電路。

推導見 docs/math/膜電位量化推導.md;定點數運算在 quant/fixed_point.py。
"""
import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp

from salt_core.quant.fixed_point import OverflowMode, RoundMode, fit_to_bits, wide_mul_shift


class QuantEventResult(NamedTuple):
    """process_event 的回傳:一筆事件之後的暫存器狀態。"""
    v_final: jax.Array     # int32,這筆事件之後的暫存器值;fire 時是 0
    is_spiked: jax.Array   # bool
    overflowed: jax.Array  # bool,寫回之前的真實值有沒有超出 i_V+f_V 位元


def process_event(v0_int: jax.Array, a_int: jax.Array, is_identity: jax.Array,
                  q_int: jax.Array, v_th_int: jax.Array | None, *, f_a: int, f_V: int,
                  i_V: int, round_mode: RoundMode | str = RoundMode.ROUND,
                  overflow_mode: OverflowMode | str = OverflowMode.WRAP
                 ) -> QuantEventResult:
    """一筆事件的整數更新:V = r(a * V) + q,全程整數。

    v0_int: 暫存器值,等於膜電位 * 2 ** f_V 的整數。
    a_int、is_identity: apply_decay_table_int 查出的衰減碼(小數 f_a 位元);is_identity(dt=0)時不衰減。
    q_int: 權重整數碼,加進暫存器前左移 f_V 位。
    v_th_int: 整數門檻;None 時不 fire、不 reset。
    fire 判斷比的是溢位處理之後的值,因為硬體比較電路讀的是暫存器裡的位元;fire 時硬 reset 成 0。
    位元寬度超過 fixed_point 的上限(MAX_SHIFT_BITS、MAX_REGISTER_BITS)時 raise ValueError。
    """
    v0_int = jnp.asarray(v0_int, dtype=jnp.int32)
    a_int = jnp.asarray(a_int, dtype=jnp.int32)
    q_int = jnp.asarray(q_int, dtype=jnp.int32)

    decayed_active = wide_mul_shift(a_int, v0_int, shift_bits=f_a, round_mode=round_mode)
    decayed = jnp.where(is_identity, v0_int, decayed_active)
    v_unfitted = decayed + q_int * (1 << f_V)
    fitted, overflowed = fit_to_bits(v_unfitted, total_bits=i_V + f_V,
                                     overflow_mode=overflow_mode)

    if v_th_int is None:
        return QuantEventResult(v_final=fitted, is_spiked=jnp.zeros_like(fitted, dtype=bool),
                                overflowed=overflowed)
    is_spiked = fitted >= jnp.asarray(v_th_int, dtype=jnp.int32)
    v_final = jnp.where(is_spiked, jnp.zeros_like(fitted), fitted)
    return QuantEventResult(v_final=v_final, is_spiked=is_spiked, overflowed=overflowed)


class QuantLayerResult(NamedTuple):
    """一層整數 forward 的結果。第 t 步處理佇列第 t 欄;沒有 s_value、s_spike(沒有梯度)。"""
    spike_mask: jax.Array       # (n, queue_len) bool
    spike_event_idx: jax.Array  # (n, queue_len) int32,佇列欄位,跟 FloatLayerResult 同一個慣例
    v_final: jax.Array          # (n,) int32 暫存器值;QuantBackend.readout 之後是物理尺度 float32
    overflowed: jax.Array       # (n, queue_len) bool,這一步寫回之前的值有沒有超出 i_V + f_V 位元


def run_layer(a_int: jax.Array, is_identity: jax.Array, q_int: jax.Array,
              v_th_int: jax.Array | None, *, f_a: int, f_V: int, i_V: int,
              round_mode: RoundMode | str = RoundMode.ROUND,
              overflow_mode: OverflowMode | str = OverflowMode.WRAP
             ) -> QuantLayerResult:
    """一層的整數掃描:每步處理一筆事件,步數等於佇列長度。

    a_int、is_identity、q_int: (n, queue_len)。每一欄都照實套用,不看真事件數:不是真事件的位置
        要在建佇列時做成 dt=0、權重 0;conv 真事件之後的欄是真的要衰減。
    v_th_int: 純量或 (n,);None 時這層不 fire。
    其餘參數整層共用,原樣交給 process_event。
    """
    result, _v_steps = _run_layer_scan(a_int, is_identity, q_int, v_th_int,
                                       f_a=f_a, f_V=f_V, i_V=i_V, round_mode=round_mode,
                                       overflow_mode=overflow_mode, trace=False)
    return result


def run_layer_traced(a_int: jax.Array, is_identity: jax.Array, q_int: jax.Array,
                     v_th_int: jax.Array | None, *, f_a: int, f_V: int, i_V: int,
                     round_mode: RoundMode | str = RoundMode.ROUND,
                     overflow_mode: OverflowMode | str = OverflowMode.WRAP
                    ) -> tuple[QuantLayerResult, jax.Array]:
    """同 run_layer,另外回傳 v_steps:(n, queue_len) 每步寫回之後的暫存器值,最後一欄等於 v_final。

    量溢位要看逐步值:膜電位可能中途衝高再衰減下來,只看 v_final 會漏掉。
    """
    return _run_layer_scan(a_int, is_identity, q_int, v_th_int,
                           f_a=f_a, f_V=f_V, i_V=i_V, round_mode=round_mode,
                           overflow_mode=overflow_mode, trace=True)


def _run_layer_scan(a_int: jax.Array, is_identity: jax.Array, q_int: jax.Array,
                    v_th_int: jax.Array | None, *, f_a: int, f_V: int, i_V: int,
                    round_mode: RoundMode | str, overflow_mode: OverflowMode | str,
                    trace: bool) -> tuple[QuantLayerResult, jax.Array | None]:
    """兩個公開函式共用的掃描內核。回傳 (result, v_steps),trace=False 時 v_steps 是 None。"""
    n_out_neurons, queue_len = a_int.shape
    # None(不 fire)是沒有 leaf 的 pytree,vmap 直接原樣傳給 process_event
    v_th_arr = (None if v_th_int is None else
                jnp.broadcast_to(jnp.asarray(v_th_int, dtype=jnp.int32), (n_out_neurons,)))
    step_fn = functools.partial(process_event, f_a=f_a, f_V=f_V, i_V=i_V,
                                round_mode=round_mode, overflow_mode=overflow_mode)
    vmapped_step = jax.vmap(step_fn)

    def step(v, t):
        step_result = vmapped_step(v, a_int[:, t], is_identity[:, t], q_int[:, t], v_th_arr)
        ys = (step_result.is_spiked, step_result.overflowed)
        if trace:
            ys = ys + (step_result.v_final,)
        return step_result.v_final, ys

    v_final, ys = jax.lax.scan(step, jnp.zeros(n_out_neurons, dtype=jnp.int32),
                               jnp.arange(queue_len))
    spike_mask, overflowed = ys[0], ys[1]

    # scan 疊出來是 (queue_len, n),轉成 (n, queue_len)
    spike_event_idx = jnp.broadcast_to(jnp.arange(queue_len), (n_out_neurons, queue_len))
    result = QuantLayerResult(spike_mask=spike_mask.T, spike_event_idx=spike_event_idx,
                                   v_final=v_final, overflowed=overflowed.T)
    if not trace:
        return result, None
    v_steps = ys[2]
    return result, v_steps.T
