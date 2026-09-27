"""整數掃描(膜電位量化模擬):一次一筆事件更新整數膜電位暫存器,模擬 FPGA 的逐事件
電路。推導見 docs/math/膜電位量化推導.md。定點數運算電路在 quant.fixed_point。
"""
import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp

from salt_core.quant.fixed_point import OverflowMode, RoundMode, fit_to_bits, wide_mul_shift



class QuantEventResult(NamedTuple):
    """`process_event` 的回傳:一筆事件更新完之後的整數膜電位狀態。"""
    v_final: jax.Array     # 這筆事件之後的膜電位(int32,spike 時已硬重置為 0)
    is_spiked: jax.Array   # bool
    overflowed: jax.Array  # bool,寫回之前的真實值有沒有超出 i_V+f_V 位元


def process_event(v0_int: jax.Array, a_int: jax.Array, is_identity: jax.Array,
                  q_int: jax.Array, v_th_int: jax.Array | None, *, f_a: int, f_V: int,
                  i_V: int, round_mode: RoundMode | str = RoundMode.ROUND,
                  overflow_mode: OverflowMode | str = OverflowMode.WRAP
                 ) -> QuantEventResult:
    """整數單一事件更新,對應 $\\tilde V_k=r(a_k\\tilde V_{k-1})+q_k$。

    暫存器值是 $\\tilde V\\cdot2^{f_V}$ 的整數;`a_int` 是 Q0.`f_a` 的衰減碼;
    `q_int` 是權重整數碼,加進暫存器前左移 `f_V` 位。全程只有整數運算,
    沒有 $s_c$、沒有浮點除法。一次只處理一筆事件,因為硬體每筆事件更新後
    就立刻捨入。沒有 surrogate gradient,直接硬判斷 `>=`、硬重置成 0。

    `a_int`/`is_identity` 來自 `quant.codes.apply_decay_table_int`:
    `is_identity=True`(Δt=0)時跳過衰減,`decayed` 直接等於 `v0_int`。

    `v_th_int=None` 代表這層不 fire(例如膜電位回歸的輸出層):不做 fire
    判斷、不重置,暫存器一路累積。

    溢位照 `overflow_mode` 處理(`quant.fixed_point.fit_to_bits`,預設繞回,定案
    理由見 docs/math/膜電位量化推導.md「溢位政策」節):`overflowed` 只回報
    有沒有發生,`v_final` 是寫回之後的值。fire 判斷比的也是寫回之後的值,
    因為硬體比較電路讀到的就是暫存器裡的位元。

    位元寬度限制:`f_a <= quant.fixed_point.MAX_SHIFT_BITS`、
    `i_V + f_V <= quant.fixed_point.MAX_REGISTER_BITS`,超過時由
    `quant.fixed_point` 的 primitive raise `ValueError`(理由見該模組說明)。
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
    """一層跑完一次整數版 forward 的輸出(chunk_size 恆為 1)。沒有
    `s_value`/`s_spike`:這條路沒有 surrogate gradient,rate coding 要用的話
    直接對 `spike_mask` 加總。`L` 是佇列長度,第 `t` 步處理佇列第 `t` 欄。
    """
    spike_mask: jax.Array       # shape (n_out_neurons, L) bool
    spike_event_idx: jax.Array  # shape (n_out_neurons, L) int32,就是佇列自己的欄位索引
                                 # (跟 LayerForwardResult 同一個局部 index 慣例)
    v_final: jax.Array          # shape (n_out_neurons,) 暫存器值 int32;
                                 # QuantBackend.readout 換過之後是物理尺度 float32
    overflowed: jax.Array       # shape (n_out_neurons, L) bool,每一步寫回暫存器
                                 # 之前的真實值有沒有超出 i_V+f_V 位元


def run_layer(a_int: jax.Array, is_identity: jax.Array, q_int: jax.Array,
              v_th_int: jax.Array | None, *, f_a: int, f_V: int, i_V: int,
              round_mode: RoundMode | str = RoundMode.ROUND,
              overflow_mode: OverflowMode | str = OverflowMode.WRAP
             ) -> QuantLayerResult:
    """整數版掃描:每一步只處理一筆事件(硬體每筆事件更新後就立刻捨入),
    所以第 `t` 步就是佇列第 `t` 欄,掃描長度等於佇列長度 `a_int.shape[1]`,
    不需要浮點版那套滑動視窗跟 pointer。

    `a_int`/`is_identity`/`q_int`:shape `(n_out_neurons, L)`,呼叫端先用
    `quant.codes.apply_decay_table_int` 查好表、準備好權重整數碼(這個函式只跑
    遞迴)。**每一欄都照實套用**,不看真事件數:佇列裡不是真事件的位置,
    佇列建構那一步就要做成 identity(Δt=0、權重 0),conv 的 catch-up 欄則是
    真的要衰減(見 `connectivity.conv.build_conv_structure`、
    `connectivity.fc.build_fc_structure`)。

    `v_th_int`:純量或 shape `(n_out_neurons,)`;`None` 代表這層不 fire。
    `f_a`/`f_V`/`i_V`/`round_mode`/`overflow_mode` 整層共用,原樣交給
    `process_event`。
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
    """跟 `run_layer` 跑一模一樣的掃描,額外回傳每步(寫回之後)
    的暫存器值 `v_steps`,shape `(n_out_neurons, L)`,最後一欄等於
    `result.v_final`。溢位驗證要看逐步值:神經元可能中途衝到峰值再衰減下來,
    只看 v_final 會漏掉。
    """
    return _run_layer_scan(a_int, is_identity, q_int, v_th_int,
                           f_a=f_a, f_V=f_V, i_V=i_V, round_mode=round_mode,
                           overflow_mode=overflow_mode, trace=True)


def _run_layer_scan(a_int: jax.Array, is_identity: jax.Array, q_int: jax.Array,
                    v_th_int: jax.Array | None, *, f_a: int, f_V: int, i_V: int,
                    round_mode: RoundMode | str, overflow_mode: OverflowMode | str,
                    trace: bool):
    """`run_layer`/`run_layer_traced` 共用的 scan
    內核。回傳 `(QuantLayerResult, v_steps)`;`trace=False` 時後者是
    `None`。"""
    n_out_neurons, queue_len = a_int.shape
    # None(不 fire)是沒有 leaf 的 pytree,vmap 直接原樣傳給 process_event
    v_th_arr = (None if v_th_int is None else
                jnp.broadcast_to(jnp.asarray(v_th_int, dtype=jnp.int32), (n_out_neurons,)))
    step_fn = functools.partial(process_event, f_a=f_a, f_V=f_V, i_V=i_V,
                                round_mode=round_mode, overflow_mode=overflow_mode)
    vmapped_step = jax.vmap(step_fn)

    def step(v, t):
        step_result = vmapped_step(v, a_int[:, t], is_identity[:, t], q_int[:, t], v_th_arr)
        # scan 會把每一步的 ys 疊成陣列回傳給外面,step 自己不讀它
        ys = (step_result.is_spiked, step_result.overflowed)
        if trace:
            ys = ys + (step_result.v_final,)
        return step_result.v_final, ys

    v_final, ys = jax.lax.scan(step, jnp.zeros(n_out_neurons, dtype=jnp.int32),
                               jnp.arange(queue_len))
    spike_mask, overflowed = ys[0], ys[1]

    # scan 的疊代軸在最前面,shape 是 (L, n_out_neurons),轉成
    # (n_out_neurons, L) 給呼叫端用
    spike_event_idx = jnp.broadcast_to(jnp.arange(queue_len), (n_out_neurons, queue_len))
    result = QuantLayerResult(spike_mask=spike_mask.T, spike_event_idx=spike_event_idx,
                                   v_final=v_final, overflowed=overflowed.T)
    if not trace:
        return result, None
    v_steps = ys[2]
    return result, v_steps.T
