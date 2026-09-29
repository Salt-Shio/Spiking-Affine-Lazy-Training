"""浮點版一層的掃描:每顆神經元用 chunk 大小的視窗反覆跑 process_chunk,fire 之後從下一筆
事件重新開始,直到步數用完。

演算法見 docs/math/單狀態仿射平行掃描推導.md「Chunk 內投機執行(speculative execution),流程不變、
每一步更輕量」;整數版在 quant/scan.py。
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from salt_core.float.affine import AffineMap, normalize_real_events, process_chunk


class FloatLayerResult(NamedTuple):
    """一層 forward 的結果,也是解碼器讀的東西。s_value 已經不含 pad 事件,可以直接整個加總。"""
    spike_mask: jax.Array       # (n, max_steps) 這一步有沒有 fire
    spike_event_idx: jax.Array  # (n, max_steps) fire 的是佇列第幾欄;FC 就是全域事件 index,
                                # conv 要查 local_to_global_j。沒 fire 的位置沒有意義
    s_spike: jax.Array          # (n, max_steps) fire 那筆事件自己的 s,forward 等於 1;下一層的 event_gain
    s_value: jax.Array          # (n, max_steps) 這一步有效事件的 s 加總
    v_final: jax.Array          # (n,) 跑完 max_steps 步之後的膜電位


def _pad_queue(maps: AffineMap, pad_len: int) -> AffineMap:
    """佇列尾端補 pad_len 筆不作用的映射(a=1, b=0),讓最後一個視窗不會讀出界。"""
    n_out_neurons = maps.a.shape[0]
    pad_a = jnp.ones((n_out_neurons, pad_len), dtype=maps.a.dtype)
    pad_b = jnp.zeros((n_out_neurons, pad_len), dtype=maps.b.dtype)
    return AffineMap(a=jnp.concatenate([maps.a, pad_a], axis=1),
                      b=jnp.concatenate([maps.b, pad_b], axis=1))


def run_layer(maps: AffineMap, v_th: float, chunk_size: int, max_steps: int,
              n_real_events: jax.Array | int,
              alpha: float = 2.0) -> FloatLayerResult:
    """一層的浮點掃描。

    maps: a、b 形狀 (n, queue_len),每顆神經元一條佇列。
    v_th: fire 門檻。alpha: surrogate gradient 的平滑程度,見 surrogate.py。
    chunk_size: 每步最多處理幾筆事件;fire 時那一步停在 fire 那筆。
    max_steps: 掃描步數,要夠跑完整條佇列;chunk_size=1 或每筆都 fire 時要 queue_len 步。
    n_real_events: 純量或 (n,),前幾筆是真事件;之後的位置不算進 s_value。
    回傳 FloatLayerResult。s_value 是每步有效事件的 s 加總(有 fire 時加到 fire 那筆為止,
    沒 fire 時只加真事件),每筆真事件剛好被加一次;定義見 docs/math/不套閘與soft-reset梯度推導.md
    「決定的解法:「加總這個窗口裡所有有效位置」,取代「只挑一個代表位置」」。
    """
    result, _v_steps, _pointer_steps = _run_layer_scan(
        maps, v_th, chunk_size, max_steps, n_real_events, alpha, trace=False)
    return result


def run_layer_traced(maps: AffineMap, v_th: float, chunk_size: int, max_steps: int,
                     n_real_events: jax.Array | int, alpha: float = 2.0
                     ) -> tuple[FloatLayerResult, jax.Array, jax.Array]:
    """同 run_layer,另外回傳逐步軌跡。不進訓練熱路徑。

    回傳 (result, v_steps, pointer_steps):
    result: 跟 run_layer 同一個掃描內核算出的 FloatLayerResult。
    v_steps: (n, max_steps) 每步結束(套過 reset)的膜電位,最後一欄等於 v_final。
    pointer_steps: (n, max_steps) int,每步開始時處理到佇列第幾欄。
    """
    return _run_layer_scan(maps, v_th, chunk_size, max_steps, n_real_events, alpha,
                            trace=True)


def _run_layer_scan(maps: AffineMap, v_th: float, chunk_size: int, max_steps: int,
                     n_real_events: jax.Array | int, alpha: float, *, trace: bool
                     ) -> tuple[FloatLayerResult, jax.Array | None, jax.Array | None]:
    """兩個公開函式共用的掃描內核。回傳 (result, v_steps, pointer_steps),trace=False 時後兩個是 None。"""
    n_out_neurons = maps.a.shape[0]
    n_real = normalize_real_events(n_real_events, n_out_neurons)
    padded = _pad_queue(maps, chunk_size)
    gather = jax.vmap(lambda arr, idx: jnp.take(arr, idx, mode='clip'))
    chunk_idx_range = jnp.arange(chunk_size)

    def take_chunk(pointer):
        idx = pointer[:, None] + jnp.arange(chunk_size)[None, :]  # shape (n_out_neurons, chunk_size)
        return AffineMap(a=gather(padded.a, idx), b=gather(padded.b, idx))

    neuron_idx_range = jnp.arange(n_out_neurons)

    def step(carry, _):
        v, pointer = carry
        chunk = take_chunk(pointer)
        chunk_result = jax.vmap(process_chunk, in_axes=(0, 0, None, None))(v, chunk, v_th, alpha)

        n_real_remaining = jnp.clip(n_real - pointer, 0, chunk_size)
        n_valid_in_chunk = jnp.where(chunk_result.is_spiked, chunk_result.spike_idx + 1,
                                      n_real_remaining)
        chunk_valid_mask = chunk_idx_range[None, :] < n_valid_in_chunk[:, None]  # shape (n_out_neurons, chunk_size)
        s_value = jnp.sum(jnp.where(chunk_valid_mask, chunk_result.s_sequence, 0.0), axis=1)

        # fire 那筆事件自己的 s,process_chunk 已經算過,這裡只是取出來
        spike_idx_clamped = jnp.minimum(chunk_result.spike_idx, chunk_size - 1)
        s_spike = chunk_result.s_sequence[neuron_idx_range, spike_idx_clamped]

        n_consumed = jnp.where(chunk_result.is_spiked, chunk_result.spike_idx + 1, chunk_size)
        spike_event_idx = pointer + chunk_result.spike_idx
        new_pointer = pointer + n_consumed
        ys = (chunk_result.is_spiked, spike_event_idx, s_spike, s_value)
        if trace:
            ys = ys + (chunk_result.v_final, pointer)
        return (chunk_result.v_final, new_pointer), ys

    init = (jnp.zeros(n_out_neurons, dtype=maps.a.dtype), jnp.zeros(n_out_neurons, dtype=jnp.int32))
    (v_final, _), ys = jax.lax.scan(step, init, None, length=max_steps)
    spike_mask, spike_event_idx, s_spike, s_value = ys[:4]

    # scan 疊出來是 (max_steps, n),轉成 (n, max_steps)
    result = FloatLayerResult(spike_mask=spike_mask.T, spike_event_idx=spike_event_idx.T,
                              s_spike=s_spike.T, s_value=s_value.T, v_final=v_final)
    if not trace:
        return result, None, None
    v_step, pointer_step = ys[4], ys[5]
    return result, v_step.T, pointer_step.T
