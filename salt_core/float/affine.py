"""單狀態 LIF 的仿射運算:仿射映射、合成、一個 chunk 的 fire 偵測、掃描步數公式。

一筆事件把膜電位 x 更新成 a * x + b,a = (1 - 1/tau) ** dt,b 是權重。
推導見 docs/math/單狀態仿射平行掃描推導.md;整數版的單步更新在 quant/scan.py。
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from salt_core.float.surrogate import atan_spike


class AffineMap(NamedTuple):
    """x -> a * x + b,對應一筆事件對膜電位的更新映射(衰減 + 累加權重)。"""
    a: jax.Array
    b: jax.Array


def combine(left: AffineMap, right: AffineMap) -> AffineMap:
    """合成兩個仿射映射,right 是 left 之後發生的映射:right ∘ left。"""
    a = right.a * left.a
    b = right.a * left.b + right.b
    return AffineMap(a=a, b=b)


def create_affine_maps(n_ms: jax.Array, w: jax.Array, tau: float) -> AffineMap:
    """一批事件各自的仿射映射:a = (1 - 1/tau) ** n_ms,b = w。逐元素算,a 跟 n_ms 同形狀,b 跟 w 同形狀。"""
    a = (1.0 - 1.0 / tau) ** jnp.asarray(n_ms, dtype=jnp.float32)
    b = jnp.asarray(w, dtype=jnp.float32)
    return AffineMap(a=a, b=b)


def normalize_real_events(n_real_events: jax.Array | int, n_out_neurons: int) -> jax.Array:
    """每顆神經元佇列裡的真事件數,統一成 (n_out_neurons,) int32 陣列。

    n_real_events: 純量(所有神經元共用)或 (n_out_neurons,) 陣列(每顆各自的數)。
    """
    arr = jnp.asarray(n_real_events, dtype=jnp.int32)
    return jnp.broadcast_to(arr, (n_out_neurons,))


def mask_pad_events(maps: AffineMap,
                     n_real_events: jax.Array | int) -> AffineMap:
    """每顆神經元超過真事件數的位置蓋成不作用的映射(a=1, b=0)。

    maps: a、b 形狀 (n, queue_len)。pad 位置的時間、增益可能是任意值,蓋掉之後不影響膜電位。
    n_real_events: 純量或 (n,)。等於 queue_len 時原樣回傳。
    """
    n_out_neurons, queue_len = maps.a.shape
    n_real = normalize_real_events(n_real_events, n_out_neurons)
    real_mask = jnp.arange(queue_len)[None, :] < n_real[:, None]  # (n_out_neurons, queue_len)
    return AffineMap(a=jnp.where(real_mask, maps.a, 1.0),
                      b=jnp.where(real_mask, maps.b, 0.0))


def spike_step_upper_bound(b: jax.Array, v_th: float, chunk_size: int) -> jax.Array:
    """掃描步數上界,證明見 docs/math/掃描步數上界推導.md。

    b: (..., queue_len) 佇列裡每筆事件的 b。
    m* = min(b > 0 的筆數, floor(正的 b 的總和 / v_th)),上界 = m* + ceil((queue_len - m*) / chunk_size)。
    回傳 int32,比 b 少最後一軸。
    """
    queue_len = b.shape[-1]
    positive = jnp.where(b > 0, b, 0.0)
    m = jnp.sum(b > 0, axis=-1)
    energy_bound = jnp.floor(jnp.sum(positive, axis=-1) / v_th)
    m_star = jnp.minimum(m.astype(jnp.float32), energy_bound)
    steps = m_star + jnp.ceil((queue_len - m_star) / chunk_size)
    return steps.astype(jnp.int32)


def base_scan_steps(queue_len: int, chunk_size: int) -> int:
    """不 fire 時掃完整條佇列的步數:每步吃滿 chunk_size 筆,ceil(queue_len / chunk_size)。"""
    return -(-queue_len // chunk_size)


def safe_extra_steps(queue_len: int, chunk_size: int) -> int:
    """一定夠的額外步數:總步數等於佇列長度,每步至少吃一筆。"""
    return queue_len - base_scan_steps(queue_len, chunk_size)


def extra_steps_upper_bound(b: jax.Array, v_th: float, chunk_size: int) -> jax.Array:
    """fire 讓掃描比基本步數多跑的步數上界 = spike_step_upper_bound - base_scan_steps。

    b 同 spike_step_upper_bound。回傳 int32;不 fire 或 chunk_size=1 時是 0。
    """
    return spike_step_upper_bound(b, v_th, chunk_size) - base_scan_steps(b.shape[-1], chunk_size)


class FloatChunkResult(NamedTuple):
    v_final: jax.Array     # chunk 結束後的膜電位,有 fire 時是 reset 之後的值
    is_spiked: jax.Array   # bool scalar,這個 chunk 內是否有 spike
    spike_idx: jax.Array   # 第一次 spike 的事件索引(0-based);沒 spike 則等於 chunk 長度
    v_sequence: jax.Array  # (chunk_size,) 假設都不 reset 算出的逐事件膜電位
    s_sequence: jax.Array  # (chunk_size,) 可微分的 fire 強度,forward 是 0 或 1


def process_chunk(v0: jax.Array, maps: AffineMap, v_th: float,
                   alpha: float = 2.0) -> FloatChunkResult:
    """一個 chunk:先假設都不 fire,用 associative_scan 算出每筆事件後的膜電位,再找第一筆
    v >= v_th 的事件 fire、reset,之後的事件丟掉(下一步從它們重新開始)。

    v0: chunk 開始時的膜電位。
    maps: a、b 形狀 (chunk_size,),依時間排序。
    fire 判斷用 atan_spike;選中哪一筆是離散選擇,不求梯度。
    為什麼 fire 只可能發生在事件那一刻,見 docs/math/單狀態仿射平行掃描推導.md
    「Fire/reset:這裡才是真正的簡化,不是形式上的簡化」。
    """
    composed = jax.lax.associative_scan(combine, maps)
    v_sequence = composed.a * v0 + composed.b

    s_sequence = atan_spike(v_sequence - v_th, alpha)
    spiked_mask = jax.lax.stop_gradient(s_sequence) >= 0.5
    any_spiked = jnp.any(spiked_mask)
    spike_idx = jnp.where(any_spiked, jnp.argmax(spiked_mask), v_sequence.shape[0])

    # soft reset:forward 的 s 是 1,(1-s)*v 等於 0;backward 經過 s 的 surrogate,理由見 docs/問題紀錄.md
    # 「決策:不套閘 + soft reset,不是硬 reset」。沒 fire 時 spike_idx 是 chunk 長度,夾回範圍內才能取值。
    spike_idx_clamped = jnp.minimum(spike_idx, v_sequence.shape[0] - 1)
    v_after_spike = (1.0 - s_sequence[spike_idx_clamped]) * v_sequence[spike_idx_clamped]
    v_silent = v_sequence[-1]
    v_final = jnp.where(any_spiked, v_after_spike, v_silent)

    return FloatChunkResult(v_final=v_final, is_spiked=any_spiked, spike_idx=spike_idx,
                            v_sequence=v_sequence, s_sequence=s_sequence)
