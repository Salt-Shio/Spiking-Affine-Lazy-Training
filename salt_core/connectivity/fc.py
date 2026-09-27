"""FC 層的事件佇列建構,分兩段:結構段(build_fc_structure)只看事件,算 Δt;
數值段用權重算出仿射映射(fc_float_values)或取出整數權重碼(fc_weight_codes)。
推導見 docs/math/全連接forward訓練範例.md。
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from salt_core.core import AffineMap, create_affine_maps, mask_pad_events


class FCQueueStructure(NamedTuple):
    """FC 佇列的結構段,只由事件決定。每顆輸出神經元看到同一串事件,所以沒有神經元軸。"""
    delta_t: jax.Array        # (n_events,) float32,跟前一筆事件的間隔,第一筆跟 t=0 比;pad 位置是 0
    source_idx: jax.Array     # (n_events,) 每筆事件的來源神經元,對應 w 的欄
    n_real_events: jax.Array  # int32 純量,前幾筆是真事件


def build_fc_structure(event_times: jax.Array, event_source_idx: jax.Array,
                       n_real_events: jax.Array | int) -> FCQueueStructure:
    """FC 佇列的結構段。

    event_times: (n_events,) 已排序的事件時間,整數 ms。
    event_source_idx: (n_events,) 每筆事件的來源神經元。
    n_real_events: 純量,前幾筆是真事件,其餘位置的 Δt 是 0。
    """
    event_times = jnp.asarray(event_times, dtype=jnp.float32)
    n_real = jnp.asarray(n_real_events, dtype=jnp.int32)
    gap_ms = jnp.diff(event_times, prepend=jnp.zeros(1, dtype=event_times.dtype))
    is_real = jnp.arange(event_times.shape[0]) < n_real
    return FCQueueStructure(delta_t=jnp.where(is_real, gap_ms, 0.0),
                            source_idx=jnp.asarray(event_source_idx), n_real_events=n_real)


def fc_float_values(structure: FCQueueStructure, w: jax.Array, tau: float,
                    event_gain: jax.Array | None) -> AffineMap:
    """FC 佇列的浮點數值段:a = (1 - 1/tau) ** delta_t,b = w[:, 來源] * event_gain。

    w: (n_out, n_in) 權重,w[i, j] 是 j -> i 的權重。
    event_gain: (n_events,) 乘進權重的增益。接在上一層後面時傳上一層的 s_spike,
        理由見 docs/問題紀錄.md。None 等於全 1。
    回傳 AffineMap,a、b 形狀 (n_out, n_events);pad 位置是 a=1、b=0。
    """
    weights = w[:, structure.source_idx]
    if event_gain is not None:
        weights = weights * jnp.asarray(event_gain, dtype=weights.dtype)[None, :]
    maps = create_affine_maps(structure.delta_t, weights, tau)
    maps = AffineMap(a=jnp.broadcast_to(maps.a[None, :], maps.b.shape), b=maps.b)
    return mask_pad_events(maps, structure.n_real_events)


def fc_weight_codes(structure: FCQueueStructure, q: jax.Array) -> jax.Array:
    """FC 佇列的整數數值段:每筆事件的整數權重碼 q[:, 來源],pad 位置是 0。

    q: (n_out, n_in) 整數權重碼。
    回傳 int32,形狀 (n_out, n_events)。
    """
    is_real = jnp.arange(structure.delta_t.shape[0]) < structure.n_real_events
    return jnp.where(is_real[None, :], q[:, structure.source_idx], 0).astype(jnp.int32)
