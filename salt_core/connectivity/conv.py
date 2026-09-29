"""conv 層的事件佇列建構,分兩段:結構段(build_conv_structure)只看事件,決定每個空間位置
收哪些事件、每欄的 dt 跟 kernel 位置;數值段用權重算出仿射映射(conv_float_values)或取出整數
權重碼(conv_weight_codes)。

推導見 docs/math/conv事件佇列建構推導.md、docs/math/conv事件佇列壓縮版推導.md。
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from salt_core.float.affine import AffineMap, create_affine_maps


def _axis_candidates(i: jax.Array, K: int, S: int, P: int, N: int,
                      O_max: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    """單軸(y 或 x)每筆事件的 N 個候選輸出位置 o、合不合法、kernel 位置 k。

    i: (n_events,) 事件在這一軸的座標。O_max: 這一軸的輸出尺寸。
    公式見 docs/math/conv事件佇列建構推導.md「連接結構:o,k 公式」。
    回傳 (o, valid, k),形狀都是 (n_events, N)。
    """
    o_min = (i + P - K + 1 + S - 1) // S       # ceil((i+P-K+1)/S)
    upper = (i + P) // S                        # floor((i+P)/S)
    r = jnp.arange(N, dtype=i.dtype)
    o = o_min[:, None] + r[None, :]              # (n_events, N)
    valid = (o <= upper[:, None]) & (o >= 0) & (o < O_max)
    k = i[:, None] - o * S + P                   # (n_events, N)
    return o, valid, k


def unravel_conv_source(event_source_idx: jax.Array, H_in: int,
                         W_in: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    """扁平的來源神經元編號 c*H_in*W_in + y*W_in + x -> (x, y, c)。

    回傳 (x, y, c),形狀都跟 event_source_idx 一樣。
    """
    hw_in = H_in * W_in
    c = event_source_idx // hw_in
    remainder = event_source_idx % hw_in
    y = remainder // W_in
    x = remainder % W_in
    return x, y, c


class ConvQueueStructure(NamedTuple):
    """conv 佇列的結構段,只由事件決定。第一維是空間位置 oy*w_out+ox,跟 oc 無關。"""
    local_to_global_j: jax.Array  # (n_spatial, max_queue_len) int32,局部欄 -> 全域事件 index,空欄是 n_events
    n_real_events: jax.Array      # (n_spatial,) int32,真 tap 數;可能超過 max_queue_len,出界偵測用
    delta_t: jax.Array            # (n_spatial, max_queue_len) float32,見 _delta_t_three_regimes
    tap_c: jax.Array              # (n_spatial, max_queue_len) int32,每欄的 kernel 位置,已夾進合法範圍
    tap_ky: jax.Array             # (n_spatial, max_queue_len) int32
    tap_kx: jax.Array             # (n_spatial, max_queue_len) int32
    n_input_events: jax.Array     # int32 純量,輸入的真事件數


def _compress_candidates(n_flat: jax.Array, j_flat: jax.Array, n_out_spatial: int,
                          max_queue_len: int, n_events: int) -> tuple[jax.Array, jax.Array]:
    """候選清單 -> 每個空間位置自己的佇列。演算法見 docs/math/conv事件佇列壓縮版推導.md
    「演算法:排序 + 分段重置計數」。

    n_flat: (C,) 每個候選的目標空間位置;不合法的候選標成 >= n_out_spatial 的任意值。
    j_flat: (C,) 每個候選的全域事件 index,0 <= j < n_events。
    n_out_spatial: h_out * w_out。
    max_queue_len: 每個空間位置佇列的長度,放不下的候選丟掉。
    n_events: 全域事件數,空欄填這個值。
    回傳 (local_to_global_j, n_real_per_neuron):
        local_to_global_j: (n_out_spatial, max_queue_len) int32,第 r 欄是這個位置第 r 個事件的全域 index。
        n_real_per_neuron: (n_out_spatial,) int32,合法候選數,可能超過 max_queue_len(給出界偵測用)。
    """
    order = jnp.lexsort((j_flat, n_flat))
    sorted_n = n_flat[order]
    sorted_j = j_flat[order]

    C = n_flat.shape[0]
    idx_range = jnp.arange(C)
    is_start = jnp.concatenate([jnp.array([True]), sorted_n[1:] != sorted_n[:-1]])
    start_positions = jnp.where(is_start, idx_range, -1)
    last_start = jax.lax.cummax(start_positions)
    local_rank = idx_range - last_start

    # 不合法、放不下的候選寫到多開的一格(垃圾桶)再切掉,不用 mode='drop'(開 GPU determinism flag 時梯度會算錯),見
    # docs/問題紀錄.md「洞見:JAX/XLA 在 GPU 上的規約運算不保證可重現,連同一顆 seed 都不例外」。
    is_invalid_n = sorted_n >= n_out_spatial
    safe_n = jnp.where(is_invalid_n, n_out_spatial, sorted_n)
    safe_rank = jnp.where((local_rank >= max_queue_len) | is_invalid_n, max_queue_len, local_rank)

    local_to_global_j_padded = jnp.full((n_out_spatial + 1, max_queue_len + 1), n_events, dtype=jnp.int32)
    local_to_global_j_padded = local_to_global_j_padded.at[safe_n, safe_rank].set(sorted_j)
    local_to_global_j = local_to_global_j_padded[:n_out_spatial, :max_queue_len]

    # 段內 local_rank+1 的最大值就是合法候選數(scatter 不保證順序,所以取 max)。
    # 超過 max_queue_len 的也要算進去,出界偵測靠這個數。
    n_real_per_neuron_padded = jnp.zeros((n_out_spatial + 1,), dtype=jnp.int32)
    real_local_rank = jnp.where(is_invalid_n, 0, local_rank + 1)
    n_real_per_neuron_padded = n_real_per_neuron_padded.at[safe_n].max(real_local_rank)
    n_real_per_neuron = n_real_per_neuron_padded[:n_out_spatial]

    return local_to_global_j, n_real_per_neuron


def _delta_t_three_regimes(t_gathered: jax.Array, n_real_per_neuron: jax.Array,
                           global_last_time: jax.Array) -> jax.Array:
    """佇列每一欄的 dt,三段規則(推導見 docs/math/conv事件佇列壓縮版推導.md「修正規則」):
    真事件的欄是跟前一筆的差(第一筆跟 t=0 比);真事件之後的第一欄(catch-up)是
    global_last_time - 這個位置最後一筆事件的時間;其餘是 0。

    t_gathered: (n_out, max_queue_len) 每欄對應的事件時間。
    n_real_per_neuron: (n_out,) 真事件數。
    global_last_time: 純量,最後一筆真輸入事件的時間。
    回傳 (n_out, max_queue_len) float。
    """
    n_out, max_queue_len = t_gathered.shape
    col_idx = jnp.arange(max_queue_len, dtype=jnp.int32)[None, :]
    n_real = n_real_per_neuron[:, None]
    is_real = col_idx < n_real
    is_catchup = col_idx == n_real

    delta_t_real = jnp.diff(t_gathered, axis=1,
                            prepend=jnp.zeros((n_out, 1), dtype=t_gathered.dtype))
    last_real_col = jnp.clip(n_real_per_neuron - 1, 0, max_queue_len - 1)
    t_last_real = jnp.take_along_axis(t_gathered, last_real_col[:, None], axis=1)[:, 0]
    delta_t_catchup = global_last_time - t_last_real

    return jnp.where(is_real, delta_t_real, jnp.where(is_catchup, delta_t_catchup[:, None], 0.0))


def build_conv_structure(event_times: jax.Array, x: jax.Array, y: jax.Array, c: jax.Array,
                         k: int, s: int, p: int, h_out: int, w_out: int, max_queue_len: int,
                         n_real_events: jax.Array | int) -> ConvQueueStructure:
    """conv 佇列的結構段:每個空間位置收哪些事件、每欄的 Δt 跟 kernel 位置。

    event_times, x, y, c: (n_events,) 已排序的事件時間(整數 ms)跟座標。
    k, s, p: kernel 大小、stride、padding。h_out, w_out: 輸出面尺寸。
    max_queue_len: 每個空間位置的佇列長度,放不下的事件丟掉,n_real_events 照實回報。
    n_real_events: 前幾筆是真事件,其餘是 pad,不進任何佇列。
    """
    event_times = jnp.asarray(event_times, dtype=jnp.float32)
    x = jnp.asarray(x, dtype=jnp.int32)
    y = jnp.asarray(y, dtype=jnp.int32)
    c = jnp.asarray(c, dtype=jnp.int32)
    n_events = event_times.shape[0]
    n_input_events = jnp.asarray(n_real_events, dtype=jnp.int32)
    n_candidates = (k - 1) // s + 1
    n_spatial = h_out * w_out

    o_y, valid_y, _ = _axis_candidates(y, k, s, p, n_candidates, h_out)  # (n_events, N)
    o_x, valid_x, _ = _axis_candidates(x, k, s, p, n_candidates, w_out)  # (n_events, N)
    is_real_event = jnp.arange(n_events) < n_input_events
    valid_2d = valid_y[:, :, None] & valid_x[:, None, :] & is_real_event[:, None, None]

    n_2d = o_y[:, :, None] * w_out + o_x[:, None, :]  # (n_events, N, N) 空間位置 id
    j_2d = jnp.broadcast_to(jnp.arange(n_events, dtype=jnp.int32)[:, None, None],
                            (n_events, n_candidates, n_candidates))
    n_flat = jnp.where(valid_2d, n_2d, n_spatial).reshape(-1)
    local_to_global_j, n_real_per_position = _compress_candidates(
        n_flat, j_2d.reshape(-1), n_spatial, max_queue_len, n_events)

    # 空欄夾到最後一筆真事件,不能夾到陣列最後一格:那可能是時間極大的 pad,Δt 會溢位。
    event_j = jnp.minimum(local_to_global_j, n_input_events - 1)
    delta_t = _delta_t_three_regimes(event_times[event_j], n_real_per_position,
                                    event_times[n_input_events - 1])

    oy_grid = (jnp.arange(n_spatial, dtype=jnp.int32) // w_out)[:, None]
    ox_grid = (jnp.arange(n_spatial, dtype=jnp.int32) % w_out)[:, None]
    # 非真 tap 的欄位 kernel 位置可能越界(負 index 會 wraparound),夾進合法範圍;這些欄位的 b 是 0。
    tap_ky = jnp.clip(y[event_j] - oy_grid * s + p, 0, k - 1)
    tap_kx = jnp.clip(x[event_j] - ox_grid * s + p, 0, k - 1)
    return ConvQueueStructure(local_to_global_j=local_to_global_j,
                              n_real_events=n_real_per_position, delta_t=delta_t,
                              tap_c=c[event_j], tap_ky=tap_ky, tap_kx=tap_kx,
                              n_input_events=n_input_events)


def _gather_taps(structure: ConvQueueStructure, w: jax.Array) -> jax.Array:
    """每個 (oc, 空間位置, 欄) 的 kernel 位置對應的權重,(oc, n_spatial, max_queue_len),dtype 同 w。"""
    return jax.vmap(
        lambda oc_w: oc_w[structure.tap_c, structure.tap_ky, structure.tap_kx])(w)


def _real_tap_mask(structure: ConvQueueStructure) -> jax.Array:
    """(n_spatial, max_queue_len) bool,真 tap 的欄位是 True。"""
    max_queue_len = structure.delta_t.shape[1]
    return jnp.arange(max_queue_len)[None, :] < structure.n_real_events[:, None]


def conv_float_values(structure: ConvQueueStructure, w: jax.Array, tau: float,
                      event_gain: jax.Array | None) -> AffineMap:
    """conv 佇列的浮點數值段:a = (1 - 1/tau) ** delta_t,b = 權重 * event_gain,非真 tap 的 b 是 0。

    w: (oc, ic, k, k) 權重。
    event_gain: (n_events,) 乘進權重的增益。接在上一層後面時傳上一層的 s_spike,
        理由見 docs/問題紀錄.md。None 等於全 1。
    回傳 AffineMap,a、b 形狀 (oc*n_spatial, max_queue_len),神經元編號 = oc*n_spatial + 空間位置。
    """
    oc = w.shape[0]
    n_spatial, max_queue_len = structure.delta_t.shape
    weight_vals = _gather_taps(structure, w)  # (oc, n_spatial, max_queue_len)
    if event_gain is not None:
        event_j = jnp.minimum(structure.local_to_global_j, structure.n_input_events - 1)
        gain = jnp.asarray(event_gain, dtype=weight_vals.dtype)[event_j]
        weight_vals = weight_vals * gain[None, :, :]
    b = jnp.where(_real_tap_mask(structure)[None, :, :], weight_vals, 0.0).reshape(
        oc * n_spatial, max_queue_len)
    maps = create_affine_maps(structure.delta_t, b, tau)
    return AffineMap(a=tile_channels(maps.a, oc), b=maps.b)


def conv_weight_codes(structure: ConvQueueStructure, q: jax.Array) -> jax.Array:
    """conv 佇列的整數數值段:每欄的整數權重碼,非真 tap 是 0。

    q: (oc, ic, k, k) 整數權重碼。
    回傳 int32,形狀 (oc*n_spatial, max_queue_len),神經元編號同 conv_float_values。
    """
    oc = q.shape[0]
    n_spatial, max_queue_len = structure.delta_t.shape
    codes = jnp.where(_real_tap_mask(structure)[None, :, :], _gather_taps(structure, q), 0)
    return codes.astype(jnp.int32).reshape(oc * n_spatial, max_queue_len)


def tile_channels(values: jax.Array, oc: int) -> jax.Array:
    """逐空間位置的值 (n_spatial, ...) 展開成逐神經元 (oc*n_spatial, ...),每個 channel 一份。"""
    n_spatial = values.shape[0]
    return jnp.broadcast_to(values[None], (oc, *values.shape)).reshape(
        oc * n_spatial, *values.shape[1:])
