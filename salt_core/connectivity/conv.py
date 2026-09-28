"""Conv 層的事件佇列建構,分兩段:結構段(build_conv_structure)只看事件,
決定每個空間位置收哪些事件、Δt、kernel 位置;數值段用權重算出仿射映射
(conv_float_values)或取出整數權重碼(conv_weight_codes)。推導見 docs/math/conv事件佇列建構推導.md、
docs/math/conv事件佇列壓縮版推導.md。
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from salt_core.float.affine import AffineMap, create_affine_maps


def _axis_candidates(i: jax.Array, K: int, S: int, P: int, N: int,
                      O_max: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    """單軸(y 或 x 共用同一條公式)算出 N 個候選輸出位置、合法性、tap。
    對應推導文件第 1、4.1 節。全程整數運算,不經過浮點數 ceil/floor
    (第 1 節已證明 (n+S-1)//S 這條式子對正負分子都成立,不需要分支)。

    回傳 (o, valid, k),shape 都是 (n_events, N)。
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
    """扁平 neuron id -> (x,y,c) 三元組,對應推導文件第 9.2 節(第 6 節攤平
    公式的反運算)。上一層的 OC_in 會自動變成這一層的 IC,不用額外傳。

    只在 conv 接 conv 時需要呼叫(上一層 stream.extract_output_events_fc
    吐出來的是扁平 id);第一層例外——資料端原生就給 (x,y,c),不經過這個
    函式(第 9.3 節)。

    回傳 (x,y,c),shape 都跟 event_source_idx 一樣。
    """
    hw_in = H_in * W_in
    c = event_source_idx // hw_in
    remainder = event_source_idx % hw_in
    y = remainder // W_in
    x = remainder % W_in
    return x, y, c


class ConvQueueStructure(NamedTuple):
    """conv 佇列的結構段,只由事件決定。第一維是空間位置 oy*w_out+ox,跟 oc 無關;
    L = max_queue_len。"""
    local_to_global_j: jax.Array  # (n_spatial, L) int32,局部欄 -> 全域事件 index,空欄是 n_events
    n_real_events: jax.Array      # (n_spatial,) int32,真 tap 數;可能超過 L,出界偵測用
    delta_t: jax.Array            # (n_spatial, L) float32,真 tap / catch-up / identity 三段規則
    tap_c: jax.Array              # (n_spatial, L) int32,每欄的 kernel 位置,已夾進合法範圍
    tap_ky: jax.Array             # (n_spatial, L) int32
    tap_kx: jax.Array             # (n_spatial, L) int32
    n_input_events: jax.Array     # int32 純量,輸入的真事件數


def _compress_candidates(n_flat: jax.Array, j_flat: jax.Array, n_out_spatial: int,
                          max_queue_len: int, n_events: int) -> tuple[jax.Array, jax.Array]:
    """候選清單 -> 每顆(空間)神經元自己的壓縮佇列。對應推導文件第 2 節。

    n_flat: shape (C,),每個候選的目標神經元 id——只含空間位置 $(o_y,o_x)$
      攤平後的 id,不含 oc(第 1 節:感受野篩選跟 oc 無關,同一個空間位置
      的 OC 個神經元共用同一份篩選結果,呼叫端只需要對 H_out*W_out 做一次,
      不用對 OC*H_out*W_out 各做一次)。不合法的候選,呼叫端要先把 n 改標記
      成 >= n_out_spatial 的保證越界值(哪個值都可以)。
    j_flat: shape (C,),每個候選對應的全域事件 index,值域 [0, n_events)。
    n_out_spatial: H_out*W_out(不含 oc)。
    max_queue_len: 文件記法 $L$,壓縮後每顆神經元佇列的固定長度上限。
    n_events: 全域事件總數,只用來選一個不會跟真實 j 值搞混的 sentinel。

    排序依 (n,j) 為主/次鍵:全域事件列表本身已經照時間排序,j 本身就是時間
    先後順序,不需要另外排序時間(第 2 節)。`jnp.lexsort` 的慣例是「最後一個
    key 是主鍵」,所以呼叫時 n_flat 要放在 tuple 最後面。

    局部 rank 用「分段重置計數」算(第 2 節):is_start 標記每一段(同一個 n
    的連續區間)的起點,對 is_start 出現的位置索引做累進最大值(cummax,跟
    float.affine.combine 的 associative_scan 是同一類運算),每個位置減掉「目前這段
    的起點」就是段內的局部 rank(0-based)。

    回傳 (local_to_global_j, n_real_per_neuron):
      local_to_global_j: shape (n_out_spatial, max_queue_len) int32。第
        (n, local_rank) 格是神經元 n 第 local_rank 個相關事件的全域 index;
        沒被寫到的格子填 n_events(值域外的 sentinel,呼叫端拿它去 gather
        event_times/x/y/c 時,JAX 的 clip 模式會夾到最後一個真實事件,不會
        crash/NaN——但這些位置本來就會被 n_real_per_neuron 蓋成 catch-up/
        identity,夾到什麼值不影響最終結果,見 build_conv_structure)。
      n_real_per_neuron: shape (n_out_spatial,) int32,神經元真正收到的
        合法 tap 數(不含 catch-up、不含 identity)。

    $L$ 太小、真的放不下全部候選的情況(第 7.2 節「L 出界」問題):不合法候選
    跟溢出候選,都用底下「垃圾桶」機制丟棄,不會 crash;`n_real_per_neuron`
    如實回報真正的候選數(可能超過 `max_queue_len`),主動偵測「是不是真的
    丟過東西」是第 7.2 節動態 L 修正機制的責任,不是這個函式(第 1 階段)要
    做的事。

    **不用 `mode='drop'`,改用「垃圾桶」(2026-09-18,問題紀錄第八節)**:
    `mode='drop'` 讓 scatter 的 index 陣列真的帶越界值,這個組合(多個邏輯
    獨立樣本合併進同一次呼叫 + scatter 的 index 真的越界)會踩到 XLA 一個
    GPU determinism 相關的 codegen bug(`--xla_gpu_deterministic_ops=true`
    開著、外層 batch `vmap` 時,梯度會算錯——forward 不受影響,純粹是
    backward 的 scatter-add 出錯,細節見 `docs/問題紀錄.md` 第八節、
    `archive/xla_repro/`)。改法:scatter 目標陣列的兩個維度都多開一格當「垃圾桶」
    (`n_out_spatial+1`、`max_queue_len+1`),不合法/溢出的候選全部指去
    垃圾桶座標——保證是合法範圍內的 index,scatter 從頭到尾不需要真的丟棄
    任何一次寫入;事後把垃圾桶那一整格切掉,效果跟原本完全一樣。跟
    `mode='drop'` 版本逐位元等價,已用多組測資(一般情況、全部合法、全部
    不合法、L 溢出、空清單)驗證過,見 `archive/xla_repro/verify_trash_row_equivalence.py`。
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

    # 垃圾桶座標:n_out_spatial(對應「候選不合法」的既有 sentinel 慣例)、
    # max_queue_len(候選溢出 L 時的落點)——兩者都保證落在 padded 陣列的
    # 合法範圍內,scatter 不需要 mode='drop'。
    is_invalid_n = sorted_n >= n_out_spatial
    safe_n = jnp.where(is_invalid_n, n_out_spatial, sorted_n)
    safe_rank = jnp.where((local_rank >= max_queue_len) | is_invalid_n, max_queue_len, local_rank)

    local_to_global_j_padded = jnp.full((n_out_spatial + 1, max_queue_len + 1), n_events, dtype=jnp.int32)
    local_to_global_j_padded = local_to_global_j_padded.at[safe_n, safe_rank].set(sorted_j)
    local_to_global_j = local_to_global_j_padded[:n_out_spatial, :max_queue_len]

    # 對每個 n scatter-max(local_rank+1):同一段內 local_rank 嚴格遞增
    # 0,1,...,count-1,段內最後一筆的 local_rank+1 剛好等於這段的合法候選數
    # (=這個神經元的 n_real_events),用 max 而不是取最後一筆,是因為 scatter
    # 不保證處理順序,但這裡任一筆的 local_rank+1 都 <= count,取 max 恆等於
    # count,不用依賴處理順序。**這個 scatter 只看 n 合不合法,跟
    # local_rank 有沒有超過 max_queue_len 完全無關**——local_rank+1 就算
    # 超過 max_queue_len 也要照樣參與 max,這樣下游才能靠這個數字偵測「真的
    # 需要比 L 更大的容量」,不能沿用上面 local_to_global_j 那個溢出判斷。
    n_real_per_neuron_padded = jnp.zeros((n_out_spatial + 1,), dtype=jnp.int32)
    real_local_rank = jnp.where(is_invalid_n, 0, local_rank + 1)
    n_real_per_neuron_padded = n_real_per_neuron_padded.at[safe_n].max(real_local_rank)
    n_real_per_neuron = n_real_per_neuron_padded[:n_out_spatial]

    return local_to_global_j, n_real_per_neuron


def _delta_t_three_regimes(t_gathered: jax.Array, n_real_per_neuron: jax.Array,
                           global_last_time: jax.Array) -> jax.Array:
    """壓縮版佇列每一欄的 Δt(推導文件第 4.3、4.4 節)。

    t_gathered: shape (n_out, L),第 (n, col) 格是 local_to_global_j[n,col]
      對應的全域事件時間(呼叫端用 event_times[local_to_global_j] 算好再傳
      進來,這個函式不知道、也不需要知道 local_to_global_j 本身)。
    n_real_per_neuron: shape (n_out,),見 `_compress_candidates`。
    global_last_time: 純量,「全域最後一筆事件的時間」(第 4.3 節)——這裡
      刻意讓呼叫端算好傳進來,因為呼叫端可能還要處理 n_real_events(pad
      事件)的情況,「全域最後一筆」在那種情況下指的是最後一筆真事件,不是
      陣列最後一格,這個函式本身不處理 pad,只認呼叫端給的這個值。

    三段規則:
      col < n_real_per_neuron[n]   (真 tap):自己這條子序列裡跟前一筆的差,
        col=0 跟 t=0 比(問題紀錄第七節同一個基準)
      col == n_real_per_neuron[n]  (catch-up):全域最後一筆時間 - 這個神經元
        自己最後一筆相關事件的時間
      col > n_real_per_neuron[n]   (identity):Δt 定義成 0

    回傳浮點 delta_t,shape (n_out, L)。
    """
    n_out, L = t_gathered.shape
    col_idx = jnp.arange(L, dtype=jnp.int32)[None, :]
    n_real = n_real_per_neuron[:, None]
    is_real = col_idx < n_real
    is_catchup = col_idx == n_real

    delta_t_real = jnp.diff(t_gathered, axis=1,
                            prepend=jnp.zeros((n_out, 1), dtype=t_gathered.dtype))
    last_real_col = jnp.clip(n_real_per_neuron - 1, 0, L - 1)
    t_last_real = jnp.take_along_axis(t_gathered, last_real_col[:, None], axis=1)[:, 0]
    delta_t_catchup = global_last_time - t_last_real

    return jnp.where(is_real, delta_t_real, jnp.where(is_catchup, delta_t_catchup[:, None], 0.0))


def build_conv_structure(event_times: jax.Array, x: jax.Array, y: jax.Array, c: jax.Array,
                         k: int, s: int, p: int, h_out: int, w_out: int, max_queue_len: int,
                         n_real_events: jax.Array | int) -> ConvQueueStructure:
    """conv 佇列的結構段:每個空間位置收哪些事件、每欄的 Δt 跟 kernel 位置。

    event_times, x, y, c: (n_events,) 已排序的事件時間(整數 ms)跟座標。
    k, s, p: kernel 大小、stride、padding。h_out, w_out: 輸出面尺寸。
    max_queue_len: 每個空間位置的佇列長度 L,放不下的事件丟掉,n_real_events 照實回報。
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
    """每個 (oc, 空間位置, 欄) 的 kernel 位置對應的權重,(oc, n_spatial, L),dtype 同 w。"""
    return jax.vmap(
        lambda oc_w: oc_w[structure.tap_c, structure.tap_ky, structure.tap_kx])(w)


def _real_tap_mask(structure: ConvQueueStructure) -> jax.Array:
    """(n_spatial, L) bool,真 tap 的欄位是 True。"""
    max_queue_len = structure.delta_t.shape[1]
    return jnp.arange(max_queue_len)[None, :] < structure.n_real_events[:, None]


def conv_float_values(structure: ConvQueueStructure, w: jax.Array, tau: float,
                      event_gain: jax.Array | None) -> AffineMap:
    """conv 佇列的浮點數值段:a = (1 - 1/tau) ** delta_t,b = 權重 * event_gain,非真 tap 的 b 是 0。

    w: (oc, ic, k, k) 權重。
    event_gain: (n_events,) 乘進權重的增益。接在上一層後面時傳上一層的 s_spike,
        理由見 docs/問題紀錄.md。None 等於全 1。
    回傳 AffineMap,a、b 形狀 (oc*n_spatial, L),神經元編號 = oc*n_spatial + 空間位置。
    """
    oc = w.shape[0]
    n_spatial, max_queue_len = structure.delta_t.shape
    weight_vals = _gather_taps(structure, w)  # (oc, n_spatial, L)
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
    回傳 int32,形狀 (oc*n_spatial, L),神經元編號同 conv_float_values。
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
