"""Conv 層的事件佇列建構:全域事件包(座標三元組 (x,y,c) + 時間)+ 權重張量
-> 每顆神經元自己的壓縮仿射映射佇列(`build_conv_queue_compressed`)。對應推導
見 docs/math/conv事件佇列建構推導.md + docs/math/conv事件佇列壓縮版推導.md。

跟 connectivity/fc.py 的關係:fc.py 的 build_fc_queue 靠一維 event_source_idx
直接當 W 的 column;conv 的權重要看事件座標相對輸出位置的偏移(kernel tap),
一維 index 不夠用,事件要換成 (x,y,c) 三元組(推導文件第 0、2 節)。

歷史:曾有一個「密集版」`build_conv_queue`(陣列寬度 = 全域事件數,不相關的
格子只衰減不加權重),step 4d 移除——production 端全部改用壓縮版,密集版只
剩測試在用,而測試改用 `salt_core/tests/_reference.py` 的透明 numpy 參考當
對照(比密集版更獨立,連避免越界 index 的手法都不一樣)。`_axis_candidates`
(單軸候選幾何)兩版共用,留著。
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from salt_core.core import AffineMap, create_affine_maps


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

    只在 conv 接 conv 時需要呼叫(上一層 layer_chain.extract_output_events
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


def receptive_field_tap_count(x: jax.Array, y: jax.Array, S: int, P: int,
                               H_out: int, W_out: int, K: int,
                               n_real_events: jax.Array | int) -> jax.Array:
    """每個空間輸出位置有幾筆「真事件」是它的合法 tap。純幾何——只看事件座標、
    K/S/P、n_real_events,不看 channel / 權重 / tau(候選合不合法只看座標,見
    docs/math/conv事件佇列建構推導.md 第 1、7 節)。回傳 shape (H_out*W_out,) int32,
    OC 軸不影響空間合法性,所以只算空間、不含 oc。`n_real_events` 必填:沒有
    padding 就傳事件總數(等於「所有事件都可能是合法 tap」)。

    init_k 掃描的 receptive-field 正規化 firing rate、$L_1$ 佇列長度量測都用這個。
    之前是拿全 1 權重跑一次 `build_conv_queue`、數 `b != 0` 反推——把重量級的
    forward 建構器當幾何 oracle。
    """
    x = jnp.asarray(x, dtype=jnp.int32)
    y = jnp.asarray(y, dtype=jnp.int32)
    n_events = x.shape[0]
    N = (K - 1) // S + 1
    o_y, valid_y, _ = _axis_candidates(y, K, S, P, N, H_out)  # (n_events, N)
    o_x, valid_x, _ = _axis_candidates(x, K, S, P, N, W_out)  # (n_events, N)
    valid_2d = valid_y[:, :, None] & valid_x[:, None, :]        # (n_events, N, N)
    is_real = jnp.arange(n_events) < jnp.asarray(n_real_events, dtype=jnp.int32)
    valid_2d = valid_2d & is_real[:, None, None]

    o_flat = o_y[:, :, None] * W_out + o_x[:, None, :]          # (n_events, N, N)
    o_flat = jnp.where(valid_2d, o_flat, H_out * W_out).reshape(-1)  # 不合法標成越界
    counts = jnp.zeros((H_out * W_out,), dtype=jnp.int32)
    return counts.at[o_flat].add(valid_2d.reshape(-1).astype(jnp.int32), mode='drop')


def conv_layer_receptive_field_firing_rate(spike_mask: jax.Array, x: jax.Array, y: jax.Array,
                                            S: int, P: int, H_out: int, W_out: int, K: int,
                                            OC: int, n_real_events: jax.Array | int
                                            ) -> jax.Array:
    """給定一次 forward 的 spike_mask,算「每個神經元 spike 數 / 自己真正的
    感受野事件數」,只對感受野事件數 > 0 的神經元取平均(感受野事件數 0 代表
    這個神經元這個樣本完全沒機會 fire,不是「fire 比例是 0」,排除掉才不會把
    平均往下拉)。回傳純量。

    spike_mask: shape (OC*H_out*W_out, max_steps)。感受野事件數是純空間量
    (`receptive_field_tap_count`),tile 到 OC 個 channel 後跟 spike_mask 的
    神經元軸對齊。
    """
    spatial = receptive_field_tap_count(x, y, S, P, H_out, W_out, K, n_real_events)
    opportunity = jnp.tile(spatial, OC)  # (OC*H_out*W_out,)
    spike_count = jnp.sum(spike_mask, axis=1)
    has_opp = opportunity > 0
    rate_per_neuron = jnp.where(has_opp, spike_count / jnp.maximum(opportunity, 1), 0.0)
    return jnp.sum(rate_per_neuron) / jnp.maximum(jnp.sum(has_opp), 1)



# ============================================================================
# 壓縮版(任務 7 第二階段):每顆神經元只留自己的相關事件,不是全域事件數。
# 對應推導見 docs/math/conv事件佇列壓縮版推導.md 全文,這裡的每個區塊都對應
# 該文件的章節,跟上面密集版的關係、為什麼需要壓縮版,見該文件開頭。
#
# 段 1-3(壓縮佇列核心、run_layer_forward 呼叫端傳陣列、extract_output_events
# 查表)已完成。event_gain 支援(段 1 當時刻意排除,因為 conv1 直接吃原始事件、
# 沒有上游層)在段 4 補上——`ConvNetCompressed` 的 conv2 需要它,跟密集版
# `build_conv_queue` 對稱:上一層的 s_spike 乘進權重,重新接通跨層梯度路徑。
# ============================================================================


class CompressedConvQueue(NamedTuple):
    """壓縮版 build_conv_queue 的回傳型別,對應推導文件第 5.1 節列的三樣
    東西。三個欄位 shape 的第一維都是 OC*H_out*W_out,跟密集版的
    n_out_neurons 是同一個量,方便之後 run_layer_forward/extract_output_events
    (第 2、3 階段)當成密集版的直接替代品接上去。
    """
    maps: AffineMap              # a/b shape (OC*H_out*W_out, L)
    local_to_global_j: jax.Array  # shape (OC*H_out*W_out, L) int32,(神經元,局部欄)-> 全域事件 index
    n_real_events: jax.Array     # shape (OC*H_out*W_out,) int32,只算真正的 tap,不含 catch-up/identity


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
    core.combine 的 associative_scan 是同一類運算),每個位置減掉「目前這段
    的起點」就是段內的局部 rank(0-based)。

    回傳 (local_to_global_j, n_real_per_neuron):
      local_to_global_j: shape (n_out_spatial, max_queue_len) int32。第
        (n, local_rank) 格是神經元 n 第 local_rank 個相關事件的全域 index;
        沒被寫到的格子填 n_events(值域外的 sentinel,呼叫端拿它去 gather
        event_times/x/y/c 時,JAX 的 clip 模式會夾到最後一個真實事件,不會
        crash/NaN——但這些位置本來就會被 n_real_per_neuron 蓋成 catch-up/
        identity,夾到什麼值不影響最終結果,見主函式)。
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
    `xla_repro/`)。改法:scatter 目標陣列的兩個維度都多開一格當「垃圾桶」
    (`n_out_spatial+1`、`max_queue_len+1`),不合法/溢出的候選全部指去
    垃圾桶座標——保證是合法範圍內的 index,scatter 從頭到尾不需要真的丟棄
    任何一次寫入;事後把垃圾桶那一整格切掉,效果跟原本完全一樣。跟
    `mode='drop'` 版本逐位元等價,已用多組測資(一般情況、全部合法、全部
    不合法、L 溢出、空清單)驗證過,見 `xla_repro/verify_trash_row_equivalence.py`。
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


def _affine_with_catchup(t_gathered: jax.Array, n_real_per_neuron: jax.Array,
                          global_last_time: jax.Array, tau: float,
                          b_real: jax.Array) -> AffineMap:
    """局部 Δt + 補位規則。對應推導文件第 3、4 節,是壓縮版跟密集版唯一的
    數值差異來源:密集版的 a 全域算一次、所有神經元共用;壓縮版每一列要用
    自己篩選後的子序列重算 Δt,而且補位不能直接補 identity(第 4.1、4.2
    節已經用反例證明過,直接補 identity 會跟密集版算出不同的 v_final)。

    t_gathered: shape (n_out, L),第 (n, col) 格是 local_to_global_j[n,col]
      對應的全域事件時間(呼叫端用 event_times[local_to_global_j] 算好再傳
      進來,這個函式不知道、也不需要知道 local_to_global_j 本身)。
    n_real_per_neuron: shape (n_out,),見 `_compress_candidates`。
    global_last_time: 純量,「全域最後一筆事件的時間」(第 4.3 節)——這裡
      刻意讓呼叫端算好傳進來,因為呼叫端可能還要處理 n_real_events(pad
      事件)的情況,「全域最後一筆」在那種情況下指的是最後一筆真事件,不是
      陣列最後一格,這個函式本身不處理 pad,只認呼叫端給的這個值。
    b_real: shape (n_out, L),第 (n, col) 格是真的 tap 才有意義的權重值
      (通常是 W[oc, c_gathered, k_y_gathered, k_x_gathered] gather 出來的),
      非真 tap 的位置數值不重要,這個函式會用 is_real mask 蓋掉。

    回傳 AffineMap,a/b shape 都是 (n_out, L)。

    三段規則(第 4.3、4.4 節):
      col < n_real_per_neuron[n]        (真 tap):Δt 用自己這條子序列的
        diff(col=0 是「跟 t=0 的差」,問題紀錄第七節同一個基準),b 用真權重
      col == n_real_per_neuron[n]       (catch-up):Δt = 全域最後一筆時間 -
        這個神經元自己最後一筆相關事件的時間,b=0(純衰減)
      col > n_real_per_neuron[n]        (identity):a=1, b=0

    n_real_per_neuron[n]==max_queue_len(剛好收滿,見第 1 階段測試重點)時,
    col 的值域 [0,L) 永遠不會等於 n_real_per_neuron[n](=L),自然沒有
    catch-up/identity 格,不需要另外特判。n_real_per_neuron[n]==0(這個神經元
    完全沒有相關事件)時,catch-up 格的 Δt 算出來的值沒有實際意義,但因為
    b 全部是 0,電壓從頭到尾停在 0,不管 a 是多少都不影響 v_final(這是純
    衰減不會讓電壓憑空出現的引理,單狀態仿射平行掃描推導第 4 節),不需要
    特判成別的分支。
    """
    n_out, L = t_gathered.shape
    col_idx = jnp.arange(L, dtype=jnp.int32)[None, :]
    n_real = n_real_per_neuron[:, None]
    is_real = col_idx < n_real
    is_catchup = col_idx == n_real

    delta_t_real = jnp.diff(t_gathered, axis=1,
                             prepend=jnp.zeros((n_out, 1), dtype=t_gathered.dtype))
    naive = create_affine_maps(delta_t_real, b_real, tau)

    last_real_col = jnp.clip(n_real_per_neuron - 1, 0, L - 1)
    t_last_real = jnp.take_along_axis(t_gathered, last_real_col[:, None], axis=1)[:, 0]
    delta_t_catchup = global_last_time - t_last_real
    a_catchup = (1.0 - 1.0 / tau) ** delta_t_catchup

    a = jnp.where(is_real, naive.a, jnp.where(is_catchup, a_catchup[:, None], 1.0))
    b = jnp.where(is_real, naive.b, 0.0)
    return AffineMap(a=a, b=b)


def build_conv_queue_compressed(event_times: jax.Array, x: jax.Array, y: jax.Array,
                                 c: jax.Array, W: jax.Array, tau: float, S: int, P: int,
                                 H_out: int, W_out: int, max_queue_len: int,
                                 n_real_events: jax.Array | int,
                                 event_gain: jax.Array | None = None
                                 ) -> CompressedConvQueue:
    """`build_conv_queue`(密集版)的壓縮版本。參數意義跟密集版完全相同,
    只多一個 `max_queue_len`(文件記法 $L$,見推導文件第 7 節怎麼決定這個
    數字,這個函式不管這件事,只管給定 $L$ 之後怎麼建構佇列)。

    event_gain: 跟密集版 `build_conv_queue` / `build_fc_queue` 意義完全相同——
      上一層 chunk_scan 回傳的 s_spike,可微分增益,預設 None(等同全 1)。
      密集版對「事件軸」整批乘;壓縮版每個 (神經元, 局部欄) 對應一個全域
      事件 `local_to_global_j[n,col]`,用同一個 `safe_j` gather 出對應的 gain
      再乘進 `weight_vals`。非真 tap 位置的 `weight_vals` 本來就無意義,乘完
      照樣被 `_affine_with_catchup` 的 is_real mask 蓋成 b=0,跟密集版「沒
      寫入的位置保持 0」是同一個保證。`ConvNetCompressed` 的 conv2 靠這個
      參數把 conv1 自己的 s_spike 乘進來,重新接通對 conv1 權重的梯度路徑。

    跟密集版的關係:第 1-4 步(候選產生:_axis_candidates、座標合法性、
    pad 事件過濾)完全共用密集版同一套邏輯,不重新實作;差異只在密集版把
    候選直接 scatter 進「跟全域事件數等長」的陣列,壓縮版先把候選壓成
    「每顆神經元自己的固定長度 L」再 scatter(推導文件第 1、2 節)。

    n_real_events(必填,沒有 padding 就傳事件總數)的處理(密集版第 8.1 節
    「pad 事件偽裝成合法座標」的危險,壓縮版一樣存在,而且推導文件本身沒有
    明講,是這裡延伸密集版既有處理方式補上的):候選合法性除了座標篩選,
    額外要求事件本身是真的(
    j < n_real_events),否則 pad 事件的假座標 (0,0,c=0) 會被壓縮版當成
    真正的 tap 收進佇列,污染 n_real_events_per_neuron 跟 (a,b)。「全域
    最後一筆事件的時間」(catch-up 用)相應改成「最後一筆真事件的時間」
    (event_times[n_real_events-1]),不是陣列最後一格——道理跟密集版
    `mask_pad_events` 把 pad 那段蓋成 identity 的效果一致:pad 事件不該
    貢獻任何真實衰減。

    回傳 CompressedConvQueue,三個欄位 shape 第一維都是 OC*H_out*W_out。
    """
    event_times = jnp.asarray(event_times, dtype=jnp.float32)
    x = jnp.asarray(x, dtype=jnp.int32)
    y = jnp.asarray(y, dtype=jnp.int32)
    c = jnp.asarray(c, dtype=jnp.int32)
    n_events = event_times.shape[0]
    # 真事件數,整支函式共用一個(沒有 padding 就傳 n_events = 整條都是真事件)。
    # is_real_event 過濾、safe_j 夾界、catch-up 的 global_last_time 都用它。
    effective_n_events = jnp.asarray(n_real_events, dtype=jnp.int32)

    OC, IC, K, K2 = W.shape
    assert K == K2, f"kernel 必須是方形,拿到 shape={W.shape}"
    N = (K - 1) // S + 1
    n_out_spatial = H_out * W_out

    o_y, valid_y, _ = _axis_candidates(y, K, S, P, N, H_out)  # (n_events, N)
    o_x, valid_x, _ = _axis_candidates(x, K, S, P, N, W_out)  # (n_events, N)
    valid_2d = valid_y[:, :, None] & valid_x[:, None, :]        # (n_events, N, N)

    # 見函式說明:候選合法性額外要求事件本身是真的,不只是座標落在感受野內
    # ——這是密集版第 8.1 節危險在壓縮版的對應處理(None 時 effective_n_events
    # =n_events,is_real_event 全 True,這步退化成 no-op)。
    is_real_event = jnp.arange(n_events) < effective_n_events  # (n_events,)
    valid_2d = valid_2d & is_real_event[:, None, None]

    n_2d = o_y[:, :, None] * W_out + o_x[:, None, :]  # (n_events, N, N) 攤平空間位置 id
    j_2d = jnp.broadcast_to(jnp.arange(n_events, dtype=jnp.int32)[:, None, None],
                             (n_events, N, N))

    # 不合法標成保證越界的 n_out_spatial(第 7 節同一招,scatter 用 mode='drop' 丟棄)
    n_flat = jnp.where(valid_2d, n_2d, n_out_spatial).reshape(-1)
    j_flat = j_2d.reshape(-1)

    local_to_global_j, n_real_per_neuron_spatial = _compress_candidates(
        n_flat, j_flat, n_out_spatial, max_queue_len, n_events)

    # gather 出每個 (空間神經元, 局部欄) 對應的事件座標/通道/時間。sentinel
    # 位置(=n_events)不能放著讓 JAX 預設的 clip 模式夾到「陣列最後一格」
    # ——有 pad 事件時,陣列最後一格可能就是 pad 事件本身(時間是超大的假
    # 值,例如 layer_chain.py 用的 1e12),夾到那裡會讓 catch-up 算出
    # Δt=真實時間-1e12 這種天文數字,(1-1/tau)^(巨大負數) 會 overflow 成
    # inf,inf*v0(=0) 在 process_chunk 裡會變 NaN——即使這個位置最終會被
    # is_real mask 蓋成 0,NaN*0 還是 NaN,污染不會被蓋掉(這是問題紀錄
    # extract_output_events 那段筆記提過的同一類坑,只是這裡換了個地方
    # 出現)。修法:sentinel 明確夾到「最後一筆真事件」(effective_n_events-1),
    # 不是「陣列最後一格」,保證湊出來的時間永遠是有意義的真實時間,Δt 不會
    # 出現這種天文數字。
    safe_j = jnp.minimum(local_to_global_j, effective_n_events - 1)
    x_g = x[safe_j]
    y_g = y[safe_j]
    c_g = c[safe_j]
    t_g = event_times[safe_j]  # (n_out_spatial, L)

    oy_grid = (jnp.arange(n_out_spatial, dtype=jnp.int32) // W_out)[:, None]
    ox_grid = (jnp.arange(n_out_spatial, dtype=jnp.int32) % W_out)[:, None]
    k_y_g = y_g - oy_grid * S + P
    k_x_g = x_g - ox_grid * S + P

    # 權重 gather 才需要 oc(第 1 節:篩選/排序跟 oc 無關,只有這一步要對
    # 每個 oc 各做一次)。非真 tap 位置的 k_y_g/k_x_g 可能落在 [0,K) 之外
    # (甚至可能是負數,JAX gather 對負數 index 是 wraparound,見問題紀錄
    # 第五節那個教訓)。這裡 clip 進 [0,K-1] 是為了語意乾淨:讓每一格
    # gather 出的都是「某個合法權重」,不依賴 wraparound / clip 的邊界行為
    # (夾到哪個值不影響最終數值——這些位置的 b 反正會被 _affine_with_catchup
    # 的 is_real mask 蓋成 0)。
    # 注意(問題紀錄第八節,2026-09-08 特徵化):`--xla_gpu_deterministic_ops=true`
    # 開著 + 外層 batch jax.vmap + jax.grad 時,壓縮版這條路徑的梯度會錯
    # (rel ~0.4;forward 不受影響)。這個 clip **不是** 那個問題的修法——
    # batch=1 不 clip 也對、batch>=2 clip 了也錯。訓練期就是不開這個 flag。
    safe_k_y = jnp.clip(k_y_g, 0, K - 1)
    safe_k_x = jnp.clip(k_x_g, 0, K - 1)
    weight_vals = jax.vmap(lambda oc_w: oc_w[c_g, safe_k_y, safe_k_x])(W)  # (OC, n_out_spatial, L)

    if event_gain is not None:
        # 見函式說明:每個 (神經元, 局部欄) 對應全域事件 safe_j,gather 出
        # 對應的 gain 乘進 weight_vals(對 OC 軸廣播)。非真 tap 位置乘完照樣
        # 被 _affine_with_catchup 的 is_real mask 蓋成 b=0。
        gain_g = jnp.asarray(event_gain, dtype=weight_vals.dtype)[safe_j]  # (n_out_spatial, L)
        weight_vals = weight_vals * gain_g[None, :, :]

    global_last_time = event_times[effective_n_events - 1]

    maps_per_oc = jax.vmap(
        lambda b_oc: _affine_with_catchup(t_g, n_real_per_neuron_spatial, global_last_time,
                                           tau, b_oc)
    )(weight_vals)  # maps_per_oc.a/.b shape (OC, n_out_spatial, L)

    n_out_neurons = OC * n_out_spatial
    a_final = maps_per_oc.a.reshape(n_out_neurons, max_queue_len)
    b_final = maps_per_oc.b.reshape(n_out_neurons, max_queue_len)
    local_to_global_j_final = jnp.broadcast_to(
        local_to_global_j[None, :, :], (OC, n_out_spatial, max_queue_len)
    ).reshape(n_out_neurons, max_queue_len)
    n_real_events_final = jnp.broadcast_to(
        n_real_per_neuron_spatial[None, :], (OC, n_out_spatial)
    ).reshape(n_out_neurons)

    return CompressedConvQueue(maps=AffineMap(a=a_final, b=b_final),
                                local_to_global_j=local_to_global_j_final,
                                n_real_events=n_real_events_final)
