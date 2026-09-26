"""FC(全連接)層的事件佇列建構:全域事件包 + 權重矩陣 -> 每顆輸出神經元自己的
仿射映射序列。對應推導見 docs/math/全連接forward訓練範例.md 第 1~3 節。

無延遲(d_ij=0,見該文件第 6 節),所以全部輸出神經元看到的事件順序、時間完全
相同,只有權重不同:N(事件間隔)只要算一次,權重才需要對每顆輸出神經元分別
從權重矩陣 gather。輸出格式固定是 (輸出神經元數 n_out_neurons, 事件數
n_total_events),跟連接結構(FC/conv)無關,是 core.py/chunk_scan.py 認的
統一介面(見 docs/TODO.md 任務 4)。
"""
import jax
import jax.numpy as jnp

from salt_core.core import AffineMap, create_affine_maps, mask_pad_events


def fc_delta_t(event_times: jax.Array, n_out_neurons: int) -> jax.Array:
    """FC 佇列裡每筆事件的整數 Δt,廣播成 shape (n_out_neurons, n_total_events)。

    跟 `build_fc_queue` 內部算 `n_ms` 用的是同一個基準(跟 t=0 的差,不是跟
    前一筆事件的差,`prepend=0` 才對得上手算數字)——FC 沒有 conv 壓縮版
    那種逐神經元不同子序列/catch-up 的複雜度,所有輸出神經元看到同一組事件、
    同一組 Δt,只有權重不同(見 `build_fc_queue` 說明),不需要像
    `connectivity.conv._delta_t_three_regimes` 那樣分三段處理。

    供整數版量化查表直接用,不需要反推 `log(a)/log(1-1/tau)`(見
    docs/問題紀錄.md 第十九節)。這裡刻意不改 `build_fc_queue` 的回傳型別
    (現有一堆呼叫端把它的回傳值直接當 `AffineMap` 用,改動範圍太大),用一
    個獨立函式讓量化路徑自己另外呼叫。pad 位置(超過 n_real_events)算出來
    的 Δt 可能是誇張的數字沒關係,下游(`chunk_scan._run_layer_scan_int`)
    本來就會強制 pad 位置當 identity,不會用到這裡算出來的值。
    """
    event_times = jnp.asarray(event_times, dtype=jnp.float32)
    n_ms = jnp.diff(event_times, prepend=jnp.zeros(1, dtype=event_times.dtype))
    return jnp.broadcast_to(n_ms, (n_out_neurons, event_times.shape[0])).astype(jnp.int32)


def build_fc_queue(event_times: jax.Array, event_source_idx: jax.Array, W: jax.Array,
                    tau: float, n_real_events: jax.Array | int,
                    event_gain: jax.Array | None = None) -> AffineMap:
    """
    event_times: shape (n_total_events,),全域事件時間(整數 ms,已依時間遞增排序)
    event_source_idx: shape (n_total_events,),每筆事件的來源神經元 index
      (對應 W 的 column)
    W: shape (n_out_neurons, n_in_neurons),權重矩陣,W[i, j] 是 a_j -> b_i 的權重
    tau: 衰減時間常數

    event_gain: shape (n_total_events,),可選,每筆事件自己的可微分增益,預設
      None(等同全 1,行為完全不變)。這一層如果是接在上一層之後(而不是外部
      原始輸入),要傳上一層 chunk_scan.run_layer_forward 回傳的 s_spike
      (對應這筆事件的那個位置)——**不能傳 s_value**。原因:單純用
      event_source_idx 從 W gather 權重,是離散索引操作,對「被索引的 W」有
      梯度,但不會讓上一層的權重出現在這個算式裡;乘上 event_gain=s_spike
      之後,因為 s_spike 是用 atan_spike 算出來、forward 精確等於 1 的可微分
      量(不是寫死的常數 1),數值不變但多了一條路徑讓下一層的 loss 能反傳回
      上一層的權重——跟 core.py 的 soft reset (1-s)*v 是同一個技巧,只是用在
      跨層事件傳遞而不是自身 reset。

    n_real_events: 這批事件裡,前面幾筆是真實事件(必填)。沒有 padding 就傳
      事件總數(全部都是真實事件)。多層串接時,extract_output_events 回傳的
      佇列是「固定上限、後面補 pad 事件」的格式,傳入真實筆數之後,n_real_events
      之後的位置會被強制蓋成 identity 映射(a=1, b=0),不管那些位置的
      event_times/event_gain 算出什麼奇怪的值,都不會產生 NaN、也不可能誤觸發
      spike(純衰減不會自己跨過門檻,見 docs/math/單狀態仿射平行掃描推導.md 第 4
      節引理)。

    回傳 AffineMap,a/b shape 都是 (n_out_neurons, n_total_events)。
    """
    event_times = jnp.asarray(event_times, dtype=jnp.float32)
    # N 是跟「模擬起始時刻 t=0」的差,不是跟第一筆事件自己的時間差——神經元的
    # m_last 初始值是 0,不是第一筆事件的時間本身,兩者只有在第一筆事件剛好
    # 發生在 t=0 時才會一樣。prepend=0 才能讓算出來的係數跟
    # docs/math/全連接forward訓練範例.md 第 4 節的手算數字對上。
    #
    # 這裡刻意不重用 fc_delta_t(雖然算式看起來一樣):fc_delta_t 內部會轉成
    # int32(給整數量化路徑用,那裡 Δt 保證是整數 ms),但這個函式服務的是
    # 訓練用的浮點路徑,呼叫端(例如測試用的隨機資料)不保證 event_times 是
    # 整數,轉 int32 會截斷小數、悄悄改變衰減係數——這個 bug 曾經真的讓
    # test_multi_layer_random_gradient.py 的 s_value 算錯(45 的落差),
    # 修正時發現的,不是憑空猜的風險。
    n_ms = jnp.diff(event_times, prepend=jnp.zeros(1, dtype=event_times.dtype))
    weights = W[:, event_source_idx]  # shape (n_out_neurons, n_total_events)
    if event_gain is not None:
        weights = weights * jnp.asarray(event_gain, dtype=weights.dtype)[None, :]
    n_ms_batched = jnp.broadcast_to(n_ms, weights.shape)
    maps = create_affine_maps(n_ms_batched, weights, tau)  # 回傳 a, b 仿射映射序列

    # 沒有 padding 時呼叫端傳事件總數,mask_pad_events 退化成 no-op。
    return mask_pad_events(maps, n_real_events)
