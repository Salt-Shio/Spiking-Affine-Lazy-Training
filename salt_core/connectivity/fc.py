"""FC(全連接)層的事件佇列建構:全域事件包 + 權重矩陣 -> 每顆輸出神經元自己的
仿射映射序列。對應推導見 docs/math/全連接forward訓練範例.md 第 1~3 節。

無延遲(d_ij=0,見該文件第 6 節),所以全部輸出神經元看到的事件順序、時間完全
相同,只有權重不同:N(事件間隔)只要算一次,權重才需要對每顆輸出神經元分別
從權重矩陣 gather。輸出格式固定是 (輸出神經元數 n_out_neurons, 事件數
n_total_events),跟連接結構(FC/conv)無關,是 core.py/chunk_scan.py 認的
統一介面(見 docs/TODO.md 任務 4)。
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from salt_core.core import (AffineMap, create_affine_maps, mask_pad_events,
                            normalize_real_events)


class FCQueue(NamedTuple):
    """`build_fc_queue` 的回傳型別,欄位對齊 conv 的 `CompressedConvQueue`
    (`maps` + `delta_t`),兩種層的量化路徑用同一種方式取 Δt。"""
    maps: AffineMap     # a/b shape (n_out_neurons, n_total_events)
    delta_t: jax.Array  # shape (n_out_neurons, n_total_events) float32,`maps.a` 就是用它算的;
                        # 每顆輸出神經元都一樣(FC 沒有逐神經元的子序列),pad 位置是 0


def build_fc_queue(event_times: jax.Array, event_source_idx: jax.Array, W: jax.Array,
                    tau: float, n_real_events: jax.Array | int,
                    event_gain: jax.Array | None = None) -> FCQueue:
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
      節引理)。pad 位置的 Δt 也定義成 0,整數版量化查表才會把它當 identity。

    回傳 `FCQueue`:`maps` 的 a/b、`delta_t` shape 都是 (n_out_neurons, n_total_events)。
    """
    event_times = jnp.asarray(event_times, dtype=jnp.float32)
    weights = W[:, event_source_idx]  # shape (n_out_neurons, n_total_events)
    if event_gain is not None:
        weights = weights * jnp.asarray(event_gain, dtype=weights.dtype)[None, :]
    n_out_neurons, n_total_events = weights.shape

    # N 是跟前一筆事件的差,第一筆跟「模擬起始時刻 t=0」比,不是跟第一筆事件
    # 自己的時間比——神經元的 m_last 初始值是 0。prepend=0 才能讓算出來的
    # 係數跟 docs/math/全連接forward訓練範例.md 第 4 節的手算數字對上。
    n_ms = jnp.diff(event_times, prepend=jnp.zeros(1, dtype=event_times.dtype))
    n_real = normalize_real_events(n_real_events, n_out_neurons, n_total_events)
    is_real = jnp.arange(n_total_events)[None, :] < n_real[:, None]
    delta_t = jnp.where(is_real, n_ms[None, :], 0.0)
    maps = create_affine_maps(delta_t, weights, tau)

    # 沒有 padding 時呼叫端傳事件總數,mask_pad_events 退化成 no-op。
    return FCQueue(maps=mask_pad_events(maps, n_real_events), delta_t=delta_t)
