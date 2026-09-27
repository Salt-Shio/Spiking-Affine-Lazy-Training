"""單狀態事件驅動 LIF 神經元:仿射合成/平行掃描/spike-reset 核心運算。

只有一個狀態 V,沒有電流變數 I,事件到達當下立刻用整數 ms 差解析衰減、
立刻判斷是否 spike。完整推導見 docs/math/單狀態仿射平行掃描推導.md。

本檔案不處理:
- 幫每個神經元建構事件佇列(全連接/conv 的連接關係,屬於後續任務)
- chunk 與 chunk 之間的序列迴圈(屬於訓練迴圈,後續任務)

單步運算 `process_chunk`:給定一個已排序、已切好的 chunk 怎麼算,訓練跟浮點
推論用。整數版的單步更新在 `salt_core/quant/scan.py`。
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from salt_core.surrogate import atan_spike


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
    """一批事件各自的仿射映射:a = (1 - 1/tau) ** n_ms(離散 Euler 衰減),b = w。
    逐元素計算;回傳的 a 形狀跟 n_ms 一樣,b 跟 w 一樣。"""
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
    """把 maps 的事件軸(最後一軸)裡,超過該神經元真事件數的位置強制蓋成
    identity 映射(a=1, b=0)。

    FC/conv 的佇列建構都要處理同一個問題:多層串接時,
    extract_output_events 用「固定上限、後面補 pad 事件」表示不定長度的佇列,
    pad 位置的 event_times/event_gain 不管算出什麼奇怪的值,都不該讓下游
    的膜電位變化或誤觸發 spike。蓋成 identity 之後,pad 位置對 associative
    scan 的合成結果完全沒有貢獻(a=1 表示不衰減、b=0 表示不加權重),純衰減
    也不可能自己跨過門檻(見 docs/math/單狀態仿射平行掃描推導.md 第 4 節引理),
    所以安全。原本 fc.py/conv.py 各自重複同一段五行邏輯,收成這個共用函式
    (見 docs/問題紀錄.md 第二節)。

    n_real_events 交給 `normalize_real_events` 統一成 (n_out_neurons,) 陣列。
    沒有 padding 的呼叫端傳 n_total_events(每個位置都是真事件),此函式就退化
    成 no-op。
    """
    n_out_neurons, n_total_events = maps.a.shape
    n_real = normalize_real_events(n_real_events, n_out_neurons)
    real_mask = jnp.arange(n_total_events)[None, :] < n_real[:, None]  # (n_out_neurons, n_total_events)
    return AffineMap(a=jnp.where(real_mask, maps.a, 1.0),
                      b=jnp.where(real_mask, maps.b, 0.0))


def spike_step_upper_bound(b: jax.Array, v_th: float, chunk_size: int) -> jax.Array:
    """一顆(或一批)神經元的掃描步數上界(證明見
    docs/math/掃描步數上界推導.md)。`b`:shape `(..., L)`,佇列裡每筆事件自己的
    仿射偏移(`AffineMap.b`,正負號 = 對應權重的正負號)。回傳比 `b`少最後一軸
    的 int32 陣列。

    推導的兩個獨立上界,取更緊的:
      m   = #{b_i > 0}                         (只看正負號)
      S/v_th = Σ_{b_i>0} b_i / v_th 再取 floor  (權重大小 / 門檻的能量預算)
    取 m* = min(m, floor(S/v_th)),步數上界 = m* + ceil((L - m*) / chunk_size)。

    `v_th` 遠大於典型 `b` 量級時(例如非 fire 層的 v_th=1e9),`floor(S/v_th)`
    會壓到 0,正確反映「這層事實上不會 fire」。
    """
    L = b.shape[-1]
    positive = jnp.where(b > 0, b, 0.0)
    m = jnp.sum(b > 0, axis=-1)
    energy_bound = jnp.floor(jnp.sum(positive, axis=-1) / v_th)
    m_star = jnp.minimum(m.astype(jnp.float32), energy_bound)
    steps = m_star + jnp.ceil((L - m_star) / chunk_size)
    return steps.astype(jnp.int32)


class ChunkForwardResult(NamedTuple):
    v_final: jax.Array     # chunk 結束後的膜電位(若有 spike,已套用硬重置)
    is_spiked: jax.Array   # bool scalar,這個 chunk 內是否有 spike
    spike_idx: jax.Array   # 第一次 spike 的事件索引(0-based);沒 spike 則等於 chunk 長度
    v_sequence: jax.Array  # shape (chunk_size,),假設全程不 reset 算出的逐事件電壓序列
    s_sequence: jax.Array  # shape (chunk_size,),可微分的 spike 強度序列,forward 精確等於 0/1


def process_chunk(v0: jax.Array, maps: AffineMap, v_th: float,
                   alpha: float = 2.0) -> ChunkForwardResult:
    """處理一個 chunk 內已排序的事件,偵測第一次 spike 並 reset,之後事件全部丟棄。

    maps 的 leading axis 是 chunk 內的事件數(chunk_size),maps.a[i]/maps.b[i]
    是第 i 筆事件(0-based,依真實時間遞增排序)自己的仿射映射。先假設整個
    chunk 都不會 spike,用 associative_scan 平行合成前綴映射,一次算出
    v_sequence(對應 docs 推導的 x_k);純衰減不可能自己跨過門檻(推導文件
    第 4 節引理),所以 spike 只可能發生在 v_sequence 本身的某個位置,直接
    比大小找第一個成立的索引即可,不需要求根。

    spike 判斷用 atan_spike(surrogate.py,對照 spikingjelly ATan 公式重刻):
    forward 精確等於硬判斷 v>=v_th,數值行為不變;backward 用平滑近似讓梯度
    能穿過這個原本不可微分的判斷點。「哪個事件是第一個 spike」這個離散選擇
    本身用 jax.lax.stop_gradient 明確標成常數——只對「有沒有 spike」這個值
    套用 surrogate,不對「選中哪個 index」求梯度,這是業界標準做法。
    """
    composed = jax.lax.associative_scan(combine, maps)
    v_sequence = composed.a * v0 + composed.b  # 對應 docs 推導的 x_k

    s_sequence = atan_spike(v_sequence - v_th, alpha)
    spiked_mask = jax.lax.stop_gradient(s_sequence) >= 0.5
    any_spiked = jnp.any(spiked_mask)
    spike_idx = jnp.where(any_spiked, jnp.argmax(spiked_mask), v_sequence.shape[0])

    # 用可微分的 soft reset 建構 spike 分支的結束電壓:s 在 spike 位置的 forward
    # 值精確是 1.0,所以 (1-s)*v 在 forward 上精確等於硬重置的 0.0,backward
    # 則會經過 s 的 surrogate 梯度。spike_idx 沒 spike 時是 sentinel(=chunk
    # 長度),clip 成合法索引只是為了安全 gather,實際要不要採用這個分支由
    # 下面的 jnp.where 決定。
    spike_idx_clamped = jnp.minimum(spike_idx, v_sequence.shape[0] - 1)
    v_after_spike = (1.0 - s_sequence[spike_idx_clamped]) * v_sequence[spike_idx_clamped]
    v_silent = v_sequence[-1]
    v_final = jnp.where(any_spiked, v_after_spike, v_silent)

    return ChunkForwardResult(v_final=v_final, is_spiked=any_spiked, spike_idx=spike_idx,
                               v_sequence=v_sequence, s_sequence=s_sequence)
