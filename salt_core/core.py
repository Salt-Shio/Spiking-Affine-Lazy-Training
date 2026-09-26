"""單狀態事件驅動 LIF 神經元:仿射合成/平行掃描/spike-reset 核心運算。

只有一個狀態 V,沒有電流變數 I,事件到達當下立刻用整數 ms 差解析衰減、
立刻判斷是否 spike。完整推導見 docs/math/單狀態仿射平行掃描推導.md。

本檔案不處理:
- 幫每個神經元建構事件佇列(全連接/conv 的連接關係,屬於後續任務)
- chunk 與 chunk 之間的序列迴圈(屬於訓練迴圈,後續任務)
只實作「給定一個已排序、已切好的 chunk」該怎麼算。

`process_event_int` 是另一條路:整數尺度的膜電位量化模擬(見
docs/問題紀錄.md 第十七節),全程只有整數運算,不碰 `s_c`,也不是這裡的
`process_chunk` 泛化出來的,是獨立的函式——`process_chunk` 訓練熱路徑不動
一行。**不開 `jax_enable_x64`**:實測過這個全域設定會讓 `jax.lax.scan`/
`jnp.argmax` 等等一大票地方的預設整數 dtype 從 int32 變成 int64,直接讓
訓練熱路徑一堆既有測試炸掉(69 個),不是原本評估的「幾乎不影響」——所以
`process_event_int` 全程留在 `int32`,呼叫端要自己保證 `f_a+i_V+f_V` 不超過
`int32` 能安全相乘的範圍(見 `process_event_int` 文件的說明跟顯式檢查)。
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from salt_core.quantize import wide_mul_shift, wrap_to_bits
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
    """一批事件各自的仿射映射:a = (1 - 1/tau)^n_ms(離散 Euler 衰減),b = w。
    n_ms、w 的 shape 是這批事件的數量(leading axis),逐元素套用同一個 tau,
    回傳的 a/b 也是同樣 shape——不是單一事件的映射,是一整批。"""
    a = (1.0 - 1.0 / tau) ** jnp.asarray(n_ms, dtype=jnp.float32)
    b = jnp.asarray(w, dtype=jnp.float32)
    return AffineMap(a=a, b=b)


def normalize_real_events(n_real_events: jax.Array | int,
                           n_out_neurons: int, n_total_events: int) -> jax.Array:
    """把「這條佇列前面幾筆是真事件」這個資訊統一成一種形式:shape
    `(n_out_neurons,)` 的 int32 陣列,每個元素是那顆神經元佇列裡的真事件數。

    接受兩種輸入:
      - 純量(int 或 traced scalar):所有神經元共用同一個真事件數
        (密集版佇列:所有神經元看同一條事件軸)。沒有 padding 就傳
        n_total_events(整條都是真事件)。
      - shape (n_out_neurons,) 陣列:每顆神經元各自的真事件數
        (壓縮版佇列:每顆神經元有自己的子序列)。

    `n_total_events` 只用來對照理解,不參與計算(留著是因為呼叫端已經有它、
    未來若要加範圍檢查也用得到)。正規化之後,`mask_pad_events` /
    `run_layer_forward` 內部只需要處理陣列一種形式(見 docs/問題紀錄.md
    第二節「n_real_events 傳法不一致」)。
    """
    del n_total_events
    arr = jnp.asarray(n_real_events, dtype=jnp.int32)
    return jnp.broadcast_to(arr, (n_out_neurons,))


def mask_pad_events(maps: AffineMap,
                     n_real_events: jax.Array | int) -> AffineMap:
    """把 maps 的事件軸(最後一軸)裡,超過該神經元真事件數的位置強制蓋成
    identity 映射(a=1, b=0)。

    `build_fc_queue`/`build_conv_queue` 都要處理同一個問題:多層串接時,
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
    n_real = normalize_real_events(n_real_events, n_out_neurons, n_total_events)
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


def process_chunk(v0: jax.Array, maps: AffineMap, v_th: float, alpha: float = 2.0,
                   round_step: jax.Array | float | None = None,
                   round_mode: str = "round") -> ChunkForwardResult:
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

    round_step:預設 None,訓練/既有呼叫端行為不變。給定時對「衰減後、還沒加
    這筆事件貢獻」的值套用定點捨入(docs/math/膜電位量化推導.md 的 r(·):
    $\\tilde V_k=r(a_k\\tilde V_{k-1})+q_k$),捨完才加上這筆事件的貢獻——
    **順序不能反過來**:round 模式下「先加再捨」跟「先捨再加」數學上等價
    (round 建立在 floor 上,floor 對「加整數平移」不分正負號都成立),但
    truncate(向零捨去)對負數的方向跟正數相反,一旦這筆事件的貢獻讓值跨過
    0,「先加再捨」跟「先捨再加」會算出不同答案(反例:衰減值 -0.6、貢獻
    +1、格距 1:文件順序 trunc(-0.6)+1=0+1=1.0,若先加總再捨會變成
    trunc(-0.6+1)=trunc(0.4)=0——膜電位本來就會出現負值,不是邊緣情況)。
    只有 chunk_size=1(呼叫端每個 chunk 剛好一筆事件)時,這個捨入才對應
    硬體「每筆事件更新後立刻捨入」的語意;chunk_size>1 時 composed 是
    associative_scan 合成多筆事件之後的結果,對它套用不會等於逐筆捨入,
    呼叫端要自己保證 chunk_size=1。

    這裡故意不用一個外部傳入的捨入函式(呼叫端一度這樣設計過):
    `process_chunk` 是逐神經元 `jax.vmap` 出來的,每個 channel 的捨入格距
    `s_c * 2^-f_V` 不一樣,格距必須跟 v_th 一樣當成 vmap 的陣列參數
    (`in_axes=0`)才能逐 channel 各自生效;一個共用的 Python 函式物件做不到
    這件事(vmap 只能讓它对每顆神經元的輸入做一樣的事)。round_mode(要
    round 還是 truncate)是全域一次決定的設計選擇,不會逐 channel 不同,
    維持純量/靜態字串即可。

    這裡不做 clip/飽和——溢位規則(飽和 or wrap-around)還沒定案(見推導
    文件「目前狀態」表),這個函式只管捨入,不管數值範圍,呼叫端要自己
    另外驗證有沒有溢位。
    """
    if round_mode not in ("round", "truncate"):
        raise ValueError(f"round_mode 必須是 'round' 或 'truncate',給的是 {round_mode!r}")
    composed = jax.lax.associative_scan(combine, maps)
    if round_step is None:
        v_sequence = composed.a * v0 + composed.b  # 對應 docs 推導的 x_k
    else:
        decayed = composed.a * v0
        scaled = decayed / round_step
        rounded_decay = (jnp.round(scaled) if round_mode == "round" else jnp.trunc(scaled)) * round_step
        v_sequence = rounded_decay + composed.b  # r(a_k * V_{k-1}) + q_k,順序對齊推導文件

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


# ============================================================================
# 整數尺度的膜電位量化模擬(見 docs/問題紀錄.md 第十七節)。跟上面
# process_chunk 那條路完全獨立,不共用、不影響訓練熱路徑。
# ============================================================================

class EventStepResultInt(NamedTuple):
    """`process_event_int` 的回傳:一筆事件更新完之後的整數膜電位狀態。"""
    v_final: jax.Array     # 這筆事件之後的膜電位(int32,spike 時已硬重置為 0)
    is_spiked: jax.Array   # bool
    overflowed: jax.Array  # bool,這次寫回暫存器有沒有溢位(繞回去之前的真實值有沒有超出 i_V+f_V 位元)


def process_event_int(v0_int: jax.Array, a_int: jax.Array, is_identity: jax.Array,
                      q_int: jax.Array, v_th_int: jax.Array, *, f_a: int, f_V: int,
                      i_V: int, round_mode: str = "round") -> EventStepResultInt:
    """整數尺度單一事件更新,對應 $\\tilde V_k=r(a_k\\tilde V_{k-1})+q_k$,
    全程只有整數運算,沒有 `s_c`、沒有浮點除法(見 docs/問題紀錄.md 第十七
    節)。

    跟 `process_chunk` 的差異:
    - 這裡永遠只處理**一筆事件**,呼叫端要保證 `chunk_size=1`——理由跟
      `process_chunk` 文件裡「round_step 只有 chunk_size=1 時才對應硬體
      逐事件捨入語意」完全一樣,單一事件不需要 associative_scan 合成、
      不需要在 chunk 內找「第一個 spike」。
    - 沒有 surrogate gradient:這條路是凍結好的模型事後拿整數模擬硬體行為
      用的,不會對輸出做反向傳播,直接硬判斷 `>=`、硬重置成 0,不需要
      `atan_spike`。

    `a_int`/`is_identity` 來自 `quantize.apply_decay_table_int`:
    `is_identity=True`(Δt=0)時跳過衰減這一步(`decayed=v0_int` 原封不動,
    `Q0.f_a` 整數格式存不下剛好等於 1 的衰減係數,見該函式文件)。

    衰減這一步(`quantize.wide_mul_shift`)不檢查溢位——該函式的底層
    `round_shift` 已經證明過,只要 `a_int<2^f_a`(衰減嚴格小於 1),捨入後的
    量值不可能超過 `v0_int` 本身。真正會讓量值變大、需要檢查的地方是加上
    `q_int` 之後,交給 `wrap_to_bits` 處理。

    溢位當下依然照第十七節定案的政策(直接繞回去)算出 `v_final`,`overflowed`
    只是讓呼叫端知道發生了,不代表這個位置的結果被丟棄或修正——選對 `i_V`
    是呼叫端自己的責任。fire 判斷(`>= v_th_int`)比較的是**繞回去之後**的
    `wrapped` 值,不是繞回去之前的真實值:硬體暫存器物理上只留得住繞回去
    之後的位元,比較電路讀到的就是這個被溢位污染過的值,這正是選錯 `i_V`
    會讓 fire 判斷跟著出錯的原因。

    輸入的整數陣列(`v0_int`/`a_int`/`q_int`/`v_th_int`)全程留在 JAX 預設的
    `int32`——**這裡本來想用 `int64` 當乘法暫存空間避免自己先溢位,實測發現
    `jax_enable_x64` 是全域設定,一開下去會讓 `jax.lax.scan`/`jnp.argmax`
    等等一大票地方的預設整數 dtype 從 int32 變成 int64,直接讓訓練熱路徑
    一堆既有測試炸掉,不能這樣做(見 docs/問題紀錄.md 第十七節)**。改成
    在 `int32` 容器裡算,乘法本身透過 `quantize.wide_mul_shift` 拆成高低兩半
    分開乘(細節見該函式文件),不需要真的湊出 `a_int*v0_int` 這個寬乘積,
    所以這裡的位元寬度限制拆成兩個各自獨立的檢查(比原本「`f_a+i_V+f_V`
    綁在一起不能超過 30」寬鬆很多):`f_a` 自己不能超過 15(`wide_mul_shift`
    內部低位那一半的乘積是 `2*f_a` 位元);`i_V+f_V`(暫存器總寬度)不能
    超過 30(`wrap_to_bits` 的 mask/加上 `q_int` 貢獻那一步需要的安全邊際)。
    """
    if round_mode not in ("round", "truncate"):
        raise ValueError(f"round_mode 必須是 'round' 或 'truncate',給的是 {round_mode!r}")
    if f_a > 15:
        raise ValueError(
            f"f_a={f_a} 超過 15,wide_mul_shift 內部的低位乘積(2*f_a 位元)會先"
            "溢位 int32。這個專案的 jax_enable_x64 不能開(見上面的說明),選"
            "小一點的 f_a。")
    if i_V + f_V > 30:
        raise ValueError(
            f"i_V+f_V={i_V + f_V} 超過 30,暫存器總寬度容不進 int32 的安全範圍"
            "(wrap_to_bits 的位元遮罩、加上 q_int 貢獻都需要安全邊際)。選窄"
            "一點的 i_V/f_V 組合。")
    v0_int = jnp.asarray(v0_int, dtype=jnp.int32)
    a_int = jnp.asarray(a_int, dtype=jnp.int32)
    q_int = jnp.asarray(q_int, dtype=jnp.int32)
    v_th_int = jnp.asarray(v_th_int, dtype=jnp.int32)

    decayed_active = wide_mul_shift(a_int, v0_int, shift_bits=f_a, mode=round_mode)
    decayed = jnp.where(is_identity, v0_int, decayed_active)
    raw = decayed + q_int * (1 << f_V)
    wrapped, overflowed = wrap_to_bits(raw, total_bits=i_V + f_V)

    is_spiked = wrapped >= v_th_int
    v_final = jnp.where(is_spiked, jnp.zeros_like(wrapped), wrapped)
    return EventStepResultInt(v_final=v_final, is_spiked=is_spiked, overflowed=overflowed)
