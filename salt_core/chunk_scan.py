"""多顆神經元(shape (n_out_neurons, S) 的 AffineMap 佇列,任何連接方式都適用)
的 chunk 化序列消化迴圈:對每顆神經元,反覆用 core.process_chunk 消化固定
大小的滑動視窗,spike 後從下一筆事件重新起跑,直到整條佇列消化完。對照
docs/math/單狀態仿射平行掃描推導.md 第 5 節、Bullet Trains
snn/dynamics.py 的 run_events_scan/associative_scan_step,但拿掉了求根跟
delay/sort_idx 那些跟這個專案無關的複雜度。

`run_layer_forward_int`/`run_layer_forward_int_traced` 是整數版(膜電位量化
模擬):chunk_size 恆為 1,逐事件呼叫 `core.process_event_int`。
"""
import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp

from salt_core.core import (AffineMap, normalize_real_events, process_chunk,
                            process_event_int)
from salt_core.fixed_point import OverflowMode, RoundMode


class LayerForwardResult(NamedTuple):
    """一層跑完一次 forward 的原始輸出。**這是給 readout / 解碼器的公開穩定
    契約**(見 salt_core/decoder.py),不是內部產物:任何輸出編碼(膜電位
    回歸讀 v_final、頻率/群體讀 s_value)要的量都在這裡,而且 pad 步對
    s_value 的貢獻已在 run_layer_forward 內歸零,呼叫端可以直接對整個
    (n_out_neurons, max_steps) 陣列加總、不必自己處理 padding。
    """
    spike_mask: jax.Array       # shape (n_out_neurons, max_steps),該步是否真的輸出一個 spike
    spike_event_idx: jax.Array  # shape (n_out_neurons, max_steps),這條佇列自己的局部
                                 # index(0-based)——密集版佇列(build_fc_queue/
                                 # build_conv_queue)這個 index 就是全域事件 index,可以
                                 # 直接查 event_times;壓縮版佇列(build_conv_queue_compressed)
                                 # 不是,這是「這顆神經元自己壓縮佇列裡的第幾欄」,要先查
                                 # local_to_global_j(見 connectivity/conv.py)才是全域事件
                                 # index,見 layer_chain.py extract_output_events 的說明
    s_spike: jax.Array          # shape (n_out_neurons, max_steps),spike 事件自己的 s
    s_value: jax.Array          # shape (n_out_neurons, max_steps),視窗內有效事件的 s 加總
    v_final: jax.Array          # shape (n_out_neurons,),消化完 max_steps 步之後的膜電位


def _pad_queue(maps: AffineMap, pad_len: int) -> AffineMap:
    """在每顆神經元佇列尾端補 identity 映射(a=1, b=0)的 pad 事件,純粹讓滑動
    視窗不會讀出界。identity 映射不改變狀態、也不可能觸發 spike,補多少筆都
    不影響正確性,只是浪費一點計算。"""
    n_out_neurons = maps.a.shape[0]
    pad_a = jnp.ones((n_out_neurons, pad_len), dtype=maps.a.dtype)
    pad_b = jnp.zeros((n_out_neurons, pad_len), dtype=maps.b.dtype)
    return AffineMap(a=jnp.concatenate([maps.a, pad_a], axis=1),
                      b=jnp.concatenate([maps.b, pad_b], axis=1))


def run_layer_forward(maps: AffineMap, v_th: float, chunk_size: int, max_steps: int,
                       n_real_events: jax.Array | int,
                       alpha: float = 2.0) -> LayerForwardResult:
    """消化 maps(shape (n_out_neurons, L))整條佇列——`L`(`maps.a.shape[1]`)
    密集版(build_fc_queue/build_conv_queue)是全域事件數,壓縮版
    (build_conv_queue_compressed)是每顆神經元自己的壓縮佇列長度
    (max_queue_len),兩者對這個函式來說是同一件事:就是這條佇列有幾欄。

    max_steps 要保證涵蓋最壞情況(每筆事件都 spike,一次只能消化一筆):每一步
    至少消化 1 筆真實事件,所以取 max_steps = L(這條佇列的長度,不管密集版
    還是壓縮版)永遠安全(正確性優先,還沒針對速度調參,對應 docs/TODO.md
    任務 4 尚未驗證的訓練速度那一項)。

    alpha 是 core.process_chunk 內 atan_spike 的平滑程度參數,見 surrogate.py。

    n_real_events:maps 裡「前面幾筆是真實事件」的數量(必填)。沒有 padding
    的呼叫端傳 maps.a.shape[1](整條都當真實事件)。多層串接時,
    layer_chain.extract_output_events 回傳的佇列是「固定上限、後面補 pad
    事件」的格式(見該檔案說明),這時要把它回傳的真實筆數明確傳進來,不能
    讓這裡自己用 maps 的 shape 反推——不然 n_valid_in_chunk(下面)會把 pad
    事件也當成真實事件去加總 s_value,重演跟 _pad_queue 同一類「padding 步驟
    偷偷貢獻梯度」的問題。

    s_value 的定義(這個窗口裡「有幾個位置算數」,把這些位置的 s 全部加起來,
    不是只挑一個代表值):
      n_valid_in_chunk = spike_idx + 1                                (有 spike,0-based)
                       = clip(n_real_events - pointer, 0, chunk_size)  (沒 spike)
      s_value          = sum(s_sequence[:n_valid_in_chunk])
    有 spike 時只加到 spike 的位置為止,跟 v_sequence「spike 之後全部丟棄」
    是同一條規則;沒 spike 時只加真正的真實事件、排除 n_real_events 之後的
    假事件(_pad_queue 補的視窗安全邊界、或呼叫端傳入的 pad 事件,道理相同)。
    這樣「一顆神經元處理完整條佇列,每筆真實事件自己的 s 都恰好被加總一次」
    ——包括 max_steps 這個安全上限跑出真實事件範圍之後的空轉步驟:那時
    n_real_events-pointer 是負的,clip 成 0,n_valid_in_chunk=0,自動貢獻 0,
    不需要呼叫端另外處理。loss = sum(s_value) 這種頻率編碼/spike-count
    類型的 loss,可以直接對整個 (n_out_neurons, max_steps) 陣列加總,不用管
    chunk_size 切得多細、spike 發生在哪一步。

    s_spike:shape (n_out_neurons, max_steps),該步「如果真的 spike」,spike
    那個事件自己的 s(= core.process_chunk 內部 soft reset 用的
    s_sequence[spike_idx_clamped],這裡只是再取一次,不重算 atan_spike)。
    spike_mask 為 False 的位置數值沒有意義。**不要跟 s_value 搞混**:s_value
    是這個窗口內「所有有效事件的 s 加總」,spike 位置之前還有其他事件時,
    s_value 會把它們也算進去;s_spike 只單獨代表「這一次 spike,spike 事件
    自己的強度」,forward 精確等於 1(有 spike 時)。跨層串接要把這顆神經元
    的 spike 當成下一層的一筆輸入事件時,必須用 s_spike 當那筆事件的可微分
    「增益」,不能用 s_value——用 s_value 會把 spike 之前非 spike 事件的
    surrogate 斜率也混進跨層的梯度路徑,等於把「不套閘」的設計精神在跨層
    邊界上悄悄破壞掉。

    回傳(LayerForwardResult):
      spike_mask: shape (n_out_neurons, max_steps),該步是否真的輸出一個 spike
      spike_event_idx: shape (n_out_neurons, max_steps),這條佇列自己的局部
        index(0-based,見 LayerForwardResult 欄位說明:密集版就是全域事件
        index,壓縮版要另外查表才是);spike_mask 為 False 的位置數值沒有意義
      s_spike: shape (n_out_neurons, max_steps),見上方說明
      s_value: shape (n_out_neurons, max_steps),見上方定義
      v_final: shape (n_out_neurons,),消化完 max_steps 步之後的膜電位
    """
    result, _v_steps, _pointer_steps = _run_layer_scan(
        maps, v_th, chunk_size, max_steps, n_real_events, alpha, trace=False)
    return result


def run_layer_forward_traced(maps: AffineMap, v_th: float, chunk_size: int, max_steps: int,
                              n_real_events: jax.Array | int, alpha: float = 2.0
                              ) -> tuple[LayerForwardResult, jax.Array, jax.Array]:
    """跟 `run_layer_forward` 跑一模一樣的掃描,但額外把每步的膜電位與佇列
    指標疊出來。回傳 `(result, v_steps, pointer_steps)`:

    - `result`:`LayerForwardResult`,逐位元等於 `run_layer_forward`(共用 scan
      內核,只差多疊兩條 per-step 輸出)。
    - `v_steps`:`(n_out_neurons, max_steps)`,每步 chunk 結束(套過 soft reset)
      的膜電位 = 完整膜電位軌跡,最後一欄 = `result.v_final`。
    - `pointer_steps`:`(n_out_neurons, max_steps)` int,每步「這顆神經元從佇列
      第幾欄開始消化」(消化前)。`chunk_size=1` 時就是步序號;`chunk_size>1`
      時因 spike 提前收而不均勻。配 `local_to_global_j`(壓縮 conv)/ 直接當
      全域 index(密集)+ event_times 可還原每步的真實毫秒。

    定位是週期性深 probe,不進訓練熱路徑;呼叫端(`layers` 的 `forward_traced` /
    `run_network_traced`)負責組成 `LayerForwardTrace` 並 `stop_gradient`。
    """
    return _run_layer_scan(maps, v_th, chunk_size, max_steps, n_real_events, alpha,
                            trace=True)


def _run_layer_scan(maps: AffineMap, v_th: float, chunk_size: int, max_steps: int,
                     n_real_events: jax.Array | int, alpha: float, *, trace: bool):
    """`run_layer_forward` / `run_layer_forward_traced` 共用的 scan 內核。
    回傳 `(LayerForwardResult, v_steps, pointer_steps)`;`trace=False` 時後兩個
    是 `None`,且 graph 跟舊版逐位元相同(`trace` 是 Python 靜態 bool,
    `if trace` 分支在 trace 期被消掉)。
    """
    n_out_neurons, n_total_events = maps.a.shape
    # 統一成 (n_out_neurons,) int32:密集版傳純量(所有神經元同一個數)、
    # 壓縮版傳逐神經元陣列、沒傳代表整條都是真事件——正規化之後底下只處理
    # 陣列一種形式(見 core.normalize_real_events)。
    n_real = normalize_real_events(n_real_events, n_out_neurons, n_total_events)
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

        # spike 事件自己的 s(不是加總後的 s_value),core.process_chunk 內部
        # soft reset 已經算過同一個值,這裡只是再 gather 一次,不重算 atan_spike。
        spike_idx_clamped = jnp.minimum(chunk_result.spike_idx, chunk_size - 1)
        s_spike = chunk_result.s_sequence[neuron_idx_range, spike_idx_clamped]

        n_consumed = jnp.where(chunk_result.is_spiked, chunk_result.spike_idx + 1, chunk_size)
        spike_event_idx = pointer + chunk_result.spike_idx
        new_pointer = pointer + n_consumed
        ys = (chunk_result.is_spiked, spike_event_idx, s_spike, s_value)
        if trace:
            ys = ys + (chunk_result.v_final, pointer)
        return (chunk_result.v_final, new_pointer), ys
        # step 不用這個 ys tuple 的內容,但 scan 會把每一步疊成陣列當回傳值給外面用

    init = (jnp.zeros(n_out_neurons, dtype=maps.a.dtype), jnp.zeros(n_out_neurons, dtype=jnp.int32))
    (v_final, _), ys = jax.lax.scan(step, init, None, length=max_steps)
    spike_mask, spike_event_idx, s_spike, s_value = ys[:4]

    # scan 的疊代軸在最前面,shape 是 (max_steps, n_out_neurons),轉成
    # (n_out_neurons, max_steps) 給呼叫端用
    result = LayerForwardResult(spike_mask=spike_mask.T, spike_event_idx=spike_event_idx.T,
                                 s_spike=s_spike.T, s_value=s_value.T, v_final=v_final)
    if not trace:
        return result, None, None
    v_step, pointer_step = ys[4], ys[5]
    return result, v_step.T, pointer_step.T


# ============================================================================
# 整數版掃描(膜電位量化模擬)
# ============================================================================

class LayerForwardResultInt(NamedTuple):
    """一層跑完一次整數版 forward 的輸出(chunk_size 恆為 1)。沒有
    `s_value`/`s_spike`:這條路沒有 surrogate gradient,rate coding 要用的話
    直接對 `spike_mask` 加總。`L` 是佇列長度,第 `t` 步處理佇列第 `t` 欄。
    """
    spike_mask: jax.Array       # shape (n_out_neurons, L) bool
    spike_event_idx: jax.Array  # shape (n_out_neurons, L) int32,就是佇列自己的欄位索引
                                 # (跟 LayerForwardResult 同一個局部 index 慣例)
    v_final: jax.Array          # shape (n_out_neurons,) 暫存器值 int32;
                                 # layers.dequantize_v_final 換過之後是物理尺度 float32
    overflowed: jax.Array       # shape (n_out_neurons, L) bool,每一步寫回暫存器
                                 # 之前的真實值有沒有超出 i_V+f_V 位元


def run_layer_forward_int(a_int: jax.Array, is_identity: jax.Array, q_int: jax.Array,
                          v_th_int: jax.Array | None, *, f_a: int, f_V: int, i_V: int,
                          round_mode: RoundMode | str = RoundMode.ROUND,
                          overflow_mode: OverflowMode | str = OverflowMode.WRAP
                          ) -> LayerForwardResultInt:
    """整數版掃描:每一步只處理一筆事件(硬體每筆事件更新後就立刻捨入),
    所以第 `t` 步就是佇列第 `t` 欄,掃描長度等於佇列長度 `a_int.shape[1]`,
    不需要浮點版那套滑動視窗跟 pointer。

    `a_int`/`is_identity`/`q_int`:shape `(n_out_neurons, L)`,呼叫端先用
    `quantize.apply_decay_table_int` 查好表、準備好權重整數碼(這個函式只跑
    遞迴)。**每一欄都照實套用**,不看真事件數:佇列裡不是真事件的位置,
    佇列建構那一步就要做成 identity(Δt=0、權重 0),conv 的 catch-up 欄則是
    真的要衰減(見 `connectivity.conv.build_conv_queue_compressed`、
    `connectivity.fc.build_fc_queue`)。

    `v_th_int`:純量或 shape `(n_out_neurons,)`;`None` 代表這層不 fire。
    `f_a`/`f_V`/`i_V`/`round_mode`/`overflow_mode` 整層共用,原樣交給
    `core.process_event_int`。
    """
    result, _v_steps = _run_layer_scan_int(a_int, is_identity, q_int, v_th_int,
                                           f_a=f_a, f_V=f_V, i_V=i_V, round_mode=round_mode,
                                           overflow_mode=overflow_mode, trace=False)
    return result


def run_layer_forward_int_traced(a_int: jax.Array, is_identity: jax.Array, q_int: jax.Array,
                                 v_th_int: jax.Array | None, *, f_a: int, f_V: int, i_V: int,
                                 round_mode: RoundMode | str = RoundMode.ROUND,
                                 overflow_mode: OverflowMode | str = OverflowMode.WRAP
                                 ) -> tuple[LayerForwardResultInt, jax.Array]:
    """跟 `run_layer_forward_int` 跑一模一樣的掃描,額外回傳每步(寫回之後)
    的暫存器值 `v_steps`,shape `(n_out_neurons, L)`,最後一欄等於
    `result.v_final`。為什麼溢位驗證要看逐步值,見
    `layers.ConvLayer.forward_quantized_traced`。
    """
    return _run_layer_scan_int(a_int, is_identity, q_int, v_th_int,
                               f_a=f_a, f_V=f_V, i_V=i_V, round_mode=round_mode,
                               overflow_mode=overflow_mode, trace=True)


def _run_layer_scan_int(a_int: jax.Array, is_identity: jax.Array, q_int: jax.Array,
                        v_th_int: jax.Array | None, *, f_a: int, f_V: int, i_V: int,
                        round_mode: RoundMode | str, overflow_mode: OverflowMode | str,
                        trace: bool):
    """`run_layer_forward_int`/`run_layer_forward_int_traced` 共用的 scan
    內核。回傳 `(LayerForwardResultInt, v_steps)`;`trace=False` 時後者是
    `None`。"""
    n_out_neurons, queue_len = a_int.shape
    # None(不 fire)是沒有 leaf 的 pytree,vmap 直接原樣傳給 process_event_int
    v_th_arr = (None if v_th_int is None else
                jnp.broadcast_to(jnp.asarray(v_th_int, dtype=jnp.int32), (n_out_neurons,)))
    step_fn = functools.partial(process_event_int, f_a=f_a, f_V=f_V, i_V=i_V,
                                round_mode=round_mode, overflow_mode=overflow_mode)
    vmapped_step = jax.vmap(step_fn)

    def step(v, t):
        step_result = vmapped_step(v, a_int[:, t], is_identity[:, t], q_int[:, t], v_th_arr)
        # scan 會把每一步的 ys 疊成陣列回傳給外面,step 自己不讀它
        ys = (step_result.is_spiked, step_result.overflowed)
        if trace:
            ys = ys + (step_result.v_final,)
        return step_result.v_final, ys

    v_final, ys = jax.lax.scan(step, jnp.zeros(n_out_neurons, dtype=jnp.int32),
                               jnp.arange(queue_len))
    spike_mask, overflowed = ys[0], ys[1]

    # scan 的疊代軸在最前面,shape 是 (L, n_out_neurons),轉成
    # (n_out_neurons, L) 給呼叫端用
    spike_event_idx = jnp.broadcast_to(jnp.arange(queue_len), (n_out_neurons, queue_len))
    result = LayerForwardResultInt(spike_mask=spike_mask.T, spike_event_idx=spike_event_idx,
                                   v_final=v_final, overflowed=overflowed.T)
    if not trace:
        return result, None
    v_steps = ys[2]
    return result, v_steps.T
