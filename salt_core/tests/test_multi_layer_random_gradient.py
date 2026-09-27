"""用一個跟被測系統完全獨立、逐事件跑的序列參考實作(不用 core.py 的
associative_scan/chunk 化機制),拿中等規模(幾十筆事件、每層好幾顆神經元)
的隨機資料當 oracle,交叉驗證真正的 build_fc_queue+run_layer_forward+
extract_output_events 串接管線,包括:

  1. 疊加兩次跨層(layer1->layer2->layer3),不是只驗證過一次
  2. 量夠大、多神經元的情況,不是只有手算得動的小例子
  3. 跨層梯度不只測過 v_final 型 loss,也測 s_value(頻率編碼加總)型 loss

外部原始輸入的事件時間用連續亂數 jitter 生成,這一層不會真的同分。但**層與
層之間的輸出不是這樣**:多個下游神經元共用同一顆上游神經元的同一次 fire
當觸發事件時,輸出時間戳記會是精確相等的真同分——事件夠密集時真的會發生
(不是理論上的邊界案例,調參這份測試資料時就實際踩到過),所以這裡的參考
實作(_sequential_multi_layer)一樣要用 (times, step_idx) 複合鍵排序,跟
layer_chain.extract_output_events 用同一條 tie-break 規則,兩邊才對得起來。
tie-break 規則本身的正確性(該排哪個在前)由 test_layer_chain_tiebreak.py
系列獨立、確定性地覆蓋,這裡只是必須讓兩套實作用「同一條」規則,不重複驗證
規則本身對不對。

序列參考實作(_sequential_layer)刻意跟 test_surrogate.py 的
_sequential_lif_with_surrogate 用同一條遞迴公式(不套閘、fire 時 soft
reset),只是這裡 vmap 到多顆神經元、多層之間用 jnp.nonzero()(不帶 size,
在 jax.grad 這種 eager 執行的情境下可以用動態 shape,不需要真正的
pipeline 才需要處理的固定長度/n_real_events 那一套)手動合併、排序——這條路徑
完全不經過 chunk_scan.py/layer_chain.py 的任何程式碼,是真正獨立的第二套
實作,不是拿同一段程式碼互相比對。
"""

import jax
import jax.numpy as jnp

from salt_core.chunk_scan import run_layer_forward
from salt_core.connectivity.fc import build_fc_queue
from salt_core.layer_chain import extract_output_events
from salt_core.surrogate import atan_spike

TOL = 2e-3


def assert_allclose(actual, expected, msg, tol=TOL):
    actual = float(actual)
    expected = float(expected)
    assert abs(actual - expected) < tol, f"{msg}: got {actual}, expected {expected}"


# ---------------------------------------------------------------------------
# 獨立的序列參考實作(oracle):不套閘、fire 時 soft reset,逐事件跑,不截斷
# ---------------------------------------------------------------------------

def _sequential_layer(event_times, event_source_idx, event_gain, W, tau, v_th, alpha):
    """對 W 的每個 row(輸出神經元),逐事件跑序列版。跟 test_surrogate.py 的
    _sequential_lif_with_surrogate 是同一條遞迴公式,這裡 vmap 到多顆神經元、
    多筆事件共用同一份時間軸(FC 無延遲)。

    回傳:
      fired: shape (m, S) bool
      s_seq: shape (m, S),每筆事件自己的 s,連續處理、不截斷
      v_final: shape (m,)
    """
    n_ms = jnp.diff(event_times, prepend=jnp.zeros(1, dtype=event_times.dtype))
    weights = W[:, event_source_idx]  # (m, S)
    if event_gain is not None:
        weights = weights * event_gain[None, :]
    decay = (1.0 - 1.0 / tau) ** n_ms  # (S,),FC 無延遲,全部神經元共用同一份

    def per_neuron(w_row):
        def step(v, inputs):
            a, w = inputs
            h = v * a + w
            s = atan_spike(h - v_th, alpha)
            fired = jax.lax.stop_gradient(s) >= 0.5
            v_new = jnp.where(fired, (1.0 - s) * h, h)
            return v_new, (fired, s)
        v_final, (fired_seq, s_seq) = jax.lax.scan(step, jnp.asarray(0.0), (decay, w_row))
        return v_final, fired_seq, s_seq

    v_final, fired, s_seq = jax.vmap(per_neuron)(weights)
    return fired, s_seq, v_final


def _sequential_multi_layer(event_times, event_source_idx, weight_matrices, tau, v_th, alpha):
    """串接多層 _sequential_layer,層與層之間用動態長度的 nonzero 手動合併,
    排序用跟 layer_chain.extract_output_events 完全一樣的 (times, step_idx)
    複合鍵(lexsort 最後一個 key 是主鍵)——不能只用 times 排序:多個下游神經元
    共用同一顆上游神經元的同一次 fire 當觸發事件時,輸出時間戳記會是精確相等
    的真同分(不是量化造成的假同分),這種情況在事件夠密集時真的會發生
    (這個檔案調參時就踩到過)。同分時處理順序會影響「fire 之後 reset、剩餘
    貢獻怎麼疊加」的結果,兩套實作的 tie-break 規則不一致就會讓 forward 數值
    對不起來,不是梯度計算的問題。回傳最後一層的 (fired, s_seq, v_final)。"""
    times, source_idx, gain = event_times, event_source_idx, None
    result = None
    for li, W in enumerate(weight_matrices):
        fired, s_seq, v_final = _sequential_layer(times, source_idx, gain, W, tau, v_th, alpha)
        result = (fired, s_seq, v_final)
        if li == len(weight_matrices) - 1:
            break
        neuron_idx, step_idx = jnp.nonzero(fired)
        new_times = times[step_idx]
        new_gain = s_seq[neuron_idx, step_idx]
        order = jnp.lexsort((step_idx, new_times))
        times, source_idx, gain = new_times[order], neuron_idx[order], new_gain[order]
    return result


# ---------------------------------------------------------------------------
# 真正的 pipeline:build_fc_queue + run_layer_forward + extract_output_events
# ---------------------------------------------------------------------------

def _real_pipeline_multi_layer(event_times, event_source_idx, weight_matrices, tau, v_th,
                                alpha, chunk_size):
    """串接真正的多層管線,回傳最後一層的 (v_final, s_value)。"""
    times, source_idx, gain = event_times, event_source_idx, None
    n_real_events = event_times.shape[0]
    v_final = s_value = None
    for li, W in enumerate(weight_matrices):
        maps = build_fc_queue(times, source_idx, W, tau, event_gain=gain,
                               n_real_events=n_real_events).maps
        max_steps = maps.a.shape[1]
        spike_mask, spike_event_idx, s_spike, s_value, v_final = run_layer_forward(
            maps, v_th, chunk_size=chunk_size, max_steps=max_steps, alpha=alpha,
            n_real_events=n_real_events)
        if li < len(weight_matrices) - 1:
            times, source_idx, gain, n_real_events = extract_output_events(
                spike_mask, spike_event_idx, s_spike, times)
    return v_final, s_value


# ---------------------------------------------------------------------------
# 隨機資料生成(固定種子,不是每次重跑都不一樣)
# ---------------------------------------------------------------------------

def _make_event_stream(key, num_events, num_sources):
    k_gap, k_src = jax.random.split(key)
    gaps = jax.random.uniform(k_gap, (num_events,), minval=0.3, maxval=2.0)
    times = jnp.cumsum(gaps)
    source_idx = jax.random.randint(k_src, (num_events,), 0, num_sources)
    return times, source_idx


def _make_weight_matrices(key, sizes, low, high):
    """sizes = [n0, m1, m2, ...],回傳每層 (m_l, n_{l-1}) 的權重矩陣列表。"""
    keys = jax.random.split(key, len(sizes) - 1)
    return [jax.random.uniform(k, (sizes[i + 1], sizes[i]), minval=low, maxval=high)
            for i, k in enumerate(keys)]


# 調參時已用獨立診斷腳本確認過,這組設定在每一層都有真實 fire,不會退化成
# 空事件包——下面 _check_gradient_matches_reference 也會在每次呼叫時重新
# 斷言參考梯度不是全部趨近 0,不是只在調參當下驗證過一次就假設之後都成立。
TAU = 4.0
V_TH = 1.0
ALPHA = 2.0
W_LOW, W_HIGH = 0.15, 0.6      # 第一層(接外部原始輸入)的權重範圍
W_LOW_2, W_HIGH_2 = 0.3, 1.1   # 第二層以後的權重範圍(事件變稀疏,權重拉高)


def _build_case(data_seed, weight_seed, num_events, sizes):
    times, source_idx = _make_event_stream(jax.random.PRNGKey(data_seed), num_events, sizes[0])
    key = jax.random.PRNGKey(weight_seed)
    k1, krest = jax.random.split(key)
    W1 = _make_weight_matrices(k1, sizes[:2], W_LOW, W_HIGH)[0]
    rest = _make_weight_matrices(krest, sizes[1:], W_LOW_2, W_HIGH_2) if len(sizes) > 2 else []
    return times, source_idx, [W1] + rest


def _check_gradient_matches_reference(weight_matrices, event_times, event_source_idx,
                                       loss_type, chunk_size):
    def real_loss(weight_matrices):
        v_final, s_value = _real_pipeline_multi_layer(
            event_times, event_source_idx, weight_matrices, TAU, V_TH, ALPHA, chunk_size)
        return jnp.sum(v_final) if loss_type == "v_final" else jnp.sum(s_value)

    def ref_loss(weight_matrices):
        fired, s_seq, v_final = _sequential_multi_layer(
            event_times, event_source_idx, weight_matrices, TAU, V_TH, ALPHA)
        return jnp.sum(v_final) if loss_type == "v_final" else jnp.sum(s_seq)

    real_value, real_grad = jax.value_and_grad(real_loss)(weight_matrices)
    ref_value, ref_grad = jax.value_and_grad(ref_loss)(weight_matrices)

    assert_allclose(real_value, ref_value, f"loss({loss_type}, chunk_size={chunk_size}) forward 值")

    for li, (g_real, g_ref) in enumerate(zip(real_grad, ref_grad)):
        assert g_real.shape == g_ref.shape
        diff = jnp.max(jnp.abs(g_real - g_ref))
        assert float(diff) < TOL, (
            f"loss({loss_type}, chunk_size={chunk_size}): layer{li} 權重梯度最大誤差 "
            f"{float(diff)} 超過容忍值,\nreal={g_real}\nref ={g_ref}")
        # 確認不是「兩邊剛好都算出 0」這種退化情況矇混過關
        assert float(jnp.max(jnp.abs(g_ref))) > 1e-4, (
            f"layer{li} 的參考梯度幾乎全是 0,這組資料/權重規模測不到東西")


# ---------------------------------------------------------------------------
# 兩層:v_final 型、s_value 型 loss 各測一次,各掃兩種 chunk_size
# ---------------------------------------------------------------------------

def test_two_layer_random_gradient_v_final():
    times, source_idx, weights = _build_case(
        data_seed=0, weight_seed=0, num_events=30, sizes=[4, 6, 3])
    for chunk_size in [1, 4]:
        _check_gradient_matches_reference(weights, times, source_idx, "v_final", chunk_size)


def test_two_layer_random_gradient_s_value():
    times, source_idx, weights = _build_case(
        data_seed=0, weight_seed=0, num_events=30, sizes=[4, 6, 3])
    for chunk_size in [1, 4]:
        _check_gradient_matches_reference(weights, times, source_idx, "s_value", chunk_size)


# ---------------------------------------------------------------------------
# 三層(疊加兩次跨層):v_final、s_value 各測一次
# ---------------------------------------------------------------------------

def test_three_layer_random_gradient_v_final():
    times, source_idx, weights = _build_case(
        data_seed=1, weight_seed=1, num_events=34, sizes=[4, 6, 5, 3])
    for chunk_size in [1, 5]:
        _check_gradient_matches_reference(weights, times, source_idx, "v_final", chunk_size)


def test_three_layer_random_gradient_s_value():
    times, source_idx, weights = _build_case(
        data_seed=1, weight_seed=1, num_events=34, sizes=[4, 6, 5, 3])
    for chunk_size in [1, 5]:
        _check_gradient_matches_reference(weights, times, source_idx, "s_value", chunk_size)
