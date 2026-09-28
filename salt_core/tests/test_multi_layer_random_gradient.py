"""多層 FC 管線(佇列建構 + run_layer_forward + extract_output_events_fc)跟獨立的逐事件參考實作比。

參考實作(_sequential_layer)不用 associative_scan、不分 chunk,逐事件跑同一條遞迴(不套閘、fire 時
soft reset,同 test_surrogate.py),層與層之間用動態長度的 nonzero 合併,不經過 float/scan.py、
stream.py。涵蓋:
  1. 三層(兩次跨層)。
  2. 幾十筆事件、每層多顆神經元的隨機資料。
  3. v_final 型跟 s_value 型 loss 的跨層梯度。
  4. fire 落在 chunk 邊界、同一個 chunk 裡 fire 兩次。

層與層之間會有真的同時間戳記:多顆下游神經元被上游同一次 fire 觸發。參考實作跟管線用同一條
(times, step_idx) 排序規則才對得起來;規則本身在 test_stream_tiebreak.py 測。
"""

import jax
import jax.numpy as jnp

from salt_core.float.scan import run_layer_forward
from salt_core.connectivity.fc import build_fc_structure, fc_float_values
from salt_core.stream import extract_output_events_fc
from salt_core.float.surrogate import atan_spike

TOL = 2e-3


def assert_allclose(actual, expected, msg, tol=TOL):
    actual = float(actual)
    expected = float(expected)
    assert abs(actual - expected) < tol, f"{msg}: got {actual}, expected {expected}"


# ---------------------------------------------------------------------------
# 獨立的序列參考實作(oracle):不套閘、fire 時 soft reset,逐事件跑,不截斷
# ---------------------------------------------------------------------------

def _sequential_layer(event_times, event_source_idx, event_gain, W, tau, v_th, alpha):
    """W 的每一列(輸出神經元)逐事件跑遞迴,所有神經元共用同一條時間軸(FC 無延遲)。


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
    """串接多層 _sequential_layer,層與層之間用 (times, step_idx) 排序合併,同 extract_output_events_fc。
    同時間戳記時處理順序會影響 reset 之後的累加,排序規則不同 forward 就對不起來。
    回傳最後一層的 (fired, s_seq, v_final)。"""
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
# 真正的 pipeline:FC 佇列建構 + run_layer_forward + extract_output_events_fc
# ---------------------------------------------------------------------------

def _real_pipeline_multi_layer(event_times, event_source_idx, weight_matrices, tau, v_th,
                                alpha, chunk_size):
    """串接真正的多層管線,回傳最後一層的 (v_final, s_value)。"""
    times, source_idx, gain = event_times, event_source_idx, None
    n_real_events = event_times.shape[0]
    v_final = s_value = None
    for li, W in enumerate(weight_matrices):
        maps = fc_float_values(build_fc_structure(times, source_idx, n_real_events), W, tau, gain)
        max_steps = maps.a.shape[1]
        spike_mask, spike_event_idx, s_spike, s_value, v_final = run_layer_forward(
            maps, v_th, chunk_size=chunk_size, max_steps=max_steps, alpha=alpha,
            n_real_events=n_real_events)
        if li < len(weight_matrices) - 1:
            times, source_idx, gain, n_real_events = extract_output_events_fc(
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


# 這組設定每一層都有真的 fire;_check_gradient_matches_reference 每次也會斷言參考梯度不是全部接近 0。
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


# ---------------------------------------------------------------------------
# 單層構造案例:fire 落在 chunk 邊界、同一個 chunk 裡 fire 兩次
# ---------------------------------------------------------------------------

# 7 筆事件間隔 1 ms、各來自不同來源,1 顆輸出神經元,在第 2、5 筆 fire。
# 手算(a = 0.75):0.5 -> 0.975 -> 1.03125 fire 歸零 -> 0.9 -> 0.875 -> 1.60625 fire。
# chunk_size 2、3、4 讓 fire 落在 chunk 邊界或中間,7 讓兩次 fire 在同一個 chunk。
_TWO_FIRE_TIMES = jnp.arange(1.0, 8.0)
_TWO_FIRE_SOURCES = jnp.arange(7)
_TWO_FIRE_W = jnp.array([[0.5, 0.6, 0.3, 0.9, 0.2, 0.95, 0.1]])


def _real_fire_events(chunk_size):
    maps = fc_float_values(build_fc_structure(_TWO_FIRE_TIMES, _TWO_FIRE_SOURCES, 7),
                           _TWO_FIRE_W, TAU, None)
    result = run_layer_forward(maps, V_TH, chunk_size=chunk_size, max_steps=7, alpha=ALPHA,
                               n_real_events=7)
    return sorted(int(idx) for idx, fired in zip(result.spike_event_idx[0], result.spike_mask[0])
                  if bool(fired))


def test_single_layer_two_fires_matches_reference_across_chunk_sizes():
    fired, _s_seq, _v_final = _sequential_layer(_TWO_FIRE_TIMES, _TWO_FIRE_SOURCES, None,
                                                _TWO_FIRE_W, TAU, V_TH, ALPHA)
    assert [int(i) for i in jnp.nonzero(fired[0])[0]] == [2, 5], "參考實作應該在第 2、5 筆 fire"
    for chunk_size in [1, 2, 3, 4, 7]:
        assert _real_fire_events(chunk_size) == [2, 5], f"chunk_size={chunk_size} fire 位置不對"
        for loss_type in ("v_final", "s_value"):
            _check_gradient_matches_reference([_TWO_FIRE_W], _TWO_FIRE_TIMES, _TWO_FIRE_SOURCES,
                                              loss_type, chunk_size)
