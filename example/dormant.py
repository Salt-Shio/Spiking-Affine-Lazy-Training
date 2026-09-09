"""Dormant neuron 診斷(Sokar et al. 2023, arXiv:2302.12902)。

每顆神經元的活動 h_i = 一批固定探測樣本上的平均 spike 數;
score_i = |h_i| / (層內 |h| 平均);score_i <= tau 即 tau-dormant。
因為分母是層平均,score 對神經元的平均恆為 1 —— 這是「活動在神經元間
怎麼分布」的形狀指標。平均 firing rate 看不到「少數飽和 + 多數休眠」
(兩者可以有同樣的平均),dormant fraction 看得到,見
docs/math/初始權重尺度推導.md 步驟 7。

只當紀錄指標:不進 loss、不進 init_k 選值。conv 隱藏層才有意義
(FC 輸出層 v_th=1e9 純積分器、不 fire)。
"""
import jax
import jax.numpy as jnp
import numpy as np

from salt_core.layers import ConvLayer, raw_events_to_stream


def _per_neuron_spikes(layers, params, in_stream) -> dict:
    """鏡射 salt_core.layers.run_network,但保留每個 ConvLayer 的逐神經元
    spike 數(sum(spike_mask, axis=1),與 LayerDiag.spike_count 同一個量)。
    回傳 {conv_layer_name: (n_neurons,) 陣列}。"""
    stream = in_stream
    out = {}
    for layer, w in zip(layers, params):
        stream, result, _diag = layer.forward(w, stream)
        if isinstance(layer, ConvLayer):
            out[layer.name] = jnp.sum(result.spike_mask, axis=1)
    return out


def dormant_report(layers, params, probe_batch, *, tau: float = 0.1,
                   chunk: int = 16) -> dict:
    """在固定探測批次上量每個 conv 隱藏層的 dormant fraction。

    probe_batch: (event_times, x, y, c, n_real_events),leading axis = 樣本數。
    分 chunk 做 vmap forward(對齊 salt_core.calibrate 避免整批建壓縮佇列 OOM)。

    回傳 {conv_layer_name: {"dormant_frac": float, "act_p90p10": float}}。
    act_p90p10 = 逐神經元平均活動的 90/10 百分位比值(連續版伴隨指標)。
    """
    et, x, y, c, nr = probe_batch
    first = layers[0]
    n = int(et.shape[0])

    @jax.jit
    def chunk_activity(p, e, xx, yy, cc, rr):
        streams = jax.vmap(raw_events_to_stream, in_axes=(0, 0, 0, 0, 0, None, None))(
            e, xx, yy, cc, rr, first.h_in, first.w_in)
        return jax.vmap(lambda s: _per_neuron_spikes(layers, p, s))(streams)

    totals: dict | None = None
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        acts = chunk_activity(params, et[lo:hi], x[lo:hi], y[lo:hi],
                              c[lo:hi], nr[lo:hi])
        acts = {k: np.asarray(jnp.sum(v, axis=0)) for k, v in acts.items()}
        totals = acts if totals is None else {k: totals[k] + acts[k] for k in totals}

    report = {}
    for name, tot in (totals or {}).items():
        mean_abs = np.abs(tot) / n
        denom = float(np.mean(mean_abs))
        if denom <= 0.0:
            report[name] = {"dormant_frac": 1.0, "act_p90p10": float("nan")}
            continue
        score = mean_abs / denom
        p10, p90 = np.percentile(mean_abs, [10.0, 90.0])
        report[name] = {
            "dormant_frac": float(np.mean(score <= tau)),
            "act_p90p10": float(p90 / p10) if p10 > 0.0 else float("inf"),
        }
    return report
