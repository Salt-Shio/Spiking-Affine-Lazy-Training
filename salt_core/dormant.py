"""Dormant neuron 診斷(Sokar et al. 2023, arXiv:2302.12902)。

每顆神經元的活動 h_i = 一批固定探測樣本上的平均活動量;
score_i = |h_i| / (層內 |h| 平均);score_i <= tau 即 tau-dormant。
因為分母是層平均,score 對神經元的平均恆為 1 —— 這是「活動在神經元間
怎麼分布」的形狀指標。平均 firing rate 看不到「少數飽和 + 多數休眠」
(兩者可以有同樣的平均),dormant fraction 看得到,見
docs/math/初始權重尺度推導.md 步驟 7。

活動量兩種取法(見該推導步驟 7.1),`dormant_report` 的 `activity` 選:
- `"spike"`(預設):每樣本 spike 數,`sum(spike_mask, axis=1)`。
- `"s_value"`:每樣本 `sum_t s_value`,連續版,對全 0 的死神經元也能排序。

只當紀錄指標:不進 loss、不進 init_k 選值。conv 隱藏層才有意義
(FC 輸出層 v_th=1e9 純積分器、不 fire)。

放 salt_core:活動分布是 SNN 通用除錯量,不是 example 專屬;ReDo(訓練中
回收休眠神經元)也靠 `dormant_score` 挑回收對象。這裡自己逐層跑 forward
收逐神經元活動,不碰 `LayerDiag`、不碰訓練熱路徑(見 docs/監測規格.md §5)。
只依賴 salt_core.layers,不碰 data / example。
"""
import jax
import jax.numpy as jnp
import numpy as np

from salt_core.layers import ConvLayer, raw_events_to_stream


def dormant_score(activity, *, tau: float = 0.1) -> dict:
    """逐神經元活動量 `(n,)`(已在探測樣本上平均、非負)-> dormant 統計。

    純歸約,不跑 forward —— ReDo 挑回收對象、`dormant_report` 寫紀錄都用這個。

    回傳 {"dormant_frac": float, "score": (n,) ndarray}:
      dormant_frac = #{score_i <= tau} / n,score_i = activity_i / 層平均。
    層平均為 0(整層全死)時 dormant_frac = 1.0、score 全 0。
    """
    activity = np.abs(np.asarray(activity, dtype=np.float64))
    denom = float(np.mean(activity))
    if denom <= 0.0:
        return {"dormant_frac": 1.0, "score": np.zeros_like(activity)}
    score = activity / denom
    return {
        "dormant_frac": float(np.mean(score <= tau)),
        "score": score,
    }


def _per_neuron_activity(layers, params, in_stream, *, use_s_value: bool) -> dict:
    """逐層跑 `layer.forward`(鏡射 salt_core.layers.run_network),保留每個
    ConvLayer 的逐神經元活動量。回傳 {conv_layer_name: (n_neurons,) 陣列}。

    `use_s_value=False`:`sum(spike_mask, axis=1)`(每樣本 spike 數)。
    `use_s_value=True`:`sum(s_value, axis=1)`(連續版)。
    """
    stream = in_stream
    out = {}
    for layer, w in zip(layers, params):
        stream, result, _diag = layer.forward(w, stream)
        if isinstance(layer, ConvLayer):
            per_step = result.s_value if use_s_value else result.spike_mask
            out[layer.name] = jnp.sum(per_step, axis=1)
    return out


def dormant_report(layers, params, probe_batch, *, tau: float = 0.1,
                   activity: str = "spike", chunk: int = 16) -> dict:
    """在固定探測批次上量每個 conv 隱藏層的 dormant 統計。

    probe_batch: (event_times, x, y, c, n_real_events),leading axis = 樣本數。
    分 chunk 做 vmap forward(避免整批一次建構壓縮佇列 OOM)。

    回傳 {conv_layer_name: {"dormant_frac": float}}。
    """
    if activity not in ("spike", "s_value"):
        raise ValueError(f"activity 必須是 'spike' 或 's_value',給的是 {activity!r}")
    use_s_value = activity == "s_value"
    et, x, y, c, nr = probe_batch
    first = layers[0]
    n = int(et.shape[0])

    @jax.jit
    def chunk_activity(p, e, xx, yy, cc, rr):
        streams = jax.vmap(raw_events_to_stream, in_axes=(0, 0, 0, 0, 0, None, None))(
            e, xx, yy, cc, rr, first.h_in, first.w_in)
        return jax.vmap(lambda s: _per_neuron_activity(
            layers, p, s, use_s_value=use_s_value))(streams)

    totals: dict | None = None
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        acts = chunk_activity(params, et[lo:hi], x[lo:hi], y[lo:hi],
                              c[lo:hi], nr[lo:hi])
        acts = {k: np.asarray(jnp.sum(v, axis=0)) for k, v in acts.items()}
        totals = acts if totals is None else {k: totals[k] + acts[k] for k in totals}

    report = {}
    for name, tot in (totals or {}).items():
        stats = dormant_score(np.abs(tot) / n, tau=tau)
        report[name] = {"dormant_frac": stats["dormant_frac"]}
    return report
