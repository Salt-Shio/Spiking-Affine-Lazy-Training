"""在固定探測批次上量指定層的 dormant 統計,訓練腳本每個 epoch 寫進 metrics.csv。

歸約公式在 salt_core.dormant.dormant_score;這裡決定挑哪幾層、怎麼分批跑、出界時怎麼重算。
只當紀錄指標,不進 loss。
"""
import jax
import jax.numpy as jnp
import numpy as np

from salt_core.capacity import grown_to_fit_batch
from salt_core.dormant import dormant_score
from salt_core.network import Network, RawEvents

from example.utils import take_raw_events


def _make_chunk_activity(network: Network, layer_names: tuple, use_s_value: bool):
    """jit 過的 (params, 一批 RawEvents) -> ({層名: (B, n_neurons) 活動量}, 每層 LayerDiag)。"""

    @jax.jit
    def chunk_activity(params, raw_batch: RawEvents):
        output = network.apply_batched(params, raw_batch)
        activity = {}
        for layer, result in zip(network.layers, output.results):
            if layer.name in layer_names:
                per_step = result.s_value if use_s_value else result.spike_mask
                activity[layer.name] = jnp.sum(per_step, axis=-1)
        return activity, output.diags

    return chunk_activity


def dormant_report(network: Network, params, probe: RawEvents, policies: dict, *,
                   layer_names, tau: float = 0.1, activity: str = "spike",
                   chunk: int = 16) -> tuple[dict, int]:
    """在 probe 上量 layer_names 每層的 dormant 比例。

    probe: 一批 RawEvents(leading axis = 樣本數)。分 chunk 跑,避免整批建壓縮佇列 OOM。
    policies: 層名 -> GrowthPolicy;某個 chunk 容量出界時放大重算,放大只在這次呼叫內有效。
    activity: "spike" 用每樣本 spike 數;"s_value" 用每樣本 s_value 加總(連續版,
        全 0 的死神經元也能排序),取法見 docs/math/初始權重尺度推導.md 步驟 7.1。
    回傳 ({層名: {"dormant_frac": float}}, 重算次數)。activity 不合法時 raise ValueError。
    """
    if activity not in ("spike", "s_value"):
        raise ValueError(f"activity 必須是 'spike' 或 's_value',給的是 {activity!r}")
    layer_names = tuple(layer_names)
    use_s_value = activity == "s_value"
    n = int(probe.event_times.shape[0])
    layers = network.layers
    chunk_activity = _make_chunk_activity(network, layer_names, use_s_value)
    regrows = 0

    totals: dict | None = None
    for lo in range(0, n, chunk):
        chunk_raw = take_raw_events(probe, slice(lo, min(lo + chunk, n)))
        acts, diags = chunk_activity(params, chunk_raw)
        grown = grown_to_fit_batch(layers, policies, diags)
        while grown is not layers:
            layers = grown
            chunk_activity = _make_chunk_activity(network.replace_layers(layers), layer_names,
                                                  use_s_value)
            regrows += 1
            acts, diags = chunk_activity(params, chunk_raw)
            grown = grown_to_fit_batch(layers, policies, diags)
        acts = {k: np.asarray(jnp.sum(v, axis=0)) for k, v in acts.items()}
        totals = acts if totals is None else {k: totals[k] + acts[k] for k in totals}

    report = {}
    for name, tot in (totals or {}).items():
        stats = dormant_score(np.abs(tot) / n, tau=tau)
        report[name] = {"dormant_frac": stats["dormant_frac"]}
    return report, regrows
