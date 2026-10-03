"""量化模型的執行跟分析:量 M、整批整數 forward、容量放大、溢位分析。

example/quantize.py、example/analysis/golden_output.py 共用。
"""
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from example.utils import take_input_events
from salt_core.capacity import LayerDiag, grown_to_fit
from salt_core.network import InputEvents, Network
from salt_core.quant.backend import QuantBackend
from salt_core.quant.calibrate import merge_v_ranges, v_abs_max_per_channel, v_range_per_channel
from salt_core.quant.convert import LayerQuantSpec
from salt_core.quant.fixed_point import OverflowMode, RoundMode

# overflow_events 最多列幾筆,其餘只算進筆數
MAX_OVERFLOW_EVENTS = 50


def _batches(raw: InputEvents, batch_size: int):
    n = raw.event_times.shape[0]
    for start in range(0, n, batch_size):
        yield take_input_events(raw, slice(start, start + batch_size))


def calibrate_v_abs_max(network: Network, params, raw: InputEvents,
                        batch_size: int) -> list[np.ndarray]:
    """raw 的全部樣本分批跑浮點 forward,回傳對齊 layers 的每層逐 channel 膜電位最大量值 M。

    network 每層要 chunk_size=1,否則 v_range_per_channel raise ValueError。
    """
    run = jax.jit(lambda raw_batch: network.apply_batched(params, raw_batch, trace=True).traces)
    per_batch = []
    for raw_batch in _batches(raw, batch_size):
        # 一批的軌跡 (B, n, 步數) 併成 (n, B*步數),當一筆樣本取極值
        traces = [trace._replace(v_steps=np.moveaxis(np.asarray(trace.v_steps), 0, 1)
                                 .reshape(trace.v_steps.shape[1], -1))
                  for trace in run(raw_batch)]
        per_batch.append(v_range_per_channel(network.layers, traces))
    return v_abs_max_per_channel(merge_v_ranges(per_batch))


def quant_layer_specs(base: LayerQuantSpec, n_layers: int,
                      out_per_channel: bool = True) -> list[LayerQuantSpec]:
    """每層的量化規格:前面的層都用 base,最後一層是不 fire 的膜電位回歸輸出層,
    逐 channel 或整層共用 s_c 照 out_per_channel。"""
    return [base] * (n_layers - 1) + [base._replace(per_channel=out_per_channel, fires=False)]


class QuantSplitOutput(NamedTuple):
    """quant_forward_split 的回傳,N 是樣本數。"""
    v_final_int: np.ndarray     # (N, n_out) 輸出層暫存器值
    preds: np.ndarray           # (N,)
    spike_count: np.ndarray     # (N, n_layers)
    truncated: np.ndarray       # (N, n_layers) bool,容量出界
    overflowed: np.ndarray      # (N, n_layers) bool,暫存器溢位過
    v_unfitted_min: np.ndarray  # (N, n_layers) 這層所有神經元寫回暫存器之前的真實值的最小
    v_unfitted_max: np.ndarray  # (N, n_layers) 同上,最大
    needed: dict                # 層名 -> 旋鈕 -> (N,) 需要的容量;沒有容量的層是空 dict


# 存檔、比對用的逐筆輸出
REFERENCE_FIELDS = ("v_final_int", "preds", "spike_count", "truncated", "overflowed")
_ARRAY_FIELDS = REFERENCE_FIELDS + ("v_unfitted_min", "v_unfitted_max")


def quant_forward_split(network: Network, decoder, params, round_mode: RoundMode | str,
                        raw: InputEvents, batch_size: int) -> QuantSplitOutput:
    """raw 的全部樣本分批跑整數 forward。params 當 jit 常數(含 Python 整數 f_a、f_V、i_V)。"""
    backend = QuantBackend(round_mode=round_mode)

    @jax.jit
    def run_batch(raw_batch):
        output = network.apply_batched(params, raw_batch, backend=backend)
        scores, _ = jax.vmap(decoder.decode)(backend.readout(output.last, params[-1]))
        n = scores.shape[0]
        truncated = [jnp.zeros(n, dtype=bool) if layer.capacity is None
                     else ~layer.capacity.fits(diag)
                     for layer, diag in zip(network.layers, output.diags)]
        return {"v_final_int": output.last.v_final,
                "preds": jnp.argmax(scores, axis=1),
                "spike_count": jnp.stack([jnp.sum(r.spike_mask, axis=(1, 2))
                                          for r in output.results], axis=1),
                "truncated": jnp.stack(truncated, axis=1),
                "overflowed": jnp.stack([jnp.any(r.overflowed, axis=(1, 2))
                                         for r in output.results], axis=1),
                "v_unfitted_min": jnp.stack([jnp.min(r.v_unfitted_min, axis=1)
                                             for r in output.results], axis=1),
                "v_unfitted_max": jnp.stack([jnp.max(r.v_unfitted_max, axis=1)
                                             for r in output.results], axis=1),
                "needed": [diag.needed for diag in output.diags]}

    parts = [jax.tree_util.tree_map(np.asarray, run_batch(raw_batch))
             for raw_batch in _batches(raw, batch_size)]
    arrays = {key: np.concatenate([part[key] for part in parts]) for key in _ARRAY_FIELDS}
    needed = {layer.name: {knob: np.concatenate([part["needed"][i][knob] for part in parts])
                           for knob in parts[0]["needed"][i]}
              for i, layer in enumerate(network.layers)}
    return QuantSplitOutput(**arrays, needed=needed)


def reference_arrays(out: QuantSplitOutput) -> dict[str, np.ndarray]:
    """存檔、比對用的逐筆輸出(REFERENCE_FIELDS 那幾個)。"""
    return {key: getattr(out, key) for key in REFERENCE_FIELDS}


def grown_for_needed(layers: list, policies: dict, needed: dict) -> list:
    """用每個旋鈕在所有樣本裡的最大需求放大每一層,公式照 policies。

    needed: 層名 -> 旋鈕 -> 逐筆需求陣列。
    """
    diags = [LayerDiag(spike_count=0, firing_rate=0.0,
                       needed={knob: int(v.max()) for knob, v in needed[layer.name].items()})
             for layer in layers]
    return grown_to_fit(layers, policies, diags)


def run_until_fits(network: Network, decoder, params, round_mode: RoundMode | str,
                   raw: InputEvents, batch_size: int,
                   policies: dict) -> tuple[Network, QuantSplitOutput, int]:
    """跑整數 forward,有樣本容量出界就放大容量重跑。回傳 (放得下的網路, 輸出, 重跑次數)。"""
    n_regrow = 0
    out = quant_forward_split(network, decoder, params, round_mode, raw, batch_size)
    while out.truncated.any():
        network = network.replace_layers(grown_for_needed(network.layers, policies, out.needed))
        out = quant_forward_split(network, decoder, params, round_mode, raw, batch_size)
        n_regrow += 1
    return network, out, n_regrow


# ============================================================================
# 溢位分析
# ============================================================================

def with_extra_i_V(params, extra: list[int]) -> list:
    """每層的 i_V 各加 extra[i] 位元,其他參數不變。"""
    return [p._replace(i_V=p.i_V + e) for p, e in zip(params, extra)]


def headroom_bits(params, out: QuantSplitOutput) -> list[float | None]:
    """每層暫存器的餘裕:寫回之前的真實值離暫存器上下限還差幾個位元,負數代表溢位。

    暫存器 w = i_V + f_V 位元,範圍 [-2**(w-1), 2**(w-1) - 1];
    餘裕 = (w - 1) - log2(max(-最小值, 最大值 + 1))。整層都是 0 時回傳 None。
    """
    result = []
    for i, p in enumerate(params):
        peak = max(-int(out.v_unfitted_min[:, i].min()), int(out.v_unfitted_max[:, i].max()) + 1)
        result.append(None if peak <= 0 else round(p.i_V + p.f_V - 1 - math.log2(peak), 3))
    return result


def min_extra_bits(network: Network, decoder, params, round_mode: RoundMode | str,
                   raw: InputEvents, batch_size: int, policies: dict,
                   out: QuantSplitOutput) -> tuple[list[int], int]:
    """最上游有溢位的層 i_V 加 1 重跑,直到所有樣本都不溢位。只看實際的溢位旗標。

    一次只加最上游那層:上游繞回的錯誤值會讓下游跟著溢位,上游修好才知道下游真正要多少。
    out: 用 params 跑出的結果,作為第一輪。
    回傳 (每層要加的位元數, 重跑輪數);每層都是在上游定案後剛好不溢位的最小值。
    暫存器超過 fixed_point.MAX_REGISTER_BITS 時 raise ValueError。
    """
    extra = [0] * len(params)
    rounds = 0
    while out.overflowed.any():
        extra[int(np.argmax(out.overflowed.any(axis=0)))] += 1
        _, out, _ = run_until_fits(network, decoder, with_extra_i_V(params, extra), round_mode,
                                   raw, batch_size, policies)
        rounds += 1
    return extra, rounds


def _neuron_position(layer, neuron: int) -> tuple[int, int, int]:
    """神經元編號 -> (channel, y, x);FC 是 (神經元, 0, 0)。"""
    shape = layer.unflatten_neurons(np.zeros(layer.n_neurons)).shape
    return tuple(int(v) for v in np.unravel_index(neuron, shape))


def overflow_events(network: Network, params, round_mode: RoundMode | str, raw: InputEvents,
                    out: QuantSplitOutput) -> list[dict]:
    """溢位發生在哪裡:每個溢位的 (樣本, 層, 神經元) 列第一次溢位的那一步。

    sample 是 raw 裡的編號。最多列 MAX_OVERFLOW_EVENTS 筆。
    v_before、v_after 是那一步寫回前後的暫存器值(fire 時 v_after 是 0);
    true_min、true_max 是這顆神經元整筆樣本寫回之前的真實值極值。
    """
    backend = QuantBackend(round_mode=round_mode)
    run = jax.jit(lambda one: network.apply(params, one, backend=backend, trace=True))
    events = []
    for i in np.nonzero(out.overflowed.any(axis=1))[0]:
        output = run(take_input_events(raw, int(i)))
        for layer, p, result, trace in zip(network.layers, params, output.results, output.traces):
            overflowed = np.asarray(result.overflowed)
            v_steps = np.asarray(trace.v_steps)
            for neuron in np.nonzero(overflowed.any(axis=1))[0]:
                step = int(np.argmax(overflowed[neuron]))
                channel, y, x = _neuron_position(layer, int(neuron))
                events.append({
                    "sample": int(i),
                    "layer": layer.name, "channel": channel, "y": y, "x": x,
                    "time_ms": float(np.asarray(trace.event_ms)[neuron, step]),
                    "v_before": int(v_steps[neuron, step - 1]) if step > 0 else 0,
                    "v_after": int(v_steps[neuron, step]),
                    "spiked": bool(np.asarray(result.spike_mask)[neuron, step]),
                    "true_min": int(np.asarray(result.v_unfitted_min)[neuron]),
                    "true_max": int(np.asarray(result.v_unfitted_max)[neuron]),
                    "register_range": [-2 ** (p.i_V + p.f_V - 1), 2 ** (p.i_V + p.f_V - 1) - 1]})
                if len(events) >= MAX_OVERFLOW_EVENTS:
                    return events
    return events


def overflow_summary(network: Network, decoder, params, round_mode: RoundMode | str,
                     raw: InputEvents, batch_size: int, policies: dict,
                     out: QuantSplitOutput) -> dict:
    """report 用的溢位分析:每層溢位筆數、餘裕、最少要加幾位元、溢位位置。

    有層用飽和時只回傳溢位筆數(飽和過的樣本數)跟餘裕:飽和是刻意讓暫存器變窄,
    每筆樣本都會碰到上下限,最少要加幾位元、溢位位置沒有意義。
    """
    names = [layer.name for layer in network.layers]
    summary = {
        "overflowed_samples": {name: int(out.overflowed[:, i].sum())
                               for i, name in enumerate(names)},
        "headroom_bits": dict(zip(names, headroom_bits(params, out))),
    }
    if any(OverflowMode(p.overflow_mode) is OverflowMode.SATURATE for p in params):
        return summary
    extra, rounds = min_extra_bits(network, decoder, params, round_mode, raw, batch_size,
                                   policies, out)
    return {
        **summary,
        "min_extra_bits": dict(zip(names, extra)),
        "min_extra_bits_rounds": rounds,
        "overflow_events": overflow_events(network, params, round_mode, raw, out),
    }
