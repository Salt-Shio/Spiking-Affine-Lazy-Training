"""92.75% 那次 run 的黃金輸出:val set 逐筆的 forward 結果,重構時拿來比對。

save:用那次 run 的容量跑一次,記下哪些樣本出界;有出界就放大容量重跑到
    全部不出界,把結果存成黃金輸出。
compare:用黃金輸出記下的容量重跑,逐筆比對。預測類別、每層 spike 數要相等,
    v_final 容差 V_FINAL_ATOL。有不同就回傳非 0。
save_quant / compare_quant:量化版 forward,規格固定為 QUANT_SPEC,容量用 save 記下的;
    要先有 save 的結果。比對全部逐值相等。

用法:python -m example.analysis.golden_output {save,compare,save_quant,compare_quant}
輸出:experiments/<那次 run>/golden/ 底下的 golden.npz、report.yaml、
    golden_quant.npz、report_quant.yaml
"""
import argparse
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np
import yaml

from data.src.nmnist import NMNISTDataset
from example.models.conv_net import ConvNetCompressed, build_decoder, build_growth_policies
from example.paths import DATASET_ROOT, EXPERIMENTS_DIR
from example.utils import load_run_params, load_run_record, rebuild_layers
from salt_core.capacity import Capacity, LayerDiag, grown_to_fit
from salt_core.layers import (dequantize_v_final, raw_events_to_stream,
                              run_network_quantized_traced, run_network_traced)
from salt_core.quant.calibrate import merge_v_ranges, v_abs_max_per_channel, v_range_per_channel
from salt_core.quant.convert import LayerQuantSpec, build_quantized_params

RUN_DIR = EXPERIMENTS_DIR / "conv_compressed_compressed_scale_10k_20260919_050446"
GOLDEN_DIR = RUN_DIR / "golden"
# 跟 e2e 訓練測試同一個容差
V_FINAL_ATOL = 1e-4
# 量化版:每層同一組位元寬度,輸出層不 fire
QUANT_SPEC = dict(bits=8, f_a=10, f_V=10, round_mode="round", overflow_mode="wrap")
# 量膜電位範圍 M 用 val 的前幾筆
QUANT_CALIBRATION_SAMPLES = 50


def load_run():
    """回傳 (layers, decoder, params, val_split, batch_size),全部照那次 run 的設定。"""
    run_record = load_run_record(str(RUN_DIR))
    data_cfg = run_record["config"]["data"]
    layers = rebuild_layers(run_record)
    decoder = build_decoder(run_record["config"]["model"], layers)
    decoder.validate(layers[-1])
    params = load_run_params(str(RUN_DIR), layers, "best")
    dataset = NMNISTDataset(DATASET_ROOT, max_events=int(data_cfg["max_events"]))
    split = dataset.build_split(seed=int(data_cfg["seed_val"]),
                                n_samples=int(data_cfg["val_size"]), which="val")
    batch_size = int(run_record["config"]["train"]["batch_size"])
    return layers, decoder, params, split, batch_size


def capacity_of(layers: list) -> dict:
    """層名 -> 旋鈕 -> 容量值。沒有容量的層不列。"""
    return {layer.name: dict(layer.capacity) for layer in layers if layer.capacity is not None}


def with_capacity(layers: list, capacity: dict) -> list:
    return [layer.with_capacity(Capacity(**capacity[layer.name])) if layer.name in capacity
            else layer for layer in layers]


def forward_split(layers, decoder, params, split, batch_size: int) -> dict:
    """整個 split 分批跑 forward。回傳:v_final (N, n_class)、preds (N,)、
    spike_count (N, n_layers)、needed {層名: {旋鈕: (N,)}}。
    """
    net = ConvNetCompressed(layers)

    @jax.jit
    def run_batch(weights, event_times, x, y, c, n_real):
        result, diags = net.apply_batched(weights, event_times, x, y, c, n_real)
        scores, _ = jax.vmap(decoder.decode)(result)
        return scores, diags

    parts = []
    n = split.labels.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        scores, diags = run_batch(params, split.event_times[start:end], split.x[start:end],
                                  split.y[start:end], split.c[start:end],
                                  split.n_real_events[start:end])
        parts.append((np.asarray(scores),
                      np.stack([np.asarray(d.spike_count) for d in diags], axis=1),
                      [jax.tree_util.tree_map(np.asarray, d.needed) for d in diags]))
    v_final = np.concatenate([p[0] for p in parts])
    return {"v_final": v_final, "preds": np.argmax(v_final, axis=1),
            "spike_count": np.concatenate([p[1] for p in parts]),
            "needed": {layer.name: {knob: np.concatenate([p[2][i][knob] for p in parts])
                                    for knob in parts[0][2][i]}
                       for i, layer in enumerate(layers)}}


def overflow_report(layers: list, out: dict) -> dict:
    """層名 -> 旋鈕 -> 出界的樣本 index list。沒有出界的旋鈕不列。"""
    report = {}
    for layer in layers:
        if layer.capacity is None:
            continue
        for knob, limit in layer.capacity.items():
            idx = np.nonzero(out["needed"][layer.name][knob] > limit)[0]
            if idx.size:
                report.setdefault(layer.name, {})[knob] = idx.tolist()
    return report


def grown_layers(layers: list, out: dict) -> list:
    """照那次 run 的放大公式,用整個 split 的最大需求放大每一層。"""
    policies = build_growth_policies(load_run_record(str(RUN_DIR))["config"]["model"], layers)
    diags = [LayerDiag(spike_count=0, firing_rate=0.0,
                       needed={knob: int(v.max()) for knob, v in out["needed"][layer.name].items()})
             for layer in layers]
    return grown_to_fit(layers, policies, diags)


def accuracy(out: dict, labels: np.ndarray) -> float:
    return float(np.mean(out["preds"] == labels))


def save() -> None:
    layers, decoder, params, split, batch_size = load_run()
    labels = np.asarray(split.labels)
    run_out = forward_split(layers, decoder, params, split, batch_size)
    run_overflow = overflow_report(layers, run_out)
    golden_layers, golden_out, n_regrow = layers, run_out, 0
    while overflow_report(golden_layers, golden_out):
        golden_layers = grown_layers(golden_layers, golden_out)
        golden_out = forward_split(golden_layers, decoder, params, split, batch_size)
        n_regrow += 1

    GOLDEN_DIR.mkdir(exist_ok=True)
    np.savez(GOLDEN_DIR / "golden.npz", labels=labels, preds=golden_out["preds"],
             v_final=golden_out["v_final"], spike_count=golden_out["spike_count"])
    changed = np.nonzero(run_out["preds"] != golden_out["preds"])[0]
    report = {
        "layer_names": [layer.name for layer in layers],
        "run_capacity": capacity_of(layers),
        "golden_capacity": capacity_of(golden_layers),
        "n_regrow": n_regrow,
        "run_overflow_samples": run_overflow,
        "run_accuracy": accuracy(run_out, labels),
        "golden_accuracy": accuracy(golden_out, labels),
        "pred_changed_samples": changed.tolist(),
        "golden_max_needed": {
            layer.name: {"spike_count": int(golden_out["spike_count"][:, i].max()),
                         **{knob: int(v.max())
                            for knob, v in golden_out["needed"][layer.name].items()}}
            for i, layer in enumerate(layers)},
    }
    with open(GOLDEN_DIR / "report.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(report, f, allow_unicode=True, sort_keys=False)
    print(yaml.safe_dump(report, allow_unicode=True, sort_keys=False))


def compare() -> int:
    with open(GOLDEN_DIR / "report.yaml", "r", encoding="utf-8") as f:
        report = yaml.safe_load(f)
    golden = np.load(GOLDEN_DIR / "golden.npz")
    layers, decoder, params, split, batch_size = load_run()
    layers = with_capacity(layers, report["golden_capacity"])
    out = forward_split(layers, decoder, params, split, batch_size)

    pred_diff = np.nonzero(out["preds"] != golden["preds"])[0]
    spike_diff = np.nonzero(np.any(out["spike_count"] != golden["spike_count"], axis=1))[0]
    v_err = np.abs(out["v_final"] - golden["v_final"]).max(axis=1)
    v_diff = np.nonzero(v_err > V_FINAL_ATOL)[0]
    print(f"accuracy {accuracy(out, golden['labels']):.4f}"
          f"(黃金輸出 {report['golden_accuracy']:.4f})")
    print(f"v_final 最大誤差 {v_err.max():.3g}")
    for name, idx in (("預測類別", pred_diff), ("spike 數", spike_diff),
                      ("v_final", v_diff)):
        print(f"{name}不同:{idx.size} 筆 {idx[:20].tolist()}")
    return int(bool(pred_diff.size or spike_diff.size or v_diff.size))


def split_sample(split, i: int) -> tuple:
    return (split.event_times[i], split.x[i], split.y[i], split.c[i], split.n_real_events[i])


def sample_stream(layers: list, split, i: int):
    first = layers[0]
    return raw_events_to_stream(*split_sample(split, i), h_in=first.h_in, w_in=first.w_in)


def build_golden_quant_params(layers: list, params, split) -> list:
    """照 QUANT_SPEC 算每層的量化參數,M 用 split 前 QUANT_CALIBRATION_SAMPLES 筆量。"""
    per_sample = [v_range_per_channel(layers, run_network_traced(
                      layers, sample_stream(layers, split, i), params))
                  for i in range(QUANT_CALIBRATION_SAMPLES)]
    v_abs_max = v_abs_max_per_channel(merge_v_ranges(per_sample))
    spec = LayerQuantSpec(bits=QUANT_SPEC["bits"], f_a=QUANT_SPEC["f_a"], f_V=QUANT_SPEC["f_V"],
                          overflow_mode=QUANT_SPEC["overflow_mode"])
    specs = [spec] * (len(layers) - 1) + [spec._replace(fires=False)]
    return build_quantized_params(layers, params, specs, v_abs_max)


def quant_forward_split(layers: list, decoder, quant_params: list, split) -> dict:
    """整個 split 逐筆跑量化版 forward。回傳 numpy 陣列:v_final_int (N, n_class)、
    preds (N,),spike_count、truncated、overflowed 各 (N, n_layers)。"""
    rows = []
    for i in range(split.labels.shape[0]):
        traces = run_network_quantized_traced(layers, sample_stream(layers, split, i),
                                              quant_params, round_mode=QUANT_SPEC["round_mode"])
        last_result = traces[-1][0]
        scores, _ = decoder.decode(dequantize_v_final(last_result, quant_params[-1]))
        rows.append({
            "v_final_int": np.asarray(last_result.v_final),
            "preds": int(np.argmax(np.asarray(scores))),
            "spike_count": [int(np.asarray(r.spike_mask).sum()) for r, _v, _d in traces],
            "truncated": [bool(d.queue_truncated) or bool(d.output_truncated)
                          for _r, _v, d in traces],
            "overflowed": [bool(np.asarray(r.overflowed).any()) for r, _v, _d in traces],
        })
    return {key: np.array([row[key] for row in rows]) for key in rows[0]}


def load_quant_run() -> tuple:
    """回傳 (layers, decoder, quant_params, split)。容量用 save 記下的,chunk_size 改 1。"""
    with open(GOLDEN_DIR / "report.yaml", "r", encoding="utf-8") as f:
        report = yaml.safe_load(f)
    layers, decoder, params, split, _batch_size = load_run()
    layers = [layer.with_chunk_size(1) for layer in with_capacity(layers, report["golden_capacity"])]
    return layers, decoder, build_golden_quant_params(layers, params, split), split


def save_quant() -> None:
    layers, decoder, quant_params, split = load_quant_run()
    out = quant_forward_split(layers, decoder, quant_params, split)
    labels = np.asarray(split.labels)
    np.savez(GOLDEN_DIR / "golden_quant.npz", labels=labels, **out)
    names = [layer.name for layer in layers]
    report = {
        "spec": QUANT_SPEC,
        "calibration_samples": QUANT_CALIBRATION_SAMPLES,
        "i_V": {name: p.i_V for name, p in zip(names, quant_params)},
        "accuracy": accuracy(out, labels),
        "truncated_samples": {name: int(out["truncated"][:, i].sum())
                              for i, name in enumerate(names)},
        "overflowed_samples": {name: int(out["overflowed"][:, i].sum())
                               for i, name in enumerate(names)},
    }
    with open(GOLDEN_DIR / "report_quant.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(report, f, allow_unicode=True, sort_keys=False)
    print(yaml.safe_dump(report, allow_unicode=True, sort_keys=False))


def compare_quant() -> int:
    with open(GOLDEN_DIR / "report_quant.yaml", "r", encoding="utf-8") as f:
        report = yaml.safe_load(f)
    golden = np.load(GOLDEN_DIR / "golden_quant.npz")
    layers, decoder, quant_params, split = load_quant_run()
    i_V = {layer.name: p.i_V for layer, p in zip(layers, quant_params)}
    out = quant_forward_split(layers, decoder, quant_params, split)

    print(f"accuracy {accuracy(out, golden['labels']):.4f}(黃金輸出 {report['accuracy']:.4f})")
    print(f"i_V {i_V}(黃金輸出 {report['i_V']})")
    n_diff = int(i_V != report["i_V"])
    for key in ("preds", "v_final_int", "spike_count", "truncated", "overflowed"):
        diff = out[key] != golden[key]
        idx = np.nonzero(diff.reshape(diff.shape[0], -1).any(axis=1))[0]
        print(f"{key} 不同:{idx.size} 筆 {idx[:20].tolist()}")
        n_diff += idx.size
    return int(bool(n_diff))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("save", "compare", "save_quant", "compare_quant"))
    args = parser.parse_args()
    if args.mode == "save":
        save()
    elif args.mode == "compare":
        sys.exit(compare())
    elif args.mode == "save_quant":
        save_quant()
    else:
        sys.exit(compare_quant())


if __name__ == "__main__":
    main()
