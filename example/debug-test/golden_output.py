"""92.75% 那次 run 的黃金輸出:val set 逐筆的 forward 結果,重構時拿來比對。

save:用那次 run 的容量跑一次,記下哪些樣本出界;有出界就放大容量重跑到
    全部不出界,把結果存成黃金輸出。
compare:用黃金輸出記下的容量重跑,逐筆比對。預測類別、每層 spike 數要相等,
    v_final 容差 V_FINAL_ATOL。有不同就回傳非 0。

用法:python example/debug-test/golden_output.py {save,compare}
輸出:experiments/<那次 run>/golden/golden.npz、report.yaml
"""
import argparse
import dataclasses
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np
import yaml

from data.src.nmnist import NMNISTDataset
from example.models.conv_net import ConvNetCompressed, build_decoder
from example.paths import DATASET_ROOT, EXPERIMENTS_DIR
from example.utils import load_run_params, load_run_record, rebuild_layers
from salt_core.layers import LayerDiag

RUN_DIR = EXPERIMENTS_DIR / "conv_compressed_compressed_scale_10k_20260919_050446"
GOLDEN_DIR = RUN_DIR / "golden"
# 跟 e2e 訓練測試同一個容差
V_FINAL_ATOL = 1e-4
CAPACITY_KNOBS = ("L", "max_out_spikes", "max_steps")
# 容量旋鈕 -> 出界時超過它的診斷欄位
OVERFLOW_FIELDS = {"L": "max_real_queue", "max_out_spikes": "n_out_spikes",
                   "max_steps": "min_steps_needed"}
DIAG_FIELDS = ("spike_count", "max_real_queue", "n_out_spikes", "min_steps_needed")


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
    """層名 -> 三個容量旋鈕的值。沒有容量旋鈕的層(FC)不列。"""
    return {layer.name: {knob: int(getattr(layer, knob)) for knob in CAPACITY_KNOBS}
            for layer in layers if all(hasattr(layer, knob) for knob in CAPACITY_KNOBS)}


def with_capacity(layers: list, capacity: dict) -> list:
    return [dataclasses.replace(layer, **capacity[layer.name]) if layer.name in capacity
            else layer for layer in layers]


def forward_split(layers, decoder, params, split, batch_size: int) -> dict:
    """整個 split 分批跑 forward。回傳 numpy 陣列:
    v_final (N, n_class)、preds (N,),以及 DIAG_FIELDS 每個 (N, n_layers)。
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
                      {f: np.stack([np.asarray(getattr(d, f)) for d in diags], axis=1)
                       for f in DIAG_FIELDS}))
    v_final = np.concatenate([p[0] for p in parts])
    out = {"v_final": v_final, "preds": np.argmax(v_final, axis=1)}
    for f in DIAG_FIELDS:
        out[f] = np.concatenate([p[1][f] for p in parts])
    return out


def overflow_report(layers: list, out: dict) -> dict:
    """層名 -> 旋鈕 -> 出界的樣本 index list。沒有出界的旋鈕不列。"""
    report = {}
    for i, layer in enumerate(layers):
        for knob, limit in capacity_of([layer]).get(layer.name, {}).items():
            needed = out[OVERFLOW_FIELDS[knob]][:, i]
            idx = np.nonzero(needed > limit)[0]
            if idx.size:
                report.setdefault(layer.name, {})[knob] = idx.tolist()
    return report


def grown_layers(layers: list, out: dict) -> list:
    """照訓練的放大公式,用整個 split 的最大需求放大每一層。"""
    grown = []
    for i, layer in enumerate(layers):
        diag = LayerDiag(spike_count=0, firing_rate=0.0,
                         **{f: int(out[f][:, i].max()) for f in OVERFLOW_FIELDS.values()})
        grown.append(layer.grown_to_fit(diag))
    return grown


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
            layer.name: {f: int(golden_out[f][:, i].max()) for f in DIAG_FIELDS}
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("save", "compare"))
    args = parser.parse_args()
    if args.mode == "save":
        save()
    else:
        sys.exit(compare())


if __name__ == "__main__":
    main()
