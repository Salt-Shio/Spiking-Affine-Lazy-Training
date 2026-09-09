"""在 test set(或 val set)上評估一個已完成的訓練 run。

訓練腳本(`example/train_conv_compressed.py`)只在訓練過程中看 val set、用它挑
best_params。test set 是刻意分開、只在需要一個「最終、沒被調參污染」的數字時
才碰的——所以獨立成這支腳本,不焊進訓練迴圈,也不會每次訓練自動跑。

用法:
  python -m example.eval_test <exp_dir> [--which test|val] [--n N] [--seed S]
                          [--params best|final]

<exp_dir> 是一次訓練的輸出目錄(裡面要有 run.yaml + best_params.npz /
params.npz)。網路形狀從 run.yaml 的 config 快照重建,壓縮容量(L /
max_out_spikes)用 run.yaml 記的訓練結束時的最終值(訓練中可能長大過),
權重從 npz 載入。結果寫進 <exp_dir>/eval_<which>.yaml,不改 run.yaml。
"""
import argparse
import dataclasses
import datetime
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import yaml

from data.src.nmnist import NMNISTDataset
from salt_core.layers import ConvLayer
from example.models.conv_net import ConvNetCompressed, build_decoder, build_network
from example.paths import DATASET_ROOT
from example.train_conv_compressed import make_evaluate_accuracy

TEST_POOL_SIZE = 10000   # N-MNIST Test/ 全量(見 data.src.nmnist.build_split）


def _load_run_record(exp_dir: str) -> dict:
    with open(os.path.join(exp_dir, "run.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _rebuild_layers(run_record: dict) -> list:
    """從 run.yaml 重建 layer list:形狀吃 config 快照,壓縮容量吃訓練結束的
    最終值(final_capacity)。init_k 不用套——評估的權重直接從 npz 載入,
    init_k 只在 init_weight / 校準時才有意義。"""
    layers = build_network(run_record["config"]["model"])
    final_capacity = run_record.get("final_capacity", {})
    rebuilt = []
    for layer in layers:
        cap = final_capacity.get(layer.name)
        if cap is not None and isinstance(layer, ConvLayer):
            layer = dataclasses.replace(
                layer, L=int(cap["L"]), max_out_spikes=int(cap["max_out_spikes"]))
        rebuilt.append(layer)
    return rebuilt


def _load_params(exp_dir: str, layers: list, which_params: str) -> tuple:
    fname = "best_params.npz" if which_params == "best" else "params.npz"
    data = np.load(os.path.join(exp_dir, fname))
    return tuple(data[layer.name] for layer in layers)


def evaluate_run(exp_dir: str, which: str, n_samples: int | None,
                  seed: int, which_params: str) -> dict:
    run_record = _load_run_record(exp_dir)
    model_cfg = run_record["config"]["model"]
    data_cfg = run_record["config"]["data"]

    layers = _rebuild_layers(run_record)
    net = ConvNetCompressed(layers)
    decoder = build_decoder(model_cfg, layers)
    decoder.validate(layers[-1])
    params = _load_params(exp_dir, layers, which_params)

    dataset = NMNISTDataset(DATASET_ROOT, max_events=int(data_cfg["max_events"]))
    if n_samples is None:
        n_samples = TEST_POOL_SIZE if which == "test" else int(data_cfg["val_size"])
    split = dataset.build_split(seed=seed, n_samples=n_samples, which=which)

    eval_batch_size = int(data_cfg.get("batch_size")
                          or run_record["config"]["train"]["batch_size"])
    evaluate_accuracy = make_evaluate_accuracy(net, decoder, eval_batch_size)
    accuracy, preds = evaluate_accuracy(params, split)

    return {
        "which": which,
        "n_samples": int(n_samples),
        "seed": int(seed),
        "params": which_params,
        "accuracy": accuracy,
        "git_commit": run_record.get("git_commit", ""),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exp_dir", help="一次訓練的輸出目錄")
    parser.add_argument("--which", choices=("test", "val"), default="test")
    parser.add_argument("--n", type=int, default=None,
                        help="抽幾筆(預設:test 全量 10000、val 用 config 的 val_size)")
    parser.add_argument("--seed", type=int, default=0,
                        help="抽樣 seed(決定抽哪些、順序),預設 0")
    parser.add_argument("--params", choices=("best", "final"), default="best",
                        help="用 best_params.npz(預設)還是 params.npz")
    args = parser.parse_args()

    result = evaluate_run(args.exp_dir, args.which, args.n, args.seed, args.params)

    out_path = os.path.join(args.exp_dir, f"eval_{args.which}.yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(result, f, allow_unicode=True, sort_keys=False)

    print(f"{args.which} accuracy = {result['accuracy']:.4f} "
          f"(n={result['n_samples']}, seed={result['seed']}, params={result['params']})")
    print(f"寫入 {out_path}")


if __name__ == "__main__":
    main()
