"""在 test set(或 val set)上評估一個已完成的訓練 run。

訓練腳本(`example/train_conv_compressed.py`)只在訓練過程中看 val set、用它挑
best_params。test set 是刻意分開、只在需要一個「最終、沒被調參污染」的數字時
才碰的——所以獨立成這支腳本,不焊進訓練迴圈,也不會每次訓練自動跑。

用法:
  python -m example.eval_test <exp_dir> [--which test|val] [--n N] [--seed S]
                          [--params best|final]

<exp_dir> 是一次訓練的輸出目錄(裡面要有 train/run.yaml + train/best_params.npz /
train/params.npz)。網路形狀從 run.yaml 的 config 快照重建,壓縮容量(L /
max_out_spikes)用 run.yaml 記的訓練結束時的最終值(訓練中可能長大過),
權重從 npz 載入。

輸出(獨立的 eval/ 子資料夾,不改 train/ 底下任何東西):
  <exp_dir>/eval/<which>.yaml       accuracy / loss / confusion_matrix + 中繼資料
  <exp_dir>/eval/<which>_preds.npz  逐樣本的 preds + labels,要重算別的東西不用重跑整個評估

評估用的 `make_evaluate`(`example/utils.py`)跟訓練熱路徑(`run_epochs` 每個
epoch 對 val split 的檢查)共用同一份——兩邊要的計算完全一樣(分批 vmap 算
scores、導出 accuracy/loss/preds),只是誰用哪個回傳值不同。
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
from example.models.conv_net import N_CLASSES, ConvNetCompressed, build_decoder, build_network
from example.paths import DATASET_ROOT
from example.utils import EVAL_DIRNAME, TRAIN_DIRNAME, load_params_npz, make_evaluate

TEST_POOL_SIZE = 10000   # N-MNIST Test/ 全量(見 data.src.nmnist.build_split）


def _load_run_record(exp_dir: str) -> dict:
    with open(os.path.join(exp_dir, TRAIN_DIRNAME, "run.yaml"), "r", encoding="utf-8") as f:
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
    return load_params_npz(os.path.join(exp_dir, TRAIN_DIRNAME, fname), layers)


def _confusion_matrix(labels: np.ndarray, preds: np.ndarray, n_classes: int) -> np.ndarray:
    """列 = 真實類別、欄 = 預測類別,標籤就是類別的數字 index(N-MNIST 是 0–9)。"""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (labels, preds), 1)
    return cm


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
    evaluate = make_evaluate(net, decoder, eval_batch_size)
    accuracy, loss, preds = evaluate(params, split)
    labels = np.asarray(split.labels)
    confusion_matrix = _confusion_matrix(labels, preds, N_CLASSES)

    eval_dir = os.path.join(exp_dir, EVAL_DIRNAME)
    os.makedirs(eval_dir, exist_ok=True)
    np.savez(os.path.join(eval_dir, f"{which}_preds.npz"), preds=preds, labels=labels)

    return {
        "which": which,
        "n_samples": int(n_samples),
        "seed": int(seed),
        "params": which_params,
        "accuracy": accuracy,
        "loss": loss,
        "confusion_matrix": confusion_matrix.tolist(),
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

    out_path = os.path.join(args.exp_dir, EVAL_DIRNAME, f"{args.which}.yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(result, f, allow_unicode=True, sort_keys=False)
    preds_path = os.path.join(args.exp_dir, EVAL_DIRNAME, f"{args.which}_preds.npz")

    print(f"{args.which} accuracy = {result['accuracy']:.4f}  loss = {result['loss']:.4f} "
          f"(n={result['n_samples']}, seed={result['seed']}, params={result['params']})")
    print(f"寫入 {out_path}")
    print(f"preds 存到 {preds_path}")


if __name__ == "__main__":
    main()
