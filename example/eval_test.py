"""在 test set(或 val set)上評估一個跑完的訓練 run。訓練時只看 val;test 只在要最終數字時跑。

用法:
  python -m example.eval_test <exp_dir> [--which test|val] [--n N] [--seed S] [--params best|final]

exp_dir 裡要有 train/run.yaml 跟 train/best_params.npz、train/params.npz;網路跟權重從 npz 讀。
輸出在 exp_dir/eval/,不改 train/:
  <which>.yaml       accuracy、loss、confusion_matrix 跟中繼資料
  <which>_preds.npz  逐樣本的 preds 跟 labels
"""
import argparse
import datetime
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import yaml

from data.src.nmnist import NMNISTDataset
from example.models.conv_net import N_CLASSES, build_decoder, build_growth_policies
from example.paths import DATASET_ROOT
from example.utils import EVAL_DIRNAME, load_run_params, load_run_record, make_evaluate


def _confusion_matrix(labels: np.ndarray, preds: np.ndarray, n_classes: int) -> np.ndarray:
    """列是真實類別、欄是預測類別。"""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (labels, preds), 1)
    return cm


def evaluate_run(exp_dir: str, which: str, n_samples: int | None,
                  seed: int, which_params: str) -> dict:
    run_record = load_run_record(exp_dir)
    model_cfg = run_record["config"]["model"]
    data_cfg = run_record["config"]["data"]

    network, params = load_run_params(exp_dir, which_params)
    layers = network.layers
    decoder = build_decoder(model_cfg, layers)
    decoder.validate(layers[-1])

    dataset = NMNISTDataset(DATASET_ROOT, max_events=int(data_cfg["max_events"]))
    if n_samples is None:
        n_samples = dataset.pool_size("test") if which == "test" else int(data_cfg["val_size"])
    split = dataset.build_split(seed=seed, n_samples=n_samples, which=which)

    eval_batch_size = int(run_record["config"]["train"]["batch_size"])
    evaluate = make_evaluate(network, decoder, eval_batch_size,
                             build_growth_policies(model_cfg, layers))
    accuracy, loss, preds, capacity_regrows = evaluate(params, split)
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
        "capacity_regrows": capacity_regrows,
        "confusion_matrix": confusion_matrix.tolist(),
        "git_commit": run_record.get("git_commit", ""),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exp_dir", help="一次訓練的輸出目錄")
    parser.add_argument("--which", choices=("test", "val"), default="test")
    parser.add_argument("--n", type=int, default=None,
                        help="抽幾筆(預設:test 全量、val 用 config 的 val_size)")
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
