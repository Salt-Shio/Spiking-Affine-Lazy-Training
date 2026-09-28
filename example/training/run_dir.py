"""實驗目錄的寫入:experiments/<run>/train/ 底下的 run.yaml、metrics.csv、params.npz、best_params.npz。

讀取端(load_run_record、load_run_params、weight_snapshot_path)在 example/utils.py,
評估、replay、分析腳本共用。
"""
import datetime
import os

import yaml

from example.paths import REPO_ROOT
from example.utils import TRAIN_DIRNAME, get_git_commit_hash
from salt_core.io import save_weights


def make_exp_dir(run_name: str, exp_root: str) -> str:
    """建 <exp_root>/conv_compressed_<run_name>_<時間戳>/train/,回傳 run 目錄。"""
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(exp_root, f"conv_compressed_{run_name}_{date_str}")
    os.makedirs(os.path.join(exp_dir, TRAIN_DIRNAME), exist_ok=True)
    return exp_dir


def run_header(cfg: dict) -> dict:
    """run.yaml 開頭的欄位:config 快照、git commit、XLA_FLAGS、時間。"""
    return {
        "config": cfg,
        "git_commit": get_git_commit_hash(str(REPO_ROOT)),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def write_run_record(exp_dir: str, run_record: dict) -> None:
    """整份覆寫 <exp_dir>/train/run.yaml。"""
    with open(os.path.join(exp_dir, TRAIN_DIRNAME, "run.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(run_record, f, allow_unicode=True, sort_keys=False)


def write_weights(exp_dir: str, network, params, best) -> None:
    """params.npz 帶結束時的網路,best_params.npz 帶 best epoch 當下的網路。"""
    train_dir = os.path.join(exp_dir, TRAIN_DIRNAME)
    save_weights(os.path.join(train_dir, "params.npz"), network, params)
    save_weights(os.path.join(train_dir, "best_params.npz"), best.network, best.params)


def metrics_csv_path(exp_dir: str) -> str:
    return os.path.join(exp_dir, TRAIN_DIRNAME, "metrics.csv")
