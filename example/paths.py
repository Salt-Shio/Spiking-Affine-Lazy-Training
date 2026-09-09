"""應用層(`example/`)共用的路徑錨點。

repo 根目錄由這個檔案自己的位置往上一層推得(`example/paths.py` -> repo 根),
不靠環境變數、不動 `sys.path`。dataset 路徑由 `data/` 自己提供(`data.paths`),
這裡只轉出來,讓訓練腳本一次 import 拿齊。
"""
from pathlib import Path

from data.paths import DATASET_ROOT  # noqa: F401  (轉出,讓 example.paths 當單一入口)

REPO_ROOT = Path(__file__).resolve().parent.parent

CONFIGS_DIR = REPO_ROOT / "configs"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"


def resolve_config(path: str) -> Path:
    """把命令列給的 config 路徑解成絕對路徑:絕對路徑原樣用,相對路徑一律
    相對 repo 根目錄(不是當前工作目錄),讓在任何 cwd 下行為一致。"""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p
