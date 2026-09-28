"""example/ 共用的路徑。repo 根目錄由這個檔的位置推得,不靠環境變數;dataset 路徑轉出 data.paths 的。
"""
from pathlib import Path

from data.paths import DATASET_ROOT  # noqa: F401  轉出

REPO_ROOT = Path(__file__).resolve().parent.parent

CONFIGS_DIR = REPO_ROOT / "configs"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"


def resolve_config(path: str) -> Path:
    """命令列給的路徑 -> 絕對路徑:相對路徑一律相對 repo 根目錄,不看當前工作目錄。"""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p
