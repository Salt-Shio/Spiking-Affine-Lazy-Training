"""`data/` 自己的資源路徑。datasets 目錄就在這個 package 底下,直接由本檔
位置推得,不需要 repo 根目錄、不靠環境變數。
"""
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parent / "datasets" / "N-MNIST"
