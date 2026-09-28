"""data/ 的資源路徑。datasets 目錄在這個 package 底下,由本檔位置推得,不靠環境變數。
"""
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parent / "datasets" / "N-MNIST"
