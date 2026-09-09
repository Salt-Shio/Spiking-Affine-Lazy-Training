"""共用小工具:決定性種子設定、git commit hash——訓練跑起來之後,每一次
`experiments/` 記錄都要能回答「這是哪個 code 版本、哪個亂數種子跑出來的」。
"""
import subprocess

import jax
import numpy as np


def set_seed(seed: int) -> jax.Array:
    """設定 numpy 的全域亂數種子,回傳一個 JAX PRNGKey。

    JAX 沒有「設一次全域種子,之後所有呼叫都決定性」這種東西——呼叫端要自己
    把回傳的 key 一路往下 `jax.random.split`/傳遞下去,沒有明確傳 key 的
    `jax.random` 呼叫,JAX 本身就不允許,不是這裡沒設好。這個函式主要是為了
    (1) 順便處理少數還會用到 numpy 亂數的地方(例如某些第三方套件內部),
    (2) 統一入口,呼叫端不用自己記兩套種子設定方式。
    """
    np.random.seed(seed)
    return jax.random.PRNGKey(seed)


def get_git_commit_hash(repo_dir: str | None = None) -> str:
    """回傳 repo_dir(預設目前工作目錄)所在 git 倉庫的 commit hash。

    抓不到就回傳 "unknown"(例如根本不在 git 倉庫裡、或環境沒裝 git)——
    這只是實驗記錄的附加資訊,不該因為抓不到就讓整個訓練/評估腳本掛掉。
    """
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir,
                                capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return "unknown"
