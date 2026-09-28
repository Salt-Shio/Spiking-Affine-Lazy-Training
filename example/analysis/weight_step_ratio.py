"""驗證 docs/問題紀錄.md「洞見:訓練中期梯度突然爆炸」
候選機制表格的第一條「輸出無上限,曲率無界增長」,另一種作法:
surrogate gradient 讓二階微分沒有唯一定義,不算 Hessian,改看逐 epoch 的權重變化量 ||θ_{t+1} - θ_t||。
線性穩定性分析 e_{t+1} = (1 - ηλ) e_t 推出 Δθ_{t+1} / Δθ_t ≈ 1 - ηλ,連續兩步的步長比就是放大倍率。

用法:
  python -m example.analysis.weight_step_ratio <exp_dir_name>
"""
import argparse
import glob
import os
import re

import numpy as np

from example.paths import EXPERIMENTS_DIR
from example.utils import WEIGHTS_DIRNAME
from salt_core.io import load_weights

EXPLOSION_EPOCHS = [39, 49, 64, 75]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir_name")
    args = parser.parse_args()

    exp_dir = os.path.join(EXPERIMENTS_DIR, args.exp_dir_name)
    weights_dir = os.path.join(exp_dir, WEIGHTS_DIRNAME)
    paths = sorted(glob.glob(os.path.join(weights_dir, "epoch_*.npz")))
    epochs = sorted(int(re.search(r"epoch_(\d+)\.npz", p).group(1)) for p in paths)

    all_params = {}
    for epoch in epochs:
        path = os.path.join(weights_dir, f"epoch_{epoch:03d}.npz")
        all_params[epoch] = load_weights(path)[1]

    def flat(params):
        return np.concatenate([np.asarray(w).ravel() for w in params])

    steps = {}
    for t in epochs[:-1]:
        steps[t] = flat(all_params[t + 1]) - flat(all_params[t])

    print(f"{'epoch t->t+1':>14} {'step_norm':>12} {'ratio vs prev':>14}")
    prev_norm = None
    for t in epochs[:-1]:
        norm = float(np.linalg.norm(steps[t]))
        ratio_str = f"{norm / prev_norm:>14.4f}" if prev_norm else f"{'':>14}"
        marker = " <-- 爆炸(loss 這個 epoch 噴出)" if (t + 1) in EXPLOSION_EPOCHS else ""
        print(f"{t:>6}->{t+1:<6} {norm:>12.6f} {ratio_str}{marker}")
        prev_norm = norm


if __name__ == "__main__":
    main()
