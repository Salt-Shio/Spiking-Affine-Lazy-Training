"""驗證候選機制表格(docs/問題紀錄.md §十五)第一條「輸出無上限,曲率無界
增長」的另一種作法:不直接算 Hessian(surrogate gradient 讓二階微分沒有
唯一定義,見對話紀錄),改用逐 epoch 實際觀察到的權重變化量 ||θ_{t+1}-θ_t||
本身——梯度下降的線性穩定性分析(e_{t+1}=(1-ηλ)e_t)可以推出
Δθ_{t+1}/Δθ_t ≈ (1-ηλ),所以連續兩步的步長比值本身就約等於我們想找的
放大倍率,不需要碰任何二階微分。

用法:
  python -m example.analysis.weight_step_ratio <exp_dir_name>
"""
import argparse
import glob
import os
import re

import numpy as np

from example.models.conv_net import build_network
from example.paths import EXPERIMENTS_DIR
from example.utils import WEIGHTS_DIRNAME, load_params_npz, load_run_record

EXPLOSION_EPOCHS = [39, 49, 64, 75]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir_name")
    args = parser.parse_args()

    exp_dir = os.path.join(EXPERIMENTS_DIR, args.exp_dir_name)
    run_record = load_run_record(exp_dir)
    layers = build_network(run_record["config"]["model"])
    names = [layer.name for layer in layers]

    weights_dir = os.path.join(exp_dir, WEIGHTS_DIRNAME)
    paths = sorted(glob.glob(os.path.join(weights_dir, "epoch_*.npz")))
    epochs = sorted(int(re.search(r"epoch_(\d+)\.npz", p).group(1)) for p in paths)

    all_params = {}
    for epoch in epochs:
        path = os.path.join(weights_dir, f"epoch_{epoch:03d}.npz")
        all_params[epoch] = load_params_npz(path, layers)

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
