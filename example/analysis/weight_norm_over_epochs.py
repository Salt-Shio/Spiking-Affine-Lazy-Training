"""驗證 docs/問題紀錄.md「洞見:訓練中期梯度突然爆炸」
候選機制表格的第一條「輸出無上限,曲率無界增長」:
逐 epoch 讀權重快照(weight_snapshot_every=1),算每層 ||W||(Frobenius norm),看是否隨 epoch 增長、
增長的時間點是否對得上爆炸點(epoch 38->39、48->49、62->64、73->75)。

用法:
  python -m example.analysis.weight_norm_over_epochs <exp_dir_name>
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

    names = [layer.name for layer in load_weights(paths[0])[0].layers]
    print(f"找到 {len(epochs)} 份權重快照(epoch {epochs[0]}..{epochs[-1]})")
    print(f"\n{'epoch':>6} " + " ".join(f"{n + '_norm':>14}" for n in names))
    for epoch in epochs:
        path = os.path.join(weights_dir, f"epoch_{epoch:03d}.npz")
        _network, params = load_weights(path)
        norms = [float(np.linalg.norm(np.asarray(w))) for w in params]
        marker = " <-- 爆炸點" if epoch in EXPLOSION_EPOCHS else ""
        print(f"{epoch:>6} " + " ".join(f"{v:>14.4f}" for v in norms) + marker)


if __name__ == "__main__":
    main()
