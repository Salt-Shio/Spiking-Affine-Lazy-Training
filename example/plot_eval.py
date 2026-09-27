"""把 `example/eval_test.py` 寫進 `eval/<which>.yaml` 的結果畫成一張圖,存回
同一個 `eval/` 子資料夾——這是這次訓練 run 的產物(跟 `train/metrics.csv`/
`train/params.npz` 同一類),不是資料集 EDA(那是 `data/viz/` + `notebooks/`
的事,兩者關注點不同:EDA 是互動瀏覽原始資料,這裡是把一次跑完的評估結果
存成報告用的圖檔)。

用法:
  python -m example.plot_eval <exp_dir> [--which test|val]

讀 <exp_dir>/eval/<which>.yaml,畫 confusion matrix,存
<exp_dir>/eval/<which>_confusion.png。
"""
import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import LinearSegmentedColormap

from data.src.nmnist import CLASS_NAMES
from example.utils import EVAL_DIRNAME
from viz.style import apply_style

# 循序色階(連續量值的 heatmap 用),跟 data/viz/nmnist.py 的 OFF_COLOR
# 同一組色票(dataviz skill references/palette.md 的藍色 100->700 階),
# 不用 matplotlib 內建的通用色盤。
_SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
                    "#184f95", "#0d366b"]
_CONFUSION_CMAP = LinearSegmentedColormap.from_list("confusion_blue", _SEQUENTIAL_BLUE)


def plot_confusion_matrix(cm: np.ndarray, class_names, ax, title: str | None = None):
    """把一份 `(n_classes, n_classes)` confusion matrix 畫在給定的 Axes 上,
    回傳 `(ax, im)`——存不存檔、排版是呼叫端的事。列 = 真實類別、欄 = 預測
    類別,每格標數字,依格子深淺切換文字顏色維持可讀對比。
    """
    im = ax.imshow(cm, cmap=_CONFUSION_CMAP)
    n = len(class_names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("預測類別")
    ax.set_ylabel("真實類別")
    if title:
        ax.set_title(title)

    vmax = cm.max() if cm.max() > 0 else 1
    for i in range(n):
        for j in range(n):
            color = "white" if cm[i, j] > vmax * 0.6 else "#1a1a1a"
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                    color=color, fontsize=8)
    return ax, im


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exp_dir", help="一次訓練的輸出目錄")
    parser.add_argument("--which", choices=("test", "val"), default="test")
    args = parser.parse_args()
    matplotlib.use("Agg")
    apply_style()

    yaml_path = os.path.join(args.exp_dir, EVAL_DIRNAME, f"{args.which}.yaml")
    with open(yaml_path, "r", encoding="utf-8") as f:
        result = yaml.safe_load(f)

    cm = np.asarray(result["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    _, im = plot_confusion_matrix(
        cm, CLASS_NAMES, ax,
        title=f"{args.which}  acc={result['accuracy']:.4f}  loss={result['loss']:.4f}")
    fig.colorbar(im, ax=ax, label="樣本數")
    fig.tight_layout()

    out_path = os.path.join(args.exp_dir, EVAL_DIRNAME, f"{args.which}_confusion.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"寫入 {out_path}")


if __name__ == "__main__":
    main()
