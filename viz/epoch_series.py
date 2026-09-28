"""逐 epoch 的純量表:讀檔跟繪圖。

資料形狀是 list[dict[str, float]],一列一個 epoch(epoch 欄是 int,其餘是 float,可能有 inf、nan),
例如 metrics.csv。EpochSeriesPlot.render 呼叫一次畫一次;哪些欄疊在同一張子圖由呼叫端用 groups 決定。
"""
import csv
import math

import matplotlib.pyplot as plt


def read_epoch_series_csv(path: str) -> list[dict]:
    """逐 epoch 一列的 csv 讀成 list[dict]:epoch 欄轉 int,其餘轉 float(inf、nan 字串也可以)。
    欄位照檔案表頭。"""
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = {"epoch": int(raw["epoch"])}
            for key, value in raw.items():
                if key != "epoch":
                    row[key] = float(value)
            rows.append(row)
    return rows


class EpochSeriesPlot:
    """每組一張子圖,x 軸是 epoch,同一組的欄疊成多條線(超過一條才加 legend)。

    groups: {子圖標題: [欄名, ...]},照順序排成網格;None 時每個非 epoch 欄自己一組。
    """

    def __init__(self, groups: dict | None = None, ncols: int = 4,
                 subplot_size: tuple = (3.2, 2.4)):
        self._groups = groups
        self._ncols = ncols
        self._subplot_size = subplot_size

    def render(self, rows: list):
        """畫一張新的 Figure 回傳,不存檔。"""
        if not rows:
            raise ValueError("rows 是空的,沒有東西可畫")
        epochs = [r["epoch"] for r in rows]
        groups = self._groups
        if groups is None:
            groups = {col: [col] for col in rows[0].keys() if col != "epoch"}

        names = list(groups.keys())
        n = len(names)
        ncols = min(self._ncols, n)
        nrows = math.ceil(n / ncols)
        w, h = self._subplot_size
        fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), squeeze=False)
        flat_axes = list(axes.flat)

        for ax, name in zip(flat_axes, names):
            cols = groups[name]
            for col in cols:
                values = [r[col] for r in rows]
                ax.plot(epochs, values, marker=".", markersize=3, linewidth=1,
                        label=col if len(cols) > 1 else None)
            ax.set_title(name, fontsize=9)
            ax.set_xlabel("epoch", fontsize=8)
            ax.tick_params(labelsize=7)
            if len(cols) > 1:
                ax.legend(fontsize=6)

        for ax in flat_axes[n:]:
            ax.set_visible(False)

        fig.tight_layout()
        return fig
