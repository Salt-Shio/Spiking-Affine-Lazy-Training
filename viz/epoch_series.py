"""「逐 epoch 純量表」這種資料性質的讀取 + 繪圖。

這種形狀:一列一個 epoch,每一欄是一個純量隨 epoch 累積的序列(`metrics.csv`
現在的樣子,也是 `MetricsLog.rows` 在記憶體裡的樣子)。跟資料來源、跟訓練是
否還在跑無關,轉出來都是同一個型別:`list[dict[str, float]]`(`epoch` 欄是
int,其餘欄是 float,含 inf/nan)。

`EpochSeriesPlot.render` 只認這個形狀,呼叫一次就是靜態畫一次;要動態呈現,
呼叫端自己決定何時、拿什麼樣的快照重複呼叫這同一個方法——渲染器不內建任何
「即時模式」,也不知道資料是從檔案讀來的還是從一個活的物件拿來的。

哪些欄該疊在同一張子圖裡比較(例如同一種指標、不同層)不是這裡的知識——那是
特定資料來源命名慣例的知識,由呼叫端透過 `groups` 傳入(見
`example/plot_metrics.py` 怎麼替 `metrics.csv` 的欄位組出分組)。不傳的話
退化成一欄一張子圖。

現況:只接了「讀 csv 靜態畫一次」這條路(`read_epoch_series_csv`)。接訓練中
即時來源(例如包一層薄殼讀 `MetricsLog.rows`)是之後的事,還沒做。
"""
import csv
import math

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt


def read_epoch_series_csv(path: str) -> list[dict]:
    """把逐 epoch 一列的 csv 讀成 `list[dict[str, float]]`。

    `epoch` 欄轉 int,其餘欄轉 float(`float()` 原生看得懂 `inf`/`nan` 字串,
    像 `metrics.csv` 裡休眠層的 `act_p90p10` 就會出現)。欄位有哪些、有幾欄
    不在這裡假設,照檔案表頭本身。
    """
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
    """把 `list[dict[str, float]]` 畫成一張圖:每組一張子圖,x 軸是 epoch,
    同一組裡的欄疊成多條線(> 1 條線才加 legend)。

    `groups`:`{子圖標題: [欄名, ...]}`,依傳入的順序排列成網格;`None` 時
    退化成「每個非 epoch 欄自己一組」,維持沒有分組資訊時的最小可用行為。
    """

    def __init__(self, groups: dict | None = None, ncols: int = 4,
                 subplot_size: tuple = (3.2, 2.4)):
        self._groups = groups
        self._ncols = ncols
        self._subplot_size = subplot_size

    def render(self, rows: list):
        """回傳畫好的 `matplotlib.figure.Figure`,不存檔——存不存、存哪裡是
        呼叫端的事(跟 `data/viz/nmnist.py`、`example/plot_eval.py` 同一個
        慣例)。每次呼叫都從頭畫一張新的:不重複利用前一次的 Figure/Axes,
        這裡先不處理「重畫效率」——呼叫一次是靜態用法、呼叫端自己重複呼叫是
        動態用法,兩者用的是同一份邏輯。
        """
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
