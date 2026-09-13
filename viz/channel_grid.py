"""「一堆各自獨立的 2D 圖排成網格」這種資料性質的還原 + 繪圖。

`unflatten_channels` 是 conv 層攤平神經元陣列的還原(通用幾何運算,不知道
`epoch`/`quantity` 這些字眼);`ImageGridPlot` 吃一串已經算好的 `(H, W)` 圖 +
標題,排成網格畫出來,每一格完全獨立、各自的色階範圍——不假設同一批圖之間
有任何關係(可能是不同 epoch、不同 quantity、不同 channel 的任意組合),所以
不像 `epoch_series` 的 `groups` 那樣把同組疊在一起比較,這裡「同時比較」就是
並排本身。

哪個 `(epoch, quantity, channel)` 三元組要解析成哪一張圖、網格要擺幾格,都是
呼叫端(`example/`)的知識,見 `example/notebooks/plot_channel_grid.ipynb`。
"""
import math

import numpy as np

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt


def unflatten_channels(flat: np.ndarray, oc: int, h: int, w: int) -> np.ndarray:
    """把 conv 層攤平的 `(n,)` 神經元陣列(`n = oc*h*w`,channel-major:
    `flat_index = c*h*w + y*w + x`)還原成 `(oc, h, w)`,單純 reshape,不用
    transpose(攤平公式見 `salt_core/connectivity/conv.py` 的佇列建構跟
    `salt_core/tests/test_conv_geometry.py`,兩處確認一致)。"""
    flat = np.asarray(flat)
    expected = oc * h * w
    if flat.shape[-1] != expected:
        raise ValueError(f"長度 {flat.shape[-1]} 跟 oc*h*w={expected}(oc={oc}, "
                         f"h={h}, w={w})對不上")
    return flat.reshape(oc, h, w)


class ImageGridPlot:
    """把一串 `(H, W)` 圖排成網格,每格獨立標題 + 獨立色階範圍(不假設不同格
    之間可以共用色階——它們可能是完全不同的量、不同的 channel、不同的
    epoch)。`ncols` 決定每列幾格,列數自動算,補不滿的格子隱藏,跟
    `viz.epoch_series.EpochSeriesPlot` 同一套版面邏輯。"""

    def __init__(self, ncols: int = 4, subplot_size: tuple = (3.2, 2.8), cmap: str = "viridis"):
        self._ncols = ncols
        self._subplot_size = subplot_size
        self._cmap = cmap

    def render(self, images: list, titles: list | None = None):
        """回傳畫好的 `matplotlib.figure.Figure`,不存檔(呼叫端的事)。每次
        呼叫都從頭畫一張新的,沒有重複利用前一次的 Figure/Axes。"""
        if not images:
            raise ValueError("images 是空的,沒有東西可畫")
        if titles is not None and len(titles) != len(images):
            raise ValueError(f"titles 長度({len(titles)})要跟 images 長度"
                             f"({len(images)})一樣")
        titles = titles if titles is not None else [None] * len(images)

        n = len(images)
        ncols = min(self._ncols, n)
        nrows = math.ceil(n / ncols)
        w, h = self._subplot_size
        fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), squeeze=False)
        flat_axes = list(axes.flat)

        for ax, img, title in zip(flat_axes, images, titles):
            im = ax.imshow(img, cmap=self._cmap)
            if title:
                ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for ax in flat_axes[n:]:
            ax.set_visible(False)

        fig.tight_layout()
        return fig
