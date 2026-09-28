"""各自獨立的 2D 圖排成網格。

ImageGridPlot:一串 (H, W) 圖加標題排成網格,每格各自的色階範圍,不假設圖之間有關係。
ChannelGridAnimation:動畫版。呼叫端用 add_row(*panels) 一排一排加,排跟排互不影響;panel 是符合
AnimatedPanel 的物件,自己決定要畫什麼,這裡不知道裡面是 conv 還是 FC。
版面用 fig.subfigures() 加 layout="constrained",由 matplotlib 量文字大小排間距;固定比例的留白在
換字體或圖高時會重疊。
"""
import math
from typing import Protocol

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap, to_rgb

# 離散值(由呼叫端 render(discrete=...) 指定)用兩色兩刻度,連續色階會冒出 0.25 這種沒意義的刻度。
# NaN(不同尺寸的層補齊的格子)畫成洋紅,跟離散的 False(灰)、連續的低值(深色)分得開。
_PAD_COLOR = "#ff00ff"
# 離散值 False 的底色,ColorOverlayPanel 沒有 channel 亮的地方也用同一個顏色
_DISCRETE_OFF_COLOR = "#d9d9d9"
_TWO_VALUE_CMAP = ListedColormap([_DISCRETE_OFF_COLOR, "#d62728"]).with_extremes(bad=_PAD_COLOR)


def _normalize_titles(titles: list | None, n: int) -> list:
    if titles is not None and len(titles) != n:
        raise ValueError(f"titles 長度({len(titles)})要跟項目數({n})一樣")
    return titles if titles is not None else [None] * n


def _normalize_discrete(discrete: bool | list | None, n: int) -> list:
    if discrete is None:
        return [False] * n
    if isinstance(discrete, bool):
        return [discrete] * n
    if len(discrete) != n:
        raise ValueError(f"discrete 長度({len(discrete)})要跟項目數({n})一樣")
    return list(discrete)


def _normalize_extents(extents: tuple | list | None, n: int) -> list:
    """extent 是 (left, right, bottom, top);單一 tuple 套到全部 n 格,None 用 imshow 的像素座標。"""
    if extents is None:
        return [None] * n
    if isinstance(extents, tuple):
        return [extents] * n
    if len(extents) != n:
        raise ValueError(f"extents 長度({len(extents)})要跟項目數({n})一樣")
    return list(extents)


def _make_grid_axes(n: int, ncols: int, subplot_size: tuple):
    """ImageGridPlot 的版面:算列數、開 Figure、補不滿的格子隱藏,每格同一個尺寸。
    回傳 (fig, 前 n 格的 axes list)。"""
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    w, h = subplot_size
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), squeeze=False)
    flat_axes = list(axes.flat)
    for ax in flat_axes[n:]:
        ax.set_visible(False)
    return fig, flat_axes[:n]


def _style_axis(fig, ax, img: np.ndarray, title: str | None, is_discrete: bool,
                 cmap: str, vmin: float | None = None, vmax: float | None = None,
                 extent: tuple | None = None, xlabel: str | None = None,
                 ylabel: str | None = None):
    """單格繪製,ImageGridPlot 跟 ChannelGridAnimation 共用。回傳 imshow 的 AxesImage(動畫用 set_data 更新)。

    fig: 呼叫 colorbar 用,Figure 或 SubFigure 都可以。
    img: (H, W) 時套色階(離散或連續)並畫 colorbar;(H, W, 3) 是已經算好的 RGB,不套色階、不畫 colorbar。
    extent: 給了才畫座標刻度,並用 aspect='auto';沒給時拿掉刻度,形狀由呼叫端用 set_box_aspect 控制。
    """
    aspect = "auto" if extent is not None else None
    if img.ndim == 3:
        im = ax.imshow(np.clip(img, 0.0, 1.0), extent=extent, aspect=aspect)
    elif is_discrete:
        im = ax.imshow(img, cmap=_TWO_VALUE_CMAP, vmin=0, vmax=1, extent=extent, aspect=aspect)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1])
        cbar.ax.set_yticklabels(["False", "True"])
    else:
        continuous_cmap = plt.get_cmap(cmap).with_extremes(bad=_PAD_COLOR)
        im = ax.imshow(img, cmap=continuous_cmap, vmin=vmin, vmax=vmax, extent=extent, aspect=aspect)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if title:
        ax.set_title(title, fontsize=9)
    if extent is None:
        ax.set_xticks([])
        ax.set_yticks([])
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    return im


class ImageGridPlot:
    """一串 (H, W) 圖排成網格,每格獨立標題、獨立色階範圍。ncols 決定每列幾格,補不滿的格子隱藏。"""

    def __init__(self, ncols: int = 4, subplot_size: tuple = (3.2, 2.8), cmap: str = "viridis"):
        self._ncols = ncols
        self._subplot_size = subplot_size
        self._cmap = cmap

    def render(self, images: list, titles: list | None = None, discrete: bool | list | None = None,
               extents: tuple | list | None = None, xlabel: str | None = None,
               ylabel: str | None = None):
        """畫一張新的 Figure 回傳,不存檔。

        discrete: 哪幾格用離散色階,單一 bool 套到全部,或跟 images 對齊的 list;None 全部是連續值。
        extents: 哪幾格畫座標刻度,(left, right, bottom, top) 單一 tuple 套到全部;沒給的拿掉刻度。
        xlabel、ylabel 套到全部格。
        """
        if not images:
            raise ValueError("images 是空的,沒有東西可畫")
        titles = _normalize_titles(titles, len(images))
        discrete = _normalize_discrete(discrete, len(images))
        extents = _normalize_extents(extents, len(images))

        fig, axes = _make_grid_axes(len(images), self._ncols, self._subplot_size)
        for ax, img, title, is_discrete, extent in zip(axes, images, titles, discrete, extents):
            _style_axis(fig, ax, np.asarray(img), title, is_discrete, self._cmap,
                        extent=extent, xlabel=xlabel, ylabel=ylabel)

        fig.tight_layout()
        return fig


class AnimatedPanel(Protocol):
    """ChannelGridAnimation.add_row 吃的介面:有下面這些屬性跟 frame(t) 的物件就行。
    panel 只描述要畫什麼,排在哪一排由呼叫 add_row 的人決定。"""
    title: str | None
    discrete: bool
    extent: tuple | None
    xlabel: str | None
    ylabel: str | None
    #: 連續值的色階範圍 (vmin, vmax);離散值填 None。
    value_range: tuple | None
    n_frames: int

    def frame(self, t: int) -> np.ndarray:
        """第 t 幀的圖:(H, W) 套色階,或 (H, W, 3) 已算好的 RGB。同一個 panel 每幀形狀要一樣。"""
        ...


class ColorOverlayPanel:
    """幾個形狀一樣的 panel 疊成一張 RGB:每個 panel 配一個顏色 (r, g, b)(各 0~1),值當亮度乘上顏色
    相加,超過 1 的裁到 1。本身也是一個 AnimatedPanel。

    所有 panel 在某個像素都是 0 的地方畫 background(預設同離散 False 的底色),不畫成加法算出的黑色。
    """

    def __init__(self, panels: list, colors: list, background: tuple = to_rgb(_DISCRETE_OFF_COLOR)):
        if not panels:
            raise ValueError("panels 是空的,沒有東西可疊")
        if len(panels) != len(colors):
            raise ValueError(f"panels 數量({len(panels)})要跟 colors 數量({len(colors)})一樣")
        n_frames_seen = {p.n_frames for p in panels}
        if len(n_frames_seen) != 1:
            raise ValueError(f"每個 panel 的 n_frames 要一樣,收到 {sorted(n_frames_seen)}")
        self._panels = panels
        self._colors = colors
        self._background = background
        self.n_frames = n_frames_seen.pop()
        self.discrete = True   # RGB 圖不套色階,這個值不影響呈現
        self.title = " + ".join(p.title for p in panels if p.title)
        self.extent = None
        self.xlabel = None
        self.ylabel = None
        self.value_range = None

    def frame(self, t: int) -> np.ndarray:
        frames = [np.asarray(p.frame(t), dtype=np.float64) for p in self._panels]
        shapes = {f.shape for f in frames}
        if len(shapes) != 1:
            raise ValueError(f"被疊的 panel 圖形狀要一致,收到 {sorted(shapes)}")
        h, w = frames[0].shape
        out = np.zeros((h, w, 3), dtype=np.float64)
        for frame, color in zip(frames, self._colors):
            for k in range(3):
                out[:, :, k] += frame * color[k]
        out = np.clip(out, 0.0, 1.0)
        any_active = np.any([f != 0 for f in frames], axis=0)
        out[~any_active] = self._background
        return out


class ChannelGridAnimation:
    """動畫網格:add_row 一排一排加 panel,用 set_data 逐幀更新。只管排版、播放、colorbar、座標軸。"""

    def __init__(self, subplot_size: tuple = (3.2, 2.8), cmap: str = "viridis"):
        self._subplot_size = subplot_size
        self._cmap = cmap
        self._rows: list[list] = []
        self._row_heights: list[float] = []
        self._row_widths: list[list | None] = []

    def add_row(self, *panels, height: float = 1.0,
                widths: list | None = None) -> "ChannelGridAnimation":
        """加一排,由上到下照呼叫順序。回傳 self,可以串接呼叫。

        height: 這排跟其他排的高度比重,預設 1.0(全部等分)。
        widths: 這排每個 panel 的寬度比重,跟 panels 對齊;不給就等寬。例如 [2, 1] 是第一個兩倍寬。
        """
        if not panels:
            raise ValueError("add_row 至少要給一個 panel")
        if widths is not None and len(widths) != len(panels):
            raise ValueError(f"widths 長度({len(widths)})要跟 panels 數量({len(panels)})一樣")
        self._rows.append(list(panels))
        self._row_heights.append(height)
        self._row_widths.append(widths)
        return self

    def build(self, frame_labels: list | None = None, interval: int = 50) -> FuncAnimation:
        """建動畫,回傳 FuncAnimation,不存檔。

        frame_labels: 跟 n_frames 對齊的字串(例如真實毫秒),給了就逐幀顯示在 suptitle。
        """
        if not self._rows:
            raise ValueError("還沒有任何一排 panel(先呼叫 add_row),沒有東西可畫")
        panels = [panel for row in self._rows for panel in row]
        n_frames_seen = {p.n_frames for p in panels}
        if len(n_frames_seen) != 1:
            raise ValueError(f"每個 panel 的 n_frames 要一樣,收到 {sorted(n_frames_seen)}")
        n_frames = n_frames_seen.pop()
        if n_frames == 0:
            raise ValueError("n_frames 是 0,沒有東西可畫")
        if frame_labels is not None and len(frame_labels) != n_frames:
            raise ValueError(f"frame_labels 長度({len(frame_labels)})要跟 "
                             f"n_frames({n_frames})一樣")

        base_w, base_h = self._subplot_size
        max_row_len = max(len(row) for row in self._rows)
        figsize = (base_w * max_row_len, base_h * sum(self._row_heights))
        fig = plt.figure(figsize=figsize, layout="constrained")
        subfigs = np.atleast_1d(fig.subfigures(nrows=len(self._rows), ncols=1,
                                                height_ratios=self._row_heights))

        ims = []
        for subfig, row, row_widths in zip(subfigs, self._rows, self._row_widths):
            axes = np.atleast_1d(subfig.subplots(1, len(row), width_ratios=row_widths))
            for ax, panel in zip(axes, row):
                img0 = np.asarray(panel.frame(0))
                vmin, vmax = (None, None) if panel.discrete else panel.value_range
                im = _style_axis(subfig, ax, img0, panel.title, panel.discrete, self._cmap,
                                  vmin, vmax, extent=panel.extent, xlabel=panel.xlabel,
                                  ylabel=panel.ylabel)
                # 有 extent 時比例由座標軸決定;沒有時維持圖片自己的長寬比,不被同一排的其他 panel 拉伸
                if panel.extent is None:
                    h, w = img0.shape[:2]  # RGB 合成圖是 (h, w, 3),只取前兩維
                    ax.set_box_aspect(h / w)
                ims.append((im, panel))

        suptitle = fig.suptitle(frame_labels[0]) if frame_labels else None

        def _update(frame_idx):
            updated = []
            for im, panel in ims:
                im.set_data(panel.frame(frame_idx))
                updated.append(im)
            if frame_labels:
                suptitle.set_text(frame_labels[frame_idx])
            return updated

        return FuncAnimation(fig, _update, frames=n_frames, interval=interval, blit=False)
