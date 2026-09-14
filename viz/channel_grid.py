"""「一堆各自獨立的 2D 圖排成網格」這種資料性質的還原 + 繪圖。

`unflatten_channels` 是 conv 層攤平神經元陣列的還原(通用幾何運算,不知道
`epoch`/`quantity` 這些字眼);`ImageGridPlot` 吃一串已經算好的 `(H, W)` 圖 +
標題,排成網格畫出來,每一格完全獨立、各自的色階範圍——不假設同一批圖之間
有任何關係(可能是不同 epoch、不同 quantity、不同 channel 的任意組合),所以
不像 `epoch_series` 的 `groups` 那樣把同組疊在一起比較,這裡「同時比較」就是
並排本身。`ChannelGridAnimation` 是同一套版面/色階邏輯的動畫版:吃一串疊起來
的 frame,逐 frame `set_data` 更新,給「依 `event_ms` 重取樣出來的一串快照」
播放用(見 `viz/time_resample.py`)。

哪個 `(epoch, quantity, channel)` 三元組要解析成哪一張圖、網格要擺幾格,都是
呼叫端(`example/`)的知識,不在這裡假設。
"""
import math

import numpy as np

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

# 離散值(例如 spike_mask)用固定兩色 + 兩檔 tick 的色階,不套連續色階(不然
# 色條會冒出 0.25/0.75 這種對兩值資料沒有意義的小數刻度)。哪一格是離散值
# 由呼叫端用 `render(discrete=...)` 明講——這裡不會去檢查 dtype 或值域猜,
# 猜值域曾經是為了 s_value(已拔除,forward 值恆為 0.0/1.0 的 float 陣列)
# 這個情境存在,現在沒有實際用例撐著,是死邏輯。
# NaN 畫成一個跟任何色階都不會搞混的顏色(洋紅)——`ChannelGridAnimation` 把
# 不同幾何尺寸的層 padding 進同一個陣列時,padding 出來的格子就是 NaN,要讓
# 呼叫端一眼看出「這裡沒有神經元」,不能讓它悄悄融進灰色(離散值的 False)
# 或 viridis 深色端(連續值的低值)裡變得無法分辨。
_PAD_COLOR = "#ff00ff"
_TWO_VALUE_CMAP = ListedColormap(["#d9d9d9", "#d62728"]).with_extremes(bad=_PAD_COLOR)


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
    """`extent` 是 `(left, right, bottom, top)`,單一 tuple 廣播到全部格、
    `None`(預設)= 沿用 `imshow` 自己的 pixel index 座標(conv 那種排列本來
    就沒有座標意義的情境)。"""
    if extents is None:
        return [None] * n
    if isinstance(extents, tuple):
        return [extents] * n
    if len(extents) != n:
        raise ValueError(f"extents 長度({len(extents)})要跟項目數({n})一樣")
    return list(extents)


def _make_grid_axes(n: int, ncols: int, subplot_size: tuple):
    """`ImageGridPlot`/`ChannelGridAnimation` 共用的版面配置:算 nrows/ncols、
    開 `Figure`,補不滿的格子隱藏。回傳 `(fig, 前 n 格的 axes list)`。"""
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
    """`ImageGridPlot`/`ChannelGridAnimation` 共用的單格繪製邏輯,回傳
    `imshow` 的 `AxesImage`(動畫要用它的 `set_data` 逐 frame 更新)。

    `extent`(`(left, right, bottom, top)`)給了才畫真實座標軸刻度(例如 FC
    neuron index / 相對時間偏移這種有意義的座標);沒給就沿用 conv 那種
    「pixel 位置本身沒有座標意義」的預設,直接拿掉刻度。"""
    aspect = "auto" if extent is not None else None
    if is_discrete:
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

    def render(self, images: list, titles: list | None = None, discrete: bool | list | None = None,
               extents: tuple | list | None = None, xlabel: str | None = None,
               ylabel: str | None = None):
        """回傳畫好的 `matplotlib.figure.Figure`,不存檔(呼叫端的事)。每次
        呼叫都從頭畫一張新的,沒有重複利用前一次的 Figure/Axes。

        `discrete`:哪幾格要用離散值(固定兩色)色階,呼叫端明講——這裡不看
        dtype、不看值域猜。單一 `bool` 廣播到全部格;跟 `images` 對齊的
        list 則逐格指定;`None`(預設)= 全部當連續值。

        `extents`:哪幾格要用真實座標軸刻度(`(left,right,bottom,top)`,單一
        tuple 廣播到全部格),沒給的格子維持原本「pixel 位置沒有座標意義」
        拿掉刻度的預設。`xlabel`/`ylabel` 套用到全部格。"""
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


class ChannelGridAnimation:
    """跟 `ImageGridPlot` 同一套版面/色階邏輯(`discrete` 明講,不猜),但吃
    一串疊起來的 frame,用 `imshow.set_data` 逐 frame 更新畫成動畫,不用每個
    frame 重新畫整張圖。給「依 `event_ms` 重取樣出來的一串 `(oc,h,w)` 快照」
    這種資料排成網格動畫用,不知道 `spike_mask`/`v_steps`/真實毫秒這些字眼。"""

    def __init__(self, ncols: int = 4, subplot_size: tuple = (3.2, 2.8), cmap: str = "viridis"):
        self._ncols = ncols
        self._subplot_size = subplot_size
        self._cmap = cmap

    def build(self, frames: np.ndarray, titles: list | None = None,
              discrete: bool | list | None = None, frame_labels: list | None = None,
              extents: tuple | list | None = None, xlabel: str | None = None,
              ylabel: str | None = None, interval: int = 50) -> FuncAnimation:
        """`frames` 形狀 `(n_frames, n_images, H, W)`。`titles`/`discrete`/
        `extents`(座標軸刻度,見 `ImageGridPlot.render`)/`xlabel`/`ylabel`
        跟 `ImageGridPlot.render` 同一套規則,逐格(`n_images`)指定,不逐
        frame 變動(窗口本身的座標範圍不會隨播放改變,只有畫面內容變)。
        `frame_labels`(跟 `n_frames` 對齊的字串 list,例如真實毫秒的顯示
        文字)給了就畫在 `fig.suptitle` 上隨 frame 更新,不給就不顯示。連續值
        (非 discrete)的色階範圍用整段 `frames` 的 min/max 固定住,動畫全程
        顏色可比較,不會每個 frame 自動重新縮放。回傳
        `matplotlib.animation.FuncAnimation`,不存檔(呼叫端的事)。"""
        frames = np.asarray(frames)
        if frames.ndim != 4:
            raise ValueError(f"frames 要是 (n_frames, n_images, H, W) 4 維,收到 shape={frames.shape}")
        n_frames, n_images = frames.shape[:2]
        if n_frames == 0 or n_images == 0:
            raise ValueError("frames 是空的,沒有東西可畫")
        titles = _normalize_titles(titles, n_images)
        discrete = _normalize_discrete(discrete, n_images)
        extents = _normalize_extents(extents, n_images)
        if frame_labels is not None and len(frame_labels) != n_frames:
            raise ValueError(f"frame_labels 長度({len(frame_labels)})要跟 "
                             f"n_frames({n_frames})一樣")

        fig, axes = _make_grid_axes(n_images, self._ncols, self._subplot_size)
        ims = []
        for j, (ax, title, is_discrete, extent) in enumerate(zip(axes, titles, discrete, extents)):
            vmin = vmax = None
            if not is_discrete:
                vmin = float(np.nanmin(frames[:, j]))
                vmax = float(np.nanmax(frames[:, j]))
            ims.append(_style_axis(fig, ax, frames[0, j], title, is_discrete,
                                    self._cmap, vmin, vmax, extent=extent,
                                    xlabel=xlabel, ylabel=ylabel))

        suptitle = fig.suptitle(frame_labels[0]) if frame_labels else None

        def _update(frame_idx):
            for j, im in enumerate(ims):
                im.set_data(frames[frame_idx, j])
            if frame_labels:
                suptitle.set_text(frame_labels[frame_idx])
            return ims

        fig.tight_layout()
        return FuncAnimation(fig, _update, frames=n_frames, interval=interval, blit=False)
