"""「一堆各自獨立的 2D 圖排成網格」這種資料性質的還原 + 繪圖。

`unflatten_channels` 是 conv 層攤平神經元陣列的還原(通用幾何運算,不知道
`epoch`/`quantity` 這些字眼);`ImageGridPlot` 吃一串已經算好的 `(H, W)` 圖 +
標題,排成網格畫出來,每一格完全獨立、各自的色階範圍——不假設同一批圖之間
有任何關係(可能是不同 epoch、不同 quantity、不同 channel 的任意組合),所以
不像 `epoch_series` 的 `groups` 那樣把同組疊在一起比較,這裡「同時比較」就是
並排本身。

`ChannelGridAnimation` 是同一套色階邏輯的動畫版,但排版方式不一樣:呼叫端用
`add_row(*panels)` 一排一排加,每排各自的 panel 平分那排的寬度,排跟排之間
互不影響(改一排放幾個、放什麼,不會牽動別的排)。吃的是一串
`AnimatedPanel`(見該類別的說明)——每個 panel 自己知道要顯示什麼內容、要不要
離散色階、座標範圍、標籤,完全不知道自己會被排在第幾排、跟誰同一排。這裡也
完全不知道 panel 裡面裝的是 conv 還是 FC,只負責把每排的 panel 排出來、
播放。`(epoch, quantity, channel)` 這種三元組要解析成哪一張圖,是
`example/` 那邊實作 panel 物件時才知道的語意,不在這裡假設。

版面用 `matplotlib` 的 `fig.subfigures()`(每排一個獨立的子畫布)+
`layout="constrained"`,不是手動猜留白比例——`constrained_layout` 會實際
量每個 axes 的 title/label/colorbar 的文字大小去排間距,才不會像固定比例
常數那樣,換字體大小或圖高就重疊(這裡吃過虧)。
"""
import math
from typing import Protocol

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap, to_rgb

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
# 離散值「False」的底色——跟 ColorOverlayPanel 沒有任何 channel 亮的底色共用
# 同一個常數,兩種畫法看起來才是同一套視覺語言,不是疊色圖另外挑了一個顏色。
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
    """`ImageGridPlot` 用的版面配置:算 nrows/ncols、開 `Figure`,補不滿的格子
    隱藏,每一格固定同一個尺寸(`ImageGridPlot` 排的是任意不相關的圖,沒有
    「這格該多寬多高」這種資訊可以參考)。回傳 `(fig, 前 n 格的 axes list)`。"""
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
    `imshow` 的 `AxesImage`(動畫要用它的 `set_data` 逐 frame 更新)。`fig`
    只用來呼叫 `.colorbar(...)`,可以是 `Figure` 也可以是 `SubFigure`
    (兩者都有 `colorbar` 方法)。

    `extent`(`(left, right, bottom, top)`)給了才畫真實座標軸刻度(例如 FC
    neuron index / 相對時間偏移這種有意義的座標),同時代表這張圖的座標軸
    本身已經決定了顯示比例,所以連帶用 `aspect='auto'`(不套用 `imshow`
    預設的 `aspect='equal'`,不然座標軸的等比例會跟資料的 pixel 形狀打架);
    沒給 `extent` 就沿用 conv 那種「pixel 位置本身沒有座標意義」的預設,直接
    拿掉刻度,形狀改由呼叫端另外用 `ax.set_box_aspect` 控制。

    `img` 形狀 `(H, W)` 時走色階(離散/連續)+ colorbar 這條路;形狀
    `(H, W, 3)` 時代表已經是算好的 RGB 合成圖(例如 `ColorOverlayPanel` 把
    幾個 channel 疊成一張圖)——顏色本身就是最終呈現,不是「數值 map 到
    顏色」,不套 `cmap`/`vmin`/`vmax`,也不畫 colorbar(沒有單一數值刻度
    可以標)。"""
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


class AnimatedPanel(Protocol):
    """`ChannelGridAnimation.add_row` 吃的最小介面。任何物件只要有這些屬性 +
    `frame(t)` 方法就能丟進去——這裡不知道也不需要知道實際是什麼型別(conv
    層的某個 channel、FC 層的某段 neuron 時間窗口,或別的東西),那些語意
    知識是呼叫端(`example/`)的事,不在 `viz/` 假設。這個介面完全不帶任何
    位置資訊——panel 只負責描述「要畫什麼」,排在第幾排、跟誰同一排是呼叫端
    呼叫 `add_row` 時才決定的事,見 `ChannelGridAnimation`。"""
    title: str | None
    discrete: bool
    extent: tuple | None
    xlabel: str | None
    ylabel: str | None
    #: 連續值的色階固定範圍 `(vmin, vmax)`;離散值不需要,填 `None`。
    value_range: tuple | None
    n_frames: int

    def frame(self, t: int) -> np.ndarray:
        """回傳第 `t` 幀的圖:`(H, W)`(套色階,見 `discrete`/`value_range`)
        或已經算好的 `(H, W, 3)` RGB 合成圖(例如 `ColorOverlayPanel`,顏色
        本身就是最終呈現,不套色階)。同一個 panel 每次呼叫的形狀要一致
        (跨 panel 可以不一樣——不同形狀的 panel 混在同一組動畫時,各自維持
        自己該有的長寬比例,不會被拉伸,見 `ChannelGridAnimation.build`)。"""
        ...


class ColorOverlayPanel:
    """把幾個形狀一致的 `AnimatedPanel` 疊成一張 RGB 合成圖:每個 panel 配一
    個顏色(`(r, g, b)`,各 0~1),該 panel 的值當亮度乘上這個顏色疊加——
    多個 panel 同時在同一個像素有值,顏色直接相加(超過 1.0 的部分裁切到
    1.0)。不知道被疊的 panel 裡面裝的是 input 的哪個 channel、conv 的哪個
    channel——只是把幾個既有的 scalar panel 組合成一個新的、`frame(t)` 回傳
    RGB 的 panel,一樣可以丟進 `ChannelGridAnimation.add_row`(跟其他 panel
    同一排、或自己一排都行)。

    全部 panel 在某個像素都是 0(沒有任何 channel 亮)的地方畫 `background`
    (預設跟 `_DISCRETE_OFF_COLOR` 同一個顏色,離散圖「False」的底色)——
    不能讓「疊起來剛好全部是 0」跟「數學上加總出來的黑色」搞混,前者是
    「這裡沒事發生」,後者只是加法的副作用,兩者視覺上要跟其他離散圖的底色
    一致,不是另外發明一個黑色底。"""

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
        self.discrete = True   # 疊完是 RGB 圖,不套色階,這個欄位不影響呈現
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
    """跟 `ImageGridPlot` 同一套色階邏輯,但排版方式不一樣:用
    `add_row(*panels, height=..., widths=...)` 一排一排加,`height` 控制這
    排多高、`widths` 控制排裡每個 panel 多寬(兩個都不給就是全部等分),排跟
    排彼此完全獨立——改一排放幾個 panel、多高多寬,不會牽動別的排。用
    `imshow.set_data` 逐 frame 更新畫成動畫,不用每個 frame 重新畫整張圖。
    這裡完全不知道 panel 裡面裝的是 conv 還是 FC、`spike_mask`/`v_steps`/
    真實毫秒這些字眼——只負責排版、播放、colorbar/座標軸這些純繪圖的事。"""

    def __init__(self, subplot_size: tuple = (3.2, 2.8), cmap: str = "viridis"):
        self._subplot_size = subplot_size
        self._cmap = cmap
        self._rows: list[list] = []
        self._row_heights: list[float] = []
        self._row_widths: list[list | None] = []

    def add_row(self, *panels, height: float = 1.0,
                widths: list | None = None) -> "ChannelGridAnimation":
        """加一排。`height` 是這排跟其他排的高度比重(預設 1.0,全部排都用
        預設值就是等分整張畫布;某一排的圖天生比較扁,想要那排矮一點,
        `height=0.6` 之類的就好,不影響其他排)。`widths` 是這排裡每個 panel
        的寬度比重,跟 `panels` 對齊(不給就是預設全部等寬,跟以前一樣;
        想要某個 panel 比別的寬,例如 `widths=[2, 1]` 表示第一個是第二個的
        兩倍寬)。呼叫幾次就有幾排,由上到下照呼叫順序排。回傳 `self`,方便
        串接呼叫(`grid.add_row(a, b).add_row(c)`)。"""
        if not panels:
            raise ValueError("add_row 至少要給一個 panel")
        if widths is not None and len(widths) != len(panels):
            raise ValueError(f"widths 長度({len(widths)})要跟 panels 數量({len(panels)})一樣")
        self._rows.append(list(panels))
        self._row_heights.append(height)
        self._row_widths.append(widths)
        return self

    def build(self, frame_labels: list | None = None, interval: int = 50) -> FuncAnimation:
        """`frame_labels`(跟 panel 的 `n_frames` 對齊的字串 list,例如真實
        毫秒的顯示文字)給了就畫在 `fig.suptitle` 上隨 frame 更新,不給就不
        顯示。回傳 `matplotlib.animation.FuncAnimation`,不存檔(呼叫端的
        事)。"""
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
                # extent 給了代表座標軸已經決定了顯示比例(見 _style_axis);
                # 沒給的維持圖片自己真正的長寬比例,不會被同一排的其他 panel
                # 拉伸。
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
