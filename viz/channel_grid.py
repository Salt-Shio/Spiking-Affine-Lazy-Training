"""「一堆各自獨立的 2D 圖排成網格」這種資料性質的還原 + 繪圖。

`unflatten_channels` 是 conv 層攤平神經元陣列的還原(通用幾何運算,不知道
`epoch`/`quantity` 這些字眼);`ImageGridPlot` 吃一串已經算好的 `(H, W)` 圖 +
標題,排成網格畫出來,每一格完全獨立、各自的色階範圍——不假設同一批圖之間
有任何關係(可能是不同 epoch、不同 quantity、不同 channel 的任意組合),所以
不像 `epoch_series` 的 `groups` 那樣把同組疊在一起比較,這裡「同時比較」就是
並排本身。`ChannelGridAnimation` 是同一套版面/色階邏輯的動畫版,但吃的是一串
`AnimatedPanel`(見該類別的說明)——每個 panel 自己知道要顯示什麼內容、要不要
離散色階、座標範圍、標籤,這裡只負責排版、播放,完全不知道 panel 裡面裝的是
conv 還是 FC。`(epoch, quantity, channel)` 這種三元組要解析成哪一張圖,是
`example/` 那邊實作 panel 物件時才知道的語意,不在這裡假設。
"""
import math
from typing import Protocol

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


def _cells_spanned(row_spec, col_spec) -> list:
    """`row_spec`/`col_spec` 各自是 `int`(單一格)或 `slice`(橫跨一段範圍,
    跟 `gridspec[row, col]` 原生語法一致)。回傳這個 position 實際佔用的所有
    `(row, col)` 格子,給重疊檢查/自動排版避讓用。"""
    rows = [row_spec] if isinstance(row_spec, int) else list(range(row_spec.start, row_spec.stop))
    cols = [col_spec] if isinstance(col_spec, int) else list(range(col_spec.start, col_spec.stop))
    return [(r, c) for r in rows for c in cols]


def _make_ratio_grid_axes(panels: list, shapes: list, ncols: int, subplot_size: tuple):
    """`ChannelGridAnimation` 用的版面配置:每一格分到的實際空間跟著這一格
    panel 自己的長寬比例走,不是每格都用同一個尺寸硬套——不然形狀差很多的
    panel(conv 接近正方形、FC 窗口又寬又扁)雖然靠 `set_box_aspect` 不會被拉
    伸變形,但硬塞進同樣大的格子裡,形狀跟原本比例差越多的那格,實際能用的
    面積就越小,看起來像是被「壓縮」。

    自動規則:把每個 panel 的長寬比例 `aspect = h/w` 換算成一組寬/高的配置
    比重(`1/sqrt(aspect)`、`sqrt(aspect)`),讓每個 panel 分到的『面積』大致
    相等,只有形狀(寬高比)不同。同一欄/列裡有多個 panel 時,那欄/列的寬/高
    取最大值(要放得下最需要空間的那個)。

    想手動蓋掉自動規則的話,在 panel 物件自己身上設(不是這裡的參數):

    - `panel.position = (row, col)` 指定要放在第幾列第幾欄。`row`/`col`
      各自可以是單一整數,也可以是 `slice(start, stop)` 橫跨好幾列/欄(例如
      `slice(0, 2)` 橫跨欄 0~1)——橫跨的那個維度不會貢獻寬/高比重給任何
      單一欄/列(橫跨多欄本來就沒有「這一欄該多寬」這種單一答案),而是直接
      吃橫跨範圍內、由其他沒有橫跨的 panel 已經決定好的總寬度,不會逼任何
      一欄/列被迫跟著變寬/變高。沒設 `position` 的 panel 自動排進剩下沒被
      佔用的格子,可以只挑幾個 panel 設,其他維持自動。
    - `panel.size_ratio = (寬倍率, 高倍率)` 直接指定這一格的寬高倍率,蓋掉
      用圖片形狀自動算出來的比重(橫跨多欄/列的那個維度會被忽略,理由同上)。

    兩者都是 panel 物件的一般屬性,`viz/` 這裡只是讀,不知道也不管是誰、為
    什麼設的。回傳 `(fig, 對齊 panels 順序的 axes list)`。"""
    n = len(panels)
    positions = [getattr(p, "position", None) for p in panels]

    occupied = set()
    for pos in positions:
        if pos is None:
            continue
        for cell in _cells_spanned(pos[0], pos[1]):
            if cell in occupied:
                raise ValueError(f"panel.position 有重疊的格子:{cell}")
            occupied.add(cell)

    auto_ncols = min(ncols, n)
    max_explicit_row = max((r for r, _ in occupied), default=-1)
    max_explicit_col = max((c for _, c in occupied), default=-1)
    nrows = max(math.ceil(n / auto_ncols), max_explicit_row + 1)
    ncols = max(auto_ncols, max_explicit_col + 1)

    free_cells = ((row, col) for row in range(nrows) for col in range(ncols)
                  if (row, col) not in occupied)
    resolved_positions = []
    for pos in positions:
        if pos is not None:
            resolved_positions.append(pos)
            continue
        try:
            resolved_positions.append(next(free_cells))
        except StopIteration:
            raise ValueError("自動排版的格子不夠放——明講的 panel.position "
                             "跟自動排版的其他 panel 衝突太多") from None

    # `None` = 這欄/列還沒有任何 panel 貢獻過比重——不能拿固定的 1.0 當底線
    # 再取 max,不然明講 `size_ratio` 想要小於 1.0 的高度會被硬拉回 1.0(這
    # 裡曾經真的這樣壞過)。真的沒有 panel 落在的欄/列才補回 1.0 當預設。
    width_ratios = [None] * ncols
    height_ratios = [None] * nrows
    for panel, (h, w), (row_spec, col_spec) in zip(panels, shapes, resolved_positions):
        size_ratio = getattr(panel, "size_ratio", None)
        if size_ratio is not None:
            width_ratio, height_ratio = size_ratio
        else:
            aspect = h / w
            width_ratio, height_ratio = 1.0 / math.sqrt(aspect), math.sqrt(aspect)
        # 橫跨多欄/列的那個維度不貢獻比重(見上面說明),只有單一格的維度才算。
        if isinstance(col_spec, int):
            width_ratios[col_spec] = (width_ratio if width_ratios[col_spec] is None
                                      else max(width_ratios[col_spec], width_ratio))
        if isinstance(row_spec, int):
            height_ratios[row_spec] = (height_ratio if height_ratios[row_spec] is None
                                       else max(height_ratios[row_spec], height_ratio))
    width_ratios = [1.0 if r is None else r for r in width_ratios]
    height_ratios = [1.0 if r is None else r for r in height_ratios]

    base_w, base_h = subplot_size
    # 用 constrained layout,不要事後呼叫 fig.tight_layout()——tight_layout
    # 對「格子寬高比例不一致 + 每格都掛了 colorbar」這種自訂 GridSpec 會重新
    # 推算版面,把明講的 position/size_ratio 排版整個打亂(實測會發生,不是
    # 猜的)。constrained layout 從一開始就照著這個 GridSpec 調間距,不會事後
    # 重新洗牌。
    fig = plt.figure(figsize=(base_w * sum(width_ratios), base_h * sum(height_ratios)),
                      layout="constrained")
    gridspec = fig.add_gridspec(nrows, ncols, width_ratios=width_ratios, height_ratios=height_ratios)
    axes = [fig.add_subplot(gridspec[row, col]) for row, col in resolved_positions]
    return fig, axes


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


class AnimatedPanel(Protocol):
    """`ChannelGridAnimation.build` 吃的最小介面。任何物件只要有這些屬性 +
    `frame(t)` 方法就能丟進去——這裡不知道也不需要知道實際是什麼型別(conv
    層的某個 channel、FC 層的某段 neuron 時間窗口,或別的東西),那些語意
    知識是呼叫端(`example/`)的事,不在 `viz/` 假設。"""
    title: str | None
    discrete: bool
    extent: tuple | None
    xlabel: str | None
    ylabel: str | None
    #: 連續值的色階固定範圍 `(vmin, vmax)`;離散值不需要,填 `None`。
    value_range: tuple | None
    n_frames: int

    def frame(self, t: int) -> np.ndarray:
        """回傳第 `t` 幀的 `(H, W)` 圖。同一個 panel 每次呼叫的形狀要一致
        (跨 panel 可以不一樣——不同形狀的 panel 混在同一組動畫時,各自維持
        自己該有的長寬比例,不會被拉伸,見 `ChannelGridAnimation.build`)。"""
        ...

    # 以下兩個是選填的,不設就維持自動排版(見 `_make_ratio_grid_axes`)——
    # 不是每個 panel 都要有,`viz/` 用 `getattr(panel, "position", None)` 讀,
    # 沒設就是 `None`。
    #: 手動指定要放在第幾列第幾欄 `(row, col)`,蓋掉自動照 list 順序排列。
    position: tuple | None
    #: 手動指定這一格的 `(寬倍率, 高倍率)`,蓋掉根據圖片形狀自動算出來的比重。
    size_ratio: tuple | None


class ChannelGridAnimation:
    """跟 `ImageGridPlot` 同一套版面/色階邏輯,但吃一串「知道怎麼呈現自己」
    的 panel 物件(見 `AnimatedPanel`),用 `imshow.set_data` 逐 frame 更新
    畫成動畫,不用每個 frame 重新畫整張圖。這裡完全不知道 panel 裡面裝的是
    conv 還是 FC、`spike_mask`/`v_steps`/真實毫秒這些字眼——只負責排版、
    播放、colorbar/座標軸這些純繪圖的事。"""

    def __init__(self, ncols: int = 4, subplot_size: tuple = (3.2, 2.8), cmap: str = "viridis"):
        self._ncols = ncols
        self._subplot_size = subplot_size
        self._cmap = cmap

    def build(self, panels: list, frame_labels: list | None = None,
              interval: int = 50) -> FuncAnimation:
        """`panels`:一串 `AnimatedPanel`。`frame_labels`(跟 panel 的
        `n_frames` 對齊的字串 list,例如真實毫秒的顯示文字)給了就畫在
        `fig.suptitle` 上隨 frame 更新,不給就不顯示。回傳
        `matplotlib.animation.FuncAnimation`,不存檔(呼叫端的事)。"""
        if not panels:
            raise ValueError("panels 是空的,沒有東西可畫")
        n_frames_seen = {p.n_frames for p in panels}
        if len(n_frames_seen) != 1:
            raise ValueError(f"每個 panel 的 n_frames 要一樣,收到 {sorted(n_frames_seen)}")
        n_frames = n_frames_seen.pop()
        if n_frames == 0:
            raise ValueError("n_frames 是 0,沒有東西可畫")
        if frame_labels is not None and len(frame_labels) != n_frames:
            raise ValueError(f"frame_labels 長度({len(frame_labels)})要跟 "
                             f"n_frames({n_frames})一樣")

        frame0s = [np.asarray(p.frame(0)) for p in panels]
        fig, axes = _make_ratio_grid_axes(panels, [img0.shape for img0 in frame0s],
                                           self._ncols, self._subplot_size)
        ims = []
        for ax, panel, img0 in zip(axes, panels, frame0s):
            vmin, vmax = (None, None) if panel.discrete else panel.value_range
            im = _style_axis(fig, ax, img0, panel.title, panel.discrete, self._cmap,
                              vmin, vmax, extent=panel.extent, xlabel=panel.xlabel,
                              ylabel=panel.ylabel)
            # 不同 panel 形狀可能差很多(conv 接近正方形、FC 窗口又寬又扁)——
            # 沒有明講 size_ratio 時,讓每個 panel 維持自己真正的長寬比例,
            # 不會被同一格的框硬拉伸;明講了 size_ratio 就是呼叫端自己選擇要
            # 拉寬/壓扁,box 形狀改跟著 size_ratio 走,不是被資料的真實比例
            # 卡住。
            size_ratio = getattr(panel, "size_ratio", None)
            if size_ratio is not None:
                width_ratio, height_ratio = size_ratio
                ax.set_box_aspect(height_ratio / width_ratio)
            else:
                h, w = img0.shape
                ax.set_box_aspect(h / w)
            ims.append(im)

        suptitle = fig.suptitle(frame_labels[0]) if frame_labels else None

        def _update(frame_idx):
            for im, panel in zip(ims, panels):
                im.set_data(panel.frame(frame_idx))
            if frame_labels:
                suptitle.set_text(frame_labels[frame_idx])
            return ims

        return FuncAnimation(fig, _update, frames=n_frames, interval=interval, blit=False)
