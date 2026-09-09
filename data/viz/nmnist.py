"""N-MNIST 資料集的視覺化工具。純函式,吃一個 `matplotlib.axes.Axes`、畫完
回傳,呼叫端(notebook)自己負責排版、存不存檔。互動控制項(逐時間區段的
slider/播放)也放 notebook——用
`ipywidgets` 直接綁 `binned_image`,不在函式庫裡包一個自己開視窗的東西
(規格書「data/ 資料前處理與視覺化架構規範」:notebooks 換成真正的 Jupyter,
影片檢視在 notebook 裡即時重畫,不是獨立 GUI 視窗、也不是預先渲染好的幀)。

N-MNIST 每個樣本是「事件流重建出一張圖」,不是「座標點本身就是資料」,
所以這裡有兩種不同用途的影像重建:

- `accumulated_image`:從 t=0 累積到某個時間點為止的全部事件——這正是規格書
  「事件數截斷長度為什麼改成 2000」那次調查用來判斷「輪廓多早就看得出來」的
  同一套邏輯,看的是「到目前為止的完整輪廓」。
- `binned_image`:只看**單一時間區段內**的事件,不累積前面區段的——例如
  [5,6) ms 這格只顯示 5~6ms 之間發生的事件,0~4ms 的不該留在畫面上。逐區段
  播放時 notebook 用 `ipywidgets` 的 slider 反覆呼叫這個函式,每次只重畫
  當下那一格。

參數的定位(規格書「data/ 資料前處理與視覺化架構規範」):`bin_ms`、`up_to`、
`start_ms`/`end_ms`、`bins`、`vmax` 都是純繪圖旋鈕,沒有「要跟資料集對齊」的
問題,由 notebook 明確傳入。

色盤:ON/OFF 是極性(polarity),用兩個通過色盲安全門檻驗證的 hex 值
(dataviz skill references/palette.md 分類色盤 slot 1/2)。
"""
import numpy as np

from data.src.nmnist import CHANNEL_NAMES, IMG_SIZE

# 極性(OFF/ON)專用的兩個色值,通過色盲安全門檻驗證(dataviz skill
# references/palette.md 分類色盤 slot 1/2)。
OFF_COLOR = "#2a78d6"
ON_COLOR = "#eb6834"


def _accumulate_counts(x: np.ndarray, y: np.ndarray, c: np.ndarray,
                        up_to: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """把事件流累加成 (IMG_SIZE, IMG_SIZE) 的 OFF/ON 次數影像。`up_to` 是只取
    前幾筆事件(事件本來就照時間遞增排序,取「前面」等於取「較早」,跟
    data/src/nmnist.py 的截斷邏輯是同一個假設);None 代表全部事件都累加。
    回傳 (off_count, on_count),不在這裡先相減——`accumulated_image` 才決定
    要不要畫成差值,拆開回傳給其他用途(例如只想看 ON 或只想看 OFF)保留彈性。
    """
    n = x.shape[0] if up_to is None else min(up_to, x.shape[0])
    off_count = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.int32)
    on_count = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.int32)
    xs, ys, cs = np.asarray(x[:n]), np.asarray(y[:n]), np.asarray(c[:n])
    np.add.at(off_count, (ys[cs == 0], xs[cs == 0]), 1)
    np.add.at(on_count, (ys[cs == 1], xs[cs == 1]), 1)
    return off_count, on_count


def accumulated_image(x: np.ndarray, y: np.ndarray, c: np.ndarray, ax,
                       up_to: int | None = None, title: str | None = None):
    """把事件流(前 `up_to` 筆,None 代表全部)畫成一張 ON-count 減 OFF-count
    的差值影像——正值(ON 較多)用 ON_COLOR 方向、負值(OFF 較多)用 OFF_COLOR
    方向、0 是中性灰,對應 dataviz skill「diverging = 兩色 + 中性灰中點」的
    規則,不是隨便挑一個 matplotlib colormap。"""
    off_count, on_count = _accumulate_counts(x, y, c, up_to)
    diff = (on_count - off_count).astype(np.float32)
    vmax = max(1.0, float(np.abs(diff).max()))

    im = ax.imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="upper")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    n_shown = x.shape[0] if up_to is None else min(up_to, x.shape[0])
    ax.set_title(title if title is not None else f"accumulated events (n={n_shown})")
    return ax, im


def _counts_in_range(x: np.ndarray, y: np.ndarray, c: np.ndarray, t_ms: np.ndarray,
                      start_ms: float, end_ms: float) -> tuple[np.ndarray, np.ndarray]:
    """只算時間落在 [start_ms, end_ms) 這個區段內的事件,不含區段以外的
    (前面、後面都不算)——`_accumulate_counts` 是「到某個時間點為止全部」,
    這個函式是「只有這個區段當下」,兩種語意不能混用。"""
    t_ms = np.asarray(t_ms)
    mask = (t_ms >= start_ms) & (t_ms < end_ms)
    off_count = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.int32)
    on_count = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.int32)
    xs, ys, cs = np.asarray(x)[mask], np.asarray(y)[mask], np.asarray(c)[mask]
    np.add.at(off_count, (ys[cs == 0], xs[cs == 0]), 1)
    np.add.at(on_count, (ys[cs == 1], xs[cs == 1]), 1)
    return off_count, on_count


def binned_image(x: np.ndarray, y: np.ndarray, c: np.ndarray, t_ms: np.ndarray, ax,
                  start_ms: float, end_ms: float, vmax: float | None = None):
    """畫出**單一時間區段** [start_ms, end_ms) 內的事件(不含區段外的),跟
    `accumulated_image` 的差別見模組開頭說明。`vmax` 讓多個區段共用同一個
    色階範圍(逐區段播放時,顏色深淺才能在不同幀之間直接比較,不會因為每幀
    自己重算 vmax 而失真)。"""
    off_count, on_count = _counts_in_range(x, y, c, t_ms, start_ms, end_ms)
    diff = (on_count - off_count).astype(np.float32)
    if vmax is None:
        vmax = max(1.0, float(np.abs(diff).max()))

    im = ax.imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="upper")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    n_shown = int(off_count.sum() + on_count.sum())
    ax.set_title(f"t in [{start_ms:g}, {end_ms:g}) ms (n={n_shown})")
    return ax, im


def n_time_bins(t_ms: np.ndarray, bin_ms: float) -> int:
    """給定事件時間跟區段寬度,算逐區段播放要幾格(涵蓋到最後一筆事件)。
    notebook 的 slider 上限用這個算,繪圖邏輯本身在 `binned_image`。"""
    t_ms = np.asarray(t_ms)
    max_ms = float(t_ms.max()) if t_ms.size > 0 else 0.0
    return int(max_ms // bin_ms) + 1


def diff_vmax(x: np.ndarray, y: np.ndarray, c: np.ndarray) -> float:
    """整段事件的 |ON-count 減 OFF-count| 最大值,當逐區段播放共用的色階上限
    (每格自己重算 vmax 的話,不同幀的顏色深淺沒辦法直接比較)。"""
    off_count, on_count = _accumulate_counts(x, y, c, up_to=None)
    return max(1.0, float(np.abs(on_count - off_count).max()))


def plot_time_histogram(t_ms: np.ndarray, c: np.ndarray, ax, bins: int = 30):
    """全體事件時間分佈(中性色長條,當背景基準)疊上 OFF/ON 各自的分佈
    (階梯線)。這裡的 channel 直接就是 `c` 陣列本身,不需要 event_source_idx
    反查的額外一步。"""
    t_ms = np.asarray(t_ms); c = np.asarray(c)
    bin_edges = np.linspace(t_ms.min(), t_ms.max(), bins + 1)

    ax.hist(t_ms, bins=bin_edges, color="#898781", alpha=0.5, density=True, label="all events")
    for ch, color in ((0, OFF_COLOR), (1, ON_COLOR)):
        ax.hist(t_ms[c == ch], bins=bin_edges, histtype="step", linewidth=1.5, color=color,
                density=True, label=CHANNEL_NAMES[ch])

    ax.set_xlabel("event time (ms)")
    ax.set_ylabel("density")
    ax.set_title("Event time distribution (pooled vs OFF/ON)")
    ax.legend(fontsize=8, framealpha=0.9)
    return ax
