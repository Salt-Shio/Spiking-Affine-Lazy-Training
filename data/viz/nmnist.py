"""N-MNIST 的視覺化。純函式,吃一個 matplotlib Axes、畫完回傳,排版跟存檔由呼叫端決定;
逐區段播放的互動控制放在 notebook(ipywidgets)。

兩種影像重建:
- accumulated_image:從頭累積到某一筆為止的全部事件,看到目前為止的完整輪廓。
- binned_image:只看一個時間區段內的事件,不含前面區段的。
bin_ms、up_to、start_ms、end_ms、bins、vmax 都是繪圖參數,由呼叫端給。
"""
import numpy as np

from data.src.nmnist import CHANNEL_NAMES, IMG_SIZE

# OFF、ON 兩個顏色,色盲可分辨
OFF_COLOR = "#2a78d6"
ON_COLOR = "#eb6834"


def _accumulate_counts(x: np.ndarray, y: np.ndarray, c: np.ndarray,
                        up_to: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """事件流累加成 (IMG_SIZE, IMG_SIZE) 的 OFF、ON 次數影像,回傳 (off_count, on_count)。

    up_to: 只取前幾筆(事件照時間遞增,前面就是較早的);None 是全部。
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
    """前 up_to 筆事件(None 是全部)畫成 ON 次數減 OFF 次數的差值影像:正值偏 ON_COLOR、負值偏
    OFF_COLOR、0 是中性灰。"""
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
    """只算時間落在 [start_ms, end_ms) 的事件,回傳 (off_count, on_count)。"""
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
    """畫 [start_ms, end_ms) 這一段的事件(差值影像,同 accumulated_image)。

    vmax: 色階上限;逐區段播放時各幀共用同一個值,顏色深淺才能互相比較。
    """
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
    """事件時間照 bin_ms 分段要幾格(涵蓋到最後一筆事件),給 slider 當上限。"""
    t_ms = np.asarray(t_ms)
    max_ms = float(t_ms.max()) if t_ms.size > 0 else 0.0
    return int(max_ms // bin_ms) + 1


def diff_vmax(x: np.ndarray, y: np.ndarray, c: np.ndarray) -> float:
    """整段事件的 |ON 次數 - OFF 次數| 最大值,當逐區段播放共用的色階上限。"""
    off_count, on_count = _accumulate_counts(x, y, c, up_to=None)
    return max(1.0, float(np.abs(on_count - off_count).max()))


def plot_time_histogram(t_ms: np.ndarray, c: np.ndarray, ax, bins: int = 30):
    """事件時間分布:全部事件用中性色長條當背景,疊上 OFF、ON 各自的分布(階梯線)。"""
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
