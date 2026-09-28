"""N-MNIST 事件相機資料集的讀取跟前處理。規格見 docs/規格書.md「N-MNIST 資料集」。

檔案是逐檔讀的二進位,用 numpy 在 host 端處理,最後才轉成 jax.Array。
常數只放資料集事實,可調的值是 NMNISTDataset 的建構參數,理由見 docs/問題紀錄.md
「決策:data/ 的常數只放資料集事實,實驗可調的值當建構參數」。
"""
import glob
import os
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

#: 影像單軸尺寸(N-MNIST 固定 34×34)。
IMG_SIZE = 34

#: channel index -> polarity:c=0 是 OFF、c=1 是 ON。模型的輸入 channel 數用這個算。
CHANNEL_NAMES: tuple[str, ...] = ("off", "on")

#: 類別 index -> 名稱(手寫數字 0~9)。
CLASS_NAMES: tuple[str, ...] = tuple(str(d) for d in range(10))

#: train/val 切分的預設值(規格書定 10%)。切分用自己的 seed,不受 build_split 的抽樣 seed 影響。
_DEFAULT_VAL_FRACTION = 0.1
_DEFAULT_VAL_SPLIT_SEED = 0


def _decode_bin_file(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """讀一個 .bin 檔,回傳 (x, y, c, t_ms),照檔案原本的順序(就是時間遞增)。

    每筆事件 5 bytes:byte0=X、byte1=Y、byte2 最高位=polarity、byte2 低 7 bit + byte3 + byte4 是
    23-bit timestamp(微秒)。t_ms = t_微秒 // 1000。檔案大小不是 5 的倍數時 raise ValueError。
    """
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size % 5 != 0:
        raise ValueError(f"{path} 檔案大小 {raw.size} bytes 不是 5 的倍數,"
                          f"不符合 N-MNIST 40-bit AER 格式")
    raw = raw.reshape(-1, 5)
    x = raw[:, 0].astype(np.int32)
    y = raw[:, 1].astype(np.int32)
    if x.min() < 0 or x.max() >= IMG_SIZE or y.min() < 0 or y.max() >= IMG_SIZE:
        raise ValueError(f"{path} 座標超出 [0,{IMG_SIZE}) 範圍:"
                          f"x=[{x.min()},{x.max()}], y=[{y.min()},{y.max()}]")
    c = (raw[:, 2] >> 7).astype(np.int32)  # polarity,對齊 CHANNEL_NAMES
    t_us = (((raw[:, 2] & 0x7F).astype(np.uint32) << 16) |
            (raw[:, 3].astype(np.uint32) << 8) | raw[:, 4].astype(np.uint32))
    t_ms = (t_us // 1000).astype(np.int32)
    return x, y, c, t_ms


def _list_files(dataset_root: str, split_dir: str) -> tuple[list[str], np.ndarray]:
    """split_dir(Train 或 Test)底下全部 .bin 檔跟標籤,照 (digit, 檔名) 排序。"""
    files: list[str] = []
    labels: list[int] = []
    for digit in range(len(CLASS_NAMES)):
        digit_dir = os.path.join(dataset_root, split_dir, str(digit))
        digit_files = sorted(glob.glob(os.path.join(digit_dir, "*.bin")))
        files.extend(digit_files)
        labels.extend([digit] * len(digit_files))
    return files, np.asarray(labels, dtype=np.int32)


class NMNISTSplit(NamedTuple):
    event_times: jax.Array     # shape (N, max_events) int32,ms,已排序
    x: jax.Array               # shape (N, max_events) int32
    y: jax.Array               # shape (N, max_events) int32
    c: jax.Array               # shape (N, max_events) int32,0=OFF,1=ON
    n_real_events: jax.Array   # shape (N,) int32,前 n_real_events[i] 筆是真的
    labels: jax.Array          # shape (N,) int32
    labels_onehot: jax.Array   # shape (N, 10)


class NMNISTDataset:
    """N-MNIST 前處理器。一個實例綁定一組實驗設定(截斷長度、train/val 切分),build_split 用它產出各個 split。

    dataset_root: N-MNIST/ 目錄,底下有 Train/、Test/,各自再分 0~9 子目錄。
    max_events: 每個樣本對齊的事件數,多的截掉(只留較早的事件),少的補 pad。沒有預設值。
    val_fraction: Train/ 切多少比例當 val。
    val_split_seed: 切 train/val 用的 seed,跟 build_split 的 seed 分開。
    """

    def __init__(self, dataset_root: str, max_events: int,
                 val_fraction: float = _DEFAULT_VAL_FRACTION,
                 val_split_seed: int = _DEFAULT_VAL_SPLIT_SEED) -> None:
        if max_events <= 0:
            raise ValueError(f"max_events 必須是正整數,拿到 {max_events}")
        if not 0.0 < val_fraction < 1.0:
            raise ValueError(f"val_fraction 必須落在 (0, 1),拿到 {val_fraction}")
        self.dataset_root = dataset_root
        self.max_events = int(max_events)
        self.val_fraction = float(val_fraction)
        self.val_split_seed = int(val_split_seed)
        # _train_val_pools 的快取:切分結果只跟建構參數有關,整個實例算一次
        self._train_val_pools_cache: tuple | None = None

    def _train_val_pools(self) -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
        """Train/ 用 val_split_seed 切成 train pool 跟 val pool,回傳 (train_files, train_labels,
        val_files, val_labels)。第一次算完快取在實例上。"""
        if self._train_val_pools_cache is not None:
            return self._train_val_pools_cache

        files, labels = _list_files(self.dataset_root, "Train")
        n_total = len(files)
        perm = np.random.RandomState(self.val_split_seed).permutation(n_total)
        n_val = int(round(n_total * self.val_fraction))
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        train_files = [files[i] for i in train_idx]
        val_files = [files[i] for i in val_idx]
        self._train_val_pools_cache = (train_files, labels[train_idx],
                                       val_files, labels[val_idx])
        return self._train_val_pools_cache

    def _pool(self, which: str) -> tuple[list[str], np.ndarray]:
        """which 對應的 pool:(檔案清單, 標籤)。which 不是 train/val/test 時 raise ValueError。"""
        if which == "train":
            pool_files, pool_labels, _, _ = self._train_val_pools()
        elif which == "val":
            _, _, pool_files, pool_labels = self._train_val_pools()
        elif which == "test":
            pool_files, pool_labels = _list_files(self.dataset_root, "Test")
        else:
            raise ValueError(f"which 必須是 'train'/'val'/'test',拿到 {which!r}")
        return pool_files, pool_labels

    def pool_size(self, which: str) -> int:
        """which("train"/"val"/"test")的 pool 總筆數,實際數檔案。"""
        return len(self._pool(which)[0])

    def build_split(self, seed: int, n_samples: int, which: str) -> NMNISTSplit:
        """從 which 的 pool 抽 n_samples 筆,建一個 split。

        which: "train"、"val" 從切好的 pool 抽;"test" 用 Test/ 全部 10000 筆。三個 pool 不重疊。
        seed: 決定抽哪幾筆、順序,不影響 pool 的邊界。
        每個樣本對齊 max_events:多的截掉;少的補 pad,x、y、c 補 0,event_times 補最後一筆真事件的時間
        (維持非遞減)。pad 靠 n_real_events 排除,座標 (0,0,0) 看起來是合法的,不能靠座標擋。
        """
        pool_files, pool_labels = self._pool(which)
        if n_samples > len(pool_files):
            raise ValueError(f"which={which!r} 的 pool 只有 {len(pool_files)} 筆,"
                              f"要求 n_samples={n_samples} 超過上限")

        max_events = self.max_events
        idx = np.random.RandomState(seed).choice(len(pool_files), size=n_samples, replace=False)

        event_times = np.zeros((n_samples, max_events), dtype=np.int32)
        xs = np.zeros((n_samples, max_events), dtype=np.int32)
        ys = np.zeros((n_samples, max_events), dtype=np.int32)
        cs = np.zeros((n_samples, max_events), dtype=np.int32)
        n_real = np.zeros((n_samples,), dtype=np.int32)
        labels = np.zeros((n_samples,), dtype=np.int32)

        for i, sample_idx in enumerate(idx):
            path = pool_files[sample_idx]
            x, y, c, t_ms = _decode_bin_file(path)
            n = x.shape[0]
            if n > max_events:
                # 只留前 max_events 筆:事件照時間遞增,前面就是較早的
                x, y, c, t_ms = x[:max_events], y[:max_events], c[:max_events], t_ms[:max_events]
                n = max_events
            event_times[i, :n] = t_ms
            if n < max_events:
                event_times[i, n:] = t_ms[-1] if n > 0 else 0
            xs[i, :n] = x
            ys[i, :n] = y
            cs[i, :n] = c
            n_real[i] = n
            labels[i] = pool_labels[sample_idx]

        labels_jax = jnp.asarray(labels)
        labels_onehot = jax.nn.one_hot(labels_jax, len(CLASS_NAMES), dtype=jnp.float32)

        return NMNISTSplit(
            event_times=jnp.asarray(event_times), x=jnp.asarray(xs), y=jnp.asarray(ys),
            c=jnp.asarray(cs), n_real_events=jnp.asarray(n_real),
            labels=labels_jax, labels_onehot=labels_onehot,
        )
