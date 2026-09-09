"""N-MNIST 事件相機資料集讀取 + 前處理。規格見 docs/規格書.md「N-MNIST 資料集」
與「data/ 資料前處理與視覺化架構規範」兩節,這裡只負責照規格實作。

原始格式(N-MNIST 官方 40-bit AER,規格書已用全部 70000 個檔案實測驗證):
每筆事件 5 bytes——byte0=X(8 bit)、byte1=Y(8 bit)、byte2 最高位=polarity、
byte2 低 7 bit + byte3 + byte4 組成 23-bit timestamp(微秒)。已額外抽樣驗證
(300 個檔案,涵蓋 Train/Test)檔案內事件本來就照 timestamp 遞增排序、
座標落在 [0,34) 範圍內,不需要另外排序或做範圍檢查以外的清理。

這裡是真實檔案的 host-side IO(逐檔讀 varying-length 二進位檔),沒辦法用 jax
向量化,前處理用 numpy 在 host 端做,最後才轉成 jax.Array。

**參數的所有權(規格書「data/ 資料前處理與視覺化架構規範」)**:所有「實驗可調」
的數字——事件截斷長度、Train/Val 切分比例與其 seed——都是 `NMNISTDataset` 的
建構參數,不是模組常數。模組層級只留「資料集本身 100% 不可調的事實」:影像尺寸、
類別數、polarity→channel 對照。截斷長度 `max_events` 沒有預設值,呼叫端(notebook /
config)一定要明確指定,不讓函式庫替實驗決定這個值。
"""
import glob
import os
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

#: 影像單軸尺寸(N-MNIST 官方規格,固定 34×34),資料集事實,不是可調參數。
IMG_SIZE = 34

#: channel index -> polarity 的唯一權威對照表,對應規格書「Polarity → channel」
#: 那一列(c=0 是 OFF、c=1 是 ON)。build_conv_queue 的 IC 軸、模型算輸入
#: channel 數的地方都要 import 這個常數,不要各自重寫字面值。資料集事實,
#: 不是可調參數。
CHANNEL_NAMES: tuple[str, ...] = ("off", "on")

#: 類別 index -> 名稱的唯一權威對照表(N-MNIST 就是手寫數字 0~9)。
#: 資料集事實,不是可調參數。
CLASS_NAMES: tuple[str, ...] = tuple(str(d) for d in range(10))

#: Train/ 底下 60000 筆切多少當 val、用哪顆 seed 切——這是「實驗設定」,擺成
#: `NMNISTDataset` 的建構參數預設值(規格書「Train/Val 切分」定案切 10%)。
#: 切分邊界要固定、不能受 build_split 呼叫端傳入的 sampling seed 影響——不然
#: 同一個 seed 換了 n_samples,train_pool/val_pool 的邊界會跟著漂移,兩次呼叫
#: 可能拿到有重疊的樣本。
_DEFAULT_VAL_FRACTION = 0.1
_DEFAULT_VAL_SPLIT_SEED = 0


def _decode_bin_file(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """讀一個 .bin 檔,回傳 (x, y, c, t_ms)——照檔案原生順序(已驗證等於時間
    遞增順序,規格書「同毫秒 tie-break:檔案原始順序」)。t_ms = t_微秒 // 1000
    (floor,規格書「時間量化」)。

    純檔案格式解碼,不吃任何實驗參數,維持模組層級函式。
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
    """列出 split_dir(Train 或 Test)底下全部 .bin 檔案跟對應標籤,照
    (digit, 檔名) 排序——決定性的原始順序,shuffle/抽樣交給呼叫端另外處理。

    純目錄掃描,不吃實驗參數,維持模組層級函式。
    """
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
    """N-MNIST 前處理器。一個實例綁定一組固定的實驗設定(截斷長度、Train/Val
    切分),`build_split` 用這組設定產出各個 split。

    Parameters
    ----------
    dataset_root:
        `N-MNIST/` 目錄(底下有 `Train/`、`Test/`,各自再分 0~9 子目錄)。
    max_events:
        每個樣本對齊的事件數(規格書「事件數截斷/padding 長度」)。超過的截斷成
        只取前 `max_events` 筆(事件已依時間遞增排序,取「前面」等於取「較早」);
        不足的 pad。**沒有預設值**——這是記憶體/實驗取捨的產物(規格書當前值
        2000,原本是全資料集真實最大值 8183),函式庫不替實驗決定,呼叫端一定要
        明確指定。
    val_fraction:
        Train/ 底下切多少比例當 val(規格書定案 10%)。
    val_split_seed:
        切 train_pool / val_pool 用的固定 seed——跟 `build_split` 的 sampling
        seed 分開,不然換 n_samples 時切分邊界會漂移。
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
        #: `_train_val_pools` 的快取。切分結果只跟 dataset_root / val_fraction /
        #: val_split_seed 有關,這三個建構後不變,所以整個實例算一次就好——
        #: build_split 每次呼叫(train / val 各一次、多個 split 反覆呼叫)不必
        #: 重新 glob 60000 個檔名、重新 permutation。
        self._train_val_pools_cache: tuple | None = None

    def _train_val_pools(self) -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
        """把 Train/ 底下 60000 筆用固定 `val_split_seed` 切成
        train_pool(1 - val_fraction)/val_pool(val_fraction),回傳
        (train_files, train_labels, val_files, val_labels)。第一次呼叫算完就
        快取在實例上,之後直接回傳同一份。"""
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

    def build_split(self, seed: int, n_samples: int, which: str) -> NMNISTSplit:
        """建一個 split。

        which: "train"/"val" 從 `_train_val_pools` 切出來的固定 pool 抽樣;"test"
          直接用 Test/ 全部 10000 筆當 pool。三個 pool 互不重疊。
        seed: 決定「這個 pool 裡抽哪 n_samples 筆、抽出來的順序」,不影響 pool
          本身的邊界(train_pool/val_pool 的切分固定用 `val_split_seed`)。

        每個樣本的事件數對齊 `self.max_events`:超過的截斷成只取前 max_events 筆
        (較早的事件);不足的 pad——x/y/c pad 成 0、event_times pad 成該樣本
        最後一筆真實時間(維持陣列非遞減,避免 diff 出現誤導性的負跳躍)。pad
        位置的實際數值不影響正確性,下游一律靠 n_real_events 搭配
        core.mask_pad_events 強制蓋成 identity 映射(見
        docs/math/conv事件佇列建構推導.md 第 8.1 節:pad 座標 (0,0,c=0) unravel 後
        看起來完全合法,必須靠 n_real_events 而不是座標合法性檢查來擋)。
        """
        if which == "train":
            pool_files, pool_labels, _, _ = self._train_val_pools()
        elif which == "val":
            _, _, pool_files, pool_labels = self._train_val_pools()
        elif which == "test":
            pool_files, pool_labels = _list_files(self.dataset_root, "Test")
        else:
            raise ValueError(f"which 必須是 'train'/'val'/'test',拿到 {which!r}")

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
                # 截斷(規格書 2026-09-03 改定案):只取前 max_events 筆——事件
                # 本來就照時間遞增排序,取「前面」等於取「較早」的事件,砍掉的是
                # 同一個樣本後面的掃視,不是隨機丟資料。理由/視覺化驗證見規格書
                # 該條目旁的說明,不重複列在這裡。
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
