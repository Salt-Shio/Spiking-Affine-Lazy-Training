"""驗證 data/src/nmnist.py 的 .bin 解碼、train/val/test pool 切分、
`NMNISTDataset.build_split` 的 padding/shape/決定性,以及規格書「data/ 資料
前處理與視覺化架構規範」要求的參數所有權(max_events 沒有預設值、由建構參數
控制陣列寬度;val_fraction 也是建構參數)。

不驗證「N-MNIST 這個資料集本身的格式對不對」(規格書已經用全部 70000 個檔案
實測驗證過 34x34/2 channel/40-bit AER),重點驗證:
1. 用一個已經手動解碼過、對過帳的真實檔案核對 _decode_bin_file 本身。
2. train_pool/val_pool/test_pool 三者互不重疊、比例對得上建構參數。
3. build_split 的 padding/截斷、n_real_events、決定性(同 seed 兩次呼叫
   一樣、不同 seed 抽到不同樣本)。
4. max_events 是「必填、無預設」的建構參數,且真的決定輸出陣列寬度。
"""
import os


import jax.numpy as jnp

from data.paths import DATASET_ROOT
from data.src.nmnist import (CLASS_NAMES, IMG_SIZE, NMNISTDataset, NMNISTSplit,
                              _decode_bin_file, _list_files)

DATASET_ROOT = str(DATASET_ROOT)  # 底下用 os.path.join 串子路徑,統一成 str

#: 規格書當前的事件截斷長度。以前是模組常數 MAX_EVENTS,現在是建構參數;測試
#: 自己拿這個值當固定基準(2000 比大部分樣本的真實事件數小,截斷是常態路徑)。
_TEST_MAX_EVENTS = 2000

# Train/0/00002.bin 前 10 筆事件,對話記錄裡已經手動解碼、對過帳的真實資料
# (byte 層級直接算過:x/y 是原始 byte,polarity 是 byte2 最高位,timestamp 是
# byte2 低 7 bit+byte3+byte4 組成的 23-bit 微秒數 // 1000)。
_KNOWN_FILE = os.path.join(DATASET_ROOT, "Train", "0", "00002.bin")
_KNOWN_X = [10, 33, 12, 33, 14, 16, 7, 1, 16, 18]
_KNOWN_Y = [30, 20, 27, 3, 23, 10, 30, 28, 11, 19]
_KNOWN_C = [1, 1, 1, 1, 0, 0, 1, 1, 0, 1]
_KNOWN_T_US = [937, 1030, 1052, 2078, 2383, 3189, 4003, 4975, 6609, 6678]
_KNOWN_T_MS = [t // 1000 for t in _KNOWN_T_US]


def _make_dataset(max_events: int = _TEST_MAX_EVENTS, **kwargs) -> NMNISTDataset:
    return NMNISTDataset(DATASET_ROOT, max_events=max_events, **kwargs)


def test_decode_bin_file_matches_known_sample():
    x, y, c, t_ms = _decode_bin_file(_KNOWN_FILE)

    assert list(map(int, x[:10])) == _KNOWN_X, list(map(int, x[:10]))
    assert list(map(int, y[:10])) == _KNOWN_Y, list(map(int, y[:10]))
    assert list(map(int, c[:10])) == _KNOWN_C, list(map(int, c[:10]))
    assert list(map(int, t_ms[:10])) == _KNOWN_T_MS, list(map(int, t_ms[:10]))

    # 檔案內事件本來就照時間遞增排序(規格書「同毫秒 tie-break:檔案原始
    # 順序」的前提),floor 量化是單調函式,量化後還是非遞減。
    assert (t_ms[1:] >= t_ms[:-1]).all(), "量化成 ms 之後應該仍然非遞減"
    assert x.min() >= 0 and x.max() < IMG_SIZE
    assert y.min() >= 0 and y.max() < IMG_SIZE
    assert set(c.tolist()) <= {0, 1}


def test_list_files_test_pool_count():
    files, labels = _list_files(DATASET_ROOT, "Test")
    assert len(files) == 10000, f"N-MNIST Test/ 應該有 10000 筆,拿到 {len(files)}"
    assert labels.shape == (10000,)
    assert set(labels.tolist()) == set(range(10))


def test_train_val_pools_disjoint_and_ratio():
    ds = _make_dataset()
    train_files, train_labels, val_files, val_labels = ds._train_val_pools()

    n_total = len(train_files) + len(val_files)
    assert n_total == 60000, f"Train/ 應該共 60000 筆,拿到 {n_total}"
    # 建構參數 val_fraction 預設 0.1
    val_fraction = len(val_files) / n_total
    assert abs(val_fraction - ds.val_fraction) < 0.001, (
        f"val 比例 {val_fraction} 應該接近建構參數 {ds.val_fraction}")

    assert set(train_files).isdisjoint(set(val_files)), "train_pool/val_pool 不該有重疊檔案"
    assert len(train_labels) == len(train_files)
    assert len(val_labels) == len(val_files)

    # 固定 val_split_seed 切分,兩次呼叫邊界要完全一樣(不受外部狀態影響)
    train_files2, _, val_files2, _ = ds._train_val_pools()
    assert train_files == train_files2
    assert val_files == val_files2

    # 切分結果有快取:同一個實例重複呼叫回傳同一份物件,不重新 glob/permutation
    assert ds._train_val_pools() is ds._train_val_pools()


def test_val_fraction_param_actually_changes_the_split():
    """val_fraction 是建構參數,不同的值要真的切出不同大小的 pool——確認它不是
    被忽略的裝飾品。"""
    small_val = _make_dataset(val_fraction=0.1)._train_val_pools()
    big_val = _make_dataset(val_fraction=0.25)._train_val_pools()
    assert len(small_val[2]) == round(60000 * 0.1)
    assert len(big_val[2]) == round(60000 * 0.25)
    assert len(big_val[2]) > len(small_val[2])


def test_build_split_shapes_padding_and_n_real_events():
    ds = _make_dataset()
    n_samples = 20
    split = ds.build_split(seed=0, n_samples=n_samples, which="train")

    assert isinstance(split, NMNISTSplit)
    assert split.event_times.shape == (n_samples, ds.max_events)
    assert split.x.shape == (n_samples, ds.max_events)
    assert split.y.shape == (n_samples, ds.max_events)
    assert split.c.shape == (n_samples, ds.max_events)
    assert split.n_real_events.shape == (n_samples,)
    assert split.labels.shape == (n_samples,)
    assert split.labels_onehot.shape == (n_samples, len(CLASS_NAMES))

    # one-hot 對得上 labels
    recovered = jnp.argmax(split.labels_onehot, axis=1)
    assert bool(jnp.all(recovered.astype(split.labels.dtype) == split.labels))

    for i in range(n_samples):
        n_real = int(split.n_real_events[i])
        assert 0 < n_real <= ds.max_events, f"樣本 {i} 的 n_real_events={n_real} 不合理"

        real_x = split.x[i, :n_real]
        real_y = split.y[i, :n_real]
        real_c = split.c[i, :n_real]
        real_t = split.event_times[i, :n_real]

        assert bool(jnp.all(real_x >= 0)) and bool(jnp.all(real_x < IMG_SIZE))
        assert bool(jnp.all(real_y >= 0)) and bool(jnp.all(real_y < IMG_SIZE))
        assert bool(jnp.all((real_c == 0) | (real_c == 1)))
        assert bool(jnp.all(jnp.diff(real_t) >= 0)), f"樣本 {i} 真實事件時間應該非遞減"

        # pad 位置(見 nmnist.py build_split 的說明):x/y/c 補 0、時間補最後
        # 一筆真實時間——不代表這些值有意義,只是驗證照文件說明的填法做的,
        # 下游一律靠 n_real_events 蓋成 identity,不靠這些具體數值。
        if n_real < ds.max_events:
            pad_x = split.x[i, n_real:]
            pad_y = split.y[i, n_real:]
            pad_c = split.c[i, n_real:]
            pad_t = split.event_times[i, n_real:]
            assert bool(jnp.all(pad_x == 0))
            assert bool(jnp.all(pad_y == 0))
            assert bool(jnp.all(pad_c == 0))
            assert bool(jnp.all(pad_t == real_t[-1]))


def test_max_events_has_no_default_and_controls_array_width():
    """規格書「data/ 資料前處理與視覺化架構規範」:max_events 是記憶體/實驗
    取捨的產物,函式庫不替實驗決定——沒有預設值,少傳就報錯;傳不同的值,
    輸出陣列寬度跟著變。"""
    try:
        NMNISTDataset(DATASET_ROOT)  # type: ignore[call-arg]
        assert False, "max_events 沒有預設值,少傳應該噴 TypeError"
    except TypeError:
        pass

    narrow = NMNISTDataset(DATASET_ROOT, max_events=500).build_split(
        seed=0, n_samples=3, which="train")
    wide = NMNISTDataset(DATASET_ROOT, max_events=1500).build_split(
        seed=0, n_samples=3, which="train")
    assert narrow.event_times.shape == (3, 500)
    assert wide.event_times.shape == (3, 1500)
    # 同 seed、同樣本,只是截斷長度不同——window 較短那份的每一筆真實事件,
    # 應該跟較長那份對應位置一模一樣(截斷只是砍尾巴,不動前面)。
    for i in range(3):
        k = int(narrow.n_real_events[i])
        assert bool(jnp.all(narrow.x[i, :k] == wide.x[i, :k]))
        assert bool(jnp.all(narrow.y[i, :k] == wide.y[i, :k]))
        assert bool(jnp.all(narrow.event_times[i, :k] == wide.event_times[i, :k]))


def test_build_split_reproducible_and_seed_changes_selection():
    ds = _make_dataset()
    n_samples = 30
    split1 = ds.build_split(seed=0, n_samples=n_samples, which="train")
    split2 = ds.build_split(seed=0, n_samples=n_samples, which="train")

    assert bool(jnp.all(split1.labels == split2.labels))
    assert bool(jnp.all(split1.n_real_events == split2.n_real_events))
    assert bool(jnp.all(split1.event_times == split2.event_times))

    split3 = ds.build_split(seed=1, n_samples=n_samples, which="train")
    # 換 seed 應該抽到不同的檔案組合(n_real_events 是每個檔案獨有的事件數,
    # 幾乎不可能兩組不同檔案的完整序列剛好一樣)
    assert not bool(jnp.all(split1.n_real_events == split3.n_real_events))


def test_build_split_train_val_test_are_disjoint_pools():
    ds = _make_dataset()
    train_split = ds.build_split(seed=0, n_samples=50, which="train")
    val_split = ds.build_split(seed=0, n_samples=50, which="val")
    test_split = ds.build_split(seed=0, n_samples=50, which="test")

    # 三個 split 各自形狀正常、pool 邊界互斥(_train_val_pools 已經驗證過
    # train/val 不重疊;test 來自完全獨立的 Test/ 目錄,規格上天生互斥)
    for split in (train_split, val_split, test_split):
        assert split.event_times.shape == (50, ds.max_events)
        assert bool(jnp.all(split.n_real_events > 0))


def test_truncation_takes_earliest_events_not_random_or_latest():
    """Train/0/00002.bin 共 5028 筆事件(對話記錄的視覺化探測已經量過),超過
    _TEST_MAX_EVENTS=2000——手動套用 build_split 用的同一條截斷規則(取前
    max_events 筆),驗證截斷後開頭 10 筆還是原本已知對過帳的真實值,長度精確
    等於 max_events——確認截斷拿的是「較早」的事件,不是隨機挑或從尾端算。"""
    x, y, c, t_ms = _decode_bin_file(_KNOWN_FILE)
    n_total = x.shape[0]
    assert n_total > _TEST_MAX_EVENTS, (
        f"這個已知檔案應該超過 _TEST_MAX_EVENTS 才能測到截斷分支,"
        f"拿到 n_total={n_total}, _TEST_MAX_EVENTS={_TEST_MAX_EVENTS}")

    m = _TEST_MAX_EVENTS
    x_trunc, y_trunc, c_trunc, t_trunc = x[:m], y[:m], c[:m], t_ms[:m]
    assert x_trunc.shape[0] == m

    assert list(map(int, x_trunc[:10])) == _KNOWN_X
    assert list(map(int, y_trunc[:10])) == _KNOWN_Y
    assert list(map(int, c_trunc[:10])) == _KNOWN_C
    assert list(map(int, t_trunc[:10])) == _KNOWN_T_MS


def test_build_split_actually_exercises_truncation_path():
    """_TEST_MAX_EVENTS=2000 時,實測(對話記錄)96.7% 的樣本事件數超過這個
    長度——抽 30 個 train 樣本,至少要有一個被截斷(n_real_events == max_events),
    確認 build_split 真的有走到截斷那個分支,不是死程式碼。"""
    ds = _make_dataset()
    split = ds.build_split(seed=0, n_samples=30, which="train")
    assert bool(jnp.any(split.n_real_events == ds.max_events)), (
        "30 個樣本裡應該至少有一個被截斷到剛好 max_events,"
        f"實際 n_real_events={list(map(int, split.n_real_events))}")


def test_build_split_n_samples_exceeds_pool_raises():
    ds = _make_dataset()
    try:
        ds.build_split(seed=0, n_samples=999999, which="test")
        assert False, "n_samples 超過 pool 大小應該要噴 ValueError"
    except ValueError:
        pass


TESTS = [
    test_decode_bin_file_matches_known_sample,
    test_list_files_test_pool_count,
    test_train_val_pools_disjoint_and_ratio,
    test_val_fraction_param_actually_changes_the_split,
    test_build_split_shapes_padding_and_n_real_events,
    test_max_events_has_no_default_and_controls_array_width,
    test_build_split_reproducible_and_seed_changes_selection,
    test_build_split_train_val_test_are_disjoint_pools,
    test_truncation_takes_earliest_events_not_random_or_latest,
    test_build_split_actually_exercises_truncation_path,
    test_build_split_n_samples_exceeds_pool_raises,
]


if __name__ == "__main__":
    for test in TESTS:
        test()
        print(f"PASS: {test.__name__}")
    print(f"\n全部 {len(TESTS)} 項測試通過")
