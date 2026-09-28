"""example 測試共用的訓練設定、合成資料跟執行工具。"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from data.src.nmnist import CLASS_NAMES, NMNISTSplit
from example.models.conv_net import build_network
from example.train_conv_compressed import TrainResult, load_nmnist_data, train
from example.training.loop import TrainData, epoch_permutation
from example.training.run_dir import make_exp_dir
from example.utils import set_seed, split_raw_events, take_raw_events
from salt_core.capacity import reduce_over_batch
from salt_core.float.affine import safe_extra_steps


def run_train(cfg: dict, data: TrainData, root) -> TrainResult:
    """在 root 底下建實驗目錄,用 cfg 在 data 上跑 train()。"""
    return train(cfg, data, make_exp_dir(cfg["run_name"], str(root)))


# ============================================================================
# 真實資料:eval、replay 測試用的已訓練實驗目錄
# ============================================================================

def reference_cfg() -> dict:
    """真實 N-MNIST 的小規模訓練(conv 8 -> conv 16 -> FC 10):seed 42、容量給足不出界、
    2 epoch、每個 epoch 存權重快照。"""
    def _conv(oc, max_queue_len, max_out_spikes, init_k):
        return {"type": "conv", "oc": oc, "k": 3, "s": 2, "p": 1,
                "tau": 16.0, "v_th": 1.0, "alpha": 2.0, "chunk_size": 1,
                "max_queue_len": max_queue_len, "max_out_spikes": max_out_spikes, "init_k": init_k,
                "max_queue_len_grow_factor": 2.0, "out_grow_factor": 2.0}
    return {
        "run_name": "reference",
        "model": {
            "decoder": "membrane_regression",
            "input_shape": [2, 34, 34],
            "layers": [
                _conv(8, 185, 8000, 8.0),
                _conv(16, 5000, 35000, 64.0),
                {"type": "fc", "name": "out", "n_out": 10, "tau": 16.0,
                 "v_th": 1.0e9, "alpha": 2.0, "chunk_size": 512, "init_k": 5.0,
                 "max_out_spikes": 1},
            ],
        },
        "data": {"max_events": 2000, "train_size": 16, "val_size": 8,
                 "seed_train": 0, "seed_val": 0},
        "train": {"lr": 1.0e-2, "epochs": 2, "batch_size": 4, "seed": 42,
                  "weight_snapshot_every": 1, "dormant_layers": ["conv1", "conv2"]},
    }


def run_reference(root) -> TrainResult:
    cfg = reference_cfg()
    return run_train(cfg, load_nmnist_data(cfg["data"]), root)


# ============================================================================
# 合成資料:出界測試用
# ============================================================================

SYNTH_N_TRAIN = 9
SYNTH_BATCH_SIZE = 2       # 每個 epoch 4 個 batch,排在最後的 1 筆湊不滿 batch、不用
SYNTH_MAX_EVENTS = 16
SYNTH_OC = 4
SYNTH_GROW = 2.0           # 容量旋鈕的放大倍率
_PIXEL = 3                # 事件全部放在 (x, y) = (3, 3)

SYNTH_HIDDEN = 4          # 隱藏 FC 的神經元數
SYNTH_HIDDEN_CHUNK = 4    # 隱藏 FC 的 chunk_size;大於 1 時 fire 會多花步數
SYNTH_HIDDEN_INIT_K = 8.0

# 不測的旋鈕給理論上限,保證不會出界。k=3、s=2、p=1 時一個輸入位置最多落在 2 x 2 個輸出位置的
# 感受野裡,每筆輸入事件最多讓 4 x oc 顆神經元各 fire 一次;FC 每筆輸入事件最多讓每顆神經元
# 各 fire 一次。額外步數的上限是總步數等於輸入流長度(每步至少吃一筆)。
CONV1_QUEUE_BOUND = SYNTH_MAX_EVENTS
CONV1_OUT_BOUND = CONV1_QUEUE_BOUND * 4 * SYNTH_OC
CONV2_QUEUE_BOUND = CONV1_OUT_BOUND
CONV2_OUT_BOUND = CONV2_QUEUE_BOUND * 4 * SYNTH_OC
HIDDEN_OUT_BOUND = CONV2_OUT_BOUND * SYNTH_HIDDEN
HIDDEN_EXTRA_STEPS_BOUND = safe_extra_steps(CONV2_OUT_BOUND, SYNTH_HIDDEN_CHUNK)


class Synthetic(NamedTuple):
    """seed:訓練用的 seed。needs:每筆訓練樣本的事件數,也就是 conv1 的佇列需求。"""
    seed: int
    data: TrainData
    needs: np.ndarray


def _split(n_events: np.ndarray) -> NMNISTSplit:
    """第 i 筆樣本有 n_events[i] 筆事件,全部在同一個像素,時間 0, 1, 2, ... ms。"""
    n = len(n_events)
    times = np.zeros((n, SYNTH_MAX_EVENTS), dtype=np.int32)
    for i, count in enumerate(n_events):
        times[i, :count] = np.arange(count)
    pixel = np.full((n, SYNTH_MAX_EVENTS), _PIXEL, dtype=np.int32)
    labels = jnp.asarray(np.arange(n) % len(CLASS_NAMES), dtype=jnp.int32)
    return NMNISTSplit(event_times=jnp.asarray(times), x=jnp.asarray(pixel), y=jnp.asarray(pixel),
                       c=jnp.zeros((n, SYNTH_MAX_EVENTS), dtype=jnp.int32),
                       n_real_events=jnp.asarray(n_events, dtype=jnp.int32), labels=labels,
                       labels_onehot=jax.nn.one_hot(labels, len(CLASS_NAMES), dtype=jnp.float32))


def synthetic_setup() -> Synthetic:
    """合成的訓練、驗證資料跟 seed。

    挑一顆 seed,讓 epoch 0 湊不滿 batch 而沒用到的那筆,epoch 1 會用到。需求最大的樣本(12 筆
    事件)放在那個位置;epoch 0 其餘 8 筆照順序是 2, 3, ..., 9 筆事件,所以 epoch 0 各 batch 的
    最大需求是 3, 5, 7, 9。
    """
    seed = next(s for s in range(100)
                if epoch_permutation(s, SYNTH_N_TRAIN, 1)[-1]
                != epoch_permutation(s, SYNTH_N_TRAIN, 0)[-1])
    order = epoch_permutation(seed, SYNTH_N_TRAIN, 0)
    needs = np.empty(SYNTH_N_TRAIN, dtype=np.int32)
    needs[order[:-1]] = 2 + np.arange(SYNTH_N_TRAIN - 1)
    needs[order[-1]] = 12
    data = TrainData(train=_split(needs), val=_split(np.array([3, 5, 7, 9])))
    return Synthetic(seed=seed, data=data, needs=needs)


def batch_needs(synth: Synthetic, epoch: int) -> list[int]:
    """第 epoch 個 epoch 各 batch 的 conv1 最大佇列需求,依 batch 順序。"""
    order = epoch_permutation(synth.seed, SYNTH_N_TRAIN, epoch)
    n_batches = SYNTH_N_TRAIN // SYNTH_BATCH_SIZE
    return [int(synth.needs[order[b * SYNTH_BATCH_SIZE:(b + 1) * SYNTH_BATCH_SIZE]].max())
            for b in range(n_batches)]


def synthetic_cfg(run_name: str, seed: int, *, conv1: dict | None = None,
                  conv2: dict | None = None, hidden: dict | None = None,
                  grow: float = SYNTH_GROW, epochs: int = 3) -> dict:
    """合成資料用的網路(輸入 2x8x8、conv -> conv -> 會 fire 的 FC -> FC 10)跟訓練設定。

    conv1、conv2、hidden:覆寫該層的容量(例如 {"max_queue_len": 1}),沒給的旋鈕是理論上限。
    grow:所有旋鈕共用的放大倍率。縮小關掉,每個 epoch 存權重快照。
    """
    policy = {"max_queue_len_grow_factor": grow, "out_grow_factor": grow,
              "max_extra_steps_grow_factor": grow}

    def _conv(max_queue_len, max_out_spikes, overrides):
        entry = {"type": "conv", "oc": SYNTH_OC, "k": 3, "s": 2, "p": 1,
                 "tau": 16.0, "v_th": 1.0, "alpha": 2.0, "chunk_size": 1, "init_k": 8.0,
                 "max_queue_len": max_queue_len, "max_out_spikes": max_out_spikes, **policy}
        entry.update(overrides or {})
        return entry
    hidden_entry = {"type": "fc", "name": "hidden", "n_out": SYNTH_HIDDEN, "tau": 16.0,
                    "v_th": 1.0, "alpha": 2.0, "chunk_size": SYNTH_HIDDEN_CHUNK,
                    "init_k": SYNTH_HIDDEN_INIT_K, "max_out_spikes": HIDDEN_OUT_BOUND,
                    "max_extra_steps": HIDDEN_EXTRA_STEPS_BOUND, **policy}
    hidden_entry.update(hidden or {})
    return {
        "run_name": run_name,
        "model": {
            "decoder": "membrane_regression",
            "input_shape": [2, 8, 8],
            "layers": [
                _conv(CONV1_QUEUE_BOUND, CONV1_OUT_BOUND, conv1),
                _conv(CONV2_QUEUE_BOUND, CONV2_OUT_BOUND, conv2),
                hidden_entry,
                {"type": "fc", "name": "out", "n_out": len(CLASS_NAMES), "tau": 16.0,
                 "v_th": 1.0e9, "alpha": 2.0, "chunk_size": 256, "init_k": 5.0,
                 "max_out_spikes": 1},
            ],
        },
        "train": {"lr": 1.0e-2, "epochs": epochs, "batch_size": SYNTH_BATCH_SIZE, "seed": seed,
                  "max_steps_reestimate_every": 0, "weight_snapshot_every": 1},
    }


def first_batch_needed(cfg: dict, synth: Synthetic) -> dict:
    """照 cfg 的容量、用訓練開始時的初始權重,算第一個 batch 各層的需求:層名 -> 旋鈕名 -> 值。"""
    network = build_network(cfg["model"])
    params = network.init(set_seed(cfg["train"]["seed"]))
    idx = epoch_permutation(synth.seed, SYNTH_N_TRAIN, 0)[:SYNTH_BATCH_SIZE]
    raw = take_raw_events(split_raw_events(synth.data.train), idx)
    output = jax.jit(network.apply_batched)(params, raw)
    return {layer.name: {knob: int(v) for knob, v in reduce_over_batch(diag).needed.items()}
            for layer, diag in zip(network.layers, output.diags)}
