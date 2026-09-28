"""訓練 checkpoint 的存讀:接著練需要的全部狀態,每個 epoch 結束覆寫一份。

內容:網路描述(含當下容量)、權重、optimizer 狀態、shuffle key、epoch、best、紀錄
(已完成 epoch 的 metrics 列、容量事件)。先寫暫存檔再換名,當機不會留下寫一半的檔。
"""
import json
import os
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from salt_core.io import weights_from_arrays, weights_to_arrays
from salt_core.network import Network

_BEST_PREFIX = "best__"


class Best(NamedTuple):
    """val_accuracy 最好的 epoch。network 是那個 epoch 當下的網路(含容量)。"""
    params: tuple
    network: Network
    val_accuracy: float
    epoch: int


class CheckpointState(NamedTuple):
    network: Network      # 下一個 epoch 要用的網路(含容量)
    params: tuple         # 對齊 network.layers
    opt_state: object
    shuffle_key: jax.Array
    epoch: int            # 最後完成的 epoch,續練從 epoch+1 開始
    best: Best
    history: dict         # 可以存成 json 的紀錄:metrics_rows、capacity_events


class Checkpointer:
    """持有一個 checkpoint 檔路徑,負責覆寫 / 讀回。

    last_epoch:最近一次存或讀的 epoch,沒有時是 None,給出界訊息用。
    """

    def __init__(self, path: str):
        self.path = path
        self.last_epoch: int | None = None

    def exists(self) -> bool:
        return os.path.isfile(self.path)

    def save(self, state: CheckpointState) -> None:
        """權重照 salt_core.io 的格式存;opt_state 拆成 leaves 逐一存,讀回時用 template 組回去。"""
        opt_leaves, _ = jax.tree_util.tree_flatten(state.opt_state)
        arrays = weights_to_arrays(state.network, state.params)
        best_arrays = weights_to_arrays(state.best.network, state.best.params)
        arrays.update({_BEST_PREFIX + key: value for key, value in best_arrays.items()})
        arrays.update({f"opt_leaf__{i}": np.asarray(leaf) for i, leaf in enumerate(opt_leaves)})
        arrays["shuffle_key"] = np.asarray(state.shuffle_key)
        arrays["epoch"] = np.asarray(state.epoch)
        arrays["best_val_accuracy"] = np.asarray(state.best.val_accuracy)
        arrays["best_epoch"] = np.asarray(state.best.epoch)
        arrays["history"] = np.asarray(json.dumps(state.history))
        tmp_path = self.path.removesuffix(".npz") + ".tmp.npz"
        np.savez(tmp_path, **arrays)
        os.replace(tmp_path, self.path)
        self.last_epoch = int(state.epoch)

    def load(self, *, opt_state_template) -> CheckpointState:
        """opt_state_template 只用來取 treedef(同一個 optimizer 對同形狀權重 init 的結果),
        不使用數值。"""
        with np.load(self.path) as data:
            network, params = weights_from_arrays(data)
            best_network, best_params = weights_from_arrays(
                {key.removeprefix(_BEST_PREFIX): data[key] for key in data.files
                 if key.startswith(_BEST_PREFIX)})
            opt_leaves_template, treedef = jax.tree_util.tree_flatten(opt_state_template)
            opt_leaves = [jnp.asarray(data[f"opt_leaf__{i}"])
                          for i in range(len(opt_leaves_template))]
            state = CheckpointState(
                network=network, params=params,
                opt_state=jax.tree_util.tree_unflatten(treedef, opt_leaves),
                shuffle_key=jnp.asarray(data["shuffle_key"]), epoch=int(data["epoch"]),
                best=Best(params=best_params, network=best_network,
                          val_accuracy=float(data["best_val_accuracy"]),
                          epoch=int(data["best_epoch"])),
                history=json.loads(str(data["history"])))
        self.last_epoch = state.epoch
        return state
