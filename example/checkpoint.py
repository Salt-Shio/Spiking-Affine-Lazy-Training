"""訓練 checkpoint 的存 / 讀。

壓縮版訓練的動態放大會「退回最近的 checkpoint + 重編譯續練」,所以每個 epoch 結束
覆蓋寫一份最新的。存網路描述(含當下容量)、權重、optimizer 狀態、shuffle key、epoch。
"""
import os
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from salt_core.io import weights_from_arrays, weights_to_arrays
from salt_core.network import Network


class CheckpointState(NamedTuple):
    network: Network      # 存檔當下的網路(含容量)
    params: tuple         # 對齊 network.layers
    opt_state: object
    shuffle_key: jax.Array
    epoch: int            # 這份 checkpoint 最後完成的 epoch,續練從 epoch+1 開始


class Checkpointer:
    """持有一個 checkpoint 檔路徑,負責覆蓋寫 / 讀回。

    `_make_exp_dir` 每次訓練都開一個新的時間戳目錄,所以 `exists()`(檔在不在)
    等價於「這次 run 存過 checkpoint 沒」。`last_epoch` 是最近一次 `save()` 對應
    的 epoch(沒存過是 `None`),只給出界訊息用、不參與邏輯判斷。
    """

    def __init__(self, path: str):
        self.path = path
        self.last_epoch: int | None = None

    def exists(self) -> bool:
        return os.path.isfile(self.path)

    def save(self, *, network: Network, params, opt_state, shuffle_key: jax.Array,
             epoch: int) -> None:
        """權重跟網路描述照 salt_core.io 的格式存;opt_state 是巢狀 pytree,拆成 leaves
        逐一存,讀回時用 opt_state_template 的 treedef 組回去。"""
        opt_leaves, _ = jax.tree_util.tree_flatten(opt_state)
        save_dict = weights_to_arrays(network, params)
        save_dict.update({f"opt_leaf__{i}": np.asarray(leaf)
                          for i, leaf in enumerate(opt_leaves)})
        save_dict["shuffle_key"] = np.asarray(shuffle_key)
        save_dict["epoch"] = np.asarray(epoch)
        np.savez(self.path, **save_dict)
        self.last_epoch = int(epoch)

    def load(self, *, opt_state_template) -> CheckpointState:
        """opt_state_template 只用來取 treedef(同一個 optimizer 對同形狀權重 init 的結果),
        不使用數值。"""
        with np.load(self.path) as data:
            network, params = weights_from_arrays(data)
            opt_leaves_template, treedef = jax.tree_util.tree_flatten(opt_state_template)
            opt_leaves = [jnp.asarray(data[f"opt_leaf__{i}"])
                          for i in range(len(opt_leaves_template))]
            return CheckpointState(network=network, params=params,
                                   opt_state=jax.tree_util.tree_unflatten(treedef, opt_leaves),
                                   shuffle_key=jnp.asarray(data["shuffle_key"]),
                                   epoch=int(data["epoch"]))
