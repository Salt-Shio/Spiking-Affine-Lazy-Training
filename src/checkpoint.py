"""訓練 checkpoint 的存 / 讀。

從 `train_conv_compressed.py` 拆出來(reviewer 早就要「checkpoint 用另一個檔
管理」)。壓縮版訓練的動態放大會「退回最近的 checkpoint + 重編譯續練」,所以
每個 epoch 結束覆蓋寫一份最新的。

**只還原 params / opt_state / shuffle_key / epoch**——容量旋鈕(各層的 L /
max_out_spikes)不存:沒有跨行程 resume,放大後的 layer list 活在 `train()`
的 local scope、跨 while 迴圈迭代累積。
"""
import os

import jax
import jax.numpy as jnp
import numpy as np


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

    def save(self, *, params, opt_state, shuffle_key: jax.Array, epoch: int) -> None:
        """params 按位置各存一個陣列;opt_state 是 optax 巢狀 pytree,np.savez
        只能存平面陣列,先 tree_flatten 拆成 leaves 逐一存,讀回時用同一個
        optimizer 對同形狀 params 重新 init 拿到相同 treedef 再 unflatten。"""
        opt_leaves, _ = jax.tree_util.tree_flatten(opt_state)
        save_dict = {f"param__{i}": np.asarray(w) for i, w in enumerate(params)}
        save_dict.update({f"opt_leaf__{i}": np.asarray(leaf)
                          for i, leaf in enumerate(opt_leaves)})
        save_dict["shuffle_key"] = np.asarray(shuffle_key)
        save_dict["epoch"] = np.asarray(epoch)
        np.savez(self.path, **save_dict)
        self.last_epoch = int(epoch)

    def load(self, *, params_template, opt_state_template):
        """template 只用來取結構(層數、opt_state 的 treedef),不使用數值。
        回傳 `(params, opt_state, shuffle_key, epoch)`——params 是對齊 layer
        list 的位置 tuple,epoch 是這份 checkpoint「最後完成」的 epoch 編號
        (續練從 epoch+1 開始)。"""
        data = np.load(self.path)
        params = tuple(jnp.asarray(data[f"param__{i}"])
                       for i in range(len(params_template)))
        opt_leaves_template, treedef = jax.tree_util.tree_flatten(opt_state_template)
        opt_leaves = [jnp.asarray(data[f"opt_leaf__{i}"])
                      for i in range(len(opt_leaves_template))]
        opt_state = jax.tree_util.tree_unflatten(treedef, opt_leaves)
        shuffle_key = jnp.asarray(data["shuffle_key"])
        epoch = int(data["epoch"])
        return params, opt_state, shuffle_key, epoch
