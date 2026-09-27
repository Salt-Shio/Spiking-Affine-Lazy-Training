"""example 測試共用的小規模訓練設定跟執行工具。"""
import contextlib
import io
import os

import yaml

from example.train_conv_compressed import train


def base_cfg(run_name: str, seed: int, conv2_L_init: int, grow: float, epochs: int,
             conv1_L_init: int = 185, conv1_max_out_init: int = 8000,
             conv2_max_out_init: int = 35000, train_size: int = 16, val_size: int = 8) -> dict:
    """真實 N-MNIST 的小規模訓練設定(conv 8 -> conv 16 -> FC 10)。

    grow: 四個容量旋鈕共用的放大倍率。
    max_out 預設給足,seed 1..42 的小規模不會出界。
    """
    def _conv(oc, L, max_out):
        return {"type": "conv", "oc": oc, "k": 3, "s": 2, "p": 1,
                "tau": 16.0, "v_th": 1.0, "alpha": 2.0, "chunk_size": 1,
                "L": L, "max_out_spikes": max_out, "init_k": 8.0 if oc == 8 else 64.0,
                "L_grow_factor": grow, "out_grow_factor": grow}
    return {
        "run_name": run_name,
        "model": {
            "decoder": "membrane_regression",
            "input_shape": [2, 34, 34],
            "layers": [
                _conv(8, conv1_L_init, conv1_max_out_init),
                _conv(16, conv2_L_init, conv2_max_out_init),
                {"type": "fc", "name": "out", "n_out": 10, "tau": 16.0,
                 "v_th": 1.0e9, "alpha": 2.0, "chunk_size": 512, "init_k": 5.0},
            ],
        },
        "data": {
            "max_events": 2000, "train_size": train_size, "val_size": val_size,
            "seed_train": 0, "seed_val": 0,
        },
        "train": {
            "lr": 1.0e-2, "epochs": epochs, "batch_size": 4, "seed": seed,
        },
    }


def reference_cfg() -> dict:
    """seed 42、conv2 L 給足不出界的 2 epoch 訓練,每個 epoch 存權重。

    出界測試拿它當「一開始就給夠容量」的對照組,eval、replay 測試拿它當已訓練的實驗目錄。
    """
    cfg = base_cfg("reference", seed=42, conv2_L_init=5000, grow=2.0, epochs=2)
    cfg["train"]["weight_snapshot_every"] = 1
    return cfg


def write_yaml(cfg: dict, root) -> str:
    """把 cfg 寫成 root 底下的 <run_name>.yaml,回傳路徑。"""
    path = os.path.join(root, f"{cfg['run_name']}.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    return path


def run_capture(cfg: dict, root):
    """用 cfg 在 root 底下跑 train(),回傳 (train() 的結果, 訓練過程印出的文字)。"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = train(write_yaml(cfg, root), exp_root=str(root))
    return result, buf.getvalue()
