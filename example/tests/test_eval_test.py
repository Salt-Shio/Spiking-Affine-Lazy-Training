"""`example/eval_test.py` 的測試。

- `_confusion_matrix`:純函式,手算小例子驗證。
- `evaluate_run`:對一個用真正 `train()` 跑出來的小 exp_dir 跑評估,檢查
  accuracy/loss/confusion_matrix/`eval_<which>_preds.npz` 彼此一致——不重算
  loss 本身的數值對不對(跟 `train_step` 用同一個 `optax.softmax_cross_entropy`,
  信任 optax),只驗證這幾個輸出是不是從同一份 `preds`/`labels` 導出的。
"""
import os
import shutil

import numpy as np
import yaml

from example.eval_test import _confusion_matrix, evaluate_run
from example.models.conv_net import N_CLASSES
from example.paths import EXPERIMENTS_DIR
from example.train_conv_compressed import train
from example.utils import EVAL_DIRNAME

_TEST_TEMP = os.path.join(EXPERIMENTS_DIR, "TEST_TEMP_EVAL")
shutil.rmtree(_TEST_TEMP, ignore_errors=True)
os.makedirs(_TEST_TEMP, exist_ok=True)

_TMP_DIR = os.path.join(_TEST_TEMP, "_configs")
os.makedirs(_TMP_DIR, exist_ok=True)


def _write_yaml(cfg: dict) -> str:
    path = os.path.join(_TMP_DIR, f"{cfg['run_name']}.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    return path


def _tiny_cfg(run_name: str) -> dict:
    """小規模、容量給足避免出界——動態放大機制是
    test_train_conv_compressed.py 的事,這裡只要一個能訓練完、能評估的 exp_dir。
    """
    def _conv(oc, init_k):
        return {"type": "conv", "oc": oc, "k": 3, "s": 2, "p": 1,
                "tau": 16.0, "v_th": 1.0, "alpha": 2.0, "chunk_size": 1,
                "L": 200, "max_out_spikes": 5000, "init_k": init_k}
    return {
        "run_name": run_name,
        "model": {
            "decoder": "membrane_regression",
            "input_shape": [2, 34, 34],
            "layers": [
                _conv(8, 8.0),
                _conv(16, 64.0),
                {"type": "fc", "name": "out", "n_out": 10, "tau": 16.0,
                 "v_th": 1.0e9, "alpha": 2.0, "chunk_size": 512, "init_k": 5.0},
            ],
        },
        "data": {"max_events": 2000, "train_size": 16, "val_size": 8,
                 "seed_train": 0, "seed_val": 0},
        "train": {"lr": 1.0e-2, "epochs": 1, "batch_size": 4, "seed": 0},
    }


def _trained_exp_dir() -> str:
    exp_dir, _net, _params, _train_split, _val_split, _run_record = train(
        _write_yaml(_tiny_cfg("eval_test_fixture")), exp_root=_TEST_TEMP)
    return exp_dir


# ============================================================================
# A. _confusion_matrix:手算
# ============================================================================

def test_confusion_matrix_hand():
    labels = np.array([0, 0, 1, 2, 2, 2])
    preds = np.array([0, 1, 1, 2, 2, 0])
    cm = _confusion_matrix(labels, preds, n_classes=3)
    expected = np.array([[1, 1, 0],
                         [0, 1, 0],
                         [1, 0, 2]])
    np.testing.assert_array_equal(cm, expected)
    assert cm.sum() == len(labels)


# ============================================================================
# B. evaluate_run:對真正訓練出的 exp_dir 跑
# ============================================================================

def test_evaluate_run_outputs_consistent():
    exp_dir = _trained_exp_dir()
    result = evaluate_run(exp_dir, which="val", n_samples=8, seed=0, which_params="best")

    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["loss"] == result["loss"] and result["loss"] >= 0.0  # 非 nan、非負

    cm = np.asarray(result["confusion_matrix"])
    assert cm.shape == (N_CLASSES, N_CLASSES)
    assert cm.sum() == result["n_samples"] == 8
    # 對角線總和 / n_samples 應該等於 accuracy——同一份 preds 導出的兩個量。
    np.testing.assert_allclose(cm.trace() / result["n_samples"], result["accuracy"], atol=1e-6)

    preds_path = os.path.join(exp_dir, EVAL_DIRNAME, "val_preds.npz")
    assert os.path.isfile(preds_path)
    data = np.load(preds_path)
    assert data["preds"].shape == (8,)
    assert data["labels"].shape == (8,)
    assert np.all((data["preds"] >= 0) & (data["preds"] < N_CLASSES))
    cm_rebuilt = _confusion_matrix(data["labels"], data["preds"], N_CLASSES)
    np.testing.assert_array_equal(cm_rebuilt, cm)


TESTS = [
    test_confusion_matrix_hand,
    test_evaluate_run_outputs_consistent,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n全部 {len(TESTS)} 項測試通過")
