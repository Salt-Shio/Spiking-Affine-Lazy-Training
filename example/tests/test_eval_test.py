"""`example/eval_test.py` 的測試。

- `_confusion_matrix`:純函式,手算小例子驗證。
- `evaluate_run`:對共用的參考訓練(conftest.py 的 reference_run)跑評估,檢查
  accuracy/loss/confusion_matrix/`eval_<which>_preds.npz` 彼此一致——不重算
  loss 本身的數值對不對(跟 `train_step` 用同一個 `optax.softmax_cross_entropy`,
  信任 optax),只驗證這幾個輸出是不是從同一份 `preds`/`labels` 導出的。
"""
import os

import numpy as np

from example.eval_test import _confusion_matrix, evaluate_run
from example.models.conv_net import N_CLASSES
from example.utils import EVAL_DIRNAME


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

def test_evaluate_run_outputs_consistent(reference_run):
    (exp_dir, *_), _stdout = reference_run
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
