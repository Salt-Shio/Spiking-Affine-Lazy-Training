"""example 測試共用的 fixture。"""
import pytest

from example.tests._train_runs import run_reference


@pytest.fixture(scope="session")
def reference_run(tmp_path_factory):
    """整套測試只跑一次的真實資料訓練(設定見 _train_runs.reference_cfg),寫在暫存目錄。
    回傳 TrainResult。"""
    return run_reference(tmp_path_factory.mktemp("reference_run"))
