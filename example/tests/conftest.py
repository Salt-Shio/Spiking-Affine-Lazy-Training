"""example 測試共用的 fixture。"""
import pytest

from example.tests._train_runs import reference_cfg, run_capture


@pytest.fixture(scope="session")
def reference_run(tmp_path_factory):
    """整套測試只跑一次的參考訓練(設定見 reference_cfg),寫在暫存目錄。

    回傳 (train() 的結果, 訓練過程印出的文字)。
    """
    return run_capture(reference_cfg(), tmp_path_factory.mktemp("reference_run"))
