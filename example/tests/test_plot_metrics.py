"""example/plot_metrics.py 的單元測試——只測這個專案特有的欄位分組知識
(`group_metrics_columns`)+ 端到端 CLI,渲染邏輯本身的測試在
viz/tests/test_epoch_series.py。"""
import os

import matplotlib
matplotlib.use("Agg")

from example.plot_metrics import group_metrics_columns, plot_metrics


def test_group_metrics_columns_matches_real_header():
    """欄名取自實際跑出來的 metrics.csv 表頭(見對話紀錄的真實 run),確保
    分組邏輯跟 example/metrics_log.py 現在真的在寫的欄名對得上。"""
    columns = ["train_loss", "val_accuracy",
               "conv1_L", "conv1_max_out", "conv1_obs_queue", "conv1_obs_out",
               "conv2_L", "conv2_max_out", "conv2_obs_queue", "conv2_obs_out",
               "conv1_firing_rate", "conv2_firing_rate", "out_firing_rate",
               "conv1_grad_norm", "conv2_grad_norm", "out_grad_norm",
               "conv1_dormant_frac", "conv1_act_p90p10",
               "conv2_dormant_frac", "conv2_act_p90p10"]

    groups = group_metrics_columns(columns)

    assert groups["train_loss"] == ["train_loss"]
    assert groups["val_accuracy"] == ["val_accuracy"]
    assert groups["L"] == ["conv1_L", "conv2_L"]
    assert groups["max_out"] == ["conv1_max_out", "conv2_max_out"]
    assert groups["obs_queue"] == ["conv1_obs_queue", "conv2_obs_queue"]
    assert groups["obs_out"] == ["conv1_obs_out", "conv2_obs_out"]
    assert groups["firing_rate"] == ["conv1_firing_rate", "conv2_firing_rate", "out_firing_rate"]
    assert groups["grad_norm"] == ["conv1_grad_norm", "conv2_grad_norm", "out_grad_norm"]
    assert groups["dormant_frac"] == ["conv1_dormant_frac", "conv2_dormant_frac"]
    assert groups["act_p90p10"] == ["conv1_act_p90p10", "conv2_act_p90p10"]
    # 8 組指標字尾 + 2 個獨立欄 = 10 組,不是 20 張各自的子圖
    assert len(groups) == 10


def test_group_metrics_columns_groups_decoder_metrics_together():
    columns = ["train_loss", "decoder_mse", "decoder_ttfs_gap"]

    groups = group_metrics_columns(columns)

    assert groups["decoder"] == ["decoder_mse", "decoder_ttfs_gap"]
    assert groups["train_loss"] == ["train_loss"]


def test_plot_metrics_end_to_end(tmp_path):
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    with open(train_dir / "metrics.csv", "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,conv1_firing_rate,conv2_firing_rate\n")
        for e in range(3):
            f.write(f"{e},{1.0 / (e + 1)},{0.1 * e},{0.2 * e}\n")

    out_path = plot_metrics(str(tmp_path))

    assert out_path == str(train_dir / "metrics.png")
    assert os.path.isfile(out_path)
    assert os.path.getsize(out_path) > 0


TESTS = [
    test_group_metrics_columns_matches_real_header,
    test_group_metrics_columns_groups_decoder_metrics_together,
    test_plot_metrics_end_to_end,
]

if __name__ == "__main__":
    import pathlib
    import tempfile

    for t in TESTS:
        if "tmp_path" in t.__code__.co_varnames:
            with tempfile.TemporaryDirectory() as d:
                t(pathlib.Path(d))
        else:
            t()
        print(f"PASS: {t.__name__}")
    print(f"\n全部 {len(TESTS)} 項通過")
