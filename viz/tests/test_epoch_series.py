"""viz/epoch_series.py 的單元測試(合成 csv/rows,不需要訓練 run,不碰任何
`example`/`salt_core` 的知識——這個套件本來就不依賴它們)。"""
import math
import os

import matplotlib
matplotlib.use("Agg")

from viz.epoch_series import EpochSeriesPlot, read_epoch_series_csv


def _write_csv(path: str, header: list, rows: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in header) + "\n")


def test_read_epoch_series_csv_parses_types(tmp_path):
    path = str(tmp_path / "metrics.csv")
    _write_csv(path, ["epoch", "train_loss", "example_ratio"],
               [{"epoch": 0, "train_loss": 1.5, "example_ratio": "inf"},
                {"epoch": 1, "train_loss": 0.7, "example_ratio": 3.2}])

    rows = read_epoch_series_csv(path)

    assert len(rows) == 2
    assert rows[0] == {"epoch": 0, "train_loss": 1.5, "example_ratio": math.inf}
    assert isinstance(rows[0]["epoch"], int)
    assert isinstance(rows[0]["train_loss"], float)
    assert rows[1]["example_ratio"] == 3.2


def test_render_ungrouped_creates_one_subplot_per_column():
    rows = [{"epoch": e, "loss": 1.0 / (e + 1), "acc": 0.1 * e, "grad_norm": float(e)}
            for e in range(5)]

    fig = EpochSeriesPlot(ncols=2).render(rows)

    visible = [ax for ax in fig.axes if ax.get_visible()]
    hidden = [ax for ax in fig.axes if not ax.get_visible()]
    assert len(visible) == 3
    assert len(hidden) == 1
    assert {ax.get_title() for ax in visible} == {"loss", "acc", "grad_norm"}
    for ax in visible:
        xdata, _ = ax.lines[0].get_data()
        assert list(xdata) == [0, 1, 2, 3, 4]
        assert ax.get_legend() is None


def test_render_grouped_overlays_lines_with_legend():
    rows = [{"epoch": e, "conv1_firing_rate": 0.1 * e, "conv2_firing_rate": 0.2 * e,
             "train_loss": 1.0 / (e + 1)} for e in range(3)]
    groups = {"firing_rate": ["conv1_firing_rate", "conv2_firing_rate"],
              "train_loss": ["train_loss"]}

    fig = EpochSeriesPlot(groups=groups, ncols=2).render(rows)

    titles = [ax.get_title() for ax in fig.axes if ax.get_visible()]
    assert titles == ["firing_rate", "train_loss"]
    fr_ax, loss_ax = [ax for ax in fig.axes if ax.get_visible()]
    assert len(fr_ax.lines) == 2
    assert fr_ax.get_legend() is not None
    assert len(loss_ax.lines) == 1
    assert loss_ax.get_legend() is None


def test_render_handles_inf_values_without_crashing():
    rows = [{"epoch": 0, "example_ratio": math.inf}, {"epoch": 1, "example_ratio": 2.0}]

    fig = EpochSeriesPlot().render(rows)

    assert len(fig.axes) == 1


def test_render_empty_rows_raises():
    try:
        EpochSeriesPlot().render([])
    except ValueError:
        pass
    else:
        raise AssertionError("預期 render([]) 要拋 ValueError")


def test_render_static_csv_end_to_end(tmp_path):
    """完整走一次:寫 csv → 讀 → 畫 → 存檔,模擬靜態用法。"""
    path = str(tmp_path / "metrics.csv")
    _write_csv(path, ["epoch", "train_loss", "val_accuracy"],
               [{"epoch": e, "train_loss": 1.0 / (e + 1), "val_accuracy": 0.1 * e}
                for e in range(3)])

    rows = read_epoch_series_csv(path)
    fig = EpochSeriesPlot().render(rows)
    out_path = str(tmp_path / "metrics.png")
    fig.savefig(out_path)

    assert os.path.isfile(out_path)
    assert os.path.getsize(out_path) > 0


TESTS = [
    test_read_epoch_series_csv_parses_types,
    test_render_ungrouped_creates_one_subplot_per_column,
    test_render_grouped_overlays_lines_with_legend,
    test_render_handles_inf_values_without_crashing,
    test_render_empty_rows_raises,
    test_render_static_csv_end_to_end,
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
