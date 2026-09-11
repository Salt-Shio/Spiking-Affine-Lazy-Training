"""example/inspect_traces.py 的單元測試(合成 npz,不需要訓練 run)。"""
import contextlib
import io
import os

import numpy as np

from example.inspect_traces import _resolve_traces_dir, report_full, report_summary
from example.trace_store import layer_names


def _write_summary(traces_dir: str, epochs, layers: dict) -> None:
    out = {"epochs": np.asarray(epochs, dtype=np.int32)}
    for name, n in layers.items():
        rng = np.random.default_rng(hash(name) % 2**32)
        E = len(epochs)
        out[f"{name}__spike_count"] = rng.random((E, n)).astype(np.float32)
        out[f"{name}__s_value_sum"] = rng.random((E, n)).astype(np.float32)
        out[f"{name}__v_final"] = rng.standard_normal((E, n)).astype(np.float32)
        out[f"{name}__idle_frac"] = rng.random((E, n)).astype(np.float32)
    np.savez(os.path.join(traces_dir, "summary.npz"), **out)


def _write_full(traces_dir: str, epoch: int, layers: dict, S=2, steps=12) -> None:
    out = {}
    for name, n in layers.items():
        rng = np.random.default_rng((hash(name) + epoch) % 2**32)
        out[f"{name}__spike_mask"] = (rng.random((S, n, steps)) > 0.7)
        out[f"{name}__s_value"] = rng.random((S, n, steps)).astype(np.float32)
        out[f"{name}__v_steps"] = rng.standard_normal((S, n, steps)).astype(np.float32)
        ms = rng.random((S, n, steps)).astype(np.float32) * 30.0
        ms[:, :, steps // 2:] = np.nan
        out[f"{name}__event_ms"] = ms
    np.savez(os.path.join(traces_dir, f"full_epoch_{epoch:03d}.npz"), **out)


def test_layer_names_order():
    files = ["epochs", "conv1__spike_count", "conv1__v_final",
             "conv2__spike_count", "out__idle_frac"]
    assert layer_names(files) == ["conv1", "conv2", "out"]


def test_resolve_traces_dir(tmp_path):
    (tmp_path / "traces").mkdir()
    assert _resolve_traces_dir(str(tmp_path)) == str(tmp_path / "traces")
    assert _resolve_traces_dir(str(tmp_path / "traces")) == str(tmp_path / "traces")


def test_report_summary_runs(tmp_path):
    layers = {"conv1": 40, "conv2": 24, "out": 10}
    _write_summary(str(tmp_path), [0, 5, 10], layers)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report_summary(str(tmp_path), tau=0.1, activity="spike", top_k=3)
    txt = buf.getvalue()
    assert "summary.npz" in txt
    for name in layers:
        assert f"[{name}]" in txt
    assert "dormant_frac" in txt and "趨勢" in txt and "整段沒醒" in txt


def test_report_summary_missing(tmp_path):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report_summary(str(tmp_path), tau=0.1, activity="s_value", top_k=5)
    assert "沒有" in buf.getvalue()


def test_report_full_runs(tmp_path):
    layers = {"conv1": 40, "conv2": 24, "out": 10}
    _write_full(str(tmp_path), 20, layers)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report_full(str(tmp_path), 20, sample=0, neuron=None)
    txt = buf.getvalue()
    assert "full_epoch_020.npz" in txt
    assert "空轉步比例" in txt and "神經元" in txt


def test_report_full_missing_lists_available(tmp_path):
    layers = {"conv1": 8}
    _write_full(str(tmp_path), 20, layers)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report_full(str(tmp_path), 5, sample=0, neuron=None)
    txt = buf.getvalue()
    assert "沒有" in txt and "full_epoch_020.npz" in txt


def test_report_full_sample_out_of_range(tmp_path):
    layers = {"conv1": 8}
    _write_full(str(tmp_path), 0, layers, S=2)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report_full(str(tmp_path), 0, sample=9, neuron=None)
    assert "超出範圍" in buf.getvalue()


TESTS = [
    test_layer_names_order,
    test_resolve_traces_dir,
    test_report_summary_runs,
    test_report_summary_missing,
    test_report_full_runs,
    test_report_full_missing_lists_available,
    test_report_full_sample_out_of_range,
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
