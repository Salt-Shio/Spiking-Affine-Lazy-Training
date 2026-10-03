"""example/quantize.py。

- spec_name、check_quant_config、quant_layer_specs:小例子。
- quantize、check:對共用的參考訓練(conftest.py 的 reference_run)產生量化資料夾,
  報告跟參考輸出要一致,check 重跑要逐筆相同;參考輸出被改過時 check 要抓到。
"""
import os

import numpy as np
import pytest
import yaml

from example.quantize import (MODEL_FILENAME, QUANT_DIRNAME, REFERENCE_FILENAME, REPORT_FILENAME,
                              check, check_quant_config, quant_layer_specs, quantize, spec_name)
from salt_core.io import load_quantized
from salt_core.quant.convert import LayerQuantSpec
from salt_core.quant.fixed_point import RoundMode

SPEC = {"bits": 8, "f_a": 10, "f_V": 10, "round_mode": "round", "out_granularity": "per_channel",
        "clip_percentile": 100, "overflow_mode": "wrap"}


def _cfg(run: str, **spec_overrides) -> dict:
    """參考訓練的 val 只有 8 筆:量 M 用前 4 筆,驗證用全部。"""
    return {"source": {"run": run, "params": "best"},
            "calibration": {"n_samples": 4},
            "verify": {"n_samples": None},
            "spec": {**SPEC, **spec_overrides}}


# ============================================================================
# A. 小例子
# ============================================================================

def test_spec_name():
    assert spec_name(SPEC) == "b8_fa10_fv10_round_pc_clip100_wrap"
    assert spec_name({**SPEC, "out_granularity": "per_tensor", "clip_percentile": 99.5,
                      "round_mode": "truncate"}) == "b8_fa10_fv10_truncate_pt_clip99.5_wrap"


def test_check_quant_config_unknown_key_raises():
    cfg = _cfg("x")
    cfg["spec"]["bit"] = 8
    with pytest.raises(ValueError, match="bit"):
        check_quant_config(cfg)


def test_check_quant_config_missing_section_raises():
    cfg = _cfg("x")
    del cfg["verify"]
    with pytest.raises(ValueError, match="verify"):
        check_quant_config(cfg)


def test_check_quant_config_bad_granularity_raises():
    with pytest.raises(ValueError, match="out_granularity"):
        check_quant_config(_cfg("x", out_granularity="per_layer"))


def test_quant_layer_specs_only_last_layer_changes():
    base = LayerQuantSpec(bits=4, f_a=8, f_V=6, clip_percentile=90.0)
    specs = quant_layer_specs(base, 3, out_per_channel=False)
    assert specs[:2] == [base, base]
    assert specs[2] == base._replace(per_channel=False, fires=False)


# ============================================================================
# B. quantize、check:對真正訓練出的 run 跑
# ============================================================================

@pytest.fixture(scope="module")
def quant_dir(reference_run):
    exp_dir = reference_run.exp_dir
    return quantize(_cfg(os.path.basename(exp_dir)), experiments_dir=os.path.dirname(exp_dir))


def test_quantize_writes_folder_under_source_run(reference_run, quant_dir):
    assert quant_dir == os.path.join(reference_run.exp_dir, QUANT_DIRNAME, spec_name(SPEC))
    for name in (MODEL_FILENAME, REFERENCE_FILENAME, REPORT_FILENAME):
        assert os.path.isfile(os.path.join(quant_dir, name))


def test_report_matches_reference_and_model(quant_dir):
    with open(os.path.join(quant_dir, REPORT_FILENAME), encoding="utf-8") as f:
        report = yaml.safe_load(f)
    reference = np.load(os.path.join(quant_dir, REFERENCE_FILENAME))
    model = load_quantized(os.path.join(quant_dir, MODEL_FILENAME))

    assert reference["preds"].shape == (8,)
    assert report["data"]["n_verify"] == 8
    assert report["accuracy"] == pytest.approx(np.mean(reference["preds"] == reference["labels"]))
    assert report["truncated_samples"] == {layer.name: 0 for layer in model.network.layers}
    assert report["i_V"] == {layer.name: p.i_V for layer, p in zip(model.network.layers,
                                                                   model.params)}
    assert model.round_mode is RoundMode.ROUND
    assert [layer.chunk_size for layer in model.network.layers] == [1] * 3
    assert model.params[-1].v_th_int is None


def test_check_rerun_matches(quant_dir):
    assert check(quant_dir) == 0


def test_check_detects_changed_reference(quant_dir, tmp_path):
    changed_dir = tmp_path / "changed"
    changed_dir.mkdir()
    for name in (MODEL_FILENAME, REPORT_FILENAME):
        (changed_dir / name).write_bytes(open(os.path.join(quant_dir, name), "rb").read())
    reference = dict(np.load(os.path.join(quant_dir, REFERENCE_FILENAME)))
    reference["v_final_int"] = reference["v_final_int"].copy()
    reference["v_final_int"][3, 0] += 1
    np.savez(changed_dir / REFERENCE_FILENAME, **reference)
    assert check(str(changed_dir)) == 1


def test_quantize_existing_folder_raises(reference_run, quant_dir):
    exp_dir = reference_run.exp_dir
    with pytest.raises(FileExistsError):
        quantize(_cfg(os.path.basename(exp_dir)), experiments_dir=os.path.dirname(exp_dir))


def test_quantize_calibration_more_than_verify_raises(reference_run):
    exp_dir = reference_run.exp_dir
    cfg = _cfg(os.path.basename(exp_dir), bits=4)
    cfg["calibration"]["n_samples"] = 6
    cfg["verify"]["n_samples"] = 5
    with pytest.raises(ValueError, match="calibration"):
        quantize(cfg, experiments_dir=os.path.dirname(exp_dir))
