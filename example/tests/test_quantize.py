"""example/quantize.py。

- spec_name、quant_dir_name、check_quant_config:小例子。
- quantize、check、evaluate:對共用的參考訓練(conftest.py 的 reference_run)產生量化資料夾,
  報告跟參考輸出、模型要一致,check 重跑要逐筆相同;參考輸出被改過時 check 要抓到。
"""
import os

import numpy as np
import pytest
import yaml

from example.quantize import (MODEL_FILENAME, QUANT_DIRNAME, REFERENCE_FILENAME, REPORT_FILENAME,
                              check, check_quant_config, evaluate, quant_dir_name, quantize,
                              spec_name)
from example.utils import EVAL_DIRNAME
from salt_core.io import load_quantized
from salt_core.quant.fixed_point import RoundMode

SPEC = {"bits": 8, "f_a": 10, "f_V": 10, "round_mode": "round", "out_granularity": "per_channel",
        "clip_percentile": 100, "overflow_mode": "wrap", "guard_bits": 1, "i_V": None}


def _cfg(run: str, **spec_overrides) -> dict:
    """參考訓練的 val 只有 8 筆:量 M 用 val 前 4 筆,驗證用全部。"""
    return {"source": {"run": run, "params": "best"},
            "calibration": {"split": "val", "n_samples": 4},
            "verify": {"n_samples": None},
            "spec": {**SPEC, **spec_overrides}}


# ============================================================================
# A. 小例子
# ============================================================================

def test_spec_name():
    assert spec_name(SPEC) == "b8_fa10_fv10_round_pc_clip100_wrap_g1"
    assert spec_name({**SPEC, "out_granularity": "per_tensor", "clip_percentile": 99.5,
                      "round_mode": "truncate", "guard_bits": 0}) == \
        "b8_fa10_fv10_truncate_pt_clip99.5_wrap_g0"
    assert spec_name({**SPEC, "overflow_mode": "saturate", "guard_bits": 0,
                      "i_V": {"conv1": 9, "conv2": 11, "out": 11}}) == \
        "b8_fa10_fv10_round_pc_clip100_saturate_iv9-11-11"


def test_quant_dir_name_has_weight_source_and_calibration():
    cfg = _cfg("x")
    assert quant_dir_name(cfg) == "best_val4_b8_fa10_fv10_round_pc_clip100_wrap_g1"
    cfg["source"]["params"] = 59
    cfg["calibration"] = {"split": "train", "n_samples": 10000}
    assert quant_dir_name(cfg) == "e59_train10000_b8_fa10_fv10_round_pc_clip100_wrap_g1"
    cfg["source"]["params"] = "last"
    with pytest.raises(ValueError, match="source.params"):
        quant_dir_name(cfg)


def test_check_quant_config_unknown_key_raises():
    cfg = _cfg("x")
    cfg["spec"]["bit"] = 8
    with pytest.raises(ValueError, match="bit"):
        check_quant_config(cfg)


def test_check_quant_config_missing_key_raises():
    cfg = _cfg("x")
    del cfg["spec"]["guard_bits"]
    with pytest.raises(ValueError, match="guard_bits"):
        check_quant_config(cfg)


def test_check_quant_config_missing_section_raises():
    cfg = _cfg("x")
    del cfg["verify"]
    with pytest.raises(ValueError, match="verify"):
        check_quant_config(cfg)


@pytest.mark.parametrize("section, key, value", [
    ("spec", "out_granularity", "per_layer"),
    ("calibration", "split", "test"),
    ("spec", "guard_bits", -1),
    ("spec", "i_V", {"conv1": 0}),
    ("spec", "i_V", 9),
])
def test_check_quant_config_bad_value_raises(section, key, value):
    cfg = _cfg("x")
    cfg[section][key] = value
    with pytest.raises(ValueError, match=key):
        check_quant_config(cfg)


def test_check_quant_config_i_V_with_guard_bits_raises():
    cfg = _cfg("x", i_V={"conv1": 9})
    with pytest.raises(ValueError, match="guard_bits"):
        check_quant_config(cfg)


# ============================================================================
# B. quantize、check、evaluate:對真正訓練出的 run 跑
# ============================================================================

def _quantize(reference_run, cfg):
    exp_dir = reference_run.exp_dir
    return quantize(cfg, experiments_dir=os.path.dirname(exp_dir))


@pytest.fixture(scope="module")
def quant_dir(reference_run):
    return _quantize(reference_run, _cfg(os.path.basename(reference_run.exp_dir)))


def _report(quant_dir) -> dict:
    with open(os.path.join(quant_dir, REPORT_FILENAME), encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_quantize_writes_folder_under_source_run(reference_run, quant_dir):
    assert quant_dir == os.path.join(reference_run.exp_dir, QUANT_DIRNAME,
                                     "best_val4_" + spec_name(SPEC))
    for name in (MODEL_FILENAME, REFERENCE_FILENAME, REPORT_FILENAME):
        assert os.path.isfile(os.path.join(quant_dir, name))


def test_report_matches_reference_and_model(quant_dir):
    report = _report(quant_dir)
    reference = np.load(os.path.join(quant_dir, REFERENCE_FILENAME))
    model = load_quantized(os.path.join(quant_dir, MODEL_FILENAME))
    names = [layer.name for layer in model.network.layers]

    assert reference["preds"].shape == (8,)
    assert report["data"]["verify"]["n"] == 8
    assert report["data"]["calibration"] == {"split": "val", "seed": 0, "n": 4}
    assert report["accuracy"] == pytest.approx(np.mean(reference["preds"] == reference["labels"]))
    assert report["truncated_samples"] == {name: 0 for name in names}
    assert report["overflowed_samples"] == {name: int(reference["overflowed"][:, i].sum())
                                            for i, name in enumerate(names)}
    assert report["i_V"] == {name: p.i_V for name, p in zip(names, model.params)}
    assert set(report["headroom_bits"]) == set(names)
    assert model.round_mode is RoundMode.ROUND
    assert [layer.chunk_size for layer in model.network.layers] == [1] * 3
    assert model.params[-1].v_th_int is None


def test_guard_bits_add_to_calibrated_i_V(reference_run, quant_dir):
    """同一份校準,guard_bits 0 跟 1 的 i_V 差 1;report 的 i_V_calibrated 相同。"""
    no_guard_dir = _quantize(reference_run,
                             _cfg(os.path.basename(reference_run.exp_dir), guard_bits=0))
    with_guard, no_guard = _report(quant_dir), _report(no_guard_dir)

    assert with_guard["i_V_calibrated"] == no_guard["i_V_calibrated"] == no_guard["i_V"]
    assert with_guard["i_V"] == {name: v + 1 for name, v in no_guard["i_V"].items()}


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


def test_evaluate_val_matches_reference(quant_dir):
    """val、seed 0、8 筆就是驗證樣本,評估輸出要跟 reference.npz 相同,並寫進 eval/。"""
    result = evaluate(quant_dir, "val", None, 0)
    outputs = np.load(os.path.join(quant_dir, EVAL_DIRNAME, "val_outputs.npz"))
    reference = np.load(os.path.join(quant_dir, REFERENCE_FILENAME))

    assert result["n_samples"] == 8
    assert os.path.isfile(os.path.join(quant_dir, EVAL_DIRNAME, "val.yaml"))
    for key in reference.files:
        np.testing.assert_array_equal(outputs[key], reference[key])
    assert result["accuracy"] == _report(quant_dir)["accuracy"]


def test_quantize_existing_folder_raises(reference_run, quant_dir):
    with pytest.raises(FileExistsError):
        _quantize(reference_run, _cfg(os.path.basename(reference_run.exp_dir)))


def test_quantize_val_calibration_more_than_val_size_raises(reference_run):
    cfg = _cfg(os.path.basename(reference_run.exp_dir), bits=4)
    cfg["calibration"]["n_samples"] = 9
    with pytest.raises(ValueError, match="val_size"):
        _quantize(reference_run, cfg)


def test_given_i_V_replaces_calibrated(reference_run, quant_dir):
    """直接指定 i_V:暫存器用指定值,report 的 i_V_calibrated 照樣是校準值;
    飽和模式不算最少要加幾位元、溢位位置。"""
    calibrated = _report(quant_dir)["i_V_calibrated"]
    given = {name: v - 1 for name, v in calibrated.items()}
    given_dir = _quantize(reference_run, _cfg(os.path.basename(reference_run.exp_dir),
                                              overflow_mode="saturate", guard_bits=0, i_V=given))
    report = _report(given_dir)
    model = load_quantized(os.path.join(given_dir, MODEL_FILENAME))

    assert os.path.basename(given_dir).endswith(
        "_saturate_iv" + "-".join(str(v) for v in given.values()))
    assert report["i_V"] == given
    assert [p.i_V for p in model.params] == list(given.values())
    assert report["i_V_calibrated"] == calibrated
    assert set(report["headroom_bits"]) == set(given)
    assert "min_extra_bits" not in report and "overflow_events" not in report


def test_given_i_V_wrong_layer_names_raises(reference_run):
    cfg = _cfg(os.path.basename(reference_run.exp_dir), guard_bits=0, i_V={"conv": 9})
    with pytest.raises(ValueError, match="層名"):
        _quantize(reference_run, cfg)


def test_register_below_threshold_raises(reference_run, quant_dir):
    """f_V=0、i_V=1:暫存器最大值是 0,到不了門檻,永遠不會 fire。"""
    names = list(_report(quant_dir)["i_V"])
    cfg = _cfg(os.path.basename(reference_run.exp_dir), f_V=0, guard_bits=0,
               i_V={name: 1 for name in names})
    with pytest.raises(ValueError, match="不會 fire"):
        _quantize(reference_run, cfg)
