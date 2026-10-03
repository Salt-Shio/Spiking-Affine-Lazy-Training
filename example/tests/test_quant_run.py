"""example/quant_run.py 的溢位分析。

用共用的參考訓練(conftest.py 的 reference_run)產生的量化模型,把 i_V 調小製造溢位:
- i_V 加寬不改變沒溢位樣本的結果(逐位元相同)。
- min_extra_bits 加回去之後所有樣本都不溢位,而且每層不超過調小的位元數。
- headroom_bits 有溢位的層是負數;overflow_events 列出的位置都對得上溢位旗標。
"""
import os

import numpy as np
import pytest

from data.src.nmnist import NMNISTDataset
from example.models.conv_net import build_decoder, build_growth_policies
from example.paths import DATASET_ROOT
from example.quant_run import (MAX_OVERFLOW_EVENTS, headroom_bits, min_extra_bits,
                               overflow_events, quant_forward_split, reference_arrays,
                               with_extra_i_V)
from example.quantize import MODEL_FILENAME, quantize
from example.utils import load_run_record, split_input_events
from salt_core.io import load_quantized

SHRINK = 4  # 每層 i_V 減幾個位元來製造溢位
BATCH_SIZE = 4


@pytest.fixture(scope="module")
def setup(reference_run):
    exp_dir = reference_run.exp_dir
    cfg = {"source": {"run": os.path.basename(exp_dir), "params": "best"},
           "calibration": {"split": "val", "n_samples": 4},
           "verify": {"n_samples": None},
           "spec": {"bits": 8, "f_a": 8, "f_V": 4, "round_mode": "round",
                    "out_granularity": "per_channel", "clip_percentile": 100,
                    "overflow_mode": "wrap", "guard_bits": 0, "i_V": None}}
    model = load_quantized(os.path.join(quantize(cfg, experiments_dir=os.path.dirname(exp_dir)),
                                        MODEL_FILENAME))
    data_cfg = load_run_record(exp_dir)["config"]["data"]
    split = NMNISTDataset(DATASET_ROOT, max_events=data_cfg["max_events"]).build_split(
        seed=data_cfg["seed_val"], n_samples=data_cfg["val_size"], which="val")
    raw = split_input_events(split, model.network.input_shape)
    decoder = build_decoder({"decoder": "membrane_regression"}, model.network.layers)
    policies = build_growth_policies(load_run_record(exp_dir)["config"]["model"],
                                     model.network.layers)
    narrow = with_extra_i_V(model.params, [-SHRINK] * len(model.params))
    out = quant_forward_split(model.network, decoder, model.params, model.round_mode, raw,
                              BATCH_SIZE)
    narrow_out = quant_forward_split(model.network, decoder, narrow, model.round_mode, raw,
                                     BATCH_SIZE)
    return dict(model=model, raw=raw, decoder=decoder, policies=policies, narrow=narrow,
                out=out, narrow_out=narrow_out)


def test_narrowed_registers_overflow(setup):
    assert setup["narrow_out"].overflowed.any(), "測試前提:i_V 調小之後要有溢位"


def test_wider_i_V_keeps_results_of_samples_without_overflow(setup):
    model = setup["model"]
    wide = quant_forward_split(model.network, setup["decoder"],
                               with_extra_i_V(model.params, [3] * len(model.params)),
                               model.round_mode, setup["raw"], BATCH_SIZE)
    clean = ~setup["out"].overflowed.any(axis=1)
    assert clean.any()
    for key, value in reference_arrays(setup["out"]).items():
        np.testing.assert_array_equal(getattr(wide, key)[clean], value[clean], err_msg=key)


def test_min_extra_bits_removes_all_overflow(setup):
    model = setup["model"]
    extra, rounds = min_extra_bits(model.network, setup["decoder"], setup["narrow"],
                                   model.round_mode, setup["raw"], BATCH_SIZE, setup["policies"],
                                   setup["narrow_out"])
    fixed = quant_forward_split(model.network, setup["decoder"],
                                with_extra_i_V(setup["narrow"], extra), model.round_mode,
                                setup["raw"], BATCH_SIZE)

    assert rounds >= 1
    assert not fixed.overflowed.any()
    assert all(0 <= e for e in extra)
    # 每層各自少 1 位元就會溢位:只加到剛好夠,不會多加
    for i, e in enumerate(extra):
        if e == 0:
            continue
        one_less = list(extra)
        one_less[i] -= 1
        again = quant_forward_split(model.network, setup["decoder"],
                                    with_extra_i_V(setup["narrow"], one_less), model.round_mode,
                                    setup["raw"], BATCH_SIZE)
        assert again.overflowed.any()


def test_headroom_negative_exactly_for_overflowed_layers(setup):
    headroom = headroom_bits(setup["narrow"], setup["narrow_out"])
    overflowed = setup["narrow_out"].overflowed.any(axis=0)
    for h, ov in zip(headroom, overflowed):
        assert (h < 0) == bool(ov)


def test_overflow_events_point_at_overflowed_samples_and_layers(setup):
    model = setup["model"]
    events = overflow_events(model.network, setup["narrow"], model.round_mode, setup["raw"],
                             setup["narrow_out"])
    names = [layer.name for layer in model.network.layers]

    assert 0 < len(events) <= MAX_OVERFLOW_EVENTS
    for event in events:
        assert setup["narrow_out"].overflowed[event["sample"], names.index(event["layer"])]
        low, high = event["register_range"]
        assert event["true_min"] < low or event["true_max"] > high
