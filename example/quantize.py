"""量化實驗:浮點 run 的權重照量化規格轉成整數模型,跑滿驗證樣本,存成可以獨立重跑的資料夾。

產生:讀來源 run 的權重 -> 校準樣本量膜電位範圍 M -> 算每層量化參數(i_V 加 guard_bits)->
    跑驗證樣本(容量出界就放大重跑)-> 寫 experiments/<來源 run>/quant/<資料夾名>/。
檢查:只讀量化資料夾跟資料集,重跑驗證樣本,逐筆比對 reference.npz。有不同就回傳非 0。
評估:只讀量化資料夾跟資料集,在 test(或 val)上跑,寫進資料夾的 eval/。

用法(路徑相對於 repo 根目錄,或給絕對路徑):
  python -m example.quantize configs/quant/baseline.yaml
  python -m example.quantize --check experiments/<來源 run>/quant/<資料夾名>
  python -m example.quantize --eval experiments/<來源 run>/quant/<資料夾名> [--which test|val] [--n N]
輸出:model.npz(salt_core.io.save_quantized 格式)、reference.npz(逐筆輸出)、report.yaml;
    --eval 寫 eval/<which>.yaml、eval/<which>_outputs.npz
"""
import argparse
import dataclasses
import datetime
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import yaml

from data.src.nmnist import NMNISTDataset, NMNISTSplit
from example.models.conv_net import build_decoder, build_growth_policies
from example.paths import DATASET_ROOT, EXPERIMENTS_DIR, REPO_ROOT, resolve_config
from example.quant_run import (QuantSplitOutput, calibrate_v_abs_max, overflow_summary,
                               quant_forward_split, quant_layer_specs, reference_arrays,
                               run_until_fits, with_extra_i_V)
from example.utils import (EVAL_DIRNAME, WEIGHTS_DIRNAME, get_git_commit_hash, load_config,
                           load_run_params, load_run_record, make_evaluate, split_input_events,
                           weight_snapshot_path)
from salt_core.capacity import GrowthPolicy
from salt_core.io import load_quantized, load_weights, save_quantized
from salt_core.network import Network
from salt_core.quant.convert import LayerQuantSpec, build_quantized_params

QUANT_DIRNAME = "quant"
MODEL_FILENAME = "model.npz"
REFERENCE_FILENAME = "reference.npz"
REPORT_FILENAME = "report.yaml"

# config 各節認得的 key
_SECTION_KEYS = {
    "source": {"run", "params"},
    "calibration": {"split", "n_samples"},
    "verify": {"n_samples"},
    "spec": {"bits", "f_a", "f_V", "round_mode", "out_granularity", "clip_percentile",
             "overflow_mode", "guard_bits"},
}
_OUT_GRANULARITIES = ("per_channel", "per_tensor")
_CALIBRATION_SPLITS = ("val", "train")


# ============================================================================
# config 跟資料夾名
# ============================================================================

def check_quant_config(cfg: dict) -> None:
    """config 缺節、缺 key、有不認得的 key、out_granularity 或 calibration.split 不合法、
    guard_bits 是負數時 raise ValueError。"""
    unknown = sorted(set(cfg) - set(_SECTION_KEYS))
    if unknown:
        raise ValueError(f"config 最外層有不認得的 key:{unknown},認得的是 {sorted(_SECTION_KEYS)}")
    for section, allowed in _SECTION_KEYS.items():
        if section not in cfg:
            raise ValueError(f"config 少了 {section} 節")
        keys = set(cfg[section])
        if keys != allowed:
            raise ValueError(f"config 的 {section} 不認得 {sorted(keys - allowed)}、"
                             f"少了 {sorted(allowed - keys)}")
    if cfg["spec"]["out_granularity"] not in _OUT_GRANULARITIES:
        raise ValueError(f"spec.out_granularity 要是 {_OUT_GRANULARITIES} 之一,"
                         f"拿到 {cfg['spec']['out_granularity']!r}")
    if cfg["calibration"]["split"] not in _CALIBRATION_SPLITS:
        raise ValueError(f"calibration.split 要是 {_CALIBRATION_SPLITS} 之一,"
                         f"拿到 {cfg['calibration']['split']!r}")
    if int(cfg["spec"]["guard_bits"]) < 0:
        raise ValueError(f"spec.guard_bits 不能是負數,拿到 {cfg['spec']['guard_bits']}")


def spec_name(spec: dict) -> str:
    """規格組成的名字,例如 b8_fa10_fv10_round_pc_clip100_wrap_g1。"""
    granularity = "pc" if spec["out_granularity"] == "per_channel" else "pt"
    return (f"b{spec['bits']}_fa{spec['f_a']}_fv{spec['f_V']}_{spec['round_mode']}_"
            f"{granularity}_clip{spec['clip_percentile']:g}_{spec['overflow_mode']}_"
            f"g{spec['guard_bits']}")


def _source_tag(which) -> str:
    """權重來源的標記:best、final、e<epoch>。不是這三種時 raise ValueError。"""
    if which in ("best", "final"):
        return which
    if isinstance(which, int) and not isinstance(which, bool):
        return f"e{which}"
    raise ValueError(f"source.params 要是 best、final 或 epoch 編號,拿到 {which!r}")


def quant_dir_name(cfg: dict) -> str:
    """量化資料夾名:權重來源、校準樣本、規格,例如 best_val50_b8_fa10_fv10_round_pc_clip100_wrap_g1。"""
    calibration = cfg["calibration"]
    return (f"{_source_tag(cfg['source']['params'])}_"
            f"{calibration['split']}{calibration['n_samples']}_{spec_name(cfg['spec'])}")


# ============================================================================
# 產生量化資料夾
# ============================================================================

def _load_source_params(source_dir: str, which) -> tuple[Network, tuple, int]:
    """回傳 (網路, 浮點權重, 權重的 epoch)。which 是 "best"、"final" 或 epoch 編號(已經檢查過)。"""
    run_record = load_run_record(source_dir)
    if which == "best":
        network, params = load_run_params(source_dir, "best")
        return network, params, int(run_record["best"]["epoch"])
    if which == "final":
        network, params = load_run_params(source_dir, "final")
        return network, params, int(run_record["config"]["train"]["epochs"]) - 1
    network, params = load_weights(weight_snapshot_path(
        os.path.join(source_dir, WEIGHTS_DIRNAME), which))
    return network, params, which


def _build_split(max_events: int, which: str, seed: int, n: int) -> NMNISTSplit:
    """N-MNIST 的 which split 抽 n 筆。同一個 seed 抽 n 筆,等於抽更多筆時的前 n 筆。"""
    return NMNISTDataset(DATASET_ROOT, max_events=max_events).build_split(
        seed=seed, n_samples=n, which=which)


def _data_meta(cfg: dict, data_cfg: dict) -> dict:
    """取樣方式:驗證用 run 的 val 前幾筆;校準用 val 或 train 前幾筆(seed 照 run)。
    驗證或 val 校準的筆數超過 run 的 val_size 時 raise ValueError。"""
    val_size = int(data_cfg["val_size"])
    verify = {"split": "val", "seed": int(data_cfg["seed_val"]),
              "n": int(cfg["verify"]["n_samples"] or val_size)}
    calibration_split = cfg["calibration"]["split"]
    calibration = {"split": calibration_split,
                   "seed": int(data_cfg["seed_val" if calibration_split == "val" else "seed_train"]),
                   "n": int(cfg["calibration"]["n_samples"])}
    for name, part in (("驗證", verify), ("校準", calibration)):
        if part["split"] == "val" and part["n"] > val_size:
            raise ValueError(f"{name}樣本數 {part['n']} 超過 run 的 val_size={val_size}")
    return {"max_events": int(data_cfg["max_events"]), "val_size": val_size,
            "verify": verify, "calibration": calibration}


def _split_of(data_meta: dict, part: str) -> NMNISTSplit:
    """data_meta 記的 verify 或 calibration 取樣。"""
    sampling = data_meta[part]
    return _build_split(data_meta["max_events"], sampling["split"], sampling["seed"], sampling["n"])


def _build_params(network: Network, float_params, spec_cfg: dict,
                  v_abs_max: list[np.ndarray]) -> list:
    """照 config 的 spec 節算每層的 QuantizedLayerParams,i_V 再加 guard_bits。"""
    base = LayerQuantSpec(bits=spec_cfg["bits"], f_a=spec_cfg["f_a"], f_V=spec_cfg["f_V"],
                          clip_percentile=float(spec_cfg["clip_percentile"]),
                          overflow_mode=spec_cfg["overflow_mode"])
    specs = quant_layer_specs(base, len(network.layers),
                              out_per_channel=spec_cfg["out_granularity"] == "per_channel")
    params = build_quantized_params(network.layers, float_params, specs, v_abs_max)
    return with_extra_i_V(params, [int(spec_cfg["guard_bits"])] * len(params))


def _capacity(network: Network) -> dict:
    return {layer.name: dict(layer.capacity) for layer in network.layers
            if layer.capacity is not None}


def _report(metadata: dict, network: Network, params: list, out: QuantSplitOutput,
            labels: np.ndarray, float_accuracy: float, n_regrow: int, overflow: dict) -> dict:
    """report.yaml 的內容。overflow 是 overflow_summary 的回傳。"""
    names = [layer.name for layer in network.layers]
    guard_bits = int(metadata["spec"]["guard_bits"])
    return {
        "source": metadata["source"],
        "spec": metadata["spec"],
        "data": metadata["data"],
        "i_V_calibrated": {name: int(p.i_V) - guard_bits for name, p in zip(names, params)},
        "i_V": {name: int(p.i_V) for name, p in zip(names, params)},
        "register_bits": {name: int(p.i_V + p.f_V) for name, p in zip(names, params)},
        "accuracy": float(np.mean(out.preds == labels)),
        "float_accuracy": float_accuracy,
        **overflow,
        "truncated_samples": {name: int(out.truncated[:, i].sum()) for i, name in enumerate(names)},
        "n_regrow": n_regrow,
        "capacity": _capacity(network),
        "git_commit": metadata["git_commit"],
        "timestamp": metadata["timestamp"],
    }


def quantize(cfg: dict, experiments_dir: str | os.PathLike = EXPERIMENTS_DIR) -> str:
    """照 cfg 產生量化資料夾,回傳它的路徑。來源 run 是 experiments_dir/<source.run>。

    資料夾已經存在時 raise FileExistsError。config 不合法(見 check_quant_config)、樣本數超過
    run 的 val_size、或 run 的 decoder 不是 membrane_regression 時 raise ValueError。
    """
    check_quant_config(cfg)
    spec_cfg, source_cfg = cfg["spec"], cfg["source"]
    source_dir = os.path.join(experiments_dir, source_cfg["run"])
    out_dir = os.path.join(source_dir, QUANT_DIRNAME, quant_dir_name(cfg))
    if os.path.exists(out_dir):
        raise FileExistsError(f"{out_dir} 已經存在,要重產請先刪掉")

    run_record = load_run_record(source_dir)
    model_cfg = run_record["config"]["model"]
    decoder_kind = model_cfg.get("decoder", "membrane_regression")
    if decoder_kind != "membrane_regression":
        raise ValueError("目前只支援 membrane_regression 輸出層(最後一層不 fire)")
    data_meta = _data_meta(cfg, run_record["config"]["data"])
    float_network, float_params, epoch = _load_source_params(source_dir, source_cfg["params"])
    decoder = build_decoder(model_cfg, float_network.layers)
    policies = build_growth_policies(model_cfg, float_network.layers)
    batch_size = int(run_record["config"]["train"]["batch_size"])

    network = float_network.replace_layers(
        [layer.with_chunk_size(1) for layer in float_network.layers])
    calibration_raw = split_input_events(_split_of(data_meta, "calibration"), network.input_shape)
    v_abs_max = calibrate_v_abs_max(network, float_params, calibration_raw, batch_size)
    quant_params = _build_params(network, float_params, spec_cfg, v_abs_max)

    split = _split_of(data_meta, "verify")
    raw = split_input_events(split, network.input_shape)
    network, out, n_regrow = run_until_fits(network, decoder, quant_params, spec_cfg["round_mode"],
                                            raw, batch_size, policies)
    overflow = overflow_summary(network, decoder, quant_params, spec_cfg["round_mode"], raw,
                                batch_size, policies, out)
    float_accuracy, _, _, _ = make_evaluate(float_network, decoder, batch_size, policies)(
        float_params, split)

    metadata = {
        "source": {"run": source_cfg["run"], "params": source_cfg["params"], "epoch": epoch},
        "spec": dict(spec_cfg),
        "data": data_meta,
        "decoder": decoder_kind,
        "batch_size": batch_size,
        "growth": {name: dataclasses.asdict(policy) for name, policy in policies.items()},
        "v_abs_max": {layer.name: m.tolist() for layer, m in zip(network.layers, v_abs_max)},
        "git_commit": get_git_commit_hash(str(REPO_ROOT)),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    labels = np.asarray(split.labels)
    os.makedirs(out_dir)
    save_quantized(os.path.join(out_dir, MODEL_FILENAME), network, quant_params,
                   spec_cfg["round_mode"], metadata)
    np.savez(os.path.join(out_dir, REFERENCE_FILENAME), labels=labels, **reference_arrays(out))
    with open(os.path.join(out_dir, REPORT_FILENAME), "w", encoding="utf-8") as f:
        yaml.safe_dump(_report(metadata, network, quant_params, out, labels, float_accuracy,
                               n_regrow, overflow), f, allow_unicode=True, sort_keys=False)
    return out_dir


# ============================================================================
# 檢查、評估量化資料夾
# ============================================================================

def check(quant_dir: str) -> int:
    """重跑 quant_dir 的模型,跟 reference.npz 逐筆比對,印出結果。全部相同回傳 0,否則 1。"""
    model = load_quantized(os.path.join(quant_dir, MODEL_FILENAME))
    with np.load(os.path.join(quant_dir, REFERENCE_FILENAME)) as npz:
        reference = dict(npz)
    split = _split_of(model.metadata["data"], "verify")
    raw = split_input_events(split, model.network.input_shape)
    labels = np.asarray(split.labels)
    decoder = build_decoder({"decoder": model.metadata["decoder"]}, model.network.layers)
    out = quant_forward_split(model.network, decoder, model.params, model.round_mode, raw,
                              int(model.metadata["batch_size"]))

    n_diff = 0
    if not np.array_equal(labels, reference["labels"]):
        print("labels 不同:資料集或取樣方式跟產生時不一樣")
        n_diff += 1
    print(f"accuracy {float(np.mean(out.preds == labels)):.4f}"
          f"(參考 {float(np.mean(reference['preds'] == reference['labels'])):.4f},"
          f"n={labels.shape[0]})")
    for key, value in reference_arrays(out).items():
        diff = value != reference[key]
        idx = np.nonzero(diff.reshape(diff.shape[0], -1).any(axis=1))[0]
        print(f"{key} 不同:{idx.size} 筆 {idx[:20].tolist()}")
        n_diff += idx.size
    return int(bool(n_diff))


def evaluate(quant_dir: str, which: str, n_samples: int | None, seed: int) -> dict:
    """在 which("test" 或 "val")抽 n_samples 筆跑量化模型,結果寫進 quant_dir/eval/,回傳評估內容。

    n_samples 是 None 時:test 用全部,val 用來源 run 的 val_size。容量出界時放大評估用的容量重跑,
    存檔的模型不變。
    """
    model = load_quantized(os.path.join(quant_dir, MODEL_FILENAME))
    meta = model.metadata
    if n_samples is None:
        n_samples = (NMNISTDataset(DATASET_ROOT, max_events=meta["data"]["max_events"])
                     .pool_size("test") if which == "test" else meta["data"]["val_size"])
    split = _build_split(meta["data"]["max_events"], which, seed, n_samples)
    raw = split_input_events(split, model.network.input_shape)
    labels = np.asarray(split.labels)
    decoder = build_decoder({"decoder": meta["decoder"]}, model.network.layers)
    policies = {name: GrowthPolicy(**policy) for name, policy in meta["growth"].items()}
    batch_size = int(meta["batch_size"])
    network, out, n_regrow = run_until_fits(model.network, decoder, model.params, model.round_mode,
                                            raw, batch_size, policies)
    result = {
        "which": which, "n_samples": int(n_samples), "seed": int(seed),
        "accuracy": float(np.mean(out.preds == labels)),
        **overflow_summary(network, decoder, model.params, model.round_mode, raw, batch_size,
                           policies, out),
        "n_regrow": n_regrow,
        "capacity": _capacity(network),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    eval_dir = os.path.join(quant_dir, EVAL_DIRNAME)
    os.makedirs(eval_dir, exist_ok=True)
    np.savez(os.path.join(eval_dir, f"{which}_outputs.npz"), labels=labels, **reference_arrays(out))
    with open(os.path.join(eval_dir, f"{which}.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(result, f, allow_unicode=True, sort_keys=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", nargs="?", help="量化 config 的路徑")
    parser.add_argument("--check", metavar="QUANT_DIR", help="重跑這個量化資料夾,比對參考輸出")
    parser.add_argument("--eval", metavar="QUANT_DIR", help="在 test 或 val 上評估這個量化資料夾")
    parser.add_argument("--which", choices=("test", "val"), default="test", help="--eval 用的資料")
    parser.add_argument("--n", type=int, default=None,
                        help="--eval 抽幾筆(預設:test 全部、val 用來源 run 的 val_size)")
    parser.add_argument("--seed", type=int, default=0, help="--eval 的抽樣 seed,預設 0")
    args = parser.parse_args()
    if sum(x is not None for x in (args.config, args.check, args.eval)) != 1:
        parser.error("config、--check、--eval 要剛好給一個")
    if args.check is not None:
        sys.exit(check(str(resolve_config(args.check))))
    if args.eval is not None:
        quant_dir = str(resolve_config(args.eval))
        print(yaml.safe_dump(evaluate(quant_dir, args.which, args.n, args.seed),
                             allow_unicode=True, sort_keys=False))
        print(f"寫入 {os.path.join(quant_dir, EVAL_DIRNAME)}")
        return
    out_dir = quantize(load_config(str(resolve_config(args.config))))
    with open(os.path.join(out_dir, REPORT_FILENAME), "r", encoding="utf-8") as f:
        print(f.read())
    print(f"寫入 {out_dir}")


if __name__ == "__main__":
    main()
