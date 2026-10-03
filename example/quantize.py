"""量化實驗:浮點 run 的權重照量化規格轉成整數模型,跑滿驗證樣本,存成可以獨立重跑的資料夾。

產生:讀來源 run 的權重 -> val 前幾筆量膜電位範圍 M -> 算每層量化參數 -> 跑驗證樣本
    (容量出界就放大重跑)-> 寫 experiments/<來源 run>/quant/<規格名>/。
檢查:只讀量化資料夾跟資料集,重跑驗證樣本,逐筆比對 reference.npz。有不同就回傳非 0。

用法(路徑相對於 repo 根目錄,或給絕對路徑):
  python -m example.quantize configs/quant/baseline.yaml
  python -m example.quantize --check experiments/<來源 run>/quant/<規格名>
輸出:model.npz(salt_core.io.save_quantized 格式)、reference.npz(逐筆輸出)、report.yaml
"""
import argparse
import datetime
import os
import sys
from typing import NamedTuple

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from data.src.nmnist import NMNISTDataset, NMNISTSplit
from example.models.conv_net import build_decoder, build_growth_policies
from example.paths import DATASET_ROOT, EXPERIMENTS_DIR, REPO_ROOT, resolve_config
from example.utils import (WEIGHTS_DIRNAME, get_git_commit_hash, load_config, load_run_params,
                           load_run_record, make_evaluate, split_input_events, take_input_events,
                           weight_snapshot_path)
from salt_core.capacity import LayerDiag, grown_to_fit
from salt_core.io import load_quantized, load_weights, save_quantized
from salt_core.network import InputEvents, Network
from salt_core.quant.backend import QuantBackend
from salt_core.quant.calibrate import merge_v_ranges, v_abs_max_per_channel, v_range_per_channel
from salt_core.quant.convert import LayerQuantSpec, build_quantized_params
from salt_core.quant.fixed_point import RoundMode

QUANT_DIRNAME = "quant"
MODEL_FILENAME = "model.npz"
REFERENCE_FILENAME = "reference.npz"
REPORT_FILENAME = "report.yaml"

# config 各節認得的 key
_SECTION_KEYS = {
    "source": {"run", "params"},
    "calibration": {"n_samples"},
    "verify": {"n_samples"},
    "spec": {"bits", "f_a", "f_V", "round_mode", "out_granularity", "clip_percentile",
             "overflow_mode"},
}
_OUT_GRANULARITIES = ("per_channel", "per_tensor")


# ============================================================================
# 共用:量 M、量化規格、整批量化 forward(golden_output 也用)
# ============================================================================

def calibrate_v_abs_max(network: Network, params, raw: InputEvents,
                        n_samples: int) -> list[np.ndarray]:
    """raw 前 n_samples 筆逐筆跑浮點 forward,回傳對齊 layers 的每層逐 channel 膜電位最大量值 M。

    network 每層要 chunk_size=1,否則 v_range_per_channel raise ValueError。
    """
    per_sample = [v_range_per_channel(network.layers, network.apply(
                      params, take_input_events(raw, i), trace=True).traces)
                  for i in range(n_samples)]
    return v_abs_max_per_channel(merge_v_ranges(per_sample))


def quant_layer_specs(base: LayerQuantSpec, n_layers: int,
                      out_per_channel: bool = True) -> list[LayerQuantSpec]:
    """每層的量化規格:前面的層都用 base,最後一層是不 fire 的膜電位回歸輸出層,
    逐 channel 或整層共用 s_c 照 out_per_channel。"""
    return [base] * (n_layers - 1) + [base._replace(per_channel=out_per_channel, fires=False)]


class QuantSplitOutput(NamedTuple):
    """quant_forward_split 的回傳,N 是樣本數。"""
    v_final_int: np.ndarray   # (N, n_out) 輸出層暫存器值
    preds: np.ndarray         # (N,)
    spike_count: np.ndarray   # (N, n_layers)
    truncated: np.ndarray     # (N, n_layers) bool,容量出界
    overflowed: np.ndarray    # (N, n_layers) bool,暫存器溢位過
    needed: dict              # 層名 -> 旋鈕 -> (N,) 需要的容量;沒有容量的層是空 dict


REFERENCE_FIELDS = ("v_final_int", "preds", "spike_count", "truncated", "overflowed")


def quant_forward_split(network: Network, decoder, params, round_mode: RoundMode | str,
                        raw: InputEvents, batch_size: int) -> QuantSplitOutput:
    """raw 的全部樣本分批跑整數 forward。params 當 jit 常數(含 Python 整數 f_a、f_V、i_V)。"""
    backend = QuantBackend(round_mode=round_mode)

    @jax.jit
    def run_batch(raw_batch):
        output = network.apply_batched(params, raw_batch, backend=backend)
        scores, _ = jax.vmap(decoder.decode)(backend.readout(output.last, params[-1]))
        n = scores.shape[0]
        truncated = [jnp.zeros(n, dtype=bool) if layer.capacity is None
                     else ~layer.capacity.fits(diag)
                     for layer, diag in zip(network.layers, output.diags)]
        return {"v_final_int": output.last.v_final,
                "preds": jnp.argmax(scores, axis=1),
                "spike_count": jnp.stack([jnp.sum(r.spike_mask, axis=(1, 2))
                                          for r in output.results], axis=1),
                "truncated": jnp.stack(truncated, axis=1),
                "overflowed": jnp.stack([jnp.any(r.overflowed, axis=(1, 2))
                                         for r in output.results], axis=1),
                "needed": [diag.needed for diag in output.diags]}

    n = raw.event_times.shape[0]
    parts = [jax.tree_util.tree_map(
                 np.asarray, run_batch(take_input_events(raw, slice(start, start + batch_size))))
             for start in range(0, n, batch_size)]
    arrays = {key: np.concatenate([part[key] for part in parts]) for key in REFERENCE_FIELDS}
    needed = {layer.name: {knob: np.concatenate([part["needed"][i][knob] for part in parts])
                           for knob in parts[0]["needed"][i]}
              for i, layer in enumerate(network.layers)}
    return QuantSplitOutput(**arrays, needed=needed)


def reference_arrays(out: QuantSplitOutput) -> dict[str, np.ndarray]:
    """存檔、比對用的逐筆輸出(QuantSplitOutput 去掉 needed)。"""
    return {key: getattr(out, key) for key in REFERENCE_FIELDS}


def grown_for_needed(layers: list, policies: dict, needed: dict) -> list:
    """用每個旋鈕在所有樣本裡的最大需求放大每一層,公式照 policies。

    needed: 層名 -> 旋鈕 -> 逐筆需求陣列。
    """
    diags = [LayerDiag(spike_count=0, firing_rate=0.0,
                       needed={knob: int(v.max()) for knob, v in needed[layer.name].items()})
             for layer in layers]
    return grown_to_fit(layers, policies, diags)


# ============================================================================
# 產生量化資料夾
# ============================================================================

def check_quant_config(cfg: dict) -> None:
    """config 缺節、有不認得的 key、out_granularity 不合法時 raise ValueError。"""
    unknown = sorted(set(cfg) - set(_SECTION_KEYS))
    if unknown:
        raise ValueError(f"config 最外層有不認得的 key:{unknown},認得的是 {sorted(_SECTION_KEYS)}")
    for section, allowed in _SECTION_KEYS.items():
        if section not in cfg:
            raise ValueError(f"config 少了 {section} 節")
        unknown = sorted(set(cfg[section]) - allowed)
        if unknown:
            raise ValueError(f"config 的 {section} 有不認得的 key:{unknown},認得的是 {sorted(allowed)}")
    if cfg["spec"]["out_granularity"] not in _OUT_GRANULARITIES:
        raise ValueError(f"spec.out_granularity 要是 {_OUT_GRANULARITIES} 之一,"
                         f"拿到 {cfg['spec']['out_granularity']!r}")


def spec_name(spec: dict) -> str:
    """規格組成的資料夾名,例如 b8_fa10_fv10_round_pc_clip100_wrap。"""
    granularity = "pc" if spec["out_granularity"] == "per_channel" else "pt"
    return (f"b{spec['bits']}_fa{spec['f_a']}_fv{spec['f_V']}_{spec['round_mode']}_"
            f"{granularity}_clip{spec['clip_percentile']:g}_{spec['overflow_mode']}")


def _load_source_params(source_dir: str, which) -> tuple[Network, tuple, int]:
    """回傳 (網路, 浮點權重, 權重的 epoch)。which 是 "best"、"final" 或 epoch 編號。"""
    run_record = load_run_record(source_dir)
    if which == "best":
        network, params = load_run_params(source_dir, "best")
        return network, params, int(run_record["best"]["epoch"])
    if which == "final":
        network, params = load_run_params(source_dir, "final")
        return network, params, int(run_record["config"]["train"]["epochs"]) - 1
    if isinstance(which, int):
        network, params = load_weights(weight_snapshot_path(
            os.path.join(source_dir, WEIGHTS_DIRNAME), which))
        return network, params, which
    raise ValueError(f"source.params 要是 best、final 或 epoch 編號,拿到 {which!r}")


def _val_head(max_events: int, seed_val: int, val_size: int, n: int) -> NMNISTSplit:
    """照 run 的設定切出 val split(val_size 筆),取前 n 筆。n 超過 val_size 時 raise ValueError。"""
    if n > val_size:
        raise ValueError(f"驗證樣本數 {n} 超過 val_size={val_size}")
    dataset = NMNISTDataset(DATASET_ROOT, max_events=max_events)
    split = dataset.build_split(seed=seed_val, n_samples=val_size, which="val")
    return jax.tree_util.tree_map(lambda a: a[:n], split)


def _layer_counts(names: list[str], flags: np.ndarray) -> dict[str, int]:
    return {name: int(flags[:, i].sum()) for i, name in enumerate(names)}


def _data_meta(cfg: dict, data_cfg: dict) -> dict:
    """取樣方式:val 的切法跟量 M、驗證各用前幾筆。校準筆數超過驗證筆數時 raise ValueError。"""
    meta = {"split": "val", "max_events": int(data_cfg["max_events"]),
            "seed_val": int(data_cfg["seed_val"]), "val_size": int(data_cfg["val_size"]),
            "n_calibration": int(cfg["calibration"]["n_samples"]),
            "n_verify": int(cfg["verify"]["n_samples"] or data_cfg["val_size"])}
    if meta["n_calibration"] > meta["n_verify"]:
        raise ValueError(f"calibration.n_samples={meta['n_calibration']} 不能超過"
                         f"驗證樣本數 {meta['n_verify']}(量 M 用驗證樣本的前幾筆)")
    return meta


def _build_params(network: Network, float_params, spec_cfg: dict,
                  v_abs_max: list[np.ndarray]) -> list:
    """照 config 的 spec 節算每層的 QuantizedLayerParams。"""
    base = LayerQuantSpec(bits=spec_cfg["bits"], f_a=spec_cfg["f_a"], f_V=spec_cfg["f_V"],
                          clip_percentile=float(spec_cfg["clip_percentile"]),
                          overflow_mode=spec_cfg["overflow_mode"])
    specs = quant_layer_specs(base, len(network.layers),
                              out_per_channel=spec_cfg["out_granularity"] == "per_channel")
    return build_quantized_params(network.layers, float_params, specs, v_abs_max)


def _run_until_fits(network: Network, decoder, params, round_mode: str, raw: InputEvents,
                    batch_size: int, policies: dict) -> tuple[Network, QuantSplitOutput, int]:
    """跑整數 forward,有樣本容量出界就放大容量重跑。回傳 (放得下的網路, 輸出, 重跑次數)。"""
    n_regrow = 0
    out = quant_forward_split(network, decoder, params, round_mode, raw, batch_size)
    while out.truncated.any():
        network = network.replace_layers(grown_for_needed(network.layers, policies, out.needed))
        out = quant_forward_split(network, decoder, params, round_mode, raw, batch_size)
        n_regrow += 1
    return network, out, n_regrow


def _report(metadata: dict, network: Network, params: list, out: QuantSplitOutput,
            labels: np.ndarray, float_accuracy: float, n_regrow: int) -> dict:
    """report.yaml 的內容。"""
    names = [layer.name for layer in network.layers]
    return {
        "source": metadata["source"],
        "spec": metadata["spec"],
        "data": metadata["data"],
        "i_V": {name: int(p.i_V) for name, p in zip(names, params)},
        "register_bits": {name: int(p.i_V + p.f_V) for name, p in zip(names, params)},
        "accuracy": float(np.mean(out.preds == labels)),
        "float_accuracy": float_accuracy,
        "overflowed_samples": _layer_counts(names, out.overflowed),
        "truncated_samples": _layer_counts(names, out.truncated),
        "n_regrow": n_regrow,
        "capacity": {layer.name: dict(layer.capacity) for layer in network.layers
                     if layer.capacity is not None},
        "git_commit": metadata["git_commit"],
        "timestamp": metadata["timestamp"],
    }


def quantize(cfg: dict, experiments_dir: str | os.PathLike = EXPERIMENTS_DIR) -> str:
    """照 cfg 產生量化資料夾,回傳它的路徑。來源 run 是 experiments_dir/<source.run>。

    資料夾已經存在時 raise FileExistsError。config 不合法(見 check_quant_config)、驗證樣本數超過
    run 的 val_size、校準樣本數超過驗證樣本數、或 run 的 decoder 不是 membrane_regression 時
    raise ValueError。
    """
    check_quant_config(cfg)
    spec_cfg, source_cfg = cfg["spec"], cfg["source"]
    source_dir = os.path.join(experiments_dir, source_cfg["run"])
    out_dir = os.path.join(source_dir, QUANT_DIRNAME, spec_name(spec_cfg))
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
    split = _val_head(data_meta["max_events"], data_meta["seed_val"], data_meta["val_size"],
                      data_meta["n_verify"])
    raw = split_input_events(split, float_network.input_shape)

    network = float_network.replace_layers(
        [layer.with_chunk_size(1) for layer in float_network.layers])
    v_abs_max = calibrate_v_abs_max(network, float_params, raw, data_meta["n_calibration"])
    quant_params = _build_params(network, float_params, spec_cfg, v_abs_max)
    network, out, n_regrow = _run_until_fits(network, decoder, quant_params,
                                             spec_cfg["round_mode"], raw, batch_size, policies)
    float_accuracy, _, _, _ = make_evaluate(float_network, decoder, batch_size, policies)(
        float_params, split)

    metadata = {
        "source": {"run": source_cfg["run"], "params": source_cfg["params"], "epoch": epoch},
        "spec": dict(spec_cfg),
        "data": data_meta,
        "decoder": decoder_kind,
        "batch_size": batch_size,
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
                               n_regrow), f, allow_unicode=True, sort_keys=False)
    return out_dir


# ============================================================================
# 檢查量化資料夾
# ============================================================================

def check(quant_dir: str) -> int:
    """重跑 quant_dir 的模型,跟 reference.npz 逐筆比對,印出結果。全部相同回傳 0,否則 1。"""
    model = load_quantized(os.path.join(quant_dir, MODEL_FILENAME))
    with np.load(os.path.join(quant_dir, REFERENCE_FILENAME)) as npz:
        reference = dict(npz)
    data_meta = model.metadata["data"]
    split = _val_head(data_meta["max_events"], data_meta["seed_val"], data_meta["val_size"],
                      data_meta["n_verify"])
    raw = split_input_events(split, model.network.input_shape)
    labels = np.asarray(split.labels)
    n = labels.shape[0]
    decoder = build_decoder({"decoder": model.metadata["decoder"]}, model.network.layers)
    out = quant_forward_split(model.network, decoder, model.params, model.round_mode, raw,
                              int(model.metadata["batch_size"]))

    n_diff = 0
    if not np.array_equal(labels, reference["labels"]):
        print("labels 不同:資料集或取樣方式跟產生時不一樣")
        n_diff += 1
    print(f"accuracy {float(np.mean(out.preds == labels)):.4f}"
          f"(參考 {float(np.mean(reference['preds'] == reference['labels'])):.4f},n={n})")
    for key, value in reference_arrays(out).items():
        diff = value != reference[key]
        idx = np.nonzero(diff.reshape(diff.shape[0], -1).any(axis=1))[0]
        print(f"{key} 不同:{idx.size} 筆 {idx[:20].tolist()}")
        n_diff += idx.size
    return int(bool(n_diff))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", nargs="?", help="量化 config 的路徑")
    parser.add_argument("--check", metavar="QUANT_DIR", help="重跑這個量化資料夾,比對參考輸出")
    args = parser.parse_args()
    if (args.config is None) == (args.check is None):
        parser.error("config 跟 --check 要剛好給一個")
    if args.check is not None:
        sys.exit(check(str(resolve_config(args.check))))
    out_dir = quantize(load_config(str(resolve_config(args.config))))
    with open(os.path.join(out_dir, REPORT_FILENAME), "r", encoding="utf-8") as f:
        print(f.read())
    print(f"寫入 {out_dir}")


if __name__ == "__main__":
    main()
