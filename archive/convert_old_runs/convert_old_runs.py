"""一次性轉換:experiments/ 底下的舊 run 補上網路描述,L 改名 max_queue_len(階段 5.4)。

每個 run:
- 權重檔(weights/epoch_XXX.npz、train/params.npz、best_params.npz、checkpoint.npz)改成
  salt_core.io 的格式,網路 = config 快照 build_network + metrics.csv 那個 epoch 的容量。
  epoch_XXX 用第 XXX 列,best_params 用 best epoch,params 用最後一列,checkpoint 用檔案記的 epoch。
- run.yaml:config 快照的 L、L_grow_factor 改名;final_capacity 換成 network。
- golden/report.yaml:容量的 L 改名。
檢查:metrics.csv 最後一列的容量等於 final_capacity;寫回後讀出的權重跟原檔逐位元相等。
已經轉過的 run(run.yaml 有 network)raise。

用法(已跑過,留作紀錄):python archive/convert_old_runs/convert_old_runs.py
"""
import csv
import os

import numpy as np
import yaml

from example.metrics_log import KNOB_COLUMNS
from example.models.conv_net import build_network
from example.paths import EXPERIMENTS_DIR
from example.utils import TRAIN_DIRNAME, WEIGHTS_DIRNAME, weight_snapshot_path
from salt_core.capacity import Capacity
from salt_core.io import network_to_dict, weights_from_arrays, weights_to_arrays

_RENAMED_KEYS = {"L": "max_queue_len", "L_grow_factor": "max_queue_len_grow_factor"}


def _renamed(entry: dict) -> dict:
    return {_RENAMED_KEYS.get(key, key): value for key, value in entry.items()}


def _read_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def _networks_by_epoch(base_network, metrics_path: str) -> dict:
    """epoch -> 套上 metrics.csv 那一列容量的網路。"""
    with open(metrics_path, newline="", encoding="utf-8") as f:
        rows = {int(row["epoch"]): row for row in csv.DictReader(f)}
    return {epoch: base_network.replace_layers(
                [layer if layer.capacity is None else layer.with_capacity(Capacity(**{
                    knob: int(row[f"{layer.name}_{KNOB_COLUMNS[knob].capacity}"])
                    for knob in layer.capacity}))
                 for layer in base_network.layers])
            for epoch, row in rows.items()}


def _write_npz_checked(path: str, arrays: dict, weight_names: list) -> None:
    """寫到暫存檔再換名;寫完讀回,權重跟 arrays 逐位元相等、dtype 相同才算數。"""
    tmp_path = path + ".converting.npz"
    np.savez(tmp_path, **arrays)
    with np.load(tmp_path) as written:
        for name in weight_names:
            if written[name].dtype != arrays[name].dtype or \
                    not np.array_equal(written[name], arrays[name]):
                raise RuntimeError(f"{path} 的 {name} 寫回後不一致")
        _network, _weights = weights_from_arrays(written)
    os.replace(tmp_path, path)


def _convert_weights_file(path: str, network) -> None:
    names = [layer.name for layer in network.layers]
    with np.load(path) as old:
        if sorted(old.files) != sorted(names):
            raise RuntimeError(f"{path} 的欄位 {old.files} 跟層名 {names} 對不上")
        weights = [old[name] for name in names]
    _write_npz_checked(path, weights_to_arrays(network, weights), names)


def _convert_checkpoint(path: str, networks: dict) -> None:
    with np.load(path) as old:
        old_arrays = {key: old[key] for key in old.files}
    network = networks[int(old_arrays["epoch"])]
    names = [layer.name for layer in network.layers]
    weights = [old_arrays.pop(f"param__{i}") for i in range(len(names))]
    if any(key.startswith("param__") for key in old_arrays):
        raise RuntimeError(f"{path} 的權重份數跟層數對不上")
    arrays = weights_to_arrays(network, weights)
    arrays.update(old_arrays)
    _write_npz_checked(path, arrays, names)


def convert_run(exp_dir: str) -> dict:
    """轉一個 run,回傳轉了幾個檔。"""
    train_dir = os.path.join(exp_dir, TRAIN_DIRNAME)
    run_record = _read_yaml(os.path.join(train_dir, "run.yaml"))
    if "network" in run_record:
        raise RuntimeError(f"{exp_dir} 已經轉過")
    model_cfg = run_record["config"]["model"]
    model_cfg["layers"] = [_renamed(entry) for entry in model_cfg["layers"]]

    networks = _networks_by_epoch(build_network(model_cfg),
                                  os.path.join(train_dir, "metrics.csv"))
    last_network = networks[max(networks)]
    final_capacity = {name: _renamed(capacity)
                      for name, capacity in run_record["final_capacity"].items()}
    last_capacity = {layer.name: dict(layer.capacity) for layer in last_network.layers
                     if layer.capacity is not None}
    if last_capacity != final_capacity:
        raise RuntimeError(f"{exp_dir}:metrics.csv 最後一列 {last_capacity} "
                           f"跟 final_capacity {final_capacity} 不同")

    weights_dir = os.path.join(exp_dir, WEIGHTS_DIRNAME)
    n_snapshots = 0
    for epoch, network in networks.items():
        path = weight_snapshot_path(weights_dir, epoch)
        if os.path.isfile(path):
            _convert_weights_file(path, network)
            n_snapshots += 1
    _convert_weights_file(os.path.join(train_dir, "params.npz"), last_network)
    _convert_weights_file(os.path.join(train_dir, "best_params.npz"),
                          networks[int(run_record["best"]["epoch"])])
    _convert_checkpoint(os.path.join(train_dir, "checkpoint.npz"), networks)

    run_record = {key: value for key, value in run_record.items() if key != "final_capacity"}
    run_record["network"] = network_to_dict(last_network)
    _write_yaml(os.path.join(train_dir, "run.yaml"), run_record)

    report_path = os.path.join(exp_dir, "golden", "report.yaml")
    if os.path.isfile(report_path):
        report = _read_yaml(report_path)
        for key in ("run_capacity", "golden_capacity"):
            report[key] = {name: _renamed(capacity) for name, capacity in report[key].items()}
        _write_yaml(report_path, report)
    return {"snapshots": n_snapshots, "golden": os.path.isfile(report_path)}


def main() -> None:
    for name in sorted(os.listdir(EXPERIMENTS_DIR)):
        exp_dir = os.path.join(EXPERIMENTS_DIR, name)
        if os.path.isfile(os.path.join(exp_dir, TRAIN_DIRNAME, "run.yaml")):
            print(f"{name}: {convert_run(exp_dir)}")


if __name__ == "__main__":
    main()
