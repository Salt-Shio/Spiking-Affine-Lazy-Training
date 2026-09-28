"""conv SNN 的訓練入口:讀 config、讀 N-MNIST、建實驗目錄、訓練。

容量旋鈕從 config 的起始值開始,訓練中出界就放大、退回最近跑完的 epoch 重來,
理由見 docs/math/conv事件佇列壓縮版推導.md;縮小的規則見 docs/規格書.md。

用法(config 路徑相對於 repo 根目錄,或給絕對路徑):
  python -m example.train_conv_compressed configs/conv/baseline.yaml
"""
import argparse
import datetime
import os
import sys
from typing import NamedTuple

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# 訓練跑好幾小時,輸出導向檔案時 Python 預設整批緩衝,中途看不到進度。
sys.stdout.reconfigure(line_buffering=True)

from data.src.nmnist import NMNISTDataset
from example.checkpoint import Checkpointer
from example.metrics_log import MetricsLog
from example.models.conv_net import build_decoder, build_growth_policies, build_network
from example.paths import DATASET_ROOT, EXPERIMENTS_DIR, resolve_config
from example.training.capacity_control import CapacityControl
from example.training.loop import RunContext, TrainData, initial_state, run_training
from example.training.optim import build_optimizer
from example.training.run_dir import (make_exp_dir, metrics_csv_path, run_header, write_run_record,
                                      write_weights)
from example.utils import (TRAIN_DIRNAME, WEIGHTS_DIRNAME, load_config, split_raw_events,
                           take_raw_events)
from salt_core.io import network_to_dict
from salt_core.layers import ConvLayer
from salt_core.network import Network

# dormant 統計的固定探測樣本數(train split 的前幾筆)
_N_PROBE = 128


class TrainResult(NamedTuple):
    exp_dir: str
    network: Network      # 結束時的網路(含容量)
    params: tuple
    run_record: dict      # 寫進 run.yaml 的內容


def load_nmnist_data(data_cfg: dict) -> TrainData:
    """照 config 的 data 區塊切 N-MNIST 的 train、val split。"""
    dataset = NMNISTDataset(DATASET_ROOT, max_events=data_cfg["max_events"])
    return TrainData(
        train=dataset.build_split(seed=data_cfg["seed_train"], n_samples=data_cfg["train_size"],
                                  which="train"),
        val=dataset.build_split(seed=data_cfg["seed_val"], n_samples=data_cfg["val_size"],
                                which="val"))


def _make_context(cfg: dict, data: TrainData, network: Network, exp_dir: str) -> RunContext:
    model_cfg, train_cfg = cfg["model"], cfg["train"]
    layers = network.layers
    decoder = build_decoder(model_cfg, layers)
    decoder.validate(layers[-1])
    n_train = int(data.train.labels.shape[0])
    batch_size = min(train_cfg["batch_size"], n_train)
    train_raw = split_raw_events(data.train)
    snapshot_every = int(train_cfg.get("weight_snapshot_every", 0))
    snapshot_dir = os.path.join(exp_dir, WEIGHTS_DIRNAME)
    if snapshot_every > 0:
        os.makedirs(snapshot_dir, exist_ok=True)
    # dormant 統計只算 conv 隱藏層
    dormant_names = tuple(layer.name for layer in layers if isinstance(layer, ConvLayer))
    return RunContext(
        data=data, train_raw=train_raw,
        probe=take_raw_events(train_raw, slice(0, min(_N_PROBE, n_train))),
        batch_size=batch_size, epochs=train_cfg["epochs"], seed=train_cfg["seed"],
        optimizer=build_optimizer(train_cfg, n_train, batch_size), decoder=decoder,
        score_cap=train_cfg.get("score_cap"), dormant_names=dormant_names,
        capacity=CapacityControl(build_growth_policies(model_cfg, layers),
                                 int(train_cfg.get("max_steps_reestimate_every", 1))),
        metrics_log=MetricsLog([layer.name for layer in layers], dormant_names,
                               total_epochs=train_cfg["epochs"]),
        checkpointer=Checkpointer(os.path.join(exp_dir, TRAIN_DIRNAME, "checkpoint.npz")),
        snapshot_dir=snapshot_dir, snapshot_every=snapshot_every)


def train(cfg: dict, data: TrainData, exp_dir: str) -> TrainResult:
    """照 cfg(model、train 區塊)在 data 上訓練,產出寫進 exp_dir(要先建好 train/ 子目錄)。

    開訓時先寫一份 run.yaml(config 快照),結束時整份覆寫成完整紀錄。
    """
    network = build_network(cfg["model"])
    ctx = _make_context(cfg, data, network, exp_dir)
    header = run_header(cfg)
    write_run_record(exp_dir, header)

    network, state = run_training(ctx, network, initial_state(ctx, network))

    metrics_log = ctx.metrics_log
    capacity_layers = [layer for layer in network.layers if layer.capacity is not None]
    run_record = {
        **header,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "best": {"val_accuracy": state.best.val_accuracy, "epoch": state.best.epoch},
        "network": network_to_dict(network),
        "last_epoch_obs": ({layer.name: metrics_log.last_needed(layer) for layer in capacity_layers}
                           if metrics_log.rows else {}),
        "capacity_events": [event.to_dict() for event in ctx.capacity.events],
    }
    metrics_log.write_csv(metrics_csv_path(exp_dir))
    write_run_record(exp_dir, run_record)
    write_weights(exp_dir, network, state.params, state.best)
    print(f"訓練結束,結果存到 {exp_dir}")
    print(f"  best val_accuracy = {state.best.val_accuracy:.4f} @ epoch {state.best.epoch}")
    metrics_log.print_summary(network.layers)
    return TrainResult(exp_dir=exp_dir, network=network, params=state.params,
                       run_record=run_record)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="yaml config 檔案路徑(相對於 repo 根目錄,或絕對路徑)")
    args = parser.parse_args()
    cfg = load_config(str(resolve_config(args.config)))
    data = load_nmnist_data(cfg["data"])
    exp_dir = make_exp_dir(cfg.get("run_name", "run"), EXPERIMENTS_DIR)
    train(cfg, data, exp_dir)


if __name__ == "__main__":
    main()
