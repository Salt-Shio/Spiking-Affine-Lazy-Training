"""conv SNN 的訓練入口:讀 config、讀 N-MNIST、建實驗目錄、訓練。

容量旋鈕從 config 的起始值開始,訓練中出界就放大、退回最近跑完的 epoch 重來,
理由見 docs/math/conv事件佇列壓縮版推導.md;縮小的規則見 docs/規格書.md。

用法(路徑相對於 repo 根目錄,或給絕對路徑):
  python -m example.train configs/conv/baseline.yaml
  python -m example.train --resume experiments/<run 目錄>    # 從 checkpoint 接著練
"""
import argparse
import datetime
import os
import sys
from typing import NamedTuple

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# 訓練跑好幾小時,輸出導向檔案時 Python 預設整批緩衝,中途看不到進度。
sys.stdout.reconfigure(line_buffering=True)

import jax

from data.src.nmnist import NMNISTDataset
from example.checkpoint import Checkpointer
from example.metrics_log import MetricsLog
from example.models.conv_net import build_decoder, build_growth_policies, build_network
from example.paths import DATASET_ROOT, EXPERIMENTS_DIR, resolve_config
from example.training.capacity_control import CapacityControl
from example.training.loop import (RunContext, TrainData, TrainState, initial_state, run_training,
                                   state_from_checkpoint)
from example.training.optim import build_optimizer
from example.training.run_dir import (make_exp_dir, metrics_csv_path, params_path, resume_entry,
                                      run_header, write_run_record, write_weights)
from example.utils import (TRAIN_DIRNAME, WEIGHTS_DIRNAME, load_config, load_run_record,
                           split_input_events, take_input_events)
from salt_core.io import network_to_dict
from salt_core.network import Network

# dormant 統計的固定探測樣本數(train split 的前幾筆)
_N_PROBE = 128

# config 各節認得的 key。model.layers、model.layer_defaults 裡的 key 由 build_network 檢查。
_TOP_KEYS = {"run_name", "model", "data", "train"}
_MODEL_KEYS = {"decoder", "input_shape", "layer_defaults", "layers"}
_DATA_KEYS = {"max_events", "seed_train", "seed_val", "train_size", "val_size"}
_TRAIN_KEYS = {"lr", "epochs", "batch_size", "seed", "weight_decay", "lr_cosine_decay",
               "lr_cosine_alpha", "grad_clip_norm", "score_cap", "dormant_layers",
               "weight_snapshot_every", "shrink_check_every"}


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


def check_config_keys(cfg: dict) -> None:
    """config 裡有不認得的 key 時 raise ValueError,訊息寫出 key 跟所在的節。"""
    sections = [("最外層", cfg, _TOP_KEYS), ("model", cfg.get("model") or {}, _MODEL_KEYS),
                ("data", cfg.get("data") or {}, _DATA_KEYS),
                ("train", cfg.get("train") or {}, _TRAIN_KEYS)]
    for where, section, allowed in sections:
        unknown = sorted(set(section) - allowed)
        if unknown:
            raise ValueError(f"config 的 {where} 有不認得的 key:{unknown},認得的是 {sorted(allowed)}")


def _make_context(cfg: dict, data: TrainData, network: Network, exp_dir: str) -> RunContext:
    model_cfg, train_cfg = cfg["model"], cfg["train"]
    layers = network.layers
    decoder = build_decoder(model_cfg, layers)
    decoder.validate(layers[-1])
    n_train = int(data.train.labels.shape[0])
    batch_size = min(train_cfg["batch_size"], n_train)
    train_raw = split_input_events(data.train, network.input_shape)
    snapshot_every = int(train_cfg.get("weight_snapshot_every", 0))
    snapshot_dir = os.path.join(exp_dir, WEIGHTS_DIRNAME)
    if snapshot_every > 0:
        os.makedirs(snapshot_dir, exist_ok=True)
    # dormant 統計哪些層由 config 決定,沒填不統計
    dormant_names = tuple(train_cfg.get("dormant_layers") or ())
    unknown = sorted(set(dormant_names) - {layer.name for layer in layers})
    if unknown:
        raise ValueError(f"train.dormant_layers 有網路裡沒有的層:{unknown},"
                         f"網路的層是 {[layer.name for layer in layers]}")
    return RunContext(
        data=data, train_raw=train_raw,
        probe=take_input_events(train_raw, slice(0, min(_N_PROBE, n_train))),
        batch_size=batch_size, epochs=train_cfg["epochs"], seed=train_cfg["seed"],
        optimizer=build_optimizer(train_cfg, n_train, batch_size), decoder=decoder,
        score_cap=train_cfg.get("score_cap"), dormant_names=dormant_names,
        capacity=CapacityControl(build_growth_policies(model_cfg, layers),
                                 int(train_cfg.get("shrink_check_every", 1))),
        metrics_log=MetricsLog([layer.name for layer in layers], dormant_names,
                               total_epochs=train_cfg["epochs"]),
        checkpointer=Checkpointer(os.path.join(exp_dir, TRAIN_DIRNAME, "checkpoint.npz")),
        snapshot_dir=snapshot_dir, snapshot_every=snapshot_every)


def _starting_point(cfg: dict, ctx: RunContext, network: Network,
                    exp_dir: str) -> tuple[Network, TrainState, dict]:
    """回傳 (起始網路, 起始狀態, run.yaml 開頭)。

    沒有 checkpoint:config 建的網路、初始狀態、新的開頭。
    有 checkpoint:checkpoint 的網路跟狀態,指標、容量事件換回存檔當下的紀錄,
    沿用原本的開頭,resumes 多一筆。
    """
    if not ctx.checkpointer.exists():
        return network, initial_state(ctx, network), run_header(cfg)
    saved = ctx.checkpointer.load(
        opt_state_template=ctx.optimizer.init(network.init(jax.random.PRNGKey(0))))
    ctx.metrics_log.restore(saved.history["metrics_rows"])
    ctx.capacity.restore(saved.history["capacity_events"])
    header = load_run_record(exp_dir)
    header["resumes"].append(resume_entry(saved.epoch + 1))
    print(f"從 checkpoint 接著練:epoch {saved.epoch + 1} 開始")
    return saved.network, state_from_checkpoint(saved), header


def train(cfg: dict, data: TrainData, exp_dir: str) -> TrainResult:
    """照 cfg(model、train 區塊)在 data 上訓練,產出寫進 exp_dir(要先建好 train/ 子目錄)。

    exp_dir 有 checkpoint 時從它接著練。開訓時先寫一份 run.yaml(config 快照),
    結束時整份覆寫成完整紀錄。exp_dir 的 run 已經跑完(有 params.npz)、config 有不認得的
    key 時 raise ValueError。
    """
    check_config_keys(cfg)
    if os.path.isfile(params_path(exp_dir)):
        raise ValueError(f"{exp_dir} 已經跑完,不能再接著練")
    network = build_network(cfg["model"])
    ctx = _make_context(cfg, data, network, exp_dir)
    network, state, header = _starting_point(cfg, ctx, network, exp_dir)
    write_run_record(exp_dir, header)

    network, state = run_training(ctx, network, state)

    metrics_log = ctx.metrics_log
    capacity_layers = [layer for layer in network.layers if layer.capacity is not None]
    run_record = {
        **header,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "best": {"val_accuracy": state.best.val_accuracy, "epoch": state.best.epoch},
        "network": network_to_dict(network),
        "last_epoch_obs": ({layer.name: metrics_log.last_needed(layer) for layer in capacity_layers}
                           if metrics_log.rows else {}),
        "capacity_events": list(ctx.capacity.events),
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
    parser.add_argument("config", nargs="?",
                        help="yaml config 檔案路徑(相對於 repo 根目錄,或絕對路徑)")
    parser.add_argument("--resume", metavar="EXP_DIR",
                        help="從這個 run 目錄的 checkpoint 接著練,config 用它開訓時的快照")
    args = parser.parse_args()
    if (args.config is None) == (args.resume is None):
        parser.error("config 跟 --resume 要剛好給一個")
    if args.resume is not None:
        exp_dir = str(resolve_config(args.resume))
        cfg = load_run_record(exp_dir)["config"]
    else:
        cfg = load_config(str(resolve_config(args.config)))
        exp_dir = make_exp_dir(cfg.get("run_name", "run"), EXPERIMENTS_DIR)
    train(cfg, load_nmnist_data(cfg["data"]), exp_dir)


if __name__ == "__main__":
    main()
