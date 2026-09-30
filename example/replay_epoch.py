"""重跑某個 epoch 存下的權重,chunk_size 覆蓋成 1,拿每一步對應一筆事件的軌跡。

chunk_size > 1 時一步會合成好幾筆事件,軌跡看不到中間那幾筆。權重加原始樣本就能決定整條軌跡,所以
訓練只存權重,要看時用 chunk_size=1 重跑。結果跟訓練時的 chunk_size 算的只差 float32 捨入。
理由見 docs/監測規格.md「讀端:example/replay_epoch.py(已實作)」。

用法:
  python -m example.replay_epoch <exp_dir> <epoch> [--sample S]
"""
import argparse
import os

from data.src.nmnist import NMNISTDataset
from example.paths import DATASET_ROOT
from example.utils import (WEIGHTS_DIRNAME, grid_input_events, load_run_record,
                           weight_snapshot_path)
from salt_core.io import load_weights
from salt_core.network import Network
from salt_core.trace import summarize_trace_scalars


def load_epoch_weights(exp_dir: str, epoch: int) -> tuple[Network, tuple]:
    """回傳 (那個 epoch 當下的網路,每層換成 chunk_size=1, 那個 epoch 存的權重)。"""
    weights_dir = os.path.join(exp_dir, WEIGHTS_DIRNAME)
    network, params = load_weights(weight_snapshot_path(weights_dir, epoch))
    return network.replace_layers([layer.with_chunk_size(1) for layer in network.layers]), params


def replay_sample(exp_dir: str, epoch: int, event_times, x, y, c, n_real_events) -> list:
    """一筆原始樣本 (event_times, x, y, c, n_real_events) -> 每層的 LayerForwardTrace(chunk_size=1)。"""
    network, params = load_epoch_weights(exp_dir, epoch)
    raw = grid_input_events(event_times, x, y, c, n_real_events, network.input_shape)
    return list(network.apply(params, raw, trace=True).traces)


def load_train_sample(run_record: dict, sample: int):
    """用 run.yaml 的 config 快照重建同一份 train split,取第 sample 筆,跟訓練時是同一筆資料。"""
    data_cfg = run_record["config"]["data"]
    dataset = NMNISTDataset(DATASET_ROOT, max_events=data_cfg["max_events"])
    split = dataset.build_split(seed=data_cfg["seed_train"],
                               n_samples=data_cfg["train_size"], which="train")
    return (split.event_times[sample], split.x[sample], split.y[sample],
            split.c[sample], split.n_real_events[sample])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("exp_dir", help="experiments/<run> 目錄")
    parser.add_argument("epoch", type=int, help="要重跑的 epoch(要有對應的 weights/epoch_XXX.npz)")
    parser.add_argument("--sample", type=int, default=0, help="train split 第幾筆樣本(預設 0)")
    args = parser.parse_args()

    run_record = load_run_record(args.exp_dir)
    sample = load_train_sample(run_record, args.sample)
    network, params = load_epoch_weights(args.exp_dir, args.epoch)
    traces = network.apply(params, grid_input_events(*sample, network.input_shape),
                           trace=True).traces

    print(f"epoch={args.epoch} sample={args.sample}(chunk_size 全部強制為 1,逐事件精確)\n")
    for layer, trace in zip(network.layers, traces):
        stats = summarize_trace_scalars(trace)
        print(f"[{layer.name}]  ({stats['n']}, {stats['steps']})  "
              f"總 spike={stats['total_spikes']}  有 fire={stats['fired'].size}/{stats['n']}  "
              f"空轉步比例={stats['idle_frac']:.2f}  "
              f"v_steps∈[{stats['v_range'][0]:.3g}, {stats['v_range'][1]:.3g}]")


if __name__ == "__main__":
    main()
