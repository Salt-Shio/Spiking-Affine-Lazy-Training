"""重跑某個 epoch 存下來的權重,強制 `chunk_size=1`,拿逐事件精確軌跡。

`docs/監測規格.md`「事後精確重現」的決定(2026-09-13):不再存
`summary.npz`/`full_epoch_XXX.npz`。理由:forward 的計算結果跟 `chunk_size`
無關(逐位元相同,見 `salt_core/core.py`)——`chunk_size>1` 時,一個 scan 步
會把最多 `chunk_size` 筆真實事件的 `(a,b)` 仿射映射一次合成掉,trace 只留得住
「這步開始的第一筆事件時間」+「合成完的結果」,中間那幾筆各自的時間/貢獻
在合成的當下就已經不可逆地混在一起,無法事後從結果反推。

與其為了保留這個粒度另外設計新的存檔格式(例如存逐事件的 `(a,b)`),不如
直接利用「權重 + 原始樣本」本身就能唯一決定整條軌跡這件事:只要某個 epoch
的權重還在(`train.weight_snapshot_every` 存的 `weights/epoch_XXX.npz`),把
`chunk_size` 覆蓋成 1 重跑 `run_network_traced`,就能拿到跟訓練當下(不管
原本用哪個 `chunk_size`)逐位元一致、但完全精確、每一步對應一筆真實事件的
軌跡——不需要另存任何逐步/逐事件格式,也不用碰 `salt_core` 的核心運算。

用法:
  python -m example.replay_epoch <exp_dir> <epoch> [--sample S]
"""
import argparse
import dataclasses
import os

from data.src.nmnist import NMNISTDataset
from example.paths import DATASET_ROOT
from example.utils import (WEIGHTS_DIRNAME, load_params_npz, load_run_record,
                           rebuild_layers, weight_snapshot_path)
from salt_core.layers import ConvLayer, raw_events_to_stream, run_network_traced
from salt_core.monitor import summarize_trace_scalars


def _force_chunk_size_one(layers: list) -> list:
    """把每一層的 `chunk_size` 覆蓋成 1。`ConvLayer` 額外把 `max_steps` 補到
    `L`——`chunk_size=1` 下的安全上界(跟 `ConvLayer.__post_init__` 沒填
    `max_steps` 時的 fallback 同一個值),避免沿用舊 `chunk_size` 校準出的
    `max_steps`(可能比 `L` 小)silently 截斷掃描。`FCLayer` 的掃描步數是
    `_run_forward` 現算的(`輸入流長度 / chunk_size`),`chunk_size` 改成 1
    會自動變成完整解析度,沒有欄位需要另外調。"""
    out = []
    for layer in layers:
        if isinstance(layer, ConvLayer):
            out.append(dataclasses.replace(layer, chunk_size=1, max_steps=layer.L))
        else:
            out.append(dataclasses.replace(layer, chunk_size=1))
    return out


def load_epoch_weights(exp_dir: str, epoch: int) -> tuple[list, tuple]:
    """回傳 `(chunk_size=1 的 layers, 那個 epoch 存的權重)`。"""
    run_record = load_run_record(exp_dir)
    layers = _force_chunk_size_one(rebuild_layers(run_record))
    weights_dir = os.path.join(exp_dir, WEIGHTS_DIRNAME)
    params = load_params_npz(weight_snapshot_path(weights_dir, epoch), layers)
    return layers, params


def replay_sample(exp_dir: str, epoch: int, event_times, x, y, c, n_real_events) -> list:
    """給一筆原始樣本(跟訓練資料同格式的
    `(event_times, x, y, c, n_real_events)`),回傳每層的 `LayerForwardTrace`
    (`chunk_size=1`,逐事件精確)。"""
    layers, params = load_epoch_weights(exp_dir, epoch)
    first = layers[0]
    in_stream = raw_events_to_stream(event_times, x, y, c, n_real_events,
                                      h_in=first.h_in, w_in=first.w_in)
    return run_network_traced(layers, in_stream, params)


def load_train_sample(run_record: dict, sample: int):
    """從 `run.yaml` 存的 config 快照重建同一份決定性 train split,取第
    `sample` 筆——跟訓練時 `dataset.build_split` 用的 seed/n_samples 完全對齊,
    才能保證重跑的是同一筆原始資料。"""
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
    layers, params = load_epoch_weights(args.exp_dir, args.epoch)
    first = layers[0]
    in_stream = raw_events_to_stream(*sample, h_in=first.h_in, w_in=first.w_in)
    traces = run_network_traced(layers, in_stream, params)

    print(f"epoch={args.epoch} sample={args.sample}(chunk_size 全部強制為 1,逐事件精確)\n")
    for layer, trace in zip(layers, traces):
        stats = summarize_trace_scalars(trace)
        print(f"[{layer.name}]  ({stats['n']}, {stats['steps']})  "
              f"總 spike={stats['total_spikes']}  有 fire={stats['fired'].size}/{stats['n']}  "
              f"空轉步比例={stats['idle_frac']:.2f}  "
              f"v_steps∈[{stats['v_range'][0]:.3g}, {stats['v_range'][1]:.3g}]")


if __name__ == "__main__":
    main()
