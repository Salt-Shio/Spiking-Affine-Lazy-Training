"""路徑 A/B(docs/問題紀錄.md §十五)通用版:給一個實驗目錄 + 爆炸前一個
epoch 的權重快照 + 爆炸發生的 epoch,對整個 train split 做 forward,找
loss/spike 數離群的樣本;再重建該 epoch 實際的 batch 切法(PRNG split,
跟浮點非決定性無關,可精確重算),看那些離群樣本落在哪個 batch。

`snapshot_epoch` 若剛好是 `target_epoch - 1`(權重存夠密,`weight_snapshot_every=1`),
就是路徑 B——這份快照是真正爆炸前一刻的權重,不是近似值。

用法:
  python example/debug-test/inspect_explosion_batch.py <exp_dir_name> \
      --snapshot-epoch N --target-epoch M [--top-k K]
"""
import argparse
import csv
import dataclasses
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax

from data.src.nmnist import NMNISTDataset
from example.models.conv_net import ConvNetCompressed, build_decoder, build_network
from example.paths import DATASET_ROOT, EXPERIMENTS_DIR
from example.utils import (TRAIN_DIRNAME, WEIGHTS_DIRNAME, load_params_npz,
                           load_run_record, weight_snapshot_path)
from salt_core.layers import ConvLayer


def _rebuild_layers_at_epoch(run_record: dict, exp_dir: str, epoch: int) -> list:
    """跟 `example.utils.rebuild_layers`的差別:那個函式還原的是**整個 run
    結束時**的最終容量(`run.yaml` 的 `final_capacity`),但 `max_steps`/
    `max_out_spikes` 訓練中途可能縮小過(`ConvLayer.shrink_max_steps`/
    `shrink_max_out_spikes`,見 docs/問題紀錄.md §十五 踩過的坑)——要重建
    「某個中途 epoch 當下」的權重,必須用那個 epoch **當下**的容量,不能用
    run 結束時的容量(可能已經比當下小,會把還沒縮小前的真實輸出/掃描步數
    截斷)。這裡改成直接讀 `metrics.csv` 裡對應 epoch 那一列的
    `{層名}_max_event_queue`/`_max_layer_spikes`/`_max_steps`。"""
    layers = build_network(run_record["config"]["model"])
    metrics_path = os.path.join(exp_dir, TRAIN_DIRNAME, "metrics.csv")
    with open(metrics_path, newline="", encoding="utf-8") as f:
        rows = {int(row["epoch"]): row for row in csv.DictReader(f)}
    row = rows[epoch]
    rebuilt = []
    for layer in layers:
        if isinstance(layer, ConvLayer):
            layer = dataclasses.replace(
                layer,
                L=int(row[f"{layer.name}_max_event_queue"]),
                max_out_spikes=int(row[f"{layer.name}_max_layer_spikes"]),
                max_steps=int(row[f"{layer.name}_max_steps"]))
        rebuilt.append(layer)
    return rebuilt


def _load_layers_and_params(exp_dir: str, snapshot_epoch: int):
    run_record = load_run_record(exp_dir)
    layers = _rebuild_layers_at_epoch(run_record, exp_dir, snapshot_epoch)
    weights_dir = os.path.join(exp_dir, WEIGHTS_DIRNAME)
    params = load_params_npz(weight_snapshot_path(weights_dir, snapshot_epoch), layers)
    return run_record, layers, params


def _rebuild_train_split(run_record: dict):
    data_cfg = run_record["config"]["data"]
    dataset = NMNISTDataset(DATASET_ROOT, max_events=data_cfg["max_events"])
    return dataset.build_split(seed=data_cfg["seed_train"],
                               n_samples=data_cfg["train_size"], which="train")


def _epoch_permutation(seed: int, n_train: int, target_epoch: int) -> np.ndarray:
    """精確重算 train_conv_compressed.run_epochs 用的 shuffle 順序(純 PRNG key
    分裂,跟浮點非決定性無關)。`shuffle_key` 從 `PRNGKey(seed+1)` 開始,逐
    epoch split,要從 epoch 0 依序重放到 target_epoch,不能跳著算。"""
    shuffle_key = jax.random.PRNGKey(seed + 1)
    perm = None
    for _epoch in range(target_epoch + 1):
        shuffle_key, subkey = jax.random.split(shuffle_key)
        perm = jax.random.permutation(subkey, n_train)
    return np.asarray(perm)


def per_sample_forward(run_record: dict, layers: list, params: tuple, split,
                       batch_size: int = 20):
    """對整個 train split 分批 forward(分批純粹省記憶體,跟訓練 batch_size
    無關),回傳每筆樣本的 loss、預測類別、每層 n_out_spikes(shape 皆
    `(n_samples,)`/`{層名: (n_samples,)}`)。"""
    net = ConvNetCompressed(layers)
    decoder = build_decoder(run_record["config"]["model"], layers)
    n = split.labels.shape[0]

    @jax.jit
    def _fwd(params, event_times, x, y, c, n_real):
        result, diags = net.apply_batched(params, event_times, x, y, c, n_real)
        scores, _ = jax.vmap(decoder.decode)(result)
        return scores, diags

    all_loss, all_pred = [], []
    all_spikes = {layer.name: [] for layer in layers}
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        scores, diags = _fwd(params, split.event_times[start:end], split.x[start:end],
                             split.y[start:end], split.c[start:end],
                             split.n_real_events[start:end])
        loss = optax.softmax_cross_entropy(scores, split.labels_onehot[start:end])
        all_loss.append(np.asarray(loss))
        all_pred.append(np.asarray(jnp.argmax(scores, axis=1)))
        for layer, d in zip(layers, diags):
            all_spikes[layer.name].append(np.asarray(d.n_out_spikes))

    loss = np.concatenate(all_loss)
    pred = np.concatenate(all_pred)
    spikes = {name: np.concatenate(v) for name, v in all_spikes.items()}
    return loss, pred, spikes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("exp_dir_name", help="experiments/ 底下的目錄名稱(不含路徑)")
    parser.add_argument("--snapshot-epoch", type=int, required=True)
    parser.add_argument("--target-epoch", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    exp_dir = os.path.join(EXPERIMENTS_DIR, args.exp_dir_name)
    run_record, layers, params = _load_layers_and_params(exp_dir, args.snapshot_epoch)
    train_cfg = run_record["config"]["train"]
    split = _rebuild_train_split(run_record)
    n_train = split.labels.shape[0]

    print(f"epoch{args.snapshot_epoch} 快照,對整個 train split({n_train} 筆)做 forward...")
    loss, pred, spikes = per_sample_forward(run_record, layers, params, split)
    correct = (pred == np.asarray(split.labels))

    print(f"\n=== train split 在 epoch{args.snapshot_epoch} 快照下的 loss 分布 ===")
    print(f"  mean={loss.mean():.6f}  std={loss.std():.6f}  max={loss.max():.6f}  "
          f"accuracy={correct.mean():.4f}")

    order = np.argsort(-loss)[:args.top_k]
    print(f"\n=== loss 最大的 {args.top_k} 筆樣本(離群候選)===")
    header = f"{'idx':>6} {'loss':>10} {'correct':>8} {'n_real':>7} "
    header += " ".join(f"{name + '_spk':>10}" for name in spikes)
    print(header)
    for i in order:
        row = f"{int(i):>6} {loss[i]:>10.6f} {str(bool(correct[i])):>8} " \
              f"{int(split.n_real_events[i]):>7} "
        row += " ".join(f"{int(spikes[name][i]):>10}" for name in spikes)
        print(row)

    n_real = np.asarray(split.n_real_events)
    print(f"\n=== n_real_events 分布 ===")
    print(f"  mean={n_real.mean():.1f}  std={n_real.std():.1f}  max={n_real.max()}")

    print(f"\n=== 重建 epoch{args.target_epoch} 的 batch 切法,找離群樣本落在哪個 batch ===")
    perm = _epoch_permutation(train_cfg["seed"], n_train, args.target_epoch)
    batch_size = min(train_cfg["batch_size"], n_train)
    n_batches = n_train // batch_size

    outlier_idx = set(int(i) for i in order)
    hit_any = False
    for b in range(n_batches):
        idx = perm[b * batch_size:(b + 1) * batch_size]
        hit = outlier_idx.intersection(idx.tolist())
        if hit:
            hit_any = True
            print(f"  batch {b}: 樣本 {idx.tolist()},命中離群樣本 {sorted(hit)}")
    if not hit_any:
        print(f"  沒有任何 batch 命中前 {args.top_k} 個離群樣本")


if __name__ == "__main__":
    main()
