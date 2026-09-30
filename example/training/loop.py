"""epoch 迴圈跟容量變化時的重來。

run_epochs 跑到跑完、或容量要改為止;run_training 是外圈:容量改了就換新的一列層,
從最後一個跑完的 epoch 接著練(出界的 batch 跟那個 epoch 已套用的更新都丟掉)。
"""
from dataclasses import dataclass
from typing import NamedTuple

import jax
import numpy as np

from example.checkpoint import Best, Checkpointer, CheckpointState
from example.dormant import dormant_report
from example.metrics_log import MetricsLog
from example.training.capacity_control import CapacityControl
from example.training.step import make_train_step
from example.utils import make_evaluate, set_seed, take_input_events, weight_snapshot_path
from salt_core.io import save_weights
from salt_core.network import InputEvents, Network


class TrainData(NamedTuple):
    """訓練跟驗證資料。每個 split 要有 event_times、x、y、c、n_real_events、labels、labels_onehot。"""
    train: object
    val: object


class TrainState(NamedTuple):
    """epoch 邊界上的訓練狀態。next_epoch:下一個要跑的 epoch。"""
    params: tuple
    opt_state: object
    shuffle_key: jax.Array
    next_epoch: int
    best: Best


@dataclass(frozen=True)
class RunContext:
    """一次 run 裡不變的東西。

    train_raw: train split 轉成的 InputEvents;probe: dormant 統計用的固定樣本。
    snapshot_every: 每幾個 epoch 存一份權重快照到 snapshot_dir,0 不存。
    """
    data: TrainData
    train_raw: InputEvents
    probe: InputEvents
    batch_size: int
    epochs: int
    seed: int
    optimizer: object
    decoder: object
    score_cap: float | None
    dormant_names: tuple
    capacity: CapacityControl
    metrics_log: MetricsLog
    checkpointer: Checkpointer
    snapshot_dir: str
    snapshot_every: int


class EpochsOutcome(NamedTuple):
    """state:最後一個跑完的 epoch 之後的狀態。new_layers:容量要改時的新一列層,跑完時是 None。"""
    state: TrainState
    new_layers: list | None


def initial_shuffle_key(seed: int) -> jax.Array:
    return jax.random.PRNGKey(seed + 1)


def shuffle_epoch(shuffle_key: jax.Array, n_train: int) -> tuple[jax.Array, jax.Array]:
    """一個 epoch 的樣本順序。回傳 (下一個 epoch 用的 key, 排列)。"""
    shuffle_key, subkey = jax.random.split(shuffle_key)
    return shuffle_key, jax.random.permutation(subkey, n_train)


def epoch_permutation(seed: int, n_train: int, epoch: int) -> np.ndarray:
    """訓練 seed 為 seed 時,第 epoch 個 epoch 的樣本順序。每個 batch 依序取 batch_size 筆,
    湊不滿一個 batch 的尾巴不用。"""
    shuffle_key = initial_shuffle_key(seed)
    for _ in range(epoch + 1):
        shuffle_key, perm = shuffle_epoch(shuffle_key, n_train)
    return np.asarray(perm)


def initial_state(ctx: RunContext, network: Network) -> TrainState:
    """從頭訓練的狀態:ctx.seed 初始化權重。"""
    params = network.init(set_seed(ctx.seed))
    return TrainState(params=params, opt_state=ctx.optimizer.init(params),
                      shuffle_key=initial_shuffle_key(ctx.seed), next_epoch=0,
                      best=Best(params=params, network=network, val_accuracy=-1.0, epoch=-1))


def state_from_checkpoint(saved: CheckpointState) -> TrainState:
    return TrainState(params=saved.params, opt_state=saved.opt_state, shuffle_key=saved.shuffle_key,
                      next_epoch=saved.epoch + 1, best=saved.best)


def _history(ctx: RunContext) -> dict:
    """checkpoint 裡的紀錄:已完成 epoch 的 metrics 列、到目前為止的容量事件。"""
    return {"metrics_rows": ctx.metrics_log.rows, "capacity_events": ctx.capacity.events}


def _finish_epoch(ctx: RunContext, network: Network, layers: list, epoch: int,
                  params, evaluate, best: Best) -> Best:
    """epoch 跑完:val 評估、dormant、記指標、存 checkpoint 跟權重快照。回傳更新過的 best。"""
    val_accuracy, _val_loss, _, val_regrows = evaluate(params, ctx.data.val)
    if val_accuracy > best.val_accuracy:
        best = Best(params=params, network=network, val_accuracy=val_accuracy, epoch=epoch)
    dormant, dormant_regrows = {}, 0
    if ctx.dormant_names:
        dormant, dormant_regrows = dormant_report(network, params, ctx.probe,
                                                  ctx.capacity.policies,
                                                  layer_names=ctx.dormant_names)
    if dormant_regrows:
        print(f"[dormant 出界] epoch={epoch}: 放大探測用容量重算 {dormant_regrows} 次")
    ctx.metrics_log.finish_epoch(epoch=epoch, val_accuracy=val_accuracy, layers=layers,
                                 needed=ctx.capacity.epoch_needed, dormant=dormant,
                                 val_capacity_regrows=val_regrows,
                                 dormant_capacity_regrows=dormant_regrows)
    return best


def run_epochs(ctx: RunContext, network: Network, state: TrainState) -> EpochsOutcome:
    """從 state.next_epoch 跑到 ctx.epochs,或到容量要改為止。

    某個 batch 放不下:丟掉這個 batch,回傳這個 epoch 開始時的狀態跟放大後的層。
    epoch 跑完後照這個 epoch 的需求縮小:回傳跑完的狀態跟縮小後的層。
    checkpoint 存的是下一個 epoch 要用的網路(縮小之後);權重快照存這個 epoch 用的網路。
    """
    layers = list(network.layers)
    train_step = make_train_step(network, ctx.optimizer, ctx.decoder, ctx.score_cap)
    evaluate = make_evaluate(network, ctx.decoder, eval_batch_size=ctx.batch_size,
                             policies=ctx.capacity.policies)
    train_split = ctx.data.train
    n_train = train_split.labels.shape[0]
    n_batches = max(1, n_train // ctx.batch_size)
    params, opt_state, shuffle_key, best = (state.params, state.opt_state, state.shuffle_key,
                                            state.best)

    for epoch in range(state.next_epoch, ctx.epochs):
        shuffle_key, perm = shuffle_epoch(shuffle_key, n_train)
        ctx.metrics_log.start_epoch()
        ctx.capacity.start_epoch(layers)

        for b in range(n_batches):
            idx = perm[b * ctx.batch_size:(b + 1) * ctx.batch_size]
            out = train_step(params, opt_state, take_input_events(ctx.train_raw, idx),
                             train_split.labels_onehot[idx])
            if not bool(out.fits):
                resumed_from = ctx.checkpointer.last_epoch if ctx.checkpointer.exists() else None
                grown, event = ctx.capacity.grow(layers, out.diags, epoch=epoch, batch=b,
                                                 resumed_from_epoch=resumed_from)
                for line in event.describe():
                    print(line)
                return EpochsOutcome(state=state, new_layers=grown)

            params, opt_state = out.params, out.opt_state
            ctx.capacity.record_batch(layers, out.diags)
            ctx.metrics_log.record_batch(loss=out.loss, layers=layers, reduced_diags=out.diags,
                                         grad_norms=out.grad_norms,
                                         decoder_metrics=out.decoder_metrics)

        best = _finish_epoch(ctx, network, layers, epoch, params, evaluate, best)
        shrunk, shrink_event = ctx.capacity.shrink(layers, epoch)
        ctx.checkpointer.save(CheckpointState(
            network=network.replace_layers(shrunk), params=params, opt_state=opt_state,
            shuffle_key=shuffle_key, epoch=epoch, best=best, history=_history(ctx)))
        if ctx.snapshot_every > 0 and epoch % ctx.snapshot_every == 0:
            save_weights(weight_snapshot_path(ctx.snapshot_dir, epoch), network, params)
        state = TrainState(params=params, opt_state=opt_state, shuffle_key=shuffle_key,
                           next_epoch=epoch + 1, best=best)

        if shrink_event is not None:
            for line in shrink_event.describe():
                print(line)
            return EpochsOutcome(state=state, new_layers=shrunk)

    return EpochsOutcome(state=state, new_layers=None)


def run_training(ctx: RunContext, network: Network, state: TrainState) -> tuple[Network, TrainState]:
    """從 state 跑到 ctx.epochs 跑完;容量改了就換新的層、從 outcome.state 接著練。
    回傳 (結束時的網路, 結束時的狀態)。"""
    while True:
        outcome = run_epochs(ctx, network, state)
        state = outcome.state
        if outcome.new_layers is None:
            return network, state
        network = network.replace_layers(outcome.new_layers)
