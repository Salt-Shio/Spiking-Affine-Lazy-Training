"""ConvNetCompressed 訓練 entrypoint。

這支腳本比一般訓練多的東西,是所有壓縮容量旋鈕的**動態放大**機制(完整設計
見 docs/math/conv事件佇列壓縮版推導.md 第 7.2 節):

- 每個 conv 層有三個會出界的容量:壓縮佇列長度 `L`、輸出 spike 上界
  `max_out_spikes`、掃描步數上界 `max_steps`。三個出界訊號都在 forward 裡算
  出來、由 `LayerDiag` 帶出。
- 沒有「靜態精算一次永久有效」的做法(原本 `conv_param_search.py` 的 L1 靜態
  量測太慢已廢),一律:config 給起始猜測 → 訓練中某個 batch 偵測出界 →
  該層 `grown_to_fit` 放大 → 退回最近的 checkpoint → 用新的 layer list 重
  編譯續練。哪個旋鈕、放大多少是 `salt_core.layers.ConvLayer` 自己的知識,
  這裡只負責「作廢這個 batch、退 checkpoint、重編譯」這圈訓練編排,而且是對
  layer list 的一個**通用迴圈**,不寫死層名。
- `max_steps` 多一條**選擇性縮小**的路(`_reestimate_max_steps`,數學推導見
  docs/math/掃描步數上界推導.md):每 `train.max_steps_reestimate_every` 個
  epoch(預設 1)開始、batch 迴圈之前,用當下權重對固定探測批重估一次,比
  現有值小就縮、重編譯續跑。這條路跟出界不一樣,不需要退 checkpoint(用的是
  已經確定沒問題的當下權重),但重編譯這件事借用同一條 while 外圈。

用法(config 路徑相對於 repo 根目錄,或給絕對路徑):
  python -m example.train_conv_compressed configs/conv/compressed_baseline.yaml
"""
import argparse
import datetime
import os
import sys
from typing import NamedTuple

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# 訓練跑好幾小時,背景執行/導向檔案時 Python 預設整批緩衝,中途完全看不到進度。
sys.stdout.reconfigure(line_buffering=True)

import jax
import jax.numpy as jnp
import optax
import yaml

from data.src.nmnist import NMNISTDataset
from salt_core.layers import ConvLayer, raw_events_to_stream, run_network
from example.checkpoint import Checkpointer
from salt_core.dormant import dormant_report
from example.metrics_log import MetricsLog
from example.models.conv_net import ConvNetCompressed, build_decoder, build_network
from example.trace_probe import TraceProbe
from example.paths import DATASET_ROOT, EXPERIMENTS_DIR, REPO_ROOT, resolve_config
from example.utils import (TRACES_DIRNAME, TRAIN_DIRNAME, get_git_commit_hash,
                           make_evaluate, save_params_npz, set_seed)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_train_step(net, optimizer, decoder):
    def loss_fn(params, batch_event_times, batch_x, batch_y, batch_c, batch_n_real,
                batch_labels_onehot):
        result, diagnostics = net.apply_batched(params, batch_event_times, batch_x, batch_y,
                                                 batch_c, batch_n_real)
        # 解碼器把最後一層的 LayerForwardResult 讀成分數(膜電位回歸 = v_final、
        # 頻率/群體 = s_value 加總),網路本身不挑 readout。dec_metrics 是這個
        # 編碼特定的純量(可空,膜電位回歸就是空的)。
        scores, dec_metrics = jax.vmap(decoder.decode)(result)
        per_sample_loss = optax.softmax_cross_entropy(scores, batch_labels_onehot)
        loss = jnp.mean(per_sample_loss)
        # 每層一份「批次縮減後的 LayerDiag」:firing_rate / spike_count 取批次
        # **平均**(給 metrics.csv 當哨兵指標),max_real_queue / n_out_spikes /
        # min_steps_needed 取批次**最大**(出界偵測:只要批次裡任何一筆樣本、
        # 任何一顆神經元超過目前容量就算出界,不能被其他樣本的小值平均掉)。
        # grown_to_fit 只看後三個欄位,logging 只看前兩個。
        reduced = [d._replace(
            spike_count=jnp.mean(d.spike_count),
            firing_rate=jnp.mean(d.firing_rate),
            max_real_queue=jnp.max(d.max_real_queue),
            n_out_spikes=jnp.max(d.n_out_spikes),
            min_steps_needed=jnp.max(d.min_steps_needed)) for d in diagnostics]
        reduced_metrics = {k: jnp.mean(v) for k, v in dec_metrics.items()}
        return loss, (reduced, reduced_metrics)

    @jax.jit
    def train_step(params, opt_state, batch_event_times, batch_x, batch_y, batch_c,
                    batch_n_real, batch_labels_onehot):
        (loss, (reduced_diags, reduced_metrics)), grad = jax.value_and_grad(
            loss_fn, has_aux=True)(
            params, batch_event_times, batch_x, batch_y, batch_c, batch_n_real,
            batch_labels_onehot)
        # grad 是對齊 net.layers 的 tuple(pytree),按層名報範數。
        grad_norms = {layer.name: jnp.linalg.norm(g)
                      for layer, g in zip(net.layers, grad)}
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, reduced_diags, reduced_metrics, grad_norms

    return train_step


def _grow_layers(layers: list, reduced_diags: list) -> list:
    """對每一層問一次 `grown_to_fit`,回傳新的 layer list(沒出界的層原封不動,
    是同一個物件)。"""
    return [layer.grown_to_fit(diag) for layer, diag in zip(layers, reduced_diags)]


def _describe_growth(old_layers: list, new_layers: list, reduced_diags: list) -> str:
    """把哪些層的哪些容量旋鈕從多少放大到多少(附這個 batch 觀察到的真實值),
    組成一行給 log/測試解析。格式:
    `conv2 L 32->2100(觀察 1401); conv2 max_out 45000->68000(觀察 46500)`。
    """
    parts = []
    for old, new, d in zip(old_layers, new_layers, reduced_diags):
        if old is new or not isinstance(old, ConvLayer):
            continue
        if new.L != old.L:
            parts.append(f"{old.name} L {old.L}->{new.L}(觀察 {int(d.max_real_queue)})")
        if new.max_out_spikes != old.max_out_spikes:
            parts.append(f"{old.name} max_out {old.max_out_spikes}->{new.max_out_spikes}"
                          f"(觀察 {int(d.n_out_spikes)})")
        if new.max_steps != old.max_steps:
            parts.append(f"{old.name} max_steps {old.max_steps}->{new.max_steps}"
                          f"(觀察 {int(d.min_steps_needed)})")
    return "; ".join(parts)


def _reestimate_max_steps(layers: list, params, probe_batch, *, chunk: int = 16) -> list:
    """`max_steps` 的選擇性縮小重估(見 docs/規格書.md「conv 層 max_steps」、
    docs/math/掃描步數上界推導.md)。對固定探測批(跟 `dormant_report` 同一批、
    同樣的 chunk 化 vmap 避免 FC 密集佇列 OOM)用當下權重跑一般 `run_network`,
    取每個 conv 層這批樣本裡 `LayerDiag.min_steps_needed` 的最大值,交給
    `shrink_max_steps` 決定要不要縮。不是每個 batch 都做,呼叫端(`run_epochs`)
    決定頻率。"""
    et, x, y, c, nr = probe_batch
    first = layers[0]
    n = int(et.shape[0])

    @jax.jit
    def chunk_diags(p, e, xx, yy, cc, rr):
        streams = jax.vmap(raw_events_to_stream, in_axes=(0, 0, 0, 0, 0, None, None))(
            e, xx, yy, cc, rr, first.h_in, first.w_in)
        return jax.vmap(lambda s: run_network(layers, s, p)[1])(streams)

    maxes: list[int] | None = None
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        diags = chunk_diags(params, et[lo:hi], x[lo:hi], y[lo:hi], c[lo:hi], nr[lo:hi])
        batch_max = [int(jnp.max(d.min_steps_needed)) for d in diags]
        maxes = batch_max if maxes is None else [max(a, b) for a, b in zip(maxes, batch_max)]

    return [layer.shrink_max_steps(cand) for layer, cand in zip(layers, maxes)]


def _describe_shrink(old_layers: list, new_layers: list) -> str:
    """跟 `_describe_growth` 對應,格式:`conv1 max_steps 200->134`。"""
    parts = [f"{old.name} max_steps {old.max_steps}->{new.max_steps}"
             for old, new in zip(old_layers, new_layers) if old is not new]
    return "; ".join(parts)


class Best(NamedTuple):
    """跨 run_epochs 呼叫累積的「val_accuracy 最好的那個 epoch」。"""
    params: object
    val_accuracy: float
    epoch: int


class EpochsOutcome(NamedTuple):
    overflowed: bool
    grown_layers: list        # 出界 / 重估縮小 = 新 list;都沒有 = 原樣傳回
    final_params: object       # 最後一個完成 epoch 的 params(出界/重估時呼叫端不用)
    best: Best
    reestimated: bool = False  # max_steps 選擇性縮小觸發的重編譯,不是出界(見
                               # _reestimate_max_steps);跟 overflowed 分開記,
                               # 因為成因、要不要當「有問題」看待完全不同,不能
                               # 共用同一個欄位混在一起。


def run_epochs(*, layers, train_step, evaluate, params, opt_state,
               shuffle_key, start_epoch: int, total_epochs: int,
               train_split, val_split, batch_size: int, probe_batch,
               metrics_log, checkpointer, best: Best,
               trace_probe: "TraceProbe | None" = None,
               max_steps_reestimate_every: int = 1) -> EpochsOutcome:
    """跑 `[start_epoch, total_epochs)` 的訓練迴圈。

    **不知道「長大」這回事**:偵測到某 batch 的真實用量超過壓縮容量,就用
    `_grow_layers` 算出放大後的新 layer list、印 `[出界]`、回傳
    `EpochsOutcome(overflowed=True, grown_layers=...)`——不自己退 checkpoint /
    重編譯,那是呼叫端 `train()` 的 while 外圈。跑完整段沒出界回
    `overflowed=False`,`final_params` 是最後一個 epoch 的權重。

    **也不知道「縮小」這回事**:`max_steps_reestimate_every > 0` 時,每 N 個
    epoch 開始、batch 迴圈之前,用當下權重對探測批重估一次 `max_steps`
    (`_reestimate_max_steps`,見 docs/規格書.md「conv 層 max_steps」)。真的縮了
    就印 `[縮小]`、回傳 `EpochsOutcome(reestimated=True, grown_layers=...)`,
    一樣交給 `train()` 的 while 外圈重編譯續跑——用的是已經確定沒問題的當下
    權重,不是修正錯誤,跟出界的處理理由不同,但重編譯這件事借用同一條路。

    每個成功 epoch:val 評估 → 更新 `best` → `metrics_log.finish_epoch` →
    `checkpointer.save`。
    """
    n_train = train_split.labels.shape[0]
    n_batches = max(1, n_train // batch_size)

    for epoch in range(start_epoch, total_epochs):
        if max_steps_reestimate_every > 0 and epoch % max_steps_reestimate_every == 0:
            reestimated_layers = _reestimate_max_steps(layers, params, probe_batch)
            if reestimated_layers != layers:
                print(f"[縮小] epoch={epoch}: {_describe_shrink(layers, reestimated_layers)}")
                return EpochsOutcome(overflowed=False, grown_layers=reestimated_layers,
                                      final_params=params, best=best, reestimated=True)

        shuffle_key, subkey = jax.random.split(shuffle_key)
        perm = jax.random.permutation(subkey, n_train)
        metrics_log.start_epoch()

        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            (new_params, new_opt_state, loss, reduced_diags, reduced_metrics,
             grad_norms) = train_step(
                params, opt_state, train_split.event_times[idx], train_split.x[idx],
                train_split.y[idx], train_split.c[idx],
                train_split.n_real_events[idx], train_split.labels_onehot[idx])

            # 出界偵測:任何一層要長大,就作廢這個 batch、回報給外圈。
            grown = _grow_layers(layers, reduced_diags)
            if grown != layers:
                where = (f"退回 checkpoint(epoch={checkpointer.last_epoch})"
                         if checkpointer.exists() else "還沒有 checkpoint,退回訓練最初始狀態")
                print(f"[出界] epoch={epoch} batch={b}: "
                      f"{_describe_growth(layers, grown, reduced_diags)},{where}")
                return EpochsOutcome(overflowed=True, grown_layers=grown,
                                      final_params=params, best=best)

            params, opt_state = new_params, new_opt_state
            metrics_log.record_batch(loss=loss, layers=layers,
                                      reduced_diags=reduced_diags,
                                      grad_norms=grad_norms,
                                      decoder_metrics=reduced_metrics)

        val_accuracy, _val_loss, _ = evaluate(params, val_split)
        if val_accuracy > best.val_accuracy:
            best = Best(params=params, val_accuracy=val_accuracy, epoch=epoch)
        dormant = dormant_report(layers, params, probe_batch)
        metrics_log.finish_epoch(epoch=epoch, val_accuracy=val_accuracy, layers=layers,
                                  dormant=dormant)
        checkpointer.save(params=params, opt_state=opt_state,
                          shuffle_key=shuffle_key, epoch=epoch)

        # 逐步軌跡探測(forward-only、另一個編譯目標,不影響上面的訓練熱路徑)。
        if trace_probe is not None and trace_probe.due(epoch):
            trace_probe.run(layers, params, epoch)

    return EpochsOutcome(overflowed=False, grown_layers=layers,
                          final_params=params, best=best)


def _make_exp_dir(run_name: str, exp_root=EXPERIMENTS_DIR) -> str:
    """訓練「開始前」就建好目錄——checkpoint 要在訓練過程中(每個 epoch 結束)
    持續寫進同一個目錄。`exp_root` 預設是 `experiments/`;e2e 測試傳
    `experiments/TEST_TEMP` 進來,把測試產物跟正式 run 隔開。

    只在這裡先建 `train/`(訓練產物每個 epoch 都要寫,是唯一保證一定會用到
    的子資料夾)。`traces/`(`TraceProbe`)、`eval/`(`eval_test.py`)是條件式
    的,各自的消費者第一次要寫的時候自己建。"""
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(exp_root, f"conv_compressed_{run_name}_{date_str}")
    os.makedirs(os.path.join(exp_dir, TRAIN_DIRNAME), exist_ok=True)
    return exp_dir


def train(config_path: str, exp_root=EXPERIMENTS_DIR):
    # config 只在這裡解析一次:三個區塊各自綁好,之後 train() 只碰這三個
    # (跟 run_name),不再出現 raw_cfg[...]。raw_cfg 只留著當「輸入快照」放進
    # run 紀錄,train() 不改它。
    raw_cfg = load_config(config_path)
    model_cfg = raw_cfg["model"]
    data_cfg = raw_cfg["data"]
    train_cfg = raw_cfg["train"]
    run_name = raw_cfg.get("run_name", "run")

    dataset = NMNISTDataset(DATASET_ROOT, max_events=data_cfg["max_events"])
    train_split = dataset.build_split(seed=data_cfg["seed_train"],
                                      n_samples=data_cfg["train_size"], which="train")
    val_split = dataset.build_split(seed=data_cfg["seed_val"],
                                    n_samples=data_cfg["val_size"], which="val")

    # 一列 layer 物件,形狀完全由 model_cfg["layers"] 決定(見
    # example.models.conv_net.build_network)。動態放大 = 用 grown_to_fit 重建這個
    # list,跨 while 迴圈迭代持續累積(層名不變)。init_k 是每層必填欄位(不校準,
    # 委定值見 docs/問題紀錄.md §12),config 沒填會在 build_network 這一步就報錯。
    layers = build_network(model_cfg)

    layer_names = [layer.name for layer in layers]
    conv_names = [layer.name for layer in layers if isinstance(layer, ConvLayer)]

    # 輸出編碼:把最後一層的 LayerForwardResult 讀成分數。跟最後一層的門檻
    # 設定配套(膜電位回歸要 v_th 超大、頻率/群體要正常門檻),validate 擋
    # 掉配錯 → 靜默算垃圾。動態放大重建 layer list 不影響 decoder(它不看
    # 容量旋鈕),所以在 while 迴圈之前建一次就好。
    decoder = build_decoder(model_cfg, layers)
    decoder.validate(layers[-1])

    batch_size = min(train_cfg["batch_size"], data_cfg["train_size"])

    # dormant score(salt_core/dormant.py)每個 epoch 在這批固定樣本上量,跨 epoch 可比。
    n_probe = min(128, data_cfg["train_size"])
    probe_batch = (train_split.event_times[:n_probe], train_split.x[:n_probe],
                   train_split.y[:n_probe], train_split.c[:n_probe],
                   train_split.n_real_events[:n_probe])

    exp_dir = _make_exp_dir(run_name, exp_root)

    # 逐步軌跡探測(docs/監測規格.md §6/§7):train.probe_every > 0 才開,對固定的
    # 前 K 筆 train 樣本週期性跑 run_network_traced,dump experiments/<run>/traces/。
    trace_probe = None
    probe_every = int(train_cfg.get("probe_every", 0))
    if probe_every > 0:
        k_trace = min(int(train_cfg.get("probe_samples", 8)), data_cfg["train_size"])
        trace_probe = TraceProbe(
            traces_dir=os.path.join(exp_dir, TRACES_DIRNAME),
            probe_batch=(train_split.event_times[:k_trace], train_split.x[:k_trace],
                         train_split.y[:k_trace], train_split.c[:k_trace],
                         train_split.n_real_events[:k_trace]),
            every=probe_every, total_epochs=train_cfg["epochs"],
            full_every=int(train_cfg.get("probe_full_every", 0)),
            full_samples=int(train_cfg.get("probe_full_samples", 2)))
    # checkpoint 每個 epoch 覆蓋寫一份最新的;出界就退回它重編譯續練。
    # exp_dir 每次新目錄,所以 checkpointer.exists() 等價於「這次 run 存過沒」:
    # 沒存過就出界 -> 退回訓練最初始狀態(同一顆 seed 重新 init)。
    checkpointer = Checkpointer(os.path.join(exp_dir, TRAIN_DIRNAME, "checkpoint.npz"))

    metrics_log = MetricsLog(layer_names, conv_names, total_epochs=train_cfg["epochs"])

    # while 外圈 = 自動長大控制系統:setup params(fresh / checkpoint)→ 跑
    # run_epochs → 沒出界就結束、出界就換成放大後的 layers 重編譯再繞。
    # 「跑 epoch」本身完全在 run_epochs 裡,不知道長大這回事。
    best = Best(params=None, val_accuracy=-1.0, epoch=-1)
    while True:
        net = ConvNetCompressed(layers)
        optimizer = optax.adam(train_cfg["lr"])

        if not checkpointer.exists():
            params = net.init(set_seed(train_cfg["seed"]))
            opt_state = optimizer.init(params)
            shuffle_key = jax.random.PRNGKey(train_cfg["seed"] + 1)
            start_epoch = 0
            if best.params is None:  # 一個 epoch 都還沒完成過的 fallback
                best = best._replace(params=params)
        else:
            template_params = net.init(jax.random.PRNGKey(0))
            params, opt_state, shuffle_key, ckpt_epoch = checkpointer.load(
                params_template=template_params,
                opt_state_template=optimizer.init(template_params))
            start_epoch = ckpt_epoch + 1

        outcome = run_epochs(
            layers=layers,
            train_step=make_train_step(net, optimizer, decoder),
            evaluate=make_evaluate(net, decoder, eval_batch_size=batch_size),
            params=params, opt_state=opt_state, shuffle_key=shuffle_key,
            start_epoch=start_epoch, total_epochs=train_cfg["epochs"],
            train_split=train_split, val_split=val_split, batch_size=batch_size,
            probe_batch=probe_batch,
            metrics_log=metrics_log, checkpointer=checkpointer, best=best,
            trace_probe=trace_probe,
            max_steps_reestimate_every=int(train_cfg.get("max_steps_reestimate_every", 1)))

        best = outcome.best
        if not outcome.overflowed and not outcome.reestimated:
            params = outcome.final_params
            break
        layers = outcome.grown_layers

    best_params, best_val_accuracy, best_epoch = best
    rows = metrics_log.rows
    # 一份 write-once 的 run 紀錄:輸入 config 快照 + commit + 最終容量 + 最佳
    # 指標。輸入(raw_cfg)不被改;結果不塞回它。
    run_record = {
        "config": raw_cfg,
        "git_commit": get_git_commit_hash(str(REPO_ROOT)),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "best": {"val_accuracy": best_val_accuracy, "epoch": best_epoch},
        "final_capacity": {
            layer.name: {"L": layer.L, "max_out_spikes": layer.max_out_spikes,
                        "max_steps": layer.max_steps}
            for layer in layers if isinstance(layer, ConvLayer)},
        "last_epoch_obs": ({
            name: {"queue": rows[-1][f"{name}_obs_queue"],
                   "out": rows[-1][f"{name}_obs_out"]}
            for name in conv_names} if rows else {}),
    }
    metrics_log.write_csv(os.path.join(exp_dir, TRAIN_DIRNAME, "metrics.csv"))
    _write_experiment(run_record, params, best_params, exp_dir, layers)
    metrics_log.print_summary(layers)
    return exp_dir, net, params, train_split, val_split, run_record


def _write_experiment(run_record: dict, params, best_params, exp_dir: str,
                       layers: list) -> None:
    """把 run 紀錄 + 權重寫進 exp_dir。metrics.csv 跟結尾的逐層用量摘要由
    MetricsLog 負責(見 example/metrics_log.py)。"""
    with open(os.path.join(exp_dir, TRAIN_DIRNAME, "run.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(run_record, f, allow_unicode=True, sort_keys=False)

    save_params_npz(os.path.join(exp_dir, TRAIN_DIRNAME, "params.npz"), layers, params)
    save_params_npz(os.path.join(exp_dir, TRAIN_DIRNAME, "best_params.npz"), layers, best_params)

    best = run_record["best"]
    print(f"訓練結束,結果存到 {exp_dir}")
    print(f"  best val_accuracy = {best['val_accuracy']:.4f} @ epoch {best['epoch']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="yaml config 檔案路徑(相對於 repo 根目錄,或絕對路徑)")
    args = parser.parse_args()
    train(str(resolve_config(args.config)))
