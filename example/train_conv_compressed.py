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
- `max_steps` 跟 `max_out_spikes` 都多一條**選擇性縮小**的路
  (`_shrink_layers`):每 `train.max_steps_reestimate_every` 個**成功跑完的**
  epoch,用這個 epoch 裡所有真實 batch 觀察到的最大值決定要不要縮,比現有值
  小就縮、重編譯續跑。這條路跟出界不一樣,不需要退 checkpoint(用的是已經
  確定沒問題的當下權重),但重編譯這件事借用同一條 while 外圈。`max_steps`
  用的觀察值是 `LayerDiag.min_steps_needed`(理論上界,數學推導見
  docs/math/掃描步數上界推導.md,因為 `max_steps` 沒有天然的經驗值可用——設
  太小是掃描提早停止、靜默算錯,不像 `L`/`max_out_spikes` 是真實資料裝不下
  的被動事實);`max_out_spikes` 用的是 `LayerDiag.n_out_spikes`(真實觀察值,
  跟長大訊號同一個量,理由見 docs/問題紀錄.md 第十四節——曾經嘗試過讓
  `max_out_spikes` 也用理論上界,實測太鬆、已撤回)。長大/縮小的目標值都用
  同一個公式算(`ceil(觀察值 * factor)`,`max_steps` 用
  `max_steps_grow_factor`、`max_out_spikes` 用 `out_grow_factor`),真實需求
  沒變時兩次算出來的目標值會相等,天然不會震盪;縮小還要另外掉到
  `max_steps_shrink_threshold`/`out_shrink_threshold` 比例以下才值得觸發
  (效率門檻,不影響防震盪)。

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
from salt_core.layers import ConvLayer
from example.checkpoint import Checkpointer
from salt_core.dormant import dormant_report
from example.metrics_log import MetricsLog
from example.models.conv_net import ConvNetCompressed, build_decoder, build_network
from example.paths import DATASET_ROOT, EXPERIMENTS_DIR, REPO_ROOT, resolve_config
from example.utils import (TRAIN_DIRNAME, WEIGHTS_DIRNAME, get_git_commit_hash,
                           make_evaluate, save_params_npz, set_seed,
                           weight_snapshot_path)


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


def _describe_growth(old_layers: list, new_layers: list, reduced_diags: list) -> list[str]:
    """哪些層的哪些容量旋鈕從多少放大到多少(附這個 batch 觀察到的真實值),
    一個旋鈕一行,給呼叫端縮排印出來/測試解析。格式:
    `conv2 L 32->2100(觀察 1401)`。"""
    lines = []
    for old, new, d in zip(old_layers, new_layers, reduced_diags):
        if old is new or not isinstance(old, ConvLayer):
            continue
        if new.L != old.L:
            lines.append(f"{old.name} L {old.L}->{new.L}(觀察 {int(d.max_real_queue)})")
        if new.max_out_spikes != old.max_out_spikes:
            lines.append(f"{old.name} max_out {old.max_out_spikes}->{new.max_out_spikes}"
                          f"(觀察 {int(d.n_out_spikes)})")
        if new.max_steps != old.max_steps:
            lines.append(f"{old.name} max_steps {old.max_steps}->{new.max_steps}"
                          f"(觀察 {int(d.min_steps_needed)})")
    return lines


def _shrink_layers(layers: list, epoch_min_steps_needed: dict,
                    epoch_n_out_spikes: dict) -> list:
    """對每一層問一次 `shrink_max_steps` + `shrink_max_out_spikes`,回傳新的
    layer list(沒縮的層原封不動,是同一個物件)。`epoch_min_steps_needed`/
    `epoch_n_out_spikes`:層名 -> 這個**成功跑完的 epoch**裡,所有真實 batch
    的 `LayerDiag.min_steps_needed`/`n_out_spikes` 觀察最大值——不是探測批,
    是這個 epoch 真正跑過的訓練資料(見 docs/規格書.md「conv 層 max_steps」,
    `max_out_spikes` 比照辦理)。"""
    result = []
    for layer in layers:
        shrunk = layer.shrink_max_steps(epoch_min_steps_needed[layer.name])
        shrunk = shrunk.shrink_max_out_spikes(epoch_n_out_spikes[layer.name])
        result.append(shrunk)
    return result


def _describe_shrink(old_layers: list, new_layers: list) -> list[str]:
    """跟 `_describe_growth` 對應,一個旋鈕一行,格式:`conv1 max_steps
    200->134`。"""
    lines = []
    for old, new in zip(old_layers, new_layers):
        if old is new:
            continue
        if new.max_steps != old.max_steps:
            lines.append(f"{old.name} max_steps {old.max_steps}->{new.max_steps}")
        if new.max_out_spikes != old.max_out_spikes:
            lines.append(f"{old.name} max_out {old.max_out_spikes}->{new.max_out_spikes}")
    return lines


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
                               # _shrink_layers);跟 overflowed 分開記,
                               # 因為成因、要不要當「有問題」看待完全不同,不能
                               # 共用同一個欄位混在一起。


def run_epochs(*, layers, train_step, evaluate, params, opt_state,
               shuffle_key, start_epoch: int, total_epochs: int,
               train_split, val_split, batch_size: int, probe_batch,
               metrics_log, checkpointer, best: Best,
               weights_dir: str | None = None, weight_snapshot_every: int = 0,
               max_steps_reestimate_every: int = 1) -> EpochsOutcome:
    """跑 `[start_epoch, total_epochs)` 的訓練迴圈。

    **不知道「長大」這回事**:偵測到某 batch 的真實用量超過壓縮容量,就用
    `_grow_layers` 算出放大後的新 layer list、印 `[出界]`、回傳
    `EpochsOutcome(overflowed=True, grown_layers=...)`——不自己退 checkpoint /
    重編譯,那是呼叫端 `train()` 的 while 外圈。跑完整段沒出界回
    `overflowed=False`,`final_params` 是最後一個 epoch 的權重。

    **也不知道「縮小」這回事,但只在 epoch 成功跑完之後才問**:每個 epoch
    的 batch 迴圈裡,順便累積這個 epoch 所有真實 batch 的 `LayerDiag.
    min_steps_needed`/`n_out_spikes` 觀察最大值(`epoch_min_steps_needed`/
    `epoch_n_out_spikes`)。**epoch 成功跑完**(checkpoint 存完)之後,
    `max_steps_reestimate_every > 0` 且這個 epoch number 命中頻率時,拿這兩份
    累積值問 `_shrink_layers`(見 docs/規格書.md「conv 層 max_steps」,
    `max_out_spikes` 比照辦理)——不是探測批,是這個 epoch 真正跑過的訓練
    資料。真的縮了就印 `[縮小]`、回傳 `EpochsOutcome(reestimated=True,
    grown_layers=...)`,一樣交給 `train()` 的 while 外圈重編譯續跑,從剛存的
    checkpoint(下一個 epoch)接著練——用的是已經確定沒問題的當下權重,不是
    修正錯誤,跟出界的處理理由不同,但重編譯這件事借用同一條路。

    每個成功 epoch:val 評估 → 更新 `best` → `metrics_log.finish_epoch` →
    `checkpointer.save` → 權重快照(`weight_snapshot_every>0` 才存)→ 縮小檢查。
    """
    n_train = train_split.labels.shape[0]
    n_batches = max(1, n_train // batch_size)

    for epoch in range(start_epoch, total_epochs):
        shuffle_key, subkey = jax.random.split(shuffle_key)
        perm = jax.random.permutation(subkey, n_train)
        metrics_log.start_epoch()
        epoch_min_steps_needed = {layer.name: 0 for layer in layers}
        epoch_n_out_spikes = {layer.name: 0 for layer in layers}

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
                print(f"[出界] epoch={epoch} batch={b}: {where}")
                for line in _describe_growth(layers, grown, reduced_diags):
                    print(f"  {line}")
                return EpochsOutcome(overflowed=True, grown_layers=grown,
                                      final_params=params, best=best)

            params, opt_state = new_params, new_opt_state
            for layer, d in zip(layers, reduced_diags):
                epoch_min_steps_needed[layer.name] = max(
                    epoch_min_steps_needed[layer.name], int(d.min_steps_needed))
                epoch_n_out_spikes[layer.name] = max(
                    epoch_n_out_spikes[layer.name], int(d.n_out_spikes))
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

        # 逐 epoch 權重快照(純權重,不含 optimizer state):給事後分析工具用
        # (例如強制 chunk_size=1 重跑 run_network_traced 拿逐事件精確軌跡,
        # 見 docs/監測規格.md)。跟 checkpointer 的 checkpoint.npz 是兩回事——
        # checkpoint.npz 只為了續練,每個 epoch 覆寫;這裡逐 epoch 各自保留
        # 一份,才能事後回頭看任何一個存過的 epoch。
        if weight_snapshot_every > 0 and epoch % weight_snapshot_every == 0:
            save_params_npz(weight_snapshot_path(weights_dir, epoch), layers, params)

        # 縮小檢查:只在這個 epoch 真正成功跑完、checkpoint 也存完之後才問,
        # 用的是這個 epoch 累積的真實觀察值,不是探測批。
        if max_steps_reestimate_every > 0 and epoch % max_steps_reestimate_every == 0:
            shrunk = _shrink_layers(layers, epoch_min_steps_needed, epoch_n_out_spikes)
            if shrunk != layers:
                print(f"[縮小] epoch={epoch}:")
                for line in _describe_shrink(layers, shrunk):
                    print(f"  {line}")
                return EpochsOutcome(overflowed=False, grown_layers=shrunk,
                                      final_params=params, best=best, reestimated=True)

    return EpochsOutcome(overflowed=False, grown_layers=layers,
                          final_params=params, best=best)


def _make_exp_dir(run_name: str, exp_root=EXPERIMENTS_DIR) -> str:
    """訓練「開始前」就建好目錄——checkpoint 要在訓練過程中(每個 epoch 結束)
    持續寫進同一個目錄。`exp_root` 預設是 `experiments/`;e2e 測試傳
    `experiments/TEST_TEMP` 進來,把測試產物跟正式 run 隔開。

    只在這裡先建 `train/`(訓練產物每個 epoch 都要寫,是唯一保證一定會用到
    的子資料夾)。`weights/`(逐 epoch 權重快照)、`eval/`(`eval_test.py`)是
    條件式的,各自的消費者第一次要寫的時候自己建。"""
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

    # 逐 epoch 權重快照(docs/監測規格.md):train.weight_snapshot_every > 0 才開。
    # 只存純權重(save_params_npz,跟 params.npz 同格式),不含 optimizer state——
    # 事後要精確重現某個 epoch 當下的 forward,只需要權重,不需要訓練狀態。
    weight_snapshot_every = int(train_cfg.get("weight_snapshot_every", 0))
    weights_dir = os.path.join(exp_dir, WEIGHTS_DIRNAME)
    if weight_snapshot_every > 0:
        os.makedirs(weights_dir, exist_ok=True)
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
            weights_dir=weights_dir, weight_snapshot_every=weight_snapshot_every,
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
            name: {"queue": rows[-1][f"{name}_obs_event_queue"],
                   "out": rows[-1][f"{name}_obs_layer_spikes"]}
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
