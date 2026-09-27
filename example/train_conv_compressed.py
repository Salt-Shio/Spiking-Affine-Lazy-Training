"""ConvNetCompressed 訓練 entrypoint。

這支腳本比一般訓練多的東西,是所有壓縮容量旋鈕的**動態放大**機制(完整設計
見 docs/math/conv事件佇列壓縮版推導.md 第 7.2 節):

- 每個 conv 層有三個會出界的容量:壓縮佇列長度 `L`、輸出 spike 上界
  `max_out_spikes`、掃描步數上界 `max_steps`。三個出界訊號都在 forward 裡算
  出來、由 `LayerDiag` 帶出。
- 沒有「靜態精算一次永久有效」的做法(原本 `conv_param_search.py` 的 L1 靜態
  量測太慢已廢),一律:config 給起始猜測 → 訓練中某個 batch 偵測出界 →
  照 `salt_core.capacity.GrowthPolicy` 放大 → 退回最近的 checkpoint → 用新的
  layer list 重編譯續練。放大多少是 `GrowthPolicy` 的公式,這裡只負責「作廢
  這個 batch、退 checkpoint、重編譯」這圈訓練編排,而且是對 layer list 的一個
  **通用迴圈**,不寫死層名。
- `max_steps` 跟 `max_out_spikes` 都多一條**選擇性縮小**的路
  (`shrunk_to_observed`):每 `train.max_steps_reestimate_every` 個**成功跑完的**
  epoch,用這個 epoch 裡所有真實 batch 觀察到的最大值決定要不要縮,比現有值
  小就縮、重編譯續跑。這條路跟出界不一樣,不需要退 checkpoint(用的是已經
  確定沒問題的當下權重),但重編譯這件事借用同一條 while 外圈。`max_steps`
  用的觀察值是 `LayerDiag.needed["max_steps"]`(理論上界,數學推導見
  docs/math/掃描步數上界推導.md,因為 `max_steps` 沒有天然的經驗值可用——設
  太小是掃描提早停止、靜默算錯,不像 `L`/`max_out_spikes` 是真實資料裝不下
  的被動事實);`max_out_spikes` 用的是 `LayerDiag.needed["max_out_spikes"]`(真實觀察值,
  跟長大訊號同一個量,理由見 docs/問題紀錄.md 第十四節——曾經嘗試過讓
  `max_out_spikes` 也用理論上界,實測太鬆、已撤回)。長大/縮小的目標值都用
  同一個公式算(`ceil(觀察值 * factor)`,`max_steps` 用
  `max_steps_grow_factor`、`max_out_spikes` 用 `out_grow_factor`),真實需求
  沒變時兩次算出來的目標值會相等,天然不會震盪;縮小還要另外掉到
  `max_steps_shrink_threshold`/`out_shrink_threshold` 比例以下才值得觸發
  (效率門檻,不影響防震盪)。

用法(config 路徑相對於 repo 根目錄,或給絕對路徑):
  python -m example.train_conv_compressed configs/conv/baseline.yaml
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
from salt_core.capacity import grown_to_fit, reduce_over_batch, shrunk_to_observed
from salt_core.layers import ConvLayer
from example.checkpoint import Checkpointer
from salt_core.dormant import dormant_report
from example.metrics_log import MetricsLog
from example.models.conv_net import (ConvNetCompressed, build_decoder, build_growth_policies,
                                     build_network)
from example.paths import DATASET_ROOT, EXPERIMENTS_DIR, REPO_ROOT, resolve_config
from example.utils import (TRAIN_DIRNAME, WEIGHTS_DIRNAME, capacity_changes, describe_growth,
                           get_git_commit_hash, make_evaluate, save_params_npz,
                           set_seed, weight_snapshot_path)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_learning_rate(train_cfg: dict, data_cfg: dict, batch_size: int):
    """`train.lr_cosine_decay`(選填,預設關閉):餘弦退火,對治
    docs/math/梯度下降曲率穩定性推導.md 第 5 節提到、目前還沒處理的 $\\eta$
    那一側——`weight_decay`/`score_cap` 都只能拖住 $\\lambda$ 的成長,穩定
    門檻 $2/\\eta$ 全程固定,$\\eta$ 不會隨訓練進行變安全。單獨測過的兩次
    `weight_decay`,反彈都均勻分布在整個訓練過程、包括接近結尾的地方(見
    `docs/問題紀錄.md` 第十五節),吻合「門檻沒有隨時間變寬」這個推論。餘弦
    退火讓 $\\eta$ 隨訓練進行下降,$2/\\eta$ 因此隨訓練進行升高。

    退火的總步數用這次 run **規劃**的 epoch 數換算
    (`steps_per_epoch * epochs`),不是實際走過的 optimizer step 數——出界
    退回 checkpoint 續練時沿用同一個 schedule(不重新算);中途還沒存過
    checkpoint 就整個重來(訓練最初期)才會讓 schedule 也跟著從頭起算,這是
    預期行為,不是 bug——那個時間點 `params` 本來就也是全新初始化,schedule
    的進度(存在 `opt_state` 裡)理應跟著歸零。
    """
    if not train_cfg.get("lr_cosine_decay", False):
        return train_cfg["lr"]
    steps_per_epoch = data_cfg["train_size"] // batch_size
    total_steps = train_cfg["epochs"] * steps_per_epoch
    return optax.cosine_decay_schedule(
        init_value=train_cfg["lr"], decay_steps=total_steps,
        alpha=train_cfg.get("lr_cosine_alpha", 0.0))


def _build_optimizer(train_cfg: dict, data_cfg: dict, batch_size: int):
    """`train.grad_clip_norm`(選填,預設關閉):症狀層解法
    (docs/math/梯度下降曲率穩定性推導.md §8.1)。不改變模型的體質(§8.2 的
    `weight_decay`/§8.4 的 `score_cap` 才是根因層),只把單步梯度的長度砍到
    這個門檻,擋住 Edge of Stability 發作當下那一步的過大位移。用
    `optax.clip_by_global_norm` 接在 `optax.adamw` 前面——clip 作用在原始梯度
    上,再交給 Adam 做自適應縮放,這樣 Adam 的一階/二階動量估計吃到的也是
    被夾過的梯度,不是砍完 update 才夾(那樣動量估計還是會被爆炸的原始梯度
    污染)。之前唯一測過的一次(三機制合測)沒有乾淨隔離——真正的死亡崩潰
    元凶後來定位是同批合測的 `label_smoothing`(已撤除,§8.3),`grad_clip_norm`
    本身從未單獨驗證過。"""
    lr = _build_learning_rate(train_cfg, data_cfg, batch_size)
    adamw = optax.adamw(lr, weight_decay=train_cfg.get("weight_decay", 0.0))
    grad_clip_norm = train_cfg.get("grad_clip_norm")
    if grad_clip_norm is None:
        return adamw
    return optax.chain(optax.clip_by_global_norm(grad_clip_norm), adamw)


def _cross_entropy_loss(scores: jax.Array, batch_labels_onehot: jax.Array,
                        score_cap: float | None) -> jax.Array:
    """夾住輸出上限(docs/math/梯度下降曲率穩定性推導.md §8.4)。`score_cap`
    是 `None`(預設)時原樣通過,等價於原本的 cross entropy。設定時用
    `score_cap * tanh(scores / score_cap)` 把每個類別的分數飽和限制在
    `[-score_cap, score_cap]`,logit 差距因此有硬上限 `2*score_cap`。跟
    label smoothing(已撤除,見同節)不同:這裡不改變分類目標本身,對還沒
    逼近上限的樣本梯度幾乎不受影響,只有分數已經接近上限時才開始飽和——用
    `tanh` 而不是硬 `jnp.clip`,是因為硬裁切在超過門檻後梯度完全變成 0,等於
    製造另一種死區。eval 那邊(example/utils.py)維持原始未夾住的分數,不然
    驗證指標會被訓練用的飽和轉換污染。回傳每筆樣本的 loss,形狀 `(batch,)`。"""
    if score_cap is not None:
        scores = score_cap * jnp.tanh(scores / score_cap)
    return optax.softmax_cross_entropy(scores, batch_labels_onehot)


def make_train_step(net, optimizer, decoder, score_cap: float | None = None):
    def loss_fn(params, batch_event_times, batch_x, batch_y, batch_c, batch_n_real,
                batch_labels_onehot):
        result, diagnostics = net.apply_batched(params, batch_event_times, batch_x, batch_y,
                                                 batch_c, batch_n_real)
        # 解碼器把最後一層的 LayerForwardResult 讀成分數(膜電位回歸 = v_final、
        # 頻率/群體 = s_value 加總),網路本身不挑 readout。dec_metrics 是這個
        # 編碼特定的純量(可空,膜電位回歸就是空的)。這次單獨測試,weight_decay
        # 維持關閉(config 不填,預設 0)。
        scores, dec_metrics = jax.vmap(decoder.decode)(result)
        per_sample_loss = _cross_entropy_loss(scores, batch_labels_onehot, score_cap)
        loss = jnp.mean(per_sample_loss)
        reduced = [reduce_over_batch(d) for d in diagnostics]
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


def _describe_shrink(old_layers: list, new_layers: list) -> list[str]:
    """跟 `describe_growth` 對應,一個旋鈕一行,格式:`conv1 max_steps 200->134`。"""
    return [f"{old_layers[i].name} {knob} {old}->{new}"
            for i, knob, old, new in capacity_changes(old_layers, new_layers)]


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
                               # shrunk_to_observed);跟 overflowed 分開記,
                               # 因為成因、要不要當「有問題」看待完全不同,不能
                               # 共用同一個欄位混在一起。


def run_epochs(*, layers, policies, train_step, evaluate, params, opt_state,
               shuffle_key, start_epoch: int, total_epochs: int,
               train_split, val_split, batch_size: int, probe_batch,
               metrics_log, checkpointer, best: Best,
               weights_dir: str | None = None, weight_snapshot_every: int = 0,
               max_steps_reestimate_every: int = 1) -> EpochsOutcome:
    """跑 `[start_epoch, total_epochs)` 的訓練迴圈。

    **不知道「長大」這回事**:偵測到某 batch 的真實用量超過壓縮容量,就用
    `grown_to_fit` 算出放大後的新 layer list、印 `[出界]`、回傳
    `EpochsOutcome(overflowed=True, grown_layers=...)`——不自己退 checkpoint /
    重編譯,那是呼叫端 `train()` 的 while 外圈。跑完整段沒出界回
    `overflowed=False`,`final_params` 是最後一個 epoch 的權重。

    **也不知道「縮小」這回事,但只在 epoch 成功跑完之後才問**:每個 epoch
    的 batch 迴圈裡,順便累積這個 epoch 所有真實 batch 的 `LayerDiag.needed`
    最大值(`epoch_needed`)。**epoch 成功跑完**(checkpoint 存完)之後,
    `max_steps_reestimate_every > 0` 且這個 epoch number 命中頻率時,拿這兩份
    累積值問 `shrunk_to_observed`(見 docs/規格書.md「conv 層 max_steps」,
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
        epoch_needed = {layer.name: dict.fromkeys(layer.capacity, 0)
                        for layer in layers if layer.capacity is not None}

        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            (new_params, new_opt_state, loss, reduced_diags, reduced_metrics,
             grad_norms) = train_step(
                params, opt_state, train_split.event_times[idx], train_split.x[idx],
                train_split.y[idx], train_split.c[idx],
                train_split.n_real_events[idx], train_split.labels_onehot[idx])

            # 出界偵測:任何一層要長大,就作廢這個 batch、回報給外圈。
            grown = grown_to_fit(layers, policies, reduced_diags)
            if grown != layers:
                where = (f"退回 checkpoint(epoch={checkpointer.last_epoch})"
                         if checkpointer.exists() else "還沒有 checkpoint,退回訓練最初始狀態")
                print(f"[出界] epoch={epoch} batch={b}: {where}")
                for line in describe_growth(layers, grown, reduced_diags):
                    print(f"  {line}")
                return EpochsOutcome(overflowed=True, grown_layers=grown,
                                      final_params=params, best=best)

            params, opt_state = new_params, new_opt_state
            for layer, d in zip(layers, reduced_diags):
                for knob, value in d.needed.items():
                    epoch_needed[layer.name][knob] = max(epoch_needed[layer.name][knob],
                                                         int(value))
            metrics_log.record_batch(loss=loss, layers=layers,
                                      reduced_diags=reduced_diags,
                                      grad_norms=grad_norms,
                                      decoder_metrics=reduced_metrics)

        val_accuracy, _val_loss, _, val_regrows = evaluate(params, val_split)
        if val_accuracy > best.val_accuracy:
            best = Best(params=params, val_accuracy=val_accuracy, epoch=epoch)
        dormant, dormant_regrows = dormant_report(layers, params, probe_batch, policies)
        if dormant_regrows:
            print(f"[dormant 出界] epoch={epoch}: 放大探測用容量重算 {dormant_regrows} 次")
        metrics_log.finish_epoch(epoch=epoch, val_accuracy=val_accuracy, layers=layers,
                                  dormant=dormant, val_capacity_regrows=val_regrows,
                                  dormant_capacity_regrows=dormant_regrows)
        checkpointer.save(params=params, opt_state=opt_state,
                          shuffle_key=shuffle_key, epoch=epoch)

        # 逐 epoch 權重快照(純權重,不含 optimizer state):給事後分析工具用
        # (例如強制 chunk_size=1 重跑 run_network(..., trace=True) 拿逐事件精確軌跡,
        # 見 docs/監測規格.md)。跟 checkpointer 的 checkpoint.npz 是兩回事——
        # checkpoint.npz 只為了續練,每個 epoch 覆寫;這裡逐 epoch 各自保留
        # 一份,才能事後回頭看任何一個存過的 epoch。
        if weight_snapshot_every > 0 and epoch % weight_snapshot_every == 0:
            save_params_npz(weight_snapshot_path(weights_dir, epoch), layers, params)

        # 縮小檢查:只在這個 epoch 真正成功跑完、checkpoint 也存完之後才問,
        # 用的是這個 epoch 累積的真實觀察值,不是探測批。
        if max_steps_reestimate_every > 0 and epoch % max_steps_reestimate_every == 0:
            shrunk = shrunk_to_observed(layers, policies, epoch_needed)
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
    # 層名在動態放大縮小時不變,policies 建一次就好
    policies = build_growth_policies(model_cfg, layers)

    layer_names = [layer.name for layer in layers]
    # dormant 統計只算 conv 隱藏層(salt_core.dormant 的挑層規則)
    dormant_names = [layer.name for layer in layers if isinstance(layer, ConvLayer)]

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

    metrics_log = MetricsLog(layer_names, dormant_names, total_epochs=train_cfg["epochs"])

    # while 外圈 = 自動長大控制系統:setup params(fresh / checkpoint)→ 跑
    # run_epochs → 沒出界就結束、出界就換成放大後的 layers 重編譯再繞。
    # 「跑 epoch」本身完全在 run_epochs 裡,不知道長大這回事。
    best = Best(params=None, val_accuracy=-1.0, epoch=-1)
    while True:
        net = ConvNetCompressed(layers)
        # 優化器組裝(`weight_decay`/`grad_clip_norm`)見 `_build_optimizer`。
        # label_smoothing 已撤除,不要再用(§8.3 討論)。
        optimizer = _build_optimizer(train_cfg, data_cfg, batch_size)

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
            layers=layers, policies=policies,
            train_step=make_train_step(
                net, optimizer, decoder, score_cap=train_cfg.get("score_cap")),
            evaluate=make_evaluate(net, decoder, eval_batch_size=batch_size, policies=policies),
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
    # 一份 write-once 的 run 紀錄:輸入 config 快照 + commit + 最終容量 + 最佳
    # 指標。輸入(raw_cfg)不被改;結果不塞回它。
    run_record = {
        "config": raw_cfg,
        "git_commit": get_git_commit_hash(str(REPO_ROOT)),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "best": {"val_accuracy": best_val_accuracy, "epoch": best_epoch},
        "final_capacity": {layer.name: dict(layer.capacity)
                           for layer in layers if layer.capacity is not None},
        "last_epoch_obs": ({layer.name: metrics_log.last_needed(layer)
                            for layer in layers if layer.capacity is not None}
                           if metrics_log.rows else {}),
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
