"""共用小工具:決定性種子設定、git commit hash、批次評估、讀 run 紀錄跟權重、
`experiments/<run>/` 底下的子資料夾命名。

`train_conv_compressed.py`(訓練)跟 `eval_test.py`(事後評估)有兩處各自
刻了一份幾乎一樣的東西,收在這裡單一來源:

- `make_evaluate`:分批 vmap 算 scores、導出 accuracy/loss/preds,兩邊本來
  各刻一份。`describe_growth` 是它跟訓練共用的容量放大訊息格式。
- `load_run_record`/`load_run_params`:讀一次訓練 run 的紀錄、權重。權重檔自帶
  網路描述(`salt_core.io`),讀回來就是存檔當下的網路。
- `split_raw_events`/`take_raw_events`:資料端的 split 轉成檢查過的 `RawEvents`、
  從裡面取一批(`data/` 不依賴 `salt_core`,轉換寫在這裡)。
- `weight_snapshot_path`:`experiments/<run>/weights/epoch_XXX.npz` 的命名
  慣例——訓練那邊(`train_conv_compressed.py`)週期性寫,事後分析工具讀,
  兩邊靠這個函式對齊路徑,不是各自重複拼字串。

`TRAIN_DIRNAME`/`WEIGHTS_DIRNAME`/`EVAL_DIRNAME`:`experiments/<run>/` 底下
三個子資料夾的名字——訓練產物(`run.yaml`/`metrics.csv`/`checkpoint.npz`/
`params.npz`/`best_params.npz`)、逐 epoch 權重快照(給事後重跑 forward 的
分析工具用,見 `docs/監測規格.md`)、`eval_test.py`/`plot_eval.py` 的事後
評估,各自獨立一個資料夾。寫的一邊(`train_conv_compressed.py`)跟讀的一邊
(`eval_test.py`/`plot_eval.py`/測試)都從這裡拿名字,不是各自重複寫字串
常數,才不會兩邊漂移。
"""
import os
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

from salt_core.capacity import grown_to_fit_batch, reduce_over_batch
from salt_core.io import load_weights
from salt_core.network import Network, RawEvents

TRAIN_DIRNAME = "train"
WEIGHTS_DIRNAME = "weights"
EVAL_DIRNAME = "eval"


def set_seed(seed: int) -> jax.Array:
    """設定 numpy 的全域亂數種子,回傳一個 JAX PRNGKey。

    JAX 沒有「設一次全域種子,之後所有呼叫都決定性」這種東西——呼叫端要自己
    把回傳的 key 一路往下 `jax.random.split`/傳遞下去,沒有明確傳 key 的
    `jax.random` 呼叫,JAX 本身就不允許,不是這裡沒設好。這個函式主要是為了
    (1) 順便處理少數還會用到 numpy 亂數的地方(例如某些第三方套件內部),
    (2) 統一入口,呼叫端不用自己記兩套種子設定方式。
    """
    np.random.seed(seed)
    return jax.random.PRNGKey(seed)


def get_git_commit_hash(repo_dir: str | None = None) -> str:
    """回傳 repo_dir(預設目前工作目錄)所在 git 倉庫的 commit hash。

    抓不到就回傳 "unknown"(例如根本不在 git 倉庫裡、或環境沒裝 git)——
    這只是實驗記錄的附加資訊,不該因為抓不到就讓整個訓練/評估腳本掛掉。
    """
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir,
                                capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return "unknown"


def weight_snapshot_path(weights_dir: str, epoch: int) -> str:
    """`experiments/<run>/weights/epoch_XXX.npz` 的路徑:那個 epoch 的權重連同當下的網路
    (salt_core.io.save_weights 的格式,不含 optimizer state)。"""
    return os.path.join(weights_dir, f"epoch_{epoch:03d}.npz")


def load_run_record(exp_dir: str) -> dict:
    """讀 `<exp_dir>/train/run.yaml`,回傳整份 config 快照 + 訓練中繼資料。"""
    with open(os.path.join(exp_dir, TRAIN_DIRNAME, "run.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_raw_events(split) -> RawEvents:
    """資料端 split 的事件欄位包成 RawEvents(leading axis = 樣本數),包之前檢查時間。
    時間不合法時 raise ValueError(見 RawEvents.checked)。"""
    return RawEvents.checked(split.event_times, split.x, split.y, split.c, split.n_real_events)


def take_raw_events(raw: RawEvents, idx) -> RawEvents:
    """從一批 RawEvents 取出 idx(index 陣列或 slice)那幾筆。"""
    return jax.tree_util.tree_map(lambda a: a[idx], raw)


def load_run_params(exp_dir: str, which_params: str) -> tuple[Network, tuple]:
    """回傳 (Network, 權重)。which_params 是 "best"(best_params.npz,best epoch 當下的網路)
    或其他(params.npz,訓練結束時的網路)。"""
    fname = "best_params.npz" if which_params == "best" else "params.npz"
    return load_weights(os.path.join(exp_dir, TRAIN_DIRNAME, fname))


def capacity_changes(old_layers: list, new_layers: list):
    """逐一產生容量有變的 (層索引, 旋鈕名, 舊值, 新值)。"""
    for i, (old, new) in enumerate(zip(old_layers, new_layers)):
        if old is new or old.capacity is None:
            continue
        for knob, old_value in old.capacity.items():
            if new.capacity[knob] != old_value:
                yield i, knob, old_value, new.capacity[knob]


def describe_growth(old_layers: list, new_layers: list, reduced_diags: list) -> list[str]:
    """哪些層的哪些容量旋鈕從多少放大到多少,一個旋鈕一行。

    reduced_diags: 對齊層的 LayerDiag,needed 是這個 batch 的最大值。
    格式:conv2 max_queue_len 32->2100(觀察 1401)。
    """
    return [f"{old_layers[i].name} {knob} {old}->{new}"
            f"(觀察 {int(reduced_diags[i].needed[knob])})"
            for i, knob, old, new in capacity_changes(old_layers, new_layers)]


def _make_scores_fn(network: Network, decoder):
    @jax.jit
    def scores_fn(params, raw_batch: RawEvents):
        output = network.apply_batched(params, raw_batch)
        scores, _ = jax.vmap(decoder.decode)(output.last)
        return scores, output.diags

    return scores_fn


def make_evaluate(network: Network, decoder, eval_batch_size: int, policies: dict):
    """回傳 evaluate(params, split) -> (accuracy, loss, preds, capacity_regrows)。

    分批 vmap 算 scores。某個 batch 容量出界時,照
    policies(層名 -> GrowthPolicy)放大評估用的容量、重算那個 batch,capacity_regrows
    是這次呼叫重算的次數。放大後的容量留給之後的呼叫,network 本身的容量不變。
    """
    layers = network.layers
    scores_fn = _make_scores_fn(network, decoder)

    def evaluate(params, split):
        nonlocal layers, scores_fn
        raw = split_raw_events(split)
        n = split.labels.shape[0]
        scores_parts = []
        regrows = 0
        for start in range(0, n, eval_batch_size):
            end = min(start + eval_batch_size, n)
            batch = take_raw_events(raw, slice(start, end))
            scores, diags = scores_fn(params, batch)
            grown = grown_to_fit_batch(layers, policies, diags)
            while grown is not layers:
                print(f"[評估出界] batch={start // eval_batch_size}: 放大評估容量重算")
                reduced = [reduce_over_batch(d) for d in diags]
                for line in describe_growth(layers, grown, reduced):
                    print(f"  {line}")
                layers = grown
                scores_fn = _make_scores_fn(network.replace_layers(layers), decoder)
                regrows += 1
                scores, diags = scores_fn(params, batch)
                grown = grown_to_fit_batch(layers, policies, diags)
            scores_parts.append(scores)
        scores = jnp.concatenate(scores_parts)
        preds = jnp.argmax(scores, axis=1)
        loss = float(jnp.mean(optax.softmax_cross_entropy(scores, split.labels_onehot)))
        accuracy = float(jnp.mean((preds == split.labels).astype(jnp.float32)))
        return accuracy, loss, np.asarray(preds), regrows

    return evaluate
