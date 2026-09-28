"""example 共用的小工具:種子、git commit、config、批次評估、讀 run 紀錄跟權重、RawEvents 轉換。

TRAIN_DIRNAME、WEIGHTS_DIRNAME、EVAL_DIRNAME 是 experiments/<run>/ 底下三個子資料夾的名字:
訓練產物、逐 epoch 權重快照、事後評估。寫的一邊跟讀的一邊都從這裡拿名字。
"""
import os
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

from example.training.capacity_control import knob_changes
from salt_core.capacity import grown_to_fit_batch, reduce_over_batch
from salt_core.io import load_weights
from salt_core.network import Network, RawEvents

TRAIN_DIRNAME = "train"
WEIGHTS_DIRNAME = "weights"
EVAL_DIRNAME = "eval"


def load_config(path: str) -> dict:
    """讀 yaml config。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> jax.Array:
    """設定 numpy 的全域亂數種子,回傳 JAX PRNGKey。JAX 的亂數要呼叫端自己傳 key 下去。"""
    np.random.seed(seed)
    return jax.random.PRNGKey(seed)


def get_git_commit_hash(repo_dir: str | None = None) -> str:
    """repo_dir(預設目前工作目錄)的 git commit hash;抓不到時回傳 "unknown",不讓腳本失敗。"""
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir,
                                capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return "unknown"


def weight_snapshot_path(weights_dir: str, epoch: int) -> str:
    """weights_dir/epoch_XXX.npz:那個 epoch 的權重連同當下的網路(save_weights 格式,不含 optimizer state)。"""
    return os.path.join(weights_dir, f"epoch_{epoch:03d}.npz")


def load_run_record(exp_dir: str) -> dict:
    """讀 exp_dir/train/run.yaml,回傳整份內容(config 快照跟訓練紀錄)。"""
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


def _make_scores_fn(network: Network, decoder):
    @jax.jit
    def scores_fn(params, raw_batch: RawEvents):
        output = network.apply_batched(params, raw_batch)
        scores, _ = jax.vmap(decoder.decode)(output.last)
        return scores, output.diags, output.fits

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
            scores, diags, fits = scores_fn(params, batch)
            while not bool(jnp.all(fits)):
                print(f"[評估出界] batch={start // eval_batch_size}: 放大評估容量重算")
                grown = grown_to_fit_batch(layers, policies, diags)
                reduced = [reduce_over_batch(d) for d in diags]
                for change in knob_changes(layers, grown, [d.needed for d in reduced]):
                    print(f"  {change}")
                layers = grown
                scores_fn = _make_scores_fn(network.replace_layers(layers), decoder)
                regrows += 1
                scores, diags, fits = scores_fn(params, batch)
            scores_parts.append(scores)
        scores = jnp.concatenate(scores_parts)
        preds = jnp.argmax(scores, axis=1)
        loss = float(jnp.mean(optax.softmax_cross_entropy(scores, split.labels_onehot)))
        accuracy = float(jnp.mean((preds == split.labels).astype(jnp.float32)))
        return accuracy, loss, np.asarray(preds), regrows

    return evaluate
