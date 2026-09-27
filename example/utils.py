"""共用小工具:決定性種子設定、git commit hash、批次評估、params npz 存讀、
`experiments/<run>/` 底下的子資料夾命名。

`train_conv_compressed.py`(訓練)跟 `eval_test.py`(事後評估)有兩處各自
刻了一份幾乎一樣的東西,收在這裡單一來源:

- `make_evaluate`:分批 vmap 算 scores、導出 accuracy/loss/preds,兩邊本來
  各刻一份。`describe_growth` 是它跟訓練共用的容量放大訊息格式。
- `save_params_npz`/`load_params_npz`:`params.npz`/`best_params.npz` 的
  寫讀,key = 層名——訓練那邊寫、eval_test 這邊讀,約定只靠人記得對齊,
  現在收進同一份函式。
- `load_run_record`/`rebuild_layers`/`load_run_params`:從 `exp_dir` 重建
  一次訓練 run 的 layers/params,原本是 `eval_test.py` 私有的
  `_load_run_record`/`_rebuild_layers`/`_load_params`,現在給任何要重建
  layer 幾何/權重的呼叫端共用(例如逐 epoch 重跑 traced forward 的分析工具)。
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

from salt_core.capacity import Capacity, grown_to_fit_batch, reduce_over_batch

from example.models.conv_net import ConvNetCompressed, build_network

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


def save_params_npz(path: str, layers: list, params: tuple) -> None:
    """`params`(對齊 `layers` 的位置 tuple)存成 npz,key = 層名。訓練結束
    (`train_conv_compressed._write_experiment`)寫 `params.npz`/`best_params.npz`,
    `eval_test.py` 讀回——兩邊靠層名對齊,不是位置,收在同一份函式才不會
    兩邊 key 命名各自漂移。"""
    names = [layer.name for layer in layers]
    np.savez(path, **{n: np.asarray(w) for n, w in zip(names, params)})


def load_params_npz(path: str, layers: list) -> tuple:
    """`save_params_npz` 的反函式:讀回對齊 `layers` 的位置 tuple。"""
    data = np.load(path)
    return tuple(data[layer.name] for layer in layers)


def weight_snapshot_path(weights_dir: str, epoch: int) -> str:
    """`experiments/<run>/weights/epoch_XXX.npz` 的路徑命名慣例(不含 optimizer
    state,格式跟 `params.npz`/`best_params.npz` 一樣是 `save_params_npz` 存的
    純權重)——訓練那邊每 `train.weight_snapshot_every` 個 epoch 存一份,事後
    要精確重現某個 epoch 當下的 forward(例如強制 `chunk_size=1` 重跑
    `run_network_traced` 拿逐事件軌跡)就讀對應的這一份。"""
    return os.path.join(weights_dir, f"epoch_{epoch:03d}.npz")


def load_run_record(exp_dir: str) -> dict:
    """讀 `<exp_dir>/train/run.yaml`,回傳整份 config 快照 + 訓練中繼資料。"""
    with open(os.path.join(exp_dir, TRAIN_DIRNAME, "run.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def rebuild_layers(run_record: dict) -> list:
    """從 `run.yaml` 重建 layer list:形狀吃 config 快照,有容量的層換成訓練結束時的
    容量(`final_capacity`)。init_k 不用套——重建出來的 layers 只用來讀取幾何/
    評估權重,不重新初始化。"""
    layers = build_network(run_record["config"]["model"])
    final_capacity = run_record.get("final_capacity", {})
    return [layer.with_capacity(Capacity(**final_capacity[layer.name]))
            if layer.capacity is not None and layer.name in final_capacity else layer
            for layer in layers]


def load_run_params(exp_dir: str, layers: list, which_params: str) -> tuple:
    """`which_params` 是 `"best"`(`best_params.npz`)或其他(`params.npz`)。"""
    fname = "best_params.npz" if which_params == "best" else "params.npz"
    return load_params_npz(os.path.join(exp_dir, TRAIN_DIRNAME, fname), layers)


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
    格式:conv2 L 32->2100(觀察 1401)。
    """
    return [f"{old_layers[i].name} {knob} {old}->{new}"
            f"(觀察 {int(reduced_diags[i].needed[knob])})"
            for i, knob, old, new in capacity_changes(old_layers, new_layers)]


def _make_scores_fn(layers: list, decoder):
    net = ConvNetCompressed(layers)

    @jax.jit
    def scores_fn(params, event_times, x, y, c, n_real):
        result, diags = net.apply_batched(params, event_times, x, y, c, n_real)
        scores, _ = jax.vmap(decoder.decode)(result)
        return scores, diags

    return scores_fn


def make_evaluate(net, decoder, eval_batch_size: int, policies: dict):
    """回傳 evaluate(params, split) -> (accuracy, loss, preds, capacity_regrows)。

    分批 vmap 算 scores。某個 batch 容量出界時,照 policies(層名 -> GrowthPolicy)
    放大評估用的容量、重算那個 batch,capacity_regrows 是這次呼叫重算的次數。
    放大後的容量留給之後的呼叫,net 本身的容量不變。
    """
    layers = net.layers
    scores_fn = _make_scores_fn(layers, decoder)

    def evaluate(params, split):
        nonlocal layers, scores_fn
        n = split.labels.shape[0]
        scores_parts = []
        regrows = 0
        for start in range(0, n, eval_batch_size):
            end = min(start + eval_batch_size, n)
            batch = (split.event_times[start:end], split.x[start:end], split.y[start:end],
                     split.c[start:end], split.n_real_events[start:end])
            scores, diags = scores_fn(params, *batch)
            grown = grown_to_fit_batch(layers, policies, diags)
            while grown is not layers:
                print(f"[評估出界] batch={start // eval_batch_size}: 放大評估容量重算")
                reduced = [reduce_over_batch(d) for d in diags]
                for line in describe_growth(layers, grown, reduced):
                    print(f"  {line}")
                layers = grown
                scores_fn = _make_scores_fn(layers, decoder)
                regrows += 1
                scores, diags = scores_fn(params, *batch)
                grown = grown_to_fit_batch(layers, policies, diags)
            scores_parts.append(scores)
        scores = jnp.concatenate(scores_parts)
        preds = jnp.argmax(scores, axis=1)
        loss = float(jnp.mean(optax.softmax_cross_entropy(scores, split.labels_onehot)))
        accuracy = float(jnp.mean((preds == split.labels).astype(jnp.float32)))
        return accuracy, loss, np.asarray(preds), regrows

    return evaluate
