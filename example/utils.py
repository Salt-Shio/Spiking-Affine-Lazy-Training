"""共用小工具:決定性種子設定、git commit hash、批次評估、params npz 存讀、
`experiments/<run>/` 底下的子資料夾命名。

`train_conv_compressed.py`(訓練)跟 `eval_test.py`(事後評估)有兩處各自
刻了一份幾乎一樣的東西,收在這裡單一來源:

- `make_evaluate`:分批 vmap 算 scores、導出 accuracy/loss/preds,兩邊本來
  各刻一份。
- `save_params_npz`/`load_params_npz`:`params.npz`/`best_params.npz` 的
  寫讀,key = 層名——訓練那邊寫、eval_test 這邊讀,約定只靠人記得對齊,
  現在收進同一份函式。
- `load_run_record`/`rebuild_layers`/`load_run_params`:從 `exp_dir` 重建
  一次訓練 run 的 layers/params,原本是 `eval_test.py` 私有的
  `_load_run_record`/`_rebuild_layers`/`_load_params`,`監測規格.md` §7.3
  就寫好「等第二個消費者出現再搬」——`example/notebooks/plot_channel_grid.ipynb`
  (讀 conv 層幾何做空間圖)是第二個消費者,搬過來共用。

`TRAIN_DIRNAME`/`TRACES_DIRNAME`/`EVAL_DIRNAME`:`experiments/<run>/` 底下
三個子資料夾的名字——訓練產物(`run.yaml`/`metrics.csv`/`checkpoint.npz`/
`params.npz`/`best_params.npz`)、`trace_probe.py` 的週期性探測、
`eval_test.py`/`plot_eval.py` 的事後評估,各自獨立一個資料夾。寫的一邊
(`train_conv_compressed.py`)跟讀的一邊(`eval_test.py`/`plot_eval.py`/
測試)都從這裡拿名字,不是各自重複寫字串常數,才不會兩邊漂移。
"""
import dataclasses
import os
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

from salt_core.layers import ConvLayer

from example.models.conv_net import build_network

TRAIN_DIRNAME = "train"
TRACES_DIRNAME = "traces"
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


def load_run_record(exp_dir: str) -> dict:
    """讀 `<exp_dir>/train/run.yaml`,回傳整份 config 快照 + 訓練中繼資料。"""
    with open(os.path.join(exp_dir, TRAIN_DIRNAME, "run.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def rebuild_layers(run_record: dict) -> list:
    """從 `run.yaml` 重建 layer list:形狀吃 config 快照,壓縮容量吃訓練結束的
    最終值(`final_capacity`)。init_k 不用套——重建出來的 layers 只用來讀取
    幾何/評估權重,不重新初始化。"""
    layers = build_network(run_record["config"]["model"])
    final_capacity = run_record.get("final_capacity", {})
    rebuilt = []
    for layer in layers:
        cap = final_capacity.get(layer.name)
        if cap is not None and isinstance(layer, ConvLayer):
            layer = dataclasses.replace(
                layer, L=int(cap["L"]), max_out_spikes=int(cap["max_out_spikes"]))
        rebuilt.append(layer)
    return rebuilt


def load_run_params(exp_dir: str, layers: list, which_params: str) -> tuple:
    """`which_params` 是 `"best"`(`best_params.npz`)或其他(`params.npz`)。"""
    fname = "best_params.npz" if which_params == "best" else "params.npz"
    return load_params_npz(os.path.join(exp_dir, TRAIN_DIRNAME, fname), layers)


def make_evaluate(net, decoder, eval_batch_size: int):
    """分批 vmap 算 scores,一次導出 `(accuracy, loss, preds)`。FC 輸出層仍是
    密集版 `build_fc_queue`,記憶體隨 vmap 樣本數線性長,不能整個 split 一次
    vmap,分批的理由跟訓練熱路徑的其他分批迴圈(`dormant_report`/
    `calibration_measure`)一樣。

    訓練期(`run_epochs` 每個 epoch 對 val split 的檢查)跟事後評估
    (`eval_test.py` 對 test/val split 的一次性評估)共用這一份——兩邊要的
    計算完全一樣,只差呼叫端要不要用 `loss`/`preds`;訓練那邊現在也免費多拿到
    一個 val loss,只是目前沒有欄位記它。
    """
    @jax.jit
    def _scores(params, event_times, x, y, c, n_real):
        result, _ = net.apply_batched(params, event_times, x, y, c, n_real)
        scores, _ = jax.vmap(decoder.decode)(result)
        return scores

    def evaluate(params, split):
        n = split.labels.shape[0]
        scores_parts = []
        for start in range(0, n, eval_batch_size):
            end = min(start + eval_batch_size, n)
            scores_parts.append(_scores(
                params, split.event_times[start:end], split.x[start:end],
                split.y[start:end], split.c[start:end], split.n_real_events[start:end]))
        scores = jnp.concatenate(scores_parts)
        preds = jnp.argmax(scores, axis=1)
        loss = float(jnp.mean(optax.softmax_cross_entropy(scores, split.labels_onehot)))
        accuracy = float(jnp.mean((preds == split.labels).astype(jnp.float32)))
        return accuracy, loss, np.asarray(preds)

    return evaluate
