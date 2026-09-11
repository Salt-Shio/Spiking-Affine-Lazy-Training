"""訓練期的逐步軌跡探測(見 docs/監測規格.md §6 / §7 呼叫端政策)。

`TraceProbe` 週期性對一批**固定**的 train 樣本跑 `salt_core.run_network_traced`,
把逐神經元活動摘要疊進 `experiments/<run>/traces/summary.npz`,並每隔幾個 epoch
把少數樣本的完整 `(n, max_steps)` 軌跡另存 `full_epoch_XXX.npz`。

- forward-only、`stop_gradient`(`run_network_traced` 已保證),不進 `train_step`、
  不影響訓練熱路徑編譯 —— 另一個編譯目標。
- 探測批建構時就切好、之後不換,跨 epoch 可比,跟 `dormant_report` 同一個做法。
- FC 密集佇列記憶體隨 vmap 樣本數線性長,所以 K 筆是逐筆迴圈跑、不整批 vmap。
"""
import os

import jax
import jax.numpy as jnp
import numpy as np

from salt_core.layers import raw_events_to_stream, run_network_traced

# summary.npz 每層四個逐神經元 (n,) 摘要:
#   spike_count  該神經元在探測批上的平均總 spike 數
#   s_value_sum  Σ_t s_value 的平均(離門檻多近的連續量累積)
#   v_final      最終膜電位的平均
#   idle_frac    空轉步(event_ms = nan)比例的平均
_SUMMARY_KEYS = ("spike_count", "s_value_sum", "v_final", "idle_frac")
_FULL_FIELDS = ("spike_mask", "s_value", "v_steps", "event_ms")


def _summarise_one(trace) -> dict:
    """一層一份 `LayerForwardTrace` -> 逐神經元 `(n,)` 摘要 dict(對步軸縮減)。"""
    return {
        "spike_count": trace.spike_mask.sum(axis=1).astype(jnp.float32),
        "s_value_sum": trace.s_value.sum(axis=1),
        "v_final": trace.v_steps[:, -1],
        "idle_frac": jnp.isnan(trace.event_ms).mean(axis=1),
    }


class TraceProbe:
    """一次訓練 run 的逐步軌跡探測器。

    `probe_batch`:`(event_times, x, y, c, n_real_events)` 的 tuple,呼叫端已切成
    K 筆(建議 K = 4~8,取 train split 前 K 筆,跟 `dormant_report` 同一批)。
    `every`:每幾個 epoch 跑一次摘要(0 = 關,呼叫端不該建構這個物件)。
    `full_every`:每幾個 epoch 另存一次完整陣列(0 = 從不)。
    `full_samples`:完整陣列存前幾筆樣本(上限 K)。
    """

    def __init__(self, traces_dir: str, probe_batch, *, every: int,
                 total_epochs: int, full_every: int = 0, full_samples: int = 2):
        self._dir = traces_dir
        self._batch = tuple(np.asarray(v) for v in probe_batch)
        self._k = int(self._batch[0].shape[0])
        self._every = int(every)
        self._total_epochs = int(total_epochs)
        self._full_every = int(full_every)
        self._full_samples = min(int(full_samples), self._k)
        self._epochs: list[int] = []
        # name -> key -> list of (n,) np 陣列,一個 recorded epoch 一份
        self._summ: dict[str, dict[str, list]] = {}
        self._cache_key = None            # 上次編譯對應的 layers 物件(出界重建才換)
        self._summ_fn = None
        self._full_fn = None
        os.makedirs(self._dir, exist_ok=True)

    def due(self, epoch: int) -> bool:
        if self._every <= 0:
            return False
        return epoch % self._every == 0 or epoch == self._total_epochs - 1

    def _full_due(self, epoch: int) -> bool:
        return self._full_every > 0 and epoch % self._full_every == 0

    def _compile(self, layers: list) -> None:
        """把兩個 traced forward 編一次,快取到 layers 物件換掉(出界重建)為止。"""
        if layers is self._cache_key:
            return
        h_in, w_in = layers[0].h_in, layers[0].w_in

        @jax.jit
        def summ_fn(p, et, x, y, c, nr):
            stream = raw_events_to_stream(et, x, y, c, nr, h_in, w_in)
            return [_summarise_one(t)
                    for t in run_network_traced(layers, stream, p)]

        @jax.jit
        def full_fn(p, et, x, y, c, nr):
            stream = raw_events_to_stream(et, x, y, c, nr, h_in, w_in)
            return run_network_traced(layers, stream, p)

        self._cache_key, self._summ_fn, self._full_fn = layers, summ_fn, full_fn

    def run(self, layers: list, params, epoch: int) -> None:
        """對探測批跑一次 traced forward,更新 summary.npz;該存完整陣列就存。"""
        self._compile(layers)
        names = [layer.name for layer in layers]

        acc = None
        for i in range(self._k):
            per_layer = jax.tree_util.tree_map(
                np.asarray, self._summ_fn(params, *(v[i] for v in self._batch)))
            if acc is None:
                acc = per_layer
            else:
                acc = [{k: a[k] + d[k] for k in _SUMMARY_KEYS}
                       for a, d in zip(acc, per_layer)]
        acc = [{k: d[k] / self._k for k in _SUMMARY_KEYS} for d in acc]

        self._record(epoch, names, acc)
        self._write_summary(names)

        if self._full_due(epoch):
            self._dump_full(epoch, names, params)

    def _record(self, epoch: int, names: list, acc: list) -> None:
        if epoch in self._epochs:          # 出界重練會退回已記錄過的 epoch 號
            slot = self._epochs.index(epoch)
            for name, d in zip(names, acc):
                for k in _SUMMARY_KEYS:
                    self._summ[name][k][slot] = d[k]
            return
        self._epochs.append(epoch)
        for name, d in zip(names, acc):
            layer_d = self._summ.setdefault(name, {k: [] for k in _SUMMARY_KEYS})
            for k in _SUMMARY_KEYS:
                layer_d[k].append(d[k])

    def _write_summary(self, names: list) -> None:
        out = {"epochs": np.asarray(self._epochs, dtype=np.int32)}
        for name in names:
            for k in _SUMMARY_KEYS:
                out[f"{name}__{k}"] = np.stack(self._summ[name][k])   # (E, n)
        np.savez(os.path.join(self._dir, "summary.npz"), **out)

    def _dump_full(self, epoch: int, names: list, params) -> None:
        stacks: dict[str, list] = {f"{n}__{f}": []
                                   for n in names for f in _FULL_FIELDS}
        for i in range(self._full_samples):
            traces = self._full_fn(params, *(v[i] for v in self._batch))
            for name, t in zip(names, traces):
                for f in _FULL_FIELDS:
                    stacks[f"{name}__{f}"].append(np.asarray(getattr(t, f)))
        out = {k: np.stack(v) for k, v in stacks.items()}            # (S, n, max_steps)
        np.savez(os.path.join(self._dir, f"full_epoch_{epoch:03d}.npz"), **out)
