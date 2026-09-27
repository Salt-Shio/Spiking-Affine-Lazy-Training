"""訓練期指標紀錄:每個 epoch 的 loss / firing rate / 梯度範數 / 解碼器指標 /
壓縮容量真實用量,收成 metrics.csv 的一行,順便印進度。

從 `train_conv_compressed.py` 拆出來——原本那支腳本裡有五個平行的 per-epoch
累加 dict、一段 ~20 行 inline 組 row、一段週期性 print 再從 row 重推字串、
`_write_experiment` 又自己寫 csv + 跑一輪結尾 print。全部收進這個 class。

不管:`best_params` / `best_val_accuracy` 的挑選(那是 checkpoint 選擇、留在
`train()`)、`val_accuracy` 怎麼算(`train()` 算好傳進來)。
"""
import csv

import numpy as np

from salt_core.layers import ConvLayer


def _ratio(obs: int, cap: int) -> str:
    """`已用/容量(百分比)`,`cap` 是 0(理論上不會發生,防禦性處理)就不算百分比。"""
    return f"{obs}/{cap}({100.0 * obs / cap:.0f}%)" if cap else f"{obs}/{cap}"


class MetricsLog:
    """一次訓練 run 的逐 epoch 指標。

    `layer_names` / `conv_names` 在建構時固定(動態放大重建 layer list 不改層名、
    不改順序),之後每個 batch / epoch 把「當前的 layers」傳進來取 `.L` /
    `.max_out_spikes` 之類的即時值。
    """

    def __init__(self, layer_names: list, conv_names: list, total_epochs: int,
                 progress_every: int | None = None):
        self._layer_names = list(layer_names)
        self._conv_names = list(conv_names)
        self._total_epochs = total_epochs
        self._progress_every = progress_every or max(1, total_epochs // 20)
        self._rows: list[dict] = []
        self.start_epoch()

    def start_epoch(self) -> None:
        self._losses: list[float] = []
        self._firing = {n: [] for n in self._layer_names}
        self._grad = {n: [] for n in self._layer_names}
        self._dec: dict[str, list] = {}
        self._obs = {n: {"queue": 0, "out": 0, "steps": 0} for n in self._conv_names}

    def record_batch(self, *, loss, layers: list, reduced_diags: list,
                     grad_norms: dict, decoder_metrics: dict) -> None:
        """一個成功(沒出界)的 batch。`reduced_diags` 對齊 `layers`,元素是
        `salt_core.layers.LayerDiag`(值是 device 純量,這裡 float/int 轉)。
        """
        self._losses.append(float(loss))
        for k, v in decoder_metrics.items():
            self._dec.setdefault(k, []).append(float(v))
        for layer, d in zip(layers, reduced_diags):
            self._firing[layer.name].append(float(d.firing_rate))
            if layer.name in self._obs:
                o = self._obs[layer.name]
                o["queue"] = max(o["queue"], int(d.max_real_queue))
                o["out"] = max(o["out"], int(d.n_out_spikes))
                o["steps"] = max(o["steps"], int(d.min_steps_needed))
        for name, g in grad_norms.items():
            self._grad[name].append(float(g))

    def finish_epoch(self, *, epoch: int, val_accuracy: float, layers: list,
                     val_capacity_regrows: int, dormant_capacity_regrows: int,
                     dormant: dict | None = None) -> None:
        """組這個 epoch 的 row(欄位順序:epoch/loss/val → 逐 conv 容量+用量 →
        逐層 firing rate → 逐層 grad norm → 逐 conv dormant 指標 → 解碼器指標),
        append,該印就印。

        `dormant`:{conv_layer_name: {"dormant_frac": float}}(見
        salt_core/dormant.py),沒傳則對應欄位填 nan。
        val_capacity_regrows / dormant_capacity_regrows:val 評估、dormant 統計
        因為容量出界重算的次數。
        """
        row = {"epoch": epoch, "train_loss": float(np.mean(self._losses)),
               "val_accuracy": val_accuracy, "val_capacity_regrows": val_capacity_regrows}
        for layer in layers:
            if not isinstance(layer, ConvLayer):
                continue
            row[f"{layer.name}_max_event_queue"] = layer.L
            row[f"{layer.name}_max_layer_spikes"] = layer.max_out_spikes
            row[f"{layer.name}_max_steps"] = layer.max_steps
            row[f"{layer.name}_obs_event_queue"] = self._obs[layer.name]["queue"]
            row[f"{layer.name}_obs_layer_spikes"] = self._obs[layer.name]["out"]
            row[f"{layer.name}_obs_steps"] = self._obs[layer.name]["steps"]
        for name in self._layer_names:
            row[f"{name}_firing_rate"] = float(np.mean(self._firing[name]))
        for name in self._layer_names:
            row[f"{name}_grad_norm"] = float(np.mean(self._grad[name]))
        d = dormant or {}
        for name in self._conv_names:
            row[f"{name}_dormant_frac"] = float(d.get(name, {}).get("dormant_frac", float("nan")))
        row["dormant_capacity_regrows"] = dormant_capacity_regrows
        for k, vals in self._dec.items():
            row[f"decoder_{k}"] = float(np.mean(vals))
        self._rows.append(row)

        if epoch % self._progress_every == 0 or epoch == self._total_epochs - 1:
            self._print_progress(row, layers)

    def _print_progress(self, row: dict, layers: list) -> None:
        """人看的進度輸出,每層一行,固定 `已用/容量(百分比)` 格式(不用逗號
        分隔,cap=0 時退化成 `已用/0`,避免除以零)。"""
        dec_str = " ".join(f"{k}={row[f'decoder_{k}']:.4f}" for k in self._dec)
        print(f"epoch {row['epoch']}: loss={row['train_loss']:.4f} "
              f"val_acc={row['val_accuracy']:.4f}"
              f"{(' ' + dec_str) if dec_str else ''}")
        for l in layers:
            if isinstance(l, ConvLayer):
                o = self._obs[l.name]
                dorm = row.get(f"{l.name}_dormant_frac", float("nan"))
                dorm_str = f"  dorm={dorm:.3f}" if dorm == dorm else ""
                print(f"  {l.name:<6} 佇列 {_ratio(o['queue'], l.L)}  "
                      f"輸出spike {_ratio(o['out'], l.max_out_spikes)}  "
                      f"掃描步數 {_ratio(o['steps'], l.max_steps)}  "
                      f"fr={row[f'{l.name}_firing_rate']:.4f}{dorm_str}")
            else:
                print(f"  {l.name:<6} fr={row[f'{l.name}_firing_rate']:.4f}")

    @property
    def rows(self) -> list:
        return self._rows

    def write_csv(self, path: str) -> None:
        if not self._rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self._rows[0].keys()))
            writer.writeheader()
            writer.writerows(self._rows)

    def print_summary(self, layers: list) -> None:
        """訓練結束的逐層容量 + 最後一個 epoch 的真實用量比例。"""
        for layer in layers:
            if not isinstance(layer, ConvLayer):
                continue
            print(f"  {layer.name} 最終容量:L={layer.L} max_out_spikes={layer.max_out_spikes} "
                  f"max_steps={layer.max_steps}")
        if not self._rows:
            return
        last = self._rows[-1]
        for layer in layers:
            if not isinstance(layer, ConvLayer):
                continue
            oq = last[f"{layer.name}_obs_event_queue"]
            oo = last[f"{layer.name}_obs_layer_spikes"]
            os_ = last[f"{layer.name}_obs_steps"]
            print(f"    {layer.name} 最後一個 epoch 用量:佇列 {_ratio(oq, layer.L)}、"
                  f"輸出 spike {_ratio(oo, layer.max_out_spikes)}、"
                  f"掃描步數 {_ratio(os_, layer.max_steps)}")
