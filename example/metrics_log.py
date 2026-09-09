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
        self._obs = {n: {"queue": 0, "out": 0} for n in self._conv_names}

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
        for name, g in grad_norms.items():
            self._grad[name].append(float(g))

    def finish_epoch(self, *, epoch: int, val_accuracy: float, layers: list,
                     dormant: dict | None = None) -> None:
        """組這個 epoch 的 row(欄位順序:epoch/loss/val → 逐 conv 容量+用量 →
        逐層 firing rate → 逐層 grad norm → 逐 conv dormant 指標 → 解碼器指標),
        append,該印就印。

        `dormant`:{conv_layer_name: {"dormant_frac": float, "act_p90p10": float}}
        (見 example/dormant.py),沒傳則對應欄位填 nan。
        """
        row = {"epoch": epoch, "train_loss": float(np.mean(self._losses)),
               "val_accuracy": val_accuracy}
        for layer in layers:
            if not isinstance(layer, ConvLayer):
                continue
            row[f"{layer.name}_L"] = layer.L
            row[f"{layer.name}_max_out"] = layer.max_out_spikes
            row[f"{layer.name}_obs_queue"] = self._obs[layer.name]["queue"]
            row[f"{layer.name}_obs_out"] = self._obs[layer.name]["out"]
        for name in self._layer_names:
            row[f"{name}_firing_rate"] = float(np.mean(self._firing[name]))
        for name in self._layer_names:
            row[f"{name}_grad_norm"] = float(np.mean(self._grad[name]))
        d = dormant or {}
        for name in self._conv_names:
            row[f"{name}_dormant_frac"] = float(d.get(name, {}).get("dormant_frac", float("nan")))
            row[f"{name}_act_p90p10"] = float(d.get(name, {}).get("act_p90p10", float("nan")))
        for k, vals in self._dec.items():
            row[f"decoder_{k}"] = float(np.mean(vals))
        self._rows.append(row)

        if epoch % self._progress_every == 0 or epoch == self._total_epochs - 1:
            self._print_progress(row, layers)

    def _print_progress(self, row: dict, layers: list) -> None:
        cap_str = " ".join(
            f"{l.name}:L={l.L}(用量{self._obs[l.name]['queue']})"
            f",out={l.max_out_spikes}(用量{self._obs[l.name]['out']})"
            for l in layers if isinstance(l, ConvLayer))
        fr_str = " ".join(f"{n}_fr={row[f'{n}_firing_rate']:.4f}" for n in self._layer_names)
        dorm_str = " ".join(
            f"{n}_dorm={row[f'{n}_dormant_frac']:.3f}" for n in self._conv_names
            if f"{n}_dormant_frac" in row and row[f"{n}_dormant_frac"] == row[f"{n}_dormant_frac"])
        dec_str = " ".join(f"{k}={row[f'decoder_{k}']:.4f}" for k in self._dec)
        print(f"epoch {row['epoch']}: loss={row['train_loss']:.4f} "
              f"val_acc={row['val_accuracy']:.4f} {cap_str} {fr_str}"
              f"{(' ' + dorm_str) if dorm_str else ''}"
              f"{(' ' + dec_str) if dec_str else ''}")

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
            print(f"  {layer.name} 最終容量:L={layer.L} max_out_spikes={layer.max_out_spikes}")
        if not self._rows:
            return
        last = self._rows[-1]
        for layer in layers:
            if not isinstance(layer, ConvLayer):
                continue
            oq, oo = last[f"{layer.name}_obs_queue"], last[f"{layer.name}_obs_out"]

            def _ratio(obs, cap):
                return f"{obs}/{cap} ({100.0 * obs / cap:.1f}%)" if cap else f"{obs}/{cap}"

            print(f"    {layer.name} 最後一個 epoch 用量:佇列 {_ratio(oq, layer.L)}、"
                  f"輸出 spike {_ratio(oo, layer.max_out_spikes)}")
