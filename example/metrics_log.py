"""訓練時的逐 epoch 指標:loss、firing rate、梯度範數、解碼器指標、容量跟用量,寫成 metrics.csv 的一列,
順便印進度。best 的挑選跟 val_accuracy 的計算不在這裡。
"""
import csv
from typing import NamedTuple

import numpy as np


class KnobColumns(NamedTuple):
    """一個容量旋鈕在 metrics.csv 的兩個欄名(前面接層名)跟進度輸出的標籤。"""
    capacity: str
    needed: str
    label: str


# 欄名沿用舊 run 的命名,舊的 metrics.csv 才能用同一套程式讀。
KNOB_COLUMNS = {"max_queue_len": KnobColumns("max_event_queue", "obs_event_queue", "佇列"),
                "max_out_spikes": KnobColumns("max_layer_spikes", "obs_layer_spikes", "輸出spike"),
                "max_extra_steps": KnobColumns("max_extra_steps", "obs_extra_steps", "額外掃描步數")}


def _ratio(obs: int, cap: int) -> str:
    """「已用/容量(百分比)」;cap 是 0 時不算百分比。"""
    return f"{obs}/{cap}({100.0 * obs / cap:.0f}%)" if cap else f"{obs}/{cap}"


class MetricsLog:
    """一次訓練 run 的逐 epoch 指標。

    layer_names、dormant_names: 建構時固定;容量放大重建層時不改層名跟順序。
    每個 batch、epoch 傳進當下的 layers,讀容量的現值。
    """

    def __init__(self, layer_names: list, dormant_names: list, total_epochs: int,
                 progress_every: int | None = None):
        self._layer_names = list(layer_names)
        self._dormant_names = list(dormant_names)
        self._total_epochs = total_epochs
        self._progress_every = progress_every or max(1, total_epochs // 20)
        self._rows: list[dict] = []
        self.start_epoch()

    def start_epoch(self) -> None:
        self._losses: list[float] = []
        self._firing = {n: [] for n in self._layer_names}
        self._grad = {n: [] for n in self._layer_names}
        self._dec: dict[str, list] = {}

    def record_batch(self, *, loss, layers: list, reduced_diags: list,
                     grad_norms: dict, decoder_metrics: dict) -> None:
        """一個沒出界的 batch。reduced_diags 對齊 layers,是合併過的 LayerDiag。"""
        self._losses.append(float(loss))
        for k, v in decoder_metrics.items():
            self._dec.setdefault(k, []).append(float(v))
        for layer, d in zip(layers, reduced_diags):
            self._firing[layer.name].append(float(d.firing_rate))
        for name, g in grad_norms.items():
            self._grad[name].append(float(g))

    def finish_epoch(self, *, epoch: int, val_accuracy: float, layers: list, needed: dict,
                     val_capacity_regrows: int, dormant_capacity_regrows: int,
                     dormant: dict | None = None) -> None:
        """組這個 epoch 的一列、加進紀錄,輪到時印進度。

        欄位順序:epoch、loss、val,有容量的層的容量跟用量,逐層 firing rate,逐層梯度範數,dormant 層,
        解碼器指標。
        needed: 層名 -> 旋鈕名 -> 這個 epoch 的最大需求,有容量的層都要有。
        dormant: {層名: {"dormant_frac": float}},沒給的欄位填 nan。
        val_capacity_regrows、dormant_capacity_regrows: val 評估、dormant 統計因為容量出界重算的次數。
        """
        row = {"epoch": epoch, "train_loss": float(np.mean(self._losses)),
               "val_accuracy": val_accuracy, "val_capacity_regrows": val_capacity_regrows}
        for layer in layers:
            if layer.capacity is None:
                continue
            for knob, value in layer.capacity.items():
                row[f"{layer.name}_{KNOB_COLUMNS[knob].capacity}"] = value
            for knob in layer.capacity:
                row[f"{layer.name}_{KNOB_COLUMNS[knob].needed}"] = needed[layer.name][knob]
        for name in self._layer_names:
            row[f"{name}_firing_rate"] = float(np.mean(self._firing[name]))
        for name in self._layer_names:
            row[f"{name}_grad_norm"] = float(np.mean(self._grad[name]))
        d = dormant or {}
        for name in self._dormant_names:
            row[f"{name}_dormant_frac"] = float(d.get(name, {}).get("dormant_frac", float("nan")))
        row["dormant_capacity_regrows"] = dormant_capacity_regrows
        for k, vals in self._dec.items():
            row[f"decoder_{k}"] = float(np.mean(vals))
        self._rows.append(row)

        if epoch % self._progress_every == 0 or epoch == self._total_epochs - 1:
            self._print_progress(row, layers)

    def _print_progress(self, row: dict, layers: list) -> None:
        """進度輸出,每層一行,「已用/容量(百分比)」。"""
        dec_str = " ".join(f"{k}={row[f'decoder_{k}']:.4f}" for k in self._dec)
        print(f"epoch {row['epoch']}: loss={row['train_loss']:.4f} "
              f"val_acc={row['val_accuracy']:.4f}"
              f"{(' ' + dec_str) if dec_str else ''}")
        for l in layers:
            if l.capacity is None:
                print(f"  {l.name:<6} fr={row[f'{l.name}_firing_rate']:.4f}")
                continue
            usage = "  ".join(
                f"{KNOB_COLUMNS[knob].label} "
                f"{_ratio(row[f'{l.name}_{KNOB_COLUMNS[knob].needed}'], value)}"
                for knob, value in l.capacity.items())
            dorm = row.get(f"{l.name}_dormant_frac", float("nan"))
            dorm_str = f"  dorm={dorm:.3f}" if dorm == dorm else ""
            print(f"  {l.name:<6} {usage}  fr={row[f'{l.name}_firing_rate']:.4f}{dorm_str}")

    @property
    def rows(self) -> list:
        return self._rows

    def restore(self, rows: list) -> None:
        """從 checkpoint 接著練時,換回存檔當下已完成 epoch 的列。"""
        self._rows = list(rows)

    def last_needed(self, layer) -> dict[str, int]:
        """最後一個 epoch 這層每個容量旋鈕的最大需求。"""
        last = self._rows[-1]
        return {knob: last[f"{layer.name}_{KNOB_COLUMNS[knob].needed}"] for knob in layer.capacity}

    def write_csv(self, path: str) -> None:
        if not self._rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self._rows[0].keys()))
            writer.writeheader()
            writer.writerows(self._rows)

    def print_summary(self, layers: list) -> None:
        """訓練結束時印逐層容量跟最後一個 epoch 的用量比例。"""
        capacity_layers = [layer for layer in layers if layer.capacity is not None]
        for layer in capacity_layers:
            values = " ".join(f"{knob}={value}" for knob, value in layer.capacity.items())
            print(f"  {layer.name} 最終容量:{values}")
        if not self._rows:
            return
        for layer in capacity_layers:
            needed = self.last_needed(layer)
            usage = "、".join(f"{KNOB_COLUMNS[knob].label} {_ratio(needed[knob], value)}"
                             for knob, value in layer.capacity.items())
            print(f"    {layer.name} 最後一個 epoch 用量:{usage}")
