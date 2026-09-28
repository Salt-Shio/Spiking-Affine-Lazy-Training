"""訓練中的容量控制:什麼時候放大、縮小,每個 epoch 的需求最大值,放大縮小的事件紀錄。

放大縮小多少是 salt_core.capacity.GrowthPolicy 的公式;這裡決定什麼時候問它,並記下結果。
"""
from typing import NamedTuple

from salt_core.capacity import grown_to_fit, shrunk_to_observed


class KnobChange(NamedTuple):
    """一個容量旋鈕的變化。observed:觸發這次變化的需求量。"""
    layer: str
    knob: str
    old: int
    new: int
    observed: int

    def __str__(self) -> str:
        return f"{self.layer} {self.knob} {self.old}->{self.new}(觀察 {self.observed})"


class GrowEvent(NamedTuple):
    """訓練 batch 出界、放大後重來。resumed_from_epoch:退回的 checkpoint epoch,None 是從頭重來。"""
    epoch: int
    batch: int
    resumed_from_epoch: int | None
    changes: tuple

    def describe(self) -> list[str]:
        where = ("還沒有 checkpoint,退回訓練最初始狀態" if self.resumed_from_epoch is None
                 else f"退回 checkpoint(epoch={self.resumed_from_epoch})")
        return ([f"[出界] epoch={self.epoch} batch={self.batch}: {where}"]
                + [f"  {change}" for change in self.changes])

    def to_dict(self) -> dict:
        return {"kind": "grow", "epoch": self.epoch, "batch": self.batch,
                "resumed_from_epoch": self.resumed_from_epoch,
                "changes": [change._asdict() for change in self.changes]}


class ShrinkEvent(NamedTuple):
    """epoch 跑完後,依這個 epoch 的最大需求縮小容量。"""
    epoch: int
    changes: tuple

    def describe(self) -> list[str]:
        return ([f"[縮小] epoch={self.epoch}:"]
                + [f"  {c.layer} {c.knob} {c.old}->{c.new}" for c in self.changes])

    def to_dict(self) -> dict:
        return {"kind": "shrink", "epoch": self.epoch,
                "changes": [change._asdict() for change in self.changes]}


def knob_changes(old_layers: list, new_layers: list, needed: list) -> tuple:
    """容量有變的旋鈕,逐一轉成 KnobChange。

    needed: 對齊層的 {旋鈕名: 需求量},沒有容量的層不看。
    """
    changes = []
    for old, new, layer_needed in zip(old_layers, new_layers, needed):
        if old is new or old.capacity is None:
            continue
        for knob, old_value in old.capacity.items():
            if new.capacity[knob] != old_value:
                changes.append(KnobChange(layer=old.name, knob=knob, old=old_value,
                                          new=new.capacity[knob],
                                          observed=int(layer_needed[knob])))
    return tuple(changes)


class CapacityControl:
    """一次 run 的容量控制。

    policies: 層名 -> GrowthPolicy,有容量的層都要有。
    reestimate_every: 每幾個 epoch 檢查一次縮小,0 不檢查。
    events: 到目前為止的事件(GrowEvent、ShrinkEvent 的 to_dict()),依發生順序。
    """

    def __init__(self, policies: dict, reestimate_every: int):
        self.policies = policies
        self.reestimate_every = reestimate_every
        self.events: list = []
        self._epoch_needed: dict = {}

    def restore(self, events: list) -> None:
        """從 checkpoint 接著練時,換回存檔當下的事件紀錄。"""
        self.events = list(events)

    def start_epoch(self, layers: list) -> None:
        self._epoch_needed = {layer.name: dict.fromkeys(layer.capacity, 0)
                              for layer in layers if layer.capacity is not None}

    def record_batch(self, layers: list, diags: list) -> None:
        """一個成功的 batch。diags 對齊 layers,已經合併成一份。"""
        for layer, diag in zip(layers, diags):
            for knob, value in diag.needed.items():
                layer_needed = self._epoch_needed[layer.name]
                layer_needed[knob] = max(layer_needed[knob], int(value))

    @property
    def epoch_needed(self) -> dict:
        """這個 epoch 到目前為止,層名 -> 旋鈕名 -> 最大需求。"""
        return self._epoch_needed

    def grow(self, layers: list, diags: list, *, epoch: int, batch: int,
             resumed_from_epoch: int | None) -> tuple[list, GrowEvent]:
        """出界的 batch:回傳放大後的一列層跟這次的事件。"""
        grown = grown_to_fit(layers, self.policies, diags)
        event = GrowEvent(epoch=epoch, batch=batch, resumed_from_epoch=resumed_from_epoch,
                          changes=knob_changes(layers, grown, [d.needed for d in diags]))
        self.events.append(event.to_dict())
        return grown, event

    def shrink(self, layers: list, epoch: int) -> tuple[list, ShrinkEvent | None]:
        """epoch 跑完後照這個 epoch 的最大需求縮小;沒輪到或都沒縮時回傳 (layers, None)。"""
        if self.reestimate_every <= 0 or epoch % self.reestimate_every != 0:
            return layers, None
        shrunk = shrunk_to_observed(layers, self.policies, self._epoch_needed)
        if shrunk is layers:
            return layers, None
        needed = [self._epoch_needed.get(layer.name) for layer in layers]
        event = ShrinkEvent(epoch=epoch, changes=knob_changes(layers, shrunk, needed))
        self.events.append(event.to_dict())
        return shrunk, event
