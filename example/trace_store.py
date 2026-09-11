"""`example/trace_probe.py`(寫)/ `example/inspect_traces.py`(讀)共用的 npz
扁平命名慣例。

一份 `summary.npz` / `full_epoch_XXX.npz` 要裝「多層 x 多欄」的陣列,但 npz
本身只有一層扁平的 key -> array,所以用 `"<層名>__<欄名>"` 當 key 把兩維攤平。
這支模組是這個命名慣例唯一的實作——寫端組 key、讀端拆 key/取層名都從這裡拿,
不各自重新刻一次 `f"{a}__{b}"` / `.split("__")`。
"""
from typing import Iterable

_SEP = "__"


def pack_key(layer_name: str, field: str) -> str:
    """`(層名, 欄名)` -> npz key,例如 `("conv1", "spike_count")` -> `"conv1__spike_count"`。"""
    return f"{layer_name}{_SEP}{field}"


def unpack_key(key: str) -> tuple[str, str]:
    """npz key -> `(層名, 欄名)`。層名本身不能含 `__`(pack_key 的反函式)。"""
    name, field = key.split(_SEP, 1)
    return name, field


def layer_names(keys: Iterable[str]) -> list[str]:
    """從一堆 npz key 依首次出現順序取出不重複的層名。略過沒有 `__` 的 key
    (例如 `summary.npz` 裡的 `epochs`)。"""
    names: list[str] = []
    for key in keys:
        if _SEP not in key:
            continue
        name, _field = unpack_key(key)
        if name not in names:
            names.append(name)
    return names
