"""網路描述跟權重的存讀:Network <-> dict,權重連同網路描述存成 npz。"""
import dataclasses
import json
import os
from collections.abc import Mapping, Sequence
from typing import Any, TypedDict

import jax
import jax.numpy as jnp
import numpy as np

from salt_core.layers import ConvLayer, FCLayer, Layer
from salt_core.network import Network

FORMAT_VERSION = 1
# npz 裡放網路描述的欄位;其餘欄位是權重,key 是層名
NETWORK_KEY = "__network__"
# 網路描述裡的 type <-> 層類別
_LAYER_TYPES = {"conv": ConvLayer, "fc": FCLayer}


class NetworkDescription(TypedDict):
    """network_to_dict 的回傳。"""
    format: int                    # FORMAT_VERSION
    input_shape: list[int]         # 網路的輸入形狀
    layers: list[dict[str, Any]]   # 每層 {"type": "conv" 或 "fc", 其餘是層的欄位名 -> 值}


def _layer_type(layer: Layer) -> str:
    for type_name, layer_class in _LAYER_TYPES.items():
        if type(layer) is layer_class:
            return type_name
    raise ValueError(f"{layer.name} 的型別 {type(layer).__name__} 不能序列化")


def network_to_dict(network: Network) -> NetworkDescription:
    """網路描述:輸入網格加上每層的全部欄位(含容量),可以直接存成 json、yaml。

    層的型別不認得時 raise ValueError。
    """
    return {"format": FORMAT_VERSION,
            "input_shape": list(network.input_shape),
            "layers": [{"type": _layer_type(layer), **dataclasses.asdict(layer)}
                       for layer in network.layers]}


def network_from_dict(description: NetworkDescription) -> Network:
    """network_to_dict 的反函式。format 不是 FORMAT_VERSION、或層的 type 不認得時 raise ValueError。"""
    if description.get("format") != FORMAT_VERSION:
        raise ValueError(f"網路描述的 format 要是 {FORMAT_VERSION},"
                         f"拿到 {description.get('format')!r}")
    layers = []
    for entry in description["layers"]:
        fields = dict(entry)
        type_name = fields.pop("type")
        if type_name not in _LAYER_TYPES:
            raise ValueError(f"不認得的層 type {type_name!r}(可用:{list(_LAYER_TYPES)})")
        layers.append(_LAYER_TYPES[type_name](**fields))
    return Network(input_shape=description["input_shape"], layers=layers)


def weights_to_arrays(network: Network, weights: Sequence[jax.Array | np.ndarray]
                      ) -> dict[str, np.ndarray]:
    """npz 的欄位:每層的權重一個陣列(key 是層名),加上 NETWORK_KEY 放網路描述的 json。

    weights: 對齊 network.layers。
    層數對不上、層名重複、或層名等於 NETWORK_KEY 時 raise ValueError。
    """
    names = [layer.name for layer in network.layers]
    if len(weights) != len(names):
        raise ValueError(f"權重有 {len(weights)} 份,網路有 {len(names)} 層")
    if len(set(names)) != len(names) or NETWORK_KEY in names:
        raise ValueError(f"層名要不重複、而且不能是 {NETWORK_KEY!r},拿到 {names}")
    arrays = {name: np.asarray(w) for name, w in zip(names, weights)}
    arrays[NETWORK_KEY] = np.asarray(json.dumps(network_to_dict(network)))
    return arrays


def weights_from_arrays(arrays: Mapping[str, np.ndarray]) -> tuple[Network, tuple[jax.Array, ...]]:
    """weights_to_arrays 的反函式。回傳 (Network, 對齊 layers 的權重),權重是 JAX 陣列。

    arrays: dict 或 np.load 讀回的 npz。沒有 NETWORK_KEY 時 raise ValueError。
    """
    if NETWORK_KEY not in arrays:
        raise ValueError(f"沒有網路描述欄位 {NETWORK_KEY!r}")
    network = network_from_dict(json.loads(str(arrays[NETWORK_KEY])))
    return network, tuple(jnp.asarray(arrays[layer.name]) for layer in network.layers)


def save_weights(path: str | os.PathLike, network: Network,
                 weights: Sequence[jax.Array | np.ndarray]) -> None:
    """權重連同網路描述存成 npz,格式見 weights_to_arrays。"""
    np.savez(path, **weights_to_arrays(network, weights))


def load_weights(path: str | os.PathLike) -> tuple[Network, tuple[jax.Array, ...]]:
    """讀 save_weights 存的檔,回傳 (Network, 權重),權重是 JAX 陣列。"""
    with np.load(path) as arrays:
        return weights_from_arrays(arrays)
