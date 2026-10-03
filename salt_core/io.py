"""網路描述跟權重的存讀:Network <-> dict,權重(浮點或量化)連同網路描述存成 npz。"""
import dataclasses
import json
import os
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, TypedDict

import jax
import jax.numpy as jnp
import numpy as np

from salt_core.layers import ConvLayer, FCLayer, Layer
from salt_core.network import Network
from salt_core.quant.fixed_point import OverflowMode, RoundMode
from salt_core.quant.params import QuantizedLayerParams

FORMAT_VERSION = 1
# npz 裡放網路描述的欄位;其餘欄位是權重,key 是層名
NETWORK_KEY = "__network__"
# 量化 npz 裡放 round_mode、每層位元數跟中繼資料的欄位;陣列欄位是 "<層名>/<欄位名>"
QUANT_KEY = "__quant__"
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


def _checked_layer_names(network: Network, n_per_layer: int, what: str) -> list[str]:
    """network 的層名。n_per_layer 跟層數對不上、層名重複、或層名等於 NETWORK_KEY 時 raise ValueError。"""
    names = [layer.name for layer in network.layers]
    if n_per_layer != len(names):
        raise ValueError(f"{what}有 {n_per_layer} 份,網路有 {len(names)} 層")
    if len(set(names)) != len(names) or NETWORK_KEY in names:
        raise ValueError(f"層名要不重複、而且不能是 {NETWORK_KEY!r},拿到 {names}")
    return names


def _network_json(network: Network) -> np.ndarray:
    return np.asarray(json.dumps(network_to_dict(network)))


def _read_network(arrays: Mapping[str, np.ndarray]) -> Network:
    """npz 的 NETWORK_KEY 欄位讀回 Network。沒有這個欄位時 raise ValueError。"""
    if NETWORK_KEY not in arrays:
        raise ValueError(f"沒有網路描述欄位 {NETWORK_KEY!r}")
    return network_from_dict(json.loads(str(arrays[NETWORK_KEY])))


def weights_to_arrays(network: Network, weights: Sequence[jax.Array | np.ndarray]
                      ) -> dict[str, np.ndarray]:
    """npz 的欄位:每層的權重一個陣列(key 是層名),加上 NETWORK_KEY 放網路描述的 json。

    weights: 對齊 network.layers。
    層數對不上、層名重複、或層名等於 NETWORK_KEY 時 raise ValueError。
    """
    names = _checked_layer_names(network, len(weights), "權重")
    arrays = {name: np.asarray(w) for name, w in zip(names, weights)}
    arrays[NETWORK_KEY] = _network_json(network)
    return arrays


def weights_from_arrays(arrays: Mapping[str, np.ndarray]) -> tuple[Network, tuple[jax.Array, ...]]:
    """weights_to_arrays 的反函式。回傳 (Network, 對齊 layers 的權重),權重是 JAX 陣列。

    arrays: dict 或 np.load 讀回的 npz。沒有 NETWORK_KEY 時 raise ValueError。
    """
    network = _read_network(arrays)
    return network, tuple(jnp.asarray(arrays[layer.name]) for layer in network.layers)


def save_weights(path: str | os.PathLike, network: Network,
                 weights: Sequence[jax.Array | np.ndarray]) -> None:
    """權重連同網路描述存成 npz,格式見 weights_to_arrays。"""
    np.savez(path, **weights_to_arrays(network, weights))


def load_weights(path: str | os.PathLike) -> tuple[Network, tuple[jax.Array, ...]]:
    """讀 save_weights 存的檔,回傳 (Network, 權重),權重是 JAX 陣列。"""
    with np.load(path) as arrays:
        return weights_from_arrays(arrays)


class QuantizedModel(NamedTuple):
    """load_quantized 的回傳。"""
    network: Network
    params: tuple[QuantizedLayerParams, ...]  # 對齊 network.layers
    round_mode: RoundMode                     # QuantBackend 的捨入規則
    metadata: dict[str, Any]                  # save_quantized 時呼叫端給的內容


def save_quantized(path: str | os.PathLike, network: Network,
                   params: Sequence[QuantizedLayerParams], round_mode: RoundMode | str,
                   metadata: Mapping[str, Any]) -> None:
    """量化模型存成 npz:讀回來不需要浮點權重,直接給 QuantBackend 跑。

    陣列欄位 "<層名>/q"、"<層名>/decay_table_int"、"<層名>/v_th_int"(None 時不存)、"<層名>/scale";
    NETWORK_KEY 放網路描述;QUANT_KEY 放 round_mode、每層的 f_a、f_V、i_V、overflow_mode 跟 metadata。
    params: 對齊 network.layers。metadata 要能存成 json,否則 json.dumps raise TypeError。
    層數對不上、層名重複、或層名等於 NETWORK_KEY 時 raise ValueError。
    """
    names = _checked_layer_names(network, len(params), "量化參數")
    arrays = {NETWORK_KEY: _network_json(network)}
    layers_info = {}
    for name, p in zip(names, params):
        arrays[f"{name}/q"] = np.asarray(p.q)
        arrays[f"{name}/decay_table_int"] = np.asarray(p.decay_table_int)
        if p.v_th_int is not None:
            arrays[f"{name}/v_th_int"] = np.asarray(p.v_th_int)
        arrays[f"{name}/scale"] = np.asarray(p.scale)
        layers_info[name] = {"f_a": int(p.f_a), "f_V": int(p.f_V), "i_V": int(p.i_V),
                             "overflow_mode": OverflowMode(p.overflow_mode).value}
    arrays[QUANT_KEY] = np.asarray(json.dumps({
        "format": FORMAT_VERSION, "round_mode": RoundMode(round_mode).value,
        "layers": layers_info, "metadata": dict(metadata)}))
    np.savez(path, **arrays)


def load_quantized(path: str | os.PathLike) -> QuantizedModel:
    """讀 save_quantized 存的檔,陣列是 JAX 陣列。

    沒有 NETWORK_KEY 或 QUANT_KEY、或 format 不是 FORMAT_VERSION 時 raise ValueError。
    """
    with np.load(path) as arrays:
        network = _read_network(arrays)
        if QUANT_KEY not in arrays:
            raise ValueError(f"沒有量化設定欄位 {QUANT_KEY!r}")
        quant = json.loads(str(arrays[QUANT_KEY]))
        if quant.get("format") != FORMAT_VERSION:
            raise ValueError(f"量化設定的 format 要是 {FORMAT_VERSION},拿到 {quant.get('format')!r}")
        params = []
        for layer in network.layers:
            name, info = layer.name, quant["layers"][layer.name]
            v_th_key = f"{name}/v_th_int"
            params.append(QuantizedLayerParams(
                q=jnp.asarray(arrays[f"{name}/q"]),
                decay_table_int=jnp.asarray(arrays[f"{name}/decay_table_int"]),
                v_th_int=jnp.asarray(arrays[v_th_key]) if v_th_key in arrays else None,
                scale=jnp.asarray(arrays[f"{name}/scale"]),
                f_a=info["f_a"], f_V=info["f_V"], i_V=info["i_V"],
                overflow_mode=OverflowMode(info["overflow_mode"])))
    return QuantizedModel(network=network, params=tuple(params),
                          round_mode=RoundMode(quant["round_mode"]), metadata=quant["metadata"])
