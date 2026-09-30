"""從 config 的 model 區塊組出網路、解碼器、容量放大縮小的策略。

每層 entry 直接寫 type、name、幾何(conv 的 oc、k、s、p,fc 的 n_out),其餘欄位分成 neuron、
training、capacity、growth 四組。model.layer_defaults 放四組的共用值,層裡同一組的同名 key 蓋過它。
形狀由 Network.build 往下接。容量旋鈕是起始值,訓練中出界會放大,倍率跟門檻在 growth 組。
"""
import dataclasses

from salt_core.capacity import GrowthPolicy
from salt_core.decoder import (MembraneRegressionDecoder, PopulationDecoder,
                                RateDecoder)
from salt_core.layers import LayerSpec, conv, fc
from salt_core.network import Network

from data.src.nmnist import CLASS_NAMES

# 類別數取自資料集,給群體編碼分組用
N_CLASSES = len(CLASS_NAMES)

__all__ = ["N_CLASSES", "build_network", "build_growth_policies", "build_decoder"]

_GEOMETRY_FIELDS = {"conv": ("oc", "k", "s", "p"), "fc": ("n_out",)}
# 每組認得的 key 跟轉型:PyYAML 把 1.0e9 讀成字串(要寫 1.0e+9 才是 float)。
_GROUP_FIELDS = {
    "neuron": {"tau": float, "v_th": float},
    "training": {"alpha": float, "init_k": float},
    "capacity": {"chunk_size": int, "max_queue_len": int, "max_out_spikes": int,
                 "max_extra_steps": int},
    "growth": {f.name: float for f in dataclasses.fields(GrowthPolicy)},
}
# 傳給層類別的組;growth 留給 build_growth_policies
_LAYER_GROUPS = ("neuron", "training", "capacity")


def _merged_groups(defaults: dict, entry: dict, where: str) -> dict[str, dict]:
    """layer_defaults 跟一層 entry 的四組合併(層的值優先)、轉型。組裡有不認得的 key 時 raise ValueError。"""
    groups = {}
    for group, fields in _GROUP_FIELDS.items():
        merged = {**(defaults.get(group) or {}), **(entry.get(group) or {})}
        unknown = sorted(set(merged) - set(fields))
        if unknown:
            raise ValueError(f"{where} 的 {group} 有不認得的 key:{unknown},認得的是 {sorted(fields)}")
        groups[group] = {key: None if val is None else fields[key](val)
                         for key, val in merged.items()}
    return groups


def _layer_groups(model_cfg: dict) -> list[dict[str, dict]]:
    """每層合併過 layer_defaults 的四組,對齊 model_cfg["layers"]。
    layer_defaults 有四組以外的 key 時 raise ValueError。"""
    defaults = model_cfg.get("layer_defaults") or {}
    unknown = sorted(set(defaults) - set(_GROUP_FIELDS))
    if unknown:
        raise ValueError(f"layer_defaults 只能放 {sorted(_GROUP_FIELDS)},拿到 {unknown}")
    return [_merged_groups(defaults, entry, f"layers[{i}]")
            for i, entry in enumerate(model_cfg["layers"])]


def _layer_spec(entry: dict, groups: dict[str, dict], where: str) -> LayerSpec:
    """一層 entry -> 層描述。type 不認得、少了幾何 key、有不認得的 key 時 raise ValueError。"""
    etype = entry.get("type")
    if etype not in _GEOMETRY_FIELDS:
        raise ValueError(f"{where}:未知的 type {etype!r}(可用:{list(_GEOMETRY_FIELDS)})")
    geometry_fields = _GEOMETRY_FIELDS[etype]
    unknown = sorted(set(entry) - {"type", "name", *geometry_fields, *_GROUP_FIELDS})
    if unknown:
        raise ValueError(f"{where} 有不認得的 key:{unknown};幾何以外的欄位要放在 "
                         f"{list(_GROUP_FIELDS)} 其中一組")
    missing = [key for key in geometry_fields if key not in entry]
    if missing:
        raise ValueError(f"{where}:{etype} 少了 {missing}")
    geometry = {key: int(entry[key]) for key in geometry_fields}
    options = {key: val for group in _LAYER_GROUPS for key, val in groups[group].items()}
    if etype == "conv":
        return conv(**geometry, name=entry.get("name"), **options)
    return fc(**geometry, name=entry.get("name"), **options)


def build_network(model_cfg: dict) -> Network:
    """model config -> Network。

    input_shape: 網路的輸入形狀,例如 [C, H, W]。
    layers: 一列 entry,格式見模組說明。name 選填,沒填是 conv1、conv2、fc1 ...。
    input_shape 不是整數列、entry 格式不對、layers 是空的、conv 接在 FC 後面、幾何退化時
    raise ValueError;少了層類別的必填欄位(例如 init_k)時由層類別 raise TypeError。
    """
    try:
        input_shape = [int(v) for v in model_cfg["input_shape"]]
    except (KeyError, TypeError, ValueError) as err:
        raise ValueError("model config 的 input_shape 要是一列整數") from err
    specs = [_layer_spec(entry, groups, f"layers[{i}]")
             for i, (entry, groups) in enumerate(zip(model_cfg["layers"],
                                                     _layer_groups(model_cfg)))]
    return Network.build(input_shape, specs)


def build_growth_policies(model_cfg: dict, layers: list) -> dict:
    """有容量的層各一個 GrowthPolicy,回傳 {層名: GrowthPolicy}。

    layers: build_network(model_cfg).layers,跟 model_cfg["layers"] 一一對齊。
    倍率、門檻讀合併過 layer_defaults 的 growth 組,沒填吃 GrowthPolicy 的預設。
    """
    return {layer.name: GrowthPolicy(**groups["growth"])
            for groups, layer in zip(_layer_groups(model_cfg), layers)
            if layer.capacity is not None}


def build_decoder(model_cfg: dict, layers: list):
    """model config 的 decoder 建解碼器:membrane_regression(預設)、rate、population。
    population 把最後一層的 n_out 依序分成 N_CLASSES 組。"""
    kind = model_cfg.get("decoder", "membrane_regression")
    if kind == "membrane_regression":
        return MembraneRegressionDecoder()
    if kind == "rate":
        return RateDecoder()
    if kind == "population":
        n_out = layers[-1].n_out
        if n_out % N_CLASSES != 0:
            raise ValueError(
                f"population 編碼:最後一層 n_out={n_out} 必須整除 N_CLASSES={N_CLASSES}")
        return PopulationDecoder(n_classes=N_CLASSES, group_size=n_out // N_CLASSES)
    raise ValueError(
        f"未知的 decoder 種類:{kind!r}"
        f"(可用:membrane_regression / rate / population)")
