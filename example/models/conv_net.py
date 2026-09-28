"""從 config 的 model 區塊組出網路、解碼器、容量放大縮小的策略。

網路形狀完全由 config 決定:ic 串接、空間尺寸、FC 的 n_in 都由 build_network 推導。
容量旋鈕是起始值,訓練中出界會放大;倍率跟門檻寫在同一個 layer entry,由 build_growth_policies 讀。
"""
import dataclasses
import math

from salt_core.capacity import GrowthPolicy
from salt_core.decoder import (MembraneRegressionDecoder, PopulationDecoder,
                                RateDecoder)
from salt_core.layers import ConvLayer, FCLayer
from salt_core.network import Network

from data.src.nmnist import CLASS_NAMES

# 類別數取自資料集,給群體編碼分組用
N_CLASSES = len(CLASS_NAMES)

__all__ = ["N_CLASSES", "build_network", "build_growth_policies", "build_decoder"]

# 這些 key 轉型後才傳給層類別:PyYAML 把 1.0e9 讀成字串(要寫 1.0e+9 才是 float)。
# 不認得的 key 原樣傳,由層類別 raise TypeError。
_LAYER_INT_FIELDS = ("chunk_size", "max_queue_len", "max_out_spikes", "max_extra_steps")
_LAYER_FLOAT_FIELDS = ("tau", "v_th", "alpha", "init_k")
# layer entry 裡屬於 GrowthPolicy 的 key,不傳給層類別
_POLICY_FIELDS = tuple(f.name for f in dataclasses.fields(GrowthPolicy))


def _coerce_layer_opts(e: dict) -> dict:
    out = {}
    for key, val in e.items():
        if key in _POLICY_FIELDS:
            continue
        if val is None:
            out[key] = None
        elif key in _LAYER_INT_FIELDS:
            out[key] = int(val)
        elif key in _LAYER_FLOAT_FIELDS:
            out[key] = float(val)
        else:
            out[key] = val
    return out


def build_network(model_cfg: dict) -> Network:
    """model config -> Network(輸入網格加一列層)。

    input_shape: [C, H, W]。
    layers: 一列 entry,每個有 type(conv 或 fc)。conv 必填 oc、k、s、p;fc 必填 n_out。name 選填,
        沒填是 conv1、conv2、fc1 ...。GrowthPolicy 的 key 留給 build_growth_policies,其餘 key 傳給層類別。
    ic、空間尺寸、FC 的 n_in 由前一層推導。
    input_shape 格式不對、layers 是空的、type 不認得、conv 接在 FC 後面、幾何退化時 raise ValueError;
    少了必填 key 或多了層類別不認得的 key 時由層類別 raise。
    """
    try:
        c, h, w = (int(v) for v in model_cfg["input_shape"])
    except (KeyError, TypeError, ValueError) as err:
        raise ValueError("model config 的 input_shape 要是 [C, H, W] 三個整數") from err

    shape = (c, h, w)  # 下一層的輸入形狀:conv 之後是 (oc, h, w),FC 之後是 (n_out,)
    layers = []
    counts = {"conv": 0, "fc": 0}
    for i, entry in enumerate(model_cfg["layers"]):
        e = dict(entry)
        etype = e.pop("type", None)
        name = e.pop("name", None)
        if etype == "conv":
            if len(shape) != 3:
                raise ValueError(f"layers[{i}]:conv 要空間輸入,前一層的輸出是攤平的 {shape}")
            c, h, w = shape
            k, s, p, oc = (int(e.pop(key)) for key in ("k", "s", "p", "oc"))
            counts["conv"] += 1
            layer = ConvLayer(
                name=name or f"conv{counts['conv']}",
                ic=c, h_in=h, w_in=w, oc=oc, k=k, s=s, p=p,
                **_coerce_layer_opts(e))
            if layer.h_out < 1 or layer.w_out < 1:
                raise ValueError(
                    f"layers[{i}] 幾何退化:輸入 {h}x{w}、k={k} s={s} p={p} -> "
                    f"輸出 {layer.h_out}x{layer.w_out}")
        elif etype == "fc":
            n_out = int(e.pop("n_out"))
            counts["fc"] += 1
            layer = FCLayer(name=name or f"fc{counts['fc']}",
                            n_in=math.prod(shape), n_out=n_out, **_coerce_layer_opts(e))
        else:
            raise ValueError(
                f"layers[{i}]:未知的 type {etype!r}(可用:conv / fc)")
        layers.append(layer)
        shape = layer.output_shape

    if not layers:
        raise ValueError("model config 的 layers 是空的")
    return Network(input_shape=model_cfg["input_shape"], layers=layers)


def build_growth_policies(model_cfg: dict, layers: list) -> dict:
    """有容量的層各一個 GrowthPolicy,回傳 {層名: GrowthPolicy}。

    layers: build_network(model_cfg).layers,跟 model_cfg["layers"] 一一對齊。
    倍率、門檻讀 layer entry 裡的同名 key,沒填吃 GrowthPolicy 的預設。
    """
    return {layer.name: GrowthPolicy(**{key: float(entry[key])
                                         for key in _POLICY_FIELDS if key in entry})
            for entry, layer in zip(model_cfg["layers"], layers) if layer.capacity is not None}


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
