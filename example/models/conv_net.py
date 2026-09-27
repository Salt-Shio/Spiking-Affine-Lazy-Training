"""從訓練 yaml 組出一個壓縮版 conv SNN。

實際運算全部在 `salt_core`(`ConvLayer` / `FCLayer` / `Network`)。
這個檔案是 `src`(開發者)這端的組裝碼,做三件事:

- `build_network(model_cfg)`:讀 model config 的 `input_shape` + `layers`
  (一列 layer entry),照 `salt_core` 的 layer 約定把層物件串出來,包成 `Network`。`ic` 串接、
  空間尺寸的往下傳(`h_out` / `w_out` 是 `ConvLayer` 自己的 property)、FC 的
  `n_in` 全部由這裡推導,yaml 不必填。**網路形狀完全由 config 決定,這個
  檔案沒有寫死的幾何。**
- `build_decoder(model_cfg, layers)`:從 model config 的 `decoder` 建輸出
  解碼器(膜電位回歸 / 頻率 / 群體),把最後一層的 `LayerForwardResult` 讀成
  預測分數。網路本身不挑 readout,見 `salt_core.decoder`。
- `build_growth_policies(model_cfg, layers)`:同一份 layer entry 裡的容量放大縮小
  倍率、門檻,建成每層的 `GrowthPolicy`。

佇列長度 `L`、輸出 spike 上界 `max_out_spikes` 都是「起始猜測 + 訓練中偵測
出界就放大」(見 docs/math/conv事件佇列壓縮版推導.md 第 7.2 節、
`salt_core.capacity`、`example/train_conv_compressed.py`)。放大縮小的倍率跟門檻
寫在同一個 layer entry 裡,由 `build_growth_policies` 讀。
`init_k` 是每層必填欄位,不校準(委定值見 docs/問題紀錄.md §12),layer entry
沒填會在建層物件那一步直接報錯。
"""
import dataclasses

from salt_core.capacity import GrowthPolicy
from salt_core.decoder import (MembraneRegressionDecoder, PopulationDecoder,
                                RateDecoder)
from salt_core.layers import ConvLayer, FCLayer
from salt_core.network import Network

from data.src.nmnist import CLASS_NAMES

# 類別數是資料集的事實,不是網路形狀。單一來源取自 data.src.nmnist.CLASS_NAMES,
# 給 build_decoder 的群體分組用。
N_CLASSES = len(CLASS_NAMES)

__all__ = ["N_CLASSES", "build_network", "build_growth_policies", "build_decoder"]

# layer entry 裡這些 key 轉型後才傳給層類別(yaml 的 `1.0e9` 之類會被 parse
# 成字串——PyYAML 遵 YAML 1.1,指數要 `1.0e+9` 才算 float)。不認得的 key
# 原樣傳,讓層類別自己 TypeError。
_LAYER_INT_FIELDS = ("chunk_size", "L", "max_out_spikes", "max_steps")
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
    """`cfg["model"]` 這個 dict -> `Network`(輸入網格 + 一列 layer 物件)。

    需要:
      - `input_shape`: `[C, H, W]`,虛擬輸入網格。
      - `layers`: 一列 entry,每個 `{type: conv|fc, ...}`。
        - `conv` 必填 `oc` / `k` / `s` / `p`;空間尺寸由這裡算、`ic` 從上一層
          串。
        - `fc` 必填 `n_out`;`n_in` = 上一層攤平。
        - `GrowthPolicy` 的欄位(`L_grow_factor` 這些)留給 `build_growth_policies`。
        - 其餘 key(`tau` / `v_th` / `alpha` / `chunk_size` / `L` /
          `max_out_spikes` / `max_steps` / `init_k`)直接當關鍵字傳給層類別,
          沒填就吃類別預設(見 `salt_core.layers`)。
        - `name` 選填,沒填自動 `conv1` / `conv2` / ... / `fc1` / ...。

    config 的格式 / key 命名 / 版本管理是開發者的事:entry 少了必填 key、或
    多了層類別不認得的 key,都會在這裡直接炸出來,不做寬容處理。
    """
    try:
        c, h, w = (int(v) for v in model_cfg["input_shape"])
    except (KeyError, TypeError, ValueError) as err:
        raise ValueError("model config 的 input_shape 要是 [C, H, W] 三個整數") from err

    layers = []
    counts = {"conv": 0, "fc": 0}
    for i, entry in enumerate(model_cfg["layers"]):
        e = dict(entry)
        etype = e.pop("type", None)
        name = e.pop("name", None)
        if etype == "conv":
            k, s, p, oc = (int(e.pop(key)) for key in ("k", "s", "p", "oc"))
            counts["conv"] += 1
            layer = ConvLayer(
                name=name or f"conv{counts['conv']}",
                ic=c, h_in=h, w_in=w, oc=oc, k=k, s=s, p=p,
                **_coerce_layer_opts(e))
            # h_out / w_out 是 ConvLayer 自己算的 property(見 salt_core.layers)。
            if layer.h_out < 1 or layer.w_out < 1:
                raise ValueError(
                    f"layers[{i}] 幾何退化:輸入 {h}x{w}、k={k} s={s} p={p} -> "
                    f"輸出 {layer.h_out}x{layer.w_out}")
            layers.append(layer)
            c, h, w = oc, layer.h_out, layer.w_out
        elif etype == "fc":
            n_out = int(e.pop("n_out"))
            counts["fc"] += 1
            layers.append(FCLayer(
                name=name or f"fc{counts['fc']}",
                n_in=c * h * w, n_out=n_out, **_coerce_layer_opts(e)))
        else:
            raise ValueError(
                f"layers[{i}]:未知的 type {etype!r}(可用:conv / fc)")

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
    """從 model config 的 `decoder` 建輸出解碼器(見 `salt_core.decoder`)。
    三選一:`membrane_regression`(預設,不填也是它)/ `rate` / `population`。
    `population` 用最後一層的 `n_out`(必須整除 `N_CLASSES`),輸出神經元
    連續等分成 `N_CLASSES` 組。
    """
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
