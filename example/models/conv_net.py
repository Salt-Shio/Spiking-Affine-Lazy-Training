"""從訓練 yaml 組出一個壓縮版 conv SNN。

實際運算全部在 `salt_core.layers`(`ConvLayer` / `FCLayer` / `run_network`)。
這個檔案是 `src`(開發者)這端的組裝碼,做三件事:

- `build_network(model_cfg)`:讀 model config 的 `input_shape` + `layers`
  (一列 layer entry),照 `salt_core` 的 layer 約定把層物件串出來。`ic` 串接、
  空間尺寸的往下傳(`h_out` / `w_out` 是 `ConvLayer` 自己的 property)、FC 的
  `n_in` 全部由這裡推導,yaml 不必填。**網路形狀完全由 config 決定,這個
  檔案沒有寫死的幾何。**
- `build_decoder(model_cfg, layers)`:從 model config 的 `decoder` 建輸出
  解碼器(膜電位回歸 / 頻率 / 群體),把最後一層的 `LayerForwardResult` 讀成
  預測分數。網路本身不挑 readout,見 `salt_core.decoder`。
- `ConvNetCompressed`:把資料端原生的 `(event_times, x, y, c, n_real)` 包成
  標準事件流餵進去、回傳 `(最後一層 LayerForwardResult, [LayerDiag, ...])`;
  `init` 逐層生權重(回一個對齊 layer list 的 weight tuple)。

佇列長度 `L`、輸出 spike 上界 `max_out_spikes` 都是「起始猜測 + 訓練中偵測
出界就放大」(見 docs/math/conv事件佇列壓縮版推導.md 第 7.2 節、
`salt_core.layers.ConvLayer.grown_to_fit`、`example/train_conv_compressed.py`)。
init_k 校準見 `salt_core.calibrate`,layer entry 沒填 `init_k` 就開訓前現算。
"""
import jax

from salt_core.decoder import (MembraneRegressionDecoder, PopulationDecoder,
                                RateDecoder)
from salt_core.layers import ConvLayer, FCLayer, raw_events_to_stream, run_network

from data.src.nmnist import CLASS_NAMES

# 類別數是資料集的事實,不是網路形狀。單一來源取自 data.src.nmnist.CLASS_NAMES,
# 給 build_decoder 的群體分組用。
N_CLASSES = len(CLASS_NAMES)

__all__ = ["N_CLASSES", "build_network", "build_decoder", "ConvNetCompressed"]

# layer entry 裡這些 key 轉型後才傳給層類別(yaml 的 `1.0e9` 之類會被 parse
# 成字串——PyYAML 遵 YAML 1.1,指數要 `1.0e+9` 才算 float)。不認得的 key
# 原樣傳,讓層類別自己 TypeError。
_LAYER_INT_FIELDS = ("chunk_size", "L", "max_out_spikes")
_LAYER_FLOAT_FIELDS = ("tau", "v_th", "alpha", "init_k", "L_grow_factor", "out_grow_factor")


def _coerce_layer_opts(e: dict) -> dict:
    out = {}
    for key, val in e.items():
        if val is None:
            out[key] = None
        elif key in _LAYER_INT_FIELDS:
            out[key] = int(val)
        elif key in _LAYER_FLOAT_FIELDS:
            out[key] = float(val)
        else:
            out[key] = val
    return out


def build_network(model_cfg: dict) -> list:
    """`cfg["model"]` 這個 dict -> 一列 layer 物件。

    需要:
      - `input_shape`: `[C, H, W]`,虛擬輸入網格。
      - `layers`: 一列 entry,每個 `{type: conv|fc, ...}`。
        - `conv` 必填 `oc` / `k` / `s` / `p`;空間尺寸由這裡算、`ic` 從上一層
          串。
        - `fc` 必填 `n_out`;`n_in` = 上一層攤平。
        - 其餘 key(`tau` / `v_th` / `alpha` / `chunk_size` / `L` /
          `max_out_spikes` / `init_k` / `L_grow_factor` / `out_grow_factor`)
          直接當關鍵字傳給層類別,沒填就吃類別預設(見 `salt_core.layers`)。
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
    return layers


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


class ConvNetCompressed:
    """一列 layer 串成的壓縮版 conv SNN。實際運算在 `salt_core.layers.run_network`,
    這個 class 是「資料端事件格式 -> 標準事件流」的薄殼。

    forward:把原始事件包成標準事件流 -> `run_network` 一列層跑完 -> 回傳
    **最後一層的原始 `LayerForwardResult`**(不自己挑 v_final;怎麼把它讀成
    預測分數是解碼器的事,見 `salt_core.decoder` / `build_decoder`),連同
    每層 `LayerDiag`(spike 數 / firing rate / L 出界訊號 / 輸出上界出界訊號)。
    動態放大 = 用 `grown_to_fit` 重建 layer list 再 `ConvNetCompressed(新
    layers)`,見 train_conv_compressed.py。

    權重:`init` 回一個對齊 `self.layers` 的 tuple(一層一份陣列),就是餵給
    `run_network` / `jax.grad` 的東西——沒有寫死欄位的 NamedTuple。
    """

    def __init__(self, layers: list):
        self.layers = layers

    def init(self, key: jax.Array) -> tuple:
        keys = jax.random.split(key, len(self.layers))
        return tuple(layer.init_weight(k) for layer, k in zip(self.layers, keys))

    def apply(self, params: tuple, event_times: jax.Array, x: jax.Array,
              y: jax.Array, c: jax.Array, n_real_events: jax.Array):
        """單一樣本 forward。回傳 (最後一層 `LayerForwardResult`,
        [每層 LayerDiag])。分數 / logits 由呼叫端的解碼器從 `LayerForwardResult`
        讀出(見 `build_decoder`)。"""
        first = self.layers[0]
        in_stream = raw_events_to_stream(event_times, x, y, c, n_real_events,
                                          h_in=first.h_in, w_in=first.w_in)
        result, diags = run_network(self.layers, in_stream, params)
        return result, diags

    def apply_batched(self, params: tuple, batch_event_times: jax.Array,
                       batch_x: jax.Array, batch_y: jax.Array, batch_c: jax.Array,
                       batch_n_real_events: jax.Array):
        """對一批樣本 vmap `apply`。回傳 (批次化的 `LayerForwardResult`,
        [批次化的 LayerDiag, ...])。"""
        return jax.vmap(self.apply, in_axes=(None, 0, 0, 0, 0, 0))(
            params, batch_event_times, batch_x, batch_y, batch_c, batch_n_real_events)
