"""backend 介面:一層 forward 裡「算數值段 + 掃描」這一段怎麼做。層負責建結構、抽輸出流、
診斷,中間交給 backend。浮點版在 salt_core/float/backend.py,整數版在
salt_core/quant/backend.py。

層跟 backend 之間的約定都寫在這裡:層呼叫 Backend.scan、讀 ScanOutput;backend 只透過
ScanLayer 讀層。兩邊都只依賴這個模組,backend 不 import layers。
"""
from typing import NamedTuple, Protocol

import jax

from salt_core.connectivity.conv import ConvQueueStructure
from salt_core.connectivity.fc import FCQueueStructure
from salt_core.float.affine import AffineMap
from salt_core.float.scan import FloatLayerResult
from salt_core.quant.params import QuantizedLayerParams
from salt_core.quant.scan import QuantLayerResult

# 一層的佇列結構:層自己建,backend 只轉交回層。
QueueStructure = ConvQueueStructure | FCQueueStructure
# 一層的參數:浮點 backend 是權重陣列,整數 backend 是 QuantizedLayerParams。
LayerParams = jax.Array | QuantizedLayerParams
# 一層掃描的結果:浮點 backend 是 FloatLayerResult,整數 backend 是 QuantLayerResult。
LayerResult = FloatLayerResult | QuantLayerResult


class ScanOutput(NamedTuple):
    """backend.scan 的回傳。"""
    result: LayerResult
    spike_gain: jax.Array         # (n_neurons, 步數),下一層的 event_gain
    extra_steps_needed: jax.Array  # int32 純量,這筆樣本比基本步數多要的掃描步數;整數版不用步數上限,是 0
    v_steps: jax.Array | None     # (n_neurons, 步數) 每步結束的膜電位,trace=True 才有
    pointer_steps: jax.Array | None  # (n_neurons, 步數) 每步開始時的佇列欄位,trace=True 才有


class ScanLayer(Protocol):
    """backend 從層讀的東西:掃描設定跟佇列的數值段。只當文件用,實際靠 duck typing。"""
    v_th: float           # 以下三個給浮點 backend 的掃描用
    alpha: float
    chunk_size: int

    def float_values(self, structure: QueueStructure, w: jax.Array,
                     event_gain: jax.Array | None) -> AffineMap:
        """浮點數值段,a、b 形狀 (n_neurons, 佇列長度)。"""
        ...

    def gather_weight_codes(self, structure: QueueStructure, q: jax.Array) -> jax.Array:
        """整數數值段:每欄的整數權重碼,int32,(n_neurons, 佇列長度),非真事件是 0。"""
        ...

    def neuron_delta_t(self, structure: QueueStructure) -> jax.Array:
        """逐神經元的 Δt,(n_neurons, 佇列長度)。"""
        ...

    def neuron_n_real(self, structure: QueueStructure) -> jax.Array:
        """逐神經元的真事件數,(n_neurons,)。"""
        ...

    def scan_steps(self, structure: QueueStructure) -> int:
        """浮點掃描的步數上限。"""
        ...


class Backend(Protocol):
    """backend 的約定。只當文件用,實際靠 duck typing。"""

    def scan(self, layer: ScanLayer, structure: QueueStructure, params: LayerParams,
             event_gain: jax.Array | None, *, trace: bool) -> ScanOutput:
        """一層的數值段 + 掃描。"""
        ...

    def readout(self, result: LayerResult, params: LayerParams) -> LayerResult:
        """最後一層結果換成解碼器要的尺度。"""
        ...
