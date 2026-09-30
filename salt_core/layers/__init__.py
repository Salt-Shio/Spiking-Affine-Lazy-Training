"""層物件:建佇列、跑一層、輸出事件流,收在同一個物件裡。

層是 frozen dataclass,只有 Python 純量欄位,可以當 jax.jit 的靜態參數;權重另外傳,是唯一
被微分的東西。容量放大縮小用 with_capacity 換一個新的層物件,會觸發重新編譯。
conv、fc 建的是少了輸入面欄位的描述,交給 Network.build 接形狀。
"""
from salt_core.layers.base import Layer, LayerOutput, LayerSpec, uniform_init
from salt_core.layers.conv import ConvLayer, ConvSpec, conv
from salt_core.layers.fc import FCLayer, FCSpec, fc

__all__ = ["ConvLayer", "ConvSpec", "FCLayer", "FCSpec", "Layer", "LayerOutput", "LayerSpec",
           "conv", "fc", "uniform_init"]
