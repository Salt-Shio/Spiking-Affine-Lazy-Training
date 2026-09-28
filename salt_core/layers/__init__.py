"""層物件:建佇列、跑一層、輸出事件流,收在同一個物件裡。

層是 frozen dataclass,只有 Python 純量欄位,可以當 jax.jit 的靜態參數;權重另外傳,是唯一
被微分的東西。容量放大縮小用 with_capacity 換一個新的層物件,會觸發重新編譯。
"""
from salt_core.layers.base import Layer, LayerOutput, uniform_init
from salt_core.layers.conv import ConvLayer
from salt_core.layers.fc import FCLayer

__all__ = ["ConvLayer", "FCLayer", "Layer", "LayerOutput", "uniform_init"]
