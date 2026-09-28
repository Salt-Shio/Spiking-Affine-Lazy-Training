"""層物件:把「建佇列 → 跑一層 → 吐標準事件流」這條鏈收進一個自足的東西。

跟前三層(神經元模擬器 / 佇列建構器 / 標準事件流)的關係:

- 神經元行為(`float.scan.run_layer_forward`)、佇列建構(`connectivity/`)、
  標準事件流(`layer_chain.EventStream` + `extract_output_events_fc*`)都不動,
  這個檔案只是把它們按「一種 layer 型別」串起來,對外只露兩個約定:
  **讀一條 `EventStream`、吐一份 `LayerOutput`(輸出流、結果、診斷、軌跡)**。
  數值段跟掃描交給 backend(`salt_core.backend`、`salt_core.quant.backend`)。
- 壓縮版的內部記帳(`local_to_global_j` 查表)留在 `ConvLayer.forward` 裡自己
  清掉,呼叫端看不到。conv 的「扁平神經元編號 → (x,y,c)」也在 `ConvLayer`
  內用自己的 `h_in`/`w_in` 還原,不外洩成呼叫端的一步。

**靜態 vs 會被微分的切分**(JAX 函數式):

- 層物件本身 = frozen dataclass,只有 Python 純量欄位(幾何 / 門檻 / chunk /
  容量 / init 尺度),可雜湊 → 能當 `jax.jit` 靜態參數 / 被閉包捕捉。
  **不含任何 JAX 陣列**,不是 pytree,不會被 trace。
- 權重 = 一個 pytree(一層一份陣列),由 `run_network` 的 `weights` 參數獨立
  傳遞,是唯一餵給 `jax.grad` 的東西。
- 容量旋鈕動態放大縮小 = `with_capacity` 換出一個新的靜態層物件(觸發一次
  重編譯),公式見 `salt_core.capacity`。
"""
from salt_core.layers.base import Layer, LayerOutput, uniform_init
from salt_core.layers.conv import ConvLayer
from salt_core.layers.fc import FCLayer

__all__ = ["ConvLayer", "FCLayer", "Layer", "LayerOutput", "uniform_init"]
