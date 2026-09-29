"""整數版一層 forward 的參數 QuantizedLayerParams。由 quant.convert 產生,QuantBackend 使用。"""
from typing import NamedTuple

import jax

from salt_core.quant.fixed_point import OverflowMode


class QuantizedLayerParams(NamedTuple):
    """一層整數版 forward 要的量化設定。"""
    q: jax.Array                   # 整數權重碼(quant.codes.quantize_to_int 的 q),整數 dtype,
                                   # shape 同 layer.weight_shape
    decay_table_int: jax.Array     # quant.codes.build_decay_table_int(f_a, layer.tau)
    v_th_int: jax.Array | None     # quant.codes.v_th_to_int 的整數門檻,純量或 (n_neurons,);
                                   # None 代表這層不 fire(例如膜電位回歸的輸出層)
    scale: jax.Array               # 逐神經元的權重量化步長 s_c,純量或 (n_neurons,);
                                   # 只在 readout 換回物理尺度時用
    f_a: int                       # 衰減碼小數位元
    f_V: int                       # 暫存器小數位元
    i_V: int                       # 暫存器整數位元(含符號位)
    overflow_mode: OverflowMode | str = OverflowMode.WRAP  # 暫存器溢位處理,見 quant.fixed_point
