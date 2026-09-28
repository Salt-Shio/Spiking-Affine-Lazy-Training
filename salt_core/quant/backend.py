"""整數 backend:模擬 FPGA 逐事件更新膜電位暫存器,推導見 docs/math/膜電位量化推導.md。"""
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from salt_core.backend import ScanOutput
from salt_core.quant.codes import apply_decay_table_int
from salt_core.quant.fixed_point import OverflowMode, RoundMode
from salt_core.quant.scan import QuantLayerResult, run_layer, run_layer_traced


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


def _check_weight_codes(q: jax.Array) -> None:
    """q 不是整數 dtype 時 raise ValueError。傳成乘回 scale 的浮點權重時,轉成整數碼
    會全部變成 0,forward 照樣跑完、不會報錯。"""
    dtype = jnp.asarray(q).dtype
    if not jnp.issubdtype(dtype, jnp.integer):
        raise ValueError(
            f"q 必須是整數權重碼(quant.codes.quantize_to_int 的 q),拿到的 dtype 是 {dtype}")


@dataclass(frozen=True)
class QuantBackend:
    """整數 backend:params 是 QuantizedLayerParams,一步處理一筆事件,全程整數。

    round_mode: 乘法後的捨入規則,整個網路共用。
    """
    round_mode: RoundMode | str = RoundMode.ROUND

    def scan(self, layer, structure, params: QuantizedLayerParams,
             event_gain: jax.Array | None, *, trace: bool) -> ScanOutput:
        """一層的查表 + 取權重碼 + 整數掃描。掃描長度是佇列長度,跟 chunk_size、
        max_extra_steps 無關。event_gain 用不到:整數路徑的跨層增益恆為 1。
        params.q 不是整數 dtype 時 raise ValueError。"""
        _check_weight_codes(params.q)
        a_int, is_identity = apply_decay_table_int(layer.neuron_delta_t(structure),
                                                   params.decay_table_int)
        q_int = layer.gather_weight_codes(structure, params.q)
        scan_args = (a_int, is_identity, q_int, params.v_th_int)
        scan_kwargs = dict(f_a=params.f_a, f_V=params.f_V, i_V=params.i_V,
                           round_mode=self.round_mode, overflow_mode=params.overflow_mode)
        if trace:
            result, v_steps = run_layer_traced(*scan_args, **scan_kwargs)
            pointer_steps = result.spike_event_idx  # 第 t 步處理佇列第 t 欄
        else:
            result = run_layer(*scan_args, **scan_kwargs)
            v_steps = pointer_steps = None
        # 沒有 surrogate gradient,跨層增益用常數 1。
        spike_gain = jnp.ones_like(result.spike_mask, dtype=jnp.float32)
        return ScanOutput(result=result, spike_gain=spike_gain,
                          extra_steps_needed=jnp.zeros((), dtype=jnp.int32),
                          v_steps=v_steps, pointer_steps=pointer_steps)

    def readout(self, result: QuantLayerResult,
                params: QuantizedLayerParams) -> QuantLayerResult:
        """v_final 從暫存器值換回物理尺度 v_int * 2**-f_V * s_c(逐神經元),其他欄位不變。

        每顆輸出神經元的 s_c 可能不同,暫存器值直接比大小會選錯類別,所以解碼前要換。
        """
        v_physical = (result.v_final.astype(jnp.float32) * 2.0 ** -params.f_V
                      * jnp.asarray(params.scale, dtype=jnp.float32))
        return result._replace(v_final=v_physical)
