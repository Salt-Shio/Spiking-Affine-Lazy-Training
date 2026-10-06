"""量化 conv 層匯出成 SALT-FPGA RTL 用 $readmemh 讀的 hex 文字檔:權重寬字、逐 output channel 門檻、衰減表。

格式規格只寫在 SALT-FPGA 的 docs/SNN/Concept/Layer-RTL/Conv.md,各段的「檔案格式」。
"""
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from salt_core.layers import ConvLayer
from salt_core.quant.params import QuantizedLayerParams


def hex_line(value: int, width: int) -> str:
    """width 位元無號值寫成一行 hex,固定 ceil(width/4) 個字元,左邊補 0。

    width 小於 1、或 value 不在 0 ~ 2**width - 1 時 raise ValueError。
    """
    if width < 1:
        raise ValueError(f"位元寬要至少 1,拿到 {width}")
    if not 0 <= value < (1 << width):
        raise ValueError(f"{value} 超出 {width} 位元無號範圍")
    return f"{value:0{(width + 3) // 4}x}"


def _checked_signed(values: np.ndarray, width: int, what: str) -> np.ndarray:
    """values 超出 width 位元二補數範圍、或 width 小於 1 時 raise ValueError,否則轉成 python int 的 object 陣列。"""
    if width < 1:
        raise ValueError(f"{what}的位元寬要至少 1,拿到 {width}")
    values = np.asarray(values)
    low, high = -(1 << (width - 1)), (1 << (width - 1)) - 1
    if values.size and (values.min() < low or values.max() > high):
        raise ValueError(f"{what} 範圍 {values.min()} ~ {values.max()} 超出 {width} 位元二補數 "
                         f"{low} ~ {high}")
    return values.astype(object)


def twos_complement(value: int, width: int) -> int:
    """有號整數 -> width 位元二補數的無號值。範圍檢查由呼叫端做。"""
    return int(value) & ((1 << width) - 1)


def conv_weight_lines(q: np.ndarray, weight_width: int) -> list[str]:
    """conv 權重碼 -> 權重檔的每一行,一行一個位址。

    q: (OC, C, K, K) 整數權重碼,q[o_c, c, k_y, k_x]。
    位址 o_c*C + c;寬字 K*K*weight_width 位元,tap k_y*K + k_x 從第 (k_y*K + k_x)*weight_width 位元起。
    q 不是 4 維、最後兩維不相等、或有值超出 weight_width 位元二補數範圍時 raise ValueError。
    """
    q = np.asarray(q)
    if q.ndim != 4 or q.shape[2] != q.shape[3]:
        raise ValueError(f"q 要是 (OC, C, K, K),拿到形狀 {q.shape}")
    out_channels, in_channels, kernel_size, _ = q.shape
    codes = _checked_signed(q, weight_width, "權重碼")
    taps = codes.reshape(out_channels * in_channels, kernel_size * kernel_size)
    word_width = kernel_size * kernel_size * weight_width
    lines = []
    for kernel in taps:
        word = 0
        for tap, code in enumerate(kernel):
            word |= twos_complement(code, weight_width) << (tap * weight_width)
        lines.append(hex_line(word, word_width))
    return lines


def channel_thresholds(v_th_int: np.ndarray, out_channels: int) -> np.ndarray:
    """逐神經元門檻 -> 逐 output channel 門檻,形狀 (out_channels,)。

    v_th_int: 純量,或 (out_channels * H_out * W_out,),神經元 (o_c, o_y, o_x) 的編號
        o_c*H_out*W_out + o_y*W_out + o_x。
    長度不是 out_channels 的倍數、或同一個 output channel 的值不全相同時 raise ValueError。
    """
    v_th_int = np.asarray(v_th_int)
    if v_th_int.ndim == 0:
        return np.full(out_channels, v_th_int.item())
    if v_th_int.ndim != 1 or v_th_int.size % out_channels != 0:
        raise ValueError(f"v_th_int 形狀 {v_th_int.shape} 不能分成 {out_channels} 個 output channel")
    per_channel = v_th_int.reshape(out_channels, -1)
    mismatched = np.flatnonzero((per_channel != per_channel[:, :1]).any(axis=1))
    if mismatched.size:
        raise ValueError(f"output channel {mismatched.tolist()} 的門檻不全相同")
    return per_channel[:, 0]


def threshold_lines(thresholds: Sequence[int] | np.ndarray, membrane_width: int) -> list[str]:
    """逐 output channel 門檻 -> 門檻檔的每一行,membrane_width 位元二補數。

    有值超出 membrane_width 位元二補數範圍時 raise ValueError。
    """
    values = _checked_signed(thresholds, membrane_width, "門檻")
    return [hex_line(twos_complement(v, membrane_width), membrane_width) for v in values]


def decay_lines(decay_table_int: Sequence[int] | np.ndarray, decay_frac_width: int) -> list[str]:
    """衰減表 -> 衰減檔的每一行,第 dt-1 行是 A[dt],decay_frac_width 位元無號。

    有值超出 0 ~ 2**decay_frac_width - 1 時 raise ValueError。
    """
    return [hex_line(int(v), decay_frac_width) for v in np.asarray(decay_table_int)]


def write_mem(path: str | os.PathLike, lines: Sequence[str]) -> None:
    """一行一個值寫成文字檔,換行用 \\n。"""
    Path(path).write_text("".join(f"{line}\n" for line in lines), encoding="ascii")


def write_conv_layer_mem(out_dir: str | os.PathLike, layer: ConvLayer,
                         params: QuantizedLayerParams, weight_width: int) -> dict[str, Path]:
    """一層 conv 的三個檔寫進 out_dir:<層名>_weight.mem、<層名>_threshold.mem、<層名>_decay.mem。

    門檻寬度 i_V + f_V,衰減碼寬度 f_a,取自 params。回傳 {"weight" | "threshold" | "decay": 路徑}。
    q 形狀不是 (oc, ic, k, k)、v_th_int 是 None(這層不 fire)或長度不是神經元數時 raise ValueError;
    其餘檢查見各 *_lines。
    """
    q = np.asarray(params.q)
    if q.shape != (layer.oc, layer.ic, layer.k, layer.k):
        raise ValueError(f"{layer.name} 的 q 形狀 {q.shape},層的幾何是 "
                         f"{(layer.oc, layer.ic, layer.k, layer.k)}")
    if params.v_th_int is None:
        raise ValueError(f"{layer.name} 沒有門檻,conv 層要能 fire")
    v_th_int = np.asarray(params.v_th_int)
    n_neurons = layer.oc * layer.h_out * layer.w_out
    if v_th_int.ndim != 0 and v_th_int.shape != (n_neurons,):
        raise ValueError(f"{layer.name} 的 v_th_int 形狀 {v_th_int.shape},神經元數 {n_neurons}")
    thresholds = channel_thresholds(v_th_int, layer.oc)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    contents = {
        "weight": conv_weight_lines(q, weight_width),
        "threshold": threshold_lines(thresholds, params.i_V + params.f_V),
        "decay": decay_lines(np.asarray(params.decay_table_int), params.f_a),
    }
    paths = {}
    for kind, lines in contents.items():
        paths[kind] = out_dir / f"{layer.name}_{kind}.mem"
        write_mem(paths[kind], lines)
    return paths
