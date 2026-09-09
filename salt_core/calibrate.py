"""層的初始權重尺度校準(找 init_k)。

兩塊:

- `calibrate_init_scale`:準則無關的通用掃描(bracket + 幾何二分)。呼叫端給
  `make_weight(k) -> 權重` 跟 `measure(權重) -> 純量`,還有目標帶 `band`;
  這支函式只管「調 k 直到 measure 落進 band」,不知道 measure 量的是 firing
  rate 還是別的。用什麼準則、目標帶多少是呼叫端的決定(問題紀錄第十二節,
  firing-rate 準則本身可能有害,這裡刻意不焊死)。
- `calibrate_network`:一列 layer,照前向順序把 `init_k is None` 的層一層層
  解出來——每層用它自己的 `calibration_measure`(層型別自己知道怎麼量),
  解完就生權重、forward 一次餵下一層(下游層的輸入要上游解好、跑過才有)。
  是 `run_network` 的校準版鏡像。

FC 輸出層(v_th 純積分器,永遠不 fire)沒有 firing-rate 準則可用,`init_k`
必須明確給定;`calibrate_network` 碰到 FC 層還留 `None` 會直接報錯。
"""
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from salt_core.layers import raw_events_to_stream, uniform_init


class CalibrationResult(NamedTuple):
    name: str
    init_k: float
    measured: float          # 收斂點 measure 量到的值
    curve: tuple             # ((init_k, measured), ...) 依量測順序,掃不到時當診斷
    converged_reason: str    # "band"(落帶內)或 "width"(區間收斂到雜訊量級)


class CalibrationError(RuntimeError):
    """合理範圍 [lo, hi] 內找不到能讓 measure 落進目標帶的 init_k。"""


def calibrate_init_scale(make_weight: Callable[[float], jax.Array],
                          measure: Callable[[jax.Array], float],
                          band: tuple, *, name: str = "layer",
                          bracket_start: float = 1.0, bracket_factor: float = 2.0,
                          bracket_max_iter: int = 20, bisect_max_iter: int = 30,
                          bisect_rel_tol: float = 0.01, lo: float = 1e-3,
                          hi: float = 1e3) -> CalibrationResult:
    """調 init_k 直到 `measure(make_weight(init_k))` 落進 `band`。

    - bracket 起點 1.0:`limit = init_k/sqrt(fan_in)` 公式本身的自然基準
      (標準 U(-1/sqrt(fan_in), 1/sqrt(fan_in)) 尺度)。
    - bracket_factor 2.0:標準 doubling search。
    - bisect_rel_tol 0.01:次要收斂條件,防止小樣本量測噪聲讓迴圈在帶邊緣震盪。
    - lo/hi:掃描合理範圍,兩端都是「不可能進帶」的物理邊界(limit→0 幾乎
      不 fire;limit 遠超 v_th 幾乎每個事件都 fire)。

    掃不到(bracket 撞到 [lo,hi] 邊界、或 bisection 兩個收斂條件都沒達成)
    一律 raise `CalibrationError`,附完整掃描曲線,不靜默選最接近的值。
    """
    band_lo, band_hi = band
    curve: list = []

    def probe(k: float) -> float:
        v = float(measure(make_weight(k)))
        curve.append((k, v))
        return v

    k0 = bracket_start
    v0 = probe(k0)
    if band_lo <= v0 <= band_hi:
        return CalibrationResult(name, k0, v0, tuple(curve), "band")

    # bracket:measure 對 init_k 統計上單調遞增,太低就倍增、太高就減半,找一個
    # 「跨過帶」的另一端當 bisection 起始區間。每步先檢查是不是直接踩進帶內。
    increasing = v0 < band_lo
    k_prev, v_prev = k0, v0
    k_cur = k0
    bracket_hit = None  # (k_lo, v_lo, k_hi, v_hi):k_lo 的 measure < band_lo,k_hi 的 > band_hi
    for _ in range(bracket_max_iter):
        k_cur = k_cur * bracket_factor if increasing else k_cur / bracket_factor
        if k_cur < lo or k_cur > hi:
            raise CalibrationError(
                f"{name}: init_k 掃描超出合理範圍 [{lo},{hi}] 仍未跨過目標帶 "
                f"{band},掃描曲線:{curve}")
        v_cur = probe(k_cur)
        if band_lo <= v_cur <= band_hi:
            return CalibrationResult(name, k_cur, v_cur, tuple(curve), "band")
        if increasing and v_cur > band_hi:
            bracket_hit = (k_prev, v_prev, k_cur, v_cur)
            break
        if not increasing and v_cur < band_lo:
            bracket_hit = (k_cur, v_cur, k_prev, v_prev)
            break
        k_prev, v_prev = k_cur, v_cur
    else:
        raise CalibrationError(
            f"{name}: bracket 階段 {bracket_max_iter} 步內沒能跨過目標帶 "
            f"{band},掃描曲線:{curve}")

    k_lo, v_lo, k_hi, v_hi = bracket_hit

    # bisection:幾何二分(乘除而非加減——範圍跨數量級,線性二分會被大數值那端
    # 拖著跑,幾何二分才能在 log 尺度上均勻收斂)。
    for _ in range(bisect_max_iter):
        if k_hi / k_lo < 1.0 + bisect_rel_tol:
            # 次要收斂:區間已窄到量測噪聲量級,繼續二分只是在雜訊裡打轉。
            return CalibrationResult(name, k_hi, v_hi, tuple(curve), "width")
        k_mid = (k_lo * k_hi) ** 0.5
        v_mid = probe(k_mid)
        if band_lo <= v_mid <= band_hi:
            return CalibrationResult(name, k_mid, v_mid, tuple(curve), "band")
        if v_mid < band_lo:
            k_lo, v_lo = k_mid, v_mid
        else:
            k_hi, v_hi = k_mid, v_mid

    raise CalibrationError(
        f"{name}: bisection 階段 {bisect_max_iter} 步內沒能收斂(既沒落帶、區間"
        f"寬度也沒收斂到相對寬度 {bisect_rel_tol}),掃描曲線:{curve}")


def _sub_batches(n: int, chunk: int):
    for lo in range(0, n, chunk):
        yield lo, min(lo + chunk, n)


def _slice_stream(stream_batch, lo: int, hi: int):
    return type(stream_batch)(*(f[lo:hi] for f in stream_batch))


def _concat_streams(parts: list):
    cls = type(parts[0])
    return cls(*(jnp.concatenate([getattr(p, name) for p in parts], axis=0)
                 for name in cls._fields))


def _forward_batch(layer, w: jax.Array, stream_batch, chunk: int):
    """對一批輸入流跑 layer.forward,回傳批次輸出流。分批 vmap 避免整批一次
    建構壓縮佇列 OOM。"""
    def one(s):
        out, _r, _d = layer.forward(w, s)
        return out

    n = stream_batch.event_times.shape[0]
    parts = [jax.vmap(one)(_slice_stream(stream_batch, lo, hi))
             for lo, hi in _sub_batches(n, chunk)]
    return _concat_streams(parts)


def calibrate_network(layers: list, raw_calib_batch: tuple, *, key: jax.Array,
                       band: tuple = (0.20, 0.50), measure_chunk: int = 16,
                       **sweep_kw) -> list:
    """回傳 `layers` 的複本,所有 `init_k is None` 的層都填上校準值。

    `raw_calib_batch`:資料端原生的 (event_times, x, y, c, n_real_events),
    leading axis 是樣本數。照前向順序:每層若 `init_k is None` 就用它自己的
    `calibration_measure` 跑 `calibrate_init_scale`;不管有沒有校準,都用該層
    (解好的)init_k 生權重、forward 一次把輸出流餵給下一層。

    `**sweep_kw` 原封轉給 `calibrate_init_scale`(bracket_start / lo / hi 等)。
    """
    et, x, y, c, nr = raw_calib_batch
    stream_batch = jax.vmap(raw_events_to_stream, in_axes=(0, 0, 0, 0, 0, None, None))(
        et, x, y, c, nr, layers[0].h_in, layers[0].w_in)

    keys = jax.random.split(key, len(layers))
    resolved = []
    for i, (layer, k) in enumerate(zip(layers, keys)):
        if getattr(layer, "init_k", None) is None:
            if not hasattr(layer, "calibration_measure"):
                raise CalibrationError(
                    f"{layer.name}:這個 layer 型別沒有 firing-rate 準則可校準,"
                    f"init_k 必須在 config 明確給定")
            measure = layer.calibration_measure(stream_batch, measure_chunk)
            make_w = (lambda kk, _l=layer, _key=k:
                      uniform_init(_key, _l.weight_shape, _l.fan_in, kk))
            res = calibrate_init_scale(make_w, measure, band, name=layer.name, **sweep_kw)
            layer = _replace_init_k(layer, res.init_k)
            print(f"[calib] {layer.name}: init_k={res.init_k:.6g} "
                  f"measured={res.measured:.4f} ({res.converged_reason})")
        resolved.append(layer)
        if i < len(layers) - 1:
            w = layer.init_weight(k)
            stream_batch = _forward_batch(layer, w, stream_batch, measure_chunk)
    return resolved


def _replace_init_k(layer, init_k: float):
    from dataclasses import replace
    return replace(layer, init_k=float(init_k))
