"""測試專用的透明參考實作,**不是**產品程式碼(不掛在 salt_core 的公開介面上)。

`dense_conv_affine_map`:用最直白的方式建出 conv 層的密集仿射映射陣列
`(n_out_neurons, n_events)`——衰減 `a` 全域算一次;`b` 逐輸出 channel,在整個
「事件 x 空間輸出格」上算出每個 (事件, 輸出位置) 的 kernel tap 是否合法、
對應哪個權重,不合法的用 `jnp.where` 蓋成 0。**不用 scatter、不用 gather 的
`mode='drop'`、不用候選壓縮**,跟 `build_conv_structure` 完全不共用機制
(連「怎麼避免越界 index」都不一樣:這裡是 clip + where,壓縮版是 scatter
mode='drop')。

用途:conv 佇列建構(`build_conv_structure` + `conv_float_values`)的等價測試拿這個當 ground truth(取代原本
拿密集版 `build_conv_queue` 當對照的做法——密集版已於 step 4d 移除)。
純仿射(v_th=1e9)場景的梯度驗算用 `finite_diff_grad`(對這個參考 forward 做
中央差分),概念上等同「用有限差分驗 autodiff」,完全繞開 XLA autodiff;
少數需要真的 fire 的場景才退回 autodiff-vs-autodiff(參考仍然可微)。

跟舊密集版 `build_conv_queue` 的語意對齊(docs/math/conv事件佇列建構推導.md):
  a[neuron, j] = (1 - 1/tau) ** (t[j] - t[j-1])         全域,所有神經元相同
  b[neuron, j] = W[oc, c[j], ky, kx] * gain[j]           j 是該神經元合法 tap
               = 0                                         否則
  ky = y[j] - oy*S + P,  kx = x[j] - ox*S + P;合法 iff 0<=ky<K 且 0<=kx<K
  j >= n_real(pad 事件):a = 1, b = 0
"""
import numpy as np

import jax.numpy as jnp

from salt_core.core import AffineMap


def dense_conv_affine_map(event_times, x, y, c, W, tau, S, P, H_out, W_out,
                           gain=None, n_real=None) -> AffineMap:
    """回傳 AffineMap(a, b),shape 都是 (OC*H_out*W_out, n_events)。`W` 可以是
    numpy 或 jax 陣列;傳 traced jax 陣列時整條路徑可微。"""
    et = jnp.asarray(event_times, dtype=jnp.float32)
    x = jnp.asarray(x, dtype=jnp.int32)
    y = jnp.asarray(y, dtype=jnp.int32)
    c = jnp.asarray(c, dtype=jnp.int32)
    W = jnp.asarray(W, dtype=jnp.float32)
    OC, IC, K, K2 = W.shape
    assert K == K2, f"kernel 必須方形,拿到 {W.shape}"
    n_events = int(et.shape[0])
    gain = (jnp.ones(n_events, dtype=jnp.float32) if gain is None
            else jnp.broadcast_to(jnp.asarray(gain, dtype=jnp.float32), (n_events,)))
    n_real = n_events if n_real is None else int(n_real)

    oy_axis = jnp.arange(H_out)
    ox_axis = jnp.arange(W_out)

    # a:全域衰減 broadcast 給每一列;pad 事件(j >= n_real)蓋成 identity。
    n_ms = jnp.diff(et, prepend=jnp.zeros(1, dtype=et.dtype))
    decay = (1.0 - 1.0 / tau) ** n_ms
    if n_real < n_events:
        decay = decay.at[n_real:].set(1.0)
    a = jnp.broadcast_to(decay, (OC * H_out * W_out, n_events))

    # 每個 (事件 j, 輸出 oy, ox) 的 kernel tap 位置與合法性(純幾何,broadcast)。
    ky = y[:, None] - oy_axis[None, :] * S + P          # (n_events, H_out)
    kx = x[:, None] - ox_axis[None, :] * S + P          # (n_events, W_out)
    valid = ((ky >= 0) & (ky < K))[:, :, None] & ((kx >= 0) & (kx < K))[:, None, :]
    is_real = jnp.arange(n_events)[:, None, None] < n_real
    valid = valid & is_real                              # (n_events, H_out, W_out)
    kyc = jnp.clip(ky, 0, K - 1)
    kxc = jnp.clip(kx, 0, K - 1)

    b_channels = []
    for oc in range(OC):
        wsel = W[oc][c[:, None, None], kyc[:, :, None], kxc[:, None, :]]  # (n_events, H_out, W_out)
        b_oc = jnp.where(valid, wsel * gain[:, None, None], 0.0)
        b_channels.append(b_oc.transpose(1, 2, 0).reshape(H_out * W_out, n_events))
    b = jnp.concatenate(b_channels, axis=0)

    return AffineMap(a=a, b=b)


def finite_diff_grad(f, W, eps: float = 1e-3) -> np.ndarray:
    """對純量函式 `f(W) -> float` 逐元素做中央差分,回傳跟 W 同形狀的梯度陣列。
    `f` 每次拿到一個 numpy 陣列。"""
    W = np.asarray(W, dtype=np.float64)
    g = np.zeros_like(W)
    it = np.nditer(W, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        wp = W.copy(); wp[i] += eps
        wm = W.copy(); wm[i] -= eps
        g[i] = (float(f(wp)) - float(f(wm))) / (2.0 * eps)
        it.iternext()
    return g
