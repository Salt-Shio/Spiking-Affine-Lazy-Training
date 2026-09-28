"""測試用的參考實作,不是產品程式碼。

dense_conv_affine_map:用最直白的方式建 conv 層的密集仿射映射 (n_out_neurons, n_events),
每顆神經元看全部事件,不合法的 tap 用 where 蓋成 0。不用 scatter、不壓縮佇列,跟
build_conv_structure 不共用機制,拿來當 conv 佇列建構的對照答案:
  a[neuron, j] = (1 - 1/tau) ** (t[j] - t[j-1])   所有神經元相同
  b[neuron, j] = W[oc, c[j], ky, kx] * gain[j]    j 是這顆神經元的合法 tap,否則 0
  ky = y[j] - oy*S + P,kx = x[j] - ox*S + P;0 <= ky < K 且 0 <= kx < K 才合法
  j >= n_real(pad 事件):a = 1、b = 0
finite_diff_grad:中央差分,驗不 fire(v_th=1e9)場景的梯度,不經過 autodiff。
"""
import numpy as np

import jax.numpy as jnp

from salt_core.float.affine import AffineMap


def dense_conv_affine_map(event_times, x, y, c, W, tau, S, P, H_out, W_out,
                           gain=None, n_real=None) -> AffineMap:
    """回傳 AffineMap(a, b),形狀都是 (OC*H_out*W_out, n_events)。W 可以是 numpy 或 jax 陣列,
    jax 陣列時可微。"""
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
    """純量函式 f(W) 逐元素做中央差分,回傳跟 W 同形狀的梯度。f 拿到的是 numpy 陣列。"""
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
