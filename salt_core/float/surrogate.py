"""ATan surrogate gradient,對照 spikingjelly 原始碼公式重刻的 JAX 版本
(spikingjelly `activation_based/surrogate.py` 的 `ATan` / `atan_backward`,
公式:forward heaviside、backward 用 alpha/2 / (1 + (pi/2 * alpha * x)^2))。

forward 是精確的 heaviside(x>=0 回傳 1,否則 0),跟現有 float/scan.py 的硬
判斷完全一樣;backward 用平滑 arctan 函式的解析導數近似,讓梯度能穿過這個
原本不可微分的判斷點。這是 docs/math/單狀態仿射平行掃描推導.md 第 4
節提到「可以直接沿用 spikingjelly 式 surrogate gradient」的具體實作——留在
JAX、不是 PyTorch,不能直接呼叫 spikingjelly 的版本,公式對照原始碼重刻。
"""
from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def atan_spike(x, alpha):
    """forward:精確 heaviside,x>=0 回傳 1,否則 0。"""
    return jnp.where(x >= 0, 1.0, 0.0).astype(x.dtype)


def _atan_spike_fwd(x, alpha):
    return atan_spike(x, alpha), x


def _atan_spike_bwd(alpha, x, g):
    a = alpha / 2.0
    ax = jnp.pi * a * x
    grad_x = a / (1.0 + ax * ax) * g
    return (grad_x,)


atan_spike.defvjp(_atan_spike_fwd, _atan_spike_bwd)


def atan_smooth(x, alpha):
    """atan_spike 的 backward 公式所對應的平滑原函數,只用來驗證 backward
    公式刻得對不對(比對 jax.grad(atan_smooth) 跟 atan_spike 的 custom_vjp
    backward 是否一致),不是訓練要用的東西。"""
    return (1.0 / jnp.pi) * jnp.arctan((jnp.pi / 2.0) * alpha * x) + 0.5
