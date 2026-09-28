"""ATan surrogate gradient:forward 是精確的 heaviside,backward 用 arctan 的導數近似。

公式照 spikingjelly 的 ATan(activation_based/surrogate.py):
backward = alpha/2 / (1 + (pi/2 * alpha * x) ** 2)。
為什麼可以用 surrogate gradient 見 docs/math/單狀態仿射平行掃描推導.md。
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
    """atan_spike 的 backward 對應的平滑原函數。只給測試比對 backward 公式用,訓練不用。"""
    return (1.0 / jnp.pi) * jnp.arctan((jnp.pi / 2.0) * alpha * x) + 0.5
