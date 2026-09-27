"""測試用的小網路跟合成原始事件,給容量出界相關的測試共用。"""
from dataclasses import replace

import jax
import jax.numpy as jnp

from salt_core.capacity import GrowthPolicy
from salt_core.layers import ConvLayer, FCLayer

INPUT_SHAPE = (2, 8, 8)
MAX_LEN = 40
# 給足的容量:大約是這組合成資料實際需要量的 2~3 倍
GENEROUS = {"conv1": dict(L=MAX_LEN, max_out_spikes=512, max_steps=MAX_LEN),
            "conv2": dict(L=128, max_out_spikes=256, max_steps=1536)}


def synthetic_raw_batch(key, n_samples, max_len, h_in, w_in, ic):
    """n_samples 筆隨機原始事件,每筆長度 max_len,第 i 筆的真事件數是 max_len - i % 3。

    回傳 (event_times, x, y, c, n_real_events),leading axis 是樣本數。
    """
    ks = jax.random.split(key, n_samples * 4)
    et = jnp.zeros((n_samples, max_len), dtype=jnp.float32)
    xs = jnp.zeros((n_samples, max_len), dtype=jnp.int32)
    ys = jnp.zeros((n_samples, max_len), dtype=jnp.int32)
    cs = jnp.zeros((n_samples, max_len), dtype=jnp.int32)
    nr = []
    for i in range(n_samples):
        kt, kx, ky, kc = ks[4 * i:4 * i + 4]
        n = max_len - (i % 3)
        t = jnp.sort(jax.random.uniform(kt, (n,), minval=1.0, maxval=30.0))
        et = et.at[i, :n].set(t)
        et = et.at[i, n:].set(t[-1])
        xs = xs.at[i, :n].set(jax.random.randint(kx, (n,), 0, w_in))
        ys = ys.at[i, :n].set(jax.random.randint(ky, (n,), 0, h_in))
        cs = cs.at[i, :n].set(jax.random.randint(kc, (n,), 0, ic))
        nr.append(n)
    return et, xs, ys, cs, jnp.array(nr, dtype=jnp.int32)


def small_layers(capacity: dict = GENEROUS) -> list:
    """conv(2x8x8 -> 4x8x8) -> conv(-> 4x4x4) -> FC(10),膜電位回歸輸出層。

    capacity: 層名 -> ConvLayer 的容量欄位。
    """
    conv1 = ConvLayer(name="conv1", ic=2, h_in=8, w_in=8, oc=4, k=3, s=1, p=1,
                      init_k=5.0, chunk_size=4, **capacity["conv1"])
    conv2 = ConvLayer(name="conv2", ic=4, h_in=8, w_in=8, oc=4, k=3, s=2, p=1,
                      init_k=5.0, chunk_size=4, **capacity["conv2"])
    out = FCLayer(name="out", n_in=conv2.n_neurons, n_out=10, init_k=5.0, chunk_size=512)
    return [conv1, conv2, out]


def small_policies(layers: list) -> dict:
    """有容量的層各一個預設 GrowthPolicy。"""
    return {layer.name: GrowthPolicy() for layer in layers if layer.capacity is not None}


def with_conv_knob(layers: list, knob: str, value: int) -> list:
    """每個 conv 層的 knob 換成 value,其他層不變。"""
    return [replace(layer, **{knob: value}) if isinstance(layer, ConvLayer) else layer
            for layer in layers]


def init_params(layers: list, seed: int = 0) -> tuple:
    keys = jax.random.split(jax.random.PRNGKey(seed), len(layers))
    return tuple(layer.init_weight(k) for layer, k in zip(layers, keys))


def raw_batch(seed: int = 1, n_samples: int = 6):
    c, h, w = INPUT_SHAPE
    return synthetic_raw_batch(jax.random.PRNGKey(seed), n_samples, MAX_LEN, h, w, c)
