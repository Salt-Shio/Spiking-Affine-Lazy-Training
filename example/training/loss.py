"""訓練用的 loss。"""
import jax
import jax.numpy as jnp
import optax


def cross_entropy_loss(scores: jax.Array, labels_onehot: jax.Array,
                       score_cap: float | None) -> jax.Array:
    """每筆樣本的 softmax cross entropy,形狀 (batch,)。

    score_cap: None 時分數原樣;有值時先換成 score_cap * tanh(scores / score_cap),
        理由見 docs/math/梯度下降曲率穩定性推導.md。評估用原始分數,不經過這個轉換。
    """
    if score_cap is not None:
        scores = score_cap * jnp.tanh(scores / score_cap)
    return optax.softmax_cross_entropy(scores, labels_onehot)
