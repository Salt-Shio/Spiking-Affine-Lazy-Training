"""一個 batch 的訓練步:forward、loss、梯度、optimizer 更新。"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from example.training.loss import cross_entropy_loss
from salt_core.capacity import reduce_over_batch


class StepOutput(NamedTuple):
    params: tuple
    opt_state: object
    loss: jax.Array
    diags: list             # 對齊層的 LayerDiag,已經合併成這個 batch 一份
    decoder_metrics: dict   # 解碼器的純量指標,batch 平均
    grad_norms: dict        # 層名 -> 梯度範數
    fits: jax.Array         # 整個 batch 每筆樣本都放得下;False 時這一步的結果不可信


def make_train_step(network, optimizer, decoder, score_cap: float | None):
    """回傳 jit 過的 train_step(params, opt_state, raw_batch, labels_onehot) -> StepOutput。"""
    def loss_fn(params, raw_batch, labels_onehot):
        output = network.apply_batched(params, raw_batch)
        scores, dec_metrics = jax.vmap(decoder.decode)(output.last)
        loss = jnp.mean(cross_entropy_loss(scores, labels_onehot, score_cap))
        reduced = [reduce_over_batch(d) for d in output.diags]
        reduced_metrics = {k: jnp.mean(v) for k, v in dec_metrics.items()}
        return loss, (reduced, reduced_metrics, jnp.all(output.fits))

    @jax.jit
    def train_step(params, opt_state, raw_batch, labels_onehot) -> StepOutput:
        (loss, (diags, dec_metrics, fits)), grad = jax.value_and_grad(
            loss_fn, has_aux=True)(params, raw_batch, labels_onehot)
        grad_norms = {layer.name: jnp.linalg.norm(g) for layer, g in zip(network.layers, grad)}
        updates, opt_state = optimizer.update(grad, opt_state, params)
        return StepOutput(params=optax.apply_updates(params, updates), opt_state=opt_state,
                          loss=loss, diags=diags, decoder_metrics=dec_metrics,
                          grad_norms=grad_norms, fits=fits)

    return train_step
