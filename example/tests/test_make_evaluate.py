"""example/utils.py 的 make_evaluate:容量出界時放大評估容量重算。"""
import functools

import jax
import jax.numpy as jnp
import numpy as np

from data.src.nmnist import NMNISTSplit
from example.utils import make_evaluate
from salt_core.decoder import MembraneRegressionDecoder
from salt_core.network import Network
from salt_core.tests._small_network import (INPUT_SHAPE, init_params, raw_batch, small_layers,
                                             small_policies, with_conv_knob)

N_SAMPLES = 6
EVAL_BATCH = 3


def _split() -> NMNISTSplit:
    et, x, y, c, nr = raw_batch(seed=4, n_samples=N_SAMPLES)
    et = jnp.floor(et)  # make_evaluate 會檢查時間是整數毫秒
    labels = jax.random.randint(jax.random.PRNGKey(5), (N_SAMPLES,), 0, 10)
    return NMNISTSplit(event_times=et, x=x, y=y, c=c, n_real_events=nr, labels=labels,
                       labels_onehot=jax.nn.one_hot(labels, 10))


@functools.cache
def _generous_case():
    """給足容量的小網路:(layers, params, split, evaluate 的結果)。"""
    layers = small_layers()
    params = init_params(layers, seed=3)
    split = _split()
    evaluate = make_evaluate(Network(INPUT_SHAPE, layers), MembraneRegressionDecoder(),
                             EVAL_BATCH, small_policies(layers))
    return layers, params, split, evaluate(params, split)


def test_generous_capacity_does_not_regrow():
    *_, (_acc, _loss, _preds, regrows) = _generous_case()
    assert regrows == 0


def test_overflow_regrows_and_matches_generous():
    layers, params, split, (acc, loss, preds, _) = _generous_case()
    small = with_conv_knob(layers, "max_queue_len", 1)
    net = Network(INPUT_SHAPE, small)
    evaluate = make_evaluate(net, MembraneRegressionDecoder(), EVAL_BATCH, small_policies(small))

    got_acc, got_loss, got_preds, regrows = evaluate(params, split)
    assert regrows > 0
    np.testing.assert_array_equal(got_preds, preds)
    assert got_acc == acc
    assert abs(got_loss - loss) < 1e-5

    # 放大後的容量留給下一次呼叫;net 自己的容量不變
    assert evaluate(params, split)[3] == 0
    assert net.layers == tuple(small)
