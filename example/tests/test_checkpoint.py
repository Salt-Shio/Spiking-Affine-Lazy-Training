"""example/checkpoint.py 的存讀測試。

每種 optimizer 先走兩步,存檔、讀回,檢查權重、optimizer 狀態、shuffle key、epoch
逐值相同、網路描述相同,而且讀回後的下一步 update 跟沒存讀過的一樣。
AdamW、梯度裁剪、餘弦退火用訓練腳本的 build_optimizer 建,跟實際訓練同一條路。
"""
import jax
import jax.numpy as jnp
import optax

from example.checkpoint import Checkpointer
from example.training.optim import build_optimizer
from salt_core.layers import ConvLayer, FCLayer
from salt_core.network import Network

_SHAPES = ((8, 2, 3, 3), (16, 8, 3, 3), (10, 1296))
# 權重形狀是 _SHAPES 的網路,容量用非預設值,確認讀回的是存進去的容量
_NETWORK = Network(input_shape=(2, 34, 34), layers=(
    ConvLayer(name="conv1", ic=2, h_in=34, w_in=34, oc=8, k=3, s=2, p=1, init_k=5.0,
              max_queue_len=185, max_out_spikes=5361, max_steps=146),
    ConvLayer(name="conv2", ic=8, h_in=17, w_in=17, oc=16, k=3, s=2, p=1, init_k=5.0,
              max_queue_len=1083, max_out_spikes=3417, max_steps=540),
    FCLayer(name="out", n_in=1296, n_out=10, init_k=5.0)))
_N_TRAIN = 16
_BATCH_SIZE = 4


def _params(seed: int) -> tuple:
    keys = jax.random.split(jax.random.PRNGKey(seed), len(_SHAPES))
    return tuple(jax.random.normal(k, shape) for k, shape in zip(keys, _SHAPES))


def _grad(params) -> tuple:
    return tuple(jnp.full_like(w, 0.01) for w in params)


def _assert_trees_equal(a, b, what: str) -> None:
    leaves_a, treedef_a = jax.tree_util.tree_flatten(a)
    leaves_b, treedef_b = jax.tree_util.tree_flatten(b)
    assert treedef_a == treedef_b, f"{what} 結構不同"
    for i, (x, y) in enumerate(zip(leaves_a, leaves_b)):
        assert jnp.array_equal(jnp.asarray(x), jnp.asarray(y)), f"{what} leaf[{i}] 讀回後不一致"


def _assert_roundtrip(optimizer, path: str) -> None:
    params = _params(123)
    opt_state = optimizer.init(params)
    for _ in range(2):
        updates, opt_state = optimizer.update(_grad(params), opt_state, params)
        params = optax.apply_updates(params, updates)
    shuffle_key = jax.random.PRNGKey(999)

    ckpt = Checkpointer(path)
    ckpt.save(network=_NETWORK, params=params, opt_state=opt_state,
              shuffle_key=shuffle_key, epoch=7)
    assert ckpt.exists() and ckpt.last_epoch == 7

    loaded = ckpt.load(opt_state_template=optimizer.init(_params(0)))
    loaded_params, loaded_opt_state = loaded.params, loaded.opt_state

    assert loaded.network == _NETWORK
    _assert_trees_equal(params, loaded_params, "params")
    _assert_trees_equal(opt_state, loaded_opt_state, "opt_state")
    assert jnp.array_equal(shuffle_key, loaded.shuffle_key)
    assert loaded.epoch == 7

    next_update, _ = optimizer.update(_grad(params), opt_state, params)
    loaded_next_update, _ = optimizer.update(_grad(loaded_params), loaded_opt_state, loaded_params)
    _assert_trees_equal(next_update, loaded_next_update, "讀回後的下一步 update")


def test_adam_state_roundtrip(tmp_path):
    _assert_roundtrip(optax.adam(1e-2), str(tmp_path / "ckpt.npz"))


def test_adamw_state_roundtrip(tmp_path):
    optimizer = build_optimizer({"lr": 1e-2, "weight_decay": 1e-2}, _N_TRAIN, _BATCH_SIZE)
    _assert_roundtrip(optimizer, str(tmp_path / "ckpt.npz"))


def test_grad_clip_chain_state_roundtrip(tmp_path):
    optimizer = build_optimizer({"lr": 1e-2, "grad_clip_norm": 10.0}, _N_TRAIN, _BATCH_SIZE)
    _assert_roundtrip(optimizer, str(tmp_path / "ckpt.npz"))


def test_cosine_schedule_state_roundtrip(tmp_path):
    """schedule 的步數存在 opt_state 裡;讀回後下一步的學習率要接著走,不能歸零。"""
    optimizer = build_optimizer({"lr": 1e-2, "epochs": 2, "lr_cosine_decay": True},
                                _N_TRAIN, _BATCH_SIZE)
    _assert_roundtrip(optimizer, str(tmp_path / "ckpt.npz"))
