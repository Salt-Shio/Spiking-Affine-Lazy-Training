"""用真正的資料 + 真正的網路(不是合成資料)驗證 vmap 算出來的梯度是否正確。

可重現性(同一種方法跑兩次一樣)不能證明正確性——一個穩定算錯的方法也會
每次重現同一個錯誤答案。這裡拿一個獨立、數學上保證正確的算法當基準:
逐筆迴圈算每個樣本的梯度、平均(對應 loss_fn 的 jnp.mean),因為
「batch 平均後的梯度」剛好等於「每個樣本各自梯度的平均」(線性),跟
batch 有沒有被向量化融合無關——這是任何實作都該滿足的數學恆等式,不是
只在這個專案裡成立的巧合。

用的是這次實際訓練(20260918_135221,vmap 版)存下來的 epoch_000 權重
快照(不是隨機初始化,是真的訓練過一步的權重)+ 真正的 N-MNIST 訓練資料。
"""
import os

os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import sys

sys.path.insert(0, "/home/salt/Projects/Spiking-Affine-Lazy-Training")

import jax
import jax.numpy as jnp
import optax

from data.src.nmnist import NMNISTDataset
from example.models.conv_net import build_network, build_decoder
from example.utils import load_params_npz
from example.paths import DATASET_ROOT, EXPERIMENTS_DIR, resolve_config
from example.train_conv_compressed import load_config
import os as _os

CONFIG_PATH = resolve_config("configs/conv/verify_laxmap_determinism.yaml")
WEIGHTS_PATH = _os.path.join(
    EXPERIMENTS_DIR,
    "conv_compressed_laxmap_determinism_verify_20260918_135221",
    "weights", "epoch_000.npz",
)

raw_cfg = load_config(str(CONFIG_PATH))
model_cfg = raw_cfg["model"]
data_cfg = raw_cfg["data"]
train_cfg = raw_cfg["train"]

layers = build_network(model_cfg)
decoder = build_decoder(model_cfg, layers)
params = load_params_npz(WEIGHTS_PATH, layers)

dataset = NMNISTDataset(DATASET_ROOT, max_events=data_cfg["max_events"])
train_split = dataset.build_split(seed=data_cfg["seed_train"],
                                   n_samples=data_cfg["train_size"], which="train")

batch_size = train_cfg["batch_size"]
idx = jnp.arange(batch_size)  # 固定拿前 batch_size 筆真實訓練資料,不用隨機打亂


class _Net:
    def __init__(self, layers):
        self.layers = layers

    def apply(self, params, event_times, x, y, c, n_real_events):
        from example.models.conv_net import ConvNetCompressed
        return ConvNetCompressed(self.layers).apply(params, event_times, x, y, c, n_real_events)

    def apply_batched(self, params, batch_event_times, batch_x, batch_y, batch_c, batch_n_real):
        return jax.vmap(self.apply, in_axes=(None, 0, 0, 0, 0, 0))(
            params, batch_event_times, batch_x, batch_y, batch_c, batch_n_real)


net = _Net(layers)


def _cross_entropy(scores, labels_onehot):
    return optax.softmax_cross_entropy(scores, labels_onehot)


def loss_single(params, event_times, x, y, c, n_real, label_onehot):
    result, _ = net.apply(params, event_times, x, y, c, n_real)
    score, _ = decoder.decode(result)
    return _cross_entropy(score[None, :], label_onehot[None, :])[0]


def loss_batched_vmap(params, batch_event_times, batch_x, batch_y, batch_c, batch_n_real,
                       batch_labels_onehot):
    result, _ = net.apply_batched(params, batch_event_times, batch_x, batch_y, batch_c, batch_n_real)
    scores, _ = jax.vmap(decoder.decode)(result)
    per_sample = _cross_entropy(scores, batch_labels_onehot)
    return jnp.mean(per_sample)


batch_event_times = train_split.event_times[idx]
batch_x = train_split.x[idx]
batch_y = train_split.y[idx]
batch_c = train_split.c[idx]
batch_n_real = train_split.n_real_events[idx]
batch_labels_onehot = train_split.labels_onehot[idx]

print(f"用真實資料:batch_size={batch_size},權重來源={WEIGHTS_PATH}")

# --- 基準:逐筆迴圈算梯度,平均(數學上保證正確,不靠任何向量化技巧) ---
grad_loop = None
for i in range(batch_size):
    g = jax.grad(loss_single)(params, batch_event_times[i], batch_x[i], batch_y[i],
                               batch_c[i], batch_n_real[i], batch_labels_onehot[i])
    if grad_loop is None:
        grad_loop = [jnp.array(x) for x in g]
    else:
        grad_loop = [a + jnp.array(b) for a, b in zip(grad_loop, g)]
grad_loop = [g / batch_size for g in grad_loop]

# --- 待驗證:vmap 批次融合 ---
grad_vmap = jax.grad(loss_batched_vmap)(
    params, batch_event_times, batch_x, batch_y, batch_c, batch_n_real, batch_labels_onehot)

max_rel = 0.0
for layer, g_loop, g_vmap in zip(layers, grad_loop, grad_vmap):
    diff = jnp.max(jnp.abs(jnp.array(g_vmap) - g_loop))
    denom = jnp.max(jnp.abs(g_loop)) + 1e-12
    rel = float(diff / denom)
    max_rel = max(max_rel, rel)
    print(f"  {layer.name}: max_abs_diff={float(diff):.3e}  rel={rel:.3e}")

print(f"\n整體 max rel diff = {max_rel:.3e}  {'正確!!' if max_rel < 1e-2 else '還是錯'}")
