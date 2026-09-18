"""已確認會重現 bug 的版本(完全無 vmap,custom_vjp 手寫 scatter_add backward,
H_IN=5 小尺寸)。之後要拆解找最小觸發條件,從這份開始一步步刪,不要從空白
重建——已經證實兩次「猜的簡化版」測不出問題,不能信任重建的版本。

跑法:`python test_no_vmap_customvjp_small.py off` / `... on`。
"""
import os
import sys

flag_on = (len(sys.argv) > 1 and sys.argv[1] == "on")
if flag_on:
    os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import jax
import jax.numpy as jnp

K, S, P = 3, 2, 1
TAU = 16.0
H_IN = 5
IC, OC = 1, 1
N_EVENTS = 4
L = 4


def h_out_for(h_in):
    return (h_in + 2 * P - K) // S + 1


def _axis_candidates(i, K, S, P, N, O_max):
    o_min = (i + P - K + 1 + S - 1) // S
    upper = (i + P) // S
    r = jnp.arange(N, dtype=i.dtype)
    o = o_min[..., None] + r
    valid = (o <= upper[..., None]) & (o >= 0) & (o < O_max)
    k = i[..., None] - o * S + P
    return o, valid, k


@jax.custom_vjp
def weight_gather(W, idx_c, idx_ky, idx_kx):
    # idx_*: (BATCH, n_out, L)
    return W[:, idx_c, idx_ky, idx_kx]  # (OC, BATCH, n_out, L)


def weight_gather_fwd(W, idx_c, idx_ky, idx_kx):
    return weight_gather(W, idx_c, idx_ky, idx_kx), (W.shape, idx_c, idx_ky, idx_kx)


def weight_gather_bwd(residuals, cotangent):
    W_shape, idx_c, idx_ky, idx_kx = residuals
    OC = W_shape[0]
    indices = jnp.stack([idx_c.reshape(-1), idx_ky.reshape(-1), idx_kx.reshape(-1)], axis=-1)
    n_idx = indices.shape[0]
    updates = jnp.transpose(cotangent.reshape(OC, n_idx), (1, 0))
    dnums = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(1,), inserted_window_dims=(1, 2, 3),
        scatter_dims_to_operand_dims=(1, 2, 3),
    )
    dW = jax.lax.scatter_add(jnp.zeros(W_shape, dtype=cotangent.dtype), indices, updates, dnums,
                              indices_are_sorted=False, unique_indices=False)
    return dW, None, None, None


weight_gather.defvjp(weight_gather_fwd, weight_gather_bwd)


def build_queue_no_vmap(times_b, x_b, y_b, c_b, W, batch):
    h_out = h_out_for(H_IN)
    n_out_spatial = h_out * h_out
    N = (K - 1) // S + 1
    n_events = N_EVENTS

    o_y, valid_y, _ = _axis_candidates(y_b, K, S, P, N, h_out)
    o_x, valid_x, _ = _axis_candidates(x_b, K, S, P, N, h_out)
    valid_2d = valid_y[:, :, :, None] & valid_x[:, :, None, :]
    n_2d = o_y[:, :, :, None] * h_out + o_x[:, :, None, :]
    j_2d = jnp.broadcast_to(jnp.arange(n_events, dtype=jnp.int32)[None, :, None, None], (batch, n_events, N, N))
    b_2d = jnp.broadcast_to(jnp.arange(batch, dtype=jnp.int32)[:, None, None, None], (batch, n_events, N, N))

    n_flat = jnp.where(valid_2d, n_2d, n_out_spatial).reshape(-1)
    j_flat = j_2d.reshape(-1)
    b_flat = b_2d.reshape(-1)

    order = jnp.lexsort((j_flat, n_flat, b_flat))
    sorted_b = b_flat[order]; sorted_n = n_flat[order]; sorted_j = j_flat[order]
    C = n_flat.shape[0]
    idx_range = jnp.arange(C)
    is_start = jnp.concatenate([jnp.array([True]),
        (sorted_n[1:] != sorted_n[:-1]) | (sorted_b[1:] != sorted_b[:-1])])
    start_positions = jnp.where(is_start, idx_range, -1)
    last_start = jax.lax.cummax(start_positions)
    local_rank = idx_range - last_start

    # *** 修法:垃圾桶版,不再用 mode='drop'(見 xla_repro/verify_trash_row_equivalence.py
    # 的逐位元等價驗證,10/10 case 一致,含 L 溢出邊界)。scatter 目標陣列多開一格
    # 垃圾桶,不合法/溢出的候選全部指去垃圾桶,事後切掉。n_real_per_neuron 的
    # scatter 只看 n 合不合法,跟 local_rank 有沒有溢出 L 無關(這是原版真正的
    # 語意,之前一版誤把兩個丟棄條件混在一起,已修正並驗證過)。 ***
    trash_n = n_out_spatial
    trash_col = L
    is_invalid_n = sorted_n >= n_out_spatial
    safe_n = jnp.where(is_invalid_n, trash_n, sorted_n)
    safe_rank = jnp.where((local_rank >= L) | is_invalid_n, trash_col, local_rank)

    local_to_global_j_padded = jnp.full((batch, n_out_spatial + 1, L + 1), n_events, dtype=jnp.int32)
    local_to_global_j_padded = local_to_global_j_padded.at[sorted_b, safe_n, safe_rank].set(sorted_j)
    local_to_global_j = local_to_global_j_padded[:, :n_out_spatial, :L]

    n_real_per_neuron_padded = jnp.zeros((batch, n_out_spatial + 1), dtype=jnp.int32)
    real_local_rank = jnp.where(is_invalid_n, 0, local_rank + 1)
    n_real_per_neuron_padded = n_real_per_neuron_padded.at[sorted_b, safe_n].max(real_local_rank)
    n_real_per_neuron = n_real_per_neuron_padded[:, :n_out_spatial]

    safe_j = jnp.minimum(local_to_global_j, n_events - 1)
    batch_idx = jnp.arange(batch)[:, None, None]
    x_g = x_b[batch_idx, safe_j]; y_g = y_b[batch_idx, safe_j]; c_g = c_b[batch_idx, safe_j]
    oy_grid = (jnp.arange(n_out_spatial, dtype=jnp.int32) // h_out)[None, :, None]
    ox_grid = (jnp.arange(n_out_spatial, dtype=jnp.int32) % h_out)[None, :, None]
    k_y_g = y_g - oy_grid * S + P; k_x_g = x_g - ox_grid * S + P
    safe_k_y = jnp.clip(k_y_g, 0, K - 1); safe_k_x = jnp.clip(k_x_g, 0, K - 1)

    weight_vals = weight_gather(W, c_g, safe_k_y, safe_k_x)  # (OC, batch, n_out, L)
    weight_vals = jnp.transpose(weight_vals, (1, 0, 2, 3))  # (batch, OC, n_out, L)

    col_idx = jnp.arange(L, dtype=jnp.int32)[None, None, :]
    is_real = col_idx < n_real_per_neuron[:, :, None]
    is_real = is_real[:, None, :, :]
    weight_vals_masked = jnp.where(is_real, weight_vals, 0.0)
    return jnp.sum(weight_vals_masked)


def make_sample(seed):
    r = np.random.default_rng(seed)
    times = np.sort(r.uniform(0, 300, N_EVENTS)).astype(np.float32)
    x = r.integers(0, H_IN, N_EVENTS).astype(np.int32)
    y = r.integers(0, H_IN, N_EVENTS).astype(np.int32)
    c = r.integers(0, IC, N_EVENTS).astype(np.int32)
    return times, x, y, c


def check(batch):
    samples = [make_sample(1000 + i) for i in range(batch)]
    times_b = jnp.stack([jnp.array(s[0]) for s in samples])
    x_b = jnp.stack([jnp.array(s[1]) for s in samples])
    y_b = jnp.stack([jnp.array(s[2]) for s in samples])
    c_b = jnp.stack([jnp.array(s[3]) for s in samples])
    W0 = jnp.array((np.random.default_rng(42).normal(size=(OC, IC, K, K)) * 0.5).astype(np.float32))

    def loss_single(W, times, x, y, c):
        return build_queue_no_vmap(times[None], x[None], y[None], c[None], W, 1)

    grad_loop = jnp.zeros_like(W0)
    for i in range(batch):
        grad_loop = grad_loop + jax.grad(loss_single)(W0, times_b[i], x_b[i], y_b[i], c_b[i])

    def loss_all(W):
        return build_queue_no_vmap(times_b, x_b, y_b, c_b, W, batch)

    grad_merged = jax.grad(loss_all)(W0)

    diff = jnp.abs(grad_merged - grad_loop)
    rel = float(jnp.max(diff) / (jnp.max(jnp.abs(grad_loop)) + 1e-12))
    flag_label = "flag=on " if flag_on else "flag=off"
    print(f"{flag_label} BATCH={batch}: rel diff = {rel:.4e}  {'正確' if rel < 1e-2 else '錯'}")


if __name__ == "__main__":
    for batch in [1, 2, 3, 4]:
        check(batch)
