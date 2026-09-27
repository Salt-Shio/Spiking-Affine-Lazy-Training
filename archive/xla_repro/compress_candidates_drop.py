"""mode='drop' 版的 _compress_candidates,凍結自 commit 60fa3a7 的前一版 salt_core/connectivity/conv.py。

只給 verify_trash_row_equivalence.py 當比對基準,不維護。
回傳 (local_to_global_j, n_real_per_neuron),形狀 (n_out_spatial, max_queue_len)、(n_out_spatial,)。
"""
import jax
import jax.numpy as jnp


def _compress_candidates(n_flat: jax.Array, j_flat: jax.Array, n_out_spatial: int,
                          max_queue_len: int, n_events: int) -> tuple[jax.Array, jax.Array]:
    order = jnp.lexsort((j_flat, n_flat))
    sorted_n = n_flat[order]
    sorted_j = j_flat[order]

    C = n_flat.shape[0]
    idx_range = jnp.arange(C)
    is_start = jnp.concatenate([jnp.array([True]), sorted_n[1:] != sorted_n[:-1]])
    start_positions = jnp.where(is_start, idx_range, -1)
    last_start = jax.lax.cummax(start_positions)
    local_rank = idx_range - last_start

    local_to_global_j = jnp.full((n_out_spatial, max_queue_len), n_events, dtype=jnp.int32)
    local_to_global_j = local_to_global_j.at[sorted_n, local_rank].set(sorted_j, mode='drop')

    # 對每個 n scatter-max(local_rank+1):同一段內 local_rank 嚴格遞增
    # 0,1,...,count-1,段內最後一筆的 local_rank+1 剛好等於這段的合法候選數
    # (=這個神經元的 n_real_events),用 max 而不是取最後一筆,是因為 scatter
    # 不保證處理順序,但這裡任一筆的 local_rank+1 都 <= count,取 max 恆等於
    # count,不用依賴處理順序。
    n_real_per_neuron = jnp.zeros((n_out_spatial,), dtype=jnp.int32)
    n_real_per_neuron = n_real_per_neuron.at[sorted_n].max(local_rank + 1, mode='drop')

    return local_to_global_j, n_real_per_neuron
