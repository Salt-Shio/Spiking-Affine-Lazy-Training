"""證明「垃圾桶」版的 _compress_candidates 跟原版邏輯完全等價,不是只看
梯度測試通過——直接逐位元比對兩個函式在同一組輸入下的輸出
(local_to_global_j, n_real_per_neuron),包含邊界情況:
- 一般情況(部分合法部分不合法)
- 全部候選都合法(沒有任何丟棄)
- 全部候選都不合法(某個神經元完全沒有候選)
- L 溢出(某個神經元的合法候選數超過 max_queue_len)
"""
import sys
sys.path.insert(0, "/home/salt/Projects/Spiking-Affine-Lazy-Training")

import numpy as np
import jax
import jax.numpy as jnp

from salt_core.connectivity.conv import _compress_candidates as original


def trash_row_version(n_flat, j_flat, n_out_spatial, max_queue_len, n_events):
    """垃圾桶版:跟原版簽名、語意完全相同,只是內部不用 mode='drop'。"""
    order = jnp.lexsort((j_flat, n_flat))
    sorted_n = n_flat[order]
    sorted_j = j_flat[order]
    C = n_flat.shape[0]
    idx_range = jnp.arange(C)
    is_start = jnp.concatenate([jnp.array([True]), sorted_n[1:] != sorted_n[:-1]])
    start_positions = jnp.where(is_start, idx_range, -1)
    last_start = jax.lax.cummax(start_positions)
    local_rank = idx_range - last_start

    trash_n = n_out_spatial
    trash_col = max_queue_len
    is_invalid_n = sorted_n >= n_out_spatial

    # local_to_global_j:原版兩個丟棄條件都適用(n 不合法 OR local_rank 溢出 L)
    safe_n = jnp.where(is_invalid_n, trash_n, sorted_n)
    safe_rank = jnp.where((local_rank >= max_queue_len) | is_invalid_n, trash_col, local_rank)
    local_to_global_j_padded = jnp.full((n_out_spatial + 1, max_queue_len + 1), n_events, dtype=jnp.int32)
    local_to_global_j_padded = local_to_global_j_padded.at[safe_n, safe_rank].set(sorted_j)
    local_to_global_j = local_to_global_j_padded[:n_out_spatial, :max_queue_len]

    # n_real_per_neuron:原版是一維 scatter,只看 sorted_n 合不合法,
    # 跟 local_rank/max_queue_len 完全無關——local_rank+1 就算超過
    # max_queue_len 也要照樣參與 max,這樣下游才能靠這個數字偵測「真的需要
    # 比 L 更大的容量」(第 7.2 節動態放大機制)。之前的版本誤把 L 溢出的
    # 丟棄條件也套用在這裡,把這個數字錯誤封頂,是真的邏輯錯誤,已修正。
    n_real_padded = jnp.zeros((n_out_spatial + 1,), dtype=jnp.int32)
    real_local_rank = jnp.where(is_invalid_n, 0, local_rank + 1)
    n_real_padded = n_real_padded.at[safe_n].max(real_local_rank)
    n_real = n_real_padded[:n_out_spatial]
    return local_to_global_j, n_real


def check(name, n_flat, j_flat, n_out_spatial, max_queue_len, n_events):
    l2g_ref, nreal_ref = original(jnp.array(n_flat), jnp.array(j_flat), n_out_spatial, max_queue_len, n_events)
    l2g_new, nreal_new = trash_row_version(jnp.array(n_flat), jnp.array(j_flat), n_out_spatial, max_queue_len, n_events)
    match_l2g = bool(jnp.array_equal(l2g_ref, l2g_new))
    match_nreal = bool(jnp.array_equal(nreal_ref, nreal_new))
    status = "一致" if (match_l2g and match_nreal) else "不一致!!"
    print(f"[{name}] local_to_global_j 一致={match_l2g}  n_real 一致={match_nreal}  -> {status}")
    if not match_l2g:
        print("  ref :\n", l2g_ref)
        print("  new :\n", l2g_new)
    if not match_nreal:
        print("  ref n_real :", nreal_ref)
        print("  new n_real :", nreal_new)
    return match_l2g and match_nreal


results = []

# 1) 一般情況:隨機候選,部分合法部分不合法
rng = np.random.default_rng(0)
n_out_spatial, max_queue_len, n_events = 8, 3, 6
C = 20
n_flat = rng.integers(0, n_out_spatial + 3, C)  # 有些會 >= n_out_spatial(不合法)
n_flat = np.where(n_flat >= n_out_spatial, n_out_spatial, n_flat)  # 部分強制標成不合法 sentinel
j_flat = rng.integers(0, n_events, C)
results.append(check("一般情況(部分不合法)", n_flat, j_flat, n_out_spatial, max_queue_len, n_events))

# 2) 全部候選都合法,沒有任何丟棄
n_flat2 = rng.integers(0, n_out_spatial, C)
j_flat2 = rng.integers(0, n_events, C)
results.append(check("全部合法", n_flat2, j_flat2, n_out_spatial, max_queue_len, n_events))

# 3) 全部候選都不合法(每個都標成 sentinel),所有神經元應該 n_real=0
n_flat3 = np.full(C, n_out_spatial)
j_flat3 = rng.integers(0, n_events, C)
results.append(check("全部不合法", n_flat3, j_flat3, n_out_spatial, max_queue_len, n_events))

# 4) L 溢出:同一個神經元收到遠超過 max_queue_len 的合法候選
small_L = 2
n_flat4 = np.array([0] * 6 + [1] * 2)  # 神經元 0 收到 6 個候選(遠超過 L=2),神經元1收到2個(剛好)
j_flat4 = np.arange(8) % n_events
results.append(check("L 溢出", n_flat4, j_flat4, n_out_spatial, small_L, n_events))

# 5) 邊界:C=0(完全沒有候選,理論上不會真的發生,但測試防禦性)
n_flat5 = np.array([], dtype=np.int32)
j_flat5 = np.array([], dtype=np.int32)
results.append(check("空候選清單", n_flat5, j_flat5, n_out_spatial, max_queue_len, n_events))

# 6) 多次不同隨機 seed 的一般情況,擴大覆蓋面
for seed in range(1, 6):
    r = np.random.default_rng(seed)
    n_out_spatial_r, max_queue_len_r, n_events_r = 12, 4, 10
    C_r = 40
    n_flat_r = r.integers(0, n_out_spatial_r + 5, C_r)
    n_flat_r = np.where(n_flat_r >= n_out_spatial_r, n_out_spatial_r, n_flat_r)
    j_flat_r = r.integers(0, n_events_r, C_r)
    results.append(check(f"隨機 seed={seed}", n_flat_r, j_flat_r, n_out_spatial_r, max_queue_len_r, n_events_r))

print()
n_pass = sum(results)
print(f"總計 {n_pass}/{len(results)} 項一致" + ("，全過" if n_pass == len(results) else "，有不一致，不能當成等價"))
