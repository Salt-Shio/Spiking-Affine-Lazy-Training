# XLA GPU determinism: wrong gradients through `mode='drop'` scatter, merged batch

**封存(2026-09-27)**:這個資料夾只當 XLA bug 的查證證據,不維護。

- `verify_trash_row_equivalence.py` 比對的原版是 `compress_candidates_drop.py`
  (從 commit `60fa3a7` 的前一版凍結下來),不再 import `salt_core`,現在還能跑。
- `test_no_vmap_customvjp_small.py`、`verify_real_pipeline.py` 對應 commit
  `60fa3a7` 當時的程式碼,之後沒有跟著專案更新,只當證據。

## 現況(2026-09-18,已解決)

根因已定位、已在正式程式碼修掉並驗證(`salt_core/connectivity/conv.py::_compress_candidates`,
改用「垃圾桶」寫法取代 `mode='drop'`)。這個資料夾保留完整的查證過程跟證據,
之後如果要考慮向 jax-ml/jax 或 openxla/xla 報 issue,直接用這份就夠。

**必要條件(兩者缺一都不會觸發,已用逐步拆解驗證,見下方「拆解過程」)**:

1. 合併多個邏輯獨立樣本進同一次 `scatter-add`(單獨不夠)。
2. `_compress_candidates` 用 `mode='drop'` 處理不合法候選(候選幾何上不落在
   任何神經元的感受野內)跟溢出候選(`local_rank >= max_queue_len`)——**真的
   觸發丟棄**(不是只是「有候選被標記不合法」這個抽象概念,是 scatter 的
   index 陣列真的帶越界值、靠 `mode='drop'` 被丟棄這件事本身)。

兩者合起來,踩到 XLA 一個 GPU determinism scatter-rewrite pass 的 codegen bug:
`scatter_dims_to_operand_dims` 被錯誤地多塞一個維度、`unique_indices=true`
這個宣告是假的(細節見下方「HLO 比對結果」)。

## 拆解過程(從確認會壞的版本開始,一次拿掉一個成分)

**檔案**:`test_no_vmap_customvjp_small.py`——完全不呼叫 `jax.vmap`(batch 軸
直接編進 `_compress_candidates` 的排序複合鍵跟 scatter 目標維度,`weight`
gather 用 `jax.custom_vjp` + 手寫 `jax.lax.scatter_add`,自己指定
`dimension_numbers`,不靠 autodiff 自動產生),H_IN=5 小尺寸,BATCH=1~4、
旗標開關各測一次。

一開始(HLO metadata 裡的 `vmap(vmap())`)以為是「外層 batch vmap」本身,
後來證實推翻(見下方 HLO 段落跟拆解紀錄)。正確的拆解:

1. 拿掉「無效欄位蓋成 0」的 mask——**還是壞**,mask 不是必要成分。
2. `_axis_candidates` 的 2D(y、x)簡化成 1D(只留 y)——**還是壞**,2D 交叉
   結構不是必要成分。
3. 拿掉「候選可能不合法、用 sentinel 標記、靠 `mode='drop'` 丟棄」這個機制
   (讓所有候選都直接視為合法)——**bug 完全消失**。

修法(`safe_j = jnp.minimum(local_to_global_j, n_events-1)` 這種「全部空欄位
夾到同一個 fallback」的寫法會製造碰撞,已知會壞)嘗試過兩版都沒用:

- 空欄位改成各自不同的 fallback(對 `n_events` 取模分散)——**還是壞**。
- gather 全面改用 `mode=FILL_OR_DROP`(越界 index 直接補 0,不需要事先夾進
  合法範圍,理論上完全不會有 fallback 碰撞)——**還是壞,錯誤數字完全一樣**。

這兩次失敗確認了:問題不是「下游怎麼讀空欄位」,是 `_compress_candidates`
**自己的** `mode='drop'` scatter 有沒有真的觸發丟棄。

**真正有效的修法**:scatter 目標陣列的兩個維度都多開一格當「垃圾桶」
(`n_out_spatial+1`、`max_queue_len+1`),不合法/溢出的候選全部指去垃圾桶
座標(保證合法範圍內),scatter 從頭到尾不需要 `mode='drop'`;事後把垃圾桶
切掉。BATCH=1~4、旗標開關全部正確。

**邏輯等價驗證**(`verify_trash_row_equivalence.py`):直接拿這個新寫法跟
原版(`compress_candidates_drop.py`)逐位元比對(不是
只看梯度測試過),涵蓋一般情況、全部合法、全部不合法、L 溢出、空清單、多組
隨機 seed,10/10 一致。第一版有個真的邏輯錯誤(把 `n_real_per_neuron` 誤
封頂在 `max_queue_len`,這個數字下游動態放大機制要用來偵測「真的需要比 L
更大的容量」,封頂會讓那個機制失效)——用這個逐位元比對抓到並修正。

## HLO 比對結果

在 `hlo_off.txt`/`hlo_on.txt` 裡找同一個 instruction(`grep -n "scatter-add.5"`,
這兩個檔案是拆解過程中間版本〔`vmap(vmap())` 版本〕留下的,拿掉 `mode='drop'`
之後這個特定 instruction 不會再出現,但錯誤的 codegen 模式本身仍是同一族):

**flag 關(對)**:
```
scatter_dims_to_operand_dims={1,2,3}
```

**flag 開(錯)**:
```
scatter_dims_to_operand_dims={1,2,3,0}
indices_are_sorted=true, unique_indices=true
```

兩份都對應同一個 `op_name`:
```
jit(loss_batched)/transpose(jvp(vmap(vmap())))/scatter-add
```

**兩個具體問題**:

1. `scatter_dims_to_operand_dims` 開旗標後多了一個 `0`。scatter 目標是
   rank-4 的 `f32[1,1,3,3]`,合法的目標維度應該只有 3 個,多出來的 `0` 是
   維度標記錯誤。
2. `unique_indices=true` 這個宣告本身是假的——`mode='drop'` 讓多個空欄位
   collide 到同一個 fallback 座標,這些 index 天生不是唯一的。

**這裡的因果關係要小心讀**:HLO 裡的 `vmap(vmap())` 一度被誤認為觸發條件
本身(見拆解過程),後來證實推翻——真正必要的是「`mode='drop'` 真的觸發
丟棄」+「多樣本合併進同一次 scatter」,`vmap(vmap())` 只是「拆解時湊出
這個現象」剛好用到的技術手段之一,不是因果鏈的必要環節。

## 目前判斷(不是這個專案程式碼的問題)

XLA 的 GPU determinism scatter-rewrite pass(官方稱 "scatter determinism
expander",見
[Determinism (GPU) | OpenXLA Project](https://openxla.org/xla/determinism))
套用在「由 `mode='drop'` 產生、索引來自合併批次的 scatter-add」上時,維度
標記算錯、且錯誤地假設索引唯一,選了一條只在索引真的唯一時才正確的快速
路徑——這裡的索引 genuinely 不唯一,於是部分梯度貢獻被覆寫/丟棄而非正確
累加,不是逐位元捨入誤差,是結構性算錯。

跟 [jax-ml/jax#25878](https://github.com/jax-ml/jax/issues/25878)、
[jax-ml/jax#27796](https://github.com/jax-ml/jax/issues/27796)、
[jax-ml/jax#32875](https://github.com/jax-ml/jax/issues/32875) 同屬「vmap +
gather/scatter + grad + determinism flag」這個大家族(這三個 issue 本身都已
closed,觸發條件跟這裡對不上,不是同一個變種),但這次定位到具體
instruction、具體哪兩個欄位錯,而且找到了不依賴上游修復的工作繞法。

## 修法對正式程式碼的影響、實測效益

`salt_core/connectivity/conv.py::_compress_candidates` 改用垃圾桶寫法後,
`example/models/conv_net.py::apply_batched` 換回 `jax.vmap`(不用再犧牲效能
繞去 `jax.lax.map`)。完整雙層 conv+FC 網路、10 epoch 實測(開
`--xla_gpu_deterministic_ops=true`):

| | 時間 | 可重現性 |
|---|---|---|
| `vmap`(修好的) | 4分58秒 | 跑兩次逐位元完全一致 |
| `lax.map` | 10分38秒(2.14x 慢) | 正確但沒必要 |

## 環境

- `jax`/`jaxlib` 0.11.1
- GPU: NVIDIA GeForce RTX 5070, driver 610.57.04
- 完整分析過程見這個專案的 `docs/問題紀錄.md` 第八節。

## 還沒決定

要不要把這個(或再簡化過的純 JAX 最小案例)送到 jax-ml/jax 或 openxla/xla
開 issue——這個專案這邊已經有工作繞法(垃圾桶寫法),不再是阻塞項,純粹是
要不要順手回饋上游的問題。
