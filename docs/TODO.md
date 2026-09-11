# TODO

尚未做的事。已完成的開發歷程不記在這裡。
背景「為什麼做這件事」見 [`../README.md`](../README.md);架構見 [`架構.md`](架構.md);規格見 [`規格書.md`](規格書.md);洞見紀錄見 [`問題紀錄.md`](問題紀錄.md)。

## 已知限制(已決定接受代價,非阻塞)

- **`XLA_FLAGS=--xla_gpu_deterministic_ops=true` 開著時,壓縮版訓練梯度會錯。**
  觸發條件 = 外層 batch `jax.vmap`(batch ≥ 2)+ `jax.grad` + 這個 flag;實測
  `dL/dW` 相對誤差 ≈ 0.4–0.5,batch=1 正常。**forward 不受影響**(flag 開 / 關
  逐位元相同)——要跟 CSNN-FPGA 做 bit-exact forward 驗證時 flag 可以開。
  訓練期不設這個 flag,接受「同一顆 seed 重跑會有 float 級分岔」的代價。剝掉
  `salt_core` 的純 JAX 最小 repro 還沒 bisect 出來。細節見
  [`問題紀錄.md`](問題紀錄.md)「GPU 規約不可重現」節。

## 待處理

- **SNN 活動 monitor,ReDo 的前置。** 設計定案見 [`監測規格.md`](監測規格.md)。
  `salt_core` 側已完成:`dormant` 搬進 `salt_core/dormant.py`(`dormant_score` 純歸約
  primitive + `dormant_report` 自己逐層跑 forward,不碰 `LayerDiag`);`LayerDiag` 依
  §4.2 維持 4 欄不動;§6 的 `run_network_traced` / `LayerForwardTrace` /
  `salt_core/monitor.py`(逐層 `(n, max_steps)`:reset 後膜電位、是否 spike、`s_value`、
  對到的真實毫秒;forward-only、`stop_gradient`;`chunk_scan` 抽共用 scan 內核,
  `run_layer_forward` 逐位元不變)。測試 `test_dormant.py` / `test_monitor.py`。
  `example` 消費端(訓練側 + 讀端)已完成:`example/trace_probe.py` 的 `TraceProbe`
  週期性對固定前 K 筆 train 樣本跑 `run_network_traced`,逐神經元摘要疊進
  `experiments/<run>/traces/summary.npz`、每隔幾個 epoch 另存
  `full_epoch_XXX.npz`(完整 `(S, n, max_steps)`);`train.probe_every > 0` 才開。
  `example/inspect_traces.py` 讀 `traces/`:休眠曲線(重用 `dormant_score`)、
  整段沒醒的神經元、單神經元波形。測試 `test_trace_probe.py` / `test_inspect_traces.py`。
  未完成:`eval_test.py` 加同一套 `--trace`(共用 dump 函式);`metrics.csv` 要不要多寫
  欄位等 ReDo 準則定案再決定(§7.3);§7.3 的 dormant / ReDo 逐神經元活動來源串 ReDo 時再定。
- **monitor 這批 code 要做一次結構重整(功能正確,但寫得急)。** 已知難聞點:
  (1) `trace_probe.py` 的 `_compile` 靠 `layers is self._cache_key` 物件 identity 當
  編譯快取 key,隱晦;(2) `run()` 裡 `acc = [{k: a[k]+d[k] ...}]` 手刻 tree reduce,
  `jax.tree_util` 有現成;(3) `self._summ` 是「層名→欄名→list of (n,) 陣列」三層巢狀
  dict 原地 mutate,`_record` 還分 append / 覆寫 slot 兩條路 —— 改成「以 epoch 為 key」
  重寫時才組陣列;(4) `_SUMMARY_KEYS` 跟 `_summarise_one` 的 key 抄兩份;
  (5) npz 扁平命名 `f"{層名}__{欄名}"` 兩支檔案來回拼 / `.split("__")` 拆,該收成一個
  helper;(6) `inspect_traces.py` 的 `report_*` 計算跟 print 綁死,無法重用;
  (7) monitor 這條路橫跨 `chunk_scan` / `layers` / `monitor` / `trace_probe` /
  `inspect_traces` 五個檔。另有兩個純顯示 / 文件小 bug:`inspect_traces` 把 `epochs`
  印成 `[np.int32(0), ...]`(該用 `.tolist()`);§7.2 原本「跟 metrics.csv 一致」的措辭
  太滿(已改)。
- **`max_steps` 與 `chunk_size` 脫鉤(conv 層)。** `ConvLayer.__call__` 傳
  `max_steps=self.L`,不看 `chunk_size`;FC 層已用 `ceil(輸入流長度 / chunk_size)`。
  現在 conv `chunk_size=1` 沒差,一調大就無效——`lax.scan` 不能提前退出,會 fire 的層
  最壞情況(每個事件都 fire)強制 `max_steps=L`,`chunk_size>1` 只是每步做更多事、
  步數不變(負優化)。正解:非 fire 層 `ceil(L/chunk_size)`;會 fire 的層 `chunk_size>1`
  本質無效,要嘛別開、要嘛接受。是設計限制,不是 bug。
- **L 收縮機制(啟發式,低優先)。** 現在 `grown_to_fit` 只單向長大。fire rate 訓練中
  單調下降 → 下游事件變少 → `max_real_queue` 掉,L 有收縮空間(偵測訊號現成)。要做成
  有 hysteresis 的啟發式(連續 N epoch `max_real_queue < L × 比例` 才縮一階),否則縮完
  又要長回來 = thrash + 重編譯。
- **`example/` 單元測試覆蓋缺口。** `example/tests/` 只有一個 e2e 檔
  (`test_train_conv_compressed.py`);`metrics_log` / `checkpoint` / `run_epochs` /
  `build_network` / `build_decoder` / `dormant` / `verify_init_k` 都沒單元測試。
  test 內容可能要先整理再補,補哪幾個、補多深未定。
- **兩個「尾端假事件」機制概念上仍分開。** `core.py` 的 `mask_pad_events`(修
  `AffineMap`)跟 `chunk_scan.py` 的 `n_valid_in_chunk`(修 `s_value`)做的是同
  一件事的兩半。低優先。
- **`docs/math/` 幾份推導文件還有前身時代的殘留**:`D:\...` 絕對路徑、`event_lif/`
  舊資料夾名、指向已廢棄 firing-rate 準則的段落。要清一輪。

## 開放題(往下走才需要)

- **`init_k = √3`(Lee 變異數保持)+ ReDo(訓練中回收休眠神經元)當一個 package。**
  目前 baseline `init_k = 5.0`、不做 ReDo(2-conv 網路不需要:V6 連 √3 的 70% 休眠
  都訓到 0.795,`init_k=5` 休眠會自己降)。冷 init 的壓縮佇列記憶體優勢隨網路深度
  複利,ReDo 讓冷 init 撐得起會休眠的隱藏層(尤其 FC→FC)。真的要疊深 FC / 更深
  conv 時再評估。見 [`math/初始權重尺度推導.md`](math/初始權重尺度推導.md) 步驟 8.4 / 10。
- ~~`example/train_conv_compressed.py` 開訓前的 firing-rate 校準 pass 現在是 dead
  code`~~ **已處理(2026-09-11)**:確認沒有消費者之後,連同 `salt_core/calibrate.py`
  整支移除,見 [`問題紀錄.md`](問題紀錄.md) §12 的更新。

## 需要 CSNN-FPGA 硬體做出來才能處理

- **同分時間戳記的 tie-break 約定。** 訓練端建佇列時 tie-break 規則必須照抄硬體
  實際順序(換順序,同一組輸入最後留在神經元裡的 $V$ 不同——反例見
  [`問題紀錄.md`](問題紀錄.md)「複合鍵排序」節)。硬體還沒做出來,沒有「現有規則」
  可讀;之後由訓練端 / 硬體端訂一個簡單的共同約定(例如同分一律照 channel 編號)。
