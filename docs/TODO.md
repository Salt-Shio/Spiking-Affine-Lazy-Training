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
  **`example` 消費端 2026-09-13 整段改版**(見 [`監測規格.md`](監測規格.md) §7):
  原本的 `example/trace_probe.py`(週期性探測、`summary.npz`/`full_epoch_XXX.npz`)、
  `example/inspect_traces.py`、`example/notebooks/plot_channel_grid.ipynb` 已移除
  ——`summary.npz` 的 `s_value_sum` 證明恆等於 `spike_count`(純重複),
  `full_epoch_XXX.npz` 在 `chunk_size>1` 的層上會不可逆遺失逐事件細節。改成
  `train_conv_compressed.py` 逐 epoch 存純權重快照(`train.weight_snapshot_every`)
  +`example/replay_epoch.py` 事後強制 `chunk_size=1` 重跑 `run_network_traced`
  (forward 結果跟 `chunk_size` 無關,逐位元相同,所以能拿到精確結果)。測試
  `example/tests/test_replay_epoch.py`。
  未完成:`eval_test.py` 加同一套 `--trace` 已確認不必要(`replay_epoch.py` 已經是
  那個獨立分析工具,見監測規格 §7.3);`metrics.csv` 要不要多寫欄位等 ReDo 準則
  定案再決定;§7.3 的 dormant / ReDo 逐神經元活動來源串 ReDo 時再定——監測規格
  §7 已指出這個來源不能沿用 `replay_epoch.py`(頻率太低、非熱路徑),ReDo 真的要做
  需要全新的逐神經元、但夠便宜塞進熱訓練迴圈的活動量設計。
- ~~monitor 這批 code 要做一次結構重整(功能正確,但寫得急)~~ **已處理
  (2026-09-11)**:對照原列的 7 個難聞點逐一確認,6 個是具體技術債,已在這次
  trace 摘要邏輯收進 `salt_core/monitor.py`(§1)、`trace_store.py` 併入(§2)那批
  改動裡解決——`_compile` 快取 key 改用值比較(`==`)不再靠物件 identity;
  `run()` 的累加/平均改用 `jax.tree_util.tree_map`,不再手刻 tree reduce;
  `TraceProbe._records` 改成 `{epoch: {層名: {欄名: 陣列}}}`,賦值即覆寫,
  新增/覆寫只剩一條路;摘要欄名改從 `acc[0].keys()` 動態取,不再兩處抄同一份
  key;npz 扁平命名收進 `salt_core/monitor.py` 的 `pack_key`/`unpack_key`/
  `layer_names`,讀寫兩邊共用;`inspect_traces.py` 的 `report_*` 已拆成
  `_summarize_layer`/`_summarize_vfinal_idle`/`summarize_trace_scalars` 等純函式,
  `report_*` 只負責印,計算可重用。第 7 點(橫跨 `chunk_scan`/`layers`/`monitor`/
  `trace_probe`/`inspect_traces` 五個檔)回頭看是 [`監測規格.md`](監測規格.md)
  §4.1 三 package 分工的自然結果,不是缺陷,不用併。`epochs` 印成
  `[np.int32(0), ...]` 的小 bug 也已用 `.tolist()` 修掉。
- ~~`max_steps` 與 `chunk_size` 脫鉤(conv 層)~~ **已處理(2026-09-12)**:完整
  數學推導見 [`math/掃描步數上界推導.md`](math/掃描步數上界推導.md),實作/
  決策見 [`規格書.md`](規格書.md)「conv 層 `max_steps`」。
- ~~`max_out_spikes` 接上 `spike_step_upper_bound` 的 `m*`~~ **已嘗試、已撤
  回,不要再做**,但 `max_out_spikes` 的縮小本身**已用另一條路做出來
  (2026-09-12)**:直接吃真實觀察值 `n_out_spikes`(跟長大訊號同一個量,
  只是取 epoch 累積最大值),不靠任何理論上界,見 `ConvLayer.
  shrink_max_out_spikes`。洞見見 [`問題紀錄.md`](問題紀錄.md)「§十三」
  「§十四」:`Σm*_i`(推論5)數學上是有效上界沒錯,但在這個專案實際權重
  規模(`init_k=5.0` 量級跟門檻相近甚至更小)下鬆到離譜——實測某 batch 算出
  90185,真實峰值只有 21706,把幾乎整份出界測試矩陣打壞成「一開始就誤判出
  界重來」,已完全撤除。
- ~~`example/metrics_log.py` 沒跟上 `max_steps` 這個新容量旋鈕,而且現有排版
  本來就擠~~ **已處理(2026-09-12)**:`_obs`/CSV/`_print_progress`/
  `print_summary` 都補上 `max_steps` 的追蹤;`_print_progress` 改成每層一行、
  固定 `已用/容量(百分比)` 格式,`L`/`out`/`steps` 換成中文標籤(`佇列`/
  `輸出spike`/`掃描步數`),不再逗號空白混用擠成一行。
- **`L` 目前仍只長不縮。** firing rate 訓練中單調下降(見
  [`問題紀錄.md`](問題紀錄.md)第十節)→ 下游事件變少 → `max_real_queue`
  掉,`L` 理論上有收縮空間,偵測訊號(`max_real_queue`)現成。要做成有
  hysteresis 的啟發式(連續 N epoch `max_real_queue < L × 比例` 才縮一階),
  否則縮完又要長回來 = thrash + 重編譯。優先度低,`max_steps`/
  `max_out_spikes` 都已經有縮小路徑,`L` 是唯一還沒有的。
- **`example/` 單元測試覆蓋缺口。** `example/tests/` 只有一個 e2e 檔
  (`test_train_conv_compressed.py`);`metrics_log` / `checkpoint` / `run_epochs` /
  `build_network` / `build_decoder` / `dormant` / `verify_init_k` 都沒單元測試。
  test 內容可能要先整理再補,補哪幾個、補多深未定。
- **兩個「尾端假事件」機制概念上仍分開。** `core.py` 的 `mask_pad_events`(修
  `AffineMap`)跟 `chunk_scan.py` 的 `n_valid_in_chunk`(修 `s_value`)做的是同
  一件事的兩半。低優先。
- **`docs/math/` 幾份推導文件還有前身時代的殘留**:`D:\...` 絕對路徑、`event_lif/`
  舊資料夾名、指向已廢棄 firing-rate 準則的段落。要清一輪。
- ~~`metrics.csv` 欄名對人不友善,`_print_progress` 的可讀格式沒同步進檔案~~
  **已處理(2026-09-14)**:容量/用量六欄改名成 `max_event_queue`/
  `obs_event_queue`/`max_layer_spikes`/`obs_layer_spikes`/`max_steps`/
  `obs_steps`(`example/metrics_log.py`),`example/train_conv_compressed.py`
  的 `last_epoch_obs` 跟 `example/tests/test_train_conv_compressed.py` 的斷言
  一併跟著改;`plot_metrics.ipynb` 的 `group_metrics_columns` 從「單一字尾對
  單一分組」改成「分組名對字尾清單」(`_METRIC_GROUPS`),同一個資源的
  `max_*`/`obs_*` 現在疊在同一張子圖裡對照,順便補上一直沒跟上的
  `max_steps`/`obs_steps`(2026-09-12 加欄時漏掉,一直各自變成獨立子圖)。
- ~~`act_p90p10`(`salt_core/dormant.py` 的 `dormant_score`)可以拔掉~~
  **已處理(2026-09-14)**:`dormant_score`/`dormant_report` 回傳值、
  `MetricsLog` csv 欄、`plot_metrics.ipynb` 分組清單、
  `salt_core/tests/test_dormant.py`(含拔掉後變死碼的 `_close` helper)、
  `viz/tests/test_epoch_series.py`(借用這個名字當「有 inf 值的欄」範例,改名
  `example_ratio`)、`docs/tmp.md` 的 schema 說明都清掉了。
- **部分視覺化呼叫端程式碼,是不是本來就該用 notebook 處理,要評估。**
  `example/plot_metrics.py`(讀 csv、分組、呼叫 `viz` 畫圖)已經整支搬進
  `example/notebooks/plot_metrics.ipynb`,不再是可以被 pytest 測的獨立模組
  (2026-09-13 討論定案:呼叫端邏輯直接寫在 notebook cell 裡,不留一份可測試的
  `.py`)。這是刻意的取捨,不是疏漏,但代表這類「一次性呼叫、給人看」的程式碼
  跟平常要求的單元測試覆蓋率有衝突,還沒有明確原則判斷「這段邏輯該留在
  `example/*.py`(可測試)還是該進 notebook(互動、不測試)」,之後接觸更多
  視覺化工具會持續碰到,要找時間定一個判準。(原本設想的 `traces/` 那兩種
  資料性質——`summary.npz`/`full_epoch_XXX.npz`——2026-09-13 已經整個移除,
  見監測規格 §7;這個判準問題留給以後其他視覺化工具碰到時再定案。)

- **訓練中期梯度突然炸開,炸完 val_accuracy 回不到炸之前的水準。** 2026-09-14
  用改名後的新版 `metrics.csv`(`configs/conv/baseline.yaml`,40 epochs)第一次
  被人眼看出來——這正是欄名改名/`plot_metrics.ipynb` 分組改版想要達成的效果
  (資料本身早就在,只是之前沒對齊、沒疊在一起看不出趨勢)。實際數字(見
  `experiments/conv_compressed_compressed_maxsteps_verify_20260914_122946/train/metrics.csv`):
  epoch 26 是全程最好的一個 epoch(`val_accuracy=0.845`、`train_loss≈0.0002`,
  幾乎完美自信的分類器);epoch 27 三層 `grad_norm` 同時放大 10 倍以上
  (`out_grad_norm` 0.0017→0.032);epoch 28 徹底炸開(`out_grad_norm=2.74`、
  `train_loss=1.05`、`val_accuracy` 掉到 0.715);之後 12 個 epoch loss 慢慢
  降回 0.01~0.07,但 `val_accuracy` 只回升到 0.75~0.76,再沒回到 0.845。已
  排除是動態放大/出界重跑觸發的(`conv1`/`conv2` 的
  `max_event_queue`/`max_layer_spikes`/`max_steps` 在 epoch 22~32 全程沒變),
  看起來是 loss 逼近 0(近乎完美自信)之後某個 batch 算出異常大梯度,把權重
  跟 optimizer(Adam)動量狀態推到回不去的區域,典型的「過度自信 cross
  entropy 梯度爆炸」模式。還沒查:optimizer 有沒有配 gradient clipping、
  實際去 replay 那個 batch 看數值。優先度看之後要不要繼續訓更深/更久的網路
  再決定。

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
