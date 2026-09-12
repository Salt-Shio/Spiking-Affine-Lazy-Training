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
- **`max_steps` 與 `chunk_size` 脫鉤(conv 層)。目前最優先。** `ConvLayer.__call__`
  傳 `max_steps=self.L`,不看 `chunk_size`;FC 層已用
  `ceil(輸入流長度 / chunk_size)`。現在 conv `chunk_size=1` 沒差,一調大就無效
  ——`lax.scan` 不能提前退出,會 fire 的層最壞情況(每個事件都 fire)強制
  `max_steps=L`,`chunk_size>1` 只是每步做更多事、步數不變(負優化)。
  **數學推導(不是猜)見 [`math/掃描步數上界推導.md`](math/掃描步數上界推導.md)**:
  用「進入任一事件前 $V<v_{th}$」的不變量證明「能 fire 的事件必要條件是
  $b_i>0$」,推出比 $L$ 更緊的 spike 數上界 $m^*=\min(m,\lfloor S/v_{th}\rfloor)$
  (數正負號 vs 用權重大小/門檻算能量預算,取更緊的),進而得到步數上界
  $T\le m^*+\lceil(L-m^*)/\text{chunk\_size}\rceil$。實作規劃:建構層物件時用
  初始權重算一次 $m^*$ 當起點;每次 `grown_to_fit` 的檢查點(`L` 變或想重估)
  重算一次;中途權重正負號翻轉導致上次估的 `max_steps` 不夠(佇列在
  `max_steps` 步內沒被吃完)時,新增一個「佇列有沒有吃完」的診斷訊號
  (`_run_layer_scan` 內部已有 `pointer`,`run_layer_forward` 目前沒往外傳),
  照現在 `L` 出界一樣的方式處理(退回 checkpoint、用當下權重重算、重編譯、
  續跑)。「數學上界當主力、出界重試當安全網」,不是純粹憑感覺猜一個數字。
- ~~`max_out_spikes` 接上 `spike_step_upper_bound` 的 `m*`~~ **已嘗試、已撤
  回,不要再做。** 洞見見 [`問題紀錄.md`](問題紀錄.md)「§十三」「§十四」:
  `Σm*_i`(推論5)數學上是有效上界沒錯,但在這個專案實際權重規模
  (`init_k=5.0` 量級跟門檻相近甚至更小)下鬆到離譜——實測某 batch 算出
  90185,真實峰值只有 21706,把幾乎整份出界測試矩陣打壞成「一開始就誤判出
  界重來」。`max_steps` 用的 `T=m*+⌈(L-m*)/chunk_size⌉` 沒有這個問題(對
  `m*` 的敏感度被 `chunk_size` 打折、又天生封頂在 `L`),但 `Σm*_i` 是把已經
  鬆的量對 N 顆神經元線性加總,沒有任何上限,兩者不能類比。`max_out_spikes`
  該用的一直是 `n_out_spikes`(真實觀察值),不需要也不該借用 `max_steps`
  的機制。
- **`example/metrics_log.py` 沒跟上 `max_steps` 這個新容量旋鈕,而且現有排版
  本來就擠。** `ConvLayer` 這次多了 `max_steps` 欄位(第三個會出界的容量,
  見上面「`max_steps` 與 `chunk_size` 脫鉤」那條),但 `MetricsLog` 完全沒
  更新去接:`start_epoch`/`record_batch` 的 `_obs` 只累積 `queue`/`out`
  兩欄,沒有 `steps`;`finish_epoch` 組的 row 只有 `{name}_L`/`{name}_max_out`,
  沒有 `{name}_max_steps`,`min_steps_needed` 的批次觀察值也沒被記錄,`.csv`
  自然也沒有這欄;`_print_progress` 的 `cap_str` 跟 `print_summary` 都只印
  `L`/`max_out_spikes`,看不到 `max_steps` 現在是多少、用量多接近它。
  同時 `_print_progress`(`example/metrics_log.py`)現在這行本身已經很擠——
  一行塞 loss/val_acc/每層兩個容量(逗號分隔跟空白分隔混用)/每層 firing
  rate/dormant,越後面的欄越難掃到,直接照抄現有格式再加一個 `max_steps`
  只會更難讀,要重新想排版(例如每層一個容量小區塊、或分行),不是單純加
  一段字串接上去。 現在 `grown_to_fit` 只單向長大。fire rate 訓練中
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
