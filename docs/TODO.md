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

- **SNN 活動 monitor,ReDo 的前置。** 要記哪些資料、放哪一層、什麼形式,設計討論見
  [`監測規格.md`](監測規格.md)(尚未定案)。現在只有 `LayerDiag`(4 個純量)+ `dormant.py`
  (逐 epoch 一個 `dormant_frac`),且 `dormant.py` 在 `example/` = dev 自己寫,應搬進
  `salt_core`。ReDo 要靠 activity 分布挑回收對象,單一 frac 不夠;debug SNN 訓練也會需要。
  開放題「`init_k=√3` + ReDo package」往下走前先補這個。
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
- **`example/train_conv_compressed.py` 開訓前的 firing-rate 校準 pass 現在是 dead
  code**(沒有 config 用 `init_k: null`)。要不要整段移除是選配清理;
  `salt_core/calibrate.py` 本身留著給未來全新架構用。

## 需要 CSNN-FPGA 硬體做出來才能處理

- **同分時間戳記的 tie-break 約定。** 訓練端建佇列時 tie-break 規則必須照抄硬體
  實際順序(換順序,同一組輸入最後留在神經元裡的 $V$ 不同——反例見
  [`問題紀錄.md`](問題紀錄.md)「複合鍵排序」節)。硬體還沒做出來,沒有「現有規則」
  可讀;之後由訓練端 / 硬體端訂一個簡單的共同約定(例如同分一律照 channel 編號)。
