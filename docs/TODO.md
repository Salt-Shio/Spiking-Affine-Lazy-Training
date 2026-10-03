# TODO

還沒做的事。做完的事記在 git,決定跟洞見記在 [`問題紀錄.md`](問題紀錄.md)。
背景見 [`../README.md`](../README.md);架構見 [`架構.md`](架構.md);規格見 [`規格書.md`](規格書.md)。

## 已知限制(已決定接受代價,非阻塞)

- **determinism flag(`XLA_FLAGS=--xla_gpu_deterministic_ops=true`)預設沒開。** 沒開時同一顆 seed
  重跑會有 float 級的分岔。開 flag 時梯度會錯的根因(`_compress_candidates` 的 `mode='drop'` scatter)
  已經改用垃圾桶寫法修掉;開 flag 跑過幾個 epoch,兩次結果 100% 相同。正式尺寸開 flag 的梯度正確性
  還沒另外驗證(要跑很久)。細節見 [`問題紀錄.md`](問題紀錄.md)「洞見:JAX/XLA 在 GPU 上的規約運算
  不保證可重現,連同一顆 seed 都不例外」。

## 待處理

- **量化參數還沒定案。** 權重量化、膜電位量化的工具都做好了(`salt_core/quant/`,推導見
  [`math/權重量化推導.md`](math/權重量化推導.md)、[`math/膜電位量化推導.md`](math/膜電位量化推導.md)),
  還沒選定最後用哪一組:
  - 權重 bit width 跟 clip percentile。PTQ 實測(92.75% 那次的 `best_params.npz`,val 0.9275):8 bits
    0.9265;6、5 bits 0.9230、0.9205;4 bits per-channel 0.8985、per-tensor 0.8830;3 bits 以下崩掉
    (0.7310、0.1640)。4 bits 時 clip percentile 100 → 90 回升到 0.9185,outlier 截斷對低 bit width 有幫助。
  - i_V 要用多少樣本量膜電位範圍。2026-09-27 實測:b=8、f_a=10、f_V=10,用 val 前 50 筆量範圍算出 i_V
    (conv1 12、conv2 14、out 14),跑滿 val 2000 筆後 conv1、out 各有 1 筆暫存器溢位(繞回)。驗證溢位時
    用的是量範圍的同一批樣本,看不到這種情況;要決定量範圍用多少樣本、驗證要不要換一批。
- **量化實驗資料夾(進行中,2026-10-03 定案)。** 量化模型要像浮點版一樣有自己的一份實驗結果:只靠這份
  就能獨立重跑,附逐筆參考輸出,之後拿來比對 FPGA 的執行結果。FPGA 檔案格式不在這次範圍(見最後一節)。
  1. `salt_core/io.py` 加 `save_quantized` / `load_quantized`:網路描述、每層 `QuantizedLayerParams`
     原樣(逐神經元)、`round_mode`、呼叫端給的中繼資料,存成一個 npz。測試:存讀逐值相等、讀回來
     forward 逐位元相同。
  2. 新入口 `python -m example.quantize configs/quant/<x>.yaml`:讀來源 run 的權重 → 量 M → 算參數 →
     跑滿驗證 split(預設整個 val)→ 容量出界就放大重跑 → 寫 `experiments/<來源 run>/quant/<權重來源>_<規格名>/`
     的 `model.npz`、`reference.npz`(逐筆預測、輸出層暫存器值、每層 spike 數、溢位、出界)、`report.yaml`。
     資料夾名由權重來源跟規格自動組成,例如 `best_b8_fa10_fv10_round_pc_clip100_wrap`。
  3. `python -m example.quantize --check <量化資料夾>`:只讀這個資料夾跟資料集,重跑並逐筆比對 `reference.npz`。
  4. `golden_output.py` 的量 M、量化 forward 改用第 2 項的共用函式;改完 `compare_quant` 要 0 差異。
  5. `membrane_quantization` notebook 第 5 步的溢位驗證改用 `VERIFY_N_SAMPLES`(預設整個 val)。
- **`max_queue_len` 只長不縮。** firing rate 訓練中單調下降(見 [`問題紀錄.md`](問題紀錄.md)
  「決策:firing rate 訓練過程單調下降,判斷不是問題、不處理」)→ 下游事件變少 → 佇列需求
  `LayerDiag.needed["max_queue_len"]` 掉,max_queue_len 有收縮空間,偵測訊號現成。要做成有 hysteresis 的啟發式
  (連續 N 個 epoch 需求 < max_queue_len × 比例才縮一階),否則縮完又要長回來,反覆重編譯。優先度低,
  `max_out_spikes`、`max_extra_steps` 都已經會縮。

## 開放題(往下走才需要)

- **剪枝:評估後暫不做。** 原規劃用 `dormant_frac` 當依據做 channel-level 剪枝(理由見
  [`math/剪枝推導.md`](math/剪枝推導.md))。這個網路太小,不划算:
  - conv1 8 channel、conv2 16 channel,FC 只有一層輸出層;剪 2~3 個 channel 就動到 25%~35%。
  - 只有兩層 conv,conv1 的輸出 channel 數就是 conv2 的輸入 channel 數,沒有分散吸收的空間。
  - 92.75% 那次最後一個 epoch 的 dormant_frac:conv1 約 0.497、conv2 約 0.464,但這是逐神經元(空間位置
    × channel),不能讀成「一半 channel 可以砍」,可能只是空間稀疏(例如邊緣偵測器只在邊緣活躍)。
  網路變大、分析顯示真的有冗餘時再評估。
- **conv1 被訓練震盪撞遠之後修不回來。** 查文獻找到兩個方向:firing-rate 正則化(把 dormant 統計的
  活動量接回 loss,懲罰活動量太低的神經元)、調寬 surrogate gradient(調小 `alpha`)。都還沒實作;訓練
  演算法這條線目前已收尾,要重開時再評估。背景見 [`問題紀錄.md`](問題紀錄.md)「conv1 vs conv2 firing rate
  不對稱:震盪的下游後果,不是架構天生如此」。
- **`init_k = √3`(Lee 變異數保持)+ ReDo(訓練中回收休眠神經元)當一個 package。** 目前 baseline
  `init_k = 5.0`、不做 ReDo(2-conv 網路不需要:V6 連 √3 的 70% 休眠都訓到 0.795,`init_k=5` 休眠會自己降)。
  冷 init 的佇列記憶體優勢隨網路深度複利,ReDo 讓冷 init 撐得起會休眠的隱藏層(尤其 FC→FC)。真的要疊深
  FC / 更深 conv 時再評估,見 [`math/初始權重尺度推導.md`](math/初始權重尺度推導.md) 步驟 8.4 / 10。
  - ReDo 需要的逐神經元活動量來源還沒定:`replay_epoch.py` 太慢、不在訓練熱路徑,要另外設計一個便宜到
    能塞進訓練迴圈的量(見 [`監測規格.md`](監測規格.md) §7.3)。

## 需要 CSNN-FPGA 硬體做出來才能處理

- **同分時間戳記的 tie-break 約定。** 訓練端建佇列時 tie-break 規則必須照抄硬體實際順序(換順序,同一組
  輸入最後留在神經元裡的 V 不同——反例見 [`問題紀錄.md`](問題紀錄.md)「洞見:同一時間戳記排序,用複合鍵
  不能只用時間」)。硬體還沒做出來,沒有「現有規則」可讀;之後由訓練端、硬體端訂一個簡單的共同約定
  (例如同分一律照 channel 編號)。
- **量化模型的匯出格式。** 要等 CSNN-FPGA 那邊的規格(架構審查第 3 條)。
