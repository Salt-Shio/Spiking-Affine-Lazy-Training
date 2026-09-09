# TODO:事件驅動 LIF 訓練工具設計

給新開的 session 讀的完整脈絡,目標是讓新 agent 不用重新推導這份文件記錄的結論。這份文件是本次長對話(討論 FPGA 硬體排序問題 → 訓練端模型選擇)整理出來的結論跟任務清單,任何一步標記完成前,都要先實測驗證過,不能只憑推導就打勾。

**2026-09-06:Yin-Yang 這條線已整個移除**(`data/src/yinyang.py`、`data/viz/yinyang.py`、`data/tests/test_yinyang.py`、`configs/fc/`、`src/train.py`、`src/baseline.py`、`src/evaluate.py`、`src/models/fc_net.py`、`MODEL_REGISTRY`,以及對應的 `notebooks/`、`experiments/`、`results/` 產出物)。任務 6 以下提到 Yin-Yang / `FCNet` / `train.py` 的內容都是移除前的紀錄,保留只為交代沿革。

**2026-09-06:密集版 `ConvNet` 這條線也已整個移除**,只保留 `ConvNetCompressed`(刪 `src/train_conv.py`、`ConvNet`/`ConvNetConfig`、`configs/conv/{baseline,smoke}.yaml`、`src/tests/test_conv_net_forward.py`、`src/tests/test_conv_net_compressed_equivalence.py`)。任務 7、任務 8 段 4 提到「密集版當 ground truth 比對基準」「等價性測試」的部分是移除前的紀錄。`build_conv_queue`(密集版佇列建構)沒刪,`src/conv_param_search.py` 還在用。整體架構翻修的問題清單見 `salt_core/架構問題盤點.md`。

> **2026-09-08:這份 TODO 是前身專案(`snn-event-training`)時代的脈絡與任務清單,已被下列文件取代,不再逐條維護:**
> - `salt_core/架構問題盤點.md` —— `salt_core/` + `src/` 的問題清單與現況。
> - `salt_core/架構翻修規劃.md` —— 層 / 管線架構翻修(step 1–5,已完成)。
> - `src/訓練流程翻修規劃.md` —— 訓練腳本翻修(step 0–6,已完成)。
> - `docs/規格書.md` / `docs/問題紀錄.md` / `docs/math/` —— 規格、問題紀錄、數學推導。
>
> 以下內容(`D:\` 路徑、`event_lif/`、WSL、任務 8 的 test-eval checkbox 等)保留當前身時代參考,狀態不反映現況。仍待處理的 test set 評估腳本改追蹤在 `salt_core/架構問題盤點.md`。

## 為什麼要做這件事

### 1. spikingjelly 原本怎麼運作

`LIFNode`(`decay_input=False`,`τ=16.0`):

```python
for t in range(T):
    v = v * (1 - 1/tau) + x[t]          # x[t] 是這個 tick 全部原始事件已加總完的值
    s = surrogate_function(v - v_th)     # 前向 heaviside,反向 ATan
    v = (1 - s) * v + s * v_reset        # 硬重置
```

`t` 只是張量索引,`τ` 對「一步」無因次,「一步=1ms」是資料端的慣例,`LIFNode` 不知道也不檢查這個對應關係。

**已驗算的關鍵性質**:每個 tick 只判斷一次,不管淨輸入多大,一次 fire 就把 $V$ 砍到 $v_{reset}$。例:$v_{th}=1.0$,同 tick 內 $w_1=w_2=1.0$,$x[t]=2.0$,只 fire 一次,多出的 1.0 直接消失。

### 2. 這導致硬體實作的問題

硬體收到逐筆事件,不知道「這個 tick 還會不會有更多事件」,只能靠**看到下一筆更晚 tick 的事件**才能確定可以結算——CSNN-FPGA 原本 $m_{cur}$ 設計的規則:

$$\text{同 ms}(m=m_{cur})\text{:}\ V\leftarrow V+\tfrac w\tau\qquad\text{跨 ms}(m>m_{cur},N=m-m_{cur}-1)\text{:}\ s=\mathbb1[V\ge v_{th}],\ V\leftarrow(s?0{:}V)(1-\tfrac1\tau)^{N+1}+\tfrac w\tau,\ m_{cur}\leftarrow m$$

**反例**:神經元在 $m=0$ 累加後已超過 $v_{th}$,但同 ms 只累加不判斷;$m=1,2,3$ 是別的常被摸到的神經元先結算送出;$m=4$ 才碰到這顆神經元、觸發結算,輸出時間戳記卻是舊值 $m_{cur}=0$——下游收到的順序是先 $t=1,2$ 才 $t=0$,處理順序不等於真實時間順序,只要不同神經元被摸到的頻率不一樣就結構性必然發生。規則本身也沒定義 $m<m_{cur}$ 該怎麼辦。完整推導見 `conv_output_ordering_and_training_pivot.md`。

### 3. 為什麼換成逐事件訓練

放棄跟 spikingjelly 逐 tick 加總等效,換單狀態模型:

$$V_{new}=V_{old}\cdot(\text{衰減})+w,\quad s=\mathbb1[V_{new}\ge v_{th}],\quad t_{輸出}=t(\text{觸發這次結算的事件自己的時間})$$

**引理(已證明)**:$V(t)=V_{old}\cdot e^{-(t-t_{last})/\tau}$,純衰減不可能把 $V$ 推過門檻,所以「要不要 fire」永遠在事件到達當下就能封閉式決定,不需要等未來。等價於把 spikingjelly 的 $T$ 切到最細——一筆事件一個 tick,零輸入的 tick 天生可以跳過不算,是同一個模型的兩種講法。

**跟 spikingjelly 是不同函數,不是同一算法**:同一 tick 內 $w_1=w_2=1.0$,spikingjelly fire 一次;逐事件處理是兩次獨立判斷(E1 fire、reset;E2 再 fire)——兩次 spike。**不能沿用 spikingjelly 訓練好的權重**。

**排序天生正確(已證明)**:輸出時間戳記直接是觸發事件自己的 $t$(不像 $m_{cur}$ 那樣分離),搭配嚴格序列化處理,一層的輸出事件天生時間遞增,遞迴到多層整條 pipeline 都對,不需要額外的同步機制、watermark、或 reorder buffer。

**同分時間戳記會影響結果**:硬體是決定性電路,順序沒有模糊地帶,但換個處理順序殘留電位不同——反例:$v_{th}=1.0$,三筆同分事件 A($w{=}0.4$)、B($w{=}0.7$)、C($w{=}0.7$),A→B→C 剩 $V=0.7$,B→C→A 剩 $V=0.4$。訓練端建構佇列時,tie-break 規則必須照抄硬體實際順序,不能用通用排序函式隨便決定。

**逐事件訓練的複雜度代價**:序列長度從幾百步暴增到幾千幾萬步。逐步 BPTT 是 $O(S)$ 個序列相依步驟(不能平行)+ 記憶體隨序列長度線性成長——這是第 4 點 associative scan 技巧變成必要、不是錦上添花的原因:靠仿射變換可先合併這個數學性質,把 $O(S)$ 改寫成 $O(\log S)$ 深度的平行前綴掃描。

### 4. 為什麼 Bullet Trains 的技巧可以借

Bullet Trains 是兩狀態模型($\tau_m\dot V=-V+I,\ \tau_s\dot I=-I$),$I>0$ 時電壓能在沒有新事件時單靠殘留電流爬升,所以需要 root solver(`nr_solver`/`bisection_solver`)+ IFT 反向傳播找未來 spike 時刻。單狀態模型沒有這個性質(第 3 點引理),不需要這整套機制——**神經元類別/訓練程式碼不能直接套用**。

值得借的是 `snn/dynamics.py`(`NeuronBase.combine`/`create_element_state`/`_associative_scan_chunk`)已驗證過的平行化技巧:把逐事件仿射變換合併,`jax.lax.associative_scan` 平行掃描,深度 $O(S)\to O(\log S)$,論文量到 44 倍加速。單狀態模型每一步就是仿射變換,完全符合這套技巧的前提,而且比 Bullet Trains 自己的模型更單純(不需要 root solver)。

## 任務清單(照順序做,每項完成要先跑過、驗證過才能打勾)

這是本專案(訓練端,純軟體)自己的任務,不需要碰 CSNN-FPGA 的 BRAM banking、channel 序列化 FSM 這些硬體實作細節——那些是硬體怎麼把運算平行化、怎麼定址的問題,跟「軟體端要用什麼演算法訓練」是兩件事。

**2026-08-26 更新,以下三點是這次討論(見 `docs/math/單狀態仿射平行掃描推導.md`、`docs/math/全連接forward訓練範例.md` 完整推導)新定案的規則,直接改變下面任務清單的內容,不是背景補充**:

- **不修改 `snn-bullet-trains/`**,那份保留當唯讀參考(clone 進來的原始論文程式碼),自己的實作寫在新資料夾 `event_lif/`(跟 `snn-bullet-trains/` 平行,底下自己 `git init`)。可以參考它的作法(仿射映射、`combine`、associative scan、chunk 投機執行這套工程骨架),但程式碼自己重寫,不搬過去改。
- **確認拿掉的兩個東西,不是可以商量的簡化,是這個專案的目標(跟 FPGA forward pass 對得上)要求一定要拿掉**:(1) 電流變數 $I$——FPGA 目標模型只有單一狀態 $V$,輸入直接加進去;(2) 連接延遲 $d_{ij}$——已查證 CSNN-FPGA 文件裡所有「延遲」的提及都只是硬體 pipeline 時序,沒有連接層級的可學習延遲,保留 $d_{ij}$ 會讓模型多出一個硬體不存在的自由度。
- **衰減公式要用離散 Euler $(1-1/\tau)^N$($N$=整數 ms 差),不是連續解析 $e^{-\Delta t/\tau}$**——CSNN-FPGA 規格明確定案維持整數 ms 精度,不換連續時間(見 `conv_event_scatter_banking_derivation.md` 第 14.0、14.5 節)。

1. [x] **建立 `event_lif/` 新資料夾,獨立 git 初始化**。已建立並完成第一次 commit(`a38515c`)。
2. [x] **設計單狀態神經元的仿射合併運算子**:$m_i(x)=a_ix+w_i$($a_i=(1-1/\tau)^{N_i}$),無 $I$、無 root solve,見 `docs/math/單狀態仿射平行掃描推導.md` 第 1、2 節。實作於 `core.py`(`AffineMap`/`combine`/`create_element_state`/`process_chunk`),5 項測試全過。$\tau$/$v_{th}$ 沿用 FPGA 規格舊值,之後再調參。
3. [x] **決定訓練資料的事件時間怎麼量化成整數 ms**:訓練資料本來就是整數 ms 精度,不需要額外量化步驟,直接拿事件時間戳記算 $N=m_i-m_{last}$。
4. [x] **實作 FC 版佇列建構 + chunk 化 associative scan 訓練迴圈**:`fc_queue.py`(`build_fc_queue`)、`chunk_scan.py`(`run_layer_forward`)、多層串接(`layer_chain.py`)全部完成,forward 數字驗證通過。$N$ 計算基準的坑見 `docs/問題紀錄.md` 第七節。batch 維度、訓練速度對比 baseline 排到任務 5 之後(需要梯度才測得出東西)。
5. [x] **驗證 surrogate gradient 銜接**:JAX 版 `atan_spike`,不套閘 + soft reset 設計(理由跟工程好處見 `docs/問題紀錄.md` 第三節,完整推導見 `docs/math/不套閘與soft-reset梯度推導.md`),已整合進 `chunk_scan.py`。多層梯度串接的阻塞項已解決,不再阻塞任務 6——關鍵洞見(`event_gain` 為什麼存在、排序 tie-break 為什麼要用複合鍵)見 `docs/問題紀錄.md` 第四、六節。
6. [x] **小規模訓練跑通(FC)**:Yin-Yang 資料集,4 個里程碑全部完成。5 個 seed(42~46)重跑,peak val_accuracy 95.9%~97.0%,test accuracy 95.6%(baseline logistic regression 62.5%)。訓練不可重現的 XLA 坑、firing rate 下降不是問題的決策,見 `docs/問題紀錄.md` 第八、十節。

   - 還沒做,留給任務 7:逐層超參 per-layer、`FCNet` 通用組裝器(見問題紀錄第二節)
7. [x] **conv 架構擴充(FC → conv1→conv2→FC)**:取代任務 4 的全連接假設,換成局部連接的佇列建構邏輯,核心運算子(任務 2)不用改。資料集/架構規格見 `docs/規格書.md`,JAX gather 負數 index 的坑見 `docs/問題紀錄.md` 第五節。

   - [x] conv 佇列建構(`connectivity/conv.py`,19 測試全過,2026-09-02)
   - [x] `n_real_events` 遮罩邏輯收進 `core.py`(`mask_pad_events`,`fc.py`/`conv.py` 共用,2026-09-03)
   - [x] N-MNIST 資料集實作(`data/src/nmnist.py` + 9 測試全過,2026-09-03)。事件數截斷長度從規格書原定的 8183(全資料集最大值,無截斷)改成 **2000**——原因跟過程見下方「記憶體問題」小節,規格書已同步更新
   - [x] 網路架構(conv1→conv2→FC,`src/models/conv_net.py`)+ 訓練腳本(`src/train_conv.py`)+ config(`configs/conv/baseline.yaml`)
   - [x] `init_k` 實測:規格書候選集 {3,4,5} 用 receptive-field 版 firing rate(分母是神經元自己真正的感受野事件數,不是全部輸入事件數——原始 FC 那套公式套到 conv 會把比例壓到 <0.1%,失真)量測,三個候選都到不了 20~50% 目標(conv1 最高 14.9%@5.0,conv2 最高 2.2%@5.0),使用者決定用最接近的 5.0 直接進訓練,不繼續往上找
   - [x] **POC 訓練跑通,confirmed 模型學得會 N-MNIST**:train 1000/val 200,batch_size=4,40 epochs,約 2 小時(WSL2+RTX 5070 12GB)。train_loss 1.46→0.003(有波動,LR 沒做衰減),best val_accuracy **84.5%**(epoch 19,10 類分類、亂猜基準 10%),結果存在 `experiments/conv_baseline_20260903_035735/`

   **記憶體問題怎麼解的**:conv 事件佇列是密集陣列(長度=事件數,`docs/conv事件佇列建構推導.md` 第 10 節本來就列為擱置項目),`run_layer_forward` 對這個長度做 `jax.lax.scan` 反向傳播,記憶體需求跟長度近似成正比——規格書原定 8183(全資料集無截斷)在這台機器(12GB VRAM,WSL2)連 batch_size=2 都會 OOM。用真實資料視覺化不同截斷長度重建出的影像,發現前 200~1000 筆事件就能重建出完整輪廓(N-MNIST 對同一數字做 3 輪掃視,後面是重複資訊),但抓「輪廓夠用」的最激進長度會砍掉幾乎每個樣本 2/3 的事件;改抓「剛好緩解記憶體問題」的溫和值 **2000**(截斷 96.7% 的樣本,但砍的比例比 200~1000 溫和很多)。另外把 `chunk_size` 榨到 1(純效能參數,不影響正確性,`test_gradient_invariant_across_chunk_sizes_with_two_fires` 已驗證過)、把「事件佇列陣列長度」跟「`run_layer_forward` 要跑幾步 scan」拆成兩個獨立參數(前者留寬給 `extract_output_events` 防止靜默截斷,後者才是真正決定記憶體的量,不能共用同一個數字)。兩者疊加後,規格書原本要的 batch_size=4 反而達成了。WSL2 GPU 額外開銷、XLA BFC allocator 碎片化這兩個環境因素也查過(有實據但查不到精確拆分比例),兩者都不影響上面這條解法路徑。**2000 這個截斷長度是緩解 OOM 的權宜之計,不是真正的解法**——任務 8 的壓縮版才是把記憶體問題從根本解決。

8. [ ] **conv 事件佇列壓縮版**:每顆神經元只存自己真正相關的事件,取代密集版「每顆神經元都存全域事件數長度」的浪費做法(任務 7 記憶體問題小節那個 2000 截斷只是權宜之計)。每段「是什麼」的詳細規格見 `docs/規格書.md`「Conv 事件佇列壓縮版:實作分段規劃」一節,數學推導見 `docs/math/conv事件佇列壓縮版推導.md`。

   - [x] 段 1:壓縮佇列建構核心(`connectivity/conv.py` 新增 `build_conv_queue_compressed`,16 測試全過,含補位 catch-up 衰減的 bug 修正)
   - [x] 段 2+3:`run_layer_forward` 呼叫端驗證 + `extract_output_events` 加 local→global $j$ 查表(`layer_chain.py`,4 項新測試 + 7 個既有測試檔案零回歸)
   - [x] 段 3.5:`data/viz.py` 拆成 `data/viz/{yinyang,nmnist}.py`
   - [x] 段 3.6:`data/` 資料前處理架構重構——`NMNISTDataset`/`YinYangDataset` class 化(`max_events`/`val_fraction`/`T`/`r_big` 等全部改成建構參數,不再是模組常數)、notebooks 轉真正的 Jupyter(`.ipynb`)
   - [x] 段 3.7:conv 訓練端接上 config——`train_conv.py`/`configs/conv/baseline.yaml` 接 `NMNISTDataset`(`max_events` 從 config 讀),刪 `data/src/nmnist.py` 的過渡 shim。**只做 conv 這條線,yinyang 那條線(`train.py`/`baseline.py`/`evaluate.py`)明確決定不遷移**,`data/src/yinyang.py` 的過渡 shim 永久保留
   - [ ] **段編號不是執行順序(2026-09-04 調整)**:實際順序是 3.7(已完成)→ 段 4 → 段 3.8 → 段 5,理由跟完整設計見 `docs/規格書.md`「Conv 事件佇列壓縮版:實作分段規劃」「段 3.8 定案」兩節——init_k 掃描要對上實際拿去訓練的 `ConvNetCompressed`,不是即將降級成參考基準的密集版 `ConvNet`;`ConvNetCompressed` 可以在 $L_1$ 未量測前用開大值先建出來,順序不衝突
   - [x] 段 4:conv1→conv2→FC 整條串接——**定案新開 `ConvNetCompressed` 類別,不改動既有 `ConvNet`**(密集版永久保留當 ground truth 比對基準)。這一段 $L_1$ 還沒量,先用刻意開大、不截斷的值建出來、驗證等價性(相同 config 下 `v_final`/spike mask/輸出事件與 `ConvNet` 一致)——**2026-09-06 補打勾**:這段先前已完成(等價測試 3/3 PASS,含 production N-MNIST 規模),checkbox 之前漏勾,不是這次才做完
   - [x] 段 3.8:`init_k` 掃描 + $L_1$ 量測(2026-09-04 整段重新設計,**任務 7 舊數值 `init_k=5.0`/14.9%@5.0/2.2%@5.0 全部作廢**)——對象是段 4 做出來的 `ConvNetCompressed`。程式拆成 `src/init_k_search.py`(通用 `FiringLayerHooks`/`sweep_init_k`,8 項單元測試)+ `src/conv_param_search.py`(CLI 膠水層,5 項接線測試)+ `conv1_receptive_field_L1_batch`(`src/models/conv_net.py`)。策畫 agent review 過,13 項新測試 + `test_conv_net_forward.py` 回歸全過,零回歸。**production 真實跑分結果**:conv1 init_k=8.0(confirm firing rate 24.16%)、conv2 init_k=64.0(confirm firing rate 21.66%),兩者都在 bracket 倍增階段直接命中目標帶 `[0.20,0.50]`,沒進 bisection;$L_1$(train/val/test/overall)= 185/185/181/185。細節、掃描規模措辭修正(256 筆改成均勻隨機非嚴格分層)見 `docs/規格書.md`「段 3.8 執行結果」小節
   - [x] **段 3.8 收尾待辦(2026-09-05 review 時發現,2026-09-06 完成)**:`ConvNetCompressedConfig`/`_init_conv_params` 拆成 per-layer 欄位(`init_k_conv1`/`init_k_conv2`/`init_k_fc`),`ConvNet`(密集版)不受影響,三個值定案為 `8.0`/`64.0`/`5.0`(`init_k_fc` 決策理由見 `docs/規格書.md`「conv 網路架構」節)。`test_conv_net_compressed_equivalence.py`/`test_conv_net_forward.py` 各 3/3 PASS,策畫 agent 親自跑過驗證,零回歸。`src/models/conv_net.py` 開頭的 UTF-8 BOM 也一併清掉
   - [ ] 段 5:$L_2$ 動態偵測+ checkpoint 續練機制(**正式訓練前的必備條件**,不是可選優化)——每步訓練額外計數真實 spike 數,超過目前 $L$ 就中斷、報真實長度、退回 checkpoint 用新 $L$(乘安全倍率)重編譯繼續;checkpoint 頻率、放大倍率等啟發式參數留給訓練實測後定案
   - [ ] test set 評估(conv 版的 evaluate.py milestone,原本排在任務 7,移過來):等段 4/5 做完、`ConvNetCompressed` 真的訓練出結果之後才寫,直接用壓縮版訓練出的模型接著寫,不回頭用密集版那組 84.5% POC 的結果

## 後續整合階段才需要處理(現在不是阻塞項)

- **同分時間戳記的處理順序約定**:等 CSNN-FPGA 硬體真的做出來、要驗證訓練結果能不能跟硬體 bit-exact 吻合時才需要處理。不是訓練端去讀 CSNN-FPGA 現有的 banking/FSM 規則來配合(硬體都還沒做出來,沒有「現有規則」可讀)——應該是雙方(訓練端、硬體端)先訂一個簡單、雙方都遵守的 tie-break 約定(例如同分時間一律照 channel 編號排序),當作共同介面,不需要訓練端去逆向工程硬體細節。
- **CSNN-FPGA 的硬體規格文件已經照使用者確認過的方向改完**(`D:\Project\CSNN-FPGA\docs\SNN\Concept\conv_event_scatter_banking_derivation.md` 第 9.4、14、15 節,經使用者本人確認才動手,不是自作主張):$m_{cur}$ 改名 $m_{last}$,語意從「累加中的 ms」改成「上次更新的 ms」,**衰減維持 ms 整數精度**,不是換成連續 $\Delta t$(µs 等級的 $\Delta t$ 相對 $\tau$ 幾乎不衰減,算那麼細沒有實際好處,是本文件先前版本想太多)。實際改動只有兩件事:(1) 拿掉「同 ms 只累加不判斷」分支,每筆事件都立刻用 $N=m-m_{last}$ 做 $(1-1/\tau)^N$ 衰減 + 判斷;(2) 輸出時間戳記改成觸發判斷的事件自己的 $m$,不是舊值。$\text{MCUR\_WIDTH}$(25 bits)、$(1-1/\tau)^N$ 查表 ROM 結構完全沿用舊版,只有指數從 $N+1$ 改成 $N$。

## 參考文件位置

| 主題 | 位置 |
|---|---|
| 單狀態仿射映射/合成/平行掃描/reset 簡化完整推導(無 $I$、無 root solve) | `docs/math/單狀態仿射平行掃描推導.md` |
| FC 連接結構、佇列建構、forward/訓練具體數字例子(無 $d_{ij}$) | `docs/math/全連接forward訓練範例.md` |
| Chunk 化 forward 的梯度推導(不套閘 + soft reset,`valid_len` 加總公式,任務 5 的完整正確性依據) | `docs/math/不套閘與soft-reset梯度推導.md` |
| Bullet Trains 完整數學筆記(舊,部分結論已被本文件更新,第 1 節衰減公式有一處符號錯誤待修) | `docs/math/bullet-trains 核心仿射概念.md` |
| NeuroScale 論文(不適用,備查) | `doi:10.1038/s41467-025-65268-z`(檔案目前不在專案目錄,需使用者重新提供才能重讀) |
| Bullet Trains 兩狀態動力學、root solver | `snn-bullet-trains/snn/solvers.py`、`snn-bullet-trains/snn/configs/lif_params.py` |
| Bullet Trains associative scan 結構 | `snn-bullet-trains/snn/dynamics.py` |
| CSNN-FPGA 硬體排序問題完整推導 | `D:\Project\CSNN-FPGA\docs\SNN\Concept\conv_output_ordering_and_training_pivot.md` |
| CSNN-FPGA 硬體規格(banking/pipeline/FSM,待改的第 14 節) | `D:\Project\CSNN-FPGA\docs\SNN\Concept\conv_event_scatter_banking_derivation.md` |
| spikingjelly 逐 tick 遞迴、surrogate gradient 原始碼 | `D:\miniconda3\envs\snn\Lib\site-packages\spikingjelly\activation_based\neuron\lif.py`、`base_node.py`、`surrogate.py` |
