# Bullet Trains 數學筆記

來源:`snn-bullet-trains/`(ICML 2026,*Bullet Trains: Parallelizing Training of
Temporally Precise Spiking Neural Networks*,Morrill / Pehle / Zador)。
GitHub:https://github.com/ToddMorrill/snn-bullet-trains

本文件記錄:神經元怎麼 leak、怎麼判斷 fire、多層怎麼連接做 forward、梯度怎麼算、
以及「一筆一筆的事件為什麼沒有訓練得很慢」的機制。所有結論都對到程式碼行號,
不是猜測。

---

## 1. 神經元狀態與 leak

每顆神經元帶兩個狀態變數 $(V, I)$(電壓、突觸電流),服從線性 ODE
(`snn/configs/lif_params.py:14-15`):

$$\tau_{mem}\dot V = -V + I, \qquad \tau_{syn}\dot I = -I$$

給定經過時間 $dt$、起始狀態 $(V_0, I_0)$、且這段期間**沒有新輸入**,解析解是
(`snn/solvers.py:322-341`,函式 `voltage_state` / `input_state`):

$$I(dt) = I_0\, e^{-dt/\tau_{syn}}$$

$$V(dt) = V_0\, e^{-dt/\tau_{mem}} + I_0 \cdot \text{ratio} \cdot
\big(e^{-dt/\tau_{syn}} - e^{-dt/\tau_{mem}}\big), \qquad
\text{ratio} = \frac{\tau_{syn}}{\tau_{mem}-\tau_{syn}}$$

這就是「leak」——純衰減,$dt$ 可以是任意實數,不用切成固定時間格。

**關鍵一步**:這個解可以寫成**仿射映射**(`snn/dynamics.py:337-367`,
`NeuronBase.create_element_state`):

$$\begin{pmatrix}V_{new}\\I_{new}\end{pmatrix} =
\begin{pmatrix}a_{00} & a_{01}\\0 & a_{11}\end{pmatrix}
\begin{pmatrix}V_0\\I_0\end{pmatrix} + \begin{pmatrix}0\\w\end{pmatrix}$$

其中:

- $a_{00} = e^{-dt/\tau_{mem}}$
- $a_{11} = e^{-dt/\tau_{syn}}$
- $a_{01}$ = 上面那個 ratio 項(用 `expm1` 算以保留小 $dt$ 時的精度)
- $w$ = 這個區間**末端**到達的那筆 input spike 的權重,打進 $I$(不是直接打進 $V$)

「一個事件 = 一個仿射映射」是後面能平行化的根本原因。

---

## 2. Spike(fire)判斷與 reset

每消化完一段區間,檢查新的 $(V,I)$ 會不會讓電壓在**尚未到達下一筆輸入之前**就
衝過 $v_{th}$。做法是解 $V(t)=v_{th}$ 這個方程的根:

- **特例封閉解**(當 $\tau_{mem}=2\tau_{syn}$,是預設參數):
  `snn/solvers.py:47-87`,`analytic_spike_time_solver`,直接解出對數形式的根。
- **一般情況**:`snn/solvers.py:120-156`,Newton-Raphson 疊代求根(13 步收斂到
  機器精度);另有 `bisection_solver` 當備用求根器。

找到的 spike time 就是這顆神經元「輸出事件」的時間戳,精確到浮點數精度,不是
量化到某個時間格。Fire 之後**硬重置**:$V \leftarrow v_{reset}=0$
(`snn/dynamics.py:625`),$I$ 不重置。

---

## 3. 連接怎麼做 forward(跨神經元、跨層)

一層 `Linear`(`snn/model.py:118-330`)做的事:拿到上一層吐出來的全域事件佇列
(`raw_time`, `source_idx`),對每個輸出神經元:

1. 用 `delay[out, source_idx]` 把每筆事件的到達時間往後推(可訓練的延遲)。
2. 按新時間排序,得到這個輸出神經元自己專屬的事件到達順序(`sort_idx`)。
3. 讀取時做雙重 gather:`raw_time[s] + delay[out, source_idx[s]]` 拿到真正到達
   時間,`synaptic_weight[out, source_idx[s]]` 拿到這筆事件對這顆神經元的權重 $w$。

佇列本身**不物化排序後的完整陣列**,只存一份 `(N_out, S)` 的 permutation
(`sort_idx`),讀取時才用上面的雙重 gather 現算——這是為了省記憶體
(細節見 `snn-bullet-trains/IMPLEMENTATION-NOTES.md` 「Queue structure」一節)。

然後每顆輸出神經元各自跑第 1、2 節的迴圈:依序消化自己佇列裡的事件
(decay + 加權重進 $I$),邊消化邊檢查會不會 fire,fire 就吐一個新事件
(自己的 `raw_time`)、重置、繼續消化剩下的佇列。整層跑完,輸出又是一份全域事件
佇列,直接餵給下一層的 `Linear`。堆疊多層就是這樣一路傳下去,最後一層通常接一個
不會 fire、只做洩漏累積的 `LINeuron`(`snn/dynamics.py:1195`,
`li_associative_scan`)當作讀出層(讀它的膜電位軌跡當 logits)。

**目前只支援全連接**:`Linear` 的 `delays_transformed` 是 `(N_out, N_in)` 密集
矩陣,`_fused_delay_argsort`(`snn/model.py:204-278`)用
`block_delays[:, source_idx]` 對每個輸出神經元 gather 全部輸入事件,等於假設
「每個輸出神經元都連到每個輸入神經元」,複雜度 $O(N_{out}\times S)$。這是
conv 架構要換掉的地方,不影響第 1、2 節的核心運算。

---

## 4. 梯度下降訓練

讀出層的電壓算出 logits,過 `softmax_cross_entropy` 算 loss
(`snn/train.py:142`);`jax.value_and_grad` 對整個網路參數(weight、delay 等)
求梯度(`snn/train.py:212`);用 `optax`(標準 Adam 系列優化器)做梯度下降更新
(`snn/train.py:826-863`)。這一段是教科書標準流程,沒有特殊之處。

真正特別的是梯度怎麼「流過」spike 這個天生不可微分的判斷:

- **傳統作法**(spikingjelly 等)是拿 surrogate gradient 硬湊一個平滑函數去
  近似 Heaviside 的梯度。
- **Bullet Trains** 改成對「spike time 方程的根」直接用
  **implicit function theorem** 求梯度(`snn/solvers.py:159-183`,
  `_ift_solver_bwd`)。設 $R(V_0,I_0,\theta,t^*)=V(t^*)-v_{th}=0$($t^*$ 是
  spike time,$\theta$ 是權重/延遲等參數),對 $\theta$ 兩邊微分:

$$\frac{dt^*}{d\theta} = -\frac{\partial R/\partial\theta}{\partial R/\partial t}$$

這是精確梯度,不是近似。Newton-Raphson 與 bisection 兩個求根器共用同一個反向
傳播(因為兩者解的是同一個根,只是找法不同)。

**記憶體上的額外工程**:對整條 chunked associative scan 直接做 `jax.vjp` 會
在反向傳播時把整個 queue(`raw_time` 等)的餘切張量在 `max_steps` 個 chunk
上重複累積,batch=128 時會 OOM(見 `IMPLEMENTATION-NOTES.md` 實測數字)。
`snn/dynamics.py` 的 `run_events_scan` 因此手刻 `jax.custom_vjp`:反向傳播時
逐 chunk 重新只 gather 該 chunk 的那一小片佇列,算完局部 VJP 後
scatter-add 進一個稀疏梯度緩衝區,把梯度記憶體從「$\text{queue 長度}\times
\text{步數}$」壓回「queue 長度」。這是工程優化,不影響梯度值本身。

---

## 5. 為什麼「一筆一筆的事件」沒有訓練得很慢

如果真的一顆神經元、一筆事件、老老實實照第 1-2 節那樣一步接一步算,那就是
長度 $S$(事件數)的**序列迴圈**,GPU 平行不起來,訓練會非常慢——這正是
`snn/sequential_backend.py` 那個版本在做的事,它被留著純粹當「正確答案的對照
組」(檔案開頭自稱 "correctness oracle"),不是拿來實際訓練大模型用的。

### 5.1 仿射映射的結合律

回到第 1 節「一個事件 = 一個仿射映射」的寫法:**仿射映射的合成滿足結合律**。

$$(map_3 \circ map_2) \circ map_1 = map_3 \circ (map_2 \circ map_1)$$

要算出「消化完前 $k$ 筆事件後的狀態」,不必老實地從第 1 筆按順序疊代到第 $k$
筆——這是經典的 **parallel prefix scan** 問題(跟平行前綴和是同一類問題),
可以用 $O(\log S)$ 深度的樹狀歸約算出來,而不是 $O(S)$ 深度的序列迴圈。
`jax.lax.associative_scan`(`snn/dynamics.py:482-503`,
`NeuronBase.combine` 定義合成運算子)就是做這個樹狀合成。GPU 有幾千個核心,
「$S$ 個仿射映射各自獨立算」是完全平行的,「樹狀合成」深度只有 $\log S$
層——速度從 $O(S)$ 序列步驟壓到 $O(\log S)$ 的來源就在這裡。

### 5.2 真正的難題:reset 打斷了線性可合成性

fire 之後會硬重置 $V$,重置是一個**不連續跳變**,不是仿射映射能表達的東西。
如果佇列裡真的有事件會觸發 fire,「前後兩段」就不能直接用仿射映射合起來算。

論文的解法是 **speculative execution(投機執行)**,很像 CPU 分支預測
(`snn/dynamics.py:679-732`,`_associative_scan_chunk`):

1. 把事件佇列切成固定大小的 `chunk`(如 128 筆,`config.dynamics.chunk_size`)。
2. **先假設整個 chunk 裡都不會 fire**,大膽用 5.1 節的平行樹狀合成,把整個
   chunk 內每個位置的狀態序列一次算出來(平行、$O(\log \text{chunk\_size})$
   深度)。
3. 平行檢查 chunk 內每個區間「假設沒重置」的話電壓會不會衝過門檻,找出
   **第一個**真正觸發 fire 的位置(`emission_idx = jnp.argmax(will_spikes)`)。
4. **這個位置之後算出來的東西全部丟掉**(因為那些都是在「假裝沒重置」的錯誤
   前提下算的,一旦真的 fire、重置發生,後面全部作廢),只保留到
   `emission_idx` 為止的正確狀態,在那裡真正執行重置、吐出一個輸出事件
   (`_advance_chunk`)。
5. **外層(chunk 跟 chunk 之間)才是真的序列迴圈**(`snn/dynamics.py:73`,
   `jax.lax.scan`),因為下一個 chunk 要從「重置後的正確狀態」接著算,這無法
   避免。

### 5.3 為什麼這樣划算

代價是每個 chunk 最多只保證正確地消化到第一個 fire 為止,浪費了 chunk 內
「fire 之後那段」的運算——但這筆浪費很划算:

- chunk 內的運算原本就是平行做的(不占序列深度),白算一些也只是多用一點
  算力,不拉長時間。這是經典的「用總運算量換序列深度」的 scan 平行化取捨,
  跟平行前綴和、平行卡爾曼濾波是同一類技巧。
- 一顆神經元通常不會每收到一筆事件就 fire 一次(大多數輸入只是累積電流,
  不會馬上觸發),所以一個 chunk(比如 128 筆事件)往往可以一次性正確處理
  到底,**外層序列迴圈的步數遠少於總事件數 $S$**——外層迴圈長度大約是
  「這顆神經元總共會 fire 幾次」的量級,不是「收到幾筆事件」的量級,這是
  速度的另一半來源。

實測數字(`snn-bullet-trains/README.md:5`、Figure 1/3):最高 **44 倍**於
序列版本的加速。

---

## 6. 快速對照表

| 概念 | 位置 |
|---|---|
| 兩變量 LIF 參數($\tau_{mem}, \tau_{syn}, v_{th}$) | `snn/configs/lif_params.py:12-22` |
| leak 解析解(`voltage_state`/`input_state`) | `snn/solvers.py:322-341` |
| 封閉解 spike-time solver | `snn/solvers.py:47-93` |
| Newton-Raphson / bisection solver | `snn/solvers.py:119-247` |
| IFT 反向傳播(共用) | `snn/solvers.py:159-183` |
| 硬重置 | `snn/dynamics.py:369-373`, `:625` |
| 仿射映射建構(單一區間) | `snn/dynamics.py:337-367` |
| 仿射映射合成運算子 | `snn/dynamics.py:323-335` |
| 平行 state 序列(associative_scan) | `snn/dynamics.py:482-503` |
| 序列版 state 序列(oracle) | `snn/dynamics.py:461-480` |
| chunk 內投機執行主邏輯 | `snn/dynamics.py:679-732` |
| 外層 chunk-to-chunk 序列 scan | `snn/dynamics.py:73` |
| 全連接 Linear 層(delay + weight + argsort) | `snn/model.py:118-330` |
| 佇列 permutation 表示法(不物化 sorted queue) | `IMPLEMENTATION-NOTES.md` §"Queue structure" |
| 反向傳播記憶體優化(custom_vjp) | `IMPLEMENTATION-NOTES.md` §"Sparse custom-VJP" |
| loss / optimizer | `snn/train.py:128-226`, `:826-863` |
| sequential backend(正確性 oracle) | `snn/sequential_backend.py` |

---

## 7. 跟這個專案的關聯

- 第 1、2 節(leak、spike、reset)的解析解與求根器,是「一筆事件到達就用經過
  的真實時間解析衰減、判斷 fire」這個核心需求的直接對應,不需要改。
- 第 3 節末段提到的「全連接假設」,是唯一要換成 conv 版佇列建構邏輯的地方,
  詳見 `D:\Project\CSNN-FPGA\docs\SNN\Concept\conv_event_scatter_banking_derivation.md`。
- 第 5 節的 speculative execution 機制,理論上不依賴全連接或 conv 的差異
  ——它只依賴「事件佇列 + 仿射映射 + reset 斷點」這個結構,conv 版佇列一樣
  可以套用同一套 chunk 平行化,但沒有實測驗證過。
- **模型落差已查證、已定案(不是懸而未決的問題)**:CSNN-FPGA 硬體端與
  spikingjelly 用的是**單變量** LIF($h[m]=v[m-1](1-\tfrac1\tau)+\tfrac{x[m]}\tau$,
  輸入直接打進 $V$),Bullet Trains 是**雙變量** current-based LIF(輸入打進
  $I$,$I$ 再驅動 $V$)——兩者是不同的微分方程,不是同一個模型的兩種參數化,
  Bullet Trains 不會退化成單變量模型。**結論:Bullet Trains 現成的神經元
  類別/訓練程式碼(root solver、IFT 反向傳播)不能直接套用**,套用了反而是
  在解一個單變量模型根本不存在的問題(單變量模型的 fire 判斷永遠封閉式、
  立刻決定,不需要求根)。要借的只有第 5 節的平行化**工程技巧**(仿射合成、
  associative scan、chunk 投機執行),不是它的神經元動力學——這正是本文件
  第 5、6 節記錄的機制,跟第 1、2 節的兩變量解析解、root solver、硬重置
  本身無關,不需要跟著改。詳見 `../../README.md`「換成什麼」節、
  `../../README.md`「更新:Bullet Trains 的神經元模型跟目標模型對不上」。
