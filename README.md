# Spiking-Affine-Lazy-Training

CSNN-FPGA(一個獨立的硬體專案)的**訓練端**。

FPGA 那邊的推論電路是**事件驅動、逐筆處理**的:硬體收到一筆事件,就在當下用
「這筆事件距離這顆神經元上次更新經過的真實時間」做解析衰減、加權重、判斷要不要
fire —— 不等下一筆事件,也不湊固定時間格(tick)。這個專案訓練出的模型,forward
行為必須跟這套逐事件推論**完全對得上**,而且要支援 conv,不是只有全連接。

## 為什麼不直接用 spikingjelly 這類套件

`LIFNode` 是**逐 tick** 更新:`v = v·(1-1/τ) + x[t]`,`t` 只是張量索引。
一個 tick 內不管累積多少輸入事件,tick 結束只判斷一次要不要 fire。
例:`v_th=1`、同 tick 內 `w1=w2=1`、`x[t]=2` → 只 fire 一次,多出的 `1.0` 消失。

硬體對不上這件事:硬體收到一筆事件時,**不知道這個 tick 之後還會不會有更多事件**,
只能等收到更晚的事件才能確定可以結算。實測會導致同一層內不同神經元的輸出時間戳
新鮮度不一致,下游收到的事件順序可能違反「時間非遞減」的假設。

查過的其他現成框架(mlGeNN 底層是固定 dt 網格模擬、SparseProp 針對 autonomous
recurrent 網路、jaxsnn 只驗證過小型 dense 網路、ADSEQ 太新沒公開程式碼)也都不是
完全合適的候選,所以決定自己開發。

## 換成什麼

**單狀態事件驅動 LIF**:只有一個膜電位 $V$,沒有電流變數 $I$,輸入直接加進 $V$。

$$V \leftarrow V\cdot(1-1/\tau)^{N} + w, \qquad s = \mathbb{1}[V \ge v_{th}], \qquad
t_{\text{輸出}} = \text{觸發這次結算的事件自己的時間}$$

$N$ = 距上次更新經過的整數 ms(離散 Euler 衰減,不是連續 $e^{-\Delta t/\tau}$——
對齊 FPGA 規格的整數 ms 精度)。

**關鍵引理**:純衰減($a\in(0,1]$、沒有新的 $w$ 加進來)永遠不會讓 $V$ 變大,只會
更接近 0。所以「要不要 fire」永遠在事件到達當下就能封閉式決定,不需要等未來。
輸出時間戳直接是觸發事件自己的 $t$,搭配嚴格序列化處理,一層的輸出天生時間遞增,
遞迴到多層整條 pipeline 都對——不需要 watermark / reorder buffer。

**跟 spikingjelly 是不同的函數**,不是同一演算法的等效實作:同 tick 內 `w1=w2=1`,
spikingjelly fire 一次,逐事件處理是兩次獨立判斷(E1 fire、reset;E2 再 fire)。
**不能沿用 spikingjelly 訓練好的權重。**

## 為什麼要平行化(associative scan)

逐事件把序列長度從幾百步拉到幾千幾萬步。逐步 BPTT 是 $O(S)$ 個序列相依步驟
(不能平行)+ 記憶體隨長度線性成長。

單狀態模型每一步就是一個仿射變換 $m_i(x) = a_i x + w_i$;仿射變換可以先合併
(函數合成有結合律),把 $O(S)$ 改寫成 $O(\log S)$ 深度的平行前綴掃描
(`jax.lax.associative_scan`)。

工程骨架(仿射映射、`combine`、chunk 投機執行)借自 Bullet Trains(ICML 2026),
**但不套用它的神經元動力學**——它是兩狀態模型($\tau_m\dot V = -V + I$、
$\tau_s\dot I = -I$),$I>0$ 時電壓能在沒有新事件時單靠殘留電流爬升,所以需要
root solver;單狀態模型沒有這個性質(上面的引理),不需要那整套。

## 結構

| 資料夾 | 定位 |
|---|---|
| `salt_core/` | 核心運算,自成封閉系統。單狀態仿射 LIF、chunk 化 associative scan 前向 / 梯度、事件佇列建構、surrogate gradient |
| `data/` | N-MNIST 載入 + 視覺化,不依賴 `salt_core/` |
| `example/` | 拿 `salt_core` + `data` 組一個實際能訓練的模型:conv 網路、訓練腳本、動態容量放大、評估。換資料集 / 換架構改這裡 |

- 架構(四層切分、`EventStream` 約定、解碼器):[`docs/架構.md`](docs/架構.md)
- 規格(dataset 格式、網路超參):[`docs/規格書.md`](docs/規格書.md)
- 數學推導:[`docs/math/`](docs/math/)
- 待辦:[`docs/TODO.md`](docs/TODO.md)

## 安裝與執行

```
pip install -e . --no-deps          # 依賴清單見 requirements.txt
python -m example.train_conv_compressed configs/conv/baseline.yaml
python -m example.train_conv_compressed --resume experiments/<run 目錄>   # 行程當掉後接著練
pytest
```
