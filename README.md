# Spiking-Affine-Lazy-Training

SALT-FPGA 的**訓練端**。
訓練出的模型 forward 要跟 FPGA 的逐事件推論**逐筆對得上**,支援 conv 跟 FC。

## 重點

- **逐事件,不是逐 tick**:每筆事件到達就衰減、加權重、判斷 fire,不湊固定時間格。
- **單狀態 LIF**:只有膜電位 $V$,沒有電流 $I$。
- **平行化**:每一步是仿射變換,用 associative scan 把 $O(S)$ 序列步驟壓成 $O(\log S)$ 深度。
- **量化**:浮點權重轉成整數模型,只用整數運算推論,輸出逐位元可重現。

## 神經元模型

$$V \leftarrow V\cdot(1-1/\tau)^{N} + w, \qquad s = \mathbb{1}[V \ge v_{th}]$$

- $N$:距上次更新的整數 ms(離散 Euler 衰減,對齊 FPGA 的 ms 精度)。
- 輸出時間戳 = 觸發這次 fire 的事件自己的時間。
- 引理:純衰減只會讓 $V$ 更接近 0,所以 fire 與否在事件到達當下就能決定。
- 結果:每層輸出天生時間遞增,不需要 watermark / reorder buffer。

## 為什麼不用現成框架

訓練端的 forward 要跟硬體是**同一個函數**,要同時做到:

1. 每筆事件到達當下就決定 fire,不等同一時間格的其他事件。
2. 單狀態神經元,整數 ms 的離散衰減 $(1-1/\tau)^N$。
3. 支援 conv。
4. 能在 GPU 上平行訓練幾千到幾萬筆事件的長序列。

查過的現成框架,沒有一個同時做到這 4 點。

- **逐時間格(tick)模擬的框架**做不到第 1 點:同一格內的輸入先加總,格子結束才判斷一次 fire。
  - 硬體收到事件時不知道這格後面還有沒有事件,做不到「等格子結束」。
  - 例:`v_th=1`、同一格 `w1=w2=1`,逐 tick fire 一次,逐事件 fire 兩次。
  - 是不同的函數,權重不能沿用。
- **事件驅動的做法**:實際查證過的是 Bullet Trains(見下方致謝),做不到第 2 點。
  - 它是兩狀態(電流驅動電壓)、連續時間指數衰減的模型,跟硬體不是同一個函數。

## 致謝:snn-bullet-trains

這個專案的平行化思路大量受到 [snn-bullet-trains](https://github.com/ToddMorrill/snn-bullet-trains) 啟發。
對應論文:*Bullet Trains: Parallelizing Training of Temporally Precise Spiking Neural Networks*(Morrill, Pehle, Zador,ICML 2026)。

- **借來的**:事件當仿射映射、`combine` 合成、associative scan、chunk 投機執行。
- **沒借的**:兩狀態 $(V, I)$ 動力學跟 root solver。
  單狀態模型的 fire 判斷是封閉式,不需要求根。
- 筆記:[`docs/math/bullet-trains 核心仿射概念.md`](docs/math/bullet-trains%20核心仿射概念.md)

## 目錄

| 資料夾 | 內容 |
|---|---|
| `salt_core/` | 核心:層、網路、容量;浮點 backend(chunk 化 scan、surrogate gradient)、`quant/` 整數 backend |
| `data/` | N-MNIST 載入與視覺化,不依賴 `salt_core/`;資料集放 `data/datasets/N-MNIST/` |
| `viz/` | 通用繪圖(曲線、網格圖、動畫) |
| `example/` | 實際模型:conv 網路、訓練、評估、量化、分析、notebook。換資料集 / 架構改這裡 |
| `configs/` | `conv/` 訓練、`quant/` 量化的 yaml |
| `tools/` | `pack_snapshot.py`:把不進 git 的本機檔案打包到另一台機器 |
| `archive/` | 封存的 bug 查證證據,不維護 |

## 執行

需要 Python ≥ 3.11,依賴見 `requirements.txt`(JAX CUDA 13)。

```
pip install -e . --no-deps

# 訓練
python -m example.train configs/conv/baseline.yaml
python -m example.train --resume experiments/<run>          # 中斷後接著練

# 評估
python -m example.eval_test experiments/<run> [--which test|val]
python -m example.plot_eval experiments/<run>

# 量化
python -m example.quantize configs/quant/baseline.yaml
python -m example.quantize --check experiments/<run>/quant/<資料夾>   # 比對參考輸出
python -m example.quantize --eval  experiments/<run>/quant/<資料夾>

pytest
```

## 文件

| 文件 | 內容 |
|---|---|
| [`docs/架構.md`](docs/架構.md) | 分層、`EventStream` 約定、backend、容量、解碼器 |
| [`docs/規格書.md`](docs/規格書.md) | dataset 格式、網路超參 |
| [`docs/量化模型推論.md`](docs/量化模型推論.md) | 拿量化資料夾只用整數跑推論(給 FPGA 端) |
| [`docs/監測規格.md`](docs/監測規格.md) | 訓練監測要記的量 |
| [`docs/問題紀錄.md`](docs/問題紀錄.md) | 設計決策與理由 |
| [`docs/寫法規則.md`](docs/寫法規則.md) | docstring / 註解規則 |
| [`docs/math/`](docs/math/) | 數學推導 |
| [`docs/TODO.md`](docs/TODO.md) | 待辦 |
