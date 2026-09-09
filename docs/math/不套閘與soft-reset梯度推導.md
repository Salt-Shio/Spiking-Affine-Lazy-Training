# Chunk 化 forward 的梯度:不套閘 + soft reset

延續 `docs/單狀態仿射平行掃描推導.md`(那份文件只推 forward:仿射合成、平行掃描、fire 判斷,不含梯度)。本文件補上梯度怎麼算,對應 `docs/TODO.md` 任務 5「驗證 surrogate gradient 銜接」在 chunk 化版本(`core.py` 的 `process_chunk`)上的具體落地。

## 0. 決策脈絡(為什麼是這個設計,不是別的)

一開始想直接照 spikingjelly 的語意——**每一筆事件,不管有沒有 fire,都套用 $(1-s)$ 這個閘**(`test_surrogate.py` 最早的序列版參考模型就是這樣寫)。但這個語意會讓每一步的仿射係數依賴前一步的輸出($s_i$ 是 $h_i$ 的函數,$h_i$ 又是前一步的輸出),破壞 `associative_scan` 平行合併需要的「係數固定、不依賴輸入」這個前提——要保留這個語意,需要一整套額外的反向 associative scan(把 $g_i=(1-s_i)-\text{slope}_i h_i$ 逐點算出來、串成另一條仿射遞迴去平行算梯度),還要處理「chunk 中間真的 fire、後面事件要遮罩掉」的邊界情況,推導跟實作複雜度都高出很多。

評估過訓練效果、實作複雜度、執行速度三個角度後(對話記錄有完整比較),決定:

- **不套閘**:沒 fire 的事件維持純仿射,不套用任何 surrogate 修正——放棄「近門檻的非 fire 事件也給訓練訊號」這件事,換取平行結構完全不受影響。Bullet Trains 自己的核心梯度機制也是這個做法(近門檻訊號只用一個獨立、可選的正規化項補,不是內建在核心梯度裡),不是沒有先例的簡化。
- **fire 那一點用 soft reset,不用硬常數**:雖然「不套閘」已經放棄了非 fire 事件的訊號,但 fire 這一點本身「超過門檻多少」的資訊值得保留——如果連這裡也用硬常數(對任何權重的梯度天生是 0),整個 surrogate gradient 機制幾乎起不了作用。而且「不套閘」已經把梯度鏈長度從「總事件數」降到「只在 fire 那一點出現一次 $g$」,原本擔心軟 reset 會讓長鏈不穩定的顧慮,強度也跟著降低很多。

**這個組合已經是 `core.py` 現有的實作**(`s_sequence = atan_spike(...)`、`safe_idx`、`v_after_fire = (1-s)*x`),本文件是幫這個既有設計補上正式的正確性推導與數字驗算,不是要改程式碼。

## 1. Forward(不變,沿用既有推導)

一個 chunk 內 $S$ 筆事件,用 `associative_scan` 算出 $x_1,\dots,x_S$(純仿射,假設整段不 reset):

$$
x_i=A_i\cdot v_0+B_i,\qquad(A_i,B_i)=\text{combine}\big((a_1,w_1),\dots,(a_i,w_i)\big)
$$

$k$ 是第一個 $x_k\ge v_{th}$ 的位置(用 `jax.lax.stop_gradient` 明確標成不可微分的離散選擇,只有「有沒有 fire」這個值需要可微分,選中哪個 index 不需要——業界標準做法,Bullet Trains 的 `emission_idx` 也是這樣處理)。

$$
s_k=\text{atan\_spike}(x_k-v_{th},\ \alpha)
$$

$$
v_{final}=\begin{cases}(1-s_k)\cdot x_k & \text{有 fire}\\ x_S & \text{沒 fire}\end{cases}
$$

forward 數值上,有 fire 時 $s_k$ 精確等於 1,$(1-1)x_k=0$,跟硬重置算出來的數字完全一樣。

## 2. 有 fire 時,$v_{final}$ 對各個權重的梯度

### 2.1 $j=k$(fire 事件自己)

$$
\frac{\partial v_{final}}{\partial w_k}=\frac{\partial v_{final}}{\partial x_k}\cdot\frac{\partial x_k}{\partial w_k}
$$

$\dfrac{\partial x_k}{\partial w_k}=1$(直接項)。$\dfrac{\partial v_{final}}{\partial x_k}$ 用乘法法則展開:

$$
\frac{\partial v_{final}}{\partial x_k}=-\frac{\partial s_k}{\partial x_k}\cdot x_k+(1-s_k)=-\text{slope}_k\cdot x_k+(1-s_k)
$$

有 fire 時 $s_k=1$,第二項消失:

$$
\boxed{\frac{\partial v_{final}}{\partial w_k}=g_k:=-\text{slope}_k\cdot x_k}
$$

### 2.2 $j<k$(fire 之前的事件)

$$
\frac{\partial v_{final}}{\partial w_j}=g_k\cdot\frac{\partial x_k}{\partial w_j}
$$

因為「不套閘」,$k$ 之前每一步都是純仿射 $x_i=x_{i-1}a_i+w_i$,逐步套鏈式法則(每多一步只多乘一個那一步的衰減係數,新加的 $w$ 常數項對更早的 $w_j$ 求導是 0,不貢獻東西):

$$
\frac{\partial x_j}{\partial w_j}=1,\quad\frac{\partial x_{j+1}}{\partial w_j}=a_{j+1},\quad\frac{\partial x_{j+2}}{\partial w_j}=a_{j+1}a_{j+2},\ \dots
$$

一路推到 $k$:

$$
\frac{\partial x_k}{\partial w_j}=a_{j+1}a_{j+2}\cdots a_k
$$

合起來:

$$
\boxed{\frac{\partial v_{final}}{\partial w_j}=g_k\cdot(a_{j+1}a_{j+2}\cdots a_k)}\qquad(j<k)
$$

這串連乘就是 `associative_scan` 已經算出來的前綴係數比值($A_k/A_j$),不需要另外寫反向掃描——`jax.grad` 對現有的純仿射 `associative_scan` 直接自動微分就會得到這個答案。

### 2.3 $j>k$(fire 之後被丟棄的事件)

$$
\frac{\partial v_{final}}{\partial w_j}=0
$$

不是靠遮罩或設係數為 0 湊出來的——`v_final` 的計算式(`x_sequence[safe_idx]`)本來就只讀取 $x_k$,從來沒有引用過 $x_{k+1},\dots,x_S$,計算圖裡根本沒有這條邊,`jax.grad` 自然得到 0,不需要額外處理。

## 3. 沒 fire 時,$v_{final}=x_S$,梯度是純仿射,不需要 $g$

整個 chunk 沒人 fire,$v_{final}$ 就是 `associative_scan` 算出來的 $x_S$ 本身,沒有任何 $s$ 介入:

$$
\frac{\partial v_{final}}{\partial w_j}=a_{j+1}a_{j+2}\cdots a_S\qquad(\text{對所有 }j\le S)
$$

跟 2.2 節同樣道理,`jax.grad` 對純仿射的 `associative_scan` 直接自動微分即可,不需要任何 surrogate 介入。

## 4. 接上真正的訓練 loss

`process_chunk` 對外交付兩樣可能被下游用到的東西:$v_{final}$(傳給下一個 chunk,或最後一層的讀出值)、$s_k$(如果下游直接用「這次 fire 有多篤定」這個值,例如群體編碼式的 loss)。設:

$$
\Lambda:=\frac{\partial L}{\partial v_{final}},\qquad\mu:=\frac{\partial L}{\partial s_k}
$$

($\Lambda,\mu$ 由下游的 loss 決定,`process_chunk` 本身不需要預設是哪一種——膜電位回歸通常 $\Lambda\ne0,\mu=0$;群體編碼通常反過來)

多變數鏈式法則(同一個 $x_k$ 透過兩條不同路徑影響 $L$,兩條路徑的貢獻加總):

$$
\frac{\partial L}{\partial w_j}=\Lambda\cdot\frac{\partial v_{final}}{\partial w_j}+\mu\cdot\frac{\partial s_k}{\partial w_j}
$$

$\dfrac{\partial s_k}{\partial w_j}=\text{slope}_k\cdot(a_{j+1}\cdots a_k)$(跟 2.2 節同一串連乘,只是種子换成 $\text{slope}_k$ 不是 $g_k$)。合起來,$j<k$ 時:

$$
\frac{\partial L}{\partial w_j}=\big(\Lambda g_k+\mu\,\text{slope}_k\big)\cdot(a_{j+1}\cdots a_k)
$$

兩條路徑的種子先加總,再乘上同一串衰減連乘——不管 $\Lambda,\mu$ 實際是多少(甚至其中一個是 0),公式結構不用變。

## 5. 如果 loss 直接用到「非 fire 事件」自己的 $s_i$

第 4 節假設 loss 只透過 $v_{final}$、$s_k$(fire 那個事件自己)這兩個管道跟 $w_j$ 產生關係。**這個假設沒有涵蓋「loss 直接用到某個沒 fire 的事件自己的 $s_i$」這種情況**——這是本文件第一版的疏漏,不是原本就決定不管的範圍,對話記錄裡是先撞到 `test_chunk_scan_gradient.py` 對不上預期值,回頭查才發現這裡沒推過。

### 5.1 單一事件自己的 $s_i$,不透過任何 reset,不需要 $g$

$$
s_i=\text{atan\_spike}(x_i-v_{th},\alpha)
$$

因為「不套閘」,$x_i$ 本身是純仿射,$s_i$ 只是事後貼上去的標籤,不影響 $x_i$ 怎麼算,也不像 $v_{final}$ 那樣涉及 reset 動作。直接微分:

$$
\boxed{\frac{\partial s_i}{\partial w_j}=\text{slope}_i\cdot(a_{j+1}\cdots a_i)}\qquad(j\le i,\ j\ \text{與}\ i\ \text{在同一段、中間沒有真正 fire 過)}
$$

**這個公式對任何事件都成立,不管它自己有沒有 fire,而且完全不需要 $g$**——$g$ 只在算 $v_{final}$(涉及 reset)時才需要,單純的 $s_i$ 本身用不到。

### 5.2 如果 $j$ 在 $i$ 之前的某次 fire「之前」,中間隔了一次 reset

如果 $j\le k<i$($k$ 是中間某次真正 fire 的位置),$x_i$ 是從 $k$ 之後**重新起算**的(用 $v_{final}$ 當新的起點,不是用被丟棄的猜測值),所以要接上第 2 節推過的 $g_k$:

$$
\frac{\partial s_i}{\partial w_j}=\text{slope}_i\cdot(a_{k+1}\cdots a_i)\cdot\frac{\partial v_{final}}{\partial w_j}\bigg|_{\text{第 }k\text{ 段}}
$$

（$\dfrac{\partial v_{final}}{\partial w_j}$ 就是第 2.2 節的 $g_k\cdot(a_{j+1}\cdots a_k)$）

### 5.3 用具體數字驗算(對照 agent 撞到的例子)

$N=[0,1,4]$,$w=[0.6,0.6,0.9]$,同一個例子,$k=1$(事件1 fire)。$L=s_0+s_1+s_2$:

$$
\frac{\partial s_0}{\partial w_0}=\text{slope}_0\approx0.387727\qquad(\text{5.1 節,同段,直接項})
$$

$$
\frac{\partial s_1}{\partial w_0}=\text{slope}_1\cdot a_1\approx0.975915\times0.75\approx0.731936\qquad(\text{5.1 節,同段})
$$

$$
\frac{\partial s_2}{\partial w_0}=\text{slope}_2\cdot a_2\cdot\underbrace{(g_1\cdot a_1)}_{\partial v_{final}/\partial w_0}\approx0.910172\times0.31640625\times(-0.768533)\approx-0.221317\qquad(\text{5.2 節,跨過 }k=1\text{ 這次 fire})
$$

$$
\frac{\partial L}{\partial w_0}\approx0.387727+0.731936-0.221317\approx0.898346
$$

跟實際跑出來的預期值 $0.898341$ 對上(誤差是四捨五入)。

### 5.4 這揭露了一個 `run_layer_forward` 現有 API 的落差,不是數學缺口

上面的推導本身是完整、自洽的——問題是 `chunk_scan.py` 現有的 `run_layer_forward`,一個外層 scan 步驟只吐出**一個**代表性的 `s_value`(fire 位置的,或沒 fire 時窗口最後一個位置的),**不會把窗口裡其他事件各自的 $s_i$ 交出來**。討論過膜電位回歸、頻率編碼兩種訓練情境各自需要什麼之後,已經決定往下一節(5.5)的方向擴充——不是「整個 `s_sequence` 都吐出來」那種最大化通用性的做法(會把回傳形狀變成 3D,`不套閘`已經放棄的部分近門檻訊號等於默默加回來),是成本較低、剛好同時滿足這兩種訓練情境的中間方案。

### 5.5 決定的解法:「加總這個窗口裡所有有效位置」,取代「只挑一個代表位置」

先定義「這個窗口裡,前面幾個位置算有效」(`valid_len`),把「有 fire」「沒 fire」兩種情況統一成同一條規則:

$$
\text{valid\_len}=\begin{cases}k+1 & \text{有 fire(fire\_idx}=k\text{,0-based)},\text{只算到 fire 為止,之後被丟棄的猜測值不算}\\ \min(\text{chunk\_size},\ \max(0,\ s_{real}-\text{pointer})) & \text{沒 fire},\text{只算窗口裡真正是真實事件、不是 padding 補位的個數}\end{cases}
$$

這一步的代表值,從「挑一個位置」換成「加總前 `valid_len` 個位置」:

$$
\boxed{s_{value}=\sum_{i=0}^{\text{valid\_len}-1}s_{sequence}[i]}
$$

實作上用一個布林遮罩 $[0,1,\dots,\text{chunk\_size}-1]<\text{valid\_len}$ 跟 `s_sequence` 逐點相乘再加總即可,不需要新的機制。

**這條公式自動涵蓋三個已經驗證過的情況,不會破壞任何既有結果**:

- `chunk_size=1` 時,`valid_len` 永遠是 0 或 1,退化成原本 `safe_idx` 挑出來的那個值,不影響已經對上手算數字的結果。
- 三事件例子(`chunk_size=3`):第一步 fire 在 $k=1$,`valid_len=2`,加總 $s_0+s_1$(補回原本漏掉的 $s_0$);第二步窗口只剩事件2、其餘是 padding,`valid_len=\min(3,1)=1`,只加總事件2 自己的 $s_2$,不會被 padding 重複灌水。兩步合計 $s_0+s_1+s_2$,對上第 5.3 節驗算的 $0.898341$。
- 指標已跑出真實事件範圍、純空轉的步驟:$s_{real}-\text{pointer}$ 是負的,`max(0,\cdot)` 夾成 0,`valid_len=0`,加總範圍是空集合,自動貢獻 0——原本 `valid_mask` 想解決的「空轉步驟不該重複計入梯度」這件事,直接內建進加總範圍裡,呼叫端不需要再另外乘一次 `valid_mask`。

**已實作並驗證**:`chunk_scan.py` 的 `run_layer_forward` 已經照這個公式改完(`valid_len`/`valid_mask`/加總,`process_chunk`、`associative_scan`、指標推進規則都沒動),`tests/test_chunk_scan_gradient.py` 拿 `jax.grad` 實際跑出來的梯度,`chunk_size=1`、`chunk_size=3` 兩種切法都對上第 5.3 節手算的 $[0.898341, 0.680819, 0.910170]$,誤差在 `1e-4` 容忍度內。

## 6. 工程上的關鍵結論:不需要手寫 `custom_vjp`

整條計算圖(從 $w_j$ 到 $x_k$ 到 $s_k$/$v_{final}$)全部由 JAX 原生就懂得微分的運算組成:`associative_scan` 上完全沒有套用任何 surrogate(純仿射,係數固定,微分規則是標準的),唯一用到 `atan_spike` 的地方(`s_sequence`,已經是有 `custom_vjp` 的現成函式)是計算完 $x_1,\dots,x_S$ **之後**才發生的、單純的逐點運算,不影響前面的合成結構。

**這代表 `jax.grad` 直接對現有的 `process_chunk`(或用它組出來的 loss)求梯度,會自動算出本文件推導的全部結果,不需要為 chunk 化的 fire/reset 邏輯另外寫一個手刻的反向傳播規則**——這正是「不套閘」這個設計決策換來的最大工程好處,對照「套閘」版本需要手推、手刻整套反向 associative scan,複雜度差非常多。

## 7. 用具體數字驗算(對照 `docs/TODO.md` 手算過的例子)

$\tau=4$($a=0.75$ 對應 $N=1$),$v_{th}=1$,$\alpha=2$,$N=[0,1,4]$,$w=[0.6,0.6,0.9]$。

Forward:$x_0=0.6$($N=0$,不衰減),$x_1=0.6\times0.75+0.6=1.05$(**在這裡 fire,$k=1$,0-based**),$x_2$(丟棄,不使用)。

$$
\text{slope}_1=\frac{\alpha/2}{1+(\frac\pi2\alpha\times0.05)^2}\approx0.975915,\qquad g_1=-\text{slope}_1\times1.05\approx-1.024711
$$

$$
\frac{\partial v_{final}}{\partial w_1}=g_1\approx-1.024711
$$

$$
\frac{\partial v_{final}}{\partial w_0}=g_1\times a_1=-1.024711\times0.75\approx-0.768533
$$

$$
\frac{\partial v_{final}}{\partial w_2}=0\qquad(\text{事件 2 在 fire 之後,被丟棄})
$$

## 8. 本文件沒有涵蓋、留給後續驗證的部分

**以下三項原本記錄成待辦,現在已經完成、驗證通過,留在這裡當作驗證軌跡:**

- ~~這幾個數字還沒有用實際跑 `jax.grad` 驗證過~~——已用 `tests/test_chunk_scan_gradient.py`(`jax.grad` 對 `chunk_scan.run_layer_forward` 組出的 loss 求導)驗證,`chunk_size=1`、`chunk_size=3` 都對上第 7 節/第 5.3 節手算的數字,自動微分確實不需要手動介入就能算對,如第 6 節所述。
- ~~`test_surrogate.py` 現有的 `_sequential_lif_with_surrogate` 還是舊的「每步套閘」語意~~——已重寫成「不套閘」語意(只在真的 fire 時套 soft reset),`test_gradient_flows_through_fire_reset` 通過,可以繼續當作 `core.py` 的正確性參考。
- ~~`run_layer_forward` 要照第 5.5 節的 `valid_len` 加總方案修改,還沒實作、也還沒拿 `jax.grad` 驗證過~~——已實作並通過測試,`test_chunk_scan_stress.py`、`test_multi_layer_forward.py` 等既有測試也都還是通過的,沒有被這個改動影響。

**還沒做、真正待處理的部分:**

- **多層、多 chunk 串接時,$\Lambda$(上一個 chunk 的 $v_{final}$ 對下一個 chunk 而言,扮演它自己的 $v_0$)怎麼一路傳遞**,本文件只推了單一 chunk 內部的梯度,沒有處理 chunk 跟 chunk 之間、或跨層之間的梯度串接,留給訓練迴圈的推導處理。
