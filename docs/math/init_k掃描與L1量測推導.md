# init_k 掃描與 $L_1$ 量測:公式與收斂性推導

> **2026-09-08 現況 banner**:第 2、4、5、6、7 節推的是「用感受野正規化 firing
> rate 落進目標帶 $[0.20, 0.50]$」當 init_k 選值準則的**掃描演算法**。
> 這個**準則本身已廢棄** —— 它把 init_k 推進「單筆事件就 fire」的飽和區,
> 訓練較差(V6 實測 8/64 ≈ 0.735 vs `init_k=5` ≈ 0.80)。定案改成 committed
> `init_k = 5.0`(三層),不校準。完整推導 + V1–V6 驗證見
> `docs/math/初始權重尺度推導.md`。
> 本文件保留為 bracket + 幾何二分**演算法收斂性**的推導記錄;
> **第 1、3 節的 $o_j$ / $L_1$ 幾何量測與 firing-rate 準則無關,仍然有效**。

給新開的 session 讀的完整脈絡在 `docs/TODO.md` 任務 8 段 3.8、`docs/規格書.md`「段 3.8 定案」——那兩份文件講「決定量什麼、掃哪些層、目標帶多少」,是規格跟決策,不重複列在這裡。本文件只做一件事:把 `src/models/conv_net.py`(`_receptive_field_opportunity_count`/`conv_layer_receptive_field_firing_rate`/`conv1_receptive_field_L1_batch`)跟 `src/init_k_search.py`(`sweep_init_k`)裡實際在算的東西,寫成公式,並且證明 bracket+bisection 演算法為什麼會收斂、什麼情況下真的收斂不了——只有數學,不含程式碼實作細節。

## 1. Opportunity count:合法 tap 的定義

給定一批事件(座標三元組 $(x,y,c)$ + 時間)跟一個 conv 層的幾何(kernel $K$、stride $S$、padding $P$、輸出網格 $H_{out}\times W_{out}$),每個輸出神經元 $j$($j=1,\dots,N$,$N=OC\cdot H_{out}\cdot W_{out}$)對每筆事件 $e$ 有一個二元判斷:$e$ 是不是 $j$ 的合法輸入 tap(座標落在 $j$ 的感受野內、且不是 pad 事件)。這個判斷完全由座標幾何決定,見 `docs/conv事件佇列建構推導.md` 第 1、4、7、8.1 節,跟權重、通道 $c$ 的實際數值都無關(只有 $c$ 是否等於 $j$ 對應的輸入通道才影響合法性,但不受權重「數值」影響)。

定義 **opportunity count**:

$$
o_j \;=\; \big|\{\, e \in \text{batch} : e \text{ 是 } j \text{ 的合法、非 pad 輸入 tap} \,\}\big|
$$

這是一個純幾何量,對固定的一批事件跟固定的層幾何,$o_j$ 是常數,不隨權重、不隨 init_k 改變。

## 2. Firing rate:感受野正規化版本

給定一次真實 forward 的結果(某組候選權重下跑出來的 spike 記錄),定義每個神經元的 **spike count**:

$$
s_j \;=\; \sum_{t} \text{spike\_mask}_{j,t}
$$

樸素的 firing rate 定義($\frac1N\sum_j s_j / (\text{某個上界})$)會把「這個神經元這批事件裡根本沒有合法輸入」跟「有輸入但沒 fire」混在一起——conv 是局部連接,邊角神經元的感受野常常被 padding 佔掉大半,$o_j=0$ 是常態,不是「這個神經元學得不好」。所以定義排除掉沒有機會的神經元:

$$
S = \{\, j : o_j > 0 \,\}, \qquad
r_j = \frac{s_j}{\max(o_j,1)} \ \ (j\in S)
$$

$$
\boxed{\ \text{firing\_rate} \;=\; \frac{1}{|S|}\sum_{j\in S} r_j \;=\; \frac{1}{|S|}\sum_{j\in S}\frac{s_j}{o_j}\ }
$$

這是**逐神經元取比例、再對神經元取無加權平均**(等權重看待每個有機會的神經元,不管它 $o_j$ 大小),不是 $\big(\sum_{j\in S}s_j\big)/\big(\sum_{j\in S}o_j\big)$ 這種按 opportunity 大小加權的版本——兩者在 $o_j$ 分布不均勻時(conv1 邊角效應)答案會不同,目前程式碼選前者。

批次(多筆樣本)的版本是對每筆樣本各自算出一個純量 firing rate,再對樣本取平均——跟上面同一個公式,只是外面再套一層 $\frac1{N_{\text{samples}}}\sum_{\text{sample}}$。

## 3. $L_1$:$o_j$ 的全資料集最大值

$L_1$(壓縮佇列要留的緊緻長度)定義成:

$$
L_1 \;=\; \max_{\text{split}\in\{train,val,test\}}\ \max_{\text{sample}\in\text{split}}\ \max_{j=1,\dots,N}\ o_j(\text{sample})
$$

這是純幾何量(第 1 節定義的 $o_j$ 本身跟權重無關),跟 init_k 搜尋完全獨立,只跟事件座標分布、conv1 的 $K,S,P$ 幾何有關,量一次對這個資料集/這個幾何組合永久有效——這也是規格書「段 3.8 定案」B 節講的「training-independent」的數學原因。

## 4. init_k 到 firing rate:一個假設單調、沒有封閉解的黑盒函數

給定一層的 fan-in $F$,uniform 初始化用 $\text{limit}=k/\sqrt F$($k$=init_k)產生權重 $W\sim U(-\text{limit},\text{limit})$。定義:

$$
f(k) \;=\; \text{firing\_rate 在權重 } W(k) \text{ 下的量測值}
$$

$f$ 沒有解析形式——要知道 $f(k)$ 的值,唯一辦法是真的生一組權重、跑一次 forward、數 spike。**唯一假設的性質是統計上單調不遞減**:$k$ 越大,$|W|$ 的尺度越大,神經元的膜電位波動幅度越大,統計上越容易跨過 $v_{th}$。這不是從 LIF 方程嚴格證明出來的定理(單一次抽樣、單一組隨機權重不保證嚴格單調,只是「平均而言」),是一個經驗假設,程式碼的所有收斂性都建立在這個假設上。

兩端物理邊界(`lo=1e-3, hi=1e3`):

$$
\lim_{k\to 0} f(k) = 0 \quad(\text{幾乎沒有事件跨得過 } v_{th})
\qquad
\lim_{k\to \infty} f(k) \to 1 \quad(\text{幾乎每個事件都讓神經元 fire})
$$

只要目標帶 $[0.20,0.50]$ 跟這兩端不重疊(顯然成立),單調假設保證兩端之間存在跨帶點。

## 5. Bracket 階段:對數尺度的倍增搜尋

從 $k_0=1$(對應標準初始化尺度)開始探測 $f(k_0)$。若已經落帶,直接回傳。否則依 $f(k_0)$ 跟帶的相對位置決定方向:

$$
k_{n+1} = \begin{cases} k_n \cdot \gamma & f(k_0) < \text{band}_{lo}\ (\text{遞增方向}) \\ k_n / \gamma & f(k_0) > \text{band}_{hi}\ (\text{遞減方向}) \end{cases}
\qquad \gamma=\text{bracket\_factor}=2
$$

每步都先檢查是不是直接落帶,再檢查是不是跨過帶(前一步在帶外一側、這一步在帶外另一側或帶內)。一旦偵測到跨帶,取 $(k_{lo},k_{hi})$ 當 bisection 的初始區間,此時保證:

$$
f(k_{lo}) < \text{band}_{lo} \le \text{band}_{hi} < f(k_{hi}) \qquad (\text{或反過來,依方向而定})
\qquad
\frac{k_{hi}}{k_{lo}} = \gamma = 2
$$

**為什麼用倍增,不是線性步進**:$k$ 的合理範圍跨六個數量級($10^{-3}$ 到 $10^3$),線性步進要嘛步伐太小、掃到天荒地老,要嘛步伐太大、直接跳過目標帶。倍增在對數尺度上是均勻步進,$\log_2(10^6)\approx20$ 步就能覆蓋整個範圍——這正是 `bracket_max_iter=20` 這個數字的來源,不是隨便選的。

## 6. Bisection 階段:幾何二分,收斂速度可以精確算出來

線性二分($k_{mid}=(k_{lo}+k_{hi})/2$)在 $k$ 橫跨數量級時會被數值大的那一端拖著跑(例如 $k_{lo}=1,k_{hi}=1000$,線性中點是 500.5,離 $k_{lo}$ 很遠)。改用**幾何中點**:

$$
k_{mid} = \sqrt{k_{lo}\cdot k_{hi}}
$$

這個中點在**對數尺度上**才是真正的中點:$\log k_{mid} = \tfrac12(\log k_{lo}+\log k_{hi})$。

**收斂速度證明**:設第 $n$ 步的區間比值 $\rho_n = k_{hi}^{(n)}/k_{lo}^{(n)}$。不管 $k_{mid}$ 取代的是 $k_{lo}$ 還是 $k_{hi}$,新比值都是:

$$
\rho_{n+1} = \frac{k_{hi}}{k_{mid}} = \frac{k_{hi}}{\sqrt{k_{lo}k_{hi}}} = \sqrt{\frac{k_{hi}}{k_{lo}}} = \sqrt{\rho_n}
\qquad\text{(取代 } k_{hi} \text{ 時,對稱可得同樣結果)}
$$

所以:

$$
\rho_n = \rho_0^{1/2^n}
$$

收斂條件 $\rho_n < 1+\varepsilon$($\varepsilon=$`bisect_rel_tol`$=0.01$)所需的步數:

$$
\rho_0^{1/2^n} < 1+\varepsilon
\iff \frac{\ln\rho_0}{2^n} < \ln(1+\varepsilon)
\iff n > \log_2\!\left(\frac{\ln\rho_0}{\ln(1+\varepsilon)}\right)
$$

代入 $\rho_0=\gamma=2$,$\varepsilon=0.01$:

$$
n > \log_2\!\left(\frac{\ln 2}{\ln 1.01}\right) = \log_2\!\left(\frac{0.6931}{0.00995}\right) \approx \log_2(69.7) \approx 6.1 \quad\Rightarrow\quad n=7 \text{ 步足夠}
$$

**這個結果不依賴 firing rate 量測正不正確**——不管每一步比較 $f(k_{mid})$ 跟帶的結果選哪一側收縮,區間寬度都保證每步開根號縮小,收斂速度是演算法幾何結構本身保證的,不是統計性質。這代表 `bisect_max_iter=30` 是遠超過需要的安全邊界(理論只需要 7 步),bisection 階段「30 步內兩個收斂條件都沒達成」這個失敗分支在正常情況下不可能觸發。

## 7. 什麼情況下真的會找不到解

結合第 5、6 節:bisection 階段的區間收斂是幾何結構保證的、不會失敗;bracket 階段 20 步倍增已經覆蓋 $[10^{-3},10^3]$ 整個範圍,所以「20 步內沒跨過帶」實質上等同「撞到 $[lo,hi]$ 邊界前就已經用完步數」,兩者是同一個失敗模式。

所以唯一有意義的失敗條件是:**在 $k\in[10^{-3},10^3]$ 這整個範圍內,$f(k)$ 從頭到尾沒有進過 $[0.20,0.50]$**。根據第 4 節的物理邊界論證,這只可能發生在:

1. 這一層的幾何/動態參數($\tau$、$v_{th}$、資料本身的事件密度)讓 firing rate 對權重尺度不敏感到這六個數量級都掰不回來(架構層級的問題,不是量測噪聲)。
2. 量測邏輯本身有 bug,$f(k)$ 對 $k$ 沒有真正反應。

**單一樣本批次的統計噪聲不足以造成這個失敗**——噪聲會讓某個特定 $k$ 量到的 $f(k)$ 上下抖動,但不會讓整條曲線在六個數量級的範圍內完全不越過帶。統計噪聲造成的失敗,表現形式是「sweep 階段找到一個 $k$,但拿到大樣本(confirm)驗證時掉出帶外」,是另一個機制(`conv_param_search.py` 的 retry 迴圈想處理的問題),不是 `InitKSweepError`。
