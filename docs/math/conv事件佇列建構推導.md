# Conv 事件佇列建構推導:從輸入事件到密集仿射映射陣列

> **現況(2026-09-08)**:本文件推導的「密集」版本——每顆輸出神經元一條
> 長度 = 全域事件數的仿射映射陣列——對應的 `build_conv_queue` 曾實作過,
> 後於架構翻修 step 4d **移除**,程式碼裡不再有密集版佇列建構。實際在用的是
> **壓縮版** `salt_core/connectivity/conv.py` 的 `build_conv_queue_compressed`
> (每顆神經元只留自己感受野內的 tap,長度固定 `L`),推導見
> `docs/math/conv事件佇列壓縮版推導.md`。
>
> 這份密集版推導**保留**:§0–§4 的連接結構($o,k$ 公式、扇出數 $N$、tap
> 索引)、§6 攤平、§8 pad 座標、§9 neuron id ↔ (x,y,c) 還原這些**幾何**
> 是壓縮版原封不動沿用的基礎(`_axis_candidates` / `unravel_conv_source` /
> `receptive_field_tap_count` 都對應這裡的節次)。只有「把 tap 攤成密集陣列」
> 這個最後的組裝步驟被壓縮版取代。閱讀時把 `build_conv_queue` 理解成
> 「密集組裝法的數學描述」,不是現存函式。

延續 `docs/math/全連接forward訓練範例.md`(FC 連接結構、forward、訓練)與
`docs/math/單狀態仿射平行掃描推導.md`(仿射合成、平行掃描、fire 判斷的數學)。
這兩份文件推導的仿射映射合成、`chunk_scan` 消化佇列的邏輯,對 conv **原封不動
適用**,不在本文件重推。本文件只處理 FC 因為全連接被完全省略掉的一步:**怎麼
幫每顆輸出神經元組出它自己要用的 `(a,b)` 仿射映射陣列**,對應
`salt_core/connectivity/fc.py` 的 `build_fc_queue`。

只寫數學公式、資料形狀(shape)與設計決策,不含實際程式碼,跟 FC 文件的體例
一致。

## 0. 跟 FC 的差異起點

FC 的 `build_fc_queue`(`weights = W[:, event_source_idx]`)能一行矩陣運算
解決,是因為全連接下**每顆輸出神經元都收全部事件**,「選權重」跟「組佇列」是
同一個動作。`event_source_idx` 本質上是「(時間, 一維 index)」這對數字,這個
一維 index 直接拿去當 `W` 的 column 用。

conv 的權重要看事件座標相對輸出位置的偏移(kernel tap),不是只看「來源是
誰」就能決定,也不是每顆輸出神經元都跟每筆事件有關——一維 `event_source_idx`
不夠用,要換成 `(x_i,y_i,c_i)` 三元組。

## 1. 連接結構:$o,k$ 公式(抄自 CSNN-FPGA 硬體規格)

完整推導見 `D:\Project\CSNN-FPGA\docs\SNN\Concept\conv_event_scatter_banking_derivation.md`
第 4、5、6、10.1 節,這裡只列訓練端要用的結論。

符號:$K$ kernel size、$S$ stride、$P$ padding(方形 kernel,$H,W$ 兩軸共用
同一組)、$i$ 事件座標(單軸)、$o$ 輸出座標(單軸)、$k$ kernel tap 索引。

**單軸最大扇出數**(編譯期常數,只跟 $K,S$ 有關):

$$
N = \left\lfloor\frac{K-1}{S}\right\rfloor + 1
$$

**給定事件座標 $i$,合法輸出 $o$ 的範圍**(下界取 ceil):

$$
o_{min} = \left\lceil \frac{i+P-K+1}{S} \right\rceil,\qquad
\text{上界} = \left\lfloor\frac{i+P}{S}\right\rfloor
$$

**這兩條式子全程用整數運算,不經過浮點數,已驗證正確、不需要分正負號兩條
公式**:$i,P,K,S$ 在這個問題裡本來就都是整數,$S>0$。上界 $\lfloor(i+P)/S\rfloor$
直接是整數 floor 除法 $(i+P)\,\texttt{//}\,S$——JAX/NumPy 的整數 `//` 本身
就是真 floor(無條件捨到負無窮),不是 C/C++ 那種對負數往 0 捨去的截斷除法,
不需要另外判斷正負號(這裡 $i+P$ 本來就恆 $\ge0$,不會踩到這個問題,但公式
本身對負數也成立)。

下界的分子 $i+P-K+1$ 靠近邊界(尤其有 padding)時可能是負數,標準整數 ceil
公式:

$$
o_{min} = \left\lfloor \frac{(i+P-K+1)+S-1}{S} \right\rfloor
$$

$\lceil n/d\rceil=\lfloor(n+d-1)/d\rfloor$($d>0$)這條恆等式,只要 `//` 是
真 floor,對任意整數 $n$(不管正負)都成立——把 $n$ 寫成 $n=qS+r$
($0\le r<S$,floor 除法的標準表示法),分 $r=0$ 跟 $r>0$ 兩種情況代入驗證,
結論都對得上 $\lceil n/S\rceil$,不需要對負數分子另外寫一條公式、也不需要
`-((-n)//d)` 這種變形。

枚舉 $N$ 個候選($r=0,\dots,N-1$):

$$
o^{(r)} = o_{min} + r
$$

**合法性**(超過上界,或超出輸出網格範圍,都不合法):

$$
\text{valid}(o^{(r)}) = \Big[\,o^{(r)} \le \text{上界}\,\Big] \wedge \Big[\,0\le o^{(r)}\le O_{max}-1\,\Big]
$$

$O_{max}$ 是這一軸的輸出大小($y$ 軸代 $H_{out}$、$x$ 軸代 $W_{out}$)。

**候選數不是固定 $N$,是 $\le N$**:硬體文件第 6.1 節註記過,即使是內部事件
(不靠近邊界),候選數也會在 $\{1,\dots,N\}$ 之間浮動,不是每次都剛好
$N$——例如 conv1/conv2 的 $K=3,S=2$($N=2$),單軸候選數在 $\{1,2\}$ 交替,
2D 合起來總候選數 $\in\{1,2,4\}$。$N$ 只是「最多幾個」的上限,拿來固定陣列
形狀用,不是每筆事件的實際候選數。

**對應的 tap**:

$$
k^{(r)} = i - o^{(r)}S + P
$$

## 2. 事件格式:從一維 index 換成三元組

FC 的事件是 (時間, 一維 index),一維 index 直接當 `W` 的 column。conv 的
事件要換成 (時間, $x_i,y_i,c_i$)——$c_i$ 接手原本一維 index 的角色(選輸入
channel),$(x_i,y_i)$ 是 conv 特有、FC 完全不需要的空間座標。

conv 輸入層(感測器事件、或前一層 conv 的輸出)實際上怎麼產生這個格式的資料,
是資料前處理階段的事,不在本文件範圍——本文件只處理「已經有這個格式的事件
清單」之後,怎麼組出仿射映射陣列。

## 3. 決策:密集版,不做動態長度壓縮

### 3.1 兩種方案

- **壓縮版**:每顆輸出神經元只留真正屬於自己的事件,佇列變短,但需要一個
  跨事件的篩選 + 排序 + 找每列起點的機制,佇列長度上限(`max_queue_len`)是
  一個要另外決定的靜態值。
- **密集版**:陣列形狀跟 FC 一樣,`(n_out_neurons, n_total_events)`,寬度
  是全域事件數,不做壓縮。不相關的事件那一格,不是完全的 identity,是「只
  衰減、不加權重」($a=$真實衰減、$b=0$)——因為真實時間確實流逝了,只是
  沒有東西加進去。

### 3.2 數值驗證:兩種方案答案完全一樣

$\tau=4$($a=0.75$),神經元 A 只跟 $t=1$($w=0.6$)、$t=5$($w=0.5$)兩筆
事件有關,中間 $t=2$ 是另一顆神經元的事件、跟 A 無關。

正確答案(A 自己的物理定義,$N=5-1=4$):

$$
x_1=0.6\qquad x_2=0.75^4\times0.6+0.5=0.68984375
$$

壓縮版(A 的佇列只留 $[t{=}1,t{=}5]$,自己重新算 `diff`$=[1,4]$):算出來
就是上面這組數字。

密集版(佇列保留全部 3 筆,$t=2$ 那格 $a=0.75^1,b=0$):

$$
x_1=0.6\qquad x_2(\text{路過 }t{=}2)=0.75\times0.6=0.45\qquad
x_3=0.75^3\times0.45+0.5=0.68984375
$$

兩者一致——因為仿射合成本身有結合律(把一段時間拆成很多小段各自衰減,跟
一次衰減完整段,結果相同),這正是 `combine` 這個運算被設計出來要保證的
性質。

### 3.3 複雜度分析:密集版浪費多少

一顆神經元真正相關的事件數,跟 $K^2\times IC$ 同量級,遠小於全域事件數
$n_{total}$。密集版逼每顆神經元都掃過 $n_{total}$ 筆,浪費倍率約
$\dfrac{H_{out}W_{out}}{N^2}$——以 conv1($H_{out}=W_{out}=64,N=2$)為例,
約 1024 倍。

壓縮版要真的省到運算量,`max_queue_len` 必須明顯小於 $n_{total}$;但要
保證不漏接任何真實事件(靜默截斷會是難以察覺的正確性錯誤),需要一個有
根據的資料密度上限(例如上游事件編碼有沒有保證單一輸入像素的最大 fire
頻率)。這個問題連 CSNN-FPGA 硬體規格自己都還沒解決(該文件第 15 節第 3
項 `FIFO_DEPTH` 一樣懸而未決,要等實測),不是本文件能憑空推出正確數字的
東西。若不打算做這個論證,壓縮版唯一「保證安全」的上限就是
`n_total_events`——這樣陣列形狀退化成跟密集版一樣大,完全沒省到運算量,
還倒賠一段展開/排序的前處理成本。

### 3.4 決定

**先走密集版**,理由跟 FC 當初 `max_steps` 的決定邏輯一致:先求正確、不需要
決定一個沒把握的上限,效能問題留到有實測資料密度數據支撐時,再回頭評估
壓縮版。

## 4. 建構密集陣列:向量化路徑

以下用 $j$ 表示事件的 index($j=0,\dots,n_{events}-1$),$r_y,r_x$
表示 $y,x$ 軸候選的計數器($r=0,\dots,N-1$,是「離 $o_{min}$ 差幾格」,不是
座標本身)。

### 4.1 單軸候選(outer sum)

$$
o^{(r)}_y[j,r_y] = o_{min,y}[j] + r_y
$$

形狀從 $(n_{events},)$ 擴成 $(n_{events},N)$,是 elementwise 的 outer
sum(一個 $(n_{events},)$ 陣列 + 一個固定的 $(N,)$ 常數陣列,靠 broadcasting
撐出二維),跟 `chunk_scan.py` 的 `take_chunk` 用 `pointer[:,None] +
arange(chunk_size)[None,:]` 是同一招。$x$ 軸同理。

`valid` 也是同一形狀 $(n_{events},N)$ 的 bool,對候選跟上界/網格範圍逐元素
比較算出來。$k^{(r)}$ 一樣是 elementwise 算式,同形狀。

### 4.2 兩軸合併成 2D

$y$ 軸的候選、mask、$k_y$,形狀 $(n_{events},N)$ 擴成 $(n_{events},N,1)$;
$x$ 軸擴成 $(n_{events},1,N)$;相乘/相與自動 broadcast 成
$(n_{events},N,N)$。

$$
\text{valid\_2d}[j,r_y,r_x] = \text{valid}_y[j,r_y] \wedge \text{valid}_x[j,r_x]
$$

### 4.3 疊上 $oc$ 軸

前面都跟 $oc$ 無關,這裡才把 $oc$(固定範圍 $\text{arange}(OC)$)用同樣的
broadcast 手法疊上去,$(n_{events},N,N)$ 擴成 $(n_{events},N,N,OC)$。

$oc$ 不影響空間合法性(合法性只跟位置有沒有落在範圍內有關),所以:

$$
\text{valid\_full}[j,r_y,r_x,oc] = \text{valid\_2d}[j,r_y,r_x]
$$

(對 $oc$ 軸整個 broadcast,四個 $oc$ 共用同一個合法性判斷。)

## 5. 權重 gather

事件自己的 channel $ic[j]$,形狀 $(n_{events},)$,對 $r_y,r_x,oc$ 三個軸
整個 broadcast。$k_y[j,r_y]$、$k_x[j,r_x]$ 同前面推出來的候選,各自
broadcast 到 $(n_{events},N,N,OC)$。

固定位置 $[j,r_y,r_x,oc]$ 的權重:

$$
\text{weight}[j,r_y,r_x,oc] = W\big[\,oc,\ ic[j],\ k_y[j,r_y],\ k_x[j,r_x]\,\big]
$$

跟前面純算術的 broadcast 不一樣,這一步是**用四組同形狀的 index 陣列
(`oc` 本身、`ic[j]`、`k_y[j,r_y]`、`k_x[j,r_x]`)去對 $W$ 做 gather**,不是
逐元素算出來的數值。

## 6. Scatter 目標:先不攤平,最後才 reshape

目標陣列想成多維 $(OC,H_{out},W_{out},n_{events})$,不手動攤平成
`(n_out_neurons, n_events)`——scatter 用的 index 直接是原始座標:

- `oc_idx[j,r_y,r_x,oc] = oc`
- `oy_idx[j,r_y,r_x,oc] = o_y[j,r_y]`
- `ox_idx[j,r_y,r_x,oc] = o_x[j,r_x]`
- `event_idx[j,r_y,r_x,oc] = j`

不需要手動寫「第幾列」的攤平算式。`chunk_scan.py` 開頭
`n_out_neurons, n_total_events = maps.a.shape` 寫死吃恰好 2 維,所以最後
交給它之前,`(OC,H_out,W_out,n_events)` 用一次 `reshape` 變成
`(OC*H_out*W_out, n_events)`——`reshape` 預設攤平順序(先 $OC$、再
$H_{out}$、再 $W_{out}$)本身就等於 row-major 攤平公式
$oc\times H_{out}W_{out}+o_y\times W_{out}+o_x$,不用另外驗證兩者對不對得
上。

## 7. Invalid 處理

兩種成因:

- **(a) 真的超出網格**:$o^{(r)}$ 算出來 $<0$ 或 $\ge O_{max}$,陣列 index
  意義上真正越界。
- **(b) 沒超出網格,但這次用不到**:候選數浮動那件事(第 1 節)——$o^{(r)}$
  本身是合法座標,只是這筆事件沒過式子的上界判斷。這種比較危險:index 看起來
  完全正常,沒被擋下來的話,會把錯的權重安靜寫進一顆真實存在的神經元佇列,
  不會有任何錯誤訊息。

兩種都要靠 `valid_full` 擋,不能只靠陣列邊界檢查(對 (b) 完全沒用)。

**統一處理**:把兩種 invalid 都改寫成「保證真的越界」的 index,例如:

$$
oy\_idx = \text{valid\_full}\ ?\ o_y[j,r_y] : H_{out}
$$

($H_{out}$ 保證不在 $[0,H_{out}-1]$ 內,故意選一個一定越界的數字。)再用
`mode='drop'` 寫入,越界的更新會被直接丟棄。**JAX 的 scatter 預設(不指定
`mode`)對越界 index 的行為沒有保證,一定要顯式寫 `mode='drop'`,不能省略。**

**不會發生同格衝突,不需要 `.add()`**:同一事件 $j$ 的不同 $(r_y,r_x)$
一定對應不同整數座標,不同 $oc$ 對應到 4 維陣列裡完全不同的一層,所以同一
事件不會有兩個候選撞進同一格;不同事件因為 `event_idx` 本來就不同,也不會
撞。全程用 `.set()` 就夠。

## 8. 衰減 $n_{ms}$:沿用 FC 的全域 `diff`,不用逐列重算

密集版底下,佇列寬度維持全域事件數,所以跟 FC 一樣:對全域 `event_times`
算一次 `diff(prepend=0)`,`broadcast` 給每一列共用——因為經過的時間是客觀
事實,不管這顆神經元在不在乎這筆事件,時間照樣流逝,每一列的 $a$ 在同一個
事件欄位下數值相同(見第 3.2 節數值驗證)。這跟壓縮版「每列要自己重新算
`diff`」是不同方案下的不同做法,密集版不需要那一步。

**把第 5、6、8 節合起來,「不相關事件」那格是怎麼湊出來的**:$a$ 完全由這一
節的全域 broadcast 決定,不經過第 6 節的 scatter,每顆神經元、每個事件欄位
一律相同;`weights`(=$b$)陣列預設整個初始化成 0,第 6 節的 scatter 只在
合法 tap 位置才寫入第 5 節算出來的真實權重。兩件事分開發生、互不干涉,所以
「不相關事件」那格自然就是「全域算出來的真實 $a$」配「陣列預設值 $0$ 的
$b$」,不需要另外寫一個分支去產生這個組合——跟第 3.2 節「只衰減、不加權重」
的描述是同一件事,只是這裡把它拆開講清楚是哪一步在管 $a$、哪一步在管 $b$。

### 8.1 `n_real_events`:跨層串接時,尾端 pad 事件要強制蓋成 identity

`layer_chain.extract_output_events` 用 `jnp.nonzero(..., fill_value=0)` 補
滿固定長度,pad 位置的 `event_source_idx` 填的是 **0**——經過
`unravel_conv_source` 還原之後,會變成看起來完全合法的座標
$(oc{=}0,o_y{=}0,o_x{=}0)$,配上 `_PAD_TIME`(`1e12`)當時間。如果不擋下來,
這筆假事件會被當成真事件去 scatter,寫入 $(0,0)$ 附近真正存在的神經元,而且
因為排序後永遠在最後面(`_PAD_TIME` 保證最大),污染會直接留在 `v_final`
裡,影響下游讀出。

`build_conv_queue` 要跟 `build_fc_queue` 一樣吃一個 `n_real_events` 參數
(預設 `None` = 全部當真實事件,行為不變)。整個密集陣列(reshape 前後都
一樣)建完之後,`event_times` 的 index 超過 `n_real_events` 的那些**整欄**
(對應所有神經元),不管算出什麼權重、什麼衰減,一律強制覆寫成
$a=1,b=0$——這是尾端 padding 才能用的招(第 3.2 節區分過的「這裡沒事發生」
那種,不是「路過、只衰減」那種),因為 pad 事件排序後保證在最後面,後面
沒有真實事件需要靠正確的累積時間差,直接蓋成完全 identity 不會出錯。

### 8.2 `event_gain`:跨層可微分增益

`build_conv_queue` 一樣要接受 `event_gain` 參數(預設 `None` = 全 1),機制
跟 `build_fc_queue` 完全相同:`weights` 陣列算完之後(第 5、6 節 scatter
完,reshape 前後都可以)整批乘上 `event_gain`(對事件軸廣播)——沒有寫進的
位置本來就是 0,乘完還是 0,不需要特別處理;真正有權重的位置乘上
`event_gain`(上一層的 `s_spike`),讓梯度能穿過跨層邊界傳回上一層權重,
理由跟 `connectivity/fc.py` 的說明完全一致。

## 9. 多層串接:`layer_chain.py` 不動,座標還原內縮進連接層自己處理

### 9.1 決策

`layer_chain.py`(`extract_output_events`)**維持完全不變**,不管上一層是
FC 還是 conv,它只認扁平的 neuron id,吐出來的 `event_source_idx` 永遠是
一個扁平數字——這支檔案不需要知道、也不需要學會 $(x,y,c)$ 這種結構。

`build_conv_queue` 的對外介面**永遠只吃 $(x_i,y_i,c_i,t)$**,不接受扁平
index、不在內部判斷「這次拿到的是原始座標還是扁平 id」——跟 `build_fc_queue`
永遠只吃一維 index、介面單一,是同一個原則。

兩者中間的落差(conv 接 conv 時,上一層吐出來的是扁平 id,下一層要的是
$(x,y,c)$),用一個**獨立的小函式**接起來,不寫進上面任何一支檔案裡。

### 9.2 還原函式:攤平公式的反運算

暫定 `unravel_conv_source(event_source_idx, H_in, W_in)`,吃扁平
`event_source_idx` + 上一層自己的輸出形狀($H_{in},W_{in}$——上一層的
$OC_{in}$ 會自動變成這一層的 $IC$,不需要額外傳),還原出 $(x_i,y_i,c_i)$:

$$
c_i = \left\lfloor\frac{\text{idx}}{H_{in}W_{in}}\right\rfloor\qquad
y_i = \left\lfloor\frac{\text{idx}\bmod H_{in}W_{in}}{W_{in}}\right\rfloor\qquad
x_i = \text{idx}\bmod W_{in}
$$

這是第 6 節攤平公式的反運算,沒有新數學——實作階段建議直接寫一個
round-trip 測試(還原完再攤平回去,要等於原本的 id),驗證這條反推公式跟
第 6 節的攤平公式沒有寫岔。

### 9.3 誰負責呼叫這個還原函式

呼叫端(現在是手動串接 conv1→conv2 的膠水程式碼,之後會是通用多層組裝器
的職責,見第 10 節「通用多層組裝器」)在拿到上一層 `extract_output_events`
的結果之後、丟進下一層 `build_conv_queue` 之前,呼叫一次
`unravel_conv_source`。

**第一層例外**:第一層的輸入直接來自資料端(不經過 `layer_chain.py`),
資料端原生就會給 $(x,y,c)$,不需要呼叫這個還原函式。

### 9.4 為什麼不比照 PyTorch「全程不攤平」

PyTorch 的 `nn.Conv2d` 全程保留 $(C,H,W)$ 結構,conv 接 conv 之間完全不
攤平,只有接 `nn.Linear` 前才 flatten 一次——它能這樣做,是因為張量本身是
密集網格,沒有「扁平事件列表」這個中介格式。

我們這裡的扁平化,是因為要對時間軸稀疏編碼(`extract_output_events` 吐
的是「哪個神經元、什麼時間點 fire」的稀疏列表,不是密集張量)——這個列表
格式是既有、已驗證過的核心機制(`chunk_scan.py`/`layer_chain.py` 通過里程碑
1~4、可重現性、多 seed 驗證)。

真的要做到「全程不攤平」,`chunk_scan.py`/`layer_chain.py` 都要改(內部
`jnp.nonzero` 要吐出多維 index、`ExtractedEvents` 的欄位要從一個扁平數字
變成三個座標)——這代表要動已經驗證過的共用核心,不是新增一支隔離的
`connectivity/conv.py` 就能做完的事。

**決策:接受攤平/還原多繞一圈的成本,換取共用核心(`core.py`/
`chunk_scan.py`/`layer_chain.py`)完全不動**——新東西全部隔離在
`connectivity/conv.py`(加上這個獨立的 `unravel_conv_source`)裡面。

## 10. 本文件沒有涵蓋、留給後續的部分

- **壓縮版效能優化**:第 3.3 節已列出需要的論證(資料密度上限),等有實測事件密度數據再重新評估要不要做。
- **`W` 的 shape 慣例**($OC,IC,K,K$,搭配 row-major 攤平)沿用硬體規格文件的記法,訓練端實際 `W` 陣列的宣告方式留給實作階段。

其餘還開放的項目(通用多層組裝器、`n_real_events` 遮罩邏輯重複)跟已經解決的問題(`event_gain` 跨層梯度驗證、gather-then-drop 是否洩漏梯度)記錄在 `docs/問題紀錄.md`,過程/bug 記錄不重複寫在這份數學文件裡。
