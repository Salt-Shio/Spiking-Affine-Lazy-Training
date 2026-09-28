"""驗證 stream.extract_output_events_fc 把 layer1 的輸出接成 layer2 的輸入,
兩層串起來的數字算對。

stream.extract_output_events_fc 是純 JAX 實作,回傳固定長度(n_source_neurons*
max_steps 這個安全上限)的陣列,不是「剛好幾筆真實事件」的動態長度——前
n_real_events 筆是真實事件(已排序),後面補 pad 事件(不影響任何下游計算,
見 stream.py 說明)。下面每個測試都用 n_real_events 切出真正有意義的
前綴來檢查內容,同時也確認完整的固定長度陣列接回 FC 佇列建構(帶
n_real_events)、run_layer_forward(帶 n_real_events)之後,行為跟「陣列剛好
只有真實事件」完全一樣。

layer1 沿用 test_fc_forward.py 的例子(n=2,m=2,tau=4,v_th=1.0):$b_1$ 在 t=4
fire(唯一一次),$b_2$ 全程不 fire。所以 layer1 的輸出事件包只有一筆:
(t=4, 來源=b1)。

layer2(p=1 個輸出神經元 c1,輸入是 layer1 的 b1,b2 兩顆):W2=[[1.5, 0.4]]。
只有 b1 的事件會真的進到 layer2 的佇列(b2 沒 fire,W2 對 b2 那個權重 0.4 在
這個例子裡不會被用到,純粹補滿矩陣形狀)。

手算 layer2:唯一一筆事件 (t=4, 來源=b1),N = 4 - 0 = 4(t=0 起算),
a = 0.75^4 = 0.31640625,h = 0*a + 1.5 = 1.5 >= v_th=1.0,c1 在 t=4 fire,
reset 後 V=0。這裡的「1.5」是 W2[c1,b1]=1.5 乘上 b1 的 s_spike(forward 精確
等於 1),數值上跟不乘 s_spike 完全一樣,只是計算圖裡多一條路徑(見
stream.py 開頭的說明)。
"""

import jax.numpy as jnp

from salt_core.float.scan import run_layer_forward
from salt_core.connectivity.fc import build_fc_structure, fc_float_values
from salt_core.stream import extract_output_events_fc

TOL = 1e-4


def assert_allclose(actual, expected, msg):
    actual = float(actual)
    expected = float(expected)
    assert abs(actual - expected) < TOL, f"{msg}: got {actual}, expected {expected}"


def test_two_layer_fc_forward():
    tau = 4.0
    v_th = 1.0

    # --- layer1: 跟 test_fc_forward.py 完全一樣的設定 ---
    layer1_event_times = jnp.array([1.0, 2.0, 4.0])
    layer1_event_source_idx = jnp.array([0, 1, 0])
    W1 = jnp.array([[0.6, 0.5],
                    [0.3, 0.2]])

    layer1_maps = fc_float_values(build_fc_structure(layer1_event_times, layer1_event_source_idx, layer1_event_times.shape[0]),
                                  W1, tau, None)
    n_real_events_1 = layer1_event_times.shape[0]
    spike_mask_1, spike_event_idx_1, s_spike_1, _, v_final_1 = run_layer_forward(
        layer1_maps, v_th, chunk_size=1, max_steps=n_real_events_1, n_real_events=n_real_events_1)

    layer2_event_times, layer2_event_source_idx, layer2_event_gain, n_real_events_2 = \
        extract_output_events_fc(spike_mask_1, spike_event_idx_1, s_spike_1, layer1_event_times)

    # layer1 只該有一筆輸出事件:(t=4, 來源=b1=index0),固定長度陣列裡
    # 只有前 n_real_events_2 筆算數
    assert int(n_real_events_2) == 1, n_real_events_2
    assert_allclose(layer2_event_times[0], 4.0, "layer1 輸出事件時間")
    assert int(layer2_event_source_idx[0]) == 0, "layer1 輸出事件來源應該是 b1"
    assert_allclose(layer2_event_gain[0], 1.0, "b1 fire 的 s_spike forward 應精確等於 1")

    # --- layer2: p=1 個輸出神經元 c1,輸入是 layer1 的 b1,b2 ---
    W2 = jnp.array([[1.5, 0.4]])

    layer2_maps = fc_float_values(build_fc_structure(layer2_event_times, layer2_event_source_idx, n_real_events_2),
                                  W2, tau, layer2_event_gain)
    spike_mask_2, spike_event_idx_2, _, _, v_final_2 = run_layer_forward(
        layer2_maps, v_th, chunk_size=1, max_steps=layer2_maps.a.shape[1],
        n_real_events=n_real_events_2)

    assert bool(spike_mask_2[0].any()), "c1 應該要 fire"
    spike_step = [i for i in range(spike_mask_2.shape[1]) if bool(spike_mask_2[0, i])][0]
    spiked_event_idx = int(spike_event_idx_2[0, spike_step])
    assert_allclose(layer2_event_times[spiked_event_idx], 4.0, "c1 fire 的時間應該是 t=4")
    assert_allclose(v_final_2[0], 0.0, "c1 fire 後應硬重置成 0")


def test_two_layer_fc_forward_multi_fire_interleaved():
    """比 test_two_layer_fc_forward 更強的例子:layer1 兩顆神經元都會 fire
    (其中一顆 fire 兩次),合併後的時間順序也不是「照神經元編號分組」——
    b2 先 fire、b1 才 fire、b2 又 fire 一次——用來驗證 extract_output_events_fc
    是真的照時間排序合併,不是碰巧照 nonzero 掃描到的順序排對而已。
    tau=4(a=0.75^N,N 從 t=0 起算),v_th=1.0。

    layer1 手算軌跡:
      b1(w11=0.5,w21=0.3): e1(t=1,w=0.5)->0.5, e2(t=2,w=0.3)->0.675,
        e3(t=3,w=0.5)->1.00625 FIRE(reset), e4(t=5,w=0.3)->0.3
      b2(w12=0.5,w22=0.9): e1(t=1,w=0.5)->0.5, e2(t=2,w=0.9)->1.275 FIRE(reset),
        e3(t=3,w=0.5)->0.5, e4(t=5,w=0.9)->1.18125 FIRE(reset)
      => b1 只在 t=3 fire 一次,b2 在 t=2、t=5 各 fire 一次
      => 合併排序後餵給 layer2 的事件是 (t=2,來源b2)(t=3,來源b1)(t=5,來源b2)

    layer2 手算軌跡(N=[2,1,2],a=[0.5625,0.75,0.5625]):
      c1(w_c1b1=0.4,w_c1b2=0.6): 0.6 -> 0.85 -> 1.078125 FIRE(在 t=5,reset)
      c2(w_c2b1=0.1,w_c2b2=0.2): 0.2 -> 0.25 -> 0.340625(不 fire)
    """
    tau = 4.0
    v_th = 1.0

    # --- layer1: n=2 (a1,a2), m=2 (b1,b2) ---
    layer1_event_times = jnp.array([1.0, 2.0, 3.0, 5.0])
    layer1_event_source_idx = jnp.array([0, 1, 0, 1])
    W1 = jnp.array([[0.5, 0.3],   # b1: w11=0.5(a1), w21=0.3(a2)
                    [0.5, 0.9]])  # b2: w12=0.5(a1), w22=0.9(a2)

    layer1_maps = fc_float_values(build_fc_structure(layer1_event_times, layer1_event_source_idx, layer1_event_times.shape[0]),
                                  W1, tau, None)
    n_real_events_1 = layer1_event_times.shape[0]
    spike_mask_1, spike_event_idx_1, s_spike_1, _, v_final_1 = run_layer_forward(
        layer1_maps, v_th, chunk_size=1, max_steps=n_real_events_1, n_real_events=n_real_events_1)

    b1_fires = [i for i in range(spike_mask_1.shape[1]) if bool(spike_mask_1[0, i])]
    b2_fires = [i for i in range(spike_mask_1.shape[1]) if bool(spike_mask_1[1, i])]
    assert len(b1_fires) == 1, b1_fires
    assert len(b2_fires) == 2, b2_fires
    assert_allclose(layer1_event_times[int(spike_event_idx_1[0, b1_fires[0]])], 3.0, "b1 fire 時間")
    assert_allclose(layer1_event_times[int(spike_event_idx_1[1, b2_fires[0]])], 2.0, "b2 第一次 fire 時間")
    assert_allclose(layer1_event_times[int(spike_event_idx_1[1, b2_fires[1]])], 5.0, "b2 第二次 fire 時間")
    assert_allclose(v_final_1[0], 0.3, "b1 最終電壓")
    assert_allclose(v_final_1[1], 0.0, "b2 最終電壓(最後一筆事件剛好 fire)")

    layer2_event_times, layer2_event_source_idx, layer2_event_gain, n_real_events_2 = \
        extract_output_events_fc(spike_mask_1, spike_event_idx_1, s_spike_1, layer1_event_times)

    n2 = int(n_real_events_2)
    assert n2 == 3, n2
    # 合併排序後應該是 t=2(來源 b2) -> t=3(來源 b1) -> t=5(來源 b2),
    # 不是照神經元編號(b1 先、b2 後)分組
    assert list(map(float, layer2_event_times[:n2])) == [2.0, 3.0, 5.0], layer2_event_times[:n2]
    assert list(map(int, layer2_event_source_idx[:n2])) == [1, 0, 1], layer2_event_source_idx[:n2]
    for i in range(n2):
        assert_allclose(layer2_event_gain[i], 1.0, f"事件{i}的 s_spike forward 應精確等於 1")

    # --- layer2: p=2 (c1,c2),輸入是 layer1 的 b1,b2 ---
    W2 = jnp.array([[0.4, 0.6],   # c1: w_c1b1=0.4, w_c1b2=0.6
                    [0.1, 0.2]])  # c2: w_c2b1=0.1, w_c2b2=0.2

    layer2_maps = fc_float_values(build_fc_structure(layer2_event_times, layer2_event_source_idx, n_real_events_2),
                                  W2, tau, layer2_event_gain)
    spike_mask_2, spike_event_idx_2, _, _, v_final_2 = run_layer_forward(
        layer2_maps, v_th, chunk_size=1, max_steps=layer2_maps.a.shape[1],
        n_real_events=n_real_events_2)

    c1_fires = [i for i in range(spike_mask_2.shape[1]) if bool(spike_mask_2[0, i])]
    assert len(c1_fires) == 1, c1_fires
    assert_allclose(layer2_event_times[int(spike_event_idx_2[0, c1_fires[0]])], 5.0, "c1 fire 時間")
    assert not bool(spike_mask_2[1].any()), "c2 不應該 fire"

    assert_allclose(v_final_2[0], 0.0, "c1 fire 後應硬重置成 0")
    assert_allclose(v_final_2[1], 0.340625, "c2 最終電壓")


def test_empty_layer_output():
    """layer1 全部神經元都不 fire 的邊界情況:extract_output_events_fc 回傳的固定
    長度陣列裡 n_real_events=0(全部都是 pad 事件),接著
    FC 佇列建構/run_layer_forward 帶 n_real_events=0 處理這個「語意上等於
    空佇列」的陣列,也不該出錯——這是先前 review 提過、但沒實測過的邊界案例。

    weights 故意設得很小,兩顆神經元在三筆事件內都不可能碰到 v_th=1.0。
    """
    tau = 4.0
    v_th = 1.0

    layer1_event_times = jnp.array([1.0, 2.0, 4.0])
    layer1_event_source_idx = jnp.array([0, 1, 0])
    W1 = jnp.array([[0.05, 0.05],
                    [0.05, 0.05]])

    layer1_maps = fc_float_values(build_fc_structure(layer1_event_times, layer1_event_source_idx, layer1_event_times.shape[0]),
                                  W1, tau, None)
    n_real_events_1 = layer1_event_times.shape[0]
    spike_mask_1, spike_event_idx_1, s_spike_1, _, _ = run_layer_forward(
        layer1_maps, v_th, chunk_size=1, max_steps=n_real_events_1, n_real_events=n_real_events_1)

    assert not bool(spike_mask_1.any()), "這組 weights 不該讓任何神經元 fire"

    layer2_event_times, layer2_event_source_idx, layer2_event_gain, n_real_events_2 = \
        extract_output_events_fc(spike_mask_1, spike_event_idx_1, s_spike_1, layer1_event_times)

    assert int(n_real_events_2) == 0, n_real_events_2
    # 固定長度 = n_source_neurons(=2) * max_steps(=3),不是動態長度 0
    assert layer2_event_times.shape == (6,), layer2_event_times.shape
    assert layer2_event_source_idx.shape == (6,), layer2_event_source_idx.shape

    # --- layer2 吃這個「語意上是空事件包」的固定長度陣列,不該出錯,
    #     結果應該是「什麼都沒發生」 ---
    W2 = jnp.array([[0.5, 0.5]])
    layer2_maps = fc_float_values(build_fc_structure(layer2_event_times, layer2_event_source_idx, n_real_events_2),
                                  W2, tau, layer2_event_gain)
    assert layer2_maps.a.shape == (1, 6)

    spike_mask_2, _, _, _, v_final_2 = run_layer_forward(
        layer2_maps, v_th, chunk_size=1, max_steps=layer2_maps.a.shape[1],
        n_real_events=n_real_events_2)

    assert not bool(spike_mask_2.any())
    assert_allclose(v_final_2[0], 0.0, "沒有任何真實輸入事件,電壓應該維持初始值 0")
