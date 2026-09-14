1. train/metrics.csv —— 唯一逐 epoch 的表格
每個 epoch 一列。
欄位:epoch / train_loss / val_accuracy。
每個 conv 層:max_event_queue / max_layer_spikes / max_steps / obs_event_queue /
obs_layer_spikes / obs_steps(容量 vs 實際用量)。
每一層:firing_rate、grad_norm。
每個 conv 層:dormant_frac。
decoder 自己的指標(decoder_*,依 decoder 種類而定)。
是目前唯一能直接畫「隨 epoch 變化」折線圖的來源。

2. traces/summary.npz —— 週期性逐神經元快照
每個探測 epoch(probe_every)存一次。
每層一組 (E, n) 陣列:spike_count / s_value_sum / v_final / idle_frac。
n 是該層神經元數,conv 層是 OC×H_out×W_out 展平。
目前只有 inspect_traces.py 印文字報表,完全沒畫成圖。

3. traces/full_epoch_XXX.npz —— 完整逐步軌跡
每 probe_full_every 個 epoch 存一次,只存少數樣本(預設 2 筆),檔案很大(~44MB)。
每層 (S, n, max_steps) 四欄:spike_mask / s_value / v_steps / event_ms。
是唯一有「逐步時間軸」的資料,能還原膜電位波形、spike raster、對齊真實毫秒。
目前也只有文字輸出。

4. eval/ —— 跑分結果
test.yaml:accuracy / loss / confusion_matrix / preds。
test_confusion.png:已經有圖(混淆矩陣,plot_eval.py)。

5. train/run.yaml —— metadata,不是數列
完整 config、best(epoch+val_accuracy)、final_capacity、last_epoch_obs。
適合當圖表標題或圖例用,不是拿來畫時間序列。