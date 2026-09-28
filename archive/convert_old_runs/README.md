# 舊實驗轉換(階段 5.4)

**封存(2026-09-28)**:一次性腳本,已經對 `experiments/` 底下 10 個 run 跑過。

- 做的事:權重檔補上網路描述(`salt_core.io` 的格式),`run.yaml` 的 `final_capacity`
  換成 `network`,`L` 改名 `max_queue_len`。細節見腳本開頭。
- 轉換後的檢查:893 個 npz 的權重跟轉換前逐位元相等,每份檔的容量跟 `metrics.csv`
  那個 epoch 一致。
- `experiments/` 現在都是新格式,這支腳本不會再用到(對已轉過的 run 會 raise)。
