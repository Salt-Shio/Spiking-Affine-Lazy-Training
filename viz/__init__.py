"""通用視覺化套件,跟 `salt_core`/`data`/`example` 同一層級——不依賴任何一個。

依「資料性質」分模組,不依「檔名」/「檔案格式」分模組。渲染器只認資料轉出來
之後的形狀(例如 `epoch_series` 認的是 `list[dict[str, float]]`),不知道也
不需要知道資料是從哪個檔案、哪個訓練 run 讀出來的——連檔案內的欄位怎麼分組
比較這種「這個專案特有的命名慣例」知識都不在這裡,那是消費端的事,見
`example/notebooks/plot_metrics.ipynb`。
"""
