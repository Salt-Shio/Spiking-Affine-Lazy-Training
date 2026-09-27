"""通用視覺化套件,跟 salt_core、data、example 同一層級。

依賴:
- replay_panels.py 依賴 salt_core(把層跟 forward 軌跡轉成動畫 panel)。
- 其餘模組不依賴 salt_core、data、example。
- 測試 tests/test_replay_panels.py 另外用到 example(先跑一次小訓練當素材)。

依資料性質分模組,渲染器只認轉好的資料形狀,不知道資料從哪個檔案讀來;
欄位怎麼分組這類專案慣例由呼叫端決定,例子見 example/notebooks/plot_metrics.ipynb。
中文字型等樣式由入口程式呼叫 viz.style.apply_style() 套用,import 時不改全域設定。
"""
