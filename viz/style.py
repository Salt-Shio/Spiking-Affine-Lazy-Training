"""圖表共用樣式。"""
import matplotlib


def apply_style() -> None:
    """設定中文字型跟負號顯示,改的是 matplotlib 全域設定。

    由入口程式(命令列腳本、notebook)呼叫;import viz 的模組不會自動套用。
    """
    # 預設字型 DejaVu Sans 沒有中文字符,文泉驛正黑當 fallback。
    matplotlib.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
