"""viz 測試共用設定:固定用不開視窗的 Agg 後端,套用跟入口程式一樣的樣式。"""
import matplotlib

from viz.style import apply_style

matplotlib.use("Agg")
apply_style()
