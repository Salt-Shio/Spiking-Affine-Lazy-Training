"""把 `train/metrics.csv` 畫成圖,存回同一個 `train/` 子資料夾——這是這次
訓練 run 的產物,跟 `train/params.npz` 同一類(定位跟 `example/plot_eval.py`
對 `eval/` 做的事一樣)。

渲染邏輯本身在 `viz.epoch_series`(通用、不知道 `metrics.csv` 的欄位長怎樣);
這裡只做這個專案特有的知識——`example/metrics_log.py` 怎麼命名欄位、哪些欄
該疊在同一張子圖裡比較(同一種指標、不同層),把欄名分組再丟給 `viz` 畫。

用法:
  python -m example.plot_metrics <exp_dir>
"""
import argparse
import os

from viz.epoch_series import EpochSeriesPlot, read_epoch_series_csv
import matplotlib.pyplot as plt

from example.utils import TRAIN_DIRNAME

# 跟 example/metrics_log.py 組欄名用的 f"{name}_{suffix}" 是同一套字尾——
# 這份清單就是「哪些欄其實是同一種指標、只是不同層」的定義,只有這裡需要
# 知道這件事,viz.epoch_series 不用。長度較長的字尾排前面,避免例如
# "obs_out"/"max_out" 這種都以底線分隔、彼此不互為子字串的字尾之間互相誤判
# (目前彼此本來就不互為子字串,排序只是防禦性寫法)。
_METRIC_SUFFIXES = ["firing_rate", "grad_norm", "dormant_frac", "act_p90p10",
                    "max_out", "obs_queue", "obs_out", "L"]


def group_metrics_columns(columns: list) -> dict:
    """把 `metrics.csv` 的欄名(不含 `epoch`)分組:同一個指標字尾(不同層)
    分在一組、`decoder_*` 分在一組,其餘(`train_loss`/`val_accuracy`)各自
    獨立一組。回傳的 dict 保留欄位第一次出現的順序。"""
    groups: dict = {}
    for col in columns:
        if col.startswith("decoder_"):
            groups.setdefault("decoder", []).append(col)
            continue
        matched = next((s for s in _METRIC_SUFFIXES if col.endswith("_" + s)), None)
        if matched:
            groups.setdefault(matched, []).append(col)
        else:
            groups[col] = [col]
    return groups


def plot_metrics(exp_dir: str, ncols: int = 4) -> str:
    """讀 `<exp_dir>/train/metrics.csv`,分組畫圖,存成
    `<exp_dir>/train/metrics.png`,回傳存檔路徑。"""
    csv_path = os.path.join(exp_dir, TRAIN_DIRNAME, "metrics.csv")
    rows = read_epoch_series_csv(csv_path)
    groups = group_metrics_columns([k for k in rows[0].keys() if k != "epoch"])
    fig = EpochSeriesPlot(groups=groups, ncols=ncols).render(rows)

    out_path = os.path.join(exp_dir, TRAIN_DIRNAME, "metrics.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("exp_dir", help="一次訓練的輸出目錄(讀 <exp_dir>/train/metrics.csv)")
    parser.add_argument("--ncols", type=int, default=4)
    args = parser.parse_args()

    out_path = plot_metrics(args.exp_dir, ncols=args.ncols)
    print(f"寫入 {out_path}")


if __name__ == "__main__":
    main()
