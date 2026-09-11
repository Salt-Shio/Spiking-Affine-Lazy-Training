"""讀 experiments/<run>/traces/ 的 dump,印成看得懂的東西。

`example/trace_probe.py`(訓練期週期性探測)寫出:
  summary.npz         逐神經元 (E, n) 摘要 + epochs (E,)
  full_epoch_XXX.npz  少數樣本的完整 (S, n, max_steps) 軌跡

這支腳本把兩者攤開:每層的休眠比例隨 epoch 怎麼變、哪些神經元整段沒醒、
活動量分布、單一神經元的膜電位波形。休眠統計直接重用
`salt_core.dormant.dormant_score`,所以離線重算跟訓練期 metrics.csv 記的一致。

用法:
  python -m example.inspect_traces <exp_dir | traces_dir> [選項]

    --tau T           dormant 門檻(預設 0.1,跟 dormant_report 一致)
    --activity A      休眠用哪個活動量:spike(預設)| s_value
    --full EPOCH      也載入 full_epoch_<EPOCH>.npz 做逐步分析
    --sample S        full 分析看第幾筆樣本(預設 0)
    --neuron N        看指定神經元(預設:該樣本最活躍的那顆)
    --top K           每層列活動量最高 / 最低的 K 顆(預設 5)
"""
import argparse
import glob
import os

import numpy as np

from salt_core.dormant import dormant_score
from salt_core.monitor import (LayerForwardTrace, layer_names, pack_key,
                               summarize_trace_scalars)
from example.utils import TRACES_DIRNAME

_ACTIVITY_KEY = {"spike": "spike_count", "s_value": "s_value_sum"}


def _resolve_traces_dir(path: str) -> str:
    if os.path.isdir(os.path.join(path, TRACES_DIRNAME)):
        return os.path.join(path, TRACES_DIRNAME)
    return path


# --------------------------------------------------------------------------
# summary.npz
# --------------------------------------------------------------------------

def _summarize_final_epoch(final_row: np.ndarray, *, top_k: int) -> dict:
    """末 epoch 活動量的分布 + 排名(純計算,不印),`report_summary` 用來組字串。"""
    p10, p50, p90 = np.percentile(final_row, [10, 50, 90])
    order = np.argsort(final_row)
    top = [(int(i), float(final_row[i])) for i in order[::-1][:top_k]]
    bottom = [(int(i), float(final_row[i])) for i in order[:top_k]]
    return {"p10": p10, "p50": p50, "p90": p90, "max": float(final_row.max()),
            "top": top, "bottom": bottom}


def _summarize_layer(act: np.ndarray, *, tau: float, top_k: int) -> dict:
    """整層 summary.npz 一個活動量欄(shape (E, n))的完整統計:逐 epoch dormant、
    趨勢、整段沒醒數、末 epoch 分布 + 排名。純計算,不印,`report_summary` 負責印。
    """
    per_epoch = [dormant_score(row, tau=tau) for row in act]
    never_woke = int(np.sum(act.max(axis=0) < 1e-9))
    return {
        "n": act.shape[1],
        "per_epoch": per_epoch,
        "trend": (per_epoch[0]["dormant_frac"], per_epoch[-1]["dormant_frac"]),
        "never_woke": never_woke,
        "final": _summarize_final_epoch(act[-1], top_k=top_k),
    }


def _summarize_vfinal_idle(vf: np.ndarray, idle: np.ndarray) -> dict:
    """末 epoch 的 v_final / idle_frac 兩個摘要量(純計算,不印)。"""
    return {"mean": float(np.nanmean(vf)), "min": float(np.nanmin(vf)),
            "max": float(np.nanmax(vf)), "nonfinite": int(np.sum(~np.isfinite(vf))),
            "idle_mean": float(idle.mean())}


def report_summary(traces_dir: str, *, tau: float, activity: str, top_k: int) -> None:
    path = os.path.join(traces_dir, "summary.npz")
    if not os.path.isfile(path):
        print(f"(沒有 {path})")
        return
    s = np.load(path)
    epochs = s["epochs"]
    names = layer_names(s.files)
    act_key = _ACTIVITY_KEY[activity]

    print(f"== summary.npz ==  {len(epochs)} 個探測 epoch:{epochs.tolist()}")
    print(f"   休眠用 {act_key}(tau={tau})\n")

    for name in names:
        act = s[pack_key(name, act_key)]          # (E, n)
        layer_stats = _summarize_layer(act, tau=tau, top_k=top_k)
        print(f"[{name}]  n={layer_stats['n']}")

        print("   epoch :  dormant_frac  act_p90p10")
        for e, r in zip(epochs, layer_stats["per_epoch"]):
            print(f"   {int(e):5d} :  {r['dormant_frac']:11.3f}  {r['act_p90p10']:10.2f}")
        first, last = layer_stats["trend"]
        print(f"   趨勢   :  {first:.3f} -> {last:.3f}  (Δ {last - first:+.3f})")

        never, n = layer_stats["never_woke"], layer_stats["n"]
        print(f"   整段沒醒 :  {never}/{n}  ({100.0 * never / n:.1f}%)")

        final = layer_stats["final"]
        print(f"   末 epoch 活動量分布 :  p10={final['p10']:.4g}  p50={final['p50']:.4g}  "
              f"p90={final['p90']:.4g}  max={final['max']:.4g}")
        print(f"   末 epoch 最活躍 {top_k} :  " +
              ", ".join(f"#{i}={v:.3g}" for i, v in final["top"]))
        print(f"   末 epoch 最安靜 {top_k} :  " +
              ", ".join(f"#{i}={v:.3g}" for i, v in final["bottom"]))

        vf_idle = _summarize_vfinal_idle(s[pack_key(name, "v_final")][-1],
                                          s[pack_key(name, "idle_frac")][-1])
        print(f"   末 epoch v_final :  mean={vf_idle['mean']:.4g}  min={vf_idle['min']:.4g}  "
              f"max={vf_idle['max']:.4g}  非有限={vf_idle['nonfinite']}")
        print(f"   末 epoch idle_frac :  mean={vf_idle['idle_mean']:.3f}  "
              f"(=1 代表該神經元探測批上完全沒收到事件)\n")


# --------------------------------------------------------------------------
# full_epoch_XXX.npz
# --------------------------------------------------------------------------

def report_full(traces_dir: str, epoch: int, *, sample: int, neuron: int | None) -> None:
    path = os.path.join(traces_dir, f"full_epoch_{epoch:03d}.npz")
    if not os.path.isfile(path):
        avail = sorted(os.path.basename(p) for p in
                       glob.glob(os.path.join(traces_dir, "full_epoch_*.npz")))
        print(f"(沒有 {path};現有:{avail})")
        return
    f = np.load(path)
    names = layer_names(f.files)
    n_samples = f[pack_key(names[0], "spike_mask")].shape[0]
    if not 0 <= sample < n_samples:
        print(f"--sample {sample} 超出範圍(這份只有 {n_samples} 筆)")
        return

    print(f"\n== full_epoch_{epoch:03d}.npz ==  {n_samples} 筆樣本,看第 {sample} 筆\n")
    for name in names:
        sm = f[pack_key(name, "spike_mask")][sample]    # (n, max_steps)
        sv = f[pack_key(name, "s_value")][sample]
        vs = f[pack_key(name, "v_steps")][sample]
        ms = f[pack_key(name, "event_ms")][sample]
        trace = LayerForwardTrace(spike_mask=sm, s_value=sv, v_steps=vs, event_ms=ms)
        stats = summarize_trace_scalars(trace)
        n = stats["n"]
        fired = stats["fired"]
        print(f"[{name}]  ({n}, {stats['steps']})  總 spike={stats['total_spikes']}  "
              f"有 fire={fired.size}/{n}  "
              f"空轉步比例={stats['idle_frac']:.2f}  "
              f"v_steps∈[{stats['v_range'][0]:.3g}, {stats['v_range'][1]:.3g}]  "
              f"非有限(v/s)={stats['nonfinite_v']}/{stats['nonfinite_s']}")

        j = neuron if neuron is not None else (int(fired[0]) if fired.size else 0)
        if not 0 <= j < n:
            print(f"   --neuron {j} 超出範圍\n")
            continue
        spk = np.where(sm[j])[0]
        print(f"   神經元 {j} :  fire {spk.size} 次" +
              (f",步 {spk[:12].tolist()}{' ...' if spk.size > 12 else ''}" if spk.size else ""))
        if spk.size:
            print(f"      對應毫秒 {np.round(ms[j, spk[:12]], 2).tolist()}")
        print(f"      v_steps 前 20 步 {np.round(vs[j, :20], 3).tolist()}")
        print(f"      s_value 前 20 步 {np.round(sv[j, :20], 3).tolist()}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="exp_dir 或它底下的 traces/")
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--activity", choices=("spike", "s_value"), default="spike")
    parser.add_argument("--full", type=int, default=None, metavar="EPOCH")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--neuron", type=int, default=None)
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    traces_dir = _resolve_traces_dir(args.path)
    if not os.path.isdir(traces_dir):
        parser.error(f"找不到目錄:{traces_dir}")

    report_summary(traces_dir, tau=args.tau, activity=args.activity, top_k=args.top)
    if args.full is not None:
        report_full(traces_dir, args.full, sample=args.sample, neuron=args.neuron)


if __name__ == "__main__":
    main()
