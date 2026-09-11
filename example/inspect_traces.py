"""讀 experiments/<run>/traces/ 的 dump,印成看得懂的東西。

`example/trace_probe.py`(訓練期)/ 之後的 `eval_test --trace` 寫出:
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

_SUMMARY_KEYS = ("spike_count", "s_value_sum", "v_final", "idle_frac")
_ACTIVITY_KEY = {"spike": "spike_count", "s_value": "s_value_sum"}


def _resolve_traces_dir(path: str) -> str:
    if os.path.isdir(os.path.join(path, "traces")):
        return os.path.join(path, "traces")
    return path


def _layer_names(files) -> list:
    """summary / full npz 的 key 是 `<層名>__<欄>`,依首次出現順序取層名。"""
    names = []
    for key in files:
        if "__" not in key:
            continue
        name = key.split("__", 1)[0]
        if name not in names:
            names.append(name)
    return names


# --------------------------------------------------------------------------
# summary.npz
# --------------------------------------------------------------------------

def report_summary(traces_dir: str, *, tau: float, activity: str, top_k: int) -> None:
    path = os.path.join(traces_dir, "summary.npz")
    if not os.path.isfile(path):
        print(f"(沒有 {path})")
        return
    s = np.load(path)
    epochs = s["epochs"]
    names = _layer_names(s.files)
    act_key = _ACTIVITY_KEY[activity]

    print(f"== summary.npz ==  {len(epochs)} 個探測 epoch:{list(epochs)}")
    print(f"   休眠用 {act_key}(tau={tau})\n")

    for name in names:
        act = s[f"{name}__{act_key}"]          # (E, n)
        n = act.shape[1]
        print(f"[{name}]  n={n}")

        # 休眠比例隨 epoch
        print("   epoch :  dormant_frac  act_p90p10")
        for e, row in zip(epochs, act):
            r = dormant_score(row, tau=tau)
            print(f"   {int(e):5d} :  {r['dormant_frac']:11.3f}  {r['act_p90p10']:10.2f}")
        first = dormant_score(act[0], tau=tau)["dormant_frac"]
        last = dormant_score(act[-1], tau=tau)["dormant_frac"]
        print(f"   趨勢   :  {first:.3f} -> {last:.3f}  (Δ {last - first:+.3f})")

        # 整段沒醒的神經元(所有探測 epoch 活動量都 ~0)
        never = int(np.sum(act.max(axis=0) < 1e-9))
        print(f"   整段沒醒 :  {never}/{n}  ({100.0 * never / n:.1f}%)")

        # 最後一個探測 epoch 的活動量分布
        finalrow = act[-1]
        p10, p50, p90 = np.percentile(finalrow, [10, 50, 90])
        print(f"   末 epoch 活動量分布 :  p10={p10:.4g}  p50={p50:.4g}  p90={p90:.4g}  "
              f"max={finalrow.max():.4g}")

        order = np.argsort(finalrow)
        hi = [(int(i), float(finalrow[i])) for i in order[::-1][:top_k]]
        lo = [(int(i), float(finalrow[i])) for i in order[:top_k]]
        print(f"   末 epoch 最活躍 {top_k} :  " + ", ".join(f"#{i}={v:.3g}" for i, v in hi))
        print(f"   末 epoch 最安靜 {top_k} :  " + ", ".join(f"#{i}={v:.3g}" for i, v in lo))

        # 其他兩個摘要量(末 epoch)
        vf = s[f"{name}__v_final"][-1]
        idle = s[f"{name}__idle_frac"][-1]
        nonfinite = int(np.sum(~np.isfinite(vf)))
        print(f"   末 epoch v_final :  mean={np.nanmean(vf):.4g}  min={np.nanmin(vf):.4g}  "
              f"max={np.nanmax(vf):.4g}  非有限={nonfinite}")
        print(f"   末 epoch idle_frac :  mean={idle.mean():.3f}  "
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
    names = _layer_names(f.files)
    n_samples = f[f"{names[0]}__spike_mask"].shape[0]
    if not 0 <= sample < n_samples:
        print(f"--sample {sample} 超出範圍(這份只有 {n_samples} 筆)")
        return

    print(f"\n== full_epoch_{epoch:03d}.npz ==  {n_samples} 筆樣本,看第 {sample} 筆\n")
    for name in names:
        sm = f[f"{name}__spike_mask"][sample]    # (n, max_steps)
        sv = f[f"{name}__s_value"][sample]
        vs = f[f"{name}__v_steps"][sample]
        ms = f[f"{name}__event_ms"][sample]
        n, steps = sm.shape
        fired = np.where(sm.sum(axis=1) > 0)[0]
        print(f"[{name}]  ({n}, {steps})  總 spike={int(sm.sum())}  "
              f"有 fire={fired.size}/{n}  "
              f"空轉步比例={np.isnan(ms).mean():.2f}  "
              f"v_steps∈[{np.nanmin(vs):.3g}, {np.nanmax(vs):.3g}]  "
              f"非有限(v/s)={int(np.sum(~np.isfinite(vs)))}/{int(np.sum(~np.isfinite(sv)))}")

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
