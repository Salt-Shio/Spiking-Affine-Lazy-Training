"""Dormant neuron 診斷(Sokar et al. 2023, arXiv:2302.12902)的歸約:逐神經元活動量 ->
dormant 統計。定義跟用途見 docs/math/初始權重尺度推導.md 步驟 7。

挑哪幾層、跑哪批探測樣本是應用的決定,在 example/dormant.py。
"""
import numpy as np


def dormant_score(activity, *, tau: float = 0.1) -> dict:
    """逐神經元活動量 `(n,)`(已在探測樣本上平均、非負)-> dormant 統計。

    純歸約,不跑 forward —— ReDo 挑回收對象、`dormant_report` 寫紀錄都用這個。

    回傳 {"dormant_frac": float, "score": (n,) ndarray}:
      dormant_frac = #{score_i <= tau} / n,score_i = activity_i / 層平均。
    層平均為 0(整層全死)時 dormant_frac = 1.0、score 全 0。
    """
    activity = np.abs(np.asarray(activity, dtype=np.float64))
    denom = float(np.mean(activity))
    if denom <= 0.0:
        return {"dormant_frac": 1.0, "score": np.zeros_like(activity)}
    score = activity / denom
    return {
        "dormant_frac": float(np.mean(score <= tau)),
        "score": score,
    }
