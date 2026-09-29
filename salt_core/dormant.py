"""dormant neuron 診斷的歸約:逐神經元活動量 -> dormant 統計。

定義見 docs/math/初始權重尺度推導.md「診斷該看分布,不是平均 firing rate(dormant score)」。
挑哪幾層、用哪批樣本由呼叫端決定。
"""
from typing import TypedDict

import numpy as np
import numpy.typing as npt


class DormantScore(TypedDict):
    """dormant_score 的回傳。"""
    dormant_frac: float   # score <= tau 的比例
    score: np.ndarray     # (n,) 每顆神經元的 score


def dormant_score(activity: npt.ArrayLike, *, tau: float = 0.1) -> DormantScore:
    """逐神經元活動量 -> dormant 統計。純歸約,不跑 forward。

    activity: (n,) 已在樣本上平均的活動量,取絕對值後用。
    tau: score 小於等於它就算 dormant。

    score_i = activity_i / 層平均,dormant_frac = score <= tau 的比例。
    層平均是 0(整層全死)時 dormant_frac = 1.0、score 全 0。
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
