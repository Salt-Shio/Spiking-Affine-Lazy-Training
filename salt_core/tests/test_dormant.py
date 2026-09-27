"""salt_core/dormant.py 的測試:dormant_score(純歸約 (n,) -> dormant 統計)。用合成
活動量陣列,把「歸約公式對不對」跟「forward 對不對」(別處已驗證)分開測。
dormant_report(挑層、跑探測批)在 example,測試在 example/tests/test_dormant_report.py。
"""
import numpy as np

from salt_core.dormant import dormant_score

TOL = 1e-5


# ============================================================================
# A. dormant_score:純歸約
# ============================================================================

def test_dormant_score_uniform_activity_zero_dormant():
    """全部一樣活躍 -> score 恆 1、dormant 比例 0。"""
    stats = dormant_score(np.full(100, 0.37), tau=0.1)
    assert stats["dormant_frac"] == 0.0
    np.testing.assert_allclose(stats["score"], 1.0, atol=TOL)


def test_dormant_score_bimodal_matches_fraction():
    """20% 飽和(~1.0)+ 80% 近零(~0.02):近零那批 score < 0.1 -> dormant ~0.8。"""
    act = np.concatenate([np.full(20, 1.0), np.full(80, 0.02)])
    stats = dormant_score(act, tau=0.1)
    assert abs(stats["dormant_frac"] - 0.8) < TOL


def test_dormant_score_all_zero_is_fully_dormant():
    stats = dormant_score(np.zeros(50), tau=0.1)
    assert stats["dormant_frac"] == 1.0
    np.testing.assert_array_equal(stats["score"], np.zeros(50))


def test_dormant_score_tau_is_inclusive_and_monotone():
    act = np.concatenate([np.full(10, 1.0), np.full(90, 0.05)])
    loose = dormant_score(act, tau=0.4)["dormant_frac"]
    tight = dormant_score(act, tau=0.1)["dormant_frac"]
    assert loose == 0.9
    assert tight == 0.0
    assert loose >= tight
