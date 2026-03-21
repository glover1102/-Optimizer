"""
Confidence scoring for optimization results.

Scoring model:
 - Statistical significance: Wilson score interval for win rate
 - Minimum 30 trades required
 - Consistency: standard deviation of walk-forward window results
 - Max consecutive losses penalty
 - Grade: A (≥80), B (65-79), C (50-64), D (35-49), F (<35)
"""

from __future__ import annotations

import math
from typing import Any


_GRADE_THRESHOLDS = [
    (80, "A"),
    (65, "B"),
    (50, "C"),
    (35, "D"),
]
_MIN_TRADES = 30
_Z = 1.96  # 95 % confidence


def _wilson_lower_bound(wins: int, n: int, z: float = _Z) -> float:
    """Lower bound of the Wilson score interval."""
    if n == 0:
        return 0.0
    p_hat = wins / n
    denominator = 1 + z**2 / n
    centre = p_hat + z**2 / (2 * n)
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    return (centre - margin) / denominator


def _grade(score: float) -> str:
    for threshold, letter in _GRADE_THRESHOLDS:
        if score >= threshold:
            return letter
    return "F"


def score_result(
    total_signals: int,
    wins: int,
    sl_hits: int,
    walk_forward_scores: list[float] | None = None,
    max_consecutive_losses: int = 0,
) -> dict[str, Any]:
    """
    Compute a composite confidence score (0–100) and letter grade.

    Parameters
    ----------
    total_signals        : total number of signals from backtest
    wins                 : number of winning trades (hit TP1 or better)
    sl_hits              : number of stop-loss hits
    walk_forward_scores  : list of OOS win-rates per walk-forward window
    max_consecutive_losses : longest losing streak observed

    Returns
    -------
    {
        "confidence_score": float (0–100),
        "confidence_grade": str ("A"–"F"),
        "win_rate": float,
        "wilson_lower": float,
        "consistency_score": float,
        "sample_size_ok": bool,
    }
    """
    if total_signals < _MIN_TRADES:
        return {
            "confidence_score": 0.0,
            "confidence_grade": "F",
            "win_rate": wins / max(total_signals, 1),
            "wilson_lower": 0.0,
            "consistency_score": 0.0,
            "sample_size_ok": False,
        }

    win_rate = wins / total_signals
    wilson_lb = _wilson_lower_bound(wins, total_signals)

    # Stat-significance component (0–40 points): scaled Wilson lower bound
    stat_score = min(40.0, wilson_lb * 80)

    # Win-rate component (0–30 points)
    wr_score = min(30.0, win_rate * 50)

    # SL penalty (up to −20 points)
    sl_rate = sl_hits / total_signals
    sl_penalty = min(20.0, sl_rate * 40)

    # Consistency component from walk-forward (0–20 points)
    consistency_score = 0.0
    if walk_forward_scores and len(walk_forward_scores) > 1:
        import statistics
        std = statistics.stdev(walk_forward_scores)
        mean_wf = statistics.mean(walk_forward_scores)
        # Low std relative to mean → high consistency
        cv = std / max(mean_wf, 0.01)
        consistency_score = max(0.0, 20.0 * (1 - min(cv, 1.0)))

    # Consecutive loss penalty (up to −5 points)
    consec_penalty = min(5.0, max_consecutive_losses * 0.5)

    raw = stat_score + wr_score - sl_penalty + consistency_score - consec_penalty
    confidence_score = round(max(0.0, min(100.0, raw)), 2)

    return {
        "confidence_score": confidence_score,
        "confidence_grade": _grade(confidence_score),
        "win_rate": round(win_rate, 4),
        "wilson_lower": round(wilson_lb, 4),
        "consistency_score": round(consistency_score, 2),
        "sample_size_ok": True,
    }
