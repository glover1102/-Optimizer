"""
Discord webhook notifier for high-grade optimization results.

Sends a rich embed to the configured Discord webhook URL whenever an
optimization completes with a confidence grade of A or B.  All errors
are caught and logged — this module must never crash the optimizer.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_GRADE_COLOR = {
    "A": 0x00FF00,   # green
    "B": 0x3498DB,   # blue
}

_GRADE_EMOJI = {
    "A": "🟢",
    "B": "🔵",
}


def notify_optimization_result(result_dict: dict) -> None:
    """Send a Discord embed for a Grade-A or Grade-B optimization result.

    Parameters
    ----------
    result_dict:
        Dictionary containing optimization result fields.  Expected keys:
        symbol, timeframe, confidence_grade, confidence_score, win_rate,
        tp2_rate, tp3_rate, sl_rate, left_bars, right_bars, offset,
        atr_multiplier, atr_period, regime, walk_forward_score.
    """
    try:
        webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "")
        if not webhook_url:
            logger.debug("DISCORD_WEBHOOK_URL not set — skipping notification")
            return

        grade = result_dict.get("confidence_grade", "")
        if grade not in ("A", "B"):
            return

        symbol = result_dict.get("symbol", "UNKNOWN")
        timeframe = result_dict.get("timeframe", "")
        score = result_dict.get("confidence_score", 0)
        win_rate = result_dict.get("win_rate", 0)
        tp2_rate = result_dict.get("tp2_rate", 0)
        tp3_rate = result_dict.get("tp3_rate", 0)
        sl_rate = result_dict.get("sl_rate", 0)
        left_bars = result_dict.get("left_bars", "?")
        right_bars = result_dict.get("right_bars", "?")
        offset = result_dict.get("offset", 0)
        atr_multiplier = result_dict.get("atr_multiplier", 0)
        atr_period = result_dict.get("atr_period", "?")
        regime = result_dict.get("regime", "unknown")
        wf_score = result_dict.get("walk_forward_score", 0)

        emoji = _GRADE_EMOJI[grade]
        color = _GRADE_COLOR[grade]
        title = f"{emoji} Grade {grade} — {symbol} {timeframe}"

        embed = {
            "title": title,
            "color": color,
            "fields": [
                {
                    "name": "Score",
                    "value": f"{score:.1f}",
                    "inline": True,
                },
                {
                    "name": "Win%",
                    "value": f"{win_rate * 100:.1f}%",
                    "inline": True,
                },
                {
                    "name": "TP2%",
                    "value": f"{tp2_rate * 100:.1f}%",
                    "inline": True,
                },
                {
                    "name": "TP3%",
                    "value": f"{tp3_rate * 100:.1f}%",
                    "inline": True,
                },
                {
                    "name": "SL%",
                    "value": f"{sl_rate * 100:.1f}%",
                    "inline": True,
                },
                {
                    "name": "Params",
                    "value": (
                        f"LB={left_bars} RB={right_bars} | "
                        f"Offset={offset:.2f} | "
                        f"Mult={atr_multiplier:.2f} | "
                        f"ATR Period={atr_period}"
                    ),
                    "inline": False,
                },
                {
                    "name": "Regime",
                    "value": str(regime),
                    "inline": True,
                },
                {
                    "name": "Walk-Forward Score",
                    "value": f"{wf_score:.1f}" if wf_score is not None else "N/A",
                    "inline": True,
                },
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        payload = json.dumps({"embeds": [embed]}).encode("utf-8")
        req = Request(
            webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=10) as resp:  # noqa: S310
            logger.debug(
                "Discord notification sent for %s %s (HTTP %d)",
                symbol, timeframe, resp.status,
            )
    except (URLError, OSError) as exc:
        logger.warning("Discord notification failed (network): %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Discord notification failed: %s", exc)
