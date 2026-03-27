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


def notify_signal(symbol: str, timeframe: str, signal_dict: dict) -> None:
    """Send a Discord embed for a BUY or SELL signal recommendation.

    Parameters
    ----------
    symbol:
        Trading symbol.
    timeframe:
        Chart timeframe.
    signal_dict:
        Signal dict as returned by signal_generator.generate_signal().
    """
    try:
        webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "")
        if not webhook_url:
            return

        action = signal_dict.get("action", "HOLD")
        if action == "HOLD":
            return

        color = 0x00FF00 if action == "BUY" else 0xFF0000
        emoji = "🟢" if action == "BUY" else "🔴"
        title = f"{emoji} {action} Signal — {symbol} {timeframe}"

        entry = signal_dict.get("entry_price")
        sl = signal_dict.get("sl_price")
        tp1 = signal_dict.get("tp1_price")
        tp2 = signal_dict.get("tp2_price")
        tp3 = signal_dict.get("tp3_price")
        strength = signal_dict.get("strength", 0)
        confidence = signal_dict.get("confidence", 0.0)
        regime = signal_dict.get("regime", "unknown")
        entry_mode = signal_dict.get("entry_mode", "Pivot")
        is_confluence = signal_dict.get("is_confluence", False)

        def _fmt(v):
            return f"{v:.6g}" if v is not None else "N/A"

        fields = [
            {"name": "Entry", "value": _fmt(entry), "inline": True},
            {"name": "SL", "value": _fmt(sl), "inline": True},
            {"name": "TP1", "value": _fmt(tp1), "inline": True},
            {"name": "TP2", "value": _fmt(tp2), "inline": True},
            {"name": "TP3", "value": _fmt(tp3), "inline": True},
            {"name": "Strength", "value": f"{strength}/4", "inline": True},
            {"name": "Confidence", "value": f"{confidence * 100:.0f}%", "inline": True},
            {"name": "Regime", "value": str(regime), "inline": True},
            {"name": "Mode", "value": f"{entry_mode}{'  ⚡ Confluence' if is_confluence else ''}", "inline": True},
        ]

        embed = {
            "title": title,
            "color": color,
            "fields": fields,
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
                "Discord signal notification sent for %s %s (HTTP %d)",
                symbol, timeframe, resp.status,
            )
    except (URLError, OSError) as exc:
        logger.warning("Discord signal notification failed (network): %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Discord signal notification failed: %s", exc)


def send_startup_message() -> None:
    """Send a startup message to Discord to verify webhook connectivity."""
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "")
    if not webhook_url:
        return

    embed = {
        "title": "🤖 QTAlgo Optimizer Online",
        "description": "Optimizer is running and Discord notifications are active.",
        "color": 0x58A6FF,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    payload_data = json.dumps({"embeds": [embed]}).encode("utf-8")
    req = Request(
        webhook_url,
        data=payload_data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=10) as resp:  # noqa: S310
        logger.info("Discord startup message sent (HTTP %d)", resp.status)


def notify_signal_outcome(signal_dict: dict) -> None:
    """Send a Discord embed when a signal resolves (TP/SL hit or expired).

    Parameters
    ----------
    signal_dict:
        Dictionary containing signal fields including outcome tracking.
        Expected keys: symbol, timeframe, action, entry_price, outcome,
        outcome_price, pnl_percent, highest_tp_hit, created_at, outcome_at.
    """
    try:
        webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "")
        if not webhook_url:
            return

        outcome = signal_dict.get("outcome")
        if not outcome:
            return

        symbol = signal_dict.get("symbol", "UNKNOWN")
        timeframe = signal_dict.get("timeframe", "")
        action = signal_dict.get("action", "")
        entry_price = signal_dict.get("entry_price")
        outcome_price = signal_dict.get("outcome_price")
        pnl_percent = signal_dict.get("pnl_percent")
        highest_tp = signal_dict.get("highest_tp_hit", 0)
        created_at = signal_dict.get("created_at")
        outcome_at = signal_dict.get("outcome_at")

        # Compute duration string
        duration_str = "—"
        if created_at and outcome_at:
            try:
                if isinstance(created_at, str):
                    from datetime import datetime as _dt
                    created_at = _dt.fromisoformat(created_at)
                if isinstance(outcome_at, str):
                    from datetime import datetime as _dt
                    outcome_at = _dt.fromisoformat(outcome_at)
                delta = outcome_at - created_at
                total_mins = int(delta.total_seconds() / 60)
                if total_mins >= 60:
                    duration_str = f"{total_mins // 60}h {total_mins % 60}m"
                else:
                    duration_str = f"{total_mins}m"
            except Exception:
                pass

        def _fmt(v):
            return f"{v:.6g}" if v is not None else "N/A"

        if outcome == "expired":
            color = 0x808080
            title = f"🚫 Signal Expired — {symbol} {timeframe}"
            description = f"{action} signal expired without hitting TP or SL."
        elif outcome == "sl_hit":
            color = 0xFF0000
            title = f"❌ Stop Loss Hit — {symbol} {timeframe}"
            description = f"{action} signal stopped out."
        else:
            color = 0x00CC44
            tp_level = outcome.replace("tp", "TP").replace("_hit", "")
            title = f"🎯 {tp_level} Hit — {symbol} {timeframe}"
            description = f"{action} signal reached {tp_level}!"
            if highest_tp and highest_tp > 1:
                description += f" (Highest TP: {highest_tp})"

        fields = [
            {"name": "Action", "value": action, "inline": True},
            {"name": "Entry", "value": _fmt(entry_price), "inline": True},
            {"name": "Exit", "value": _fmt(outcome_price), "inline": True},
        ]

        if pnl_percent is not None:
            pnl_sign = "+" if pnl_percent >= 0 else ""
            fields.append({"name": "P&L", "value": f"{pnl_sign}{pnl_percent:.2f}%", "inline": True})

        fields.append({"name": "Duration", "value": duration_str, "inline": True})

        embed = {
            "title": title,
            "description": description,
            "color": color,
            "fields": fields,
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
                "Discord outcome notification sent for %s %s %s (HTTP %d)",
                symbol, timeframe, outcome, resp.status,
            )
    except (URLError, OSError) as exc:
        logger.warning("Discord outcome notification failed (network): %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Discord outcome notification failed: %s", exc)
