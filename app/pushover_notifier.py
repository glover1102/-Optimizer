"""
Pushover push-notification sender.

Sends concise alert messages to the Pushover app whenever a BUY/SELL signal
is generated, a signal outcome is resolved, or a Grade A/B optimization completes.
"""

from __future__ import annotations

import logging
import os
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

PUSHOVER_API_URL = "https://api.pushover.net/1/messages.json"


def _get_credentials() -> tuple[str, str]:
    """Return (user_key, api_token). Both must be set for notifications to send."""
    user_key = os.environ.get("PUSHOVER_USER_KEY", "")
    api_token = os.environ.get("PUSHOVER_API_TOKEN", "")
    return user_key, api_token


def _is_enabled() -> bool:
    return os.environ.get("PUSHOVER_ENABLED", "true").lower() in ("true", "1", "yes")


def _send_pushover(title: str, message: str, priority: int = 0, html: int = 1) -> None:
    """Send a single Pushover notification."""
    user_key, api_token = _get_credentials()
    if not user_key or not api_token:
        logger.debug("Pushover credentials not configured — skipping")
        return
    if not _is_enabled():
        return

    data = urlencode({
        "token": api_token,
        "user": user_key,
        "title": title,
        "message": message,
        "priority": priority,
        "html": html,
    }).encode("utf-8")

    req = Request(PUSHOVER_API_URL, data=data, method="POST")
    try:
        with urlopen(req, timeout=10) as resp:  # noqa: S310
            logger.debug("Pushover notification sent: %s (HTTP %d)", title, resp.status)
    except (URLError, OSError) as exc:
        logger.warning("Pushover notification failed (network): %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Pushover notification failed: %s", exc)


def notify_signal(symbol: str, timeframe: str, signal_dict: dict) -> None:
    """Send a Pushover alert for a BUY or SELL signal."""
    from app.notification_state import is_paused
    if is_paused():
        logger.info("Notifications paused — skipping Pushover alert for %s %s", symbol, timeframe)
        return

    try:
        action = signal_dict.get("action", "HOLD")
        if action == "HOLD":
            return

        emoji = "🟢" if action == "BUY" else "🔴"
        entry = signal_dict.get("entry_price")
        sl = signal_dict.get("sl_price")
        tp1 = signal_dict.get("tp1_price")
        tp2 = signal_dict.get("tp2_price")
        tp3 = signal_dict.get("tp3_price")
        strength = signal_dict.get("strength", 0)
        confidence = signal_dict.get("confidence", 0.0)
        regime = signal_dict.get("regime", "unknown")
        mode = signal_dict.get("entry_mode", "Hybrid")

        def _f(v):
            return f"{v:.6g}" if v is not None else "N/A"

        title = f"{emoji} {action} — {symbol} {timeframe}"
        message = (
            f"<b>Entry:</b> {_f(entry)}  <b>SL:</b> {_f(sl)}\n"
            f"<b>TP1:</b> {_f(tp1)}  <b>TP2:</b> {_f(tp2)}  <b>TP3:</b> {_f(tp3)}\n"
            f"<b>Strength:</b> {strength}/4  <b>Conf:</b> {confidence*100:.0f}%\n"
            f"<b>Regime:</b> {regime}  <b>Mode:</b> {mode}"
        )

        _send_pushover(title, message, priority=0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Pushover signal notification failed: %s", exc)


def notify_signal_outcome(signal_dict: dict) -> None:
    """Send a Pushover alert when a signal resolves."""
    from app.notification_state import is_paused
    if is_paused():
        logger.info(
            "Notifications paused — skipping Pushover outcome alert for %s %s",
            signal_dict.get("symbol", "UNKNOWN"),
            signal_dict.get("timeframe", ""),
        )
        return

    try:
        outcome = signal_dict.get("outcome")
        if not outcome:
            return

        symbol = signal_dict.get("symbol", "UNKNOWN")
        timeframe = signal_dict.get("timeframe", "")
        action = signal_dict.get("action", "")
        pnl = signal_dict.get("pnl_percent")

        if outcome == "sl_hit":
            title = f"❌ SL Hit — {symbol} {timeframe}"
        elif outcome == "expired":
            title = f"🚫 Expired — {symbol} {timeframe}"
        else:
            tp_level = outcome.replace("tp", "TP").replace("_hit", "")
            title = f"🎯 {tp_level} Hit — {symbol} {timeframe}"

        pnl_str = f"{'+' if pnl >= 0 else ''}{pnl:.2f}%" if pnl is not None else "N/A"
        message = f"<b>{action}</b> signal resolved: {outcome}\n<b>P&amp;L:</b> {pnl_str}"

        _send_pushover(title, message)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Pushover outcome notification failed: %s", exc)


def send_test_notification() -> None:
    """Send a test Pushover notification to verify connectivity."""
    _send_pushover("🔔 QTAlgo Test", "Pushover notifications are working!")
    """Send a Pushover alert for Grade A/B optimization results."""
    try:
        grade = result_dict.get("confidence_grade", "")
        if grade not in ("A", "B"):
            return

        symbol = result_dict.get("symbol", "UNKNOWN")
        timeframe = result_dict.get("timeframe", "")
        score = result_dict.get("confidence_score", 0)
        win_rate = result_dict.get("win_rate", 0)

        emoji = "🟢" if grade == "A" else "🔵"
        title = f"{emoji} Grade {grade} — {symbol} {timeframe}"
        message = f"<b>Score:</b> {score:.1f}  <b>Win:</b> {win_rate*100:.1f}%"

        _send_pushover(title, message)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Pushover optimization notification failed: %s", exc)
