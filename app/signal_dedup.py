"""
Signal deduplication helper.

Before persisting a new SignalRecommendation row and firing Discord/Pushover
notifications, callers should check whether the incoming signal is materially
different from the most-recent existing signal for the same symbol+timeframe.

A signal is considered a *duplicate* when every price field matches within a
relative tolerance of 0.01 % (to absorb floating-point noise) **and** the
action is the same.  HOLD signals are always treated as duplicates of an
existing HOLD — there is nothing actionable to re-notify about.

Usage
-----
    from app.signal_dedup import is_duplicate_signal

    dup = is_duplicate_signal(db, symbol, timeframe, sig)
    if dup:
        # Nothing to do — is_current was already refreshed by this function.
        logger.info("Skipping duplicate signal for %s %s", symbol, timeframe)
    else:
        # Proceed with insert + notifications.
        ...
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Relative tolerance for price comparison (0.01 %)
_PRICE_TOL = 0.0001


def _prices_match(a: Optional[float], b: Optional[float]) -> bool:
    """Return True if two nullable prices are equal within _PRICE_TOL."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    denom = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / denom < _PRICE_TOL


def is_duplicate_signal(db, symbol: str, timeframe: str, sig: dict) -> bool:
    """Check whether *sig* duplicates the most-recent signal for symbol+timeframe.

    Side-effects when a duplicate is detected
    ------------------------------------------
    The existing ``SignalRecommendation`` row's ``is_current`` attribute is set
    to ``True`` in-memory so that callers do **not** need to do a separate
    UPDATE.  Callers are responsible for committing this change.

    Returns
    -------
    True  — signal is a duplicate; caller should skip insert and notifications.
    False — signal is new/changed; caller should proceed normally.
    """
    try:
        from app.models import SignalRecommendation
        from sqlalchemy import desc

        # Fetch the most-recent signal for this symbol/timeframe
        existing: Optional[SignalRecommendation] = (
            db.query(SignalRecommendation)
            .filter(
                SignalRecommendation.symbol == symbol,
                SignalRecommendation.timeframe == timeframe,
            )
            .order_by(desc(SignalRecommendation.created_at))
            .first()
        )

        if existing is None:
            return False

        new_action = sig.get("action", "HOLD")

        # Different action → not a duplicate
        if existing.action != new_action:
            return False

        # HOLD vs HOLD → always a duplicate (nothing actionable to re-notify)
        if new_action == "HOLD":
            # Re-mark as current (in-memory; caller is responsible for commit)
            existing.is_current = True
            logger.debug(
                "Duplicate HOLD signal skipped for %s %s", symbol, timeframe
            )
            return True

        # BUY/SELL — compare price fields
        price_fields = [
            (sig.get("entry_price"), existing.entry_price),
            (sig.get("sl_price"), existing.sl_price),
            (sig.get("tp1_price"), existing.tp1_price),
            (sig.get("tp2_price"), existing.tp2_price),
            (sig.get("tp3_price"), existing.tp3_price),
        ]

        if all(_prices_match(a, b) for a, b in price_fields):
            # Re-mark as current (in-memory; caller is responsible for commit)
            existing.is_current = True
            logger.info(
                "Duplicate %s signal skipped for %s %s (prices unchanged)",
                new_action,
                symbol,
                timeframe,
            )
            return True

        return False

    except Exception as exc:  # noqa: BLE001
        # If the dedup check itself fails, allow the signal through to avoid
        # silently dropping legitimate signals.
        logger.warning("signal_dedup check failed for %s %s: %s", symbol, timeframe, exc)
        return False
