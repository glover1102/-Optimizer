"""
In-memory notification pause/resume toggle.

This module provides a simple global flag that can be used to temporarily
suppress all outgoing Discord and Pushover notifications without changing
any configuration or environment variables.  The state is intentionally
in-memory (not persisted) so that it resets to the default "resumed" state
on every application restart — a safe default that avoids silently dropping
alerts after a redeploy.

Usage
-----
    from app.notification_state import is_paused, pause_notifications, resume_notifications

    if is_paused():
        return  # skip notification

    pause_notifications()   # suppress all future notifications
    resume_notifications()  # allow notifications again
"""

from __future__ import annotations

_paused: bool = False


def pause_notifications() -> None:
    """Pause all outgoing Discord and Pushover notifications."""
    global _paused  # noqa: PLW0603
    _paused = True


def resume_notifications() -> None:
    """Resume outgoing Discord and Pushover notifications."""
    global _paused  # noqa: PLW0603
    _paused = False


def is_paused() -> bool:
    """Return True if notifications are currently paused."""
    return _paused
