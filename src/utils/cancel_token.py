"""Cooperative cancellation token for LLM streaming.

Streamlit blocks its main thread during a generation call, so cancellation
must be checked *inside* the streaming loop. This module gives every
generation path a shared ``threading.Event`` it can poll.
"""
from __future__ import annotations

import threading


class CancelToken:
    """Thread-safe cancel signal."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def reset(self) -> None:
        self._event.clear()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        """Raise :class:`KeyboardInterrupt` if the token has been cancelled."""
        if self._event.is_set():
            raise KeyboardInterrupt("Generation stopped by user")


_GLOBAL_TOKEN = CancelToken()


def get_cancel_token() -> CancelToken:
    """Return the process-wide cancel token."""
    return _GLOBAL_TOKEN
