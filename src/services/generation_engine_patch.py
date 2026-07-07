"""Cancellation-aware streaming wrapper.

Illustrative snippet for how to modify your existing
``generation_engine.stream()`` so a shared ``CancelToken`` can break
the loop. Merge into the existing class.
"""
from __future__ import annotations

from typing import Iterator, Optional

from src.utils.cancel_token import CancelToken


def stream_with_cancel(llm, prompt: str,
                       cancel_token: Optional[CancelToken] = None) -> Iterator[str]:
    """Yield model chunks, aborting cleanly when the token flips."""
    try:
        for chunk in llm.stream(prompt):
            if cancel_token and cancel_token.is_cancelled():
                yield "\n\n_⏹️ Stopped by user._"
                return
            text = getattr(chunk, "content", None) or str(chunk)
            yield text
    except KeyboardInterrupt:
        yield "\n\n_⏹️ Stopped by user._"
