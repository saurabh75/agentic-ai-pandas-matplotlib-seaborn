"""
Stage 4 — Retrieval Quality Pipeline

Pipeline:
  Candidate Chunks → Reranker (done in hybrid_retriever) → Dedup + Filter
  → Freshness / Permission Check → Context Builder → Grounded Context

Produces a clean, token-bounded GroundedContext ready for the LLM.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import List, Optional

from src.models.agent import GroundedContext, RetrievalCandidate, RetrievalStrategy
from src.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Heuristic token estimation (4 chars ≈ 1 token)
# ---------------------------------------------------------------------------
_CHARS_PER_TOKEN = 4
_DEFAULT_MAX_TOKENS = 6000


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


# =============================================================================
# Deduplication + Filter
# =============================================================================


class DeduplicatorFilter:
    """
    Removes near-duplicate chunks and applies minimum quality thresholds.

    Dedup strategy:
    - Exact: same source_id (chunk_index + document_hash)
    - Near-dup: MinHash or content-prefix fingerprint (first 100 chars)
    """

    def __init__(self, min_content_length: int = 50) -> None:
        self._min_len = min_content_length

    def filter(
        self,
        candidates: List[RetrievalCandidate],
        min_score: float = -99.0,
    ) -> List[RetrievalCandidate]:
        seen_ids: set[str] = set()
        seen_prefixes: set[str] = set()
        filtered: List[RetrievalCandidate] = []

        for c in candidates:
            # Minimum content length
            if len(c.content.strip()) < self._min_len:
                continue

            # Exact source dedup
            if c.source_id and c.source_id in seen_ids:
                continue
            if c.source_id:
                seen_ids.add(c.source_id)

            # Near-duplicate via content fingerprint
            fingerprint = hashlib.md5(c.content[:120].encode()).hexdigest()
            if fingerprint in seen_prefixes:
                continue
            seen_prefixes.add(fingerprint)

            filtered.append(c)

        logger.debug(
            f"Dedup+Filter: {len(candidates)} → {len(filtered)} candidates"
        )
        return filtered


# =============================================================================
# Freshness / Permission Check
# =============================================================================


class FreshnessPermissionChecker:
    """
    Validates that chunks are fresh enough and accessible.

    In the local RAG context:
    - Freshness: documents older than `max_age_days` emit a warning (not blocked)
    - Permission: always allow (no ACL system in local mode); log if missing metadata
    """

    def __init__(self, max_age_days: Optional[int] = None) -> None:
        self._max_age = max_age_days

    def check(
        self, candidates: List[RetrievalCandidate]
    ) -> tuple[List[RetrievalCandidate], List[str]]:
        """
        Returns:
            passed: Candidates that passed the check.
            warnings: Human-readable freshness/permission warnings.
        """
        passed: List[RetrievalCandidate] = []
        warnings: List[str] = []

        now = datetime.now(tz=timezone.utc)

        for c in candidates:
            upload_date_str = c.metadata.get("upload_date")

            # Freshness check (advisory only)
            if self._max_age and upload_date_str:
                try:
                    upload_dt = datetime.fromisoformat(upload_date_str)
                    if upload_dt.tzinfo is None:
                        upload_dt = upload_dt.replace(tzinfo=timezone.utc)
                    age_days = (now - upload_dt).days
                    if age_days > self._max_age:
                        warnings.append(
                            f"⚠️ '{c.metadata.get('filename', 'unknown')}' "
                            f"is {age_days} days old (threshold: {self._max_age}d)"
                        )
                except (ValueError, TypeError):
                    pass  # Malformed date — let it through

            # Permission check (local mode: always pass)
            passed.append(c)

        if warnings:
            logger.warning(f"Freshness warnings: {warnings}")

        return passed, warnings


# =============================================================================
# Context Builder
# =============================================================================


class ContextBuilder:
    """
    Assembles the final GroundedContext for the LLM.

    Responsibilities:
    - Token-budget enforcement (prevent context overload)
    - Structured context string generation with source labels
    - Metadata packaging for citation generation
    """

    def __init__(self, max_tokens: int = _DEFAULT_MAX_TOKENS) -> None:
        self._max_tokens = max_tokens

    def build(
        self,
        candidates: List[RetrievalCandidate],
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
        iterations_used: int = 1,
    ) -> GroundedContext:
        selected: List[RetrievalCandidate] = []
        total_tokens = 0

        for c in candidates:
            chunk_tokens = _estimate_tokens(c.content)
            if total_tokens + chunk_tokens > self._max_tokens:
                logger.info(
                    f"Token budget reached ({total_tokens}/{self._max_tokens}) "
                    f"— truncating at {len(selected)} chunks"
                )
                break
            selected.append(c)
            total_tokens += chunk_tokens

        logger.info(
            f"Context built: {len(selected)} chunks, "
            f"~{total_tokens} tokens, strategy={strategy.value}"
        )

        return GroundedContext(
            chunks=selected,
            total_tokens_estimate=total_tokens,
            strategy_used=strategy,
            iterations_used=iterations_used,
        )

    def to_prompt_string(self, ctx: GroundedContext) -> str:
        """Format the context for insertion into the LLM prompt."""
        parts = []
        for i, c in enumerate(ctx.chunks, 1):
            meta = c.metadata
            source = meta.get("filename", "Unknown")
            page = meta.get("page_number")
            label = f"[Source {i}] {source}"
            if page:
                label += f", page {page}"
            parts.append(f"{label}:\n{c.content}")
        return "\n\n---\n\n".join(parts)


# =============================================================================
# Quality Pipeline (façade)
# =============================================================================


class RetrievalQualityPipeline:
    """
    Orchestrates the full retrieval quality pipeline:
      Candidates → Dedup/Filter → Freshness/Permission → Context Builder
    """

    def __init__(
        self,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        max_age_days: Optional[int] = None,
        min_content_length: int = 50,
    ) -> None:
        self._deduplicator = DeduplicatorFilter(min_content_length)
        self._freshness = FreshnessPermissionChecker(max_age_days)
        self._builder = ContextBuilder(max_tokens)

    def run(
        self,
        candidates: List[RetrievalCandidate],
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
        iterations_used: int = 1,
    ) -> tuple[GroundedContext, List[str]]:
        """
        Execute the full quality pipeline.

        Returns:
            grounded_context: Clean, token-bounded context.
            warnings: Any freshness/permission advisories.
        """
        # Step 1: Dedup + Filter
        filtered = self._deduplicator.filter(candidates)

        # Step 2: Freshness + Permission
        passed, warnings = self._freshness.check(filtered)

        # Step 3: Context Builder
        context = self._builder.build(passed, strategy, iterations_used)

        return context, warnings

    def to_prompt_string(self, ctx: GroundedContext) -> str:
        return self._builder.to_prompt_string(ctx)
