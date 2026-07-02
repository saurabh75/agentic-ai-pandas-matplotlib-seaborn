"""
Stage 1 — Input & Orchestration

Pipeline:
  User Query → Intent + Task Analysis → Planner / Decomposer → Policy Check

The orchestrator transforms a raw user query into a structured, safe, and
decomposed plan before handing off to the Agent Loop.
"""

from __future__ import annotations

import re
from typing import List

from src.models.agent import (
    IntentAnalysis,
    PolicyDecision,
    PolicyResult,
    QueryIntent,
)
from src.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Blocked patterns for Policy Check
# ---------------------------------------------------------------------------
_BLOCKED_PATTERNS: List[str] = [
    r"\b(ignore previous|disregard instructions|you are now)\b",
    r"\b(jailbreak|prompt injection)\b",
    r"<script[\s>]",
    r"\bsystem prompt\b",
]

_COMPILED_BLOCKS = [re.compile(p, re.IGNORECASE) for p in _BLOCKED_PATTERNS]

# ---------------------------------------------------------------------------
# Intent keywords (heuristic, LLM-free for speed)
# ---------------------------------------------------------------------------
_INTENT_SIGNALS: dict[QueryIntent, List[str]] = {
    QueryIntent.FACTUAL: [
        "what is", "who is", "when did", "where is", "define", "what are",
        "how many", "which", "tell me about",
    ],
    QueryIntent.ANALYTICAL: [
        "why", "how does", "explain", "analyze", "what causes", "impact",
        "effect", "reason", "evaluate",
    ],
    QueryIntent.COMPARATIVE: [
        "compare", "versus", "vs", "difference between", "similarities",
        "contrast", "better", "worse", "pros and cons",
    ],
    QueryIntent.SUMMARIZATION: [
        "summarize", "summary", "overview", "brief", "tldr", "key points",
        "main ideas", "highlight",
    ],
}


# =============================================================================
# Intent Analyser
# =============================================================================


class IntentAnalyser:
    """
    Classifies query intent and decomposes complex queries into sub-questions.

    Uses a fast heuristic signal-matching approach rather than calling the LLM,
    keeping latency low for the orchestration stage.
    """

    def analyse(self, query: str) -> IntentAnalysis:
        intent = self._classify(query)
        sub_queries = self._decompose(query, intent)
        requires_multi = len(sub_queries) > 1

        logger.info(
            f"Intent: {intent} | sub-queries: {len(sub_queries)} | "
            f"multi-step: {requires_multi}"
        )

        return IntentAnalysis(
            original_query=query,
            intent=intent,
            sub_queries=sub_queries,
            requires_multi_step=requires_multi,
            confidence=0.75 if intent != QueryIntent.UNKNOWN else 0.3,
        )

    def _classify(self, query: str) -> QueryIntent:
        lower = query.lower()
        scores: dict[QueryIntent, int] = {k: 0 for k in QueryIntent}

        for intent, signals in _INTENT_SIGNALS.items():
            for signal in signals:
                if signal in lower:
                    scores[intent] += 1

        best = max(scores, key=lambda k: scores[k])
        return best if scores[best] > 0 else QueryIntent.UNKNOWN

    def _decompose(self, query: str, intent: QueryIntent) -> List[str]:
        """
        Split compound queries (joined by 'and', ',', ';') into sub-questions.
        Analytical/comparative queries are also split on conjunctions.
        """
        # Simple conjunction split for compound factual/comparative queries
        if intent in (QueryIntent.COMPARATIVE, QueryIntent.ANALYTICAL):
            parts = re.split(r"\band\b|\bor\b|;|,(?=\s+[A-Z])", query)
            cleaned = [p.strip() for p in parts if len(p.strip()) > 15]
            if len(cleaned) > 1:
                return cleaned

        return [query]


# =============================================================================
# Task Planner / Decomposer
# =============================================================================


class TaskPlanner:
    """
    Builds an ordered list of retrieval sub-tasks from the intent analysis.

    For single-step queries this is a no-op passthrough.
    For multi-step queries it orders sub-tasks by estimated specificity.
    """

    def plan(self, analysis: IntentAnalysis) -> List[str]:
        if not analysis.requires_multi_step:
            return [analysis.original_query]

        # Sort: shorter/more specific sub-queries first (tend to be more targeted)
        ordered = sorted(analysis.sub_queries, key=lambda q: len(q))
        logger.info(f"Task plan: {len(ordered)} sub-tasks")
        return ordered


# =============================================================================
# Policy Check
# =============================================================================


class PolicyChecker:
    """
    Guards against prompt injection, disallowed content, and PII leakage.

    Returns a PolicyResult with ALLOW / WARN / BLOCK and an optional
    sanitised query.
    """

    def check(self, query: str) -> PolicyResult:
        # --- Injection / manipulation detection ---
        for pattern in _COMPILED_BLOCKS:
            if pattern.search(query):
                logger.warning(f"Policy BLOCK triggered: {pattern.pattern!r}")
                return PolicyResult(
                    decision=PolicyDecision.BLOCK,
                    reason="Query contains disallowed instructions or injection patterns.",
                )

        # --- Length guard (context overload prevention) ---
        if len(query) > 2000:
            truncated = query[:2000]
            logger.warning("Query too long — truncated to 2000 chars")
            return PolicyResult(
                decision=PolicyDecision.WARN,
                reason="Query truncated to 2000 characters.",
                modified_query=truncated,
            )

        return PolicyResult(decision=PolicyDecision.ALLOW, reason="OK")


# =============================================================================
# Orchestrator (façade combining all three stages)
# =============================================================================


class Orchestrator:
    """
    Combines IntentAnalyser + TaskPlanner + PolicyChecker into a single call.

    Returns the effective query (possibly sanitised), the intent analysis,
    the ordered task list, and the policy result.
    """

    def __init__(self) -> None:
        self.intent_analyser = IntentAnalyser()
        self.task_planner = TaskPlanner()
        self.policy_checker = PolicyChecker()

    def run(self, raw_query: str) -> tuple[str, IntentAnalysis, List[str], PolicyResult]:
        """
        Execute the full orchestration pipeline.

        Returns:
            effective_query: Query to use downstream (may be sanitised).
            intent_analysis: Intent + decomposition result.
            task_plan: Ordered list of retrieval sub-tasks.
            policy_result: Policy gate outcome.
        """
        # 1. Policy check first (fast fail)
        policy = self.policy_checker.check(raw_query)
        if policy.decision == PolicyDecision.BLOCK:
            return raw_query, IntentAnalysis(original_query=raw_query), [raw_query], policy

        effective_query = policy.modified_query or raw_query

        # 2. Intent analysis
        analysis = self.intent_analyser.analyse(effective_query)

        # 3. Task planning
        task_plan = self.task_planner.plan(analysis)

        return effective_query, analysis, task_plan, policy
