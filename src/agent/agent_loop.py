"""
Stage 2 — Agent Loop

Architecture:
  Query Rewrite → Retrieval Strategy Selector → Tool/Source Selection
  → Multi-step Retrieval → Gap Detection → [loop or proceed]

The loop runs for at most `max_iterations` to prevent loop explosion.
Each iteration refines the query, adapts the strategy, and checks whether
the accumulated evidence is sufficient to answer the user's question.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from src.models.agent import (
    AgentState,
    GapAnalysis,
    QueryIntent,
    RetrievalCandidate,
    RetrievalStrategy,
)
from src.logger import get_logger

if TYPE_CHECKING:
    from langchain_ollama import OllamaLLM

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_REWRITE_PROMPT = """You are a search query optimizer.

Original user question: {original_query}
Current search query: {current_query}
Missing aspects identified: {gaps}

Rewrite the search query to specifically target the missing aspects.
Return ONLY the rewritten query, nothing else.
"""

_GAP_PROMPT = """You are an evidence evaluator.

User question: {query}

Retrieved context:
{context}

Does the retrieved context fully answer the user's question?
Answer with a JSON object in this exact format (no markdown, no extra text):
{{
  "is_sufficient": true or false,
  "missing_aspects": ["aspect1", "aspect2"],
  "confidence": 0.0 to 1.0,
  "follow_up_query": "refined query targeting missing aspects or empty string"
}}
"""

_STRATEGY_MAP: dict[QueryIntent, RetrievalStrategy] = {
    QueryIntent.FACTUAL: RetrievalStrategy.BM25,        # Keyword-precise
    QueryIntent.ANALYTICAL: RetrievalStrategy.HYBRID,   # Needs broad coverage
    QueryIntent.COMPARATIVE: RetrievalStrategy.MMR,     # Diverse sources
    QueryIntent.SUMMARIZATION: RetrievalStrategy.VECTOR,# Semantic similarity
    QueryIntent.UNKNOWN: RetrievalStrategy.HYBRID,
}


# =============================================================================
# Query Rewriter
# =============================================================================


class QueryRewriter:
    """Rewrites the current query using the LLM to target identified gaps."""

    def __init__(self, llm: "OllamaLLM") -> None:
        self._llm = llm

    def rewrite(self, state: AgentState) -> str:
        if not state.gaps:
            return state.current_query

        prompt = _REWRITE_PROMPT.format(
            original_query=state.original_query,
            current_query=state.current_query,
            gaps=", ".join(state.gaps),
        )
        try:
            rewritten = self._llm.invoke(prompt).strip()
            # Sanity: must be a non-empty single line
            rewritten = rewritten.split("\n")[0].strip()
            if not rewritten or len(rewritten) < 5:
                return state.current_query
            logger.info(f"Query rewritten: '{state.current_query}' → '{rewritten}'")
            return rewritten
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}")
            return state.current_query


# =============================================================================
# Retrieval Strategy Selector
# =============================================================================


class StrategySelector:
    """
    Selects the best retrieval strategy based on intent and iteration number.

    On later iterations (when initial strategy failed to find enough evidence)
    it escalates to a broader strategy.
    """

    def select(self, state: AgentState) -> RetrievalStrategy:
        base = _STRATEGY_MAP.get(state.intent, RetrievalStrategy.HYBRID)

        # Escalation: if we've already tried once and still have gaps, go hybrid
        if state.iterations > 0 and state.gaps:
            if base != RetrievalStrategy.HYBRID:
                logger.info(f"Strategy escalated to HYBRID on iteration {state.iterations}")
                return RetrievalStrategy.HYBRID

        logger.info(f"Strategy selected: {base} (intent={state.intent})")
        return base


# =============================================================================
# Gap Detector
# =============================================================================


class GapDetector:
    """
    Uses the LLM to evaluate whether the retrieved evidence fully answers
    the user's question, and identifies what's missing.
    """

    def __init__(self, llm: "OllamaLLM") -> None:
        self._llm = llm

    def detect(
        self,
        query: str,
        candidates: List[RetrievalCandidate],
    ) -> GapAnalysis:
        if not candidates:
            return GapAnalysis(
                is_sufficient=False,
                missing_aspects=["No documents retrieved"],
                follow_up_query=query,
                confidence=0.0,
            )

        context = "\n\n".join(
            f"[{i+1}] {c.content[:400]}"
            for i, c in enumerate(candidates[:8])
        )

        prompt = _GAP_PROMPT.format(query=query, context=context)

        try:
            raw = self._llm.invoke(prompt).strip()
            # Strip markdown fences if present
            raw = raw.replace("```json", "").replace("```", "").strip()
            import json
            data = json.loads(raw)
            return GapAnalysis(
                is_sufficient=bool(data.get("is_sufficient", False)),
                missing_aspects=data.get("missing_aspects", []),
                confidence=float(data.get("confidence", 0.5)),
                follow_up_query=data.get("follow_up_query") or None,
            )
        except Exception as e:
            logger.warning(f"Gap detection parse failed: {e}. Assuming sufficient.")
            # Fail open: if we can't parse, proceed with what we have
            return GapAnalysis(
                is_sufficient=True,
                confidence=0.4,
            )


# =============================================================================
# Agent Loop
# =============================================================================


class AgentLoop:
    """
    Controlled retrieval loop implementing the Agentic RAG architecture.

    Each iteration:
      1. Selects retrieval strategy
      2. Retrieves candidates via the hybrid retriever
      3. Detects evidence gaps
      4. Rewrites query if gaps remain
      5. Stops when evidence is sufficient or max_iterations reached

    Loop explosion is prevented by `max_iterations` (default 3).
    """

    def __init__(
        self,
        llm: "OllamaLLM",
        hybrid_retriever,
        max_iterations: int = 3,
    ) -> None:
        self._llm = llm
        self._retriever = hybrid_retriever
        self._max_iterations = max_iterations
        self._rewriter = QueryRewriter(llm)
        self._selector = StrategySelector()
        self._gap_detector = GapDetector(llm)

    def run(
        self,
        query: str,
        intent: QueryIntent = QueryIntent.UNKNOWN,
        sub_queries: List[str] | None = None,
    ) -> tuple[List[RetrievalCandidate], AgentState]:
        """
        Execute the agent loop over one or more sub-queries.

        Args:
            query: The (possibly rewritten) user query.
            intent: Classified query intent.
            sub_queries: Decomposed sub-questions from the planner.

        Returns:
            all_candidates: Deduplicated, merged candidate list.
            state: Final agent state with step log.
        """
        state = AgentState(
            original_query=query,
            current_query=query,
            intent=intent,
            max_iterations=self._max_iterations,
        )

        queries_to_run = sub_queries or [query]
        all_candidates: List[RetrievalCandidate] = []
        seen_ids: set[str] = set()

        for sub_q in queries_to_run:
            state.current_query = sub_q
            sub_candidates = self._run_single_loop(state, seen_ids)
            all_candidates.extend(sub_candidates)
            state.iterations = 0  # Reset iteration counter per sub-query

        return all_candidates, state

    def _run_single_loop(
        self,
        state: AgentState,
        seen_ids: set[str],
    ) -> List[RetrievalCandidate]:
        """Inner loop for a single query."""
        accumulated: List[RetrievalCandidate] = []

        while state.iterations < state.max_iterations:
            iteration = state.iterations + 1
            logger.info(
                f"Agent loop iteration {iteration}/{state.max_iterations} "
                f"| query='{state.current_query[:80]}'"
            )

            # 1. Select strategy
            strategy = self._selector.select(state)
            state.strategy = strategy
            state.add_step(
                "Strategy Selection",
                state.current_query,
                strategy.value,
                iteration=iteration,
            )

            # 2. Retrieve candidates
            try:
                candidates = self._retriever.retrieve(
                    query=state.current_query,
                    strategy=strategy,
                )
            except Exception as e:
                logger.error(f"Retrieval failed on iteration {iteration}: {e}")
                break

            # 3. Dedup
            new_candidates = []
            for c in candidates:
                if c.source_id not in seen_ids:
                    seen_ids.add(c.source_id)
                    new_candidates.append(c)

            accumulated.extend(new_candidates)

            state.add_step(
                "Multi-step Retrieval",
                f"Strategy={strategy.value}",
                f"Retrieved {len(candidates)} chunks ({len(new_candidates)} new)",
                total_accumulated=len(accumulated),
            )

            # 4. Gap detection
            gap = self._gap_detector.detect(state.original_query, accumulated)
            state.gaps = gap.missing_aspects

            state.add_step(
                "Gap Detection",
                f"{len(accumulated)} chunks accumulated",
                f"sufficient={gap.is_sufficient} | gaps={gap.missing_aspects}",
                confidence=gap.confidence,
            )

            state.iterations += 1

            if gap.is_sufficient:
                state.evidence_sufficient = True
                logger.info(f"Evidence sufficient after {iteration} iteration(s)")
                break

            if state.iterations >= state.max_iterations:
                logger.warning(
                    f"Max iterations ({state.max_iterations}) reached — proceeding with "
                    f"available evidence ({len(accumulated)} chunks)"
                )
                break

            # 5. Rewrite query for next iteration
            if gap.follow_up_query:
                state.current_query = gap.follow_up_query
            else:
                state.current_query = self._rewriter.rewrite(state)

            state.add_step(
                "Query Rewrite",
                f"Gaps: {gap.missing_aspects}",
                state.current_query,
            )

        return accumulated
