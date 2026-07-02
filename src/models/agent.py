"""
Pydantic models for the Agentic RAG system.

Covers every stage of the architecture:
  Input & Orchestration → Agent Loop → Retrieval Quality → Generation → Evaluation
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Enumerations
# =============================================================================


class QueryIntent(str, Enum):
    """Classification of user query intent."""
    FACTUAL = "factual"           # Who/what/when direct lookups
    ANALYTICAL = "analytical"     # Why/how requiring synthesis
    COMPARATIVE = "comparative"   # Compare/contrast across sources
    SUMMARIZATION = "summarization"  # Summarize a topic or document
    UNKNOWN = "unknown"


class RetrievalStrategy(str, Enum):
    """Available retrieval strategies."""
    VECTOR = "vector"    # Pure dense embedding search
    BM25 = "bm25"        # Sparse keyword search
    HYBRID = "hybrid"    # BM25 + vector fusion
    MMR = "mmr"          # Maximal Marginal Relevance (diversity)


class PolicyDecision(str, Enum):
    """Policy gate outcome."""
    ALLOW = "allow"
    BLOCK = "block"
    WARN = "warn"


# =============================================================================
# Stage 1: Input & Orchestration
# =============================================================================


class IntentAnalysis(BaseModel):
    """Result of intent + task analysis."""
    original_query: str
    intent: QueryIntent = QueryIntent.UNKNOWN
    sub_queries: List[str] = Field(default_factory=list, description="Decomposed sub-questions")
    requires_multi_step: bool = False
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class PolicyResult(BaseModel):
    """Policy check outcome."""
    decision: PolicyDecision = PolicyDecision.ALLOW
    reason: str = ""
    modified_query: Optional[str] = None  # Sanitised version if WARN


# =============================================================================
# Stage 2: Agent Loop
# =============================================================================


class AgentStep(BaseModel):
    """Single step recorded inside the agent loop for traceability."""
    step_name: str
    iteration: int = 0
    input_summary: str
    output_summary: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GapAnalysis(BaseModel):
    """Result of gap detection — does current evidence answer the query?"""
    is_sufficient: bool = False
    missing_aspects: List[str] = Field(default_factory=list)
    follow_up_query: Optional[str] = None  # Rewritten query targeting the gap
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class AgentState(BaseModel):
    """Mutable state carried across agent loop iterations."""
    original_query: str
    current_query: str
    intent: QueryIntent = QueryIntent.UNKNOWN
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    iterations: int = 0
    max_iterations: int = 3
    evidence_sufficient: bool = False
    steps: List[AgentStep] = Field(default_factory=list)
    all_retrieved_ids: List[str] = Field(default_factory=list)  # dedup tracking
    gaps: List[str] = Field(default_factory=list)

    def add_step(self, name: str, inp: str, out: str, **meta: Any) -> None:
        self.steps.append(AgentStep(
            step_name=name,
            iteration=self.iterations,
            input_summary=inp[:300],
            output_summary=out[:300],
            metadata=meta,
        ))


# =============================================================================
# Stage 4: Retrieval Quality Pipeline
# =============================================================================


class RetrievalCandidate(BaseModel):
    """A single retrieved chunk before quality filtering."""
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    vector_score: float = 0.0
    bm25_score: float = 0.0
    hybrid_score: float = 0.0
    reranker_score: Optional[float] = None
    source_id: str = ""  # dedup key (document_hash + chunk_index)


class GroundedContext(BaseModel):
    """Final context package passed to the LLM."""
    chunks: List[RetrievalCandidate]
    total_tokens_estimate: int = 0
    strategy_used: RetrievalStrategy = RetrievalStrategy.HYBRID
    iterations_used: int = 1


# =============================================================================
# Stage 5: Reasoning & Generation
# =============================================================================


class Citation(BaseModel):
    """A single source citation in the final answer."""
    index: int
    filename: str
    page_number: Optional[int] = None
    excerpt: str = ""   # Short supporting quote
    relevance: float = 0.0


class CitedAnswer(BaseModel):
    """Final LLM answer with inline citations and quality scores."""
    answer: str
    draft_answer: str = ""      # Pre-verification draft
    citations: List[Citation] = Field(default_factory=list)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    groundedness_score: float = Field(0.0, ge=0.0, le=1.0)
    is_grounded: bool = True
    verification_notes: str = ""


# =============================================================================
# Stage 6: Evaluation & Feedback
# =============================================================================


class EvaluationMetrics(BaseModel):
    """Per-query evaluation metrics logged for every request."""
    latency_ms: float = 0.0
    retrieval_k: int = 0
    iterations_used: int = 1
    strategy_used: str = ""
    confidence: float = 0.0
    groundedness_score: float = 0.0
    token_estimate: int = 0
    model_used: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class AgenticRAGResponse(BaseModel):
    """Top-level response object returned to the UI."""
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    confidence: float = 0.0
    groundedness_score: float = 0.0
    agent_steps: List[AgentStep] = Field(default_factory=list)
    grounded_context: Optional[GroundedContext] = None
    metrics: EvaluationMetrics = Field(default_factory=EvaluationMetrics)
    model_used: str = ""
    # Optional chart produced by the chart pipeline (matplotlib + seaborn).
    chart_path: Optional[str] = None
    chart_caption: Optional[str] = None
    # Patch v5: Data-analysis mode — Markdown result from pandas/stats engine.
    # When set, this is the authoritative answer and no LLM was called.
    analysis_markdown: Optional[str] = None
    analysis_operation: Optional[str] = None
