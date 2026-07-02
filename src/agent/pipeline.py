"""
Agentic RAG Pipeline — Top-level orchestrator connecting all 6 stages.

  Stage 1: Input & Orchestration  (orchestrator.py)
  Stage 2: Agent Loop             (agent_loop.py)
  Stage 3: Knowledge Layer        (hybrid_retriever.py)
  Stage 4: Retrieval Quality      (context_builder.py)
  Stage 5: Reasoning & Generation (generation_engine.py)
  Stage 6: Evaluation & Feedback  (evaluator.py)

Single entry-point: AgenticRAGPipeline.run(query, session_id)
"""

from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Optional

from src.agent.orchestrator import Orchestrator
from src.agent.agent_loop import AgentLoop
from src.services.hybrid_retriever import HybridRetriever
from src.services.context_builder import RetrievalQualityPipeline
from src.services.generation_engine import GenerationEngine, get_llm, clear_session
from src.services.chart_router import ChartIntentRouter
from src.services.data_extractor import extract_dataframe
from src.services.chart_renderer import render as render_chart
from src.services import data_store, pandas_executor, stats_engine
from src.services.analysis_router import AnalysisRouter
from src.evaluation.evaluator import (
    EvaluationMetrics,
    LatencyTimer,
    RetrievalEvaluator,
    get_metric_store,
)
from src.models.agent import (
    AgentStep,
    AgenticRAGResponse,
    EvaluationMetrics,
    PolicyDecision,
)
from src.logger import get_logger

logger = get_logger(__name__)


class AgenticRAGPipeline:
    """
    Full Agentic RAG pipeline following the 2026 architecture diagram.

    Instantiate once (or cache via get_pipeline()) and call .run() per query.
    """

    def __init__(
        self,
        max_iterations: int = 3,
        max_context_tokens: int = 6000,
    ) -> None:
        self._orchestrator = Orchestrator()
        llm = get_llm()
        self._retriever = HybridRetriever()
        self._agent_loop = AgentLoop(
            llm=llm,
            hybrid_retriever=self._retriever,
            max_iterations=max_iterations,
        )
        self._quality_pipeline = RetrievalQualityPipeline(
            max_tokens=max_context_tokens
        )
        self._generator = GenerationEngine()
        self._charts_enabled = os.getenv("CHARTS_ENABLED", "true").lower() == "true"
        self._chart_router = ChartIntentRouter(llm=llm) if self._charts_enabled else None
        # Patch v5: analysis short-circuit
        self._analysis_enabled = os.getenv("ANALYSIS_ENABLED", "true").lower() == "true"
        self._analysis_router = AnalysisRouter(llm=llm) if self._analysis_enabled else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------



    def run(
        self,
        raw_query: str,
        session_id: str = "default",
        stream: bool = False,
    ) -> AgenticRAGResponse:
        """
        Execute the full 6-stage pipeline for a single user query.

        Args:
            raw_query: Raw query string from the user.
            session_id: Conversation session identifier.
            stream: If True, answer is generated via streaming (no citations/verification).

        Returns:
            AgenticRAGResponse with answer, citations, metrics, and agent step log.
        """
        with LatencyTimer() as timer:
            response = self._execute(raw_query, session_id, stream)

        # Attach final latency
        response.metrics.latency_ms = timer.elapsed_ms
        response.metrics.model_used = response.model_used

        # Record in evaluation store
        get_metric_store().record(response.metrics)

        logger.info(
            f"Pipeline complete | latency={timer.elapsed_ms:.0f}ms "
            f"| confidence={response.confidence:.2f} "
            f"| groundedness={response.groundedness_score:.2f}"
        )
        return response

    def _execute(
        self,
        raw_query: str,
        session_id: str,
        stream: bool,
    ) -> AgenticRAGResponse:
        from src.config import OLLAMA_LLM_MODEL

        # ===== Patch v5: Data-analysis short-circuit =====
        # If a tabular file is registered AND the query is a stats/EDA question,
        # bypass RAG entirely and answer with pandas/statsmodels.
        analysis_resp = self._try_analysis(raw_query)
        if analysis_resp is not None:
            return analysis_resp



        # ===== Stage 1: Orchestration =====
        effective_query, intent_analysis, task_plan, policy = (
            self._orchestrator.run(raw_query)
        )

        if policy.decision == PolicyDecision.BLOCK:
            return AgenticRAGResponse(
                answer=f"⛔ Request blocked: {policy.reason}",
                model_used=OLLAMA_LLM_MODEL,
                metrics=EvaluationMetrics(model_used=OLLAMA_LLM_MODEL),
            )

        orchestration_step = AgentStep(
            step_name="Orchestration",
            iteration=0,
            input_summary=raw_query[:200],
            output_summary=(
                f"Intent={intent_analysis.intent.value} | "
                f"sub-tasks={len(task_plan)} | "
                f"policy={policy.decision.value}"
            ),
        )

        # ===== Stage 2: Agent Loop =====
        candidates, agent_state = self._agent_loop.run(
            query=effective_query,
            intent=intent_analysis.intent,
            sub_queries=task_plan if intent_analysis.requires_multi_step else None,
        )

        all_steps = [orchestration_step] + agent_state.steps

        if not candidates:
            return AgenticRAGResponse(
                answer=(
                    "I couldn't find any relevant information in your documents. "
                    "Try rephrasing your question or uploading more documents."
                ),
                agent_steps=all_steps,
                model_used=OLLAMA_LLM_MODEL,
                metrics=EvaluationMetrics(
                    retrieval_k=0,
                    iterations_used=agent_state.iterations,
                    strategy_used=agent_state.strategy.value,
                    model_used=OLLAMA_LLM_MODEL,
                ),
            )

        # ===== Stage 4: Retrieval Quality Pipeline =====
        grounded_ctx, freshness_warnings = self._quality_pipeline.run(
            candidates=candidates,
            strategy=agent_state.strategy,
            iterations_used=agent_state.iterations,
        )

        if freshness_warnings:
            all_steps.append(AgentStep(
                step_name="Freshness Check",
                iteration=agent_state.iterations,
                input_summary=f"{len(candidates)} candidates",
                output_summary=" | ".join(freshness_warnings),
            ))

        context_string = self._quality_pipeline.to_prompt_string(grounded_ctx)

        # Retrieval evaluation
        retrieval_eval = RetrievalEvaluator.estimate(grounded_ctx.chunks)

        # ===== Stage 5: Generation =====
        if stream:
            # Streaming mode — caller handles token output
            return AgenticRAGResponse(
                answer="[streaming]",
                agent_steps=all_steps,
                grounded_context=grounded_ctx,
                model_used=OLLAMA_LLM_MODEL,
                metrics=EvaluationMetrics(
                    retrieval_k=len(grounded_ctx.chunks),
                    iterations_used=agent_state.iterations,
                    strategy_used=agent_state.strategy.value,
                    token_estimate=grounded_ctx.total_tokens_estimate,
                    model_used=OLLAMA_LLM_MODEL,
                ),
            )

        cited_answer = self._generator.generate(
            question=effective_query,
            context_string=context_string,
            ctx=grounded_ctx,
            session_id=session_id,
        )

        all_steps.append(AgentStep(
            step_name="Generation & Verification",
            iteration=agent_state.iterations,
            input_summary=f"{len(grounded_ctx.chunks)} context chunks",
            output_summary=(
                f"confidence={cited_answer.confidence:.2f} | "
                f"groundedness={cited_answer.groundedness_score:.2f} | "
                f"citations={len(cited_answer.citations)}"
            ),
            metadata={"verification_notes": cited_answer.verification_notes},
        ))

        # ===== Stage 5b: Optional chart generation =====
        chart_path, chart_caption = None, None
        if self._chart_router and self._chart_router.has_keyword_intent(raw_query):
            try:
                df, src_note = extract_dataframe(
                    grounded_ctx.chunks, raw_query, llm=get_llm()
                )
                if df is not None and not df.empty:
                    spec = self._chart_router.detect(raw_query, list(df.columns))
                    if spec and spec.is_valid():
                        rendered = render_chart(df, spec, raw_query)
                        if rendered:
                            chart_path, chart_caption = rendered
                            all_steps.append(AgentStep(
                                step_name="Chart",
                                iteration=agent_state.iterations,
                                input_summary=f"{src_note}; cols={list(df.columns)[:6]}",
                                output_summary=f"{spec.chart_type} → {chart_path}",
                            ))
            except Exception as e:
                logger.warning(f"Chart pipeline failed: {e}")

        return AgenticRAGResponse(
            answer=cited_answer.answer,
            citations=cited_answer.citations,
            confidence=cited_answer.confidence,
            groundedness_score=cited_answer.groundedness_score,
            agent_steps=all_steps,
            grounded_context=grounded_ctx,
            model_used=OLLAMA_LLM_MODEL,
            chart_path=chart_path,
            chart_caption=chart_caption,
            metrics=EvaluationMetrics(
                retrieval_k=len(grounded_ctx.chunks),
                iterations_used=agent_state.iterations,
                strategy_used=agent_state.strategy.value,
                confidence=cited_answer.confidence,
                groundedness_score=cited_answer.groundedness_score,
                token_estimate=grounded_ctx.total_tokens_estimate,
                model_used=OLLAMA_LLM_MODEL,
            ),
        )

    def stream_answer(self, raw_query: str, session_id: str = "default"):
        """
        Streaming generator.
        Yields:
          - str tokens as they arrive from the LLM (or a single str for analysis)
          - a final dict {"final": AgenticRAGResponse} with metadata (citations, chart, steps, metrics)

        Runs the analysis short-circuit first, then falls back to streaming RAG.
        """
        from src.config import OLLAMA_LLM_MODEL

        # ---- Patch v5: analysis short-circuit (instant, no LLM) ----
        analysis_resp = self._try_analysis(raw_query)
        if analysis_resp is not None:
            yield analysis_resp.answer
            yield {"final": analysis_resp}
            return

        # ---- Orchestration ----
        effective_query, intent_analysis, task_plan, policy = (
            self._orchestrator.run(raw_query)
        )
        if policy.decision == PolicyDecision.BLOCK:
            msg = f"⛔ Blocked: {policy.reason}"
            yield msg
            yield {"final": AgenticRAGResponse(
                answer=msg, model_used=OLLAMA_LLM_MODEL,
                metrics=EvaluationMetrics(model_used=OLLAMA_LLM_MODEL),
            )}
            return

        # ---- Agent loop (retrieval) ----
        candidates, agent_state = self._agent_loop.run(
            query=effective_query,
            intent=intent_analysis.intent,
            sub_queries=task_plan if intent_analysis.requires_multi_step else None,
        )
        if not candidates:
            msg = "I couldn't find any relevant information. Try uploading more documents."
            yield msg
            yield {"final": AgenticRAGResponse(
                answer=msg, model_used=OLLAMA_LLM_MODEL,
                metrics=EvaluationMetrics(model_used=OLLAMA_LLM_MODEL),
            )}
            return

        grounded_ctx, _ = self._quality_pipeline.run(candidates, agent_state.strategy)
        context_string = self._quality_pipeline.to_prompt_string(grounded_ctx)

        # ---- Stream tokens ----
        full = ""
        for token in self._generator.stream(
            question=effective_query,
            context_string=context_string,
            session_id=session_id,
        ):
            full += token
            yield token

        # ---- Build citations from the streamed answer (single-pass, no 2nd LLM) ----
        # Build lightweight (no-LLM) citations from retrieved chunks — keeps
        # streaming a true single-pass, avoiding a second Ollama call.
        try:
            from src.services.generation_engine import CitationBuilder
            citations = CitationBuilder(None)._fallback_citations(grounded_ctx)
        except Exception:
            citations = []


        final_resp = AgenticRAGResponse(
            answer=full,
            citations=citations,
            confidence=0.85 if citations else 0.6,
            groundedness_score=0.85 if citations else 0.6,
            grounded_context=grounded_ctx,
            model_used=OLLAMA_LLM_MODEL,
            agent_steps=agent_state.steps,
            metrics=EvaluationMetrics(
                retrieval_k=len(grounded_ctx.chunks),
                iterations_used=agent_state.iterations,
                strategy_used=agent_state.strategy.value,
                token_estimate=grounded_ctx.total_tokens_estimate,
                model_used=OLLAMA_LLM_MODEL,
            ),
        )
        yield {"final": final_resp}



    # ------------------------------------------------------------------
    # Patch v5: Data-analysis short-circuit
    # ------------------------------------------------------------------
    def _try_analysis(self, raw_query: str) -> Optional[AgenticRAGResponse]:
        """Return an AgenticRAGResponse when the query is a data-analysis
        question against a registered tabular file. Otherwise None."""
        from src.config import OLLAMA_LLM_MODEL

        if not self._analysis_router:
            return None
        entry = data_store.get_entry()
        if entry is None:
            return None

        spec = self._analysis_router.detect(raw_query, entry["columns"], entry["name"])
        if spec is None:
            return None

        df = data_store.load_df(entry["name"])
        if df is None:
            return None

        # Route to the right engine
        if spec.operation in ("ttest", "anova", "chi2", "regression", "normality"):
            result = stats_engine.run(spec.operation, df, spec.columns, spec.params)
        elif spec.operation == "chart":
            # Delegate to the chart pipeline against the real df
            result = None
            if self._chart_router:
                try:
                    chart_spec = self._chart_router.detect(raw_query, list(df.columns))
                    if chart_spec and chart_spec.is_valid():
                        rendered = render_chart(df, chart_spec, raw_query)
                        if rendered:
                            path, cap = rendered
                            return AgenticRAGResponse(
                                answer=f"Generated chart from `{entry['name']}`.",
                                model_used=OLLAMA_LLM_MODEL,
                                chart_path=path,
                                chart_caption=cap,
                                analysis_operation="chart",
                                agent_steps=[AgentStep(
                                    step_name="Data Analysis (Chart)",
                                    iteration=0,
                                    input_summary=f"{entry['name']} ({entry['rows']} rows)",
                                    output_summary=f"{chart_spec.chart_type} → {path}",
                                )],
                                metrics=EvaluationMetrics(model_used=OLLAMA_LLM_MODEL),
                            )
                except Exception as e:
                    logger.warning(f"Chart short-circuit failed: {e}")
            if result is None:
                return None
        else:
            result = pandas_executor.run(spec.operation, df, spec.columns, spec.params)

        if result is None:
            return None

        # Optional chart hint from the result
        chart_path, chart_caption = None, None
        hint = result.get("chart_hint")
        if hint and self._charts_enabled and result.get("df") is not None:
            try:
                from src.services.chart_router import ChartSpec
                cols = list(result["df"].columns)
                spec_c = ChartSpec(chart_type=hint,
                                   x=cols[0] if cols else None,
                                   y=cols[1] if len(cols) > 1 else None)
                if spec_c.is_valid():
                    rendered = render_chart(result["df"], spec_c, raw_query)
                    if rendered:
                        chart_path, chart_caption = rendered
            except Exception as e:
                logger.debug(f"Auto-chart skipped: {e}")

        step = AgentStep(
            step_name=f"Data Analysis · {spec.operation}",
            iteration=0,
            input_summary=f"{entry['name']} · {entry['rows']} rows · cols={spec.columns or 'auto'}",
            output_summary="pandas/statsmodels executed (no LLM call)",
        )
        return AgenticRAGResponse(
            answer=result["markdown"],
            confidence=1.0,
            groundedness_score=1.0,
            model_used=OLLAMA_LLM_MODEL,
            chart_path=chart_path,
            chart_caption=chart_caption,
            analysis_markdown=result["markdown"],
            analysis_operation=spec.operation,
            agent_steps=[step],
            metrics=EvaluationMetrics(
                retrieval_k=0,
                iterations_used=0,
                strategy_used="analysis",
                confidence=1.0,
                groundedness_score=1.0,
                model_used=OLLAMA_LLM_MODEL,
            ),
        )

    def invalidate_retrieval_cache(self) -> None:
        """Call after new documents are ingested to refresh the BM25 index."""
        self._retriever.invalidate_bm25()




# =============================================================================
# Singleton pipeline (cached for Streamlit)
# =============================================================================

_pipeline: Optional[AgenticRAGPipeline] = None


def get_pipeline() -> AgenticRAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = AgenticRAGPipeline()
    return _pipeline


def reset_pipeline() -> None:
    global _pipeline
    _pipeline = None
