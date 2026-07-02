"""
Stage 5 — Reasoning & Generation

Pipeline:
  LLM Reasoning → Draft Answer → Citation Builder → Verifier / Groundedness Check
  → Final Answer (with citations + confidence)

The verifier is the key guard against hallucination: it scores how well
the draft answer is supported by the grounded context before it is
shown to the user.
"""

from __future__ import annotations

import json
import re
from typing import Dict, Generator, List, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_ollama import OllamaLLM

from src.config import OLLAMA_BASE_URL, OLLAMA_LLM_MODEL, OLLAMA_TIMEOUT
from src.models.agent import Citation, CitedAnswer, GroundedContext
from src.logger import get_logger
from src.exceptions import OllamaConnectionError

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Session store (in-memory, keyed by session_id)
# ---------------------------------------------------------------------------
_SESSION_STORE: Dict[str, ChatMessageHistory] = {}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_REASONING_SYSTEM = """You are a precise research assistant. Answer the user's question based ONLY on the provided context documents.

RULES:
1. Use ONLY information from the context — never use outside knowledge.
2. If the context does not contain sufficient information, say so clearly.
3. Reference sources using [Source N] notation inline (e.g. "According to [Source 1]...").
4. Be thorough but concise. Do not pad with filler.
5. If asked to compare or analyse, address all aspects.

CONTEXT:
{context}
"""

_CITATION_PROMPT = """Given the answer below and the source list, extract inline citations.

Answer:
{answer}

Sources (index → filename):
{source_list}

Return ONLY a JSON array of citation objects. Each object must have:
  "index": <int, 1-based source number>,
  "filename": <str>,
  "page_number": <int or null>,
  "excerpt": <str, ≤ 80 chars of supporting text from the answer>,
  "relevance": <float 0.0-1.0>

Return nothing except the JSON array.
"""

_GROUNDEDNESS_PROMPT = """You are a fact-checking assistant.

User question: {question}

Context:
{context}

Draft answer:
{answer}

Score the answer on TWO dimensions (0.0 to 1.0):
1. groundedness: How well is every claim in the answer supported by the context?
2. confidence: How complete and certain is the answer given the context?

Also provide brief verification_notes (1-2 sentences).

Return ONLY JSON in this format (no markdown, no extra text):
{{
  "groundedness_score": <float>,
  "confidence": <float>,
  "is_grounded": <true if groundedness_score >= 0.6>,
  "verification_notes": "<string>"
}}
"""

_CONTEXTUALIZE_PROMPT = """Given the conversation history and the latest user question, write a standalone question.

History:
{history}

Latest question: {question}

Standalone question (return ONLY the question, nothing else):"""


# =============================================================================
# LLM factory (cached)
# =============================================================================


_LLM_INSTANCE: Optional[OllamaLLM] = None


def get_llm() -> OllamaLLM:
    """Get a cached Ollama LLM instance, raising OllamaConnectionError if down."""
    global _LLM_INSTANCE
    if _LLM_INSTANCE is not None:
        return _LLM_INSTANCE

    logger.info(f"Initialising LLM: {OLLAMA_LLM_MODEL}")
    try:
        llm = OllamaLLM(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_LLM_MODEL,
            temperature=0.1,
            timeout=OLLAMA_TIMEOUT,
        )
        _ = llm.invoke("ping")
        _LLM_INSTANCE = llm
        logger.info("LLM initialised successfully")
        return llm
    except Exception as e:
        msg = (
            f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
            f"Ensure Ollama is running and '{OLLAMA_LLM_MODEL}' is pulled. Error: {e}"
        )
        logger.error(msg)
        raise OllamaConnectionError(msg) from e


def clear_llm_cache() -> None:
    global _LLM_INSTANCE
    _LLM_INSTANCE = None


# =============================================================================
# Session memory helpers
# =============================================================================


def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = ChatMessageHistory()
    return _SESSION_STORE[session_id]


def clear_session(session_id: str) -> None:
    _SESSION_STORE.pop(session_id, None)
    logger.info(f"Session cleared: {session_id}")


def contextualize_question(session_id: str, question: str) -> str:
    """Convert a follow-up question into a standalone query using chat history."""
    history = get_session_history(session_id)
    if not history.messages:
        return question

    history_str = ""
    for msg in history.messages[-6:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        history_str += f"{role}: {msg.content}\n"

    try:
        llm = get_llm()
        prompt = _CONTEXTUALIZE_PROMPT.format(history=history_str, question=question)
        standalone = llm.invoke(prompt).strip().split("\n")[0].strip()
        return standalone if len(standalone) > 5 else question
    except Exception as e:
        logger.warning(f"Contextualization failed: {e}")
        return question


# =============================================================================
# Citation Builder
# =============================================================================


class CitationBuilder:
    """Extracts structured citations from the draft answer via an LLM call."""

    def __init__(self, llm: OllamaLLM) -> None:
        self._llm = llm

    def build(
        self, answer: str, ctx: GroundedContext
    ) -> List[Citation]:
        source_list = "\n".join(
            f"{i+1}: {c.metadata.get('filename', 'unknown')} "
            f"(page {c.metadata.get('page_number', 'N/A')})"
            for i, c in enumerate(ctx.chunks)
        )
        prompt = _CITATION_PROMPT.format(answer=answer, source_list=source_list)

        try:
            raw = self._llm.invoke(prompt).strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            data = json.loads(raw)
            if not isinstance(data, list):
                return []
            citations = []
            for item in data:
                try:
                    citations.append(Citation(**item))
                except Exception:
                    pass
            return citations
        except Exception as e:
            logger.warning(f"Citation extraction failed: {e}. Generating fallback.")
            return self._fallback_citations(ctx)

    def _fallback_citations(self, ctx: GroundedContext) -> List[Citation]:
        return [
            Citation(
                index=i + 1,
                filename=c.metadata.get("filename", "unknown"),
                page_number=c.metadata.get("page_number"),
                excerpt="",
                relevance=c.reranker_score or c.hybrid_score or c.vector_score,
            )
            for i, c in enumerate(ctx.chunks[:5])
        ]


# =============================================================================
# Groundedness Verifier
# =============================================================================


class GroundednessVerifier:
    """
    Scores draft answer groundedness before showing it to the user.

    This is the key guard against hallucinated synthesis (failure point F in
    the architecture diagram).
    """

    def __init__(self, llm: OllamaLLM) -> None:
        self._llm = llm

    def verify(
        self,
        question: str,
        context_string: str,
        answer: str,
    ) -> dict:
        prompt = _GROUNDEDNESS_PROMPT.format(
            question=question,
            context=context_string[:4000],
            answer=answer,
        )
        try:
            raw = self._llm.invoke(prompt).strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"Groundedness verification failed: {e}")
            return {
                "groundedness_score": 0.5,
                "confidence": 0.5,
                "is_grounded": True,
                "verification_notes": "Verification unavailable.",
            }


# =============================================================================
# Generation Engine (façade)
# =============================================================================


class GenerationEngine:
    """
    Combines LLM reasoning + citation builder + groundedness verifier.

    Produces a CitedAnswer with confidence scores and inline citations.
    """

    def __init__(self) -> None:
        self._llm = get_llm()
        self._citation_builder = CitationBuilder(self._llm)
        self._verifier = GroundednessVerifier(self._llm)

    def generate(
        self,
        question: str,
        context_string: str,
        ctx: GroundedContext,
        session_id: str = "default",
        system_override: str | None = None,   # Patch v9.3
    ) -> CitedAnswer:
        """
        Run the full Reasoning → Draft → Citation → Verify pipeline.

        Patch v9.3: if `system_override` is provided (a fully-rendered senior-
        expert persona prompt with CONTEXT already embedded), use it in place
        of the generic _REASONING_SYSTEM.
        """
        # --- Contextualize question (handle follow-ups) ---
        standalone = contextualize_question(session_id, question)

        # --- Draft answer ---
        system = system_override if system_override else _REASONING_SYSTEM.format(context=context_string)
        prompt = f"{system}\n\nQuestion: {standalone}\n\nAnswer:"
        try:
            draft = self._llm.invoke(prompt).strip()
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            draft = "I encountered an error generating a response. Please try again."

        # --- Citations ---
        citations = self._citation_builder.build(draft, ctx)

        # --- Groundedness verification ---
        verdict = self._verifier.verify(standalone, context_string, draft)

        # --- Update session memory ---
        history = get_session_history(session_id)
        history.add_user_message(question)
        history.add_ai_message(draft)

        return CitedAnswer(
            answer=draft,
            draft_answer=draft,
            citations=citations,
            confidence=float(verdict.get("confidence", 0.5)),
            groundedness_score=float(verdict.get("groundedness_score", 0.5)),
            is_grounded=bool(verdict.get("is_grounded", True)),
            verification_notes=verdict.get("verification_notes", ""),
        )

    def stream(
        self,
        question: str,
        context_string: str,
        session_id: str = "default",
        system_override: str | None = None,   # Patch v9.3
    ) -> Generator[str, None, None]:
        """
        Streaming variant — yields text tokens for Streamlit st.write_stream().
        Citations and groundedness are NOT computed in streaming mode.
        """
        standalone = contextualize_question(session_id, question)
        system = system_override if system_override else _REASONING_SYSTEM.format(context=context_string)
        prompt = f"{system}\n\nQuestion: {standalone}\n\nAnswer:"
        full_answer = ""
        try:
            for chunk in self._llm.stream(prompt):
                full_answer += chunk
                yield chunk
        except Exception as e:
            yield f"\n\n❌ Streaming error: {e}"
            full_answer = f"Error: {e}"

        # Persist to memory even in streaming mode
        history = get_session_history(session_id)
        history.add_user_message(question)
        history.add_ai_message(full_answer)
