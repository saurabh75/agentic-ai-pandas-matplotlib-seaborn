"""
LLM service with conversation memory and source-aware prompting.

Handles:
- Ollama LLM initialization with connection health checks
- Conversation history management via ChatMessageHistory
- Contextualized question generation (follow-up -> standalone query)
- Source-citation prompt engineering
"""

from typing import Optional, List, Dict
from functools import lru_cache

from langchain_ollama import OllamaLLM
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ChatMessageHistory

from src.config import OLLAMA_BASE_URL, OLLAMA_LLM_MODEL, OLLAMA_TIMEOUT
from src.models.document import RAGResponse, RetrievalResult
from src.logger import get_logger
from src.exceptions import OllamaConnectionError

logger = get_logger(__name__)

_SESSION_STORE: Dict[str, ChatMessageHistory] = {}

SYSTEM_PROMPT = """You are a helpful research assistant that answers questions based ONLY on the provided context.

INSTRUCTIONS:
1. Answer using ONLY the information in the context documents.
2. If the answer is not in the context, say "I don't have enough information to answer that."
3. Cite your sources using [filename, page X] format when referencing specific information.
4. Be concise but complete. Include relevant details from the context.
5. Do not make up information or use outside knowledge.

CONTEXT:
{context}

Answer the user's question based on the above context."""

CONTEXTUALIZE_PROMPT = """Given the conversation history and the latest user question, formulate a standalone question that can be understood without the conversation history.

Conversation History:
{history}

Latest Question: {question}

Standalone Question:"""


@lru_cache(maxsize=1)
def get_llm() -> OllamaLLM:
    """Get cached Ollama LLM instance."""
    logger.info(f"Initializing LLM: {OLLAMA_LLM_MODEL}")
    try:
        llm = OllamaLLM(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_LLM_MODEL,
            temperature=0.1,
            timeout=OLLAMA_TIMEOUT,
        )
        _ = llm.invoke("Hello")
        logger.info("LLM initialized successfully")
        return llm
    except Exception as e:
        msg = (
            f"Failed to connect to Ollama LLM at {OLLAMA_BASE_URL}. "
            f"Ensure Ollama is running and '{OLLAMA_LLM_MODEL}' is pulled. "
            f"Error: {e}"
        )
        logger.error(msg)
        raise OllamaConnectionError(msg) from e


def get_session_history(session_id: str) -> ChatMessageHistory:
    """Get or create conversation history for a session."""
    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = ChatMessageHistory()
    return _SESSION_STORE[session_id]


def format_context(results: List[RetrievalResult]) -> str:
    """Format retrieval results into a context string for the LLM."""
    context_parts = []
    for i, result in enumerate(results, 1):
        meta = result.metadata
        source_info = f"[Source {i}] {meta.filename}"
        if meta.page_number:
            source_info += f", page {meta.page_number}"
        context_parts.append(f"{source_info}:\n{result.content}\n")
    return "\n".join(context_parts)


def contextualize_question(history: List, question: str) -> str:
    """Convert a follow-up question into a standalone query."""
    if not history:
        return question

    history_str = ""
    for msg in history[-6:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        history_str += f"{role}: {msg.content}\n"

    llm = get_llm()
    prompt = CONTEXTUALIZE_PROMPT.format(history=history_str, question=question)

    try:
        standalone = llm.invoke(prompt).strip()
        logger.info(f"Contextualized: '{question}' -> '{standalone}'")
        return standalone if standalone else question
    except Exception as e:
        logger.warning(f"Contextualization failed: {e}")
        return question


def generate_answer(
    question: str,
    retrieval_results: List[RetrievalResult],
    session_id: str = "default",
) -> RAGResponse:
    """Generate an answer using retrieved context and conversation history."""
    history = get_session_history(session_id)
    standalone_question = contextualize_question(history.messages, question)
    context = format_context(retrieval_results)

    prompt = SYSTEM_PROMPT.format(context=context) + f"\n\nQuestion: {standalone_question}\n\nAnswer:"

    llm = get_llm()
    try:
        answer = llm.invoke(prompt).strip()
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        return RAGResponse(
            answer="I encountered an error generating a response. Please try again.",
            sources=retrieval_results,
            model_used=OLLAMA_LLM_MODEL,
        )

    history.add_user_message(question)
    history.add_ai_message(answer)

    return RAGResponse(
        answer=answer,
        sources=retrieval_results,
        model_used=OLLAMA_LLM_MODEL,
    )


def clear_session(session_id: str = "default") -> None:
    """Clear conversation history for a session."""
    if session_id in _SESSION_STORE:
        del _SESSION_STORE[session_id]
        logger.info(f"Cleared session: {session_id}")
