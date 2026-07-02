"""
Centralized configuration for the Local RAG Agent.

All settings are loaded from environment variables with sensible defaults.
This ensures the application can be configured without modifying code,
and secrets (like API keys) never appear in version control.
"""

import os
from pathlib import Path
from typing import List, Set
from dotenv import load_dotenv

# Load .env file from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


# =============================================================================
# Ollama Configuration
# =============================================================================

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
"""Base URL for the Ollama API server."""

OLLAMA_EMBEDDING_MODEL: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
"""Model name for generating text embeddings. Must be pulled via `ollama pull`."""

OLLAMA_LLM_MODEL: str = os.getenv("OLLAMA_LLM_MODEL", "llama3.1:8b")
"""Model name for the chat/QA LLM. Must be pulled via `ollama pull`."""

OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "120"))
"""HTTP timeout in seconds for Ollama API calls."""


# =============================================================================
# Vector Store Configuration
# =============================================================================

CHROMA_PERSIST_DIR: Path = _PROJECT_ROOT / os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
"""Directory where ChromaDB persists its SQLite database and index files."""

CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "documents")
"""Name of the ChromaDB collection for document chunks."""


# =============================================================================
# Text Chunking Configuration
# =============================================================================

CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
"""Maximum number of characters per text chunk.

Larger chunks preserve more context but increase retrieval noise.
Smaller chunks improve precision but may fragment coherent passages.
"""

CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
"""Number of overlapping characters between consecutive chunks.

Ensures no semantic boundaries are lost at chunk boundaries.
"""


# =============================================================================
# Retrieval Configuration
# =============================================================================

FETCH_K: int = int(os.getenv("FETCH_K", "20"))
"""Number of chunks to retrieve from the vector store in the first stage.

This is the "recall" stage — we fetch more than we need to give the
reranker a diverse pool to select from.
"""

RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", "5"))
"""Number of chunks to return after reranking.

This is the "precision" stage — only the highest-quality chunks
are passed to the LLM, maximizing context window utility.
"""

MMR_LAMBDA: float = float(os.getenv("MMR_LAMBDA", "0.5"))
"""Lambda parameter for Maximal Marginal Relevance (MMR).

- 0.0 = Maximum diversity (ignores relevance)
- 1.0 = Maximum relevance (ignores diversity)
- 0.5 = Balanced tradeoff (recommended default)
"""


# =============================================================================
# Reranker Configuration
# =============================================================================

RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
"""HuggingFace model name for the cross-encoder reranker.

Cross-encoders jointly encode query+document pairs, producing
far more accurate relevance scores than bi-encoder embeddings.
"""

RERANKER_DEVICE: str = os.getenv("RERANKER_DEVICE", "auto")
"""Device for reranker inference: 'cpu', 'cuda', 'cuda:0', or 'auto'."""


# =============================================================================
# Document Processing Configuration
# =============================================================================

DOCUMENTS_DIR: Path = _PROJECT_ROOT / os.getenv("DOCUMENTS_DIR", "documents")
"""Directory where uploaded/ingested documents are stored."""

MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
"""Maximum allowed file size in megabytes. Prevents DoS via large uploads."""

MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024
"""Maximum file size in bytes (computed from MB setting)."""

SUPPORTED_EXTENSIONS: Set[str] = set(
    ext.strip().lower()
    for ext in os.getenv("SUPPORTED_EXTENSIONS", ".pdf,.docx,.xlsx,.xls,.pptx,.ppt,.txt,.md,.csv").split(",")
)
"""Set of supported file extensions (with leading dot, lowercase)."""


# =============================================================================
# Logging Configuration
# =============================================================================

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
"""Python logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL."""

LOG_FORMAT: str = os.getenv(
    "LOG_FORMAT",
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
"""Format string for log messages."""


# =============================================================================
# Application Configuration
# =============================================================================

APP_TITLE: str = os.getenv("APP_TITLE", "Local RAG Agent")
"""Title displayed in the Streamlit UI."""

APP_PORT: int = int(os.getenv("APP_PORT", "8501"))
"""Port for the Streamlit development server."""


# =============================================================================
# Derived / Computed Settings
# =============================================================================

def ensure_directories() -> None:
    """Create required directories if they don't exist."""
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)


# Run on import
ensure_directories()
