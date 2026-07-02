"""
Custom exception hierarchy for the Local RAG Agent.

Centralizing exceptions makes error handling more precise and allows
the UI to display user-friendly messages based on exception type.
"""


class RAGAgentError(Exception):
    """Base exception for all RAG Agent errors."""
    pass


class ConfigurationError(RAGAgentError):
    """Raised when configuration is invalid or missing."""
    pass


class OllamaConnectionError(RAGAgentError):
    """Raised when the Ollama service is unreachable or returns an error."""
    pass


class DocumentProcessingError(RAGAgentError):
    """Raised when a document cannot be parsed or processed."""
    pass


class EmptyDocumentError(DocumentProcessingError):
    """Raised when a document contains no extractable text."""
    pass


class UnsupportedFormatError(DocumentProcessingError):
    """Raised when a file type is not supported."""
    pass


class VectorStoreError(RAGAgentError):
    """Raised when ChromaDB operations fail."""
    pass


class DuplicateDocumentError(RAGAgentError):
    """Raised when attempting to ingest a document that already exists."""
    pass


class SecurityError(RAGAgentError):
    """Raised when a security policy is violated (e.g., oversized file)."""
    pass


class RerankerError(RAGAgentError):
    """Raised when the reranker model fails to load or inference fails."""
    pass
