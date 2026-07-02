"""
Pydantic data models for documents, chunks, and retrieval results.

These models enforce runtime validation, provide clear type signatures,
and serve as the contract between ingestion, retrieval, and UI layers.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator


class DocumentMetadata(BaseModel):
    """Metadata attached to every document chunk in the vector store.

    This schema is the source of truth for what metadata ChromaDB stores.
    Any changes here require rebuilding the vector database.
    """

    filename: str = Field(..., description="Original filename of the document")
    file_path: str = Field(..., description="Absolute path to the stored file")
    page_number: Optional[int] = Field(None, description="Page number (1-indexed) for paginated documents")
    upload_date: str = Field(default_factory=lambda: datetime.now().isoformat(), description="ISO 8601 timestamp of ingestion")
    document_hash: str = Field(..., description="SHA256 hash of the raw file content for deduplication")
    chunk_index: Optional[int] = Field(None, description="Index of this chunk within the document")

    @field_validator("upload_date", mode="before")
    @classmethod
    def ensure_iso_format(cls, v):
        """Ensure upload_date is always a valid ISO string."""
        if isinstance(v, datetime):
            return v.isoformat()
        return v


class IngestedDocument(BaseModel):
    """Represents a successfully ingested document with its chunks."""

    filename: str
    document_hash: str
    total_chunks: int
    upload_date: str
    status: str = "success"


class RetrievalResult(BaseModel):
    """A single chunk retrieved from the vector store with relevance information."""

    content: str = Field(..., description="The text content of the chunk")
    metadata: DocumentMetadata
    similarity_score: float = Field(..., description="Cosine similarity score from vector search (0-1)")
    reranker_score: Optional[float] = Field(None, description="Cross-encoder reranker score (higher = more relevant)")


class RAGResponse(BaseModel):
    """Complete response from the RAG pipeline to the UI."""

    answer: str = Field(..., description="The LLM-generated answer")
    sources: List[RetrievalResult] = Field(default_factory=list, description="Chunks used to generate the answer")
    query_time_ms: Optional[float] = Field(None, description="Total query processing time in milliseconds")
    model_used: str = Field("", description="Name of the LLM model that generated the answer")


class DatabaseStats(BaseModel):
    """Statistics about the current state of the vector database."""

    document_count: int = Field(0, description="Number of unique documents indexed")
    chunk_count: int = Field(0, description="Total number of chunks in the vector store")
    collection_name: str = Field("", description="Name of the ChromaDB collection")
    last_updated: Optional[str] = Field(None, description="ISO timestamp of last ingestion")
    is_available: bool = Field(False, description="Whether the vector store is accessible")
