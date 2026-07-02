"""
Vector store service with singleton caching and deduplication.

Wraps ChromaDB with:
- Singleton pattern (one instance across the app)
- Document hash-based deduplication
- Health check integration
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma

from src.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME
from src.services.embedding_service import get_embeddings_model
from src.logger import get_logger
from src.exceptions import VectorStoreError
from src.utils.chroma_utils import check_document_exists

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_vector_store(
    persist_dir: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> Chroma:
    """Get a cached ChromaDB vector store instance.

    Uses LRU cache to ensure only one vector store connection exists,
    preventing SQLite locking issues and reducing memory overhead.

    Args:
        persist_dir: Directory for ChromaDB persistence.
        collection_name: Name of the collection.

    Returns:
        Initialized Chroma instance.

    Raises:
        VectorStoreError: If the database cannot be initialized.
    """
    persist = Path(persist_dir) if persist_dir else CHROMA_PERSIST_DIR
    collection = collection_name or CHROMA_COLLECTION_NAME

    logger.info(f"Initializing vector store: {collection} @ {persist}")

    try:
        embeddings = get_embeddings_model()
        vectorstore = Chroma(
            persist_directory=str(persist),
            embedding_function=embeddings,
            collection_name=collection,
        )
        logger.info(f"Vector store ready with {vectorstore._collection.count()} chunks")
        return vectorstore

    except Exception as e:
        msg = f"Failed to initialize vector store: {e}"
        logger.error(msg)
        raise VectorStoreError(msg) from e


def is_document_indexed(document_hash: str) -> bool:
    """Check if a document already exists in the vector store.

    Args:
        document_hash: SHA256 hash of the document content.

    Returns:
        True if the document has been previously ingested.
    """
    try:
        vectorstore = get_vector_store()
        return check_document_exists(vectorstore, document_hash)
    except Exception as e:
        logger.warning(f"Could not check document existence: {e}")
        return False


def clear_vector_store_cache() -> None:
    """Clear the vector store cache.

    Use after clearing the database to force reconnection.
    """
    get_vector_store.cache_clear()
    logger.info("Vector store cache cleared")
