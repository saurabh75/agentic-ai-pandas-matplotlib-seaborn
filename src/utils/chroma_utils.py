"""
ChromaDB utility functions for introspection and health checks.

Provides safe wrappers around ChromaDB metadata queries to avoid
crashing the application when the database is empty or corrupted.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime

from langchain_chroma import Chroma

from src.logger import get_logger
from src.models.document import DatabaseStats

logger = get_logger(__name__)


def get_database_stats(vectorstore: Chroma) -> DatabaseStats:
    """Extract statistics from a ChromaDB vector store.

    Args:
        vectorstore: Initialized ChromaDB instance.

    Returns:
        DatabaseStats with document count, chunk count, and metadata.
    """
    try:
        collection = vectorstore._collection
        count = collection.count()

        # Get unique documents by hashing through metadata
        unique_docs = set()
        last_updated = None

        if count > 0:
            # Sample metadata to find unique documents
            # ChromaDB get() without IDs returns all documents
            try:
                all_meta = collection.get(include=["metadatas"])["metadatas"]
                for meta in all_meta:
                    if meta and "document_hash" in meta:
                        unique_docs.add(meta["document_hash"])
                    if meta and "upload_date" in meta:
                        if last_updated is None or meta["upload_date"] > last_updated:
                            last_updated = meta["upload_date"]
            except Exception as e:
                logger.warning(f"Could not extract full metadata: {e}")

        return DatabaseStats(
            document_count=len(unique_docs),
            chunk_count=count,
            collection_name=collection.name,
            last_updated=last_updated,
            is_available=True,
        )

    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        return DatabaseStats(
            document_count=0,
            chunk_count=0,
            collection_name="unknown",
            is_available=False,
        )


def check_document_exists(vectorstore: Chroma, document_hash: str) -> bool:
    """Check if a document with the given hash already exists in the store.

    Args:
        vectorstore: Initialized ChromaDB instance.
        document_hash: SHA256 hash of the document.

    Returns:
        True if at least one chunk with this hash exists.
    """
    try:
        # Query by metadata filter
        results = vectorstore._collection.get(
            where={"document_hash": document_hash},
            limit=1,
        )
        return len(results["ids"]) > 0
    except Exception as e:
        logger.warning(f"Document existence check failed: {e}")
        return False


def clear_database(vectorstore: Chroma) -> bool:
    """Delete all documents from the vector store.

    Args:
        vectorstore: Initialized ChromaDB instance.

    Returns:
        True if successful.
    """
    try:
        collection = vectorstore._collection
        all_ids = collection.get()["ids"]
        if all_ids:
            collection.delete(ids=all_ids)
            logger.info(f"Deleted {len(all_ids)} chunks from vector store")
        return True
    except Exception as e:
        logger.error(f"Failed to clear database: {e}")
        return False
