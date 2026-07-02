"""
ChromaDB utility functions for introspection and health checks.

Provides safe wrappers around ChromaDB metadata queries to avoid
crashing the application when the database is empty or corrupted.
"""

import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List

from langchain_chroma import Chroma

from src.logger import get_logger
from src.models.document import DatabaseStats

logger = get_logger(__name__)


def get_database_stats(vectorstore: Chroma) -> DatabaseStats:
    """Extract statistics from a ChromaDB vector store."""
    try:
        collection = vectorstore._collection
        count = collection.count()

        unique_docs = set()
        last_updated = None

        if count > 0:
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
    try:
        results = vectorstore._collection.get(
            where={"document_hash": document_hash},
            limit=1,
        )
        return bool(results and results.get("ids"))
    except Exception as e:
        logger.warning(f"Document existence check failed: {e}")
        return False


def clear_database(vectorstore: Chroma) -> bool:
    """Delete every chunk in the collection AND drop+recreate the collection.

    Row-level delete alone can leave orphan HNSW segments on disk and
    stale in-memory caches. Dropping the collection is the only reliable
    way to reset a persistent Chroma store between sessions.
    """
    try:
        client = vectorstore._client
        collection = vectorstore._collection
        name = collection.name

        # 1) best-effort row purge
        try:
            all_ids = collection.get()["ids"]
            if all_ids:
                collection.delete(ids=all_ids)
                logger.info(f"Deleted {len(all_ids)} chunks from '{name}'")
        except Exception as e:
            logger.warning(f"Row-level delete failed (continuing): {e}")

        # 2) drop the collection so HNSW state is gone
        try:
            client.delete_collection(name=name)
            logger.info(f"Dropped collection '{name}'")
        except Exception as e:
            logger.warning(f"delete_collection failed (continuing): {e}")

        # 3) recreate empty so the next get_vector_store() finds it
        try:
            client.get_or_create_collection(name=name)
        except Exception as e:
            logger.warning(f"Recreate collection failed: {e}")

        return True
    except Exception as e:
        logger.error(f"Failed to clear database: {e}")
        return False


def wipe_all_data(
    chroma_dir: Path,
    extra_dirs: Optional[List[Path]] = None,
) -> Dict[str, bool]:
    """Hard-reset: remove chroma_db/ and any listed data folders from disk.

    Call AFTER clearing the langchain cache (clear_vector_store_cache) so
    no open handles keep the sqlite/HNSW files locked on Windows.
    """
    results: Dict[str, bool] = {}
    targets = [chroma_dir] + list(extra_dirs or [])
    for d in targets:
        try:
            if d and Path(d).exists():
                shutil.rmtree(d, ignore_errors=True)
            Path(d).mkdir(parents=True, exist_ok=True)
            results[str(d)] = True
            logger.info(f"Wiped {d}")
        except Exception as e:
            results[str(d)] = False
            logger.error(f"Failed to wipe {d}: {e}")
    return results
