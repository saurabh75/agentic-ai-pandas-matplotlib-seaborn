"""
Embedding service with singleton caching for Ollama embeddings.

The OllamaEmbeddings model is expensive to initialize (establishes HTTP
connection, loads model metadata). This module caches the instance using
LangChain's @lru_cache decorator, ensuring all components share one instance.
"""

from functools import lru_cache
from typing import Optional

from langchain_ollama import OllamaEmbeddings

from src.config import OLLAMA_BASE_URL, OLLAMA_EMBEDDING_MODEL
from src.logger import get_logger
from src.exceptions import OllamaConnectionError

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embeddings_model(
    base_url: Optional[str] = None,
    model_name: Optional[str] = None,
) -> OllamaEmbeddings:
    """Get a cached OllamaEmbeddings instance.

    Uses LRU cache to ensure only one embeddings model exists in memory,
    regardless of how many times this function is called. This reduces
    memory usage and eliminates redundant HTTP connections.

    Args:
        base_url: Ollama server URL. Defaults to config value.
        model_name: Embedding model name. Defaults to config value.

    Returns:
        Initialized OllamaEmbeddings instance.

    Raises:
        OllamaConnectionError: If the Ollama server is unreachable.
    """
    url = base_url or OLLAMA_BASE_URL
    model = model_name or OLLAMA_EMBEDDING_MODEL

    logger.info(f"Initializing embeddings model: {model} @ {url}")

    try:
        embeddings = OllamaEmbeddings(
            base_url=url,
            model=model,
        )
        # Test connection with a simple embedding
        _ = embeddings.embed_query("test")
        logger.info("Embeddings model initialized successfully")
        return embeddings

    except Exception as e:
        msg = (
            f"Failed to connect to Ollama at {url}. "
            f"Ensure Ollama is running and model '{model}' is pulled. "
            f"Error: {e}"
        )
        logger.error(msg)
        raise OllamaConnectionError(msg) from e


def clear_embeddings_cache() -> None:
    """Clear the embeddings model cache.

    Call this if you need to reinitialize with different parameters
    (e.g., after pulling a new model).
    """
    get_embeddings_model.cache_clear()
    logger.info("Embeddings cache cleared")
