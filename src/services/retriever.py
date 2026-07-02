"""
Retrieval service with MMR and cross-encoder reranking.

Architecture:
  Query → Chroma MMR(fetch_k=20) → BGE Reranker → Top 5 → LLM

MMR (Maximal Marginal Relevance) ensures diverse coverage by penalizing
chunks that are too similar to already-selected chunks. This prevents
the common problem where all top-k results say the same thing.

The BGE reranker is a cross-encoder: it jointly encodes query+document,
producing more accurate relevance scores than the bi-encoder embedding
model used for initial retrieval.
"""

from typing import List, Optional
from functools import lru_cache

import torch
from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.config import FETCH_K, RETRIEVAL_K, MMR_LAMBDA, RERANKER_MODEL, RERANKER_DEVICE
from src.services.vector_store import get_vector_store
from src.logger import get_logger
from src.exceptions import RerankerError
from src.models.document import RetrievalResult, DocumentMetadata

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_reranker():
    """Get cached BGE reranker model.

    Uses LRU cache to avoid reloading the ~400MB model on every query.
    Falls back to CPU if CUDA is unavailable.

    Returns:
        CrossEncoder instance, or None if loading fails.
    """
    try:
        from sentence_transformers import CrossEncoder

        device = RERANKER_DEVICE
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading reranker: {RERANKER_MODEL} on {device}")

        reranker = CrossEncoder(
            RERANKER_MODEL,
            device=device,
            max_length=512,
        )
        logger.info("Reranker loaded successfully")
        return reranker

    except Exception as e:
        logger.error(f"Failed to load reranker: {e}")
        return None


class RAGRetriever:
    """Retrieval pipeline with MMR and cross-encoder reranking."""

    def __init__(
        self,
        vectorstore: Optional[Chroma] = None,
        fetch_k: int = FETCH_K,
        retrieval_k: int = RETRIEVAL_K,
        mmr_lambda: float = MMR_LAMBDA,
    ):
        """Initialize the retriever.

        Args:
            vectorstore: ChromaDB instance. Uses singleton if None.
            fetch_k: Number of candidates to fetch from vector store.
            retrieval_k: Number of chunks to return after reranking.
            mmr_lambda: MMR diversity/relevance tradeoff (0.0-1.0).
        """
        self.vectorstore = vectorstore or get_vector_store()
        self.fetch_k = fetch_k
        self.retrieval_k = retrieval_k
        self.mmr_lambda = mmr_lambda
        self._reranker = None

    @property
    def reranker(self):
        """Lazy-load the reranker model."""
        if self._reranker is None:
            self._reranker = get_reranker()
        return self._reranker

    def retrieve(self, query: str) -> List[RetrievalResult]:
        """Execute the full retrieval pipeline.

        Steps:
        1. MMR search: fetch_k diverse candidates from ChromaDB
        2. Reranking: score each candidate with cross-encoder
        3. Selection: return top retrieval_k by reranker score

        Args:
            query: User query string.

        Returns:
            List of RetrievalResult, sorted by relevance (best first).
        """
        logger.info(f"Retrieving for query: {query[:100]}...")

        # Step 1: MMR retrieval from ChromaDB
        mmr_docs = self.vectorstore.max_marginal_relevance_search(
            query=query,
            k=self.fetch_k,
            fetch_k=self.fetch_k * 2,
            lambda_mult=self.mmr_lambda,
        )

        logger.info(f"MMR returned {len(mmr_docs)} candidates")

        if not mmr_docs:
            return []

        # Step 2: Cross-encoder reranking
        if self.reranker:
            reranked = self._rerank(query, mmr_docs)
        else:
            logger.warning("Reranker unavailable, using MMR scores only")
            reranked = self._fallback_rank(query, mmr_docs)

        # Step 3: Return top-k
        return reranked[:self.retrieval_k]

    def _rerank(self, query: str, docs: List[Document]) -> List[RetrievalResult]:
        """Score documents with cross-encoder reranker.

        Args:
            query: Original user query.
            docs: Candidate documents from MMR.

        Returns:
            Documents sorted by reranker score (descending).
        """
        pairs = [(query, doc.page_content) for doc in docs]

        try:
            scores = self.reranker.predict(pairs)
            logger.info(f"Reranker scored {len(scores)} candidates")
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return self._fallback_rank(query, docs)

        scored_results = []
        for doc, score in zip(docs, scores):
            similarity = doc.metadata.get("score", 0.0)

            safe_meta = {
                k: v for k, v in doc.metadata.items()
                if k in DocumentMetadata.model_fields
            }
            result = RetrievalResult(
                content=doc.page_content,
                metadata=DocumentMetadata(**safe_meta),
                similarity_score=float(similarity),
                reranker_score=float(score),
            )
            scored_results.append(result)

        scored_results.sort(key=lambda x: x.reranker_score, reverse=True)
        return scored_results

    def _fallback_rank(self, query: str, docs: List[Document]) -> List[RetrievalResult]:
        """Fallback ranking when reranker is unavailable.

        Uses placeholder scores and preserves MMR order.
        """
        results = []
        for i, doc in enumerate(docs):
            safe_meta = {
                k: v for k, v in doc.metadata.items()
                if k in DocumentMetadata.model_fields
            }
            result = RetrievalResult(
                content=doc.page_content,
                metadata=DocumentMetadata(**safe_meta),
                similarity_score=1.0 - (i * 0.05),
                reranker_score=None,
            )
            results.append(result)
        return results
