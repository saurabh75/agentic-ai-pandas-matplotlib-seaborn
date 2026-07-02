"""
Stage 3 — Knowledge & Memory Layer / Hybrid Retriever

Implements three retrieval modes:
  VECTOR  — pure dense embedding similarity (Chroma)
  BM25    — sparse keyword matching (rank_bm25 over stored corpus)
  HYBRID  — reciprocal rank fusion of BM25 + vector scores
  MMR     — maximal marginal relevance for diversity

The retriever is the "Tool / Source Selection" node in the agent loop and
feeds directly into the Retrieval Quality Pipeline (Stage 4).
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, List, Optional

from langchain_chroma import Chroma

from src.config import FETCH_K, RETRIEVAL_K, MMR_LAMBDA, RERANKER_MODEL, RERANKER_DEVICE
from src.models.agent import RetrievalCandidate, RetrievalStrategy
from src.services.vector_store import get_vector_store
from src.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# BM25 Index (built on demand from ChromaDB corpus)
# =============================================================================


class BM25Index:
    """
    Sparse keyword index built from the ChromaDB text corpus.

    Rebuilt lazily on first use. Call `invalidate()` after new documents
    are ingested so the index reflects the updated corpus.
    """

    def __init__(self) -> None:
        self._corpus: List[str] = []
        self._metadatas: List[dict] = []
        self._index = None
        self._valid = False

    def build(self, vectorstore: Chroma) -> None:
        """Fetch all documents from ChromaDB and build the BM25 index."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.error("rank_bm25 not installed. Run: pip install rank-bm25")
            return

        try:
            result = vectorstore._collection.get(include=["documents", "metadatas"])
            self._corpus = result.get("documents") or []
            self._metadatas = result.get("metadatas") or []

            if not self._corpus:
                logger.warning("BM25: corpus is empty — no documents indexed yet")
                return

            tokenized = [doc.lower().split() for doc in self._corpus]
            self._index = BM25Okapi(tokenized)
            self._valid = True
            logger.info(f"BM25 index built with {len(self._corpus)} chunks")
        except Exception as e:
            logger.error(f"BM25 index build failed: {e}")

    def search(self, query: str, k: int) -> List[RetrievalCandidate]:
        if not self._valid or self._index is None:
            return []
        tokens = query.lower().split()
        scores = self._index.get_scores(tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        candidates = []
        for idx in top_idx:
            meta = self._metadatas[idx] if idx < len(self._metadatas) else {}
            source_id = (
                f"{meta.get('document_hash', 'unknown')}_{meta.get('chunk_index', idx)}"
            )
            candidates.append(
                RetrievalCandidate(
                    content=self._corpus[idx],
                    metadata=meta,
                    bm25_score=float(scores[idx]),
                    source_id=source_id,
                )
            )
        return candidates

    def invalidate(self) -> None:
        self._valid = False
        self._index = None
        logger.info("BM25 index invalidated")


# Singleton BM25 index shared across requests
_bm25_index = BM25Index()


def get_bm25_index() -> BM25Index:
    return _bm25_index


# =============================================================================
# Reranker (shared with original retriever via lru_cache)
# =============================================================================


@lru_cache(maxsize=1)
def _load_reranker():
    try:
        import torch
        from sentence_transformers import CrossEncoder

        device = RERANKER_DEVICE
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading reranker: {RERANKER_MODEL} on {device}")
        reranker = CrossEncoder(RERANKER_MODEL, device=device, max_length=512)
        logger.info("Reranker loaded")
        return reranker
    except Exception as e:
        logger.warning(f"Reranker unavailable: {e}")
        return None


# =============================================================================
# Reciprocal Rank Fusion
# =============================================================================


def _reciprocal_rank_fusion(
    vector_hits: List[RetrievalCandidate],
    bm25_hits: List[RetrievalCandidate],
    k: int = 60,
) -> List[RetrievalCandidate]:
    """
    Fuse two ranked lists using Reciprocal Rank Fusion (RRF).

    RRF score = Σ 1 / (rank + k)

    This is parameter-robust: results that appear highly in both lists
    are strongly promoted regardless of raw score magnitude differences
    between sparse and dense retrievers.
    """
    scores: Dict[str, float] = {}
    candidates: Dict[str, RetrievalCandidate] = {}

    for rank, c in enumerate(vector_hits):
        scores[c.source_id] = scores.get(c.source_id, 0.0) + 1.0 / (rank + k)
        candidates[c.source_id] = c

    for rank, c in enumerate(bm25_hits):
        scores[c.source_id] = scores.get(c.source_id, 0.0) + 1.0 / (rank + k)
        if c.source_id not in candidates:
            candidates[c.source_id] = c
        else:
            # Merge BM25 score into existing candidate
            candidates[c.source_id].bm25_score = c.bm25_score

    for sid, rrf_score in scores.items():
        candidates[sid].hybrid_score = rrf_score

    return sorted(candidates.values(), key=lambda c: c.hybrid_score, reverse=True)


# =============================================================================
# Hybrid Retriever
# =============================================================================


class HybridRetriever:
    """
    Multi-strategy retriever supporting VECTOR, BM25, HYBRID, and MMR modes.

    Sits at the intersection of the Knowledge & Memory Layer and the
    Retrieval Quality Pipeline in the Agentic RAG architecture.
    """

    def __init__(
        self,
        vectorstore: Optional[Chroma] = None,
        fetch_k: int = FETCH_K,
        retrieval_k: int = RETRIEVAL_K,
        mmr_lambda: float = MMR_LAMBDA,
    ) -> None:
        self._vectorstore = vectorstore or get_vector_store()
        self._fetch_k = fetch_k
        self._retrieval_k = retrieval_k
        self._mmr_lambda = mmr_lambda
        self._reranker = None
        self._bm25 = get_bm25_index()

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = _load_reranker()
        return self._reranker

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
    ) -> List[RetrievalCandidate]:
        """
        Retrieve and rerank candidates using the specified strategy.

        Args:
            query: Search query string.
            strategy: Which retrieval mode to use.

        Returns:
            Reranked list of RetrievalCandidate (up to retrieval_k).
        """
        if strategy == RetrievalStrategy.VECTOR:
            candidates = self._vector_search(query, self._fetch_k)
        elif strategy == RetrievalStrategy.BM25:
            candidates = self._bm25_search(query, self._fetch_k)
        elif strategy == RetrievalStrategy.MMR:
            candidates = self._mmr_search(query, self._fetch_k)
        else:  # HYBRID (default)
            candidates = self._hybrid_search(query, self._fetch_k)

        # Cross-encoder reranking
        if self.reranker and candidates:
            candidates = self._rerank(query, candidates)

        return candidates[: self._retrieval_k]

    # ------------------------------------------------------------------
    # Retrieval modes
    # ------------------------------------------------------------------

    def _vector_search(self, query: str, k: int) -> List[RetrievalCandidate]:
        try:
            results = self._vectorstore.similarity_search_with_score(query, k=k)
            candidates = []
            for doc, score in results:
                meta = doc.metadata or {}
                source_id = f"{meta.get('document_hash', 'x')}_{meta.get('chunk_index', 0)}"
                candidates.append(
                    RetrievalCandidate(
                        content=doc.page_content,
                        metadata=meta,
                        vector_score=float(1 - score),  # distance → similarity
                        source_id=source_id,
                    )
                )
            return candidates
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def _bm25_search(self, query: str, k: int) -> List[RetrievalCandidate]:
        if not self._bm25._valid:
            self._bm25.build(self._vectorstore)
        return self._bm25.search(query, k)

    def _mmr_search(self, query: str, k: int) -> List[RetrievalCandidate]:
        try:
            docs = self._vectorstore.max_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=k * 2,
                lambda_mult=self._mmr_lambda,
            )
            candidates = []
            for i, doc in enumerate(docs):
                meta = doc.metadata or {}
                source_id = f"{meta.get('document_hash', 'x')}_{meta.get('chunk_index', i)}"
                candidates.append(
                    RetrievalCandidate(
                        content=doc.page_content,
                        metadata=meta,
                        vector_score=1.0 - (i * 0.04),  # approximate decay
                        source_id=source_id,
                    )
                )
            return candidates
        except Exception as e:
            logger.error(f"MMR search failed: {e}")
            return self._vector_search(query, k)

    def _hybrid_search(self, query: str, k: int) -> List[RetrievalCandidate]:
        vector_hits = self._vector_search(query, k)
        bm25_hits = self._bm25_search(query, k)
        if not vector_hits and not bm25_hits:
            return []
        if not vector_hits:
            return bm25_hits
        if not bm25_hits:
            return vector_hits
        return _reciprocal_rank_fusion(vector_hits, bm25_hits)

    # ------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------

    def _rerank(
        self, query: str, candidates: List[RetrievalCandidate]
    ) -> List[RetrievalCandidate]:
        pairs = [(query, c.content) for c in candidates]
        try:
            scores = self.reranker.predict(pairs)
            for c, s in zip(candidates, scores):
                c.reranker_score = float(s)
            candidates.sort(key=lambda c: c.reranker_score or -math.inf, reverse=True)
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
        return candidates

    def invalidate_bm25(self) -> None:
        """Call after new documents are ingested."""
        self._bm25.invalidate()
