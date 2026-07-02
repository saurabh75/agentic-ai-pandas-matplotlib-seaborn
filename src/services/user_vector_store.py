"""Per-user vector store + multi-file KB registry.

Wraps whatever ChromaDB client you already use so the rest of the app
never has to think about user scoping.
"""
from __future__ import annotations
from typing import Any
from src.services.user_context import UserContext
from src.services.user_storage import vector_collection_name
from src.services.multi_file_kb import MultiFileKB   # from v8


class UserVectorStore:
    """Per-user Chroma collection with defense-in-depth metadata filter."""

    def __init__(self, chroma_client) -> None:
        self.client = chroma_client
        self._cache: dict[str, Any] = {}

    def collection_for(self, user: UserContext):
        name = vector_collection_name(user)
        if name not in self._cache:
            self._cache[name] = self.client.get_or_create_collection(name=name)
        return self._cache[name]

    def add(self, user: UserContext, ids, documents, embeddings, metadatas):
        # Force user_id into every metadata row
        metadatas = [{**(m or {}), "user_id": user.user_id} for m in metadatas]
        self.collection_for(user).add(
            ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas
        )

    def query(self, user: UserContext, query_embeddings, n_results=8, where=None):
        merged = {"user_id": user.user_id}
        if where:
            merged = {"$and": [merged, where]}
        return self.collection_for(user).query(
            query_embeddings=query_embeddings, n_results=n_results, where=merged
        )

    def wipe(self, user: UserContext) -> None:
        name = vector_collection_name(user)
        try:
            self.client.delete_collection(name)
        except Exception:
            pass
        self._cache.pop(name, None)


class UserKBRegistry:
    """One MultiFileKB per user, lazily created."""
    def __init__(self) -> None:
        self._kbs: dict[str, MultiFileKB] = {}

    def kb_for(self, user: UserContext) -> MultiFileKB:
        if user.safe_id not in self._kbs:
            self._kbs[user.safe_id] = MultiFileKB()
        return self._kbs[user.safe_id]

    def wipe(self, user: UserContext) -> None:
        self._kbs.pop(user.safe_id, None)
