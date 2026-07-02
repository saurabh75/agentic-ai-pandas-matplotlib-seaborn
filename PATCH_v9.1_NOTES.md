# Patch v9.1 — Clear DB + New Chat Fix

## Bugs fixed
1. **Clear DB did nothing on first click.** The old handler required checking a
   `Confirm clear?` checkbox that only appeared *after* the button was clicked,
   so a single click looked like a no-op. Replaced with an inline
   confirm/cancel dialog that clears in one confirmed click.
2. **Clear DB left stale data behind.** Row-level `collection.delete(ids=…)`
   can leave orphan HNSW segments and cached embeddings. `clear_database` now
   drops the collection, recreates it empty, then `rmtree`s
   `chroma_db/`, `documents/`, `data_store/`, `charts/`.
3. **New Chat only cleared LLM memory.** Screenshot showed
   `Session cleared: streamlit_…` but Documents / Chunks unchanged because the
   handler never touched the vector store or file folders. Now `New Chat`
   calls the same hard-reset as `Clear DB` (per your v9 preference:
   *clear files + chat*).
4. **Streamlit `@st.cache_resource` retained the old Chroma client** even
   after `clear_vector_store_cache()`. Both `cached_vector_store.clear()` and
   `cached_pipeline.clear()` are now called before `rmtree` so Windows can
   release the sqlite handles.