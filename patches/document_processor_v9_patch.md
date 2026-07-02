# document_processor.py — v9 patch

Every ingest call now receives `user: UserContext` and writes to that user's
scoped directories only.

```python
from src.services.user_storage import uploads_dir, data_dir, profiles_dir
from src.services.file_profiler import profile_file

def ingest(self, user, uploaded_file):
    saved = uploads_dir(user) / uploaded_file.name
    with open(saved, "wb") as f:
        # existing 1MB chunked write (v5.2)
        for chunk in iter(lambda: uploaded_file.read(1024*1024), b""):
            f.write(chunk)

    # ---- existing chunking / embedding, unchanged ----
    ids, docs, embs, metas = self._chunk_and_embed(saved)

    # user-scoped vector write
    self.pipeline.vector.add(user, ids=ids, documents=docs,
                             embeddings=embs, metadatas=metas)

    # profile (v8) → user-scoped cache dir
    profile = profile_file(
        path=str(saved),
        sample_text=self._sample_text,
        ollama_client=self.ollama,
        model=os.getenv("PERSONA_CLASSIFIER_MODEL"),
        cache_dir=profiles_dir(user),   # <-- pass explicit dir
    )
    self.pipeline.register_file(user, saved.name, profile)
    return profile
```

If your v8 `file_profiler.profile_file` does not accept `cache_dir`, add a
one-line kwarg forwarding to `_cache_path()`:

```python
def profile_file(..., cache_dir: Path | None = None):
    cache = (cache_dir / f"{Path(path).stem}.profile.json") if cache_dir \
            else _cache_path(path)
```
