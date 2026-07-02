# pipeline.py — v9 integration patch (multi-user)

Thread `user: UserContext` through every entry point. The pipeline becomes
stateless w.r.t. users — user state lives in `UserKBRegistry` and
`UserVectorStore`.

## 1) Imports

```python
from src.services.user_context import UserContext, get_current_user
from src.services.user_storage import data_dir, profiles_dir, charts_dir
from src.services.user_vector_store import UserVectorStore, UserKBRegistry
```

## 2) Pipeline `__init__` — replace singletons

```python
self.vector = UserVectorStore(self.chroma_client)   # was: self.collection = ...
self.kbs    = UserKBRegistry()                      # was: self.kb = MultiFileKB()
```

## 3) Every public method takes `user: UserContext`

```python
def register_file(self, user, filename, profile):
    self.kbs.kb_for(user).register(filename, profile)

def stream_answer(self, user: UserContext, query: str):
    kb = self.kbs.kb_for(user)
    files_in_scope = kb.list_files()
    persona_info = detect_persona(files_in_scope, domains=kb.domains())
    ...
    # retrieval is user-scoped by the vector store:
    results = self.vector.query(user, query_embeddings=[q_emb], n_results=8)
    ...
```

## 4) Analysis / EDA paths — use per-user parquet dir

Everywhere v5/v7 wrote to `./data_store/*.parquet`, replace with:

```python
parquet_path = data_dir(user) / f"{fname}.parquet"
```

Chart output:

```python
out = charts_dir(user) / f"{chart_id}.png"
```

## 5) New Chat / Wipe

```python
def wipe_user(self, user: UserContext) -> dict:
    from src.services.user_storage import wipe_user as fs_wipe
    self.vector.wipe(user)
    self.kbs.wipe(user)
    return fs_wipe(user)
```
