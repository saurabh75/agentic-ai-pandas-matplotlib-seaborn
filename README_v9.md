# Patch v9 — Multi-User Isolation + New Chat Reset

Adds per-user isolation to the RAG. Every user has their own vector
collection, parquet store, file profiles, charts, uploads, and chat
history. Users are identified by a **reverse-proxy-injected header**
(Option D) — no login form inside Streamlit.

## Install

1. Copy `src/services/user_context.py`, `user_storage.py`,
   `user_vector_store.py` into your project.
2. Append `.env.additions` to `.env`.
3. Apply the three patch guides in `patches/`.
4. Deploy behind nginx+oauth2-proxy or Cloudflare Access — see
   `patches/reverse_proxy_examples.txt`.

## What changes

| Layer | Before | After |
|---|---|---|
| Vector collection | `rag_chunks` | `rag_chunks_{safe_id}` per user |
| Parquet / profiles | `./data_store/…` | `./data_store/{safe_id}/…` |
| Charts | `./charts/…` | `./charts/{safe_id}/…` |
| Uploads | `./uploads/…` | `./uploads/{safe_id}/…` |
| Multi-file KB | singleton | `UserKBRegistry.kb_for(user)` |
| Chat state | `st.session_state.messages` | `st.session_state["u:{safe_id}:messages"]` |

Defense-in-depth: even though each user has an isolated Chroma
collection, every retrieval also filters `where={"user_id": user.user_id}`.

## New Chat button

Sidebar button → confirmation dialog → `pipeline.wipe_user(user)`:
- deletes user's Chroma collection
- `rmtree` on `data_store/{safe_id}`, `charts/{safe_id}`, `uploads/{safe_id}`
- drops the user's `MultiFileKB` and every `u:{safe_id}:*` session key

Other users' data is untouched.

## Security guarantees

- `user_id` regex-restricted to `[A-Za-z0-9_.\-@]{3,64}` before use
- `safe_id` = sanitized + SHA1[:8] suffix → path traversal impossible
- `AUTH_ENABLED=false` falls back to single-user "default" (backward compat)
- `AUTH_ALLOW_ANON=true` is TESTING only — do not enable in production
- Header list is configurable via `AUTH_USER_HEADER` (comma-separated)

## Backward compatibility

Set `AUTH_ENABLED=false` and the app behaves exactly like v8 with all
data under `data_store/default/`.
