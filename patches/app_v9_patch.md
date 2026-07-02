# app.py — v9 patch (multi-user + New Chat)

## 1) Resolve user at the top of the script

```python
from src.services.user_context import get_current_user

try:
    user = get_current_user()
except PermissionError as e:
    st.error("🔒 Authentication required")
    st.caption(str(e))
    st.stop()
```

## 2) Sidebar — user badge + New Chat button

```python
with st.sidebar:
    st.markdown(f"👤 **{user.user_id}**  \n`{user.source}`")
    st.divider()

    if st.button("🆕 New Chat", type="primary", use_container_width=True):
        st.session_state["_confirm_new_chat"] = True

    if st.session_state.get("_confirm_new_chat"):
        st.warning("This will delete ALL your uploaded files, embeddings, and chat history. Continue?")
        col1, col2 = st.columns(2)
        if col1.button("✅ Yes, reset everything", use_container_width=True):
            removed = pipeline.wipe_user(user)
            # per-user session keys
            for k in list(st.session_state.keys()):
                if k.startswith(f"u:{user.safe_id}:"):
                    del st.session_state[k]
            st.session_state["_confirm_new_chat"] = False
            st.success(f"Cleared: {list(removed.keys())}")
            st.rerun()
        if col2.button("❌ Cancel", use_container_width=True):
            st.session_state["_confirm_new_chat"] = False
            st.rerun()
```

## 3) Namespace ALL session_state by user

Every read/write of `st.session_state` becomes:

```python
def uk(key: str) -> str:                       # user-scoped key helper
    return f"u:{user.safe_id}:{key}"

messages    = st.session_state.setdefault(uk("messages"), [])
profiles    = st.session_state.setdefault(uk("profiles"), {})
persona     = st.session_state.get(uk("active_persona"))
```

This prevents User A ever seeing User B's chat even in a shared browser tab
(shouldn't happen with proxy headers, but defense-in-depth).

## 4) Pass user into every pipeline call

```python
pipeline.ingest(user, uploaded_file)
pipeline.stream_answer(user, query)
pipeline.register_file(user, fname, profile)
```

## 5) File Intelligence sidebar (v8) — read user-scoped profiles

```python
for fname, p in st.session_state.get(uk("profiles"), {}).items():
    ...
```
