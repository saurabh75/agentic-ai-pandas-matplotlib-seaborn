# app.py — Streamlit UI patch (v8)

## After successful upload

```python
profile = document_processor.ingest(uploaded_file)  # now returns profile
st.session_state.setdefault("profiles", {})[profile["file_name"]] = profile
```

## Sidebar — File Intelligence panel

```python
with st.sidebar:
    st.header("📁 File Intelligence")
    for fname, p in st.session_state.get("profiles", {}).items():
        with st.expander(f"📄 {fname}", expanded=True):
            st.caption(f"{p['file_type'].upper()} • {p['size_human']} • domain: **{p.get('domain','?')}**")
            if "rows" in p:
                st.write(f"Rows: {p['rows']} | Cols: {p['columns']}")
                st.write(f"Missing: {p['missing_total']} | Duplicates: {p['duplicates']}")
            elif "pages_est" in p:
                st.write(f"~{p['pages_est']} pages • {p['word_count']} words")
            st.markdown("**Summary:** " + p.get("summary", "-"))
            if p.get("main_topics"):
                st.markdown("**Topics:** " + ", ".join(map(str, p["main_topics"][:6])))
            st.markdown("**Suggested questions:**")
            for q in p.get("suggested_questions", [])[:5]:
                if st.button(q, key=f"sq_{fname}_{q}"):
                    st.session_state["queued_question"] = q
                    st.rerun()
```

## Header — active persona badge

```python
persona = st.session_state.get("active_persona") or {"label": "Awaiting file..."}
st.markdown(f"### 🎓 Active Persona: `{persona['label']}`")
```

## Chat rendering — expose chunks used

After streaming completes and the sentinel dict arrives:

```python
st.session_state["active_persona"] = meta["persona"]
with st.expander(f"🔍 Sources ({len(meta.get('chunks_used', []))} chunks)"):
    for c in meta.get("chunks_used", []):
        st.caption(f"• {c['source']} · chunk {c['chunk_id']} · score {c['score']:.3f}")
```

Existing chart rendering and analysis-markdown branches are unchanged.
