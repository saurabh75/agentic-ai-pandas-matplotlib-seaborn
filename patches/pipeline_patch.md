# pipeline.py — integration patch (v8)

Apply these edits to `src/agent/pipeline.py`. Nothing existing is removed; only new hooks are added.

## 1) Imports (top of file)

```python
from src.services.persona_manager import detect_persona
from src.services.response_formatter import build_system_suffix, fallback_no_info, NO_INFO
from src.services.multi_file_kb import MultiFileKB
from src.agent.prompts import get_persona_prompt
```

## 2) Pipeline class — add to `__init__`

```python
self.kb = MultiFileKB()          # session-scoped multi-file registry
self._active_persona = None
```

Expose a helper the UI can call after each upload:

```python
def register_file(self, filename: str, profile: dict) -> None:
    self.kb.register(filename, profile)
```

## 3) Persona resolution — call at the top of `answer()` / `stream_answer()`

```python
files_in_scope = self.kb.list_files() or [ctx_file] if (ctx_file := kwargs.get("file")) else self.kb.list_files()
persona_info = detect_persona(files_in_scope, domains=self.kb.domains())
self._active_persona = persona_info

persona_prompt = get_persona_prompt(persona_info["persona"], persona_info["domain"])
system_prompt = persona_prompt + build_system_suffix(persona_info, files_in_scope)
```

Pass `system_prompt` to the Ollama call in place of (or prepended to) the existing system message. Keep all other agentic logic (retrieval, rerank, groundedness) unchanged.

## 4) Multi-file context assembly

Right after retrieval + rerank produces `chunks`:

```python
if self.kb.is_multi():
    context_text = self.kb.consolidated_context(chunks)
    overlap = self.kb.detect_overlap(chunks)
    if overlap:
        context_text += f"\n\n[Cross-file overlapping topics: {', '.join(overlap)}]"
else:
    context_text = "\n\n".join(c["content"] for c in chunks)
```

## 5) Hallucination fallback

Before generation, if `not chunks` OR max rerank score < threshold:

```python
answer_md = fallback_no_info(persona_info, files_in_scope)
yield answer_md
yield {"persona": persona_info, "chunks": [], "charts": [], "no_info": True}
return
```

## 6) Final sentinel — extend the existing metadata dict

```python
yield {
    "persona": persona_info,
    "active_files": files_in_scope,
    "chunks_used": [{"source": c["metadata"].get("source"),
                     "chunk_id": c["metadata"].get("chunk_id"),
                     "score": c.get("score")} for c in chunks],
    "charts": charts,                # existing
    "analysis": analysis_md,         # existing (v5+)
}
```

## 7) Analysis / EDA short-circuit (already added in v5-v7)

Prepend the Data Scientist persona header to the analysis markdown so the persona rule applies even on zero-LLM paths:

```python
analysis_md = ("### 1. Active Expert Persona\nSenior Data Scientist (20 yrs) — "
               "tabular file detected.\n\n" + analysis_md)
```
