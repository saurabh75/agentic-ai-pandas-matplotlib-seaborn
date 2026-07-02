# Patch Notes — 6 GB GPU build (GTX 1660 / i5-9400 / 24 GB RAM)

## What changed

### 1. `.env` (overwritten)
- **LLM**: `llama3.1:8b` → `llama3.2:3b-instruct-q4_K_M` (fits in 6 GB VRAM with KV cache).
- **Reranker on CPU**: `RERANKER_DEVICE=auto` → `cpu`. Frees VRAM for the LLM; BGE-base is fast on a 6-core CPU.
- **Context shrunk**: `CHUNK_SIZE 1000 → 512`, `CHUNK_OVERLAP 200 → 100`, `FETCH_K 20 → 10`, `RETRIEVAL_K 5 → 3`. Smaller prompts = lower VRAM, faster turn.
- **Added Ollama tuning vars**: `OLLAMA_NUM_CTX=3072`, `OLLAMA_NUM_GPU=999`, `OLLAMA_NUM_PREDICT=512`, `OLLAMA_TEMPERATURE=0.2`, `MAX_CONTEXT_TOKENS=2048`.
  These only take effect if `src/services/generation_engine.py` forwards them to Ollama's `options` dict. If it doesn't yet, add:
  ```python
  options={
      "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", 3072)),
      "num_gpu": int(os.getenv("OLLAMA_NUM_GPU", 999)),
      "num_predict": int(os.getenv("OLLAMA_NUM_PREDICT", 512)),
      "temperature": float(os.getenv("OLLAMA_TEMPERATURE", 0.2)),
  }
  ```
- **Bigger timeout**: `OLLAMA_TIMEOUT 120 → 180` for first-token latency on CPU spill.

### 2. `app.py` — double-generation fix
The original `if prompt := st.chat_input(...)` block did:
1. `pipeline.stream_answer(...)` — full LLM generation #1
2. `pipeline.run(...)` — full LLM generation #2 (for citations/metrics)

That's ~2× latency and ~2× VRAM pressure per question. Patched to call `pipeline.run()` once and pseudo-stream the result into the UI placeholder.

## Files
- `.env` — overwritten with 6 GB-friendly defaults
- `app.py` — single-pass generation
- Original code otherwise untouched

## First-run checklist
```bash
ollama pull llama3.2:3b-instruct-q4_K_M
ollama pull nomic-embed-text
pip install -r requirements.txt   # torch will install CPU build if no CUDA torch is pre-installed
streamlit run app.py
```

If you ever want a tiny bit more quality and can spare VRAM, try
`qwen2.5:3b-instruct-q4_K_M` (often better at instruction following than llama3.2:3b).

---

## Patch v2 — ingestion stability

**Why:** `nomic-embed-text` was crashing Ollama mid-run because we sent 9000+ chunks in a single `add_documents()` call. Ollama's worker died and the next call hit a stale port (`connectex: actively refused`).

### Changes

1. **`.env`**
   - `CHUNK_SIZE=1000` (was 512) → ~5× fewer chunks per document
   - `CHUNK_OVERLAP=150` (was 100)
   - Added `EMBED_BATCH_SIZE=16`

2. **`src/services/document_processor.py`**
   - Indexing now sends chunks to Chroma/Ollama in batches of `EMBED_BATCH_SIZE`.
   - Each batch has 3-attempt exponential backoff retry on transient connection errors (refused/reset/timeout/EOF/5xx).
   - Progress log every 10 batches.

### Before ingesting

```powershell
ollama pull nomic-embed-text
$env:OLLAMA_NUM_PARALLEL=1
$env:OLLAMA_MAX_LOADED_MODELS=1
$env:OLLAMA_KEEP_ALIVE="10m"
ollama serve
```

Then in a second terminal:

```powershell
python scripts/ingest.py --dir "C:\Users\USER\Documents\PythonProjects\docs"
```

Do **not** run `streamlit run app.py` at the same time as ingestion — they fight for VRAM.

---

## Patch v3 — Chart generation (matplotlib + seaborn)

### What changed
- **New** `src/services/chart_router.py` — keyword + LLM JSON-mode intent detection. Returns a `ChartSpec` (chart_type, x, y, hue, agg, title) or `None`.
- **New** `src/services/data_extractor.py` — pulls a `pandas.DataFrame` out of the retrieved context. Order of attempts: CSV in chunk → markdown table → LLM-to-JSON fallback. Capped at 5000 rows.
- **New** `src/services/chart_renderer.py` — headless matplotlib (`Agg` backend) + seaborn theme. Supports `bar`, `line`, `scatter`, `hist`, `box`, `heatmap`, `pie`. Saves to `./charts/chart_<hash>.png`.
- **Modified** `src/models/agent.py` — `AgenticRAGResponse` now carries optional `chart_path` and `chart_caption`.
- **Modified** `src/agent/pipeline.py` — after Stage 5 (Generation), if the query has chart intent, runs the chart pipeline and attaches the PNG path. Adds an extra `AgentStep` named "Chart" so it shows up in the reasoning trace.
- **Modified** `app.py` — renders `st.image(response.chart_path, caption=...)` below the answer.
- **`.env`** — added `CHARTS_ENABLED=true`, `CHARTS_DIR=charts`, `CHART_MAX_ROWS=5000`.
- **`requirements.txt`** — added `matplotlib>=3.8`, `seaborn>=0.13`, `pandas>=2.0`.

### Install the new deps
```
pip install matplotlib seaborn pandas
```

### How it behaves
- The router only triggers on visual-intent keywords (chart, plot, trend, compare, distribution, by year/month, etc.) — normal Q&A is unaffected, no extra LLM calls.
- When intent is detected, it uses the already-loaded Ollama LLM (no extra VRAM cost) to: (a) extract a tabular JSON if context is prose, and (b) pick a chart type + columns.
- All rendering is CPU-only (Agg backend) — safe alongside your 6 GB GPU LLM.
- Failures are silent and graceful: the answer still renders, just without a chart, and a warning is logged.

### Disable
Set `CHARTS_ENABLED=false` in `.env`.

### Examples that should produce a chart
- "Plot revenue by quarter from the sales report"
- "Show the distribution of response times"
- "Compare error rates across services"
- "Trend of active users over the last 12 months"
