# 🤖 Agentic RAG Agent — 2026 Edition

> *"Not just retrieve → generate; a controlled loop of planning, retrieval, verification, and response."*

Implements the full **Agentic RAG Architecture** as a local, private, cost-free Streamlit app powered by Ollama.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  1. INPUT & ORCHESTRATION                               │
│     User Query → Intent Analysis → Planner → Policy    │
├─────────────────────────────────────────────────────────┤
│  2. AGENT LOOP                              max 3 iters │
│     Query Rewrite → Strategy Select → Retrieval        │
│     → Gap Detection → [loop or proceed]                 │
├─────────────────────────────────────────────────────────┤
│  3. KNOWLEDGE & MEMORY LAYER                            │
│     Vector DB (Chroma) + BM25 + Session Memory         │
├─────────────────────────────────────────────────────────┤
│  4. RETRIEVAL QUALITY PIPELINE                          │
│     Reranker → Dedup+Filter → Freshness → Context Bld  │
├─────────────────────────────────────────────────────────┤
│  5. REASONING & GENERATION                              │
│     LLM → Draft → Citation Builder → Groundedness Check│
│     → Final Answer (with citations + confidence %)     │
├─────────────────────────────────────────────────────────┤
│  6. EVALUATION & FEEDBACK                               │
│     Latency | Confidence | Groundedness | Iterations   │
└─────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) running locally

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Pull Ollama models
```bash
ollama pull llama3.1:8b          # LLM
ollama pull nomic-embed-text     # Embeddings
```

### 3. Run
```bash
streamlit run app.py
# or
bash start.sh
```

---

## Key Features

| Feature | Detail |
|---|---|
| **Agentic Loop** | Up to 3 retrieval iterations with automatic gap detection |
| **Query Rewriting** | LLM rewrites queries to target missing evidence |
| **Hybrid Retrieval** | BM25 + Dense Vector + Reciprocal Rank Fusion |
| **Strategy Selection** | Auto-selects VECTOR/BM25/HYBRID/MMR by intent |
| **Cross-encoder Reranking** | BGE reranker for precision |
| **Groundedness Scoring** | Verifies every answer against context before display |
| **Citation Builder** | Extracts inline [Source N] citations automatically |
| **Streaming Responses** | Real-time token streaming in UI |
| **Evaluation Dashboard** | Per-session latency, confidence, groundedness, iterations |
| **Policy Check** | Guards against prompt injection and context overload |

## Supported File Types
`.pdf` · `.docx` · `.xlsx` · `.xls` · `.pptx` · `.ppt` · `.txt` · `.md` · `.csv`

## Configuration
All settings in `.env` — see `.env` for full reference.

## CLI Usage
```bash
# Ingest documents
python scripts/ingest.py --dir ./documents/
python scripts/ingest.py --file report.pdf

# Query
python scripts/query.py "What are the key findings?"
python scripts/query.py --session s1 "Follow-up question"
```
