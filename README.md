# 🧠 Local Agentic RAG — Senior Expert Edition

A **fully local, multi-user, agentic Retrieval-Augmented Generation** system that behaves like a senior domain expert. Upload documents or datasets, ask questions in natural language, and get **grounded, cited, structured answers** — plus automated EDA, statistical testing, chart generation, and type coercion — all running on your own machine with Ollama.

> 🖥️ Optimized for modest hardware (tested on **i5-9400 / 24 GB RAM / GTX 1660 6 GB**) using quantized Llama 3.2 3B.

---

## ✨ Features

### 🤖 Agentic Intelligence
- **Dynamic Expert Personas** — auto-switches between *Senior Data Scientist* (CSV/XLSX), *Senior Domain Expert* (PDF/DOCX/MD), and *Multi-Domain Analyst* (mixed corpora).
- **20-year senior-expert framing** with strict hallucination guard — every answer opens with the active persona and cites sources.
- **Structured 7-section responses**: Persona → Executive Summary → Analysis → Findings → Recommendations → Sources → Follow-ups.
- **Cross-document reasoning** across multiple uploads simultaneously.

### 📊 Data Analysis Mode
- **Deterministic Pandas Executor** — bypasses the LLM for computable questions (nulls, duplicates, describe, groupby, outliers).
- **One-shot Full EDA** — trigger phrases like *"do complete EDA"* run all 11 steps instantly:
  1. Understand Data → 2. Shape & Dtypes → 3. Missing → 4. Duplicates → 5. Numerical → 6. Categorical → 7. Bivariate → 8. Correlation → 9. Outliers → 10. Feature Relationships → 11. Insights.
- **Analysis Tiers** — Univariate, Bivariate, Multivariate on demand.
- **~45 Statistical Tests** across 10 families: Normality (Shapiro, D'Agostino, KS), Location (t-test, Welch, Mann-Whitney, Wilcoxon), ANOVA + Tukey HSD, Kruskal-Wallis, Chi-Square, Fisher, Cramér's V, Levene, Bartlett, Pearson/Spearman/Kendall, VIF, ADF, Ljung-Box, and more.
- **EDA Memory** — follow-ups like *"write conclusion"* cite the actual results of the prior EDA (no re-hallucination).

### 📈 Visualization
- Automatic **Matplotlib + Seaborn** charts (histograms, boxplot grids, heatmaps, count plots, regression plots).
- OLS summary rendering for regression tasks.

### 🔧 Utilities
- **Type Coercion** — natural-language dtype conversion: *"convert age to string"*, *"bmi to int"*, etc.
- **Token Streaming** — real-time output via Ollama `stream=True`.
- **⏹️ Stop Button** — cancel any generation mid-stream; partial output is preserved.
- **Multi-User Isolation** — per-user Chroma collections + storage directories.
- **New Chat = Hard Reset** — wipes vectors, files, and profile cards for the current user.
- **Report Export** — one-click ZIP of the full session (report + charts + artifacts).

### 🛡️ Reliability
- Chunked file uploads (1 MB) — no more crashes on large files.
- `filetype` (pure-Python) instead of `python-magic` — no libmagic headaches on Windows.
- Batched ingestion with retries against Ollama socket resets.

---

## 🏗️ Architecture

```text
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI (app.py)                │
│  Upload · Chat · Charts · Stop · New Chat · Export ZIP  │
└──────────────────────────┬──────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │   Agent Pipeline        │  ← persona resolver
              │   src/agent/pipeline.py │
              └──┬──────┬──────┬────────┘
                 │      │      │
       ┌─────────▼─┐ ┌──▼────┐ ┌▼──────────────┐
       │  Router   │ │ RAG   │ │ Analysis Mode │
       │ (intent)  │ │ Chain │ │ pandas/scipy  │
       └─────┬─────┘ └───┬───┘ └───┬───────────┘
             │           │         │
       ┌─────▼───────────▼─────────▼──────┐
       │      Generation Engine           │
       │  Ollama (Llama 3.2 3B q4_K_M)    │
       │  streaming + system_override     │
       └────────────┬─────────────────────┘
                    │
       ┌────────────▼────────────┐
       │  ChromaDB (per user)    │  +  Parquet store (tabular)
       └─────────────────────────┘
```

---

## 📦 Requirements

- **Python** 3.10+
- **Ollama** with `llama3.2:3b-instruct-q4_K_M` pulled locally
- **RAM** ≥ 16 GB recommended (24 GB ideal)
- **GPU** optional; 6 GB VRAM tuned config ships by default

Install:

```bash
git clone https://github.com/<you>/local-agentic-rag.git
cd local-agentic-rag
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
ollama pull llama3.2:3b-instruct-q4_K_M
```

---

## ⚙️ Configuration (`.env`)

```env
OLLAMA_MODEL=llama3.2:3b-instruct-q4_K_M
OLLAMA_NUM_CTX=2048
RERANKER_DEVICE=cpu
CHART_ROUTER_USE_LLM=false
FAST_MODE=true
```

---

## 🚀 Run

```bash
streamlit run app.py
```

Open <http://localhost:8501>.

1. Sign in (or use the default user).
2. Drag in CSV / XLSX / PDF / DOCX / MD.
3. Ask anything — *"do complete EDA"*, *"find outliers in bmi"*, *"run ANOVA on charges by region"*, *"convert age to string"*, *"summarize the contract"*.

---

## 🧪 Example Prompts

| Prompt | What it triggers |
|---|---|
| `do complete EDA` | 11-step deterministic report + charts |
| `find outliers in charges` | IQR + Z-score + boxplot |
| `run chi-square on smoker vs region` | scipy chi², Cramér's V |
| `convert bmi to int` | Safe dtype coercion |
| `write conclusion` | Grounded on prior EDA memory |
| `summarize section 4` | Persona-switched to Domain Expert |

---

## 🗂️ Project Structure

```text
local-agentic-rag/
├── app.py                          # Streamlit UI
├── requirements.txt
├── .env
├── src/
│   ├── agent/
│   │   ├── pipeline.py             # Persona resolver + orchestration
│   │   └── prompts.py              # Senior-expert system prompts
│   ├── services/
│   │   ├── generation_engine.py    # Ollama streaming
│   │   ├── eda_engine.py           # 11-step full EDA
│   │   ├── statistical_tests.py    # ~45 tests
│   │   ├── analysis_tiers.py       # Uni/Bi/Multivariate
│   │   ├── type_converter.py       # Any-to-any dtype
│   │   ├── pandas_executor.py      # Deterministic tabular ops
│   │   ├── chart_renderer.py       # Matplotlib/Seaborn
│   │   ├── persona_manager.py
│   │   ├── domain_classifier.py
│   │   ├── file_profiler.py
│   │   └── response_formatter.py
│   └── utils/
│       ├── chroma_utils.py         # Per-user isolation + hard reset
│       ├── cancel_token.py         # Stop button
│       └── export_report.py        # Session ZIP
├── uploads/       # per-user
├── data_store/    # Parquet cache
└── chroma_db/     # per-user vector collections
```

---

## 🛣️ Roadmap

- [ ] Whisper-based voice input
- [ ] Time-series module (Prophet / ARIMA)
- [ ] SQL connector for direct DB analysis
- [ ] Docker Compose (Ollama + app one-shot)

---

## 📜 License

MIT — do whatever you want, just don't blame me if the LLM disagrees with your data.

---

## 🙏 Acknowledgments

Built on [Ollama](https://ollama.com), [ChromaDB](https://www.trychroma.com), [LangChain](https://www.langchain.com), [Streamlit](https://streamlit.io), [scikit-learn](https://scikit-learn.org), [statsmodels](https://www.statsmodels.org), and [scipy](https://scipy.org).
