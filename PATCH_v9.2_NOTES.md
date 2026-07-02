# Patch v9.2 — Full EDA Workflow Execution

## The bug (from screenshot)
`do professional EDA workflow` and `do all` fell into the RAG loop (11 agent
iterations, hybrid retrieval) and produced generic **descriptions** of what
EDA is — the LLM was reasoning over text chunks instead of running any
computation on your `annual-enterprise-survey-2025-financial-year-provisional.csv`.

## Fix
Intercept those phrases in the **AnalysisRouter** and run a real end-to-end
EDA on the registered DataFrame — no LLM, no retrieval, ~1–3 s on a 6 GB
machine.

### Trigger phrases (high-priority, checked BEFORE other intents)
`do all` · `do it all` · `do everything` · `run all` · `full/complete/professional/entire EDA` ·
`eda workflow` · `eda report` · `all steps` · `exploratory data analysis` ·
`profile the dataset` · `automated EDA` · bare `eda`.

### The 11 steps (executed with pandas / scipy / seaborn)
1. Understand the Data — rows, cols, dtypes, memory, head(5)
2. Shape & Data Types — dtype + non-null + unique per column
3. Missing Values — count/% table + missingness heatmap
4. Duplicates — count + sample duplicate rows
5. Numerical Analysis — describe + skew + kurtosis + histogram grid (KDE)
6. Categorical Analysis — value_counts + top-category bar grid
7. Bivariate Analysis — numeric × categorical boxplots + top means
8. Correlation — Pearson matrix + heatmap + top |r| pairs
9. Outliers — IQR bounds per numeric col + boxplot grid
10. Feature Relationships — seaborn pairplot (first 4 numeric)
11. Insights & Recommendations — automated flags for high-null cols,
    duplicates, skew, high cardinality, multicollinearity

Every step produces markdown + PNG(s). All charts render inline in Streamlit.

## Files changed
- `src/services/eda_engine.py` — **new**, the 11-step engine.
- `src/services/analysis_router.py` — new `full_eda` intent (priority 1).
- `src/models/agent.py` — `extra_chart_paths` / `extra_chart_captions`.
- `src/agent/pipeline.py` — routes `full_eda` to the engine; returns extra chart list.
- `src/utils/chroma_utils.py` — from v9.1 (hard reset + wipe helper).
- `app.py` — from v9.1 (one-click Clear DB + hard-reset New Chat) plus
  new loop that renders every extra chart under the primary one.

## Install
Unzip over your project, overwriting the listed files. No new dependencies
(matplotlib, seaborn, scipy already in v5+ requirements). Restart Streamlit.

## Try it
1. Upload a CSV / XLSX (it goes to `data_store/` as Parquet).
2. Ask: **`do all`**, **`professional EDA workflow`**, **`profile the dataset`**.
3. You should see a report with 11 sections, correlation heatmap, histogram
   grid, boxplots, pairplot, and automated recommendations — in ~1–3 s
   with **0 iterations** on the strategy badge.
