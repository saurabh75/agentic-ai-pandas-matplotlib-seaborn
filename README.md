# Local RAG Agent — Patch v9.7

Adds four features on top of v9.6:

1. **Type conversion** — `str → int → float → datetime → bool → category`, any-to-any.
2. **Analysis tiers** — deterministic Univariate, Bivariate, Multivariate reports with charts.
3. **ZIP export** — every full-analysis run produces a downloadable bundle
   (markdown report + raw stats JSON + all charts).
4. **Stop button** — cooperative cancel token that halts LLM streaming and
   preserves the partial answer with a `⏹️ Stopped by user.` marker.

## File map

```
src/services/type_converter.py         # NEW  — safe type coercion
src/services/analysis_tiers.py         # NEW  — univariate / bivariate / multivariate
src/services/analysis_router_patch.py  # PATCH — regex patterns to merge into router
src/services/generation_engine_patch.py# PATCH — cancel-aware stream() snippet
src/utils/cancel_token.py              # NEW  — threading.Event wrapper
src/utils/export_report.py             # NEW  — zip builder
app_patch_snippets.py                  # REF  — copy/paste blocks for app.py
```

## Installation

1. Unzip on top of your project root (new files land in `src/services/` and `src/utils/`).
2. Merge the patch files into their existing counterparts:
   - `analysis_router_patch.py` → add regex dicts into `analysis_router.py`.
   - `generation_engine_patch.py` → add `cancel_token` param to your `stream()`.
3. Wire `app.py` using the blocks in `app_patch_snippets.py`
   (Stop button, conversion intent, analysis tiers, zip download).

## Dependencies

```bash
pip install scikit-learn statsmodels seaborn matplotlib scipy pandas
```

## Chat examples that now work

- `convert age to int`
- `change signup_date type to datetime`
- `cast charges as string`
- `univariate analysis`
- `bivariate analysis`
- `multivariate analysis` (needs ≥3 numeric columns)
- `do all` — runs all three tiers + statistical test battery, then offers the ZIP.
- `⏹️ Stop` button appears whenever `generating=True`.

## Notes

- The stop token is process-global (`get_cancel_token()`). If you deploy
  multi-worker, move it into `st.session_state`.
- Multivariate falls back with a friendly message when the dataset has
  <3 numeric columns or <10 complete rows.
- `format_report()` gives you a ready-to-render markdown blob for every
  conversion.
