"""
Analysis Intent Router (Patch v5).

Detects whether a query should be answered by running pandas/statsmodels
against a registered tabular file, instead of the RAG pipeline.

Regex-first (~1 ms) — fast path covers ~90% of cases.
LLM classifier is deliberately optional and off by default.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

# --- intent → regex map --------------------------------------------------
_PATTERNS = {
    "describe":     r"\b(describe|summary|summarize|summarise|overview|stat[s]?|info|schema|dtypes?)\b",
    "nulls":        r"\b(null|na|nan|missing|empty|blank)\b",
    "duplicates":   r"\b(duplicate|dup|repeated|repeats)\b",
    "value_counts": r"\b(count(s)?|frequency|value[_ ]counts|how many|top \d+|unique)\b",
    "correlation":  r"\b(correlation|corr|heatmap|correlated|relationship between)\b",
    "groupby":      r"\b(group ?by|per|by (each )?\w+|average .* by|mean .* by|sum .* by)\b",
    "ttest":        r"\b(t[- ]?test|significant difference|compare .* between|differ between)\b",
    "anova":        r"\b(anova|analysis of variance|one[- ]way|compare .* across|means across)\b",
    "chi2":         r"\b(chi[- ]?squared?|chi2|contingency|independence)\b",
    "regression":   r"\b(regress|regression|ols|linear model|predict\w*)\b",
    "normality":    r"\b(normal(ly)? distributed|normality|shapiro|qq[- ]?plot|q-q)\b",
    "chart":        r"\b(plot|chart|graph|visuali[sz]e|bar chart|line chart|scatter|histogram|hist|box ?plot|violin|pie chart|pairplot|countplot)\b",
    "head":         r"\b(head|first \d+ rows?|sample|show me the data)\b",
}


@dataclass
class AnalysisSpec:
    operation: str                       # one of _PATTERNS keys
    file: Optional[str] = None           # tabular file name (None = newest)
    columns: List[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)

    def is_valid(self) -> bool:
        return self.operation in _PATTERNS


class AnalysisRouter:
    """Regex-first, tiny-LLM-fallback classifier for analysis intents."""

    def __init__(self, llm=None) -> None:
        self._llm = llm
        self._use_llm = os.getenv("ANALYSIS_ROUTER_USE_LLM", "false").lower() == "true"

    def detect(self, query: str, columns: List[str], filename: Optional[str] = None) -> Optional[AnalysisSpec]:
        q = query.lower().strip()
        # 1. regex hit
        for op, pat in _PATTERNS.items():
            if re.search(pat, q):
                cols = _extract_columns(query, columns)
                return AnalysisSpec(operation=op, file=filename, columns=cols)

        # 2. optional LLM fallback (kept short)
        if self._use_llm and self._llm is not None:
            try:
                op = self._llm_classify(query, columns)
                if op:
                    cols = _extract_columns(query, columns)
                    return AnalysisSpec(operation=op, file=filename, columns=cols)
            except Exception:
                return None
        return None

    def _llm_classify(self, query: str, columns: List[str]) -> Optional[str]:
        ops = ",".join(_PATTERNS.keys()) + ",none"
        prompt = (
            f"Classify this data-analysis question into ONE label from: {ops}.\n"
            f"Available columns: {columns}\n"
            f"Question: {query}\n"
            f"Answer with a single label, no other text."
        )
        try:
            resp = self._llm.invoke(prompt).strip().lower().split()[0]
        except Exception:
            return None
        if resp in _PATTERNS:
            return resp
        return None


def _extract_columns(query: str, columns: List[str]) -> List[str]:
    """Find column names mentioned in the query (case-insensitive, word-boundary)."""
    q = query.lower()
    hits: List[str] = []
    for c in columns:
        c_low = str(c).lower()
        if not c_low:
            continue
        # word-boundary match; underscores treated as word chars
        pat = r"(?<![a-z0-9_])" + re.escape(c_low) + r"(?![a-z0-9_])"
        if re.search(pat, q):
            hits.append(c)
    return hits
