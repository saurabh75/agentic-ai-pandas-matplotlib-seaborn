"""
Chart intent router.

Two-stage detection:
  1. Cheap regex on keywords  -> skip LLM entirely for clearly non-chart queries.
  2. Small JSON-mode LLM call -> when intent is plausible, ask the model to
     produce a ChartSpec (or null).

The LLM is the same Ollama model already loaded for generation, so no extra
VRAM is consumed. Set CHARTS_ENABLED=false in .env to disable globally.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

from src.logger import get_logger

logger = get_logger(__name__)

# Keywords that hint at a visualization request.
_CHART_KEYWORDS = re.compile(
    r"\b(chart|plot|graph|visuali[sz]e|visuali[sz]ation|trend|distribution|"
    r"histogram|bar\s*chart|line\s*chart|scatter|heat\s*map|heatmap|box\s*plot|"
    r"compare|comparison|over\s+time|by\s+(?:year|month|day|category|group))\b",
    re.IGNORECASE,
)

# Allowed chart types -> matches keys in chart_renderer.RENDERERS.
ALLOWED_TYPES = {"bar", "line", "scatter", "hist", "box", "heatmap", "pie"}


@dataclass
class ChartSpec:
    chart_type: str                    # one of ALLOWED_TYPES
    x: Optional[str] = None            # column name for X axis
    y: Optional[str] = None            # column name for Y axis (or value)
    hue: Optional[str] = None          # optional grouping column
    agg: Optional[str] = None          # sum / mean / count / none
    title: str = ""
    rationale: str = ""                # why the agent picked this chart

    def is_valid(self) -> bool:
        return self.chart_type in ALLOWED_TYPES


class ChartIntentRouter:
    """Decides whether a query should produce a chart and what kind."""

    def __init__(self, llm=None) -> None:
        self._llm = llm  # langchain Ollama LLM, optional

    # ------------------------------------------------------------------
    def has_keyword_intent(self, query: str) -> bool:
        return bool(_CHART_KEYWORDS.search(query or ""))

    # ------------------------------------------------------------------
    def detect(self, query: str, sample_columns: List[str]) -> Optional[ChartSpec]:
        """
        Return a ChartSpec or None. `sample_columns` are columns available in
        the extracted DataFrame; we pass them to the LLM so it picks real names.
        """
        if not self.has_keyword_intent(query):
            return None

        if self._llm is None or not sample_columns:
            # Fallback: pick a sensible default with the first two columns.
            if len(sample_columns) >= 2:
                return ChartSpec(
                    chart_type="bar",
                    x=sample_columns[0],
                    y=sample_columns[1],
                    title=query[:80],
                    rationale="keyword-only fallback",
                )
            return None

        prompt = (
            "You decide if a user's question should be answered with a chart.\n"
            f"Available columns: {sample_columns}\n"
            f"Allowed chart_type: {sorted(ALLOWED_TYPES)}\n"
            f'User question: "{query}"\n\n'
            "Respond with ONLY a compact JSON object, no prose, no markdown:\n"
            '{"chart_type":"bar|line|scatter|hist|box|heatmap|pie",'
            '"x":"<col or null>","y":"<col or null>","hue":"<col or null>",'
            '"agg":"sum|mean|count|none","title":"<short>","rationale":"<short>"}\n'
            'If a chart does NOT make sense, respond exactly: {"chart_type":"none"}'
        )

        try:
            raw = self._llm.invoke(prompt)
            text = getattr(raw, "content", raw) if not isinstance(raw, str) else raw
            text = text.strip()
            # Trim to the first {...} block to be tolerant of stray tokens.
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                return None
            data = json.loads(m.group(0))
            ctype = (data.get("chart_type") or "").lower()
            if ctype == "none" or ctype not in ALLOWED_TYPES:
                return None
            return ChartSpec(
                chart_type=ctype,
                x=data.get("x") or None,
                y=data.get("y") or None,
                hue=data.get("hue") or None,
                agg=(data.get("agg") or "none").lower(),
                title=data.get("title") or query[:80],
                rationale=data.get("rationale") or "",
            )
        except Exception as e:
            logger.warning(f"ChartIntentRouter LLM parse failed: {e}")
            return None
