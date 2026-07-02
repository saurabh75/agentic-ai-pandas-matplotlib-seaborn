"""
Extract a pandas DataFrame from retrieved RAG context.

Strategy (in order):
  1. Look for CSV-shaped content in any chunk (delimiter detection).
  2. Look for Markdown / pipe tables (| col | col |).
  3. Ask the LLM (optional) to convert prose numbers into a small JSON table.

Returns (df, source_note) or (None, reason).
"""

from __future__ import annotations

import io
import json
import re
from typing import List, Optional, Tuple

import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)

_MAX_ROWS = 5000


def _try_csv(text: str) -> Optional[pd.DataFrame]:
    # Heuristic: at least 2 lines, commas or semicolons or tabs present,
    # and consistent column counts on the first few rows.
    head = "\n".join(text.strip().splitlines()[:50])
    if not head or "\n" not in head:
        return None
    for sep in (",", ";", "\t", "|"):
        if sep not in head:
            continue
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, engine="python",
                             on_bad_lines="skip", nrows=_MAX_ROWS)
            if df.shape[1] >= 2 and df.shape[0] >= 2:
                # Drop fully-empty columns from markdown noise.
                df = df.dropna(axis=1, how="all")
                if df.shape[1] >= 2:
                    return df
        except Exception:
            continue
    return None


_MD_TABLE_RE = re.compile(
    r"(\|[^\n]+\|\n\|[\s\-:|]+\|\n(?:\|[^\n]+\|\n?)+)", re.MULTILINE
)


def _try_markdown_table(text: str) -> Optional[pd.DataFrame]:
    m = _MD_TABLE_RE.search(text or "")
    if not m:
        return None
    raw = m.group(1)
    lines = [ln.strip().strip("|") for ln in raw.strip().splitlines()]
    if len(lines) < 3:
        return None
    headers = [h.strip() for h in lines[0].split("|")]
    rows = []
    for ln in lines[2:]:
        cells = [c.strip() for c in ln.split("|")]
        if len(cells) == len(headers):
            rows.append(cells)
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=headers)
    # Try to coerce numerics.
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")
    return df


def _try_llm_json(text: str, query: str, llm) -> Optional[pd.DataFrame]:
    if llm is None or not text.strip():
        return None
    prompt = (
        "Extract a small structured table from the context that answers the user's "
        "question. Return ONLY compact JSON of the form "
        '{"columns":["col1","col2",...],"rows":[[..],[..]]} '
        "with at most 50 rows. If no numeric/tabular data is present, return "
        '{"columns":[],"rows":[]}.\n\n'
        f"User question: {query}\n\nContext:\n{text[:4000]}"
    )
    try:
        raw = llm.invoke(prompt)
        out = getattr(raw, "content", raw) if not isinstance(raw, str) else raw
        m = re.search(r"\{.*\}", out, re.DOTALL)
        if not m:
            return None
        data = json.loads(m.group(0))
        cols = data.get("columns") or []
        rows = data.get("rows") or []
        if not cols or not rows:
            return None
        df = pd.DataFrame(rows, columns=cols)
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="ignore")
        return df.head(_MAX_ROWS)
    except Exception as e:
        logger.warning(f"LLM JSON table extraction failed: {e}")
        return None


def extract_dataframe(
    chunks: List, query: str, llm=None
) -> Tuple[Optional[pd.DataFrame], str]:
    """
    chunks: iterable of objects with `.content` and `.metadata`.
    """
    if not chunks:
        return None, "no chunks"

    # 1) per-chunk CSV / markdown table
    for c in chunks:
        content = getattr(c, "content", "") or ""
        df = _try_csv(content) or _try_markdown_table(content)
        if df is not None:
            src = (getattr(c, "metadata", {}) or {}).get("filename", "context")
            return df, f"tabular extract from {src}"

    # 2) concatenate and ask the LLM
    joined = "\n\n".join(getattr(c, "content", "") or "" for c in chunks)
    df = _try_llm_json(joined, query, llm)
    if df is not None and len(df) > 0:
        return df, "LLM-structured extract"

    return None, "no tabular data found in context"
