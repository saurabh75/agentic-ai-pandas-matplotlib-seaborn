"""
Pandas Executor (Patch v5).

Runs a whitelisted set of pandas operations against a registered DataFrame.
NO eval/exec. Column names are validated against the schema.

Every function returns:
    { "markdown": str, "df": Optional[DataFrame], "chart_hint": Optional[str] }
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# --- helpers -------------------------------------------------------------

def _md_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df is None or len(df) == 0:
        return "_(no rows)_"
    if len(df) > max_rows:
        df = df.head(max_rows)
    try:
        return df.to_markdown(index=True)
    except Exception:
        return "```\n" + df.to_string() + "\n```"


def _numeric_cols(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def _valid_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


# --- operations ----------------------------------------------------------

def describe(df: pd.DataFrame, cols: Optional[List[str]] = None) -> Dict[str, Any]:
    cols = _valid_cols(df, cols or [])
    target = df[cols] if cols else df
    numeric = target.describe(include="all").transpose()
    md = [f"**Dataset overview** — {len(df)} rows × {len(df.columns)} columns\n"]
    md.append("**Column types:**\n")
    dtype_df = pd.DataFrame({"dtype": df.dtypes.astype(str),
                             "non_null": df.notna().sum(),
                             "nulls": df.isna().sum()})
    md.append(_md_table(dtype_df))
    md.append("\n**Statistics:**\n")
    md.append(_md_table(numeric))
    return {"markdown": "\n".join(md), "df": numeric, "chart_hint": None}


def nulls(df: pd.DataFrame, **_) -> Dict[str, Any]:
    n = df.isna().sum()
    pct = (n / len(df) * 100).round(2)
    out = pd.DataFrame({"null_count": n, "null_%": pct})
    out = out[out["null_count"] > 0] if out["null_count"].sum() > 0 else out
    total = int(df.isna().sum().sum())
    header = f"**Null / missing value report** — total nulls: **{total}** across {len(df)} rows\n"
    if total == 0:
        return {"markdown": header + "\n_No missing values anywhere._", "df": None, "chart_hint": None}
    return {"markdown": header + "\n" + _md_table(out), "df": out.reset_index(), "chart_hint": "bar"}


def duplicates(df: pd.DataFrame, **_) -> Dict[str, Any]:
    mask = df.duplicated(keep=False)
    n_dup = int(df.duplicated().sum())
    if n_dup == 0:
        return {"markdown": f"**Duplicates:** none found across {len(df)} rows.", "df": None, "chart_hint": None}
    dup_rows = df[mask].sort_values(by=list(df.columns)).head(20)
    md = [
        f"**Duplicates report** — {n_dup} duplicate row(s) detected (out of {len(df)}).",
        f"\nShowing up to 20 example duplicate rows:\n",
        _md_table(dup_rows, max_rows=20),
    ]
    return {"markdown": "\n".join(md), "df": dup_rows, "chart_hint": None}


def head(df: pd.DataFrame, params: Optional[dict] = None, **_) -> Dict[str, Any]:
    n = int((params or {}).get("n", 10))
    n = max(1, min(n, 100))
    return {"markdown": f"**First {n} rows:**\n\n" + _md_table(df.head(n)), "df": df.head(n), "chart_hint": None}


def value_counts(df: pd.DataFrame, cols: List[str], params: Optional[dict] = None) -> Dict[str, Any]:
    cols = _valid_cols(df, cols)
    if not cols:
        # fall back to first non-numeric column
        cat = [c for c in df.columns if c not in _numeric_cols(df)]
        if not cat:
            return {"markdown": "_Could not find a categorical column to count._", "df": None, "chart_hint": None}
        cols = [cat[0]]
    parts = []
    first_df = None
    for c in cols[:3]:
        vc = df[c].value_counts(dropna=False).head(20).rename_axis(c).reset_index(name="count")
        parts.append(f"**Value counts — `{c}`:**\n\n" + _md_table(vc))
        if first_df is None:
            first_df = vc
    return {"markdown": "\n\n".join(parts), "df": first_df, "chart_hint": "bar"}


def correlation(df: pd.DataFrame, cols: Optional[List[str]] = None, **_) -> Dict[str, Any]:
    num = df[_numeric_cols(df)] if not cols else df[_valid_cols(df, cols)].select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        # try dummying categoricals
        num = pd.get_dummies(df, drop_first=False).select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return {"markdown": "_Not enough numeric columns for correlation._", "df": None, "chart_hint": None}
    corr = num.corr().round(3)
    md = "**Correlation matrix** (Pearson):\n\n" + _md_table(corr)
    return {"markdown": md, "df": corr, "chart_hint": "heatmap"}


def groupby(df: pd.DataFrame, cols: List[str], params: Optional[dict] = None) -> Dict[str, Any]:
    cols = _valid_cols(df, cols)
    if not cols:
        return {"markdown": "_Please mention which column to group by._", "df": None, "chart_hint": None}
    group_col = cols[0]
    metric_cols = [c for c in cols[1:] if c in _numeric_cols(df)] or _numeric_cols(df)
    if not metric_cols:
        return {"markdown": "_No numeric columns to aggregate._", "df": None, "chart_hint": None}
    agg = df.groupby(group_col)[metric_cols].agg(["mean", "median", "count"]).round(3)
    md = f"**Group by `{group_col}`:**\n\n" + _md_table(agg)
    flat = df.groupby(group_col)[metric_cols[0]].mean().reset_index()
    return {"markdown": md, "df": flat, "chart_hint": "bar"}


# --- dispatch ------------------------------------------------------------

_OPS = {
    "describe":     describe,
    "info":         describe,
    "nulls":        nulls,
    "duplicates":   duplicates,
    "head":         head,
    "value_counts": value_counts,
    "correlation":  correlation,
    "groupby":      groupby,
}


def run(op: str, df: pd.DataFrame, cols: Optional[List[str]] = None,
        params: Optional[dict] = None) -> Optional[Dict[str, Any]]:
    fn = _OPS.get(op)
    if fn is None:
        return None
    try:
        return fn(df=df, cols=cols or [], params=params or {})
    except TypeError:
        # some ops don't take cols/params
        return fn(df=df)


def supported() -> List[str]:
    return list(_OPS.keys())
