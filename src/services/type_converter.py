"""Safe type coercion for any column to any dtype.

Supports: int, float, str, bool, datetime, category.
Use `errors='coerce'` to turn invalid cells into NaN instead of raising.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

SUPPORTED_TYPES = {
    "int": "int64", "integer": "int64", "int64": "int64",
    "float": "float64", "double": "float64", "decimal": "float64", "float64": "float64",
    "str": "string", "string": "string", "text": "string", "object": "string",
    "bool": "boolean", "boolean": "boolean",
    "datetime": "datetime64[ns]", "date": "datetime64[ns]", "timestamp": "datetime64[ns]",
    "category": "category", "categorical": "category",
}


def convert_column(df: pd.DataFrame, column: str, target: str,
                   errors: str = "coerce") -> tuple[pd.DataFrame, dict]:
    """Convert ``df[column]`` to ``target`` dtype.

    Parameters
    ----------
    df : pd.DataFrame
    column : str
        Column name to convert.
    target : str
        Any key from :data:`SUPPORTED_TYPES` (case-insensitive).
    errors : {'coerce', 'raise', 'ignore'}
        'coerce' converts invalid cells to NaN (recommended for chat UX).

    Returns
    -------
    (new_df, report) : tuple[pd.DataFrame, dict]
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found. Available: {list(df.columns)}")

    target_norm = target.lower().strip()
    if target_norm not in SUPPORTED_TYPES:
        raise ValueError(
            f"Unsupported type '{target}'. Use one of: {sorted(set(SUPPORTED_TYPES))}"
        )

    dtype = SUPPORTED_TYPES[target_norm]
    original = df[column].copy()
    before_type = str(original.dtype)
    new_df = df.copy()

    try:
        if dtype.startswith("datetime"):
            new_df[column] = pd.to_datetime(original, errors=errors)
        elif dtype == "int64":
            # str -> int needs a numeric pass first so "3.0" survives.
            new_df[column] = pd.to_numeric(original, errors=errors).astype("Int64")
        elif dtype == "float64":
            new_df[column] = pd.to_numeric(original, errors=errors)
        elif dtype == "boolean":
            truthy = {"true", "yes", "1", "y", "t"}
            falsy = {"false", "no", "0", "n", "f"}

            def _to_bool(v):
                if pd.isna(v):
                    return pd.NA
                s = str(v).strip().lower()
                if s in truthy:
                    return True
                if s in falsy:
                    return False
                return pd.NA

            new_df[column] = original.map(_to_bool).astype("boolean")
        elif dtype == "category":
            new_df[column] = original.astype("category")
        else:  # string
            new_df[column] = original.astype("string")
    except Exception as exc:
        if errors == "raise":
            raise
        return df, {"success": False, "error": str(exc), "column": column}

    n_before_null = int(original.isna().sum())
    n_after_null = int(new_df[column].isna().sum())
    n_lost = n_after_null - n_before_null

    return new_df, {
        "success": True,
        "column": column,
        "from_type": before_type,
        "to_type": str(new_df[column].dtype),
        "rows_converted": len(original) - n_lost,
        "rows_failed": n_lost,
        "sample_before": original.head(5).tolist(),
        "sample_after": new_df[column].head(5).tolist(),
    }


def bulk_convert(df: pd.DataFrame, conversions: dict[str, str]) -> tuple[pd.DataFrame, list[dict]]:
    """Convert many columns at once.

    Example
    -------
    >>> df, reports = bulk_convert(df, {"age": "int", "signup_date": "datetime"})
    """
    reports = []
    for col, tgt in conversions.items():
        df, rpt = convert_column(df, col, tgt)
        reports.append(rpt)
    return df, reports


def format_report(report: dict) -> str:
    """Render a single conversion report as human-friendly markdown."""
    if not report.get("success"):
        return f"❌ **{report.get('column')}** — {report.get('error')}"
    return (
        f"✅ **{report['column']}**: `{report['from_type']}` → `{report['to_type']}`\n"
        f"- Rows converted: {report['rows_converted']:,}\n"
        f"- Rows failed (→ NaN): {report['rows_failed']:,}\n"
        f"- Sample before: {report['sample_before']}\n"
        f"- Sample after:  {report['sample_after']}"
    )
