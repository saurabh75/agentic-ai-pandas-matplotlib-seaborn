"""
Render a matplotlib/seaborn chart from a ChartSpec + DataFrame.

Runs headless (Agg backend) so it is safe inside Streamlit / scripts.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional, Tuple

# IMPORTANT: select backend before pyplot import.
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.logger import get_logger
from src.services.chart_router import ChartSpec

logger = get_logger(__name__)

sns.set_theme(style="whitegrid", context="talk")

CHARTS_DIR = Path(os.getenv("CHARTS_DIR", "charts"))
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
CHART_MAX_ROWS = int(os.getenv("CHART_MAX_ROWS", "5000"))


def _maybe_agg(df: pd.DataFrame, spec: ChartSpec) -> pd.DataFrame:
    if not spec.agg or spec.agg == "none" or not spec.x or not spec.y:
        return df
    if spec.x not in df.columns or spec.y not in df.columns:
        return df
    try:
        gb = df.groupby(spec.x, dropna=False)[spec.y]
        if spec.agg == "sum":
            out = gb.sum().reset_index()
        elif spec.agg == "mean":
            out = gb.mean().reset_index()
        elif spec.agg == "count":
            out = gb.count().reset_index()
        else:
            return df
        return out
    except Exception:
        return df


def render(df: pd.DataFrame, spec: ChartSpec, query: str) -> Optional[Tuple[str, str]]:
    """
    Returns (image_path, caption) or None on failure.
    """
    if df is None or df.empty:
        return None
    if len(df) > CHART_MAX_ROWS:
        df = df.head(CHART_MAX_ROWS)

    df = _maybe_agg(df, spec)
    cols = list(df.columns)
    x = spec.x if spec.x in cols else (cols[0] if cols else None)
    y = spec.y if spec.y and spec.y in cols else (cols[1] if len(cols) > 1 else None)
    hue = spec.hue if spec.hue and spec.hue in cols else None

    try:
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=110)
        ctype = spec.chart_type

        if ctype == "bar":
            sns.barplot(data=df, x=x, y=y, hue=hue, ax=ax)
        elif ctype == "line":
            sns.lineplot(data=df, x=x, y=y, hue=hue, ax=ax, marker="o")
        elif ctype == "scatter":
            sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax)
        elif ctype == "hist":
            col = y or x
            sns.histplot(data=df, x=col, hue=hue, ax=ax, kde=True, bins=30)
        elif ctype == "box":
            sns.boxplot(data=df, x=x, y=y, hue=hue, ax=ax)
        elif ctype == "heatmap":
            num = df.select_dtypes(include="number")
            if num.shape[1] < 2:
                plt.close(fig)
                return None
            sns.heatmap(num.corr(), annot=True, fmt=".2f",
                        cmap="rocket_r", ax=ax)
        elif ctype == "pie":
            if not (x and y):
                plt.close(fig)
                return None
            ax.pie(df[y], labels=df[x].astype(str), autopct="%1.1f%%",
                   startangle=90)
            ax.axis("equal")
        else:
            plt.close(fig)
            return None

        ax.set_title(spec.title or query[:80])
        # Rotate long x tick labels.
        if ctype in {"bar", "line", "scatter", "box"} and x is not None:
            try:
                if df[x].dtype == object or df.shape[0] > 8:
                    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            except Exception:
                pass
        fig.tight_layout()

        h = hashlib.sha1((query + spec.chart_type + str(cols)).encode()).hexdigest()[:10]
        path = CHARTS_DIR / f"chart_{h}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

        caption = (
            f"{spec.chart_type.title()} chart"
            + (f" — {spec.rationale}" if spec.rationale else "")
        )
        return str(path), caption
    except Exception as e:
        logger.warning(f"Chart render failed: {e}")
        try:
            plt.close("all")
        except Exception:
            pass
        return None
