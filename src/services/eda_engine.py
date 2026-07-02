"""
EDA Engine (Patch v9.2) — runs the full 11-step Exploratory Data Analysis
workflow against a registered DataFrame with ZERO LLM calls.

Steps:
 1. Understand Data
 2. Shape and Data Types
 3. Missing Values
 4. Duplicates
 5. Numerical Analysis
 6. Categorical Analysis
 7. Bivariate Analysis
 8. Correlation
 9. Outliers (IQR)
10. Feature Relationships
11. Insights and Recommendations

Returns:
    {"markdown": str, "chart_paths": [str,...], "captions": [str,...]}
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.logger import get_logger

logger = get_logger(__name__)

sns.set_theme(style="whitegrid", context="notebook")

CHARTS_DIR = Path(os.getenv("CHARTS_DIR", "charts"))
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


# --- helpers -------------------------------------------------------------

def _save(fig, tag: str) -> str:
    h = hashlib.md5(f"{tag}_{np.random.rand()}".encode()).hexdigest()[:10]
    path = CHARTS_DIR / f"eda_{tag}_{h}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _md_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df is None or len(df) == 0:
        return "_(no rows)_"
    df = df.head(max_rows)
    try:
        return df.to_markdown()
    except Exception:
        return "```\n" + df.to_string() + "\n```"


def _numeric(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def _categorical(df: pd.DataFrame, max_card: int = 30) -> List[str]:
    cats = []
    for c in df.columns:
        if c in _numeric(df):
            continue
        try:
            if df[c].nunique(dropna=True) <= max_card:
                cats.append(c)
        except Exception:
            pass
    return cats


# --- individual steps ----------------------------------------------------

def _step_understand(df: pd.DataFrame) -> str:
    n_num = len(_numeric(df))
    n_cat = len(_categorical(df))
    return (
        f"### 1. Understand the Data\n\n"
        f"- **Rows:** {len(df):,}\n"
        f"- **Columns:** {len(df.columns)}\n"
        f"- **Numeric columns:** {n_num}\n"
        f"- **Categorical columns (card ≤ 30):** {n_cat}\n"
        f"- **Memory footprint:** {df.memory_usage(deep=True).sum() / 1024:.1f} KB\n"
        f"- **First 5 rows:**\n\n{_md_table(df.head(5))}\n"
    )


def _step_shape_types(df: pd.DataFrame) -> str:
    info = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null": df.notna().sum(),
        "nulls": df.isna().sum(),
        "unique": [df[c].nunique(dropna=True) for c in df.columns],
    })
    return f"### 2. Shape & Data Types\n\nShape: **{df.shape[0]} × {df.shape[1]}**\n\n{_md_table(info)}\n"


def _step_missing(df: pd.DataFrame) -> Tuple[str, List[str], List[str]]:
    n = df.isna().sum()
    pct = (n / len(df) * 100).round(2)
    out = pd.DataFrame({"nulls": n, "pct": pct})
    out = out[out["nulls"] > 0].sort_values("nulls", ascending=False)
    total = int(n.sum())
    md = [f"### 3. Missing Values\n\nTotal missing cells: **{total}**"]
    charts, caps = [], []
    if total == 0:
        md.append("\n_No missing values in the dataset._")
    else:
        md.append("\n" + _md_table(out))
        # heatmap of missingness
        try:
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.heatmap(df.isna(), cbar=False, yticklabels=False, ax=ax)
            ax.set_title("Missing-value heatmap")
            charts.append(_save(fig, "missing"))
            caps.append("Missing-value heatmap (yellow = missing)")
        except Exception as e:
            logger.debug(f"Missing heatmap skipped: {e}")
    return "\n".join(md), charts, caps


def _step_duplicates(df: pd.DataFrame) -> str:
    n_dup = int(df.duplicated().sum())
    md = f"### 4. Duplicates\n\nDuplicate rows detected: **{n_dup}** ({n_dup / len(df):.1%} of dataset)"
    if n_dup > 0:
        md += "\n\nSample duplicate rows:\n\n" + _md_table(df[df.duplicated(keep=False)].head(10))
    return md


def _step_numerical(df: pd.DataFrame) -> Tuple[str, List[str], List[str]]:
    nums = _numeric(df)
    if not nums:
        return "### 5. Numerical Analysis\n\n_No numeric columns._", [], []
    stats = df[nums].describe().T.round(3)
    stats["skew"] = df[nums].skew().round(3)
    stats["kurtosis"] = df[nums].kurtosis().round(3)
    md = f"### 5. Numerical Analysis\n\n{_md_table(stats)}"
    charts, caps = [], []
    # histograms grid (up to 6)
    sel = nums[:6]
    try:
        rows = (len(sel) + 2) // 3
        fig, axes = plt.subplots(rows, min(3, len(sel)), figsize=(14, 3.5 * rows))
        axes = np.atleast_1d(axes).flatten()
        for i, c in enumerate(sel):
            sns.histplot(df[c].dropna(), kde=True, ax=axes[i], color="steelblue")
            axes[i].set_title(c)
        for j in range(len(sel), len(axes)):
            axes[j].axis("off")
        charts.append(_save(fig, "hist"))
        caps.append("Distributions (histograms + KDE)")
    except Exception as e:
        logger.debug(f"Hist grid skipped: {e}")
    return md, charts, caps


def _step_categorical(df: pd.DataFrame) -> Tuple[str, List[str], List[str]]:
    cats = _categorical(df)
    if not cats:
        return "### 6. Categorical Analysis\n\n_No categorical columns detected._", [], []
    md = ["### 6. Categorical Analysis\n"]
    charts, caps = [], []
    for c in cats[:5]:
        vc = df[c].value_counts(dropna=False).head(10)
        md.append(f"**`{c}`** — {df[c].nunique()} unique values\n\n" + _md_table(vc.rename_axis(c).reset_index(name="count")))
    try:
        sel = cats[:min(4, len(cats))]
        rows = (len(sel) + 1) // 2
        fig, axes = plt.subplots(rows, min(2, len(sel)), figsize=(14, 4 * rows))
        axes = np.atleast_1d(axes).flatten()
        for i, c in enumerate(sel):
            top = df[c].value_counts(dropna=False).head(10)
            sns.barplot(x=top.values, y=top.index.astype(str), ax=axes[i], color="teal")
            axes[i].set_title(c)
        for j in range(len(sel), len(axes)):
            axes[j].axis("off")
        charts.append(_save(fig, "cat"))
        caps.append("Top categories per categorical column")
    except Exception as e:
        logger.debug(f"Categorical bars skipped: {e}")
    return "\n\n".join(md), charts, caps


def _step_bivariate(df: pd.DataFrame) -> Tuple[str, List[str], List[str]]:
    nums = _numeric(df)
    cats = _categorical(df)
    if not nums or not cats:
        return "### 7. Bivariate Analysis\n\n_Need at least 1 numeric and 1 categorical column._", [], []
    md = ["### 7. Bivariate Analysis (numeric × categorical)\n"]
    charts, caps = [], []
    pairs = []
    for cat in cats[:2]:
        for num in nums[:2]:
            pairs.append((cat, num))
    try:
        rows = (len(pairs) + 1) // 2
        fig, axes = plt.subplots(rows, min(2, len(pairs)), figsize=(14, 4.5 * rows))
        axes = np.atleast_1d(axes).flatten()
        for i, (cat, num) in enumerate(pairs):
            top = df[cat].value_counts().head(8).index
            sub = df[df[cat].isin(top)]
            sns.boxplot(data=sub, x=cat, y=num, ax=axes[i])
            axes[i].tick_params(axis="x", rotation=30)
            axes[i].set_title(f"{num} by {cat}")
            grp = df.groupby(cat)[num].mean().round(2).sort_values(ascending=False).head(5)
            md.append(f"**{num} by {cat}** — top means:\n\n" + _md_table(grp.reset_index()))
        for j in range(len(pairs), len(axes)):
            axes[j].axis("off")
        charts.append(_save(fig, "biv"))
        caps.append("Numeric distribution by category (boxplots)")
    except Exception as e:
        logger.debug(f"Bivariate skipped: {e}")
    return "\n\n".join(md), charts, caps


def _step_correlation(df: pd.DataFrame) -> Tuple[str, List[str], List[str]]:
    nums = _numeric(df)
    if len(nums) < 2:
        return "### 8. Correlation\n\n_Not enough numeric columns._", [], []
    corr = df[nums].corr().round(3)
    md = f"### 8. Correlation Matrix (Pearson)\n\n{_md_table(corr)}"
    charts, caps = [], []
    try:
        fig, ax = plt.subplots(figsize=(min(12, 1 + len(nums)), min(10, 1 + len(nums))))
        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax)
        ax.set_title("Correlation heatmap")
        charts.append(_save(fig, "corr"))
        caps.append("Pearson correlation heatmap")
    except Exception as e:
        logger.debug(f"Corr heatmap skipped: {e}")
    # strongest pairs
    pairs = (corr.where(~np.eye(len(corr), dtype=bool))
                 .stack().abs().sort_values(ascending=False))
    seen, top = set(), []
    for (a, b), v in pairs.items():
        key = frozenset([a, b])
        if key in seen:
            continue
        seen.add(key)
        top.append((a, b, round(corr.loc[a, b], 3)))
        if len(top) >= 5:
            break
    if top:
        md += "\n\n**Top |correlations|:**\n\n" + _md_table(pd.DataFrame(top, columns=["A", "B", "r"]))
    return md, charts, caps


def _step_outliers(df: pd.DataFrame) -> Tuple[str, List[str], List[str]]:
    nums = _numeric(df)
    if not nums:
        return "### 9. Outliers\n\n_No numeric columns._", [], []
    rows = []
    for c in nums:
        s = df[c].dropna()
        if len(s) < 5:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out = ((s < low) | (s > high)).sum()
        rows.append({"column": c, "n_outliers": int(out),
                     "pct": round(out / len(s) * 100, 2),
                     "low_bound": round(low, 3), "high_bound": round(high, 3)})
    md = "### 9. Outliers (IQR method, 1.5×)\n\n" + _md_table(pd.DataFrame(rows))
    charts, caps = [], []
    try:
        sel = nums[:6]
        cols = min(3, len(sel))
        rws = (len(sel) + cols - 1) // cols
        fig, axes = plt.subplots(rws, cols, figsize=(4.5 * cols, 3.5 * rws))
        axes = np.atleast_1d(axes).flatten()
        for i, c in enumerate(sel):
            sns.boxplot(x=df[c].dropna(), ax=axes[i], color="salmon")
            axes[i].set_title(c)
        for j in range(len(sel), len(axes)):
            axes[j].axis("off")
        charts.append(_save(fig, "box"))
        caps.append("Boxplots — outlier visualization (IQR)")
    except Exception as e:
        logger.debug(f"Boxplot grid skipped: {e}")
    return md, charts, caps


def _step_relationships(df: pd.DataFrame) -> Tuple[str, List[str], List[str]]:
    nums = _numeric(df)
    if len(nums) < 2:
        return "### 10. Feature Relationships\n\n_Not enough numeric columns for a pairplot._", [], []
    sel = nums[:4]
    charts, caps = [], []
    try:
        sample = df[sel].dropna()
        if len(sample) > 500:
            sample = sample.sample(500, random_state=0)
        g = sns.pairplot(sample, diag_kind="kde", plot_kws={"alpha": 0.6, "s": 15})
        g.fig.suptitle("Pairwise relationships", y=1.02)
        path = CHARTS_DIR / f"eda_pair_{hashlib.md5(str(sel).encode()).hexdigest()[:10]}.png"
        g.fig.savefig(path, dpi=90, bbox_inches="tight")
        plt.close(g.fig)
        charts.append(str(path))
        caps.append(f"Pairplot of {', '.join(sel)}")
    except Exception as e:
        logger.debug(f"Pairplot skipped: {e}")
    return "### 10. Feature Relationships\n\nPairwise relationships shown for: `" + ", ".join(sel) + "`", charts, caps


def _step_insights(df: pd.DataFrame) -> str:
    lines = ["### 11. Insights & Recommendations\n"]
    nulls = df.isna().sum()
    high_null = nulls[nulls / len(df) > 0.3]
    if len(high_null) > 0:
        lines.append(f"- ⚠️ Columns with >30% missing: **{', '.join(high_null.index)}** — consider dropping or imputing.")
    n_dup = int(df.duplicated().sum())
    if n_dup > 0:
        lines.append(f"- ⚠️ {n_dup} duplicate rows — recommend `df.drop_duplicates()`.")
    nums = _numeric(df)
    if nums:
        sk = df[nums].skew().abs()
        skewed = sk[sk > 1].index.tolist()
        if skewed:
            lines.append(f"- 📈 Highly skewed numeric columns: **{', '.join(skewed)}** — consider log or Box-Cox transform.")
    cats = _categorical(df)
    high_card = [c for c in cats if df[c].nunique() > 20]
    if high_card:
        lines.append(f"- 🔤 High-cardinality categoricals: **{', '.join(high_card)}** — consider grouping rare levels.")
    if len(nums) >= 2:
        corr = df[nums].corr().abs()
        np.fill_diagonal(corr.values, 0)
        strong = (corr > 0.8).sum().sum() // 2
        if strong > 0:
            lines.append(f"- 🔗 {strong} pair(s) of numeric features with |r| > 0.8 — potential multicollinearity.")
    if len(lines) == 1:
        lines.append("- ✅ No major data-quality issues detected.")
    lines.append("\n**Suggested next steps:** handle missing values, encode categoricals, scale numeric features, then run modeling.")
    return "\n".join(lines)


# --- public dispatcher ---------------------------------------------------

def run_full_eda(df: pd.DataFrame) -> Dict[str, Any]:
    """Execute all 11 steps. Returns markdown + list of chart image paths."""
    parts: List[str] = ["# 📊 Full Exploratory Data Analysis Report\n"]
    charts: List[str] = []
    caps: List[str] = []

    steps = [
        ("understand", lambda: (_step_understand(df), [], [])),
        ("shape_types", lambda: (_step_shape_types(df), [], [])),
        ("missing", lambda: _step_missing(df)),
        ("duplicates", lambda: (_step_duplicates(df), [], [])),
        ("numerical", lambda: _step_numerical(df)),
        ("categorical", lambda: _step_categorical(df)),
        ("bivariate", lambda: _step_bivariate(df)),
        ("correlation", lambda: _step_correlation(df)),
        ("outliers", lambda: _step_outliers(df)),
        ("relationships", lambda: _step_relationships(df)),
        ("insights", lambda: (_step_insights(df), [], [])),
    ]
    for name, fn in steps:
        try:
            md, ch, cp = fn()
            parts.append(md)
            charts.extend(ch)
            caps.extend(cp)
        except Exception as e:
            logger.warning(f"EDA step '{name}' failed: {e}")
            parts.append(f"### ⚠️ Step '{name}' skipped: {e}")

    return {
        "markdown": "\n\n---\n\n".join(parts),
        "chart_paths": charts,
        "captions": caps,
    }
