"""
Statistics Engine (Patch v5).

Wraps scipy.stats + statsmodels for hypothesis testing and regression.
Modeled on the reference notebook (hypothesis-testing-of-health-insurance-data).

All functions return: { "markdown": str, "chart_hint": Optional[str] }
NO LLM calls — pure numeric.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _interpret(p: float, alpha: float = 0.05) -> str:
    if p < alpha:
        return (f"p = {p:.4g} < {alpha} → **reject H₀**. "
                "There IS a statistically significant difference.")
    return (f"p = {p:.4g} ≥ {alpha} → **fail to reject H₀**. "
            "No statistically significant difference detected.")


def ttest(df: pd.DataFrame, cols: List[str], **_) -> Dict[str, Any]:
    """Two-sample independent t-test. cols = [numeric_col, group_col]."""
    try:
        from scipy import stats
    except ImportError:
        return {"markdown": "_scipy not installed._", "chart_hint": None}

    if len(cols) < 2:
        return {"markdown": "_Need a numeric column and a 2-group column, e.g. "
                            "'compare charges between smoker and non-smoker'._",
                "chart_hint": None}

    num_col = next((c for c in cols if pd.api.types.is_numeric_dtype(df[c])), None)
    grp_col = next((c for c in cols if c != num_col), None)
    if num_col is None or grp_col is None:
        return {"markdown": "_Could not identify numeric + group columns._", "chart_hint": None}

    groups = df[grp_col].dropna().unique()
    if len(groups) != 2:
        return {"markdown": f"_`{grp_col}` has {len(groups)} groups; t-test needs exactly 2. "
                            "Use ANOVA instead._",
                "chart_hint": None}

    a = df[df[grp_col] == groups[0]][num_col].dropna()
    b = df[df[grp_col] == groups[1]][num_col].dropna()
    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)  # Welch's

    md = [
        f"**Two-sample t-test (Welch)** — `{num_col}` between `{grp_col}` groups",
        "",
        f"- Group `{groups[0]}`: n={len(a)}, mean={a.mean():.4g}, std={a.std():.4g}",
        f"- Group `{groups[1]}`: n={len(b)}, mean={b.mean():.4g}, std={b.std():.4g}",
        f"- t-statistic: **{t_stat:.4f}**",
        f"- p-value: **{p_val:.4g}**",
        "",
        _interpret(p_val),
    ]
    return {"markdown": "\n".join(md), "chart_hint": "box"}


def anova(df: pd.DataFrame, cols: List[str], **_) -> Dict[str, Any]:
    """One-way ANOVA. cols = [numeric_col, group_col]."""
    try:
        from scipy import stats
    except ImportError:
        return {"markdown": "_scipy not installed._", "chart_hint": None}

    if len(cols) < 2:
        return {"markdown": "_Need numeric and group columns, e.g. "
                            "'compare charges across regions'._",
                "chart_hint": None}

    num_col = next((c for c in cols if pd.api.types.is_numeric_dtype(df[c])), None)
    grp_col = next((c for c in cols if c != num_col), None)
    if num_col is None or grp_col is None:
        return {"markdown": "_Could not identify numeric + group columns._", "chart_hint": None}

    groups = [g[num_col].dropna().values for _, g in df.groupby(grp_col) if g[num_col].notna().any()]
    if len(groups) < 2:
        return {"markdown": "_Need at least 2 non-empty groups._", "chart_hint": None}

    f_stat, p_val = stats.f_oneway(*groups)

    summary = df.groupby(grp_col)[num_col].agg(["count", "mean", "std"]).round(4)

    md = [
        f"**One-way ANOVA** — `{num_col}` across `{grp_col}`",
        "",
        f"- F-statistic: **{f_stat:.4f}**",
        f"- p-value: **{p_val:.4g}**",
        "",
        _interpret(p_val),
        "",
        "**Group summary:**",
        "",
        summary.to_markdown(),
    ]
    return {"markdown": "\n".join(md), "chart_hint": "box"}


def chi2(df: pd.DataFrame, cols: List[str], **_) -> Dict[str, Any]:
    """Chi-square test of independence between 2 categorical columns."""
    try:
        from scipy import stats
    except ImportError:
        return {"markdown": "_scipy not installed._", "chart_hint": None}

    if len(cols) < 2:
        return {"markdown": "_Need two categorical columns, e.g. 'is smoking independent of region'._",
                "chart_hint": None}

    a, b = cols[0], cols[1]
    ct = pd.crosstab(df[a], df[b])
    if ct.size == 0:
        return {"markdown": "_Empty contingency table._", "chart_hint": None}

    chi2_stat, p_val, dof, _ = stats.chi2_contingency(ct)
    md = [
        f"**Chi-square test of independence** — `{a}` vs `{b}`",
        "",
        f"- χ²: **{chi2_stat:.4f}**",
        f"- degrees of freedom: {dof}",
        f"- p-value: **{p_val:.4g}**",
        "",
        _interpret(p_val).replace("difference", "association"),
        "",
        "**Contingency table:**",
        "",
        ct.to_markdown(),
    ]
    return {"markdown": "\n".join(md), "chart_hint": "heatmap"}


def regression(df: pd.DataFrame, cols: List[str], **_) -> Dict[str, Any]:
    """OLS regression. First column = dependent, rest = independents."""
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return {"markdown": "_statsmodels not installed._", "chart_hint": None}

    if len(cols) < 2:
        return {"markdown": "_Need a dependent column and at least one predictor, e.g. "
                            "'regress charges on age bmi smoker'._",
                "chart_hint": None}

    y_col = cols[0]
    x_cols = cols[1:]

    if not pd.api.types.is_numeric_dtype(df[y_col]):
        return {"markdown": f"_Dependent variable `{y_col}` must be numeric._", "chart_hint": None}

    # sanitize column names for formula (statsmodels dislikes spaces / special chars)
    def _safe(c: str) -> str:
        return f"Q('{c}')"

    formula = f"{_safe(y_col)} ~ " + " + ".join(_safe(c) for c in x_cols)
    try:
        model = smf.ols(formula, data=df).fit()
    except Exception as e:
        return {"markdown": f"_Regression failed: {e}_", "chart_hint": None}

    coefs = pd.DataFrame({
        "coef": model.params.round(4),
        "std_err": model.bse.round(4),
        "t": model.tvalues.round(3),
        "p_value": model.pvalues.round(4),
    })
    md = [
        f"**OLS Regression** — `{y_col}` ~ {', '.join(x_cols)}",
        "",
        f"- R²: **{model.rsquared:.4f}**",
        f"- Adjusted R²: {model.rsquared_adj:.4f}",
        f"- F-statistic p-value: **{model.f_pvalue:.4g}**",
        f"- Observations: {int(model.nobs)}",
        "",
        "**Coefficients:**",
        "",
        coefs.to_markdown(),
    ]
    return {"markdown": "\n".join(md), "chart_hint": None}


def normality(df: pd.DataFrame, cols: List[str], **_) -> Dict[str, Any]:
    """Shapiro-Wilk test on a numeric column."""
    try:
        from scipy import stats
    except ImportError:
        return {"markdown": "_scipy not installed._", "chart_hint": None}

    numeric = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()[:1]
    if not numeric:
        return {"markdown": "_No numeric column to test._", "chart_hint": None}

    col = numeric[0]
    data = df[col].dropna()
    # Shapiro is best up to ~5000
    sample = data.sample(min(len(data), 5000), random_state=0) if len(data) > 5000 else data
    stat, p_val = stats.shapiro(sample)

    md = [
        f"**Shapiro-Wilk normality test** — `{col}`",
        f"(sampled {len(sample)} of {len(data)} values)",
        "",
        f"- W-statistic: **{stat:.4f}**",
        f"- p-value: **{p_val:.4g}**",
        "",
        _interpret(p_val).replace("difference", "deviation from normality"),
        "",
        f"Skew: {data.skew():.3f}   Kurtosis: {data.kurt():.3f}",
    ]
    return {"markdown": "\n".join(md), "chart_hint": "hist"}


# --- dispatch ------------------------------------------------------------

_OPS = {
    "ttest":      ttest,
    "anova":      anova,
    "chi2":       chi2,
    "regression": regression,
    "normality":  normality,
}


def run(op: str, df: pd.DataFrame, cols: Optional[List[str]] = None,
        params: Optional[dict] = None) -> Optional[Dict[str, Any]]:
    fn = _OPS.get(op)
    if fn is None:
        return None
    return fn(df=df, cols=cols or [], params=params or {})


def supported() -> List[str]:
    return list(_OPS.keys())
