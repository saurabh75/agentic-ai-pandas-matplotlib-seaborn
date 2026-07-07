"""Univariate / Bivariate / Multivariate analysis tiers.

Every function is deterministic (no LLM), returns a dict with
`tables`/`results` and a list of chart file paths so the caller can
render them in Streamlit and package them into the export zip.
"""
from __future__ import annotations

import os
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# ═════════════════════════════ UNIVARIATE ═════════════════════════════
def univariate_analysis(df: pd.DataFrame, output_dir: str) -> dict:
    """One variable at a time: distribution, central tendency, spread."""
    os.makedirs(output_dir, exist_ok=True)
    charts: list[str] = []
    tables: dict[str, dict] = {}

    for col in df.select_dtypes(include=np.number).columns:
        s = df[col].dropna()
        if len(s) < 3:
            continue
        tables[col] = {
            "mean": float(s.mean()), "median": float(s.median()),
            "mode": float(s.mode().iloc[0]) if not s.mode().empty else None,
            "std": float(s.std()), "var": float(s.var()),
            "min": float(s.min()), "max": float(s.max()),
            "range": float(s.max() - s.min()),
            "q1": float(s.quantile(.25)), "q3": float(s.quantile(.75)),
            "iqr": float(s.quantile(.75) - s.quantile(.25)),
            "skew": float(stats.skew(s)),
            "kurtosis": float(stats.kurtosis(s)),
            "cv": float(s.std() / s.mean()) if s.mean() else None,
            "shapiro_p": float(stats.shapiro(s.sample(min(len(s), 5000)))[1]),
        }
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        sns.histplot(s, kde=True, ax=axes[0]); axes[0].set_title(f"{col} — Distribution")
        sns.boxplot(x=s, ax=axes[1]); axes[1].set_title(f"{col} — Boxplot")
        stats.probplot(s, plot=axes[2]); axes[2].set_title(f"{col} — Q-Q Plot")
        path = os.path.join(output_dir, f"uni_{col}.png")
        plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
        charts.append(path)

    for col in df.select_dtypes(include=["object", "category"]).columns:
        vc = df[col].value_counts()
        if vc.empty:
            continue
        tables[col] = {
            "unique": int(df[col].nunique()),
            "mode": str(vc.index[0]),
            "top_freq": int(vc.iloc[0]),
            "entropy": float(stats.entropy(vc)),
        }
        fig, ax = plt.subplots(figsize=(8, 4))
        vc.head(15).plot(kind="bar", ax=ax)
        ax.set_title(f"{col} — Top Frequencies")
        path = os.path.join(output_dir, f"uni_{col}.png")
        plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
        charts.append(path)

    return {"tables": tables, "charts": charts}


# ═════════════════════════════ BIVARIATE ═════════════════════════════
def bivariate_analysis(df: pd.DataFrame, output_dir: str) -> dict:
    """Two variables: relationships, associations, group differences."""
    os.makedirs(output_dir, exist_ok=True)
    charts: list[str] = []
    results = {"num_num": [], "num_cat": [], "cat_cat": []}
    num = df.select_dtypes(include=np.number).columns.tolist()
    cat = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # numeric x numeric
    for a, b in combinations(num, 2):
        pair = df[[a, b]].dropna()
        if len(pair) < 3:
            continue
        x, y = pair[a].values, pair[b].values
        pr, pp = stats.pearsonr(x, y)
        sr, sp = stats.spearmanr(x, y)
        results["num_num"].append({
            "x": a, "y": b,
            "pearson_r": float(pr), "pearson_p": float(pp),
            "spearman_r": float(sr), "spearman_p": float(sp),
        })
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.regplot(x=x, y=y, ax=ax, scatter_kws={"alpha": .5})
        ax.set(xlabel=a, ylabel=b, title=f"{a} vs {b} (r={pr:.3f})")
        path = os.path.join(output_dir, f"bi_{a}_vs_{b}.png")
        plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
        charts.append(path)

    # numeric x categorical
    for n_col in num:
        for c_col in cat:
            groups = [g[n_col].dropna().values for _, g in df.groupby(c_col)]
            groups = [g for g in groups if len(g) >= 3]
            if len(groups) < 2:
                continue
            if len(groups) == 2:
                test, test_p = "Welch t", float(stats.ttest_ind(*groups, equal_var=False)[1])
            else:
                test, test_p = "ANOVA F", float(stats.f_oneway(*groups)[1])
            results["num_cat"].append({
                "numeric": n_col, "category": c_col,
                "test": test, "p_value": test_p,
            })
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.boxplot(data=df, x=c_col, y=n_col, ax=ax)
            ax.set_title(f"{n_col} by {c_col} ({test} p={test_p:.4f})")
            plt.xticks(rotation=30)
            path = os.path.join(output_dir, f"bi_{n_col}_by_{c_col}.png")
            plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
            charts.append(path)

    # categorical x categorical
    for a, b in combinations(cat, 2):
        ct = pd.crosstab(df[a], df[b])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        n = ct.sum().sum()
        v = float(np.sqrt(chi2 / (n * (min(ct.shape) - 1)))) if n else 0.0
        results["cat_cat"].append({
            "var_a": a, "var_b": b,
            "chi2": float(chi2), "dof": int(dof),
            "p_value": float(p), "cramers_v": v,
        })
        fig, ax = plt.subplots(figsize=(7, 4))
        ct.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title(f"{a} × {b} (χ²={chi2:.2f}, V={v:.3f})")
        plt.xticks(rotation=30)
        path = os.path.join(output_dir, f"bi_{a}_x_{b}.png")
        plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
        charts.append(path)

    return {"results": results, "charts": charts}


# ═════════════════════════════ MULTIVARIATE ═════════════════════════════
def multivariate_analysis(df: pd.DataFrame, output_dir: str) -> dict:
    """3+ numeric variables: correlation, PCA, clustering, VIF."""
    os.makedirs(output_dir, exist_ok=True)
    charts: list[str] = []
    results: dict = {}
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(num_cols) < 3:
        return {"error": "Multivariate analysis needs at least 3 numeric columns.",
                "charts": [], "results": {}}

    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
    except ImportError:
        return {"error": "scikit-learn not installed. Run: pip install scikit-learn",
                "charts": [], "results": {}}

    X = df[num_cols].dropna()
    if len(X) < 10:
        return {"error": "Need at least 10 complete rows for multivariate analysis.",
                "charts": [], "results": {}}
    Xs = StandardScaler().fit_transform(X)

    # 1. Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(X.corr(), annot=True, cmap="coolwarm", center=0, ax=ax, fmt=".2f")
    ax.set_title("Multivariate Correlation Matrix")
    path = os.path.join(output_dir, "multi_corr.png")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    charts.append(path)

    # 2. Pairplot
    try:
        pp = sns.pairplot(X.sample(min(500, len(X)), random_state=42), diag_kind="kde")
        path = os.path.join(output_dir, "multi_pairplot.png")
        pp.savefig(path); plt.close()
        charts.append(path)
    except Exception:
        pass

    # 3. PCA
    n_comp = min(5, len(num_cols))
    pca = PCA(n_components=n_comp)
    pcs = pca.fit_transform(Xs)
    results["pca_explained_variance"] = pca.explained_variance_ratio_.tolist()
    results["pca_cumulative"] = np.cumsum(pca.explained_variance_ratio_).tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(range(1, n_comp + 1), pca.explained_variance_ratio_)
    axes[0].set(xlabel="Principal Component", ylabel="Explained Variance", title="Scree Plot")
    axes[1].scatter(pcs[:, 0], pcs[:, 1], alpha=.5)
    axes[1].set(xlabel="PC1", ylabel="PC2", title="PCA 2D Projection")
    path = os.path.join(output_dir, "multi_pca.png")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    charts.append(path)

    # 4. K-Means elbow + clusters
    inertias = []
    for k in range(2, 8):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
        inertias.append(km.inertia_)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(2, 8), inertias, marker="o")
    ax.set(xlabel="k", ylabel="Inertia", title="K-Means Elbow Method")
    path = os.path.join(output_dir, "multi_elbow.png")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    charts.append(path)

    labels = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(Xs)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(pcs[:, 0], pcs[:, 1], c=labels, cmap="tab10", alpha=.6)
    ax.set_title("K-Means Clusters (k=3) on PCA Projection")
    path = os.path.join(output_dir, "multi_clusters.png")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    charts.append(path)

    # 5. VIF
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        results["vif"] = {
            col: float(variance_inflation_factor(Xs, i))
            for i, col in enumerate(num_cols)
        }
    except ImportError:
        results["vif"] = "statsmodels not installed"

    return {"results": results, "charts": charts}


# ═════════════════════════════ REPORT RENDERING ═════════════════════════════
def render_analysis_markdown(uni: dict, bi: dict, multi: dict) -> str:
    """Combine the three tiers into one markdown report."""
    md = ["# Complete Analysis Report", ""]

    md.append("## 1. Univariate Analysis")
    if uni.get("tables"):
        for col, stats_ in uni["tables"].items():
            md.append(f"### {col}")
            md.append("| statistic | value |\n|---|---|")
            for k, v in stats_.items():
                val = f"{v:.4f}" if isinstance(v, float) else str(v)
                md.append(f"| {k} | {val} |")
            md.append("")

    md.append("## 2. Bivariate Analysis")
    for section, rows in bi.get("results", {}).items():
        if not rows:
            continue
        md.append(f"### {section.replace('_', ' × ')}")
        md.append("| " + " | ".join(rows[0].keys()) + " |")
        md.append("|" + "|".join(["---"] * len(rows[0])) + "|")
        for r in rows:
            md.append("| " + " | ".join(
                f"{v:.4f}" if isinstance(v, float) else str(v) for v in r.values()
            ) + " |")
        md.append("")

    md.append("## 3. Multivariate Analysis")
    if multi.get("error"):
        md.append(f"_{multi['error']}_")
    else:
        res = multi.get("results", {})
        if "pca_explained_variance" in res:
            md.append("### PCA Explained Variance")
            for i, v in enumerate(res["pca_explained_variance"], 1):
                md.append(f"- PC{i}: {v:.4f} ({res['pca_cumulative'][i-1]:.4f} cumulative)")
            md.append("")
        if isinstance(res.get("vif"), dict):
            md.append("### VIF (Multicollinearity)")
            md.append("| feature | VIF |\n|---|---|")
            for k, v in res["vif"].items():
                md.append(f"| {k} | {v:.2f} |")
            md.append("")

    return "\n".join(md)
