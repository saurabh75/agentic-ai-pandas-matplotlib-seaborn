"""Generate a File Intelligence card at ingest time. Cached to disk."""
from __future__ import annotations
import os, json, hashlib
from pathlib import Path
from typing import Any

from src.services.domain_classifier import classify
from src.services.persona_manager import _ext, TABULAR_EXTS, DOC_EXTS

CACHE_DIR = Path(os.getenv("DATA_STORE_DIR", "data_store")) / "profiles"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(file_path: str) -> Path:
    h = hashlib.md5(file_path.encode()).hexdigest()[:12]
    return CACHE_DIR / f"{Path(file_path).stem}_{h}.profile.json"


def _human_size(n: int) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024: return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def _profile_tabular(path: str) -> dict:
    import pandas as pd
    ext = _ext(path)
    df = pd.read_csv(path) if ext == ".csv" else pd.read_excel(path)
    num_cols = df.select_dtypes("number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "missing_total": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "domain": "data",
        "summary": f"Tabular dataset with {df.shape[0]} rows × {df.shape[1]} columns. "
                   f"{len(num_cols)} numeric, {len(cat_cols)} categorical.",
        "main_topics": cat_cols[:5] + num_cols[:5],
        "suggested_questions": [
            "Run full EDA on this dataset",
            "Show missing values and duplicates",
            "Detect outliers in numeric columns",
            "Show correlation heatmap",
            f"Describe the distribution of {num_cols[0]}" if num_cols else "Summarize the dataset",
        ],
    }


def _profile_document(path: str, sample_text: str, ollama_client=None, model=None) -> dict:
    domain = classify(sample_text, ollama_client, model)
    pages = sample_text.count("\f") + 1
    words = len(sample_text.split())
    # naive main-topic extraction (top nouns via frequency)
    import re
    from collections import Counter
    tokens = [w.lower() for w in re.findall(r"[A-Za-z]{4,}", sample_text)]
    stop = {"this", "that", "with", "from", "have", "been", "were", "which", "their", "would", "there", "about"}
    freq = Counter(t for t in tokens if t not in stop).most_common(8)
    topics = [w for w, _ in freq]
    return {
        "pages_est": pages,
        "word_count": words,
        "domain": domain,
        "summary": sample_text[:400].replace("\n", " ").strip() + ("..." if len(sample_text) > 400 else ""),
        "main_topics": topics,
        "suggested_questions": [
            "Give me an executive summary of this document",
            "Extract key entities and action items",
            "What are the main risks and recommendations?",
            f"Summarize the section about {topics[0]}" if topics else "Summarize the document",
            "List the most important findings",
        ],
    }


def profile_file(path: str, sample_text: str = "", ollama_client=None, model=None,
                 use_cache: bool = True) -> dict:
    """Build (or load cached) profile. sample_text used for doc classification."""
    cache = _cache_path(path)
    if use_cache and cache.exists():
        try:
            return json.loads(cache.read_text())
        except Exception:
            pass

    ext = _ext(path)
    size = os.path.getsize(path) if os.path.exists(path) else 0
    base: dict[str, Any] = {
        "file_name": Path(path).name,
        "file_type": ext.lstrip("."),
        "size_bytes": size,
        "size_human": _human_size(size),
    }
    try:
        if ext in TABULAR_EXTS:
            base.update(_profile_tabular(path))
        elif ext in DOC_EXTS or sample_text:
            base.update(_profile_document(path, sample_text, ollama_client, model))
        else:
            base.update({"domain": "general", "summary": "Unrecognized file type.",
                         "main_topics": [], "suggested_questions": []})
    except Exception as e:
        base["profile_error"] = str(e)
        base.setdefault("domain", "general")

    try:
        cache.write_text(json.dumps(base, indent=2, default=str))
    except Exception:
        pass
    return base
