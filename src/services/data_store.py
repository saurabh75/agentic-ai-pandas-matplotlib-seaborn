"""
Tabular data store (Patch v5).

CSV / XLSX files uploaded to the RAG app are also persisted as parquet so the
analysis engine can run real pandas operations on the FULL dataset instead of
LLM-summarising a text-chunked version.

- Parquet files live in ./data_store/<hash>.parquet
- Metadata (name, columns, rows) lives in ./data_store/registry.json
- load_df(name) is LRU-cached so repeat queries hit RAM.

Kept intentionally small — no server, no schema evolution, no lockfile.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)

DATA_STORE_DIR = Path(os.getenv("DATA_STORE_DIR", "data_store"))
DATA_STORE_DIR.mkdir(parents=True, exist_ok=True)
_REGISTRY = DATA_STORE_DIR / "registry.json"
MAX_ROWS = int(os.getenv("MAX_ANALYSIS_ROWS", "1000000"))


def _load_registry() -> Dict[str, Dict[str, Any]]:
    if not _REGISTRY.exists():
        return {}
    try:
        return json.loads(_REGISTRY.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_registry(reg: Dict[str, Dict[str, Any]]) -> None:
    _REGISTRY.write_text(json.dumps(reg, indent=2), encoding="utf-8")


def save_tabular(file_path: Path, document_hash: str) -> Optional[Dict[str, Any]]:
    """Read a CSV/XLSX into a DataFrame, persist as parquet, register it."""
    ext = file_path.suffix.lower()
    try:
        if ext == ".csv":
            df = pd.read_csv(file_path)
        elif ext in {".xlsx", ".xls"}:
            df = pd.read_excel(file_path)
        else:
            return None
    except Exception as e:
        logger.warning(f"[data_store] Failed to read {file_path.name}: {e}")
        return None

    if len(df) > MAX_ROWS:
        logger.warning(
            f"[data_store] {file_path.name}: {len(df)} rows > MAX_ANALYSIS_ROWS "
            f"({MAX_ROWS}); truncating."
        )
        df = df.head(MAX_ROWS)

    parquet_path = DATA_STORE_DIR / f"{document_hash}.parquet"
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception as e:
        logger.warning(f"[data_store] Parquet write failed for {file_path.name}: {e}")
        return None

    entry = {
        "name": file_path.name,
        "hash": document_hash,
        "parquet": str(parquet_path.resolve()),
        "columns": list(df.columns.astype(str)),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "rows": int(len(df)),
    }
    reg = _load_registry()
    reg[file_path.name] = entry
    _save_registry(reg)
    load_df.cache_clear()
    logger.info(f"[data_store] Registered {file_path.name}: {len(df)} rows × {len(df.columns)} cols")
    return entry


def list_files() -> List[Dict[str, Any]]:
    return list(_load_registry().values())


def get_entry(name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Get a single file entry. If name is None, returns the most-recently added file."""
    reg = _load_registry()
    if not reg:
        return None
    if name:
        # exact match, then case-insensitive contains
        if name in reg:
            return reg[name]
        low = name.lower()
        for k, v in reg.items():
            if low in k.lower():
                return v
        return None
    # newest = last inserted
    return list(reg.values())[-1]


@lru_cache(maxsize=8)
def load_df(name: str) -> Optional[pd.DataFrame]:
    entry = get_entry(name)
    if not entry:
        return None
    try:
        return pd.read_parquet(entry["parquet"])
    except Exception as e:
        logger.warning(f"[data_store] Failed to load parquet {name}: {e}")
        return None


def schema_summary(entry: Dict[str, Any]) -> str:
    """One-chunk text summary of a tabular file for the RAG index."""
    lines = [
        f"Tabular dataset: {entry['name']}",
        f"Rows: {entry['rows']}  Columns: {len(entry['columns'])}",
        "",
        "Schema:",
    ]
    for col in entry["columns"]:
        dt = entry["dtypes"].get(col, "?")
        lines.append(f"  - {col} ({dt})")
    lines.append("")
    lines.append(
        "This dataset is available for direct analysis (describe, nulls, "
        "duplicates, value_counts, correlation, t-test, ANOVA, chi-square, "
        "regression, and charting)."
    )
    return "\n".join(lines)


def clear_registry() -> None:
    """Called when the vector DB is cleared."""
    reg = _load_registry()
    for entry in reg.values():
        try:
            Path(entry["parquet"]).unlink(missing_ok=True)
        except Exception:
            pass
    if _REGISTRY.exists():
        _REGISTRY.unlink()
    load_df.cache_clear()
