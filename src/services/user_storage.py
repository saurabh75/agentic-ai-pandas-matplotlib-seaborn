"""Per-user filesystem + vector collection paths. All writes MUST go through here."""
from __future__ import annotations
import os, shutil
from pathlib import Path
from src.services.user_context import UserContext

_ROOT = Path(os.getenv("DATA_STORE_DIR", "data_store"))
_CHARTS = Path(os.getenv("CHARTS_DIR", "charts"))
_UPLOADS = Path(os.getenv("UPLOADS_DIR", "uploads"))


def vector_collection_name(user: UserContext) -> str:
    return f"rag_chunks_{user.safe_id}"


def data_dir(user: UserContext) -> Path:
    p = _ROOT / user.safe_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def profiles_dir(user: UserContext) -> Path:
    p = data_dir(user) / "profiles"
    p.mkdir(parents=True, exist_ok=True)
    return p


def charts_dir(user: UserContext) -> Path:
    p = _CHARTS / user.safe_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def uploads_dir(user: UserContext) -> Path:
    p = _UPLOADS / user.safe_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def wipe_user(user: UserContext) -> dict:
    """Delete EVERYTHING belonging to this user. Other users untouched."""
    removed = {}
    for label, p in [("data", _ROOT / user.safe_id),
                     ("charts", _CHARTS / user.safe_id),
                     ("uploads", _UPLOADS / user.safe_id)]:
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
            removed[label] = str(p)
    # recreate empty dirs
    data_dir(user); charts_dir(user); uploads_dir(user)
    return removed
