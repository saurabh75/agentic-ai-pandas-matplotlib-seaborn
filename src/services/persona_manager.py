"""Resolves the active expert persona from file type(s) in scope."""
from __future__ import annotations
from pathlib import Path
from typing import Iterable

TABULAR_EXTS = {".csv", ".xlsx", ".xls", ".parquet"}
DOC_EXTS = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".txt", ".md", ".markdown", ".rtf", ".html"}

PERSONA_REGISTRY = {
    "data_scientist": {
        "label": "Senior Data Scientist (20 yrs)",
        "exts": TABULAR_EXTS,
    },
    "domain_expert": {
        "label": "Senior Domain Expert (20 yrs)",
        "exts": DOC_EXTS,
    },
    "multi_domain": {
        "label": "Senior Multi-Domain Analyst (20 yrs)",
        "exts": set(),
    },
}


def _ext(name: str) -> str:
    return Path(name).suffix.lower()


def detect_persona(files: Iterable[str], domains: dict[str, str] | None = None) -> dict:
    """Return {persona, label, domain, reason, files}.
    files: iterable of file names/paths currently in scope.
    domains: optional map filename -> classified domain (for docs).
    """
    files = [f for f in files if f]
    if not files:
        return {"persona": "domain_expert", "label": "Senior Domain Expert (20 yrs)",
                "domain": "general", "reason": "No file in scope; defaulting to generic expert.",
                "files": []}

    exts = {_ext(f) for f in files}
    tabular = exts & TABULAR_EXTS
    docs = exts & DOC_EXTS

    if tabular and not docs:
        return {"persona": "data_scientist",
                "label": PERSONA_REGISTRY["data_scientist"]["label"],
                "domain": "data", "files": files,
                "reason": f"Tabular file(s) detected ({', '.join(sorted(tabular))}) → Data Scientist persona."}

    if docs and not tabular:
        # pick dominant domain if provided
        dom = "general"
        if domains:
            counts: dict[str, int] = {}
            for f in files:
                d = domains.get(f, "general")
                counts[d] = counts.get(d, 0) + 1
            dom = max(counts, key=counts.get)
        return {"persona": "domain_expert",
                "label": f"Senior {dom.title()} Expert (20 yrs)",
                "domain": dom, "files": files,
                "reason": f"Document file(s) detected; classified domain = {dom}."}

    # mixed
    return {"persona": "multi_domain",
            "label": PERSONA_REGISTRY["multi_domain"]["label"],
            "domain": "mixed", "files": files,
            "reason": "Mixed tabular + document files in scope → Multi-Domain Analyst."}
