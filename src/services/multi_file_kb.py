"""Session-level multi-file knowledge base coordinator."""
from __future__ import annotations
from typing import Any
from collections import defaultdict


class MultiFileKB:
    """Tracks active uploaded files and enables cross-file retrieval grouping."""

    def __init__(self) -> None:
        self.files: dict[str, dict[str, Any]] = {}   # filename -> profile

    def register(self, filename: str, profile: dict) -> None:
        self.files[filename] = profile

    def remove(self, filename: str) -> None:
        self.files.pop(filename, None)

    def list_files(self) -> list[str]:
        return list(self.files.keys())

    def domains(self) -> dict[str, str]:
        return {f: p.get("domain", "general") for f, p in self.files.items()}

    def is_multi(self) -> bool:
        return len(self.files) > 1

    def group_chunks_by_source(self, chunks: list[dict]) -> dict[str, list[dict]]:
        """chunks: [{content, metadata:{source,...}, score}]"""
        grouped: dict[str, list[dict]] = defaultdict(list)
        for c in chunks:
            src = (c.get("metadata") or {}).get("source", "unknown")
            grouped[src].append(c)
        return dict(grouped)

    def detect_overlap(self, chunks: list[dict]) -> list[str]:
        """Return topics/entities appearing in >1 file (naive token overlap)."""
        import re
        from collections import Counter
        per_file: dict[str, set] = {}
        for c in chunks:
            src = (c.get("metadata") or {}).get("source", "?")
            toks = set(re.findall(r"[A-Za-z]{5,}", (c.get("content") or "").lower()))
            per_file.setdefault(src, set()).update(toks)
        if len(per_file) < 2:
            return []
        counter: Counter = Counter()
        for s in per_file.values():
            counter.update(s)
        return [t for t, n in counter.most_common(20) if n >= 2][:10]

    def consolidated_context(self, chunks: list[dict], max_per_file: int = 3) -> str:
        grouped = self.group_chunks_by_source(chunks)
        parts = []
        for src, items in grouped.items():
            parts.append(f"\n=== SOURCE: {src} ===")
            for i, c in enumerate(items[:max_per_file], 1):
                parts.append(f"[chunk {i}] {c.get('content', '')[:800]}")
        return "\n".join(parts)
