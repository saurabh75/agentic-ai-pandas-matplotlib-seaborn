"""Bundle an analysis artifact (report + charts + raw stats) into a zip."""
from __future__ import annotations

import io
import json
import os
import zipfile
from datetime import datetime


def build_analysis_zip(artifact: dict) -> bytes:
    """Package an analysis artifact into a downloadable zip.

    ``artifact`` shape (produced by the pipeline):
        {
            "dataset": "health_insurance.csv",
            "timestamp": "2026-07-07T12:34:56",
            "report_md": "<markdown>",
            "stats": {...},              # JSON-serialisable
            "charts": ["/tmp/.../a.png", ...],
        }
    """
    buf = io.BytesIO()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            f"report_{stamp}.md",
            artifact.get("report_md", "# (no report body)"),
        )
        zf.writestr(
            f"stats_{stamp}.json",
            json.dumps(artifact.get("stats", {}), indent=2, default=str),
        )
        for chart_path in artifact.get("charts", []):
            if chart_path and os.path.exists(chart_path):
                zf.write(chart_path, arcname=f"charts/{os.path.basename(chart_path)}")
        zf.writestr(
            "README.txt",
            "Analysis Report\n"
            f"Dataset:   {artifact.get('dataset', 'unknown')}\n"
            f"Generated: {artifact.get('timestamp', stamp)}\n\n"
            "Contents:\n"
            "  report_*.md   Human-readable markdown report\n"
            "  stats_*.json  Raw statistics used to build the report\n"
            "  charts/       All generated PNG charts\n",
        )

    return buf.getvalue()
