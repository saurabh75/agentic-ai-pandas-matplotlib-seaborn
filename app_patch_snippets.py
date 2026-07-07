"""Streamlit wiring for v9.7 features.

Paste each block into ``app.py`` at the marked location. Nothing here
is meant to be imported; treat it as a copy-paste reference.
"""

# ─────────────────────────────────────────────────────────────
# 1. Imports (add to the top of app.py)
# ─────────────────────────────────────────────────────────────
"""
import re
from datetime import datetime
import streamlit as st

from src.utils.cancel_token import get_cancel_token
from src.utils.export_report import build_analysis_zip
from src.services.type_converter import convert_column, format_report
from src.services.analysis_tiers import (
    univariate_analysis,
    bivariate_analysis,
    multivariate_analysis,
    render_analysis_markdown,
)
"""


# ─────────────────────────────────────────────────────────────
# 2. STOP BUTTON — render above st.chat_input
# ─────────────────────────────────────────────────────────────
"""
token = get_cancel_token()

if st.session_state.get("generating", False):
    if st.button("⏹️ Stop generation", type="primary", use_container_width=True):
        token.cancel()
        st.toast("Stopping generation…", icon="⏹️")
"""


# ─────────────────────────────────────────────────────────────
# 3. STREAM LOOP with cancel checks
# ─────────────────────────────────────────────────────────────
"""
if prompt := st.chat_input("Ask anything…"):
    token.reset()
    st.session_state["generating"] = True

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full = ""
        try:
            for chunk in pipeline.stream(prompt, cancel_token=token):
                full += chunk
                placeholder.markdown(full + "▌")
                if token.is_cancelled():
                    full += "\\n\\n_⏹️ Stopped by user._"
                    break
            placeholder.markdown(full)
        finally:
            st.session_state["generating"] = False
            token.reset()
"""


# ─────────────────────────────────────────────────────────────
# 4. TYPE CONVERSION intent handler (inside your intent router)
# ─────────────────────────────────────────────────────────────
"""
CONVERT_RE = re.compile(
    r"\\bconvert\\s+(?P<col>[\\w\\s]+?)\\s+(?:to|into|as)\\s+"
    r"(?P<type>int|integer|float|str|string|bool|boolean|datetime|date|category)\\b",
    re.IGNORECASE,
)

def try_handle_conversion(prompt: str) -> str | None:
    m = CONVERT_RE.search(prompt)
    if not m:
        return None
    df = st.session_state.get("active_df")
    if df is None:
        return "No active dataset uploaded."
    col = m.group("col").strip()
    tgt = m.group("type").strip()
    new_df, report = convert_column(df, col, tgt)
    st.session_state["active_df"] = new_df
    return format_report(report)
"""


# ─────────────────────────────────────────────────────────────
# 5. ANALYSIS TIERS + ZIP export
# ─────────────────────────────────────────────────────────────
"""
CHART_DIR = "charts"

if st.session_state.get("run_full_analysis"):
    df = st.session_state["active_df"]
    uni = univariate_analysis(df, CHART_DIR)
    bi = bivariate_analysis(df, CHART_DIR)
    multi = multivariate_analysis(df, CHART_DIR)

    report_md = render_analysis_markdown(uni, bi, multi)
    artifact = {
        "dataset": st.session_state.get("dataset_name", "dataset"),
        "timestamp": datetime.utcnow().isoformat(),
        "report_md": report_md,
        "stats": {"univariate": uni["tables"], "bivariate": bi["results"],
                  "multivariate": multi.get("results", {})},
        "charts": uni["charts"] + bi["charts"] + multi["charts"],
    }
    st.session_state["last_analysis_artifact"] = artifact

    st.markdown(report_md)
    for chart in artifact["charts"]:
        st.image(chart)

if artifact := st.session_state.get("last_analysis_artifact"):
    zip_bytes = build_analysis_zip(artifact)
    st.download_button(
        "📦 Download Full Report (ZIP)",
        data=zip_bytes,
        file_name=f"eda_report_{artifact['dataset']}_"
                  f"{datetime.now():%Y%m%d_%H%M}.zip",
        mime="application/zip",
    )
"""
