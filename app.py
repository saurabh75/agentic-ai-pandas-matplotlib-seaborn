"""
Streamlit UI for the Agentic RAG Agent (2026 Architecture Edition).

Implements all 6 stages of the Agentic RAG Architecture:
  1. Input & Orchestration
  2. Agent Loop (query rewrite, gap detection, multi-step retrieval)
  3. Knowledge & Memory Layer (hybrid BM25 + vector)
  4. Retrieval Quality Pipeline (dedup, freshness, context builder)
  5. Reasoning & Generation (citation builder, groundedness verifier)
  6. Evaluation & Feedback (latency, precision, confidence, user feedback)
"""

import sys
import time
from pathlib import Path

# =============================================================================
# Path setup
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
from dotenv import load_dotenv

env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(env_path)

APP_TITLE = os.getenv("APP_TITLE", "Agentic RAG Agent")
DOCUMENTS_DIR = PROJECT_ROOT / os.getenv("DOCUMENTS_DIR", "documents")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
SUPPORTED_EXTENSIONS = set(
    ext.strip().lower()
    for ext in os.getenv(
        "SUPPORTED_EXTENSIONS", ".pdf,.docx,.xlsx,.xls,.pptx,.ppt,.txt,.md,.csv"
    ).split(",")
)
CHROMA_PERSIST_DIR = PROJECT_ROOT / os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Imports
# =============================================================================
import streamlit as st

try:
    from src.agent.pipeline import get_pipeline, reset_pipeline, clear_session
    from src.services.document_processor import process_single_file, process_directory
    from src.services.vector_store import get_vector_store, clear_vector_store_cache
    from src.services.generation_engine import get_llm, clear_session
    from src.utils.file_utils import sanitize_filename, get_unique_storage_path
    from src.utils.chroma_utils import get_database_stats, clear_database
    from src.models.document import DatabaseStats
    from src.evaluation.evaluator import get_metric_store
    from src.logger import get_logger
    from src.exceptions import (
        DuplicateDocumentError,
        EmptyDocumentError,
        DocumentProcessingError,
        OllamaConnectionError,
    )
    IMPORTS_OK = True
except ImportError as e:
    IMPORTS_OK = False
    IMPORT_ERROR = str(e)
    if "src." not in str(e) and "No module named" in str(e):
        pkg = str(e).replace("No module named '", "").replace("'", "")
        IMPORT_ERROR = f"Missing package: {pkg}. Run: pip install -r requirements.txt"

# =============================================================================
# Page config
# =============================================================================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CSS
# =============================================================================
st.markdown("""
<style>
  .source-card {
    background: #f8f9fb;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    border-left: 4px solid #6c63ff;
  }
  .badge {
    display: inline-block;
    background: #e8e8f0;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 0.78em;
    color: #444;
    margin: 0 3px;
  }
  .badge-green  { background: #d4edda; color: #155724; }
  .badge-yellow { background: #fff3cd; color: #856404; }
  .badge-red    { background: #f8d7da; color: #721c24; }
  .step-card {
    background: #f0f4ff;
    border-radius: 6px;
    padding: 8px 14px;
    margin: 4px 0;
    border-left: 3px solid #4a6cf7;
    font-size: 0.88em;
  }
  .metric-box {
    text-align: center;
    padding: 10px 6px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    margin: 4px 0;
  }
  .metric-box h3 { margin: 0; font-size: 1.6em; }
  .metric-box p  { margin: 0; font-size: 0.82em; opacity: 0.9; }
  .eval-row { padding: 6px 0; border-bottom: 1px solid #eee; font-size: 0.9em; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Import guard
# =============================================================================
if not IMPORTS_OK:
    st.error(f"❌ Import Error: {IMPORT_ERROR}")
    st.markdown("""
    **Fix:**
    1. `pip install -r requirements.txt`
    2. Verify Ollama is running: `ollama serve`
    """)
    st.stop()

logger = get_logger("app")

# =============================================================================
# Session state
# =============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = f"streamlit_{int(time.time())}"
if "db_stats" not in st.session_state:
    st.session_state.db_stats = None

# =============================================================================
# Cached resources
# =============================================================================
@st.cache_resource(show_spinner=False)
def cached_vector_store():
    try:
        return get_vector_store()
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def cached_pipeline():
    try:
        return get_pipeline()
    except OllamaConnectionError:
        return None

# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.header("📊 Database Status")

    vectorstore = cached_vector_store()
    if vectorstore:
        try:
            stats = get_database_stats(vectorstore)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f'<div class="metric-box"><h3>{stats.document_count}</h3><p>Documents</p></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-box"><h3>{stats.chunk_count}</h3><p>Chunks</p></div>', unsafe_allow_html=True)
            if stats.last_updated:
                st.caption(f"🕐 Last updated: {stats.last_updated[:19]}")
        except Exception as e:
            st.error(f"❌ Stats error: {e}")
    else:
        st.warning("⚠️ Vector store unavailable")

    st.divider()

    # --- DB Controls ---
    st.header("🛠️ Controls")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear DB", use_container_width=True):
            if st.checkbox("Confirm clear?", key="confirm_clear"):
                try:
                    if vectorstore and clear_database(vectorstore):
                        clear_vector_store_cache()
                        reset_pipeline()
                        st.session_state.messages = []
                        st.success("✅ Cleared!")
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ {e}")
    with col2:
        if st.button("🔄 Re-index", use_container_width=True):
            with st.spinner("Re-indexing…"):
                try:
                    results = process_directory()
                    clear_vector_store_cache()
                    reset_pipeline()
                    st.success(f"✅ {len(results)} docs re-indexed")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ {e}")

    st.divider()

    # --- Upload ---
    st.header("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        f"Drop files here (max {MAX_FILE_SIZE_MB} MB each)",
        accept_multiple_files=True,
        type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
    )
    if uploaded_files:
        for uf in uploaded_files:
            if len(uf.getvalue()) > MAX_FILE_SIZE_MB * 1024 * 1024:
                st.error(f"❌ '{uf.name}' exceeds {MAX_FILE_SIZE_MB} MB")
                continue
            safe_name = sanitize_filename(uf.name)
            storage_path = get_unique_storage_path(DOCUMENTS_DIR, safe_name)
            storage_path.write_bytes(uf.getvalue())
            with st.spinner(f"Processing {uf.name}…"):
                try:
                    result = process_single_file(storage_path)
                    clear_vector_store_cache()
                    reset_pipeline()
                    pipeline = cached_pipeline()
                    if pipeline:
                        pipeline.invalidate_retrieval_cache()
                    st.success(f"✅ {result.filename} ({result.total_chunks} chunks)")
                except DuplicateDocumentError:
                    st.info(f"ℹ️ '{uf.name}' already indexed")
                except (EmptyDocumentError, DocumentProcessingError) as e:
                    st.error(f"❌ {e}")

    st.divider()

    # --- Conversation ---
    st.header("💬 Conversation")
    if st.button("🆕 New Chat", use_container_width=True):
        clear_session(st.session_state.session_id)
        st.session_state.session_id = f"streamlit_{int(time.time())}"
        st.session_state.messages = []
        st.rerun()
    st.caption(f"Session: `{st.session_state.session_id[:22]}…`")

    st.divider()

    # --- Eval Summary ---
    st.header("📈 Session Metrics")
    summary = get_metric_store().summary()
    if summary:
        st.metric("Queries", int(summary.get("total_queries", 0)))
        st.metric("Avg Latency", f"{summary.get('avg_latency_ms', 0):.0f} ms")
        st.metric("Avg Confidence", f"{summary.get('avg_confidence', 0):.2f}")
        st.metric("Avg Groundedness", f"{summary.get('avg_groundedness', 0):.2f}")
        st.metric("Avg Iterations", f"{summary.get('avg_iterations', 0):.1f}")
    else:
        st.caption("No queries yet.")

# =============================================================================
# Main area
# =============================================================================
st.title(f"🤖 {APP_TITLE}")
st.caption("💯 Local & Private | 🔄 Agentic Loop | 🔍 Hybrid BM25+Vector | ✅ Groundedness Verified")

# Health check
pipeline = cached_pipeline()
if pipeline is None:
    st.error("🔴 Cannot connect to Ollama. Run: `ollama serve`", icon="❌")
    st.stop()
else:
    st.success("🟢 Ollama connected — Agentic RAG ready", icon="✅")

# --- Chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            # Confidence & groundedness badges
            conf = msg.get("confidence", 0)
            gnd = msg.get("groundedness", 0)
            strategy = msg.get("strategy", "")
            iters = msg.get("iterations", 1)

            conf_cls = "badge-green" if conf >= 0.7 else ("badge-yellow" if conf >= 0.4 else "badge-red")
            gnd_cls  = "badge-green" if gnd  >= 0.7 else ("badge-yellow" if gnd  >= 0.4 else "badge-red")

            st.markdown(
                f'<span class="{conf_cls} badge">Confidence {conf:.0%}</span>'
                f'<span class="{gnd_cls} badge">Grounded {gnd:.0%}</span>'
                f'<span class="badge">🔍 {strategy.upper()}</span>'
                f'<span class="badge">🔁 {iters} iter</span>',
                unsafe_allow_html=True,
            )

            # Citations
            if msg.get("citations"):
                with st.expander(f"📚 Citations ({len(msg['citations'])})"):
                    for cit in msg["citations"]:
                        page = f", p. {cit['page_number']}" if cit.get("page_number") else ""
                        rel = cit.get("relevance", 0)
                        st.markdown(
                            f'<div class="source-card">'
                            f'<strong>[{cit["index"]}] {cit["filename"]}{page}</strong>'
                            f'<span class="badge">Relevance {rel:.2f}</span>'
                            + (f'<br><em style="color:#666;font-size:0.87em">{cit["excerpt"]}</em>' if cit.get("excerpt") else "")
                            + "</div>",
                            unsafe_allow_html=True,
                        )

            # Agent reasoning trace
            if msg.get("agent_steps"):
                with st.expander(f"🧠 Agent Reasoning ({len(msg['agent_steps'])} steps)"):
                    for step in msg["agent_steps"]:
                        st.markdown(
                            f'<div class="step-card">'
                            f'<strong>Step {step["iteration"]+1} · {step["step_name"]}</strong><br>'
                            f'<span style="color:#555">↳ {step["output_summary"]}</span>'
                            "</div>",
                            unsafe_allow_html=True,
                        )

# --- Chat input ---
if prompt := st.chat_input("Ask about your documents…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        full_answer = ""
        response = None
        status = st.status("🤖 Agentic RAG running…", expanded=False)

        try:
            with status:
                st.write("**Stage 1:** Orchestrating intent & policy…")
                st.write("**Stage 2:** Checking data-analysis intent…")

            # Patch v5.1: true single-pass streaming.
            # stream_answer yields str tokens; the LAST item is a dict
            # {"final": AgenticRAGResponse} carrying metadata (citations, chart,
            # steps, metrics). No second LLM call is made.
            for chunk in pipeline.stream_answer(prompt, st.session_state.session_id):
                if isinstance(chunk, dict) and "final" in chunk:
                    response = chunk["final"]
                    break
                full_answer += chunk
                answer_placeholder.markdown(full_answer + "▌")
            answer_placeholder.markdown(full_answer)

            # Render chart if the pipeline produced one.
            if response is not None and getattr(response, "chart_path", None):
                try:
                    st.image(response.chart_path,
                             caption=response.chart_caption or "Chart",
                             use_column_width=True)
                except Exception as _e:
                    st.warning(f"Chart could not be displayed: {_e}")
            status.update(label="✅ Done", state="complete", expanded=False)
            if response is None:
                st.session_state.messages.append({
                    "role": "assistant", "content": full_answer,
                    "citations": [], "agent_steps": [],
                    "confidence": 0.0, "groundedness": 0.0,
                    "strategy": "n/a", "iterations": 0,
                })
                st.stop()

            metrics = response.metrics
            conf = response.confidence
            gnd = response.groundedness_score
            strategy = metrics.strategy_used
            iters = metrics.iterations_used

            conf_cls = "badge-green" if conf >= 0.7 else ("badge-yellow" if conf >= 0.4 else "badge-red")
            gnd_cls  = "badge-green" if gnd  >= 0.7 else ("badge-yellow" if gnd  >= 0.4 else "badge-red")

            st.markdown(
                f'<span class="{conf_cls} badge">Confidence {conf:.0%}</span>'
                f'<span class="{gnd_cls} badge">Grounded {gnd:.0%}</span>'
                f'<span class="badge">🔍 {strategy.upper()}</span>'
                f'<span class="badge">🔁 {iters} iter</span>'
                f'<span class="badge">⏱ {metrics.latency_ms:.0f} ms</span>'
                f'<span class="badge">~{metrics.token_estimate} tok</span>',
                unsafe_allow_html=True,
            )

            # Store in chat history
            cit_dicts = [c.model_dump() for c in response.citations]
            step_dicts = [s.model_dump() for s in response.agent_steps]

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer,
                "citations": cit_dicts,
                "agent_steps": step_dicts,
                "confidence": conf,
                "groundedness": gnd,
                "strategy": strategy,
                "iterations": iters,
            })

            # Citations inline
            if response.citations:
                with st.expander(f"📚 Citations ({len(response.citations)})"):
                    for cit in response.citations:
                        page = f", p. {cit.page_number}" if cit.page_number else ""
                        st.markdown(
                            f'<div class="source-card">'
                            f'<strong>[{cit.index}] {cit.filename}{page}</strong>'
                            f'<span class="badge">Relevance {cit.relevance:.2f}</span>'
                            + (f'<br><em style="color:#666;font-size:0.87em">{cit.excerpt}</em>' if cit.excerpt else "")
                            + "</div>",
                            unsafe_allow_html=True,
                        )

            # Agent reasoning trace
            if response.agent_steps:
                with st.expander(f"🧠 Agent Reasoning ({len(response.agent_steps)} steps)"):
                    for step in response.agent_steps:
                        st.markdown(
                            f'<div class="step-card">'
                            f'<strong>Step {step.iteration+1} · {step.step_name}</strong><br>'
                            f'<span style="color:#555">↳ {step.output_summary}</span>'
                            "</div>",
                            unsafe_allow_html=True,
                        )

        except OllamaConnectionError:
            st.error("🔴 Ollama disconnected. Run: `ollama serve`")
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            st.error(f"❌ {e}")
            st.info("Try refreshing or checking Ollama.")
