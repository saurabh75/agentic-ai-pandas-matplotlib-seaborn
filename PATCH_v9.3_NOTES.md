# Patch v9.3 — Senior-Expert Personas ACTUALLY wired in

## The bug
`get_persona_prompt()` and `persona_manager` existed since v8 but **nothing
called them**. The LLM always saw the generic prompt:

    "You are a precise research assistant..."

So the agent sounded flat — no persona, no framework, no expert framing.

## Fix
1. `src/agent/prompts.py` — rewritten with three heavyweight senior-expert
   prompts (20-year framing, methodology checklist, tone rules, response
   format, hallucination guard):
   - **Senior Data Scientist** — tabular files (.csv/.xlsx/.parquet)
   - **Senior {domain} Expert** — documents; domain auto-classified
     (finance / healthcare / legal / technical / research / hr / business /
     insurance / general)
   - **Senior Multi-Domain Analyst** — mixed tabular + docs
2. `src/services/generation_engine.py` — `generate()` and `stream()` now
   accept `system_override=...` and use it in place of the generic prompt.
3. `src/agent/pipeline.py` — new `_resolve_persona_system()` inspects the
   filenames in the retrieved chunks, picks the persona via
   `persona_manager.detect_persona`, renders the full system prompt with
   context embedded, and passes it to both the non-streaming and
   streaming generator paths. A new **"Persona"** step is appended to
   `agent_steps` so the UI (Agent Steps expander) shows which senior
   expert answered.

## Enforced response shape (every reply)
```
Persona
Executive Summary
Analysis           (with [Source N] inline)
Key Findings
Recommendations    (with owner-role + expected impact)
Sources
Follow-up Questions
```
Every reply also opens with the fixed line, e.g.:
`Speaking as a Senior Data Scientist with 20 years of experience —`

## Hallucination guard baked into every persona
- Answer ONLY from CONTEXT + prior computed results.
- If insufficient → exact refusal sentence + list of missing inputs.
- Every number/name/quote MUST carry a [Source N] tag.
- Outside knowledge only under an explicit `General background:` label,
  max 2 sentences.

## Install
Unzip over the project (overwrites `src/agent/prompts.py`,
`src/services/generation_engine.py`, `src/agent/pipeline.py`, and installs
`src/services/persona_manager.py` if it wasn't already present). No new
Python deps. Restart Streamlit.

## Try it
- Upload `health_insurance.csv` and ask `describe the churn drivers` →
  reply opens with `Speaking as a Senior Data Scientist with 20 years…`
  and follows the 7-section format with real numbers from the pandas
  computation.
- Upload a PDF policy doc and ask `what are the termination clauses` →
  `Speaking as a Senior Legal Expert with 20 years…`
- Upload both → `Speaking as a Senior Multi-Domain Analyst with 20 years…`
  and cross-references the CSV against the PDF.

## Files in this zip
- `src/agent/prompts.py`
- `src/agent/pipeline.py`
- `src/services/generation_engine.py`
- `src/services/persona_manager.py`
- `PATCH_v9.3_NOTES.md`
