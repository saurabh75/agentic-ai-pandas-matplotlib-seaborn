"""Enforces the 7-section response format."""
from __future__ import annotations
from typing import Any

NO_INFO = "This information is not available in the uploaded documents."


def build_system_suffix(persona_info: dict, active_files: list[str]) -> str:
    files_str = ", ".join(active_files) if active_files else "none"
    return f"""

## ACTIVE PERSONA
{persona_info['label']} — {persona_info.get('reason','')}
Files in scope: {files_str}

## REQUIRED RESPONSE STRUCTURE
Format EVERY answer with these exact markdown sections:

### 1. Active Expert Persona
State the persona + one-line justification.

### 2. Executive Summary
2-4 sentences.

### 3. Detailed Analysis
Depth appropriate to the question. Reference retrieved context.

### 4. Key Findings
Bulleted list.

### 5. Recommendations
Bulleted list of actionable next steps.

### 6. Source References
List file names and chunk ids used. If none, state: "{NO_INFO}"

### 7. Suggested Follow-up Questions
3 concise follow-ups the user could ask next.

## HALLUCINATION GUARD
- Use ONLY the provided context.
- If context does not contain the answer, output exactly: "{NO_INFO}" in section 2 and stop.
"""


def fallback_no_info(persona_info: dict, files: list[str]) -> str:
    src = ", ".join(files) if files else "no files"
    return f"""### 1. Active Expert Persona
{persona_info['label']} — {persona_info.get('reason','')}

### 2. Executive Summary
{NO_INFO}

### 3. Detailed Analysis
The retrieval step did not return sufficient grounded context to answer this question.

### 4. Key Findings
- No matching content in uploaded files.

### 5. Recommendations
- Upload a document containing this topic, or rephrase the question.

### 6. Source References
Searched: {src}. No matching chunks above the confidence threshold.

### 7. Suggested Follow-up Questions
- What topics do the uploaded files cover?
- Give me an executive summary of {files[0] if files else 'the document'}.
- List key entities in the uploaded files.
"""
