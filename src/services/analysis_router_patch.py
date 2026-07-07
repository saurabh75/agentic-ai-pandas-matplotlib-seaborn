"""Router patterns added in v9.7.

Merge these into your existing ``analysis_router.py``'s ``_PATTERNS`` /
``_SINGLE_TEST_PATTERNS`` dicts. Kept as a separate file so the patch is
easy to diff.
"""

# High-priority intents (checked before generic RAG routing).
_TYPE_CONVERT_PATTERNS = [
    r"\bconvert\s+(?P<col>[\w\s]+?)\s+(?:to|into|as)\s+(?P<type>int|integer|float|str|string|bool|boolean|datetime|date|category)\b",
    r"\bchange\s+(?P<col>[\w\s]+?)\s+(?:dtype|type|column type)\s+to\s+(?P<type>\w+)\b",
    r"\bcast\s+(?P<col>[\w\s]+?)\s+as\s+(?P<type>\w+)\b",
    r"\b(?P<col>[\w\s]+?)\s+as\s+(?P<type>int|integer|float|str|string|bool|datetime|category)\b",
]

_ANALYSIS_TIER_PATTERNS = {
    "univariate": r"\b(univariate|single[- ]variable|one[- ]variable|per[- ]column)\s*(analysis|eda)?\b",
    "bivariate": r"\b(bivariate|pairwise|two[- ]variable|pair(?:wise)?)\s+(analysis|eda|relationships?)\b",
    "multivariate": r"\b(multivariate|multi[- ]variable|pca|clustering|dimensionality)\s*(analysis|eda)?\b",
}

_FOLLOWUP_PATTERNS = [
    r"\b(write|generate|give|draft)\s+(?:a\s+)?(conclusion|summary|takeaways?|recommendations?|report|writeup|write-up)\b",
    r"\b(interpret|explain|summari[sz]e|elaborate on)\s+(?:the\s+)?(above|previous|last|these|those|results?|findings?|report|eda)\b",
    r"^\s*(conclusion|summary|takeaways?|next steps?|so what|implications?)\s*\??\s*$",
    r"\bwhat (does|do) (this|these|the results?) (mean|imply|suggest)\b",
]

# Full EDA triggers (from v9.2, kept for reference).
_FULL_EDA_PATTERNS = [
    r"\b(full|complete|professional|thorough|whole)\s+(eda|analysis|report|workflow)\b",
    r"\b(do|run|perform|execute|generate)\s+(all|everything|full|the\s+full|complete)\b",
    r"^\s*(do|run)\s+all\s*$",
    r"^\s*eda\s*$",
]
