"""Deterministic Unicode tokenization for lexical retrieval."""

import re
import unicodedata

_TERMS = re.compile(r"[^\W_]+", re.UNICODE)
_MAX_QUERY_TERMS = 128


def terms(text: str) -> list[str]:
    """Return unique word and number tokens in their first-occurrence order."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return list(dict.fromkeys(_TERMS.findall(normalized)))
