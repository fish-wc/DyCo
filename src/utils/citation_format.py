"""Citation format normalization utilities.

Normalizes various legacy / model-generated citation markers into the canonical
XML form: <cite ref="n"/>.

This is especially useful for smaller models that may copy the knowledge marker
style (e.g. <REF-1>, [REF-1], <ref>1</ref>, [1]) instead of emitting <cite/>.
"""

from __future__ import annotations

import re


_REF_ANGLE_RE = re.compile(r"<\s*REF-(\d+)\s*/?\s*>", re.IGNORECASE)
_REF_TAG_RE = re.compile(r"<\s*ref\s*>\s*(\d+)\s*<\s*/\s*ref\s*>", re.IGNORECASE)
_REF_BRACKET_RE = re.compile(r"\[\s*REF-(\d+)\s*\]", re.IGNORECASE)

# Convert bare numeric bracket citations like "...文本[12]。" into cite.
# We constrain it to inline usage to avoid breaking list numbering.
_NUM_BRACKET_INLINE_RE = re.compile(
    r"(?<=[\u4e00-\u9fffA-Za-z0-9])\[(\d+)\](?=[。！？；，,\.\s])"
)


def normalize_citation_markers(text: str) -> str:
    """Normalize various citation markers into <cite ref="n"/>.

    This function is intentionally conservative: it focuses on known patterns
    that appear in this project (<REF-n>, <ref>n</ref>, [REF-n], [n]).
    """
    if not text:
        return text

    text = _REF_ANGLE_RE.sub(r'<cite ref="\1"/>', text)
    text = _REF_TAG_RE.sub(r'<cite ref="\1"/>', text)
    text = _REF_BRACKET_RE.sub(r'<cite ref="\1"/>', text)
    text = _NUM_BRACKET_INLINE_RE.sub(r'<cite ref="\1"/>', text)
    return text
