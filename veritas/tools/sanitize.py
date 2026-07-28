"""HTML sanitisation for fetched pages — ADR-006.

Fetched web content is **untrusted input**. This module strips the vectors a model
would actually act on, before the content ever enters a context window. It is *not* a
full HTML parser; it is a deliberate allowlist applied to the text the model will see.

What we strip and why:

- ``<script>`` blocks — executable code, never evidence.
- ``<style>`` blocks — presentational, not factual.
- ``<!-- ... -->`` comments — a known prompt-injection vector (hidden instructions).
- ``<iframe>``, ``<object>``, ``<embed>``, ``<svg>`` — hidden content / data URIs.
- All remaining HTML tags — the model sees plain text, not markup it might interpret
  as structure. ``<b>`` and ``<a>`` carry no factual weight the snippet didn't already
  have; keeping them only invites a model to treat tags as meaningful.

What we do *not* claim:

- This does not defeat prompt injection in general. A page can still say "ignore
  previous instructions" in plain text. The defence is *layered*: stripping removes
  the structural hiding places (scripts, comments, hidden attributes), and the
  delimited preamble in :mod:`veritas.tools.fetch_page` marks the remainder as data,
  not instruction. The regression test in ``tests/test_injection.py`` asserts the
  *combined* defence holds against a realistic poisoned page. ADR-006 is honest about
  this: mitigated, not solved.
"""

from __future__ import annotations

import re

__all__ = ["DROP_BLOCK_RE", "TEXT_ONLY_TAG_RE", "strip_untrusted_html"]

# Block-level elements whose entire content we discard (not just the tags).
# ``<script>`` and ``<style>`` are non-factual; comments are an injection vector.
# ``<iframe>``/``<object>``/``<embed>``/``<svg>`` carry hidden content or data URIs.
# Note: ``<svg>`` is paired (``</svg>``); the others may be self-closing or paired.
_DROP_BLOCK_RE = re.compile(
    r"<(script|style|iframe|object|embed|svg)\b[^>]*>.*?</\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
# HTML comments — the classic "hidden instruction" vector.
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
# Any remaining tag (open, close, self-closing). We keep the text between tags.
_TAG_RE = re.compile(r"<[^>]+>")
# Runs of whitespace from stripped tags collapse to a single space.
_WS_RE = re.compile(r"[ \t\f\v]+")
# Three-or-more newlines collapse to two (paragraph break).
_BLANK_LINES_RE = re.compile(r"\n{3,}")

# Re-exported for tests / introspection.
DROP_BLOCK_RE = _DROP_BLOCK_RE
TEXT_ONLY_TAG_RE = _TAG_RE


def strip_untrusted_html(html: str) -> str:
    """Strip scripts, styles, comments, hidden media, and all tags from ``html``.

    Returns plain text suitable for placing in a delimited data block. Whitespace is
    collapsed but line structure is preserved so snippets remain readable.
    """
    text = _DROP_BLOCK_RE.sub(" ", html)
    text = _COMMENT_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", text)
    # Decode the handful of HTML entities that survive in plain text. We do not use
    # html.unescape because we do not want to resolve numeric entities that might
    # encode control characters — only the named ones a snippet would contain.
    for entity, char in _NAMED_ENTITIES:
        text = text.replace(entity, char)
    text = _WS_RE.sub(" ", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


_NAMED_ENTITIES: list[tuple[str, str]] = [
    ("&amp;", "&"),
    ("&lt;", "<"),
    ("&gt;", ">"),
    ("&quot;", '"'),
    ("&#39;", "'"),
    ("&nbsp;", " "),
]
