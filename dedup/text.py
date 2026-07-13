"""Normalization of Russian text for deduplication."""
from __future__ import annotations

import re

from razdel import tokenize

_URL_RE = re.compile(r"https?://\S+|t\.me/\S+|@[\w_]+")
_WS_RE = re.compile(r"\s+")
_NONWORD_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)

_DIGEST_MARKERS = re.compile(
    r"главные новости|кратко:|дайджест|итоги дня|важное за|что случилось"
    r"|больше новостей к этому часу|прямо сейчас в эфире|в программе «|слушайте в эфире",
    flags=re.IGNORECASE,
)
_BULLET_RE = re.compile(r"[▪•▸►\-🟢🔴🔵🟡]\s?")


def clean(text: str) -> str:
    """Lowercase, strip URLs/mentions/punctuation, collapse whitespace."""
    t = (text or "").lower()
    t = _URL_RE.sub(" ", t)
    t = _NONWORD_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


def word_tokens(text: str) -> list[str]:
    """Word tokens for MinHash shingles (on cleaned text)."""
    return [tok.text for tok in tokenize(clean(text))]


def is_digest(text: str) -> bool:
    """True if the post is a digest/summary of multiple events."""
    if _DIGEST_MARKERS.search(text):
        return True
    if len(_BULLET_RE.findall(text)) >= 3 and len(text) > 250:
        return True
    return False
