"""Нормализация русского текста для дедупликации."""
from __future__ import annotations

import re

from razdel import tokenize

_URL_RE = re.compile(r"https?://\S+|t\.me/\S+|@[\w_]+")
_WS_RE = re.compile(r"\s+")
_NONWORD_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


def clean(text: str) -> str:
    """Lowercase, убрать URL/упоминания/пунктуацию, схлопнуть пробелы."""
    t = (text or "").lower()
    t = _URL_RE.sub(" ", t)
    t = _NONWORD_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


def word_tokens(text: str) -> list[str]:
    """Токены-слова для шинглов MinHash (на очищенном тексте)."""
    return [tok.text for tok in tokenize(clean(text))]
