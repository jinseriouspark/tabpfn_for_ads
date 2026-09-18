"""Attributes of a piece of copy that are determined by whoever wrote it.

These are the *mediators*: length, whether there is a number, a question, a
hashtag, an emoji, a call to action, a link. An LLM writing the text changes
them, and they change performance on their own. They are recomputed for every
rewrite so a variant is scored on what it actually says.
"""

from __future__ import annotations

import re

_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HASHTAG = re.compile(r"(?<!\w)#\w+")
_EMOJI = re.compile(
    "[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f900-\U0001f9ff\U0001f1e6-\U0001f1ff]"
)
_CTA_WORDS = (
    "sign up",
    "start free",
    "try it",
    "learn more",
    "get the demo",
    "see pricing",
    "join",
    "download",
    "book",
    "read more",
    "check out",
    "link in bio",
    "dm me",
)


def text_features(text: str) -> dict[str, int]:
    """Describe one piece of copy with the attributes the author controls."""
    text = text or ""
    lowered = text.lower()
    words = text.split()
    return {
        "word_count": len(words),
        "char_count": len(text),
        "n_lines": max(1, text.count("\n") + 1) if text else 0,
        "has_numeric_claim": int(any(ch.isdigit() for ch in text)),
        "has_question": int("?" in text),
        "has_emoji": int(bool(_EMOJI.search(text))),
        "n_hashtags": len(_HASHTAG.findall(text)),
        "has_link": int(bool(_URL.search(text))),
        "has_cta": int(any(phrase in lowered for phrase in _CTA_WORDS)),
    }
