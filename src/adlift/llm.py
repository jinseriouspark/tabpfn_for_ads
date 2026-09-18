"""The language-model side of the loop: read a banner, rewrite a creative.

TabPFN scores creatives; it does not write them and it cannot look at a
picture. Both of those jobs go to a language model, behind one small interface
so the rest of the package does not care which one.

``ClaudeCopyLLM``
    Uses the Anthropic SDK. Reads banner images with vision, writes copy
    variants under explicit constraints. Needs ``ANTHROPIC_API_KEY`` (or an
    ``ant auth login`` profile).

``StubCopyLLM``
    Deterministic, offline. Produces plausible extractions and variants from
    templates so the whole pipeline, the tests and the MCP server run with no
    network and no keys. It is a stand-in, not a demo of quality.

The constraints passed to ``generate_variants`` are the point of the design.
The causal and lever analyses say *which attributes* of model-written copy
hurt performance (length, dropped numbers, aspirational tone, hashtag piles).
Those become hard constraints on the rewrite, so the model polishes wording
without reintroducing the habits that cost reach.
"""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

DEFAULT_MODEL = "claude-opus-5"

BANNER_SCHEMA_HINT = """{
  "headline": "main text on the banner, verbatim",
  "body": "secondary text, verbatim, or empty string",
  "cta_text": "button or call-to-action text, or empty string",
  "claim_type": "one of: performance, price, outcome, capability, none",
  "tone": "one of: direct, aspirational, urgent, playful, technical",
  "has_numeric_claim": 0 or 1,
  "has_brand_logo": 0 or 1,
  "has_human_face": 0 or 1,
  "background_kind": "one of: photo, solid, gradient, illustration",
  "dominant_color": "one of: blue, purple, green, red, monochrome, orange",
  "text_contrast": number between 0 and 1,
  "brand": "brand name if visible, else empty string",
  "vertical": "best guess at the advertiser's industry, one lowercase word"
}"""


@dataclass(slots=True)
class RewriteConstraints:
    """Guard-rails for a rewrite, derived from what the data says hurts.

    ``medium`` is ``"ad"`` (headline, body, call to action) or ``"post"`` (one
    social-post text). ``max_hashtags`` only matters for posts.
    """

    max_words: int = 14
    keep_numeric_claim: bool = True
    keep_cta: bool = True
    tone: str = "direct"
    max_hashtags: int | None = None
    medium: str = "ad"
    avoid_words: list[str] = field(
        default_factory=lambda: [
            "discover",
            "unlock",
            "transform",
            "elevate",
            "journey",
            "seamless",
            "effortless",
            "empower",
            "reimagine",
            "excited to share",
            "thrilled",
            "game-changer",
        ]
    )

    def as_prompt(self) -> str:
        scope = "the whole post" if self.medium == "post" else "headline, body and CTA combined"
        lines = [
            f"- At most {self.max_words} words across {scope}.",
            f"- Tone: {self.tone}. Plain claims, no hype.",
        ]
        if self.keep_numeric_claim:
            lines.append("- If the original carries a number, keep a concrete number.")
        if self.keep_cta and self.medium != "post":
            lines.append("- Keep a short, literal call to action.")
        if self.max_hashtags is not None:
            lines.append(f"- At most {self.max_hashtags} hashtags.")
        if self.avoid_words:
            lines.append(f"- Never use these words: {', '.join(self.avoid_words)}.")
        return "\n".join(lines)


class CopyLLM(Protocol):
    """What the rest of the package needs from a language model."""

    name: str

    def extract_banner(self, image_bytes: bytes, media_type: str) -> dict[str, Any]:
        """Read one banner image into schema fields."""
        ...

    def generate_variants(
        self, creative: Any, n: int, constraints: RewriteConstraints
    ) -> list[dict[str, str]]:
        """Write ``n`` rewrites, each a dict with headline, body, cta_text.

        ``creative`` is anything with ``headline``, ``body`` and ``cta_text``
        attributes: an ``AdCreative`` or a ``PostDraft``.
        """
        ...


def _parse_json_block(text: str) -> Any:
    """Pull JSON out of a model reply that may be wrapped in a code fence."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start = text.find("[") if text.lstrip().startswith("[") else text.find("{")
    if start > 0:
        text = text[start:]
    return json.loads(text)


class StubCopyLLM:
    """Offline stand-in that behaves like a language model, deterministically."""

    name = "stub"

    _OPENERS = ["Cut", "Fix", "Stop paying for", "Skip", "Drop"]

    def extract_banner(self, image_bytes: bytes, media_type: str) -> dict[str, Any]:
        # The stub cannot see. It returns a well-formed record derived from the
        # image size so callers can exercise the full path without a key.
        size = len(image_bytes)
        return {
            "headline": f"Banner text unavailable offline ({size} bytes)",
            "body": "",
            "cta_text": "Learn more",
            "claim_type": "none",
            "tone": "direct",
            "has_numeric_claim": 0,
            "has_brand_logo": 1,
            "has_human_face": 0,
            "background_kind": "photo",
            "dominant_color": "blue",
            "text_contrast": 0.7,
            "brand": "",
            "vertical": "unknown",
            "_stub": True,
        }

    def generate_variants(
        self, creative: Any, n: int, constraints: RewriteConstraints
    ) -> list[dict[str, str]]:
        words = [w.strip(".,!?") for w in str(creative.headline).split() if not w.startswith("#")]
        number = next((w for w in words if any(ch.isdigit() for ch in w)), None)
        noun = " ".join(words[-2:]) if len(words) >= 2 else str(creative.headline).strip(".,!?")
        variants = []
        for index in range(n):
            opener = self._OPENERS[index % len(self._OPENERS)]
            headline = f"{opener} {noun}."
            if number and constraints.keep_numeric_claim:
                headline = f"{opener} {noun} by {number}."
            if constraints.medium == "post":
                tail = "" if index % 2 else " Worth it? Yes."
                variants.append(
                    {
                        "headline": " ".join((headline + tail).split()[: constraints.max_words]),
                        "body": "",
                        "cta_text": "",
                    }
                )
            else:
                variants.append(
                    {
                        "headline": " ".join(headline.split()[: constraints.max_words]),
                        "body": "" if index % 2 else "No setup. Cancel anytime.",
                        "cta_text": getattr(creative, "cta_text", "") or "Start free",
                    }
                )
        return variants


class ClaudeCopyLLM:
    """Language-model backend on the Anthropic SDK."""

    name = "claude"

    def __init__(self, model: str | None = None, *, effort: str = "low") -> None:
        import anthropic

        self._client = anthropic.Anthropic()
        self.model = model or os.environ.get("ADLIFT_LLM_MODEL", DEFAULT_MODEL)
        # Low effort keeps the loop interactive; these are short, well-specified
        # tasks and do not need deep reasoning.
        self._effort = effort

    def _ask(self, system: str, content: list[dict[str, Any]], max_tokens: int = 2000) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            output_config={"effort": self._effort},
            messages=[{"role": "user", "content": content}],
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("the model declined this request")
        return "".join(block.text for block in response.content if block.type == "text")

    def extract_banner(self, image_bytes: bytes, media_type: str) -> dict[str, Any]:
        encoded = base64.standard_b64encode(image_bytes).decode("utf-8")
        system = (
            "You read advertising banners and return their content as JSON. "
            "Transcribe text exactly as written. Reply with a single JSON object "
            "and nothing else."
        )
        content = [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": encoded},
            },
            {
                "type": "text",
                "text": f"Return this banner as JSON with exactly these fields:\n{BANNER_SCHEMA_HINT}",
            },
        ]
        parsed = _parse_json_block(self._ask(system, content))
        if not isinstance(parsed, dict):
            raise ValueError("banner extraction did not return a JSON object")
        return parsed

    def generate_variants(
        self, creative: Any, n: int, constraints: RewriteConstraints
    ) -> list[dict[str, str]]:
        if constraints.medium == "post":
            system = (
                "You write short social-media posts. You follow constraints exactly. "
                "Reply with a JSON array and nothing else."
            )
            prompt = (
                "Rewrite this post. Keep the point and the facts; change the wording.\n\n"
                f"Post: {creative.headline}\n\n"
                f"Constraints:\n{constraints.as_prompt()}\n\n"
                f"Return exactly {n} distinct variants as a JSON array of objects with keys "
                '"headline" (the full post text), "body" (empty string), "cta_text" (empty string).'
            )
        else:
            system = (
                "You write short advertising copy. You follow constraints exactly. "
                "Reply with a JSON array and nothing else."
            )
            prompt = (
                "Rewrite this ad creative. Keep the offer and the facts; change the wording.\n\n"
                f"Headline: {creative.headline}\n"
                f"Body: {creative.body or '(none)'}\n"
                f"CTA: {creative.cta_text or '(none)'}\n\n"
                f"Constraints:\n{constraints.as_prompt()}\n\n"
                f"Return exactly {n} distinct variants as a JSON array of objects with keys "
                '"headline", "body", "cta_text".'
            )
        parsed = _parse_json_block(self._ask(system, [{"type": "text", "text": prompt}]))
        if not isinstance(parsed, list):
            raise ValueError("variant generation did not return a JSON array")
        cleaned = []
        for item in parsed[:n]:
            if not isinstance(item, dict) or "headline" not in item:
                continue
            cleaned.append(
                {
                    "headline": str(item.get("headline", "")).strip(),
                    "body": str(item.get("body", "") or "").strip(),
                    "cta_text": str(item.get("cta_text", "") or "").strip(),
                }
            )
        return cleaned


def get_llm(backend: str = "auto", **kwargs: Any) -> CopyLLM:
    """Pick a language-model backend.

    ``auto`` uses Claude when the SDK is importable and credentials resolve,
    otherwise the offline stub. Pass ``"claude"`` or ``"stub"`` to force one.
    """
    if backend == "stub":
        return StubCopyLLM()
    if backend == "claude":
        return ClaudeCopyLLM(**kwargs)

    has_credentials = bool(
        os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    )
    if has_credentials:
        try:
            return ClaudeCopyLLM(**kwargs)
        except ImportError:
            pass
    return StubCopyLLM()
