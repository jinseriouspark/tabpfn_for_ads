"""Read an ad banner image into an ``AdCreative``.

A banner is a picture with words on it. The words are what the model reads;
the picture supplies the visual attributes (logo, face, background, contrast).
A vision-capable language model does both in one pass and returns a JSON
record that maps straight onto the schema.

Anything the extractor does not return falls back to the schema default, and
the caller decides the placement context (device, placement, budget), because
that is not something a picture can tell you.
"""

from __future__ import annotations

import io
import mimetypes
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from adlift.llm import CopyLLM, get_llm
from adlift.schema import AdCreative

_MEDIA_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}


def load_image(source: str | Path | bytes) -> tuple[bytes, str]:
    """Read image bytes and a media type from a path, URL or raw bytes.

    Args:
        source: A local path, an ``http(s)`` URL or bytes already in memory.

    Returns:
        ``(bytes, media_type)``. Images in a format the vision API does not
        accept are re-encoded to PNG.
    """
    if isinstance(source, bytes):
        raw, media_type = source, "image/png"
    else:
        text = str(source)
        if text.startswith(("http://", "https://")):
            request = Request(text, headers={"User-Agent": "adlift/0.1"})
            with urlopen(request, timeout=30) as response:  # noqa: S310 - caller-supplied URL
                raw = response.read()
                media_type = response.headers.get_content_type()
        else:
            path = Path(text)
            raw = path.read_bytes()
            media_type = mimetypes.guess_type(path.name)[0] or "image/png"

    if media_type not in _MEDIA_TYPES:
        from PIL import Image

        with Image.open(io.BytesIO(raw)) as image:
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="PNG")
        raw, media_type = buffer.getvalue(), "image/png"
    return raw, media_type


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(bool(int(value)))
    except (TypeError, ValueError):
        return default


def banner_to_creative(
    source: str | Path | bytes,
    *,
    llm: CopyLLM | None = None,
    campaign_id: str = "new",
    author: str = "human",
    context: dict[str, Any] | None = None,
) -> AdCreative:
    """Extract one banner into the canonical creative record.

    Args:
        source: Path, URL or bytes of the banner image.
        llm: Extraction backend. Defaults to whatever ``get_llm`` resolves.
        campaign_id: Campaign this creative belongs to.
        author: Who wrote the copy, ``human`` or ``llm``. The image cannot tell
            you this; it is metadata you supply.
        context: Placement fields (``device``, ``placement``, ``vertical``,
            ``audience``, ``daily_budget_usd``, ``week_index``). Anything
            omitted stays at the schema default.

    Returns:
        An ``AdCreative`` with text and visual attributes filled in.
    """
    llm = llm or get_llm()
    raw, media_type = load_image(source)
    fields = llm.extract_banner(raw, media_type)
    context = dict(context or {})

    creative = AdCreative(
        headline=str(fields.get("headline", "")).strip(),
        body=str(fields.get("body", "") or "").strip(),
        cta_text=str(fields.get("cta_text", "") or "").strip(),
        campaign_id=campaign_id,
        author=author,  # type: ignore[arg-type]
        claim_type=str(fields.get("claim_type", "unknown") or "unknown"),
        tone=str(fields.get("tone", "unknown") or "unknown"),
        has_numeric_claim=_coerce_int(fields.get("has_numeric_claim")),
        has_brand_logo=_coerce_int(fields.get("has_brand_logo")),
        has_human_face=_coerce_int(fields.get("has_human_face")),
        background_kind=str(fields.get("background_kind", "unknown") or "unknown"),
        dominant_color=str(fields.get("dominant_color", "unknown") or "unknown"),
        text_contrast=float(fields.get("text_contrast", 0.5) or 0.5),
        vertical=str(context.get("vertical") or fields.get("vertical") or "unknown"),
        placement=str(context.get("placement", "unknown")),
        device=str(context.get("device", "unknown")),
        audience=str(context.get("audience", "unknown")),
        daily_budget_usd=float(context.get("daily_budget_usd", 0.0)),
        week_index=int(context.get("week_index", 0)),
        extra={"brand": fields.get("brand", ""), "extractor": llm.name},
    )
    return creative
