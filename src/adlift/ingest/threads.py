"""Read a Threads account into the canonical post table.

Two ways in, one table out:

``fetch_threads_account``
    Pulls your own posts and their insights through the official Threads API
    (``graph.threads.net``). Needs a user access token with the
    ``threads_basic`` and ``threads_manage_insights`` permissions. Other
    people's view counts are not exposed by the API, so this path is for the
    account you own.

``threads_frame_from_csv``
    Reads an export you assembled yourself: at minimum ``text``,
    ``timestamp`` and ``views``. Everything else is derived or defaulted.

Both derive the same features: posting hour and weekday, an ISO week label
(the grouping unit), a week index (the drift axis), the text attributes the
author controls, and an engagement rate. The ``author`` column is *yours* to
fill in. Nothing here guesses whether a post was written by a model, because
a guess would be correlated with the text and would poison the estimate.

The API adapter was written against the documented v1.0 request shape but the
network policy of the environment it was built in did not allow reaching the
service, so the first live run should be treated as a check: the fetcher
prints the raw error body if the service rejects a field or a metric.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd

from adlift.schema import THREADS_SPEC
from adlift.text_features import text_features

DEFAULT_BASE_URL = "https://graph.threads.net/v1.0"
POST_FIELDS = "id,text,timestamp,media_type,permalink,is_quote_post,username"
INSIGHT_METRICS = ("views", "likes", "replies", "reposts", "quotes")
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ---------------------------------------------------------------------------
# API access
# ---------------------------------------------------------------------------


def _get(url: str, params: dict[str, Any]) -> dict[str, Any]:
    request = Request(f"{url}?{urlencode(params)}", headers={"User-Agent": "adlift/0.1"})
    try:
        with urlopen(request, timeout=60) as response:  # noqa: S310 - fixed API host
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Threads API returned {error.code} for {url}: {body[:500]}") from error


def fetch_own_posts(
    access_token: str,
    *,
    user_id: str = "me",
    limit: int = 500,
    fields: str = POST_FIELDS,
    base_url: str = DEFAULT_BASE_URL,
    include_replies: bool = False,
) -> list[dict[str, Any]]:
    """List the account's own posts, newest first, following pagination.

    Args:
        access_token: Threads user access token.
        user_id: ``me`` or a numeric Threads user id.
        limit: Stop after this many posts.
        fields: Comma-separated media fields to request.
        base_url: API root; override for a proxy or a newer version.
        include_replies: Also pull the account's replies (``/replies``) and
            mark them ``is_reply=1``.
    """
    posts: list[dict[str, Any]] = []
    edges = ["threads"] + (["replies"] if include_replies else [])
    for edge in edges:
        url = f"{base_url}/{user_id}/{edge}"
        params: dict[str, Any] = {"fields": fields, "access_token": access_token, "limit": 100}
        while url and len(posts) < limit:
            payload = _get(url, params)
            for item in payload.get("data", []):
                item["is_reply"] = int(edge == "replies")
                posts.append(item)
            next_url = payload.get("paging", {}).get("next")
            if not next_url:
                break
            url, params = next_url, {}
    return posts[:limit]


def fetch_post_insights(
    access_token: str,
    post_id: str,
    *,
    metrics: tuple[str, ...] = INSIGHT_METRICS,
    base_url: str = DEFAULT_BASE_URL,
) -> dict[str, float]:
    """Per-post insight metrics. Missing metrics come back as ``NaN``."""
    payload = _get(
        f"{base_url}/{post_id}/insights",
        {"metric": ",".join(metrics), "access_token": access_token},
    )
    out: dict[str, float] = {m: float("nan") for m in metrics}
    for entry in payload.get("data", []):
        name = entry.get("name")
        values = entry.get("values") or []
        value = values[0].get("value") if values else entry.get("total_value", {}).get("value")
        if name in out and value is not None:
            out[name] = float(value)
    return out


def fetch_account(access_token: str, *, base_url: str = DEFAULT_BASE_URL) -> dict[str, Any]:
    """Account id, username and current follower count (a point-in-time value)."""
    profile = _get(f"{base_url}/me", {"fields": "id,username", "access_token": access_token})
    try:
        insights = _get(
            f"{base_url}/me/threads_insights",
            {"metric": "followers_count", "access_token": access_token},
        )
        for entry in insights.get("data", []):
            if entry.get("name") == "followers_count":
                profile["followers_count"] = entry.get("total_value", {}).get("value")
    except RuntimeError:
        profile["followers_count"] = None
    return profile


def fetch_threads_account(
    access_token: str,
    *,
    limit: int = 500,
    include_replies: bool = False,
    tz: str = "UTC",
    base_url: str = DEFAULT_BASE_URL,
) -> pd.DataFrame:
    """Posts plus insights for the account behind ``access_token``."""
    posts = fetch_own_posts(
        access_token, limit=limit, include_replies=include_replies, base_url=base_url
    )
    for post in posts:
        post.update(fetch_post_insights(access_token, post["id"], base_url=base_url))
    return posts_to_frame(posts, tz=tz)


# ---------------------------------------------------------------------------
# Feature derivation
# ---------------------------------------------------------------------------


def _parse_timestamp(value: Any, tz: str) -> datetime | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, datetime):
        stamp = value
    else:
        text = str(value).strip().replace("Z", "+00:00")
        # The Threads API writes offsets without a colon, e.g. +0000.
        if len(text) >= 5 and text[-5] in "+-" and text[-3] != ":":
            text = f"{text[:-2]}:{text[-2:]}"
        try:
            stamp = datetime.fromisoformat(text)
        except ValueError:
            return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(ZoneInfo(tz))


def posts_to_frame(posts: list[dict[str, Any]] | pd.DataFrame, *, tz: str = "UTC") -> pd.DataFrame:
    """Turn raw posts (API dicts or a loose dataframe) into the canonical table.

    Recognised input columns: ``id``/``post_id``, ``text``, ``timestamp``,
    ``media_type`` or ``has_media``, ``is_reply``, ``topic``, ``author``,
    ``views``, ``likes``, ``replies``, ``reposts``, ``quotes``.
    """
    raw = pd.DataFrame(posts) if not isinstance(posts, pd.DataFrame) else posts.copy()
    if raw.empty:
        return pd.DataFrame(columns=THREADS_SPEC.all_columns)

    frame = pd.DataFrame(index=raw.index)
    frame["post_id"] = (
        raw["id"].astype(str)
        if "id" in raw.columns
        else raw.get(
            "post_id", pd.Series([f"post_{i:05d}" for i in range(len(raw))], index=raw.index)
        ).astype(str)
    )
    frame["text"] = raw.get("text", "").fillna("").astype(str)

    stamps = [
        _parse_timestamp(v, tz)
        for v in raw.get(
            "timestamp", raw.get("posted_at", pd.Series([None] * len(raw), index=raw.index))
        )
    ]
    frame["timestamp"] = [s.isoformat() if s else None for s in stamps]
    frame["posted_hour"] = [s.hour if s else 12 for s in stamps]
    frame["weekday"] = [WEEKDAYS[s.weekday()] if s else "unknown" for s in stamps]
    iso = [s.isocalendar() if s else None for s in stamps]
    frame["week"] = [f"{i[0]}-W{i[1]:02d}" if i else "unknown" for i in iso]
    ordinal = [s.date().toordinal() // 7 if s else None for s in stamps]
    valid = [o for o in ordinal if o is not None]
    first = min(valid) if valid else 0
    frame["week_index"] = [int(o - first) if o is not None else 0 for o in ordinal]

    if "has_media" in raw.columns:
        frame["has_media"] = raw["has_media"].fillna(0).astype(int)
    elif "media_type" in raw.columns:
        frame["has_media"] = (
            raw["media_type"].fillna("TEXT_POST").astype(str).str.upper() != "TEXT_POST"
        ).astype(int)
    else:
        frame["has_media"] = 0
    frame["is_reply"] = raw["is_reply"].fillna(0).astype(int) if "is_reply" in raw.columns else 0
    frame["topic"] = (
        raw["topic"].fillna("unknown").astype(str) if "topic" in raw.columns else "unknown"
    )
    frame["author"] = (
        raw["author"].fillna("").astype(str).str.lower().str.strip().replace("", "human")
        if "author" in raw.columns
        else "human"
    )

    features = pd.DataFrame([text_features(t) for t in frame["text"]], index=frame.index)
    for column in features.columns:
        frame[column] = features[column]

    for metric in ("views", "likes", "replies", "reposts", "quotes"):
        frame[metric] = pd.to_numeric(
            raw.get(metric, pd.Series([float("nan")] * len(raw), index=raw.index)), errors="coerce"
        )
    engagement = frame[["likes", "replies", "reposts", "quotes"]].fillna(0).sum(axis=1)
    frame["engagement_rate"] = (engagement / frame["views"].replace(0, float("nan"))).clip(0, 1)

    ordered = [c for c in THREADS_SPEC.all_columns if c in frame.columns] + ["timestamp"]
    return frame[ordered].sort_values("timestamp").reset_index(drop=True)


def threads_frame_from_csv(
    path: str | Path, *, column_map: dict[str, str] | None = None, tz: str = "UTC"
) -> pd.DataFrame:
    """Load a hand-assembled export. Needs ``text``, ``timestamp`` and ``views``."""
    raw = pd.read_csv(path)
    if column_map:
        raw = raw.rename(columns=column_map)
    missing = {"text", "timestamp", "views"} - set(raw.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns after mapping: {sorted(missing)}")
    return posts_to_frame(raw, tz=tz)


def write_label_template(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write a CSV for hand-labelling ``author`` and ``topic``.

    The template keeps the post id, timestamp, text and views so a person can
    label with the post in front of them, and leaves ``author`` blank. Fill
    it with ``human`` or ``llm`` and load the result back with
    ``threads_frame_from_csv``.
    """
    path = Path(path)
    template = frame[["post_id", "timestamp", "text", "views"]].copy()
    template["author"] = frame["author"].where(frame["author"].isin(["human", "llm"]), "")
    template["topic"] = frame["topic"].where(frame["topic"] != "unknown", "")
    template.to_csv(path, index=False)
    return path
