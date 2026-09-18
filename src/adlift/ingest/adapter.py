"""Map a platform export onto the canonical schema.

Every ad platform names its columns differently. This adapter takes a CSV plus
an optional mapping from the file's column names to schema names, fills in
what is missing with defaults and returns a frame the rest of the package
accepts. Nothing here is clever; it exists so real data has a documented way
in.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from adlift.schema import ALL_COLUMNS, GROUP_COLUMN, ID_COLUMN, TREATMENT_COLUMN

DEFAULTS: dict[str, object] = {
    "body": "",
    "cta_text": "",
    "claim_type": "unknown",
    "tone": "unknown",
    "has_numeric_claim": 0,
    "has_brand_logo": 0,
    "has_human_face": 0,
    "background_kind": "unknown",
    "dominant_color": "unknown",
    "text_contrast": 0.5,
    "vertical": "unknown",
    "placement": "unknown",
    "device": "unknown",
    "audience": "unknown",
    "daily_budget_usd": 0.0,
    "week_index": 0,
}


def frame_from_csv(
    path: str | Path,
    *,
    column_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Load a CSV export as a canonical creative table.

    Required after mapping: ``headline``, ``campaign_id``, ``author`` and
    either ``ctr`` or both ``clicks`` and ``impressions``.

    Args:
        path: The CSV file.
        column_map: ``{"their_column": "schema_column"}`` renames applied
            before validation.
    """
    frame = pd.read_csv(path)
    if column_map:
        frame = frame.rename(columns=column_map)

    required = {"headline", GROUP_COLUMN, TREATMENT_COLUMN}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns after mapping: {sorted(missing)}")

    if "ctr" not in frame.columns:
        if {"clicks", "impressions"} <= set(frame.columns):
            frame["ctr"] = frame["clicks"] / frame["impressions"].replace(0, pd.NA)
        else:
            raise ValueError("CSV needs a 'ctr' column or both 'clicks' and 'impressions'")

    frame[TREATMENT_COLUMN] = frame[TREATMENT_COLUMN].astype(str).str.lower().str.strip()
    if ID_COLUMN not in frame.columns:
        frame[ID_COLUMN] = [f"row_{i:05d}" for i in range(len(frame))]

    for column, default in DEFAULTS.items():
        if column not in frame.columns:
            frame[column] = default

    if "word_count" not in frame.columns or "char_count" not in frame.columns:
        text = frame[["headline", "body", "cta_text"]].fillna("").agg(" ".join, axis=1)
        frame["word_count"] = text.str.split().str.len()
        frame["char_count"] = text.str.len()

    for column in ("impressions", "clicks"):
        if column not in frame.columns:
            frame[column] = pd.NA

    return frame[[c for c in ALL_COLUMNS if c in frame.columns]]
