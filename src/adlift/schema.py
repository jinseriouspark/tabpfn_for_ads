"""Canonical schema for ad creatives.

Every part of AdLift speaks this one table. A banner that came out of an image
extractor, a row pulled from an ad platform export and a variant an LLM just
wrote all land in the same columns, so the model never has to care where a
creative came from.

The column groups matter for modelling:

``TEXT_COLUMNS``
    Free text read straight off the creative. TabPFN 3.5 consumes these
    natively, which is why there is no vectoriser anywhere in this project.

``CREATIVE_COLUMNS``
    Structured attributes of the creative itself (layout, colour, claim type).
    These are *mediators*: an LLM writing the copy can change them. They are
    included in prediction and excluded from the confounder set.

``CONTEXT_COLUMNS``
    Attributes of the placement, not the creative (device, vertical, budget,
    week). These are *confounders*: they influence both which creatives get
    written and how those creatives perform.

``TREATMENT_COLUMN``
    ``author`` -- "human" or "llm". The thing whose causal effect we estimate.

``GROUP_COLUMN``
    ``campaign_id``. Creatives inside one campaign share budget, audience and
    bidding, so they are not independent. Splits and TabPFN's ``group_col``
    both key off this.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import pandas as pd

Author = Literal["human", "llm"]

TREATMENT_COLUMN = "author"
GROUP_COLUMN = "campaign_id"
ID_COLUMN = "creative_id"

TEXT_COLUMNS = [
    "headline",
    "body",
    "cta_text",
]

CREATIVE_COLUMNS = [
    "word_count",
    "char_count",
    "has_numeric_claim",
    "claim_type",
    "tone",
    "has_brand_logo",
    "has_human_face",
    "background_kind",
    "dominant_color",
    "text_contrast",
]

CONTEXT_COLUMNS = [
    "vertical",
    "placement",
    "device",
    "audience",
    "daily_budget_usd",
    "week_index",
]

OUTCOME_COLUMNS = ["impressions", "clicks", "ctr"]

#: Columns handed to the model as features.
FEATURE_COLUMNS = TEXT_COLUMNS + CREATIVE_COLUMNS + CONTEXT_COLUMNS + [TREATMENT_COLUMN]

#: Columns that may confound the author -> performance relationship. Creative
#: attributes are deliberately absent: they sit on the causal path from author
#: to outcome, so conditioning on them would block part of the effect we want.
CONFOUNDER_COLUMNS = CONTEXT_COLUMNS

CATEGORICAL_COLUMNS = [
    "claim_type",
    "tone",
    "background_kind",
    "dominant_color",
    "vertical",
    "placement",
    "device",
    "audience",
    TREATMENT_COLUMN,
]

ALL_COLUMNS = [ID_COLUMN, GROUP_COLUMN] + FEATURE_COLUMNS + OUTCOME_COLUMNS


@dataclass(slots=True)
class AdCreative:
    """One ad creative, whatever produced it.

    Only ``headline`` is required. Everything else has a neutral default so an
    LLM-written variant or a half-filled platform export still round-trips
    through the same code path as a fully annotated banner.
    """

    headline: str
    body: str = ""
    cta_text: str = ""

    creative_id: str = ""
    campaign_id: str = "unknown"
    author: Author = "human"

    # Creative attributes (mediators).
    claim_type: str = "unknown"
    tone: str = "unknown"
    has_numeric_claim: int = 0
    has_brand_logo: int = 0
    has_human_face: int = 0
    background_kind: str = "unknown"
    dominant_color: str = "unknown"
    text_contrast: float = 0.5

    # Placement attributes (confounders).
    vertical: str = "unknown"
    placement: str = "unknown"
    device: str = "unknown"
    audience: str = "unknown"
    daily_budget_usd: float = 0.0
    week_index: int = 0

    # Outcomes, absent for a creative that has not run yet.
    impressions: float | None = None
    clicks: float | None = None
    ctr: float | None = None

    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        text = " ".join(part for part in (self.headline, self.body, self.cta_text) if part)
        self.word_count = len(text.split())
        self.char_count = len(text)
        if self.ctr is None and self.clicks is not None and self.impressions:
            self.ctr = float(self.clicks) / float(self.impressions)

    # ``slots=True`` needs these declared; they are derived in __post_init__.
    word_count: int = 0
    char_count: int = 0

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row.pop("extra", None)
        return row


def creatives_to_frame(creatives: list[AdCreative]) -> pd.DataFrame:
    """Build the canonical dataframe from a list of creatives."""
    if not creatives:
        return pd.DataFrame(columns=ALL_COLUMNS)
    frame = pd.DataFrame([c.to_row() for c in creatives])
    for column in ALL_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[ALL_COLUMNS]


def validate(frame: pd.DataFrame, *, require_outcome: bool = True) -> None:
    """Raise if ``frame`` cannot be used for modelling.

    Checks the things that silently ruin a causal estimate rather than the
    things pandas would catch anyway: a missing treatment column, a treatment
    with only one level, or a campaign whose creatives are all one author
    (which makes within-campaign comparison impossible).
    """
    missing = [c for c in FEATURE_COLUMNS + [GROUP_COLUMN] if c not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    levels = set(frame[TREATMENT_COLUMN].dropna().unique())
    unknown = levels - {"human", "llm"}
    if unknown:
        raise ValueError(f"{TREATMENT_COLUMN} must be 'human' or 'llm', found: {sorted(unknown)}")
    if len(levels) < 2:
        raise ValueError(
            f"{TREATMENT_COLUMN} has only {levels or 'no'} level(s); "
            "a causal contrast needs both 'human' and 'llm' rows"
        )

    if require_outcome:
        if "ctr" not in frame.columns or frame["ctr"].isna().all():
            raise ValueError("no outcome: 'ctr' is missing or entirely null")

    per_campaign = frame.groupby(GROUP_COLUMN)[TREATMENT_COLUMN].nunique()
    mixed = int((per_campaign > 1).sum())
    if mixed == 0:
        raise ValueError(
            "every campaign uses a single author, so author is perfectly confounded "
            "with campaign; no within-campaign contrast is identifiable"
        )
