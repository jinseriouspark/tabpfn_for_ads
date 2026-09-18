"""What a dataset looks like to AdLift, and the roles its columns play.

AdLift started on ad creatives and now also reads social posts. The two have
different columns but the same *structure*, and it is the structure the
estimators care about:

Text columns
    Free text read straight off the creative. TabPFN 3.5 consumes these
    natively, so there is no vectoriser anywhere in this project.

Creative columns (mediators)
    Attributes of the copy itself: length, tone, whether it carries a number
    or a hashtag. The author determines them, so they sit on the causal path
    from author to outcome. They are features for prediction and are excluded
    from the confounder set on purpose.

Context columns (confounders)
    Attributes of where and when the creative ran: placement, device, posting
    hour, week. They influence both who writes the copy and how it performs.

Treatment
    ``author``, ``human`` or ``llm``. The thing whose effect is estimated.

Group
    Rows that are not independent of each other: creatives in one campaign,
    posts in one week. Splits, TabPFN's ``group_col`` and the cluster
    bootstrap all key off this.

Outcome
    What is being predicted, with the scale the model should work on. CTR is
    modelled as is; view counts are heavy-tailed and modelled on ``log1p``.

A ``DatasetSpec`` bundles those roles. ``AD_SPEC`` and ``THREADS_SPEC`` are the
two shipped instances; any table with the same roles can declare its own.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

import numpy as np
import pandas as pd

Author = Literal["human", "llm"]
Transform = Literal["identity", "log1p", "logit"]
EffectUnit = Literal["pp", "ratio", "raw"]


@dataclass(frozen=True)
class OutcomeSpec:
    """The target column and the scale the model should see it on.

    Args:
        column: Name of the outcome column.
        label: Human name used in reports, for example ``"CTR"`` or ``"views"``.
        transform: Applied before fitting and inverted after predicting.
            ``log1p`` is right for counts with a long tail such as views.
        clip: Bounds on the natural scale applied to predictions.
        effect_unit: How to print an effect. ``pp`` multiplies by 100,
            ``ratio`` reports a percentage change, ``raw`` prints the number.
    """

    column: str
    label: str
    transform: Transform = "identity"
    clip: tuple[float | None, float | None] | None = None
    effect_unit: EffectUnit = "raw"

    def forward(self, y: Any) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if self.transform == "log1p":
            return np.log1p(np.clip(y, 0.0, None))
        if self.transform == "logit":
            p = np.clip(y, 1e-6, 1 - 1e-6)
            return np.log(p / (1 - p))
        return y

    def inverse(self, z: Any) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        if self.transform == "log1p":
            out = np.expm1(z)
        elif self.transform == "logit":
            out = 1.0 / (1.0 + np.exp(-z))
        else:
            out = z
        if self.clip is not None:
            out = np.clip(out, self.clip[0], self.clip[1])
        return out

    def format_effect(self, natural: float, *, ratio: float | None = None) -> str:
        """Print an effect the way a reader of this outcome expects it."""
        if self.effect_unit == "pp":
            return f"{natural * 100:+.3f} pp"
        if self.effect_unit == "ratio" and ratio is not None:
            return f"{ratio * 100:+.1f}% ({natural:+.1f} {self.label})"
        return f"{natural:+.4g} {self.label}"


@dataclass(frozen=True)
class DatasetSpec:
    """Column roles for one kind of dataset."""

    name: str
    text_columns: tuple[str, ...]
    creative_columns: tuple[str, ...]
    context_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    outcome: OutcomeSpec
    treatment_column: str = "author"
    group_column: str = "campaign_id"
    id_column: str = "creative_id"
    extra_outcome_columns: tuple[str, ...] = ()
    #: How held-out evaluation should split by default. Social posts arrive in
    #: time order, so their cold start is "predict next week's posts".
    default_split: Literal["group", "temporal"] = "group"
    #: Segment used for the effect breakdown in reports.
    default_segment: str = "device"

    @property
    def feature_columns(self) -> list[str]:
        return (
            list(self.text_columns)
            + list(self.creative_columns)
            + list(self.context_columns)
            + [self.treatment_column]
        )

    @property
    def confounder_columns(self) -> list[str]:
        return list(self.context_columns)

    @property
    def adjustment_sets(self) -> dict[str, list[str]]:
        return {
            # Total effect. Confounders only.
            "confounders": list(self.context_columns),
            # Direct effect, holding observable creative attributes fixed.
            "confounders+creative": list(self.context_columns) + list(self.creative_columns),
            # Everything, text included. Conditions on the treatment's own
            # output; shown only to demonstrate why that is the wrong estimand.
            "everything": (
                list(self.context_columns) + list(self.creative_columns) + list(self.text_columns)
            ),
        }

    @property
    def all_columns(self) -> list[str]:
        return (
            [self.id_column, self.group_column]
            + self.feature_columns
            + [self.outcome.column]
            + list(self.extra_outcome_columns)
        )

    @property
    def noiseless_truth_column(self) -> str:
        """Where a synthetic generator keeps the noise-free outcome."""
        return f"_truth_{self.outcome.column}_noiseless"

    def with_outcome(self, outcome: OutcomeSpec) -> DatasetSpec:
        return replace(self, outcome=outcome)


# ---------------------------------------------------------------------------
# Ad creatives
# ---------------------------------------------------------------------------

AD_SPEC = DatasetSpec(
    name="ads",
    text_columns=("headline", "body", "cta_text"),
    creative_columns=(
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
    ),
    context_columns=(
        "vertical",
        "placement",
        "device",
        "audience",
        "daily_budget_usd",
        "week_index",
    ),
    categorical_columns=(
        "claim_type",
        "tone",
        "background_kind",
        "dominant_color",
        "vertical",
        "placement",
        "device",
        "audience",
        "author",
    ),
    outcome=OutcomeSpec("ctr", "CTR", transform="identity", clip=(0.0, 1.0), effect_unit="pp"),
    treatment_column="author",
    group_column="campaign_id",
    id_column="creative_id",
    extra_outcome_columns=("impressions", "clicks"),
    default_split="group",
    default_segment="device",
)

# ---------------------------------------------------------------------------
# Threads posts (one account's timeline)
# ---------------------------------------------------------------------------

VIEWS_OUTCOME = OutcomeSpec(
    "views", "views", transform="log1p", clip=(0.0, None), effect_unit="ratio"
)
ENGAGEMENT_OUTCOME = OutcomeSpec(
    "engagement_rate", "engagement rate", transform="identity", clip=(0.0, 1.0), effect_unit="pp"
)

THREADS_SPEC = DatasetSpec(
    name="threads",
    text_columns=("text",),
    creative_columns=(
        "word_count",
        "char_count",
        "n_lines",
        "has_numeric_claim",
        "has_question",
        "has_emoji",
        "n_hashtags",
        "has_link",
        "has_cta",
    ),
    # Decided before the text is written, and moving reach on their own.
    context_columns=("posted_hour", "weekday", "has_media", "is_reply", "topic", "week_index"),
    categorical_columns=("weekday", "topic", "author"),
    outcome=VIEWS_OUTCOME,
    treatment_column="author",
    group_column="week",
    id_column="post_id",
    extra_outcome_columns=("likes", "replies", "reposts", "quotes", "engagement_rate"),
    default_split="temporal",
    default_segment="topic",
)

SPECS: dict[str, DatasetSpec] = {"ads": AD_SPEC, "threads": THREADS_SPEC}


def get_spec(name: str, *, outcome: str | None = None) -> DatasetSpec:
    """Look a spec up by name, optionally swapping the outcome.

    ``outcome`` accepts ``views`` or ``engagement`` for the Threads spec.
    """
    try:
        spec = SPECS[name]
    except KeyError as error:
        raise ValueError(f"unknown dataset spec {name!r}; pick from {sorted(SPECS)}") from error
    if outcome in (None, "", spec.outcome.column):
        return spec
    if spec is THREADS_SPEC and outcome in ("engagement", "engagement_rate"):
        return spec.with_outcome(ENGAGEMENT_OUTCOME)
    raise ValueError(f"outcome {outcome!r} is not defined for spec {name!r}")


# ---------------------------------------------------------------------------
# Backwards-compatible module constants (the ads spec)
# ---------------------------------------------------------------------------

TREATMENT_COLUMN = AD_SPEC.treatment_column
GROUP_COLUMN = AD_SPEC.group_column
ID_COLUMN = AD_SPEC.id_column
TEXT_COLUMNS = list(AD_SPEC.text_columns)
CREATIVE_COLUMNS = list(AD_SPEC.creative_columns)
CONTEXT_COLUMNS = list(AD_SPEC.context_columns)
OUTCOME_COLUMNS = ["impressions", "clicks", "ctr"]
FEATURE_COLUMNS = AD_SPEC.feature_columns
CONFOUNDER_COLUMNS = AD_SPEC.confounder_columns
CATEGORICAL_COLUMNS = list(AD_SPEC.categorical_columns)
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

    # Derived in __post_init__; declared here because of ``slots=True``.
    word_count: int = 0
    char_count: int = 0

    def __post_init__(self) -> None:
        text = " ".join(part for part in (self.headline, self.body, self.cta_text) if part)
        self.word_count = len(text.split())
        self.char_count = len(text)
        if self.ctr is None and self.clicks is not None and self.impressions:
            self.ctr = float(self.clicks) / float(self.impressions)

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row.pop("extra", None)
        return row


def creatives_to_frame(creatives: list[AdCreative]) -> pd.DataFrame:
    """Build the canonical ads dataframe from a list of creatives."""
    if not creatives:
        return pd.DataFrame(columns=ALL_COLUMNS)
    frame = pd.DataFrame([c.to_row() for c in creatives])
    for column in ALL_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[ALL_COLUMNS]


def validate(
    frame: pd.DataFrame, *, spec: DatasetSpec = AD_SPEC, require_outcome: bool = True
) -> None:
    """Raise if ``frame`` cannot be used for modelling under ``spec``.

    Checks the things that silently ruin a causal estimate rather than the
    things pandas would catch anyway: a missing treatment column, a treatment
    with only one level, or a grouping where every group has a single author
    (which makes within-group comparison impossible).
    """
    treatment, group = spec.treatment_column, spec.group_column
    missing = [c for c in spec.feature_columns + [group] if c not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns for spec {spec.name!r}: {missing}")

    levels = set(frame[treatment].dropna().unique())
    unknown = levels - {"human", "llm"}
    if unknown:
        raise ValueError(f"{treatment} must be 'human' or 'llm', found: {sorted(unknown)}")
    if len(levels) < 2:
        raise ValueError(
            f"{treatment} has only {levels or 'no'} level(s); "
            "a causal contrast needs both 'human' and 'llm' rows"
        )

    if require_outcome:
        column = spec.outcome.column
        if column not in frame.columns or frame[column].isna().all():
            raise ValueError(f"no outcome: {column!r} is missing or entirely null")

    per_group = frame.groupby(group)[treatment].nunique()
    if int((per_group > 1).sum()) == 0:
        raise ValueError(
            f"every {group} uses a single author, so author is perfectly confounded "
            f"with {group}; no within-group contrast is identifiable"
        )
