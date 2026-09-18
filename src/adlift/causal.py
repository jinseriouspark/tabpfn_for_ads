"""Does LLM-written copy actually perform better?

Averaging click-through rate by author answers a different question than the
one people think they are asking. LLM copy is not sprinkled at random across an
ad account: adoption climbs over time and concentrates in particular verticals
and placements, all of which move performance on their own. The gap you read
off a pivot table is that confounding plus whatever the copy is doing, and
there is no way to tell the two apart from the pivot table.

This module separates them, and is opinionated about one thing that is easy to
get wrong: **which columns to adjust for**.

Adjust for placement, device, vertical, audience, budget and week. Those are
chosen before the copy is written and influence both the choice of author and
the outcome, so they confound.

Do *not* adjust for the creative's own attributes -- length, tone, whether it
carries a number. An LLM writing the copy is what changes those attributes.
They sit on the causal path from author to outcome, so conditioning on them
removes part of the effect being measured. Adjusting for the text itself is
worse still: the text is the treatment's output, not a covariate.

So the default estimate is a **total effect**: what happens to performance when
an LLM writes the creative instead of a person, copy changes and all. The
mediator-adjusted estimate is reported alongside it, not as a better number but
to show how much of the effect travels through attributes you can already see.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from adlift.model import CTRModel
from adlift.schema import (
    CONFOUNDER_COLUMNS,
    CREATIVE_COLUMNS,
    GROUP_COLUMN,
    TEXT_COLUMNS,
    TREATMENT_COLUMN,
)

ADJUSTMENT_SETS: dict[str, list[str]] = {
    # Total effect. Confounders only.
    "confounders": CONFOUNDER_COLUMNS,
    # Direct effect, holding observable creative attributes fixed.
    "confounders+creative": CONFOUNDER_COLUMNS + CREATIVE_COLUMNS,
    # Everything, text included. Reported only to show why it is the wrong
    # estimand: it conditions on the treatment's own output.
    "everything": CONFOUNDER_COLUMNS + CREATIVE_COLUMNS + TEXT_COLUMNS,
}

ModelFactory = Callable[[], CTRModel]


@dataclass(slots=True)
class AteResult:
    """An average treatment effect with its uncertainty and its caveats."""

    estimand: str
    estimator: str
    adjustment: str
    ate: float
    ci_low: float
    ci_high: float
    n: int
    n_campaigns: int
    per_row_effect: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    notes: list[str] = field(default_factory=list)

    @property
    def significant(self) -> bool:
        """Whether the interval excludes zero."""
        return not (self.ci_low <= 0.0 <= self.ci_high)

    def describe(self) -> str:
        direction = "lifts" if self.ate > 0 else "lowers"
        verdict = "excludes zero" if self.significant else "includes zero"
        return (
            f"LLM authorship {direction} CTR by {self.ate * 100:.3f} points "
            f"(95% CI {self.ci_low * 100:.3f} to {self.ci_high * 100:.3f}, {verdict})"
        )


def naive_difference(frame: pd.DataFrame) -> float:
    """The pivot-table answer: mean CTR of LLM rows minus human rows.

    Kept so reports can show how far off it is, not because it is usable.
    """
    llm = frame.loc[frame[TREATMENT_COLUMN] == "llm", "ctr"].mean()
    human = frame.loc[frame[TREATMENT_COLUMN] == "human", "ctr"].mean()
    return float(llm - human)


def _cluster_bootstrap_ci(
    effects: np.ndarray,
    groups: np.ndarray,
    *,
    n_boot: int,
    alpha: float,
    seed: int,
) -> tuple[float, float]:
    """Percentile interval, resampling whole campaigns.

    Creatives inside a campaign share a budget, an audience and an unobserved
    campaign quality, so resampling rows would treat correlated observations as
    independent and produce an interval that is far too tight. Campaigns are
    the independent unit, so campaigns are what gets resampled.

    This captures sampling variation in the effect estimates. It does not
    capture uncertainty in the fitted model itself, which would require
    refitting inside every replicate.
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    index_by_group = {g: np.flatnonzero(groups == g) for g in unique}

    draws = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([index_by_group[g] for g in picked])
        draws[b] = float(effects[rows].mean())

    return (
        float(np.quantile(draws, alpha / 2)),
        float(np.quantile(draws, 1 - alpha / 2)),
    )


def s_learner_ate(
    frame: pd.DataFrame,
    *,
    model_factory: ModelFactory | None = None,
    adjustment: str = "confounders",
    n_boot: int = 400,
    alpha: float = 0.05,
    seed: int = 0,
) -> AteResult:
    """Estimate the effect by predicting each creative under both authors.

    One model is fitted on the adjustment columns plus the author flag. Every
    row is then scored twice, once with the flag set to ``llm`` and once to
    ``human``. The average gap is the effect.

    Args:
        frame: Canonical creative table with outcomes.
        model_factory: Builds the model. Defaults to the auto-resolved backend.
        adjustment: Key into ``ADJUSTMENT_SETS``.
        n_boot: Cluster-bootstrap replicates for the interval.
        alpha: One minus the coverage level.
        seed: Seed for the bootstrap.
    """
    if adjustment not in ADJUSTMENT_SETS:
        raise ValueError(
            f"unknown adjustment set {adjustment!r}; pick from {list(ADJUSTMENT_SETS)}"
        )

    columns = [c for c in ADJUSTMENT_SETS[adjustment] if c in frame.columns]
    factory = model_factory or (lambda: CTRModel())
    features = frame[columns + [TREATMENT_COLUMN]].copy()

    model = factory().fit(features, frame["ctr"].to_numpy(dtype=float), groups=frame[GROUP_COLUMN])

    as_llm = features.copy()
    as_llm[TREATMENT_COLUMN] = "llm"
    as_human = features.copy()
    as_human[TREATMENT_COLUMN] = "human"
    effects = model.predict(as_llm) - model.predict(as_human)

    groups = frame[GROUP_COLUMN].to_numpy()
    low, high = _cluster_bootstrap_ci(effects, groups, n_boot=n_boot, alpha=alpha, seed=seed)

    notes = []
    if adjustment == "confounders+creative":
        notes.append(
            "Creative attributes are mediators. This is a direct effect, not the total effect."
        )
    if adjustment == "everything":
        notes.append(
            "Adjusting for the copy itself conditions on the treatment's own output. "
            "Not a valid causal estimate; shown for contrast only."
        )

    return AteResult(
        estimand="total" if adjustment == "confounders" else "direct",
        estimator="s-learner",
        adjustment=adjustment,
        ate=float(effects.mean()),
        ci_low=low,
        ci_high=high,
        n=len(frame),
        n_campaigns=int(pd.Series(groups).nunique()),
        per_row_effect=effects,
        notes=notes,
    )


def t_learner_ate(
    frame: pd.DataFrame,
    *,
    model_factory: ModelFactory | None = None,
    adjustment: str = "confounders",
    n_boot: int = 400,
    alpha: float = 0.05,
    seed: int = 0,
) -> AteResult:
    """Estimate the effect with one model per author, then cross-predict.

    Where the S-learner can bury a weak treatment signal among many covariates,
    the T-learner cannot ignore it: the two arms are separate models. The cost
    is that each model sees only half the data. Agreement between the two
    estimators is the useful signal; disagreement means the estimate is fragile.
    """
    columns = [c for c in ADJUSTMENT_SETS[adjustment] if c in frame.columns]
    factory = model_factory or (lambda: CTRModel())

    arms: dict[str, CTRModel] = {}
    for author in ("human", "llm"):
        subset = frame[frame[TREATMENT_COLUMN] == author]
        if len(subset) < 10:
            raise ValueError(
                f"only {len(subset)} rows for author={author!r}; too few to fit an arm"
            )
        arms[author] = factory().fit(
            subset[columns], subset["ctr"].to_numpy(dtype=float), groups=subset[GROUP_COLUMN]
        )

    features = frame[columns]
    effects = arms["llm"].predict(features) - arms["human"].predict(features)

    groups = frame[GROUP_COLUMN].to_numpy()
    low, high = _cluster_bootstrap_ci(effects, groups, n_boot=n_boot, alpha=alpha, seed=seed)

    return AteResult(
        estimand="total" if adjustment == "confounders" else "direct",
        estimator="t-learner",
        adjustment=adjustment,
        ate=float(effects.mean()),
        ci_low=low,
        ci_high=high,
        n=len(frame),
        n_campaigns=int(pd.Series(groups).nunique()),
        per_row_effect=effects,
    )


def placebo_test(
    frame: pd.DataFrame,
    *,
    model_factory: ModelFactory | None = None,
    adjustment: str = "confounders",
    n_rounds: int = 5,
    seed: int = 0,
) -> dict[str, float]:
    """Reshuffle the author label and check the effect collapses.

    The labels are permuted *within* each campaign, which preserves every
    confounder exactly and destroys only the link between author and outcome. A
    pipeline that still reports a large effect on shuffled labels is measuring
    an artefact of its own construction, and its real estimate cannot be
    trusted either.

    Returns:
        The mean and maximum absolute placebo effect, and the real estimate for
        comparison.
    """
    rng = np.random.default_rng(seed)
    placebo_effects = []

    for _ in range(n_rounds):
        shuffled = frame.copy()
        shuffled[TREATMENT_COLUMN] = shuffled.groupby(GROUP_COLUMN)[TREATMENT_COLUMN].transform(
            lambda s: rng.permutation(s.to_numpy())
        )
        if shuffled[TREATMENT_COLUMN].nunique() < 2:
            continue
        result = s_learner_ate(
            shuffled, model_factory=model_factory, adjustment=adjustment, n_boot=1, seed=seed
        )
        placebo_effects.append(abs(result.ate))

    real = s_learner_ate(frame, model_factory=model_factory, adjustment=adjustment, n_boot=1)
    return {
        "placebo_mean_abs_ate": float(np.mean(placebo_effects))
        if placebo_effects
        else float("nan"),
        "placebo_max_abs_ate": float(np.max(placebo_effects)) if placebo_effects else float("nan"),
        "real_ate": float(real.ate),
        # How many times larger the real effect is than the average placebo.
        # Below roughly 3 the estimate is not distinguishable from noise.
        "ratio": float(abs(real.ate) / (float(np.mean(placebo_effects)) + 1e-12))
        if placebo_effects
        else float("nan"),
    }


def effect_by_segment(result: AteResult, frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Break a fitted effect down by one column.

    An average can hide a treatment that helps one segment and hurts another.
    This is where a copy policy stops being "use the LLM" and becomes "use the
    LLM here, not there".
    """
    if len(result.per_row_effect) != len(frame):
        raise ValueError("result and frame do not line up")
    breakdown = (
        pd.DataFrame({column: frame[column].to_numpy(), "effect": result.per_row_effect})
        .groupby(column)["effect"]
        .agg(["mean", "std", "count"])
        .sort_values("mean", ascending=False)
    )
    breakdown.columns = ["effect", "sd", "n"]
    return breakdown.reset_index()


def mediation_split(total: AteResult, direct: AteResult) -> dict[str, float]:
    """How much of the total effect runs through visible creative attributes.

    A large indirect share means the win is mostly length, tone and whether the
    copy carries a number, which a human could copy without an LLM. A small one
    means the win lives in the wording itself.
    """
    indirect = total.ate - direct.ate
    share = indirect / total.ate if total.ate else float("nan")
    return {
        "total_effect": total.ate,
        "direct_effect": direct.ate,
        "indirect_effect": indirect,
        "indirect_share": float(share),
    }


def overlap_diagnostic(frame: pd.DataFrame, adjustment: str = "confounders") -> dict[str, float]:
    """Check that both authors actually occur across the adjustment columns.

    Every estimator here answers a "what if this creative had the other author"
    question. That question has an answer only where creatives of both kinds
    exist. If LLM copy is always long and human copy is always short, then
    asking what a 30-word human-written creative would have done is asking the
    model to invent a row it has never seen, and it will happily oblige with a
    number nobody should trust.

    This fits a propensity model for author and reports how much of the sample
    sits in a region where one author is close to impossible. Treat a large
    ``share_off_support`` as a reason to narrow the claim, not as a nuisance.

    Returns:
        ``auc`` -- how predictable the author is from the adjustment columns.
        Near 0.5 is close to random assignment; near 1.0 means the two groups
        barely overlap. ``share_off_support`` -- fraction of rows whose
        propensity falls outside the range covered by the other arm.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict

    columns = [c for c in ADJUSTMENT_SETS[adjustment] if c in frame.columns]
    design = frame[columns].copy()
    for column in design.columns:
        if not pd.api.types.is_numeric_dtype(design[column]):
            design[column] = design[column].astype("category").cat.codes
    design = design.astype(float).to_numpy()

    treated = (frame[TREATMENT_COLUMN] == "llm").to_numpy().astype(int)
    propensity = cross_val_predict(
        HistGradientBoostingClassifier(max_iter=200, random_state=0),
        design,
        treated,
        cv=4,
        method="predict_proba",
    )[:, 1]

    treated_range = (propensity[treated == 1].min(), propensity[treated == 1].max())
    control_range = (propensity[treated == 0].min(), propensity[treated == 0].max())
    low = max(treated_range[0], control_range[0])
    high = min(treated_range[1], control_range[1])
    off_support = float(np.mean((propensity < low) | (propensity > high)))

    return {
        "auc": float(roc_auc_score(treated, propensity)),
        "share_off_support": off_support,
        "common_support_low": float(low),
        "common_support_high": float(high),
        "min_propensity": float(propensity.min()),
        "max_propensity": float(propensity.max()),
    }


def full_analysis(
    frame: pd.DataFrame,
    *,
    model_factory: ModelFactory | None = None,
    n_boot: int = 400,
    segment_by: str = "device",
    seed: int = 0,
) -> dict:
    """Run every estimate and diagnostic in one pass.

    This is what the command line and the MCP server both call, so a report
    always carries the caveats alongside the number rather than leaving them
    for the reader to remember.
    """
    total = s_learner_ate(frame, model_factory=model_factory, n_boot=n_boot, seed=seed)
    cross_check = t_learner_ate(frame, model_factory=model_factory, n_boot=n_boot, seed=seed)
    direct = s_learner_ate(
        frame,
        model_factory=model_factory,
        adjustment="confounders+creative",
        n_boot=n_boot,
        seed=seed,
    )

    estimator_gap = abs(total.ate - cross_check.ate)
    agreement = estimator_gap <= 0.5 * abs(total.ate) if total.ate else False

    return {
        "naive_difference": naive_difference(frame),
        "total_effect": total,
        "direct_effect": direct,
        "cross_check": cross_check,
        "mediation": mediation_split(total, direct),
        "segments": effect_by_segment(total, frame, segment_by),
        "overlap": overlap_diagnostic(frame),
        "placebo": placebo_test(frame, model_factory=model_factory, n_rounds=3, seed=seed),
        "estimators_agree": bool(agreement),
    }


def dopfn_ate(
    frame: pd.DataFrame,
    *,
    adjustment: str = "confounders",
    repo_path: str | None = None,
) -> AteResult:
    """Estimate the effect with Do-PFN, Prior Labs' causal foundation model.

    Do-PFN (Robertson et al., 2025) is pre-trained on structural causal models
    to predict interventional outcomes directly, without a causal graph. Its
    paper uses exactly the S- and T-learners in this module as baselines and
    reports that they degrade when mediators are present while Do-PFN holds
    up. This wrapper exists so the two can be compared on the same ad data.

    It is optional and experimental. Do-PFN is a research checkout, not a
    package: clone https://github.com/jr2021/Do-PFN, install its requirements
    (they include torch), and point ``ADLIFT_DOPFN_PATH`` or ``repo_path`` at
    the clone. The model loads its weights from paths relative to that
    directory, so the working directory is switched for the call.
    """
    import os
    import sys

    path = repo_path or os.environ.get("ADLIFT_DOPFN_PATH")
    if not path or not os.path.isdir(path):
        raise RuntimeError(
            "Do-PFN is not available. Clone https://github.com/jr2021/Do-PFN, install its "
            "requirements, and set ADLIFT_DOPFN_PATH to the clone."
        )

    columns = [c for c in ADJUSTMENT_SETS[adjustment] if c in frame.columns]
    design = frame[columns].copy()
    for column in design.columns:
        if not pd.api.types.is_numeric_dtype(design[column]):
            design[column] = design[column].astype("category").cat.codes
    treatment = (frame[TREATMENT_COLUMN] == "llm").to_numpy(dtype=float)
    # Do-PFN's convention: the treatment is column zero.
    X = np.column_stack([treatment, design.to_numpy(dtype=float)])
    y = frame["ctr"].to_numpy(dtype=float)

    previous_cwd = os.getcwd()
    inserted = path not in sys.path
    if inserted:
        sys.path.insert(0, path)
    try:
        os.chdir(path)
        from scripts.transformer_prediction_interface.base import DoPFNRegressor  # type: ignore

        model = DoPFNRegressor()
        model.fit(X, y)
        effects = np.asarray(model.predict_cate(X.copy()), dtype=float).ravel()
    finally:
        os.chdir(previous_cwd)
        if inserted:
            sys.path.remove(path)

    groups = frame[GROUP_COLUMN].to_numpy()
    low, high = _cluster_bootstrap_ci(effects, groups, n_boot=400, alpha=0.05, seed=0)
    return AteResult(
        estimand="total" if adjustment == "confounders" else "direct",
        estimator="do-pfn",
        adjustment=adjustment,
        ate=float(effects.mean()),
        ci_low=low,
        ci_high=high,
        n=len(frame),
        n_campaigns=int(pd.Series(groups).nunique()),
        per_row_effect=effects,
        notes=["Do-PFN research checkout; interval is a cluster bootstrap of its CATEs."],
    )
