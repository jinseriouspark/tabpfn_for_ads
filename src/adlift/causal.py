"""Does LLM-written copy actually perform better?

Averaging the outcome by author answers a different question than the one
people think they are asking. LLM copy is not sprinkled at random across an
account: adoption climbs over time and concentrates in particular verticals,
placements or topics, all of which move performance on their own. The gap you
read off a pivot table is that confounding plus whatever the copy is doing,
and there is no way to tell the two apart from the pivot table.

This module separates them, and is opinionated about one thing that is easy to
get wrong: **which columns to adjust for**.

Adjust for the context: placement, device, posting hour, topic, week. Those are
chosen before the copy is written and influence both the choice of author and
the outcome, so they confound.

Do *not* adjust for the creative's own attributes -- length, tone, whether it
carries a number or a hashtag. An LLM writing the copy is what changes those
attributes. They sit on the causal path from author to outcome, so conditioning
on them removes part of the effect being measured. Adjusting for the text
itself is worse still: the text is the treatment's output, not a covariate.

So the default estimate is a **total effect**: what happens when an LLM writes
the creative instead of a person, copy changes and all. The mediator-adjusted
estimate is reported alongside it, not as a better number but to show how much
of the effect travels through attributes you can already see.

Effects are reported on the outcome's natural scale and, when the model works
on a log scale, also as a ratio: "LLM posts get 23% fewer views" is the number
a person can act on.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from adlift.model import CTRModel
from adlift.schema import AD_SPEC, DatasetSpec

ModelFactory = Callable[[], CTRModel]


@dataclass(slots=True)
class AteResult:
    """An average treatment effect with its uncertainty and its caveats."""

    estimand: str
    estimator: str
    adjustment: str
    #: Effect on the natural scale (CTR points, views).
    ate: float
    ci_low: float
    ci_high: float
    n: int
    n_groups: int
    outcome_label: str = "outcome"
    effect_unit: str = "raw"
    #: Effect on the model's scale (log1p for counts). Equal to ``ate`` when
    #: the outcome is modelled as is.
    ate_model_scale: float = float("nan")
    ci_low_model: float = float("nan")
    ci_high_model: float = float("nan")
    #: ``exp(ate_model_scale) - 1`` for log outcomes: the multiplicative change.
    ratio: float | None = None
    per_row_effect: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    per_row_effect_model: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    notes: list[str] = field(default_factory=list)

    @property
    def significant(self) -> bool:
        """Whether the interval a reader should look at excludes zero.

        For log outcomes that is the model-scale interval; the natural-scale
        mean of a heavy-tailed outcome is carried by a few extreme rows.
        """
        if self.ratio is not None:
            return not (self.ci_low_model <= 0.0 <= self.ci_high_model)
        return not (self.ci_low <= 0.0 <= self.ci_high)

    @property
    def n_campaigns(self) -> int:  # backwards-compatible alias
        return self.n_groups

    @property
    def ratio_low(self) -> float | None:
        """Lower bound of the ratio, from the model-scale interval."""
        return float(np.expm1(self.ci_low_model)) if self.ratio is not None else None

    @property
    def ratio_high(self) -> float | None:
        return float(np.expm1(self.ci_high_model)) if self.ratio is not None else None

    def interval(self) -> str:
        """The interval a reader should look at: ratio bounds for log outcomes."""
        if self.ratio is not None:
            return f"{self.ratio_low * 100:+.1f}% to {self.ratio_high * 100:+.1f}%"
        return f"{_format(self, self.ci_low)} to {_format(self, self.ci_high)}"

    def formatted(self) -> str:
        return _format(self, self.ate, ratio=self.ratio)

    def describe(self) -> str:
        headline = self.ratio if self.ratio is not None else self.ate
        direction = "lifts" if headline > 0 else "lowers"
        verdict = "excludes zero" if self.significant else "includes zero"
        return (
            f"LLM authorship {direction} {self.outcome_label} by {self.formatted()} "
            f"(95% CI {self.interval()}, {verdict})"
        )


def _format(result: AteResult, value: float, *, ratio: float | None = None) -> str:
    if result.effect_unit == "pp":
        return f"{value * 100:+.3f} pp"
    if result.effect_unit == "ratio":
        if ratio is not None:
            return f"{ratio * 100:+.1f}% ({value:+.1f} {result.outcome_label})"
        return f"{value:+.1f} {result.outcome_label}"
    return f"{value:+.4g} {result.outcome_label}"


def _factory(model_factory: ModelFactory | None, spec: DatasetSpec) -> ModelFactory:
    """Make sure every model built for this analysis carries the right spec."""

    def build() -> CTRModel:
        model = model_factory() if model_factory is not None else CTRModel(spec=spec)
        if model.spec is not spec:
            model.spec = spec
        return model

    return build


def _columns(spec: DatasetSpec, frame: pd.DataFrame, adjustment: str) -> list[str]:
    sets = spec.adjustment_sets
    if adjustment not in sets:
        raise ValueError(f"unknown adjustment set {adjustment!r}; pick from {list(sets)}")
    return [c for c in sets[adjustment] if c in frame.columns]


def naive_difference(frame: pd.DataFrame, spec: DatasetSpec = AD_SPEC) -> float:
    """The pivot-table answer: mean outcome of LLM rows minus human rows.

    Kept so reports can show how far off it is, not because it is usable.
    """
    column, treatment = spec.outcome.column, spec.treatment_column
    llm = frame.loc[frame[treatment] == "llm", column].mean()
    human = frame.loc[frame[treatment] == "human", column].mean()
    return float(llm - human)


def naive_ratio(frame: pd.DataFrame, spec: DatasetSpec = AD_SPEC) -> float:
    """The pivot-table answer as a percentage change."""
    column, treatment = spec.outcome.column, spec.treatment_column
    llm = frame.loc[frame[treatment] == "llm", column].mean()
    human = frame.loc[frame[treatment] == "human", column].mean()
    return float(llm / human - 1.0) if human else float("nan")


def _cluster_bootstrap_cis(
    effects: list[np.ndarray],
    groups: np.ndarray,
    *,
    n_boot: int,
    alpha: float,
    seed: int,
) -> list[tuple[float, float]]:
    """Percentile intervals, resampling whole groups, one per effect array.

    Creatives inside a group share a budget, an audience and an unobserved
    quality (posts inside a week share follower count and algorithm state),
    so resampling rows would treat correlated observations as independent and
    produce an interval that is far too tight. Groups are the independent
    unit, so groups are what gets resampled. Every effect array is resampled
    with the same draws.

    This captures sampling variation in the effect estimates. It does not
    capture uncertainty in the fitted model itself, which would require
    refitting inside every replicate.
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    index_by_group = {g: np.flatnonzero(groups == g) for g in unique}

    draws = np.empty((n_boot, len(effects)))
    for b in range(n_boot):
        picked = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([index_by_group[g] for g in picked])
        for j, effect in enumerate(effects):
            draws[b, j] = float(effect[rows].mean())

    return [
        (float(np.quantile(draws[:, j], alpha / 2)), float(np.quantile(draws[:, j], 1 - alpha / 2)))
        for j in range(len(effects))
    ]


def _result(
    *,
    spec: DatasetSpec,
    estimator: str,
    adjustment: str,
    natural: np.ndarray,
    model_scale: np.ndarray,
    groups: np.ndarray,
    n_boot: int,
    alpha: float,
    seed: int,
    notes: list[str] | None = None,
) -> AteResult:
    (low, high), (low_m, high_m) = _cluster_bootstrap_cis(
        [natural, model_scale], groups, n_boot=n_boot, alpha=alpha, seed=seed
    )
    ate_model = float(model_scale.mean())
    ratio = float(np.expm1(ate_model)) if spec.outcome.transform == "log1p" else None
    return AteResult(
        estimand="total" if adjustment == "confounders" else "direct",
        estimator=estimator,
        adjustment=adjustment,
        ate=float(natural.mean()),
        ci_low=low,
        ci_high=high,
        n=len(natural),
        n_groups=int(pd.Series(groups).nunique()),
        outcome_label=spec.outcome.label,
        effect_unit=spec.outcome.effect_unit,
        ate_model_scale=ate_model,
        ci_low_model=low_m,
        ci_high_model=high_m,
        ratio=ratio,
        per_row_effect=natural,
        per_row_effect_model=model_scale,
        notes=notes or [],
    )


def s_learner_ate(
    frame: pd.DataFrame,
    *,
    model_factory: ModelFactory | None = None,
    spec: DatasetSpec = AD_SPEC,
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
        frame: Canonical table with outcomes.
        model_factory: Builds the model. Defaults to the auto-resolved backend
            with this spec.
        spec: Column roles and outcome scale.
        adjustment: Key into ``spec.adjustment_sets``.
        n_boot: Cluster-bootstrap replicates for the interval.
        alpha: One minus the coverage level.
        seed: Seed for the bootstrap.
    """
    treatment, group, outcome = spec.treatment_column, spec.group_column, spec.outcome.column
    columns = _columns(spec, frame, adjustment)
    build = _factory(model_factory, spec)
    features = frame[columns + [treatment]].copy()

    model = build().fit(features, frame[outcome].to_numpy(dtype=float), groups=frame[group])

    as_llm = features.copy()
    as_llm[treatment] = "llm"
    as_human = features.copy()
    as_human[treatment] = "human"

    llm_model = model.predict(as_llm, scale="model")
    human_model = model.predict(as_human, scale="model")
    natural = spec.outcome.inverse(llm_model) - spec.outcome.inverse(human_model)
    model_scale = llm_model - human_model

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

    return _result(
        spec=spec,
        estimator="s-learner",
        adjustment=adjustment,
        natural=natural,
        model_scale=model_scale,
        groups=frame[group].to_numpy(),
        n_boot=n_boot,
        alpha=alpha,
        seed=seed,
        notes=notes,
    )


def t_learner_ate(
    frame: pd.DataFrame,
    *,
    model_factory: ModelFactory | None = None,
    spec: DatasetSpec = AD_SPEC,
    adjustment: str = "confounders",
    n_boot: int = 400,
    alpha: float = 0.05,
    seed: int = 0,
    refit_boot: int = 0,
) -> AteResult:
    """Estimate the effect with one model per author, then cross-predict.

    Where the S-learner can bury a weak treatment signal among many covariates,
    the T-learner cannot ignore it: the two arms are separate models. The cost
    is that each model sees only part of the data. Agreement between the two
    estimators is the useful signal; disagreement means the estimate is fragile.

    Args:
        refit_boot: When positive, the interval comes from this many
            cluster-bootstrap replicates that *refit both arms* on resampled
            groups. That is the honest interval on small data: the cheap
            per-row bootstrap only resamples the effects of one fitted model
            and understates how much the model itself moves.
    """
    treatment, group, outcome = spec.treatment_column, spec.group_column, spec.outcome.column
    columns = _columns(spec, frame, adjustment)
    build = _factory(model_factory, spec)
    authors = frame[treatment].to_numpy()
    y_natural = frame[outcome].to_numpy(dtype=float)

    def fit_arms(rows: np.ndarray) -> dict[str, CTRModel] | None:
        arms: dict[str, CTRModel] = {}
        for author in ("human", "llm"):
            mask = rows[authors[rows] == author]
            if len(mask) < 10:
                return None
            arms[author] = build().fit(
                frame[columns].iloc[mask], y_natural[mask], groups=frame[group].iloc[mask]
            )
        return arms

    everything = np.arange(len(frame))
    arms = fit_arms(everything)
    if arms is None:
        raise ValueError("fewer than 10 rows for one author; too few to fit an arm")

    features = frame[columns]
    llm_model = arms["llm"].predict(features, scale="model")
    human_model = arms["human"].predict(features, scale="model")
    natural = spec.outcome.inverse(llm_model) - spec.outcome.inverse(human_model)
    groups_all = frame[group].to_numpy()

    result = _result(
        spec=spec,
        estimator="t-learner",
        adjustment=adjustment,
        natural=natural,
        model_scale=llm_model - human_model,
        groups=groups_all,
        n_boot=n_boot,
        alpha=alpha,
        seed=seed,
    )

    if refit_boot > 0:
        rng = np.random.default_rng(seed)
        unique = np.unique(groups_all)
        index_by_group = {g: np.flatnonzero(groups_all == g) for g in unique}
        draws_natural: list[float] = []
        draws_model: list[float] = []
        for _ in range(refit_boot):
            picked = rng.choice(unique, size=len(unique), replace=True)
            rows = np.concatenate([index_by_group[g] for g in picked])
            arms_b = fit_arms(rows)
            if arms_b is None:
                continue
            sub = features.iloc[rows]
            m1 = arms_b["llm"].predict(sub, scale="model")
            m0 = arms_b["human"].predict(sub, scale="model")
            draws_model.append(float((m1 - m0).mean()))
            draws_natural.append(
                float((spec.outcome.inverse(m1) - spec.outcome.inverse(m0)).mean())
            )
        if len(draws_model) >= 10:
            result.ci_low_model = float(np.quantile(draws_model, alpha / 2))
            result.ci_high_model = float(np.quantile(draws_model, 1 - alpha / 2))
            result.ci_low = float(np.quantile(draws_natural, alpha / 2))
            result.ci_high = float(np.quantile(draws_natural, 1 - alpha / 2))
            result.notes.append(
                f"Interval from {len(draws_model)} cluster-bootstrap replicates that refit both arms."
            )
    return result


def _propensity(
    frame: pd.DataFrame, columns: list[str], spec: DatasetSpec, *, seed: int = 0
) -> np.ndarray:
    """Cross-fitted probability of LLM authorship given the adjustment columns."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_predict

    design = frame[columns].copy()
    for column in design.columns:
        if not pd.api.types.is_numeric_dtype(design[column]):
            design[column] = design[column].astype("category").cat.codes
    treated = (frame[spec.treatment_column] == "llm").to_numpy().astype(int)
    propensity = cross_val_predict(
        HistGradientBoostingClassifier(max_iter=150, min_samples_leaf=10, random_state=seed),
        design.astype(float).to_numpy(),
        treated,
        cv=4,
        method="predict_proba",
    )[:, 1]
    return np.clip(propensity, 0.05, 0.95)


def dr_learner_ate(
    frame: pd.DataFrame,
    *,
    model_factory: ModelFactory | None = None,
    spec: DatasetSpec = AD_SPEC,
    adjustment: str = "confounders",
    n_boot: int = 400,
    alpha: float = 0.05,
    seed: int = 0,
    cross_fit: int = 4,
) -> AteResult:
    """Doubly-robust (AIPW) effect: outcome models corrected by propensity weights.

    The S-learner can shrink a weak treatment signal toward zero on small
    data, because a regularised model has little reason to split on one
    binary column. The T-learner does not, but leans entirely on its two
    outcome models. The doubly-robust score combines the T-learner's arms
    with a propensity model: it is unbiased if *either* the outcome models
    or the propensity model is right, which is why it is the headline here.

    The score is computed on the model scale (``log1p`` for counts) and read
    back as a ratio; the natural-scale ``ate`` is the plug-in difference of
    the two arms (or the score itself when the outcome is untransformed).
    Propensities are clipped to ``[0.05, 0.95]`` so a handful of
    near-deterministic rows cannot dominate.

    The outcome arms are **cross-fitted** over ``cross_fit`` group folds: each
    row is scored by arms that never saw its group. Without that, a flexible
    model fits its own training rows closely, the residuals vanish, and the
    doubly-robust correction silently collapses onto the T-learner.
    """
    from sklearn.model_selection import GroupKFold

    treatment, group, outcome = spec.treatment_column, spec.group_column, spec.outcome.column
    columns = _columns(spec, frame, adjustment)
    build = _factory(model_factory, spec)
    features = frame[columns]
    y_natural = frame[outcome].to_numpy(dtype=float)
    authors = frame[treatment].to_numpy()
    groups_all = frame[group].to_numpy()

    for author in ("human", "llm"):
        if int((authors == author).sum()) < 10:
            raise ValueError(f"too few rows for author={author!r}; need at least 10 to fit an arm")

    def fit_arm(rows: np.ndarray, author: str) -> CTRModel:
        mask = rows[authors[rows] == author]
        if len(mask) < 5:  # a fold with almost none of this arm: fall back to every row of it
            mask = np.flatnonzero(authors == author)
        return build().fit(features.iloc[mask], y_natural[mask], groups=frame[group].iloc[mask])

    mu1 = np.empty(len(frame))
    mu0 = np.empty(len(frame))
    n_groups = len(np.unique(groups_all))
    folds = max(2, min(cross_fit, n_groups)) if cross_fit and cross_fit > 1 else 0
    if folds:
        for train_rows, test_rows in GroupKFold(n_splits=folds).split(features, groups=groups_all):
            mu1[test_rows] = fit_arm(train_rows, "llm").predict(
                features.iloc[test_rows], scale="model"
            )
            mu0[test_rows] = fit_arm(train_rows, "human").predict(
                features.iloc[test_rows], scale="model"
            )
    else:
        everything = np.arange(len(frame))
        mu1 = fit_arm(everything, "llm").predict(features, scale="model")
        mu0 = fit_arm(everything, "human").predict(features, scale="model")

    y = spec.outcome.forward(y_natural)
    treated = (authors == "llm").astype(float)
    e = _propensity(frame, columns, spec, seed=seed)

    score = mu1 - mu0 + treated * (y - mu1) / e - (1.0 - treated) * (y - mu0) / (1.0 - e)
    # On an untransformed outcome the score already lives on the natural
    # scale, so it is the estimate. On a log scale the natural-scale number is
    # the plug-in arm difference and the ratio carries the correction.
    if spec.outcome.transform == "identity":
        natural = score
    else:
        natural = spec.outcome.inverse(mu1) - spec.outcome.inverse(mu0)

    return _result(
        spec=spec,
        estimator="dr-learner",
        adjustment=adjustment,
        natural=natural,
        model_scale=score,
        groups=groups_all,
        n_boot=n_boot,
        alpha=alpha,
        seed=seed,
        notes=[
            "Doubly robust on the model scale with cross-fitted arms; the natural-scale effect is the "
            "plug-in arm difference."
        ],
    )


def placebo_test(
    frame: pd.DataFrame,
    *,
    model_factory: ModelFactory | None = None,
    spec: DatasetSpec = AD_SPEC,
    adjustment: str = "confounders",
    n_rounds: int = 5,
    seed: int = 0,
) -> dict[str, float]:
    """Reshuffle the author label and check the effect collapses.

    The labels are permuted *within* each group, which preserves every
    confounder exactly and destroys only the link between author and outcome.
    A pipeline that still reports a large effect on shuffled labels is
    measuring an artefact of its own construction, and its real estimate
    cannot be trusted either.

    Returns:
        The mean and maximum absolute placebo effect, the real estimate, and
        their ratio. Below roughly 3 the estimate is not distinguishable from
        noise.
    """
    treatment, group = spec.treatment_column, spec.group_column
    rng = np.random.default_rng(seed)
    placebo_effects = []

    for _ in range(n_rounds):
        shuffled = frame.copy()
        shuffled[treatment] = shuffled.groupby(group)[treatment].transform(
            lambda s: rng.permutation(s.to_numpy())
        )
        if shuffled[treatment].nunique() < 2:
            continue
        result = s_learner_ate(
            shuffled,
            model_factory=model_factory,
            spec=spec,
            adjustment=adjustment,
            n_boot=1,
            seed=seed,
        )
        placebo_effects.append(abs(result.ate))

    real = s_learner_ate(
        frame, model_factory=model_factory, spec=spec, adjustment=adjustment, n_boot=1
    )
    mean_placebo = float(np.mean(placebo_effects)) if placebo_effects else float("nan")
    return {
        "placebo_mean_abs_ate": mean_placebo,
        "placebo_max_abs_ate": float(np.max(placebo_effects)) if placebo_effects else float("nan"),
        "real_ate": float(real.ate),
        "ratio": float(abs(real.ate) / (mean_placebo + 1e-12)) if placebo_effects else float("nan"),
    }


def effect_by_segment(result: AteResult, frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Break a fitted effect down by one column.

    An average can hide a treatment that helps one segment and hurts another.
    This is where a copy policy stops being "use the LLM" and becomes "use the
    LLM here, not there".
    """
    if len(result.per_row_effect) != len(frame):
        raise ValueError("result and frame do not line up")
    if column not in frame.columns:
        raise ValueError(f"segment column {column!r} is not in the frame")
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

    A large indirect share means the win (or loss) is mostly length, tone and
    whether the copy carries a number, which a human could copy without an
    LLM. A small one means it lives in the wording itself.
    """
    indirect = total.ate - direct.ate
    share = indirect / total.ate if total.ate else float("nan")
    return {
        "total_effect": total.ate,
        "direct_effect": direct.ate,
        "indirect_effect": indirect,
        "indirect_share": float(share),
    }


def overlap_diagnostic(
    frame: pd.DataFrame, adjustment: str = "confounders", spec: DatasetSpec = AD_SPEC
) -> dict[str, float]:
    """Check that both authors actually occur across the adjustment columns.

    Every estimator here answers a "what if this creative had the other author"
    question. That question has an answer only where creatives of both kinds
    exist. If LLM copy is always long and human copy is always short, then
    asking what a 30-word human-written creative would have done is asking the
    model to invent a row it has never seen, and it will happily oblige with a
    number nobody should trust.

    Returns:
        ``auc`` -- how predictable the author is from the adjustment columns.
        Near 0.5 is close to random assignment; near 1.0 means the two groups
        barely overlap. ``share_off_support`` -- fraction of rows whose
        propensity falls outside the range covered by the other arm.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict

    columns = _columns(spec, frame, adjustment)
    design = frame[columns].copy()
    for column in design.columns:
        if not pd.api.types.is_numeric_dtype(design[column]):
            design[column] = design[column].astype("category").cat.codes
    design = design.astype(float).to_numpy()

    treated = (frame[spec.treatment_column] == "llm").to_numpy().astype(int)
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
    spec: DatasetSpec = AD_SPEC,
    n_boot: int = 400,
    segment_by: str | None = None,
    seed: int = 0,
    refit_boot: int = 30,
) -> dict:
    """Run every estimate and diagnostic in one pass.

    This is what the command line and the MCP server both call, so a report
    always carries the caveats alongside the number rather than leaving them
    for the reader to remember.

    The headline is the T-learner with a refit cluster bootstrap. The
    S-learner is the cross-check: on small, heavy-tailed data it shrinks a
    weak treatment toward zero, so agreement between the two is informative
    and disagreement is a warning. A doubly-robust estimator exists in this
    module (:func:`dr_learner_ate`) but its cross-fitted score has far too
    much variance below a few hundred rows on a heavy-tailed outcome, so it
    is not part of the default report.
    """
    segment_by = segment_by or spec.default_segment
    common = {"model_factory": model_factory, "spec": spec, "n_boot": n_boot, "seed": seed}
    total = t_learner_ate(frame, refit_boot=refit_boot, **common)
    s_total = s_learner_ate(frame, **common)
    direct = s_learner_ate(frame, adjustment="confounders+creative", **common)

    # Agreement is judged on the model scale, where the estimators are
    # comparable; the natural-scale mean of a heavy-tailed outcome is not.
    scales = [total.ate_model_scale, s_total.ate_model_scale]
    same_sign = len({np.sign(v) for v in scales if v}) <= 1
    spread = abs(scales[0] - scales[1])
    agreement = bool(same_sign and spread <= max(0.5 * abs(total.ate_model_scale), 1e-9))
    treatment = spec.treatment_column

    return {
        "spec": spec,
        "naive_difference": naive_difference(frame, spec),
        "naive_ratio": naive_ratio(frame, spec),
        "total_effect": total,
        "cross_check": s_total,
        "s_learner": s_total,
        "direct_effect": direct,
        # Same estimator on both sides, so the split is internally consistent.
        "mediation": mediation_split(s_total, direct),
        "median_effect": float(np.median(total.per_row_effect)),
        "segments": effect_by_segment(total, frame, segment_by),
        "overlap": overlap_diagnostic(frame, spec=spec),
        "placebo": placebo_test(
            frame, model_factory=model_factory, spec=spec, n_rounds=3, seed=seed
        ),
        "estimators_agree": bool(agreement),
        "n_rows": int(len(frame)),
        "n_llm": int((frame[treatment] == "llm").sum()),
        "n_human": int((frame[treatment] == "human").sum()),
        # Half the interval width: the smallest effect this sample could have
        # resolved. If the estimate is smaller than this, "no effect found" is
        # a statement about the sample, not about the copy.
        "detectable_effect": float((total.ci_high - total.ci_low) / 2),
        "detectable_ratio": (
            float(np.expm1((total.ci_high_model - total.ci_low_model) / 2))
            if total.ratio is not None
            else None
        ),
    }


def dopfn_ate(
    frame: pd.DataFrame,
    *,
    spec: DatasetSpec = AD_SPEC,
    adjustment: str = "confounders",
    repo_path: str | None = None,
) -> AteResult:
    """Estimate the effect with Do-PFN, Prior Labs' causal foundation model.

    Do-PFN (Robertson et al., 2025) is pre-trained on structural causal models
    to predict interventional outcomes directly, without a causal graph. Its
    paper uses exactly the S- and T-learners in this module as baselines and
    reports that they degrade when mediators are present while Do-PFN holds
    up. This wrapper exists so the two can be compared on the same data.

    It is optional and experimental. Do-PFN is a research checkout, not a
    package: clone https://github.com/jr2021/Do-PFN, install its requirements
    (they include torch), and point ``ADLIFT_DOPFN_PATH`` or ``repo_path`` at
    the clone. The model loads its weights from paths relative to that
    directory, so the working directory is switched for the call. Do-PFN is
    fitted on the model scale of the outcome (``log1p`` for counts).
    """
    import os
    import sys

    path = repo_path or os.environ.get("ADLIFT_DOPFN_PATH")
    if not path or not os.path.isdir(path):
        raise RuntimeError(
            "Do-PFN is not available. Clone https://github.com/jr2021/Do-PFN, install its "
            "requirements, and set ADLIFT_DOPFN_PATH to the clone."
        )

    columns = _columns(spec, frame, adjustment)
    design = frame[columns].copy()
    for column in design.columns:
        if not pd.api.types.is_numeric_dtype(design[column]):
            design[column] = design[column].astype("category").cat.codes
    treatment = (frame[spec.treatment_column] == "llm").to_numpy(dtype=float)
    # Do-PFN's convention: the treatment is column zero.
    X = np.column_stack([treatment, design.to_numpy(dtype=float)])
    y_model = spec.outcome.forward(frame[spec.outcome.column].to_numpy(dtype=float))

    previous_cwd = os.getcwd()
    inserted = path not in sys.path
    if inserted:
        sys.path.insert(0, path)
    try:
        os.chdir(path)
        from scripts.transformer_prediction_interface.base import DoPFNRegressor  # type: ignore

        model = DoPFNRegressor()
        model.fit(X, y_model)
        y1 = np.asarray(model.predict_cid(X.copy(), 1), dtype=float).ravel()
        y0 = np.asarray(model.predict_cid(X.copy(), 0), dtype=float).ravel()
    finally:
        os.chdir(previous_cwd)
        if inserted:
            sys.path.remove(path)

    natural = spec.outcome.inverse(y1) - spec.outcome.inverse(y0)
    return _result(
        spec=spec,
        estimator="do-pfn",
        adjustment=adjustment,
        natural=natural,
        model_scale=y1 - y0,
        groups=frame[spec.group_column].to_numpy(),
        n_boot=400,
        alpha=0.05,
        seed=0,
        notes=["Do-PFN research checkout; interval is a cluster bootstrap of its CATEs."],
    )
