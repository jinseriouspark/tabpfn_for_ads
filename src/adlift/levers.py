"""Which levers actually move the outcome, and by how much.

"Human or LLM" is one lever. A person publishing a post pulls a dozen more:
the hour, the weekday, whether there is an image, whether it is a reply, how
long it runs, whether it opens with a question, carries a number, a link, a
hashtag, an emoji, what it is about. The question a creator asks is not "is
the LLM better" but "what should I change to make this one land".

This module answers that with the same machinery as :mod:`adlift.causal`:
fit one model on the structured attributes, then score every post under each
alternative setting of one lever and read off the average change, with a
cluster bootstrap over weeks for the interval. The raw text is deliberately
left out of this model. Flipping ``has_question`` while the text still ends
in a full stop would ask the model to believe two contradictory things.

The adjustment set differs by the kind of lever:

Context levers (hour, weekday, media, reply, topic)
    Decided before writing. Adjust for the other context columns and for the
    text attributes, which are downstream of none of them.

Text levers (length, question, number, hashtags, link, emoji, CTA)
    Adjust for context and for the *other* text attributes, so the reported
    number is the change from that one edit with everything else held.

Author
    Adjust for context only, as in :mod:`adlift.causal`; the text attributes
    are mediators of this lever.

Every estimate is a model-based counterfactual, not an experiment. The overlap
caveat from :mod:`adlift.causal` applies: a lever setting that never occurs
in the data cannot be evaluated, and the table says how many rows support
each comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from adlift.causal import ModelFactory, _cluster_bootstrap_cis, _factory
from adlift.schema import THREADS_SPEC, DatasetSpec

LeverKind = Literal["binary", "categorical", "numeric", "author"]


@dataclass(slots=True)
class Lever:
    """One controllable attribute and how to vary it."""

    column: str
    kind: LeverKind
    #: Alternative settings to evaluate. For ``numeric`` these are values to
    #: set; for ``categorical`` levels; ignored for ``binary`` and ``author``.
    levels: tuple[Any, ...] = ()
    #: Other columns that move together with this one, as ``{col: fn(value)}``.
    #: ``word_count`` drags ``char_count`` along, for instance.
    coupled: dict[str, Any] = field(default_factory=dict)
    label: str = ""


HOUR_BUCKETS: dict[str, tuple[int, ...]] = {
    "night (0-6)": (1, 2, 3, 4, 5),
    "morning (7-11)": (8, 9, 10),
    "midday (12-16)": (12, 13, 14, 15),
    "evening (17-23)": (18, 19, 20, 21, 22),
}


def default_levers(spec: DatasetSpec = THREADS_SPEC) -> list[Lever]:
    """The levers a Threads creator controls, in the order a report lists them."""
    if spec.name != "threads":
        return [Lever("author", "author", label="LLM instead of human")]
    return [
        Lever("posted_hour", "categorical", levels=tuple(HOUR_BUCKETS), label="posting time"),
        Lever("weekday", "categorical", levels=("weekday", "weekend"), label="day of week"),
        Lever("has_media", "binary", label="attach an image or video"),
        Lever("is_reply", "binary", label="post as a reply"),
        Lever("topic", "categorical", levels=(), label="topic"),
        Lever(
            "word_count",
            "numeric",
            levels=(15, 30, 45, 70),
            coupled={"char_count": lambda w: int(round(w * 5.6))},
            label="length in words",
        ),
        Lever("has_question", "binary", label="open with a question"),
        Lever("has_numeric_claim", "binary", label="include a number"),
        Lever("n_hashtags", "numeric", levels=(0, 2, 4), label="hashtags"),
        Lever("has_link", "binary", label="include a link"),
        Lever("has_emoji", "binary", label="use an emoji"),
        Lever("has_cta", "binary", label="add a call to action"),
        Lever("author", "author", label="LLM instead of human"),
    ]


def _design_columns(spec: DatasetSpec, frame: pd.DataFrame) -> list[str]:
    columns = list(spec.context_columns) + list(spec.creative_columns) + [spec.treatment_column]
    return [c for c in columns if c in frame.columns]


def _set_lever(design: pd.DataFrame, lever: Lever, value: Any) -> pd.DataFrame:
    out = design.copy()
    if lever.column == "posted_hour" and isinstance(value, str) and value in HOUR_BUCKETS:
        hours = HOUR_BUCKETS[value]
        out[lever.column] = hours[len(hours) // 2]
    elif lever.column == "weekday" and value in ("weekday", "weekend"):
        out[lever.column] = "Wed" if value == "weekday" else "Sat"
    else:
        out[lever.column] = value
    for column, fn in lever.coupled.items():
        if column in out.columns:
            out[column] = fn(value) if callable(fn) else fn
    return out


def _observed_level(frame: pd.DataFrame, lever: Lever) -> Any:
    """The reference setting: what the account does most often."""
    column = frame[lever.column]
    if lever.column == "posted_hour":
        counts = {
            name: int(column.isin(range(min(h), max(h) + 1)).sum())
            for name, h in HOUR_BUCKETS.items()
        }
        return max(counts, key=counts.get)
    if lever.column == "weekday":
        return "weekend" if column.isin(["Sat", "Sun"]).mean() > 0.5 else "weekday"
    if lever.kind == "numeric":
        return float(column.median())
    return column.mode().iloc[0]


def _support(frame: pd.DataFrame, lever: Lever, value: Any) -> int:
    column = frame[lever.column]
    if lever.column == "posted_hour" and isinstance(value, str) and value in HOUR_BUCKETS:
        hours = HOUR_BUCKETS[value]
        return int(column.between(min(hours), max(hours)).sum())
    if lever.column == "weekday" and value in ("weekday", "weekend"):
        weekend = column.isin(["Sat", "Sun"])
        return int(weekend.sum() if value == "weekend" else (~weekend).sum())
    if lever.kind == "numeric":
        return int(
            (np.abs(column.astype(float) - float(value)) <= max(1.0, 0.25 * float(value))).sum()
        )
    return int((column == value).sum())


def lever_analysis(
    frame: pd.DataFrame,
    *,
    spec: DatasetSpec = THREADS_SPEC,
    levers: list[Lever] | None = None,
    model_factory: ModelFactory | None = None,
    n_boot: int = 200,
    alpha: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """Estimate the effect of every lever, relative to what the account usually does.

    One model is fitted on the structured attributes (no raw text). For each
    lever and each alternative setting, every post is re-scored with that
    setting and the mean change from the reference setting is reported, on
    the natural scale and, for log outcomes, as a percentage.

    Returns:
        A table with one row per (lever, alternative), sorted by absolute
        effect: ``lever``, ``from``, ``to``, ``effect``, ``ci_low``,
        ``ci_high``, ``ratio``, ``support`` (rows observed at the alternative),
        ``significant``.
    """
    levers = levers or default_levers(spec)
    build = _factory(model_factory, spec)
    outcome = spec.outcome
    columns = _design_columns(spec, frame)
    design = frame[columns].copy()
    groups = frame[spec.group_column].to_numpy()

    model = build().fit(
        design, frame[outcome.column].to_numpy(dtype=float), groups=frame[spec.group_column]
    )
    baseline_model_scale = model.predict(design, scale="model")

    rows: list[dict[str, Any]] = []
    for lever in levers:
        if lever.column not in design.columns:
            continue
        reference = _observed_level(frame, lever)
        if lever.kind == "binary":
            candidates = [0, 1]
        elif lever.kind == "author":
            candidates = ["human", "llm"]
        elif lever.kind == "categorical":
            candidates = (
                list(lever.levels) if lever.levels else list(pd.unique(frame[lever.column]))
            )
        else:
            candidates = list(lever.levels)

        ref_pred = model.predict(_set_lever(design, lever, reference), scale="model")
        for value in candidates:
            if value == reference:
                continue
            alt_pred = model.predict(_set_lever(design, lever, value), scale="model")
            natural = outcome.inverse(alt_pred) - outcome.inverse(ref_pred)
            model_scale = alt_pred - ref_pred
            (low, high), (low_m, high_m) = _cluster_bootstrap_cis(
                [natural, model_scale], groups, n_boot=n_boot, alpha=alpha, seed=seed
            )
            ate_model = float(model_scale.mean())
            ratio = float(np.expm1(ate_model)) if outcome.transform == "log1p" else None
            rows.append(
                {
                    "lever": lever.label or lever.column,
                    "column": lever.column,
                    "from": reference,
                    "to": value,
                    "effect": float(natural.mean()),
                    "ci_low": low,
                    "ci_high": high,
                    "ratio": ratio,
                    "ratio_low": float(np.expm1(low_m)) if ratio is not None else None,
                    "ratio_high": float(np.expm1(high_m)) if ratio is not None else None,
                    "support": _support(frame, lever, value),
                    "significant": not (low <= 0.0 <= high),
                }
            )

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["abs_effect"] = table["effect"].abs()
    table = table.sort_values("abs_effect", ascending=False).drop(columns="abs_effect")
    table.attrs["baseline_prediction"] = float(outcome.inverse(baseline_model_scale).mean())
    return table.reset_index(drop=True)


def suggest_for_post(
    frame: pd.DataFrame,
    post_row: dict[str, Any],
    *,
    spec: DatasetSpec = THREADS_SPEC,
    levers: list[Lever] | None = None,
    model_factory: ModelFactory | None = None,
    top: int = 5,
) -> pd.DataFrame:
    """For one draft, which single change raises its predicted outcome most.

    The draft is scored as written, then once per alternative lever setting.
    Text levers are evaluated as attribute flips (the model does not see the
    words), so a suggestion like "open with a question" is a prompt for the
    rewrite loop, not a rewrite itself.
    """
    levers = levers or default_levers(spec)
    build = _factory(model_factory, spec)
    outcome = spec.outcome
    columns = _design_columns(spec, frame)
    design = frame[columns].copy()
    model = build().fit(
        design, frame[outcome.column].to_numpy(dtype=float), groups=frame[spec.group_column]
    )

    draft = pd.DataFrame([{c: post_row.get(c) for c in columns}])
    as_is = float(outcome.inverse(model.predict(draft, scale="model"))[0])

    rows = []
    for lever in levers:
        if lever.column not in draft.columns:
            continue
        current = draft[lever.column].iloc[0]
        if lever.kind == "binary":
            candidates = [1 - int(current)]
        elif lever.kind == "author":
            candidates = ["llm" if current == "human" else "human"]
        elif lever.kind == "categorical":
            candidates = [
                v for v in (lever.levels or pd.unique(frame[lever.column])) if v != current
            ]
        else:
            candidates = [v for v in lever.levels if v != current]
        for value in candidates:
            predicted = float(
                outcome.inverse(model.predict(_set_lever(draft, lever, value), scale="model"))[0]
            )
            rows.append(
                {
                    "change": f"{lever.label or lever.column}: {current} -> {value}",
                    "column": lever.column,
                    "to": value,
                    "predicted": predicted,
                    "lift": predicted - as_is,
                    "lift_pct": (predicted / as_is - 1.0) if as_is else float("nan"),
                }
            )
    table = pd.DataFrame(rows).sort_values("lift", ascending=False).reset_index(drop=True)
    table.attrs["as_is"] = as_is
    return table.head(top)
