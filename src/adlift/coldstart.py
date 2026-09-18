"""Cold-start evaluation: can we rank creatives before they ever run?

The question an ad team actually faces is not "what is this creative's exact
click-through rate". It is "I have eight new creatives and one budget; which do
I fund?" Today that is answered by spending money: split the budget evenly,
serve every creative enough impressions to read a signal, then scale the
winner. Most of that spend buys nothing but the knowledge that seven creatives
were worse. A social account faces the same choice every week: which of this
week's drafts goes out first.

So the evaluation here holds out whole groups, never individual rows. For an
ad account the group is a campaign and the split is random over campaigns. For
a social timeline the group is a week and the split is *temporal*: fit on the
past, rank the next week's posts. Either way a held-out group is a genuine cold
start.

``recall_at_k``
    Of the held-out groups, the share whose true best creative is inside the
    model's top ``k`` picks. This is the number that licenses cutting the test.

``top1_regret``
    How much outcome is given up by scaling the model's first pick instead of
    the true winner, as a fraction of the winner's value.

``exploration_saved``
    The share of the exploration budget that goes unspent if only the model's
    top ``k`` creatives are tested rather than all of them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, TimeSeriesSplit

from adlift.model import CTRModel
from adlift.schema import AD_SPEC, DatasetSpec

Split = Literal["group", "temporal"]


@dataclass(slots=True)
class ColdStartResult:
    """Scores for one model on held-out groups."""

    name: str
    backend: str
    n_test_groups: int
    spearman: float
    mae: float
    top1_regret: float
    recall_at_k: dict[int, float] = field(default_factory=dict)
    exploration_saved: dict[int, float] = field(default_factory=dict)
    fit_seconds: float = 0.0
    predict_seconds: float = 0.0
    split: str = "group"
    outcome_label: str = "outcome"

    @property
    def n_test_campaigns(self) -> int:  # backwards-compatible alias
        return self.n_test_groups

    def as_row(self) -> dict:
        row = asdict(self)
        row.pop("recall_at_k")
        row.pop("exploration_saved")
        return row


def _truth(frame: pd.DataFrame, spec: DatasetSpec) -> np.ndarray:
    """Prefer the noise-free outcome when a generator left one; else observed."""
    column = spec.noiseless_truth_column
    if column in frame.columns and frame[column].notna().all():
        return frame[column].to_numpy(dtype=float)
    return frame[spec.outcome.column].to_numpy(dtype=float)


def _group_scores(
    frame: pd.DataFrame,
    predicted: np.ndarray,
    spec: DatasetSpec,
    k_values: tuple[int, ...],
    *,
    mask: np.ndarray | None = None,
) -> tuple[dict[int, float], dict[int, float], float]:
    """Per-group selection quality, averaged over groups."""
    truth = _truth(frame, spec)
    work = pd.DataFrame(
        {"group": frame[spec.group_column].to_numpy(), "predicted": predicted, "truth": truth}
    )
    if mask is not None:
        work = work[mask]

    hits: dict[int, list[float]] = {k: [] for k in k_values}
    saved: dict[int, list[float]] = {k: [] for k in k_values}
    regrets: list[float] = []

    for _, group in work.groupby("group", sort=False):
        if len(group) < 2:
            continue
        order = group["predicted"].to_numpy().argsort()[::-1]
        truth_values = group["truth"].to_numpy()
        best_index = int(truth_values.argmax())
        best_value = float(truth_values[best_index])

        picked = float(truth_values[order[0]])
        regrets.append((best_value - picked) / best_value if best_value > 0 else 0.0)

        size = len(group)
        for k in k_values:
            top = set(order[: min(k, size)].tolist())
            hits[k].append(float(best_index in top))
            saved[k].append(max(0.0, (size - min(k, size)) / size))

    recall = {k: float(np.mean(v)) if v else float("nan") for k, v in hits.items()}
    savings = {k: float(np.mean(v)) if v else float("nan") for k, v in saved.items()}
    return recall, savings, float(np.mean(regrets)) if regrets else float("nan")


def _folds(groups: np.ndarray, n_splits: int, split: Split):
    """Yield (train_index, test_index) over rows, splitting by group."""
    unique = np.unique(groups)
    n_splits = max(2, min(n_splits, len(unique) - 1))
    if split == "group":
        splitter = GroupKFold(n_splits=n_splits)
        yield from splitter.split(np.zeros(len(groups)), groups=groups)
        return
    # Temporal: groups sort in time order (week labels are zero-padded), and
    # every fold fits on everything before its test block.
    ordered = np.sort(unique)
    for train_g, test_g in TimeSeriesSplit(n_splits=n_splits).split(ordered):
        train_set, test_set = set(ordered[train_g]), set(ordered[test_g])
        yield (
            np.flatnonzero([g in train_set for g in groups]),
            np.flatnonzero([g in test_set for g in groups]),
        )


def evaluate_cold_start(
    frame: pd.DataFrame,
    model: CTRModel,
    *,
    name: str | None = None,
    n_splits: int = 4,
    k_values: tuple[int, ...] = (1, 2, 3),
    split: Split | None = None,
    seed: int = 0,
) -> ColdStartResult:
    """Score one model on held-out groups.

    Args:
        frame: The canonical table, with outcomes.
        model: An unfitted model. It is refitted inside every fold and its
            spec decides the columns, the outcome scale and the default split.
        name: Label for reports. Defaults to the model's backend.
        n_splits: Number of folds.
        k_values: Shortlist sizes to report recall and savings for.
        split: ``group`` (random over groups) or ``temporal`` (fit on the past,
            rank the next block). Defaults to the spec's choice.
        seed: Unused by the splitters but kept so callers can pass one.

    Returns:
        A ``ColdStartResult`` aggregated across folds. With a temporal split,
        the earliest block is never scored because nothing precedes it.
    """
    del seed
    spec = model.spec
    split = split or spec.default_split
    frame = frame.reset_index(drop=True)
    groups = frame[spec.group_column].to_numpy()
    features = frame[spec.feature_columns]
    target = frame[spec.outcome.column].to_numpy(dtype=float)

    predictions = np.full(len(frame), np.nan)
    fit_seconds = 0.0
    predict_seconds = 0.0
    n_folds = 0

    for train_index, test_index in _folds(groups, n_splits, split):
        fold_model = model.clone()
        fold_model.fit(
            features.iloc[train_index],
            target[train_index],
            groups=frame[spec.group_column].iloc[train_index],
        )
        predictions[test_index] = fold_model.predict(features.iloc[test_index])
        if fold_model.report:
            fit_seconds += fold_model.report.fit_seconds
            predict_seconds += fold_model.report.predict_seconds
        n_folds += 1

    from scipy.stats import spearmanr

    scored = ~np.isnan(predictions)
    truth = _truth(frame, spec)
    recall, savings, regret = _group_scores(frame, predictions, spec, k_values, mask=scored)

    return ColdStartResult(
        name=name or model.backend,
        backend=model.backend,
        n_test_groups=int(pd.Series(groups[scored]).nunique()),
        spearman=float(spearmanr(predictions[scored], truth[scored]).statistic),
        mae=float(np.mean(np.abs(predictions[scored] - truth[scored]))),
        top1_regret=regret,
        recall_at_k=recall,
        exploration_saved=savings,
        fit_seconds=fit_seconds / max(n_folds, 1),
        predict_seconds=predict_seconds / max(n_folds, 1),
        split=split,
        outcome_label=spec.outcome.label,
    )


def random_baseline(
    frame: pd.DataFrame,
    *,
    spec: DatasetSpec = AD_SPEC,
    k_values: tuple[int, ...] = (1, 2, 3),
    seed: int = 0,
) -> ColdStartResult:
    """What you get with no model at all: pick a creative at random.

    This is the honest floor. Any model has to beat it by enough to justify
    trusting it with a budget decision.
    """
    rng = np.random.default_rng(seed)
    predicted = rng.random(len(frame))
    recall, savings, regret = _group_scores(frame, predicted, spec, k_values)
    from scipy.stats import spearmanr

    truth = _truth(frame, spec)
    return ColdStartResult(
        name="random",
        backend="none",
        n_test_groups=int(frame[spec.group_column].nunique()),
        spearman=float(spearmanr(predicted, truth).statistic),
        mae=float(np.mean(np.abs(predicted - truth))),
        top1_regret=regret,
        recall_at_k=recall,
        exploration_saved=savings,
        split="none",
        outcome_label=spec.outcome.label,
    )


def comparison_table(results: list[ColdStartResult], k: int = 2) -> pd.DataFrame:
    """Lay several models side by side, sorted by top-1 regret."""
    rows = []
    for result in results:
        rows.append(
            {
                "Model": result.name,
                "Spearman": round(result.spearman, 4),
                "Top-1 regret": round(result.top1_regret, 4),
                f"Recall@{k}": round(result.recall_at_k.get(k, float("nan")), 4),
                f"Exploration saved@{k}": round(result.exploration_saved.get(k, float("nan")), 4),
                "Fit s": round(result.fit_seconds, 2),
            }
        )
    return pd.DataFrame(rows).sort_values("Top-1 regret").reset_index(drop=True)


def learning_curve(
    frame: pd.DataFrame,
    model: CTRModel,
    *,
    start: int = 5,
    step: int = 5,
    max_context: int | None = None,
) -> pd.DataFrame:
    """How quickly the model becomes useful as an account posts.

    This is the cold start as it is actually lived: the first ``start`` rows
    are all the model knows, it ranks the next ``step`` rows, then those join
    the context and it ranks the block after that. Rows are taken in time
    order (``timestamp`` when present, otherwise group order, otherwise row
    order).

    TabPFN's whole argument is that it is usable in the first row of this
    table, where a gradient-boosted model is not.

    Returns:
        One row per step: ``n_context``, ``spearman`` (within the next
        block), ``top1_regret`` (the block treated as one shortlist),
        ``mae`` and ``fit_seconds``.
    """
    from scipy.stats import spearmanr

    spec = model.spec
    ordered = frame.copy()
    if "timestamp" in ordered.columns:
        ordered = ordered.sort_values("timestamp", kind="stable")
    elif spec.group_column in ordered.columns:
        ordered = ordered.sort_values(spec.group_column, kind="stable")
    ordered = ordered.reset_index(drop=True)

    features = ordered[spec.feature_columns]
    target = ordered[spec.outcome.column].to_numpy(dtype=float)
    truth = _truth(ordered, spec)
    limit = min(len(ordered), max_context or len(ordered))

    rows = []
    n = start
    while n + 2 <= limit:
        block = slice(n, min(n + step, limit))
        fold_model = model.clone()
        fold_model.fit(features.iloc[:n], target[:n], groups=ordered[spec.group_column].iloc[:n])
        predicted = fold_model.predict(features.iloc[block])
        block_truth = truth[block]
        best = int(block_truth.argmax())
        picked = int(np.argmax(predicted))
        regret = (
            (block_truth[best] - block_truth[picked]) / block_truth[best]
            if block_truth[best] > 0
            else 0.0
        )
        rho = spearmanr(predicted, block_truth).statistic if len(block_truth) >= 3 else float("nan")
        rows.append(
            {
                "n_context": n,
                "block_size": int(block.stop - block.start),
                "spearman": float(rho),
                "top1_regret": float(regret),
                "mae": float(np.mean(np.abs(predicted - block_truth))),
                "fit_seconds": fold_model.report.fit_seconds if fold_model.report else float("nan"),
            }
        )
        n += step
    return pd.DataFrame(rows)
