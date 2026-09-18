"""Cold-start evaluation: can we rank creatives before they ever serve?

The question an ad team actually faces is not "what is this creative's exact
click-through rate". It is "I have eight new creatives and one budget; which do
I fund?" Today that is answered by spending money: split the budget evenly,
serve every creative enough impressions to read a signal, then scale the
winner. Most of that spend buys nothing but the knowledge that seven creatives
were worse.

So the evaluation here holds out whole campaigns, never individual rows. A
held-out campaign is a genuine cold start: the model has seen none of its
creatives, none of its budget, none of its audience. Success is measured the
way the budget is spent.

``recall_at_k``
    Of the held-out campaigns, the share whose true best creative is inside the
    model's top ``k`` picks. This is the number that licenses cutting the test.

``top1_regret``
    How much click-through rate is given up by scaling the model's first pick
    instead of the true winner, as a fraction of the winner's rate.

``exploration_saved``
    The share of the exploration budget that goes unspent if only the model's
    top ``k`` creatives are tested rather than all of them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from adlift.model import CTRModel
from adlift.schema import FEATURE_COLUMNS, GROUP_COLUMN

#: Column holding the noise-free rate, when the frame came from the generator.
TRUTH_COLUMN = "_truth_ctr_noiseless"


@dataclass(slots=True)
class ColdStartResult:
    """Scores for one model on held-out campaigns."""

    name: str
    backend: str
    n_test_campaigns: int
    spearman: float
    mae: float
    top1_regret: float
    recall_at_k: dict[int, float] = field(default_factory=dict)
    exploration_saved: dict[int, float] = field(default_factory=dict)
    fit_seconds: float = 0.0
    predict_seconds: float = 0.0

    def as_row(self) -> dict:
        row = asdict(self)
        row.pop("recall_at_k")
        row.pop("exploration_saved")
        return row


def _truth(frame: pd.DataFrame) -> np.ndarray:
    """Prefer the noise-free rate when available; fall back to observed."""
    if TRUTH_COLUMN in frame.columns and frame[TRUTH_COLUMN].notna().all():
        return frame[TRUTH_COLUMN].to_numpy(dtype=float)
    return frame["ctr"].to_numpy(dtype=float)


def _campaign_scores(
    frame: pd.DataFrame, predicted: np.ndarray, k_values: tuple[int, ...]
) -> tuple[dict[int, float], dict[int, float], float]:
    """Per-campaign selection quality, averaged over campaigns."""
    truth = _truth(frame)
    work = pd.DataFrame(
        {
            GROUP_COLUMN: frame[GROUP_COLUMN].to_numpy(),
            "predicted": predicted,
            "truth": truth,
        }
    )

    hits: dict[int, list[float]] = {k: [] for k in k_values}
    saved: dict[int, list[float]] = {k: [] for k in k_values}
    regrets: list[float] = []

    for _, group in work.groupby(GROUP_COLUMN, sort=False):
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


def evaluate_cold_start(
    frame: pd.DataFrame,
    model: CTRModel,
    *,
    name: str | None = None,
    n_splits: int = 4,
    k_values: tuple[int, ...] = (1, 2, 3),
    seed: int = 0,
) -> ColdStartResult:
    """Score one model with grouped cross-validation over campaigns.

    Args:
        frame: The canonical creative table, with outcomes.
        model: An unfitted model. It is refitted inside every fold.
        name: Label for reports. Defaults to the model's backend.
        n_splits: Number of campaign folds.
        k_values: Shortlist sizes to report recall and savings for.
        seed: Unused by ``GroupKFold`` but kept so callers can pass one.

    Returns:
        A ``ColdStartResult`` aggregated across folds.
    """
    del seed
    frame = frame.reset_index(drop=True)
    groups = frame[GROUP_COLUMN].to_numpy()
    features = frame[FEATURE_COLUMNS]
    target = frame["ctr"].to_numpy(dtype=float)

    splitter = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    predictions = np.full(len(frame), np.nan)
    fit_seconds = 0.0
    predict_seconds = 0.0

    for train_index, test_index in splitter.split(features, target, groups):
        fold_model = CTRModel(
            backend=model.backend,
            model_version=model.model_version,
            thinking_effort=model.thinking_effort,
            n_estimators=model.n_estimators,
            random_state=model.random_state,
        )
        fold_model.fit(
            features.iloc[train_index],
            target[train_index],
            groups=frame[GROUP_COLUMN].iloc[train_index],
        )
        predictions[test_index] = fold_model.predict(features.iloc[test_index])
        if fold_model.report:
            fit_seconds += fold_model.report.fit_seconds
            predict_seconds += fold_model.report.predict_seconds

    from scipy.stats import spearmanr

    truth = _truth(frame)
    recall, savings, regret = _campaign_scores(frame, predictions, k_values)

    return ColdStartResult(
        name=name or model.backend,
        backend=model.backend,
        n_test_campaigns=int(pd.Series(groups).nunique()),
        spearman=float(spearmanr(predictions, truth).statistic),
        mae=float(np.mean(np.abs(predictions - truth))),
        top1_regret=regret,
        recall_at_k=recall,
        exploration_saved=savings,
        fit_seconds=fit_seconds / splitter.get_n_splits(),
        predict_seconds=predict_seconds / splitter.get_n_splits(),
    )


def random_baseline(
    frame: pd.DataFrame, *, k_values: tuple[int, ...] = (1, 2, 3), seed: int = 0
) -> ColdStartResult:
    """What you get with no model at all: pick a creative at random.

    This is the honest floor. Any model has to beat it by enough to justify
    trusting it with a budget decision.
    """
    rng = np.random.default_rng(seed)
    predicted = rng.random(len(frame))
    recall, savings, regret = _campaign_scores(frame, predicted, k_values)
    from scipy.stats import spearmanr

    return ColdStartResult(
        name="random",
        backend="none",
        n_test_campaigns=int(frame[GROUP_COLUMN].nunique()),
        spearman=float(spearmanr(predicted, _truth(frame)).statistic),
        mae=float(np.mean(np.abs(predicted - _truth(frame)))),
        top1_regret=regret,
        recall_at_k=recall,
        exploration_saved=savings,
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
