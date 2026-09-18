"""Turn results into a report a stakeholder can read.

Figures and one Markdown document. The figures use a validated
colour-blind-safe palette, thin marks and direct labels, and never a dual axis.
The document leads with the decision each result supports, and carries every
caveat next to the number it qualifies. Everything reads its labels off the
dataset spec, so the same code reports CTR points for ad creatives and view
ratios for social posts.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from adlift.causal import AteResult  # noqa: E402
from adlift.coldstart import ColdStartResult, comparison_table  # noqa: E402
from adlift.schema import AD_SPEC, DatasetSpec  # noqa: E402

# Categorical slots from the reference palette, in fixed order. Slot 1 is the
# hero (TabPFN), slot 2 the baseline, slot 3 a third model, slot 4 "no model".
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _scale(spec: DatasetSpec) -> tuple[float, str]:
    """Multiplier and axis unit for the outcome's natural scale."""
    if spec.outcome.effect_unit == "pp":
        return 100.0, f"{spec.outcome.label}, percentage points"
    return 1.0, spec.outcome.label


def _fmt(spec: DatasetSpec, value: float, *, ratio: float | None = None) -> str:
    return spec.outcome.format_effect(value, ratio=ratio)


def plot_cold_start(results: list[ColdStartResult], path: str | Path, *, k: int = 2) -> Path:
    """Two panels: how much outcome is lost by trusting each model's pick, and
    how often the true winner survives its shortlist."""
    path = Path(path)
    ordered = sorted(results, key=lambda r: r.top1_regret)
    names = [r.name for r in ordered]
    colours = [SERIES[3] if r.backend == "none" else SERIES[i % 3] for i, r in enumerate(ordered)]
    label = ordered[0].outcome_label if ordered else "outcome"

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), facecolor=SURFACE)
    fig.subplots_adjust(wspace=0.5)

    regret = [r.top1_regret * 100 for r in ordered]
    bars = axes[0].barh(names, regret, color=colours, height=0.55)
    axes[0].bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9, color=INK)
    axes[0].set_title(
        f"{label} given up by scaling the model's first pick", fontsize=10, color=INK, loc="left"
    )
    axes[0].set_xlabel("top-1 regret (lower is better)", fontsize=9, color=INK_SOFT)
    axes[0].set_xlim(0, max(regret) * 1.3 if regret else 1)
    axes[0].invert_yaxis()

    recall = [r.recall_at_k.get(k, float("nan")) * 100 for r in ordered]
    bars = axes[1].barh(names, recall, color=colours, height=0.55)
    axes[1].bar_label(bars, fmt="%.0f%%", padding=4, fontsize=9, color=INK)
    axes[1].set_title(
        f"Groups whose true winner is in the top {k}", fontsize=10, color=INK, loc="left"
    )
    axes[1].set_xlabel(f"recall@{k} (higher is better)", fontsize=9, color=INK_SOFT)
    axes[1].set_xlim(0, 115)
    axes[1].invert_yaxis()

    for ax in axes:
        _style(ax)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_effects(
    analysis: dict[str, Any],
    path: str | Path,
    *,
    truth_total: float | None = None,
    truth_direct: float | None = None,
    truth_ratios: tuple[float | None, float | None] = (None, None),
) -> Path:
    """Dot-and-interval chart of every estimate against the planted truth.

    Outcomes modelled on a log scale are drawn as percentage changes with the
    model-scale interval, which is the interval the report headlines. The
    natural-scale mean of a heavy-tailed outcome is carried by a few extreme
    rows and would put the chart at odds with the table.
    """
    path = Path(path)
    spec: DatasetSpec = analysis.get("spec", AD_SPEC)
    as_ratio = spec.outcome.effect_unit == "ratio"
    mult, unit = (100.0, f"{spec.outcome.label}, % change") if as_ratio else _scale(spec)

    def point(result: AteResult) -> tuple[float, float | None, float | None]:
        if as_ratio and result.ratio is not None:
            return result.ratio, result.ratio_low, result.ratio_high
        return result.ate, result.ci_low, result.ci_high

    naive = analysis.get("naive_ratio") if as_ratio else analysis["naive_difference"]
    rows: list[tuple[str, float, float | None, float | None, str]] = [
        ("Naive difference (pivot table)", float(naive), None, None, SERIES[3]),
        ("T-learner, total effect (headline)", *point(analysis["total_effect"]), SERIES[0]),
        ("S-learner, total effect", *point(analysis["cross_check"]), SERIES[0]),
        (
            "S-learner, direct effect (mediator-adjusted)",
            *point(analysis["direct_effect"]),
            SERIES[1],
        ),
    ]
    if isinstance(analysis.get("dopfn"), AteResult):
        rows.append(("Do-PFN, total effect", *point(analysis["dopfn"]), SERIES[2]))

    truth_t = truth_ratios[0] if as_ratio else truth_total
    truth_d = truth_ratios[1] if as_ratio else truth_direct

    fig, ax = plt.subplots(figsize=(9, 0.6 * len(rows) + 1.6), facecolor=SURFACE)
    y = np.arange(len(rows))
    for i, (_label, ate, lo, hi, colour) in enumerate(rows):
        if lo is not None and hi is not None:
            ax.plot(
                [lo * mult, hi * mult], [i, i], color=colour, linewidth=2, solid_capstyle="round"
            )
        ax.plot(
            ate * mult,
            i,
            "o",
            color=colour,
            markersize=8,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
        )
        ax.annotate(
            f"{ate * mult:+.3g}" + ("%" if as_ratio else ""),
            (ate * mult, i),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8.5,
            color=INK,
        )

    ax.axvline(0, color=INK_SOFT, linewidth=1, linestyle=":")
    label_y = len(rows) - 0.35
    if truth_t is not None:
        ax.axvline(truth_t * mult, color=SERIES[0], linewidth=1.2, linestyle="--", alpha=0.7)
        ax.text(
            truth_t * mult,
            label_y,
            "true total",
            ha="center",
            va="top",
            fontsize=8,
            color=SERIES[0],
        )
    if truth_d is not None:
        ax.axvline(truth_d * mult, color=SERIES[1], linewidth=1.2, linestyle="--", alpha=0.7)
        ax.text(
            truth_d * mult,
            label_y,
            "true direct",
            ha="center",
            va="top",
            fontsize=8,
            color=SERIES[1],
        )
    ax.set_ylim(len(rows) - 0.2, -0.6)

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9, color=INK)
    ax.set_xlabel(
        f"effect of LLM authorship on {unit} (95% cluster-bootstrap CI)", fontsize=9, color=INK_SOFT
    )
    ax.set_title(
        "What the pivot table says vs. what the data supports", fontsize=10, color=INK, loc="left"
    )
    _style(ax)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_segments(analysis: dict[str, Any], path: str | Path) -> Path:
    """Effect by segment: where the average hides a sign change."""
    path = Path(path)
    spec: DatasetSpec = analysis.get("spec", AD_SPEC)
    mult, unit = _scale(spec)
    segments: pd.DataFrame = analysis["segments"]
    column = segments.columns[0]
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(segments) + 1.4), facecolor=SURFACE)
    values = segments["effect"].to_numpy() * mult
    colours = [SERIES[0] if v >= 0 else SERIES[1] for v in values]
    bars = ax.barh(segments[column].astype(str), values, color=colours, height=0.55)
    ax.bar_label(bars, labels=[f"{v:+.3g}" for v in values], padding=4, fontsize=9, color=INK)
    ax.axvline(0, color=INK_SOFT, linewidth=1)
    ax.invert_yaxis()
    span = max(abs(values.min()), abs(values.max()), 1e-3)
    ax.set_xlim(min(values.min(), 0) - 0.3 * span, max(values.max(), 0) + 0.3 * span)
    ax.set_xlabel(f"LLM effect on {unit}", fontsize=9, color=INK_SOFT)
    ax.set_title(f"Effect by {column}", fontsize=10, color=INK, loc="left")
    _style(ax)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_levers(
    levers: pd.DataFrame, path: str | Path, *, spec: DatasetSpec, top: int = 14
) -> Path:
    """Every lever's effect with its interval, largest first."""
    path = Path(path)
    mult, unit = _scale(spec)
    table = levers.head(top).iloc[::-1]
    use_ratio = "ratio" in table.columns and table["ratio"].notna().all()
    values = (table["ratio"] * 100).to_numpy() if use_ratio else table["effect"].to_numpy() * mult
    lows = (table["ratio_low"] * 100).to_numpy() if use_ratio else table["ci_low"].to_numpy() * mult
    highs = (
        (table["ratio_high"] * 100).to_numpy() if use_ratio else table["ci_high"].to_numpy() * mult
    )
    labels = [
        f"{lever}: {source} -> {target}"
        for lever, source, target in zip(table["lever"], table["from"], table["to"], strict=True)
    ]

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(table) + 1.6), facecolor=SURFACE)
    y = np.arange(len(table))
    for i, (v, lo, hi, sig) in enumerate(
        zip(values, lows, highs, table["significant"], strict=True)
    ):
        colour = (SERIES[0] if v >= 0 else SERIES[1]) if sig else SERIES[3]
        ax.plot([lo, hi], [i, i], color=colour, linewidth=2, solid_capstyle="round")
        ax.plot(v, i, "o", color=colour, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.axvline(0, color=INK_SOFT, linewidth=1, linestyle=":")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5, color=INK)
    ax.set_xlabel(
        f"change in {spec.outcome.label} ({'%' if use_ratio else unit}) vs. what the account usually does, 95% CI",
        fontsize=9,
        color=INK_SOFT,
    )
    ax.set_title(
        "Which levers move the outcome (grey: interval includes zero)",
        fontsize=10,
        color=INK,
        loc="left",
    )
    _style(ax)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_learning_curve(curves: dict[str, pd.DataFrame], path: str | Path) -> Path:
    """Ranking quality as the account accumulates posts, one line per model.

    Each block is a handful of posts, so a single block's rank correlation is
    noisy. The line is a three-block rolling mean; the faint dots are the raw
    blocks, kept so the smoothing is visible rather than hidden.
    """
    path = Path(path)
    fig, ax = plt.subplots(figsize=(8, 3.6), facecolor=SURFACE)
    for i, (name, curve) in enumerate(curves.items()):
        colour = SERIES[i % 3]
        raw = curve["spearman"].astype(float)
        smooth = raw.rolling(3, min_periods=1, center=True).mean()
        ax.plot(curve["n_context"], raw, "o", color=colour, alpha=0.35, markersize=4)
        ax.plot(curve["n_context"], smooth, color=colour, linewidth=2, label=name)
        last = curve.assign(smooth=smooth).dropna(subset=["smooth"]).tail(1)
        if not last.empty:
            ax.annotate(
                name,
                (last["n_context"].iloc[0], last["smooth"].iloc[0]),
                textcoords="offset points",
                xytext=(6, 0),
                fontsize=8.5,
                color=colour,
                va="center",
            )
    ax.axhline(0, color=INK_SOFT, linewidth=1, linestyle=":")
    ax.set_xlabel("posts in context", fontsize=9, color=INK_SOFT)
    ax.set_ylabel("Spearman on the next block", fontsize=9, color=INK_SOFT)
    ax.set_title(
        "Learning curve: how soon the ranking becomes useful", fontsize=10, color=INK, loc="left"
    )
    if len(curves) > 1:
        ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    _style(ax)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def render_markdown(
    *,
    cold_start: list[ColdStartResult],
    analysis: dict[str, Any],
    figures: dict[str, Path],
    truth_total: float | None = None,
    truth_direct: float | None = None,
    n_rows: int,
    n_groups: int,
    backend: str,
    levers: pd.DataFrame | None = None,
    curves: dict[str, pd.DataFrame] | None = None,
    truth_ratios: tuple[float | None, float | None] = (None, None),
) -> str:
    """Write the report. Decision first, evidence second, caveats attached."""
    spec: DatasetSpec = analysis.get("spec", AD_SPEC)
    total: AteResult = analysis["total_effect"]
    direct: AteResult = analysis["direct_effect"]
    cross: AteResult = analysis["cross_check"]
    overlap = analysis["overlap"]
    placebo = analysis["placebo"]
    mediation = analysis["mediation"]
    table = comparison_table(cold_start)
    outcome = spec.outcome.label
    group = spec.group_column

    direction = "hurts" if (total.ratio if total.ratio is not None else total.ate) < 0 else "helps"
    confident = total.significant and analysis["estimators_agree"] and placebo["ratio"] >= 3
    naive_ratio = analysis.get("naive_ratio")

    lines = [
        "# AdLift report",
        "",
        f"Dataset: `{spec.name}`. {n_rows} rows across {n_groups} {group}s. Outcome: **{outcome}**. "
        f"Model backend: `{backend}`.",
        "",
        "## Decision summary",
        "",
        f"- **Cold start.** Ranking new rows with `{table.iloc[0]['Model']}` before they run gives up "
        f"{table.iloc[0]['Top-1 regret'] * 100:.1f}% of the best achievable {outcome}, against "
        f"{table[table['Model'] == 'random']['Top-1 regret'].iloc[0] * 100:.1f}% for picking at random. "
        f"Testing only its top 2 keeps the true winner {table.iloc[0]['Recall@2'] * 100:.0f}% of the time "
        f"and leaves {table.iloc[0]['Exploration saved@2'] * 100:.0f}% of the exploration budget unspent.",
        f"- **Human vs LLM copy.** Handing the brief to an LLM {direction} {outcome} by "
        f"{_fmt(spec, total.ate, ratio=total.ratio)} (95% CI {total.interval()}). "
        f"The pivot-table answer was {_fmt(spec, analysis['naive_difference'], ratio=naive_ratio)}"
        + (
            ", the wrong sign."
            if np.sign(analysis["naive_difference"]) != np.sign(total.ate)
            else "."
        )
        + (
            " This estimate passes its checks."
            if confident
            else " Treat this estimate with caution; see checks."
        ),
        f"- **Where the effect lives.** {mediation['indirect_share'] * 100:.0f}% of it runs through attributes "
        "you can see and control: length, whether it carries a number, tone, hashtags. That is the rewrite "
        "policy: let the model polish wording inside those limits.",
    ]
    if levers is not None and not levers.empty:
        top_lever = levers.iloc[0]
        lines.append(
            f"- **Biggest lever.** {top_lever['lever']}: {top_lever['from']} -> {top_lever['to']} changes "
            f"{outcome} by {_fmt(spec, top_lever['effect'], ratio=top_lever.get('ratio'))}"
            + (
                " (interval excludes zero)."
                if top_lever["significant"]
                else " (interval includes zero)."
            )
        )
    lines += [
        "",
        "## Cold-start benchmark",
        "",
        f"Held-out {group}s, never held-out rows"
        + (
            ", in time order: each fold fits on the past and ranks the next block."
            if cold_start
            and cold_start[0].split == "temporal"
            or (len(cold_start) > 1 and cold_start[1].split == "temporal")
            else "."
        ),
        "",
        table.to_markdown(index=False),
        "",
        f"![cold start]({figures['cold_start'].name})",
    ]
    if curves and "learning_curve" in figures:
        lines += [
            "",
            "### Learning curve",
            "",
            "Rows in time order. The model sees the first *n*, ranks the next block, and the block joins "
            "the context. Where a line first stays above zero is where the account can start trusting "
            "the ranking.",
            "",
            f"![learning curve]({figures['learning_curve'].name})",
        ]
    lines += [
        "",
        "## Causal estimate: LLM vs human copy",
        "",
        f"![effects]({figures['effects'].name})",
        "",
        "| Estimate | Value | 95% CI | Adjustment |",
        "|---|---|---|---|",
        f"| Naive difference | {_fmt(spec, analysis['naive_difference'], ratio=naive_ratio)} | – | none |",
        f"| T-learner total (headline) | {_fmt(spec, total.ate, ratio=total.ratio)} | {total.interval()} | confounders |",
        f"| S-learner total | {_fmt(spec, cross.ate, ratio=cross.ratio)} | {cross.interval()} | confounders |",
        f"| S-learner direct | {_fmt(spec, direct.ate, ratio=direct.ratio)} | {direct.interval()} | confounders + creative attributes |",
    ]
    if isinstance(analysis.get("dopfn"), AteResult):
        d = analysis["dopfn"]
        lines.append(
            f"| Do-PFN total | {_fmt(spec, d.ate, ratio=d.ratio)} | {d.interval()} | learned |"
        )
    if truth_total is not None:
        lines.append(
            f"| **Planted truth, total** | **{_fmt(spec, truth_total, ratio=truth_ratios[0])}** | – | – |"
        )
    if truth_direct is not None:
        lines.append(
            f"| **Planted truth, direct** | **{_fmt(spec, truth_direct, ratio=truth_ratios[1])}** | – | – |"
        )

    lines += [
        "",
        "### Checks",
        "",
        f"- Estimators agree: **{'yes' if analysis['estimators_agree'] else 'no'}** "
        f"(T-learner {_fmt(spec, total.ate, ratio=total.ratio)}, S-learner {_fmt(spec, cross.ate, ratio=cross.ratio)}). "
        "On small, heavy-tailed data the S-learner shrinks a weak treatment toward zero, so the T-learner "
        "is the headline and its interval comes from a bootstrap that refits both arms.",
        (
            f"- Median per-row effect: {_fmt(spec, analysis['median_effect'])}. When the mean and the median "
            "disagree in sign, a few outliers are carrying the mean; read the ratio."
            if "median_effect" in analysis
            else ""
        ),
        f"- Placebo: shuffling the author label inside each {group} gives a mean effect of "
        f"{_fmt(spec, placebo['placebo_mean_abs_ate'])}; the real estimate is {placebo['ratio']:.1f}x larger.",
        f"- Overlap on the confounder set: author is predictable with AUC {overlap['auc']:.2f}; "
        f"{overlap['share_off_support'] * 100:.1f}% of rows fall outside common support. "
        + (
            "The total effect is identifiable from this data."
            if overlap["share_off_support"] < 0.1
            else "Common support is thin; narrow the claim."
        ),
        "- Smallest effect this sample could resolve: about "
        + (
            f"{analysis['detectable_ratio'] * 100:+.0f}%"
            if analysis.get("detectable_ratio") is not None
            else _fmt(spec, analysis["detectable_effect"])
        )
        + f" ({analysis['n_llm']} LLM rows, {analysis['n_human']} human rows). An estimate inside that band "
        "is a statement about the sample, not about the copy.",
        "- The direct (mediator-adjusted) estimate conditions on attributes the author determines. LLM copy is "
        "systematically longer and shaped differently, so the two arms barely overlap on those columns and "
        "the direct effect is poorly identified. It is reported for the mediation split, not as a headline.",
        "",
        "## Effect by segment",
        "",
        f"![segments]({figures['segments'].name})",
        "",
        analysis["segments"].round(5).to_markdown(index=False),
    ]
    if levers is not None and not levers.empty:
        show = levers[
            ["lever", "from", "to", "effect", "ci_low", "ci_high", "support", "significant"]
        ].copy()
        if "ratio" in levers.columns and levers["ratio"].notna().any():
            show.insert(4, "change_pct", (levers["ratio"] * 100).round(1))
        lines += [
            "",
            "## Levers: what to change to make the next one land",
            "",
            "Every post re-scored under each alternative setting of one lever, relative to what the account "
            "usually does. Model-based counterfactuals, not experiments; `support` is how many rows were "
            "actually observed at the alternative.",
            "",
            f"![levers]({figures['levers'].name})" if "levers" in figures else "",
            "",
            show.round(4).to_markdown(index=False),
        ]
    lines += [
        "",
        "## Method",
        "",
        "- **Model.** TabPFN 3.5 through `tabpfn-client` reads the text directly; no vectoriser. Fitting is "
        "in-context with `fit_with_cache`, so each fold and each rewrite is seconds and repeated predictions "
        "skip the forward pass.",
        f"- **Splits.** Held-out {group}s. Rows in one {group} share an audience and an unobserved state; "
        "splitting rows would leak. Social timelines use a temporal split.",
        "- **Effect.** Headline: T-learner (one model per author, cross-predicted), interval from a cluster "
        "bootstrap that refits both arms. Cross-check: S-learner (one model on confounders + author, each row "
        "scored under both). A doubly-robust (AIPW) estimator is in the library but is unstable below a few "
        "hundred rows on heavy-tailed outcomes, so it is not reported by default. "
        f"T-learner: one model per author, cross-predicted. Intervals: cluster bootstrap over {group}s. "
        "Log-scale outcomes are reported as ratios.",
        "- **Adjustment set.** Context columns only. Creative attributes are mediators and are excluded from "
        "the confounder set on purpose.",
        "- **Related work.** Do-PFN (Robertson et al., 2025) uses these meta-learners as baselines and its "
        "'Confounder + Mediator' case study is this problem's graph. Drift-Resilient TabPFN addresses the "
        "rising LLM adoption over time that confounds the naive comparison.",
    ]
    return "\n".join(line for line in lines if line)


def results_json(
    *,
    cold_start: list[ColdStartResult],
    analysis: dict[str, Any],
    truth_total: float | None,
    truth_direct: float | None,
    levers: pd.DataFrame | None = None,
    curves: dict[str, pd.DataFrame] | None = None,
) -> str:
    def ate(r: AteResult) -> dict[str, Any]:
        d = asdict(r)
        d.pop("per_row_effect", None)
        d.pop("per_row_effect_model", None)
        return d

    spec: DatasetSpec = analysis.get("spec", AD_SPEC)
    payload: dict[str, Any] = {
        "spec": spec.name,
        "outcome": spec.outcome.column,
        "cold_start": [
            {**r.as_row(), "recall_at_k": r.recall_at_k, "exploration_saved": r.exploration_saved}
            for r in cold_start
        ],
        "naive_difference": analysis["naive_difference"],
        "naive_ratio": analysis.get("naive_ratio"),
        "total_effect": ate(analysis["total_effect"]),
        "direct_effect": ate(analysis["direct_effect"]),
        "cross_check": ate(analysis["cross_check"]),
        "s_learner": ate(analysis["s_learner"]) if "s_learner" in analysis else None,
        "median_effect": analysis.get("median_effect"),
        "mediation": analysis["mediation"],
        "overlap": analysis["overlap"],
        "placebo": analysis["placebo"],
        "estimators_agree": analysis["estimators_agree"],
        "detectable_effect": analysis.get("detectable_effect"),
        "detectable_ratio": analysis.get("detectable_ratio"),
        "segments": analysis["segments"].to_dict(orient="records"),
        "truth_total": truth_total,
        "truth_direct": truth_direct,
    }
    if isinstance(analysis.get("dopfn"), AteResult):
        payload["dopfn"] = ate(analysis["dopfn"])
    if levers is not None:
        payload["levers"] = levers.to_dict(orient="records")
    if curves:
        payload["learning_curves"] = {
            name: c.to_dict(orient="records") for name, c in curves.items()
        }
    return json.dumps(
        payload,
        indent=2,
        default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o),
    )
