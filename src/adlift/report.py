"""Turn results into a report a stakeholder can read.

Two figures and one Markdown document. The figures use a validated
colour-blind-safe palette, thin marks and direct labels, and never a dual axis.
The document leads with the decision each result supports, and carries every
caveat next to the number it qualifies.
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


def plot_cold_start(results: list[ColdStartResult], path: str | Path, *, k: int = 2) -> Path:
    """Two panels: how much CTR is lost by trusting each model's pick, and how
    often the true winner survives its shortlist."""
    path = Path(path)
    ordered = sorted(results, key=lambda r: r.top1_regret)
    names = [r.name for r in ordered]
    colours = [SERIES[3] if r.backend == "none" else SERIES[i % 3] for i, r in enumerate(ordered)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), facecolor=SURFACE)
    fig.subplots_adjust(wspace=0.5)

    regret = [r.top1_regret * 100 for r in ordered]
    bars = axes[0].barh(names, regret, color=colours, height=0.55)
    axes[0].bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9, color=INK)
    axes[0].set_title(
        "CTR given up by scaling the model's first pick", fontsize=10, color=INK, loc="left"
    )
    axes[0].set_xlabel("top-1 regret (lower is better)", fontsize=9, color=INK_SOFT)
    axes[0].set_xlim(0, max(regret) * 1.3)
    axes[0].invert_yaxis()

    recall = [r.recall_at_k.get(k, float("nan")) * 100 for r in ordered]
    bars = axes[1].barh(names, recall, color=colours, height=0.55)
    axes[1].bar_label(bars, fmt="%.0f%%", padding=4, fontsize=9, color=INK)
    axes[1].set_title(
        f"Campaigns whose true winner is in the top {k}", fontsize=10, color=INK, loc="left"
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
) -> Path:
    """Dot-and-interval chart of every estimate against the planted truth."""
    path = Path(path)
    rows: list[tuple[str, float, float | None, float | None, str]] = [
        ("Naive difference (pivot table)", analysis["naive_difference"], None, None, SERIES[3]),
        (
            "S-learner, total effect",
            analysis["total_effect"].ate,
            analysis["total_effect"].ci_low,
            analysis["total_effect"].ci_high,
            SERIES[0],
        ),
        (
            "T-learner, total effect",
            analysis["cross_check"].ate,
            analysis["cross_check"].ci_low,
            analysis["cross_check"].ci_high,
            SERIES[0],
        ),
        (
            "S-learner, direct effect (mediator-adjusted)",
            analysis["direct_effect"].ate,
            analysis["direct_effect"].ci_low,
            analysis["direct_effect"].ci_high,
            SERIES[1],
        ),
    ]
    if "dopfn" in analysis and isinstance(analysis["dopfn"], AteResult):
        r = analysis["dopfn"]
        rows.append(("Do-PFN, total effect", r.ate, r.ci_low, r.ci_high, SERIES[2]))

    fig, ax = plt.subplots(figsize=(9, 0.6 * len(rows) + 1.6), facecolor=SURFACE)
    y = np.arange(len(rows))
    for i, (_label, ate, lo, hi, colour) in enumerate(rows):
        if lo is not None and hi is not None:
            ax.plot([lo * 100, hi * 100], [i, i], color=colour, linewidth=2, solid_capstyle="round")
        ax.plot(
            ate * 100,
            i,
            "o",
            color=colour,
            markersize=8,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
        )
        ax.annotate(
            f"{ate * 100:+.3f}",
            (ate * 100, i),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8.5,
            color=INK,
        )

    ax.axvline(0, color=INK_SOFT, linewidth=1, linestyle=":")
    # Planted truths as dashed guides, labelled below the last row so the
    # labels never collide with the title or the estimates.
    label_y = len(rows) - 0.35
    if truth_total is not None:
        ax.axvline(truth_total * 100, color=SERIES[0], linewidth=1.2, linestyle="--", alpha=0.7)
        ax.text(
            truth_total * 100,
            label_y,
            "true total",
            ha="center",
            va="top",
            fontsize=8,
            color=SERIES[0],
        )
    if truth_direct is not None:
        ax.axvline(truth_direct * 100, color=SERIES[1], linewidth=1.2, linestyle="--", alpha=0.7)
        ax.text(
            truth_direct * 100,
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
        "effect of LLM authorship on CTR, percentage points (95% cluster-bootstrap CI)",
        fontsize=9,
        color=INK_SOFT,
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
    segments: pd.DataFrame = analysis["segments"]
    column = segments.columns[0]
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(segments) + 1.4), facecolor=SURFACE)
    values = segments["effect"].to_numpy() * 100
    colours = [SERIES[0] if v >= 0 else SERIES[1] for v in values]
    bars = ax.barh(segments[column].astype(str), values, color=colours, height=0.55)
    ax.bar_label(bars, labels=[f"{v:+.3f}" for v in values], padding=4, fontsize=9, color=INK)
    ax.axvline(0, color=INK_SOFT, linewidth=1)
    ax.invert_yaxis()
    # Leave room for the value label beyond the bar end on both sides.
    span = max(abs(values.min()), abs(values.max()), 1e-3)
    ax.set_xlim(min(values.min(), 0) - 0.3 * span, max(values.max(), 0) + 0.3 * span)
    ax.set_xlabel("LLM effect on CTR, percentage points", fontsize=9, color=INK_SOFT)
    ax.set_title(f"Effect by {column}", fontsize=10, color=INK, loc="left")
    _style(ax)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def _fmt_pp(x: float) -> str:
    return f"{x * 100:+.3f} pp"


def render_markdown(
    *,
    cold_start: list[ColdStartResult],
    analysis: dict[str, Any],
    figures: dict[str, Path],
    truth_total: float | None = None,
    truth_direct: float | None = None,
    n_rows: int,
    n_campaigns: int,
    backend: str,
) -> str:
    """Write the report. Decision first, evidence second, caveats attached."""
    total: AteResult = analysis["total_effect"]
    direct: AteResult = analysis["direct_effect"]
    cross: AteResult = analysis["cross_check"]
    overlap = analysis["overlap"]
    placebo = analysis["placebo"]
    mediation = analysis["mediation"]
    table = comparison_table(cold_start)

    direction = "hurts" if total.ate < 0 else "helps"
    confident = total.significant and analysis["estimators_agree"] and placebo["ratio"] >= 3

    lines = [
        "# AdLift report",
        "",
        f"Account: {n_rows} creatives across {n_campaigns} campaigns. Model backend: `{backend}`.",
        "",
        "## Decision summary",
        "",
        f"- **Cold start.** Ranking new creatives with `{table.iloc[0]['Model']}` before they serve gives up "
        f"{table.iloc[0]['Top-1 regret'] * 100:.1f}% of the best achievable CTR, against "
        f"{table[table['Model'] == 'random']['Top-1 regret'].iloc[0] * 100:.1f}% for picking at random. "
        f"Testing only its top 2 keeps the true winner {table.iloc[0]['Recall@2'] * 100:.0f}% of the time "
        f"and leaves {table.iloc[0]['Exploration saved@2'] * 100:.0f}% of the exploration budget unspent.",
        f"- **Human vs LLM copy.** Handing the brief to an LLM {direction} CTR by "
        f"{_fmt_pp(total.ate)} (95% CI {_fmt_pp(total.ci_low)} to {_fmt_pp(total.ci_high)}). "
        f"The pivot-table answer was {_fmt_pp(analysis['naive_difference'])}"
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
        "you can see and control: copy length, whether it carries a number, tone. That is the rewrite policy: "
        "let the model polish wording inside those limits.",
        "",
        "## Cold-start benchmark",
        "",
        "Held-out campaigns, never held-out rows. Each fold fits on the other campaigns and ranks the "
        "creatives in campaigns it has never seen.",
        "",
        table.to_markdown(index=False),
        "",
        f"![cold start]({figures['cold_start'].name})",
        "",
        "## Causal estimate",
        "",
        f"![effects]({figures['effects'].name})",
        "",
        "| Estimate | Value | 95% CI | Adjustment |",
        "|---|---|---|---|",
        f"| Naive difference | {_fmt_pp(analysis['naive_difference'])} | – | none |",
        f"| S-learner total | {_fmt_pp(total.ate)} | {_fmt_pp(total.ci_low)} to {_fmt_pp(total.ci_high)} | confounders |",
        f"| T-learner total | {_fmt_pp(cross.ate)} | {_fmt_pp(cross.ci_low)} to {_fmt_pp(cross.ci_high)} | confounders |",
        f"| S-learner direct | {_fmt_pp(direct.ate)} | {_fmt_pp(direct.ci_low)} to {_fmt_pp(direct.ci_high)} | confounders + creative attributes |",
    ]
    if "dopfn" in analysis and isinstance(analysis["dopfn"], AteResult):
        d = analysis["dopfn"]
        lines.append(
            f"| Do-PFN total | {_fmt_pp(d.ate)} | {_fmt_pp(d.ci_low)} to {_fmt_pp(d.ci_high)} | learned |"
        )
    if truth_total is not None:
        lines.append(f"| **Planted truth, total** | **{_fmt_pp(truth_total)}** | – | – |")
    if truth_direct is not None:
        lines.append(f"| **Planted truth, direct** | **{_fmt_pp(truth_direct)}** | – | – |")

    lines += [
        "",
        "### Checks",
        "",
        f"- Estimators agree: **{'yes' if analysis['estimators_agree'] else 'no'}** "
        f"(S-learner {_fmt_pp(total.ate)}, T-learner {_fmt_pp(cross.ate)}).",
        f"- Placebo: shuffling the author label inside campaigns gives a mean effect of "
        f"{_fmt_pp(placebo['placebo_mean_abs_ate'])}; the real estimate is {placebo['ratio']:.1f}x larger.",
        f"- Overlap on the confounder set: author is predictable with AUC {overlap['auc']:.2f}; "
        f"{overlap['share_off_support'] * 100:.1f}% of rows fall outside common support. "
        + (
            "The total effect is identifiable from this data."
            if overlap["share_off_support"] < 0.1
            else "Common support is thin; narrow the claim."
        ),
        "- The direct (mediator-adjusted) estimate conditions on copy attributes that the author "
        "determines. LLM copy is systematically longer and less numeric, so the two arms barely overlap on "
        "those columns and the direct effect is poorly identified. It is reported for the mediation split, "
        "not as a headline.",
        "",
        "## Effect by segment",
        "",
        f"![segments]({figures['segments'].name})",
        "",
        analysis["segments"].round(5).to_markdown(index=False),
        "",
        "## Method",
        "",
        "- **Model.** TabPFN 3.5 through `tabpfn-client` reads the creative text directly; no vectoriser. "
        "Fitting is in-context, so each fold and each rewrite is seconds.",
        "- **Splits.** `GroupKFold` on campaign. Rows in one campaign share budget, audience and an "
        "unobserved quality; splitting rows would leak.",
        "- **Effect.** S-learner: one model on confounders + author, every row scored under both authors. "
        "T-learner: one model per author, cross-predicted. Intervals: cluster bootstrap over campaigns.",
        "- **Adjustment set.** Placement, device, vertical, audience, budget, week. Creative attributes are "
        "mediators and are excluded from the confounder set on purpose.",
        "- **Related work.** Do-PFN (Robertson et al., 2025) uses these meta-learners as baselines and its "
        "'Confounder + Mediator' case study is this problem's graph. Drift-Resilient TabPFN addresses the "
        "rising LLM adoption over time that confounds the naive comparison.",
    ]
    return "\n".join(lines)


def results_json(
    *,
    cold_start: list[ColdStartResult],
    analysis: dict[str, Any],
    truth_total: float | None,
    truth_direct: float | None,
) -> str:
    def ate(r: AteResult) -> dict[str, Any]:
        d = asdict(r)
        d.pop("per_row_effect", None)
        return d

    payload = {
        "cold_start": [
            {**r.as_row(), "recall_at_k": r.recall_at_k, "exploration_saved": r.exploration_saved}
            for r in cold_start
        ],
        "naive_difference": analysis["naive_difference"],
        "total_effect": ate(analysis["total_effect"]),
        "direct_effect": ate(analysis["direct_effect"]),
        "cross_check": ate(analysis["cross_check"]),
        "mediation": analysis["mediation"],
        "overlap": analysis["overlap"],
        "placebo": analysis["placebo"],
        "estimators_agree": analysis["estimators_agree"],
        "segments": analysis["segments"].to_dict(orient="records"),
        "truth_total": truth_total,
        "truth_direct": truth_direct,
    }
    if "dopfn" in analysis and isinstance(analysis["dopfn"], AteResult):
        payload["dopfn"] = ate(analysis["dopfn"])
    return json.dumps(payload, indent=2, default=float)
