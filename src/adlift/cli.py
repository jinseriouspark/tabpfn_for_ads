"""Command line for AdLift.

Every command runs offline with the gradient-boosted baseline and the stub
language model, and upgrades itself to hosted TabPFN 3.5 and Claude when
``TABPFN_TOKEN`` and ``ANTHROPIC_API_KEY`` are present.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="adlift",
    help="Real-time causal copy testing for ad creatives, powered by TabPFN 3.5.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _load_history(data: Path | None, seed: int, n_campaigns: int):
    """A CSV if given, else the synthetic account with known ground truth."""
    from adlift.datasets.synth import make_ad_account
    from adlift.ingest.adapter import frame_from_csv
    from adlift.schema import validate

    if data is None:
        frame = make_ad_account(n_campaigns=n_campaigns, seed=seed)
    else:
        frame = frame_from_csv(data)
    validate(frame)
    return frame


def _context_from(
    vertical: str, placement: str, device: str, audience: str, budget: float, week: int
) -> dict:
    return {
        "vertical": vertical,
        "placement": placement,
        "device": device,
        "audience": audience,
        "daily_budget_usd": budget,
        "week_index": week,
    }


@app.command()
def synth(
    out: Path = typer.Option(Path("data/synthetic_account.csv"), help="Where to write the CSV."),
    n_campaigns: int = typer.Option(120, help="Number of campaigns."),
    seed: int = typer.Option(7, help="Random seed."),
    keep_truth: bool = typer.Option(False, help="Keep the hidden _truth_* columns."),
) -> None:
    """Generate a synthetic ad account with a known causal effect."""
    from adlift.datasets.synth import make_ad_account, true_ate

    frame = make_ad_account(n_campaigns=n_campaigns, seed=seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    (frame if keep_truth else frame[[c for c in frame.columns if not c.startswith("_")]]).to_csv(
        out, index=False
    )
    console.print(
        f"wrote {len(frame)} creatives / {frame.campaign_id.nunique()} campaigns -> {out}"
    )
    console.print(
        f"planted total effect {true_ate(frame, 'total') * 100:+.3f} pp, "
        f"direct effect {true_ate(frame, 'direct') * 100:+.3f} pp"
    )


@app.command()
def benchmark(
    data: Path | None = typer.Option(None, help="Creative CSV. Omit for the synthetic account."),
    backend: str = typer.Option("auto", help="auto | client | local | baseline"),
    n_splits: int = typer.Option(4, help="Campaign folds."),
    seed: int = typer.Option(7),
    n_campaigns: int = typer.Option(120),
) -> None:
    """Cold-start benchmark: rank creatives in campaigns the model has never seen."""
    from adlift.coldstart import comparison_table, evaluate_cold_start, random_baseline
    from adlift.model import CTRModel, resolve_backend

    frame = _load_history(data, seed, n_campaigns)
    chosen = resolve_backend(backend)
    results = [random_baseline(frame, seed=seed)]
    results.append(
        evaluate_cold_start(
            frame, CTRModel(backend="baseline"), name="GBDT + TF-IDF", n_splits=n_splits
        )
    )
    if chosen != "baseline":
        label = "TabPFN 3.5" if chosen == "client" else "TabPFN (local)"
        results.append(
            evaluate_cold_start(frame, CTRModel(backend=chosen), name=label, n_splits=n_splits)
        )
    _print_table(comparison_table(results))


@app.command()
def analyze(
    data: Path | None = typer.Option(None, help="Creative CSV. Omit for the synthetic account."),
    backend: str = typer.Option("auto", help="auto | client | local | baseline"),
    n_boot: int = typer.Option(400, help="Cluster-bootstrap replicates."),
    segment_by: str = typer.Option("device", help="Column for the segment breakdown."),
    seed: int = typer.Option(7),
    n_campaigns: int = typer.Option(120),
    dopfn: bool = typer.Option(False, help="Also run Do-PFN if ADLIFT_DOPFN_PATH is set."),
) -> None:
    """Estimate the causal effect of LLM-written copy on CTR, with checks."""
    from adlift.causal import dopfn_ate, full_analysis
    from adlift.datasets.synth import true_ate
    from adlift.model import CTRModel

    frame = _load_history(data, seed, n_campaigns)
    analysis = full_analysis(
        frame, model_factory=lambda: CTRModel(backend=backend), n_boot=n_boot, segment_by=segment_by
    )
    if dopfn:
        try:
            analysis["dopfn"] = dopfn_ate(frame)
        except RuntimeError as error:
            console.print(f"[yellow]{error}[/yellow]")

    total, direct = analysis["total_effect"], analysis["direct_effect"]
    console.rule("Effect of LLM authorship on CTR")
    console.print(f"naive difference       {analysis['naive_difference'] * 100:+.3f} pp")
    console.print(
        f"S-learner total        {total.ate * 100:+.3f} pp  "
        f"[{total.ci_low * 100:+.3f}, {total.ci_high * 100:+.3f}]"
    )
    console.print(f"T-learner total        {analysis['cross_check'].ate * 100:+.3f} pp")
    console.print(f"S-learner direct       {direct.ate * 100:+.3f} pp  (mediator-adjusted)")
    if "dopfn" in analysis:
        console.print(f"Do-PFN total           {analysis['dopfn'].ate * 100:+.3f} pp")
    if "_truth_total_effect" in frame.columns:
        console.print(
            f"[bold]planted truth total    {true_ate(frame, 'total') * 100:+.3f} pp[/bold]"
        )
        console.print(
            f"[bold]planted truth direct   {true_ate(frame, 'direct') * 100:+.3f} pp[/bold]"
        )
    console.print()
    console.print(
        f"estimators agree: {analysis['estimators_agree']} | "
        f"placebo ratio: {analysis['placebo']['ratio']:.1f}x | "
        f"overlap AUC: {analysis['overlap']['auc']:.2f} | "
        f"off support: {analysis['overlap']['share_off_support'] * 100:.1f}%"
    )
    console.print(
        f"indirect share (via visible copy attributes): "
        f"{analysis['mediation']['indirect_share'] * 100:.0f}%"
    )
    console.print()
    _print_table(analysis["segments"].round(5))


@app.command()
def demo(
    out: Path = typer.Option(Path("reports"), help="Output directory."),
    data: Path | None = typer.Option(None, help="Creative CSV. Omit for the synthetic account."),
    backend: str = typer.Option("auto"),
    n_boot: int = typer.Option(400),
    n_splits: int = typer.Option(4),
    seed: int = typer.Option(7),
    n_campaigns: int = typer.Option(120),
    dopfn: bool = typer.Option(False),
) -> None:
    """Run everything and write a Markdown report with figures and JSON."""
    from adlift.causal import dopfn_ate, full_analysis
    from adlift.coldstart import evaluate_cold_start, random_baseline
    from adlift.datasets.synth import true_ate
    from adlift.model import CTRModel, resolve_backend
    from adlift.report import (
        plot_cold_start,
        plot_effects,
        plot_segments,
        render_markdown,
        results_json,
    )

    out.mkdir(parents=True, exist_ok=True)
    frame = _load_history(data, seed, n_campaigns)
    chosen = resolve_backend(backend)
    console.print(
        f"backend: {chosen} | rows: {len(frame)} | campaigns: {frame.campaign_id.nunique()}"
    )

    with console.status("cold-start benchmark"):
        cold = [random_baseline(frame, seed=seed)]
        cold.append(
            evaluate_cold_start(
                frame, CTRModel(backend="baseline"), name="GBDT + TF-IDF", n_splits=n_splits
            )
        )
        if chosen != "baseline":
            label = "TabPFN 3.5" if chosen == "client" else "TabPFN (local)"
            cold.append(
                evaluate_cold_start(frame, CTRModel(backend=chosen), name=label, n_splits=n_splits)
            )

    with console.status("causal analysis"):
        analysis = full_analysis(
            frame, model_factory=lambda: CTRModel(backend=chosen), n_boot=n_boot
        )
        if dopfn:
            try:
                analysis["dopfn"] = dopfn_ate(frame)
            except RuntimeError as error:
                console.print(f"[yellow]{error}[/yellow]")

    has_truth = "_truth_total_effect" in frame.columns
    truth_total = true_ate(frame, "total") if has_truth else None
    truth_direct = true_ate(frame, "direct") if has_truth else None

    figures = {
        "cold_start": plot_cold_start(cold, out / "cold_start.png"),
        "effects": plot_effects(
            analysis, out / "effects.png", truth_total=truth_total, truth_direct=truth_direct
        ),
        "segments": plot_segments(analysis, out / "segments.png"),
    }
    markdown = render_markdown(
        cold_start=cold,
        analysis=analysis,
        figures=figures,
        truth_total=truth_total,
        truth_direct=truth_direct,
        n_rows=len(frame),
        n_campaigns=int(frame.campaign_id.nunique()),
        backend=chosen,
    )
    (out / "report.md").write_text(markdown)
    (out / "results.json").write_text(
        results_json(
            cold_start=cold, analysis=analysis, truth_total=truth_total, truth_direct=truth_direct
        )
    )
    console.print(f"report -> {out / 'report.md'}")
    console.print(f"figures -> {', '.join(str(p) for p in figures.values())}")


@app.command()
def score(
    headline: str = typer.Option(..., help="Creative headline."),
    body: str = typer.Option("", help="Body copy."),
    cta: str = typer.Option("", help="Call to action."),
    author: str = typer.Option("human", help="human | llm"),
    data: Path | None = typer.Option(None, help="History CSV. Omit for the synthetic account."),
    backend: str = typer.Option("auto"),
    vertical: str = typer.Option("saas"),
    placement: str = typer.Option("social_feed"),
    device: str = typer.Option("mobile"),
    audience: str = typer.Option("prospecting"),
    budget: float = typer.Option(300.0),
    week: int = typer.Option(20),
    seed: int = typer.Option(7),
) -> None:
    """Predict CTR for a creative that has never served."""
    from adlift.loop import CopyCoach
    from adlift.model import CTRModel
    from adlift.schema import AdCreative

    frame = _load_history(data, seed, 120)
    creative = AdCreative(
        headline=headline,
        body=body,
        cta_text=cta,
        author=author,  # type: ignore[arg-type]
        campaign_id="new",
        **_context_from(vertical, placement, device, audience, budget, week),
    )
    coach = CopyCoach(frame, model=CTRModel(backend=backend)).fit()
    [scored] = coach.score([creative])
    console.print(
        json.dumps(
            {
                **scored.to_dict(),
                "fit_seconds": round(coach.fit_seconds, 3),
                "backend": coach.model.backend,
            },
            indent=2,
        )
    )


@app.command()
def revise(
    headline: str = typer.Option(..., help="Draft headline."),
    body: str = typer.Option(""),
    cta: str = typer.Option(""),
    author: str = typer.Option("human", help="Who wrote the draft: human | llm"),
    n: int = typer.Option(6, help="Number of variants."),
    data: Path | None = typer.Option(None),
    backend: str = typer.Option("auto"),
    llm: str = typer.Option("auto", help="auto | claude | stub"),
    vertical: str = typer.Option("saas"),
    placement: str = typer.Option("social_feed"),
    device: str = typer.Option("mobile"),
    audience: str = typer.Option("prospecting"),
    budget: float = typer.Option(300.0),
    week: int = typer.Option(20),
    seed: int = typer.Option(7),
) -> None:
    """Have an LLM rewrite a draft under data-derived limits; TabPFN ranks the results."""
    from adlift.llm import get_llm
    from adlift.loop import CopyCoach
    from adlift.model import CTRModel
    from adlift.schema import AdCreative

    frame = _load_history(data, seed, 120)
    draft = AdCreative(
        headline=headline,
        body=body,
        cta_text=cta,
        author=author,  # type: ignore[arg-type]
        campaign_id="new",
        **_context_from(vertical, placement, device, audience, budget, week),
    )
    coach = CopyCoach(frame, model=CTRModel(backend=backend), llm=get_llm(llm)).fit()
    result = coach.revise(draft, n_variants=n)

    table = Table(
        title=f"Draft {result.draft.predicted_ctr * 100:.2f}% -> best {result.best.predicted_ctr * 100:.2f}%  "
        f"(fit {result.fit_seconds:.1f}s, score {result.score_seconds:.2f}s)"
    )
    table.add_column("#", justify="right")
    table.add_column("pred CTR", justify="right")
    table.add_column("lift", justify="right")
    table.add_column("words", justify="right")
    table.add_column("copy")
    table.add_row(
        "draft",
        f"{result.draft.predicted_ctr * 100:.2f}%",
        "",
        str(draft.word_count),
        f"{draft.headline} / {draft.body} / {draft.cta_text}".strip(" /"),
    )
    for v in result.variants:
        c = v.creative
        table.add_row(
            str(v.rank),
            f"{v.predicted_ctr * 100:.2f}%",
            f"{v.lift_vs_draft * 100:+.2f}",
            str(c.word_count),
            f"{c.headline} / {c.body} / {c.cta_text}".strip(" /"),
        )
    console.print(table)
    console.print(
        f"constraints: <= {result.constraints.max_words} words, tone {result.constraints.tone}, "
        f"keep number: {result.constraints.keep_numeric_claim}"
    )
    for note in result.notes:
        console.print(f"[dim]{note}[/dim]")


@app.command()
def ingest(
    image: str = typer.Argument(..., help="Path or URL of a banner image."),
    llm: str = typer.Option("auto", help="auto | claude | stub"),
    campaign: str = typer.Option("new"),
    author: str = typer.Option("human"),
    vertical: str = typer.Option(""),
    placement: str = typer.Option("unknown"),
    device: str = typer.Option("unknown"),
    audience: str = typer.Option("unknown"),
    budget: float = typer.Option(0.0),
    week: int = typer.Option(0),
) -> None:
    """Read a banner image into the canonical creative record (JSON)."""
    from adlift.ingest.banner import banner_to_creative
    from adlift.llm import get_llm

    context = _context_from(vertical or "unknown", placement, device, audience, budget, week)
    if not vertical:
        context.pop("vertical")
    creative = banner_to_creative(
        image, llm=get_llm(llm), campaign_id=campaign, author=author, context=context
    )
    console.print(json.dumps({**creative.to_row(), "extra": creative.extra}, indent=2, default=str))


def _print_table(frame) -> None:
    table = Table(show_lines=False)
    for column in frame.columns:
        table.add_column(str(column), justify="right" if frame[column].dtype != object else "left")
    for _, row in frame.iterrows():
        table.add_row(*[f"{v:.4f}" if isinstance(v, float) else str(v) for v in row])
    console.print(table)


if __name__ == "__main__":
    app()
