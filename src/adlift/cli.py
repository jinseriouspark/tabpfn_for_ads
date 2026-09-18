"""Command line for AdLift.

Every command runs offline with the gradient-boosted baseline and the stub
language model, and upgrades itself to hosted TabPFN 3.5 and Claude when
``TABPFN_TOKEN`` and ``ANTHROPIC_API_KEY`` are present.

Two datasets are built in: ``--spec ads`` (creatives with CTR) and
``--spec threads`` (one account's posts with views). Your own data comes in
through ``--data``.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="adlift",
    help="Real-time causal copy testing for ads and social posts, powered by TabPFN 3.5.",
    no_args_is_help=True,
    add_completion=False,
)
threads_app = typer.Typer(help="Threads: fetch your account, label it, score and rewrite drafts.")
app.add_typer(threads_app, name="threads")
console = Console()

SPEC_OPTION = typer.Option("ads", help="ads | threads")
OUTCOME_OPTION = typer.Option("", help="threads only: views (default) | engagement")


def _spec(name: str, outcome: str = ""):
    from adlift.schema import get_spec

    return get_spec(name, outcome=outcome or None)


def _load_history(data: Path | None, spec, seed: int, n_groups: int):
    """A CSV if given, else the synthetic account for this spec."""
    from adlift.schema import validate

    if data is None:
        if spec.name == "threads":
            from adlift.datasets.synth_threads import make_threads_account

            frame = make_threads_account(n_weeks=max(4, n_groups), seed=seed)
        else:
            from adlift.datasets.synth import make_ad_account

            frame = make_ad_account(n_campaigns=n_groups, seed=seed)
    elif spec.name == "threads":
        from adlift.ingest.threads import threads_frame_from_csv

        frame = threads_frame_from_csv(data)
    else:
        from adlift.ingest.adapter import frame_from_csv

        frame = frame_from_csv(data)
    validate(frame, spec=spec)
    return frame


def _truths(frame, spec):
    """Planted truths when the frame came from a generator, else Nones."""
    if "_truth_total_effect" not in frame.columns:
        return None, None
    return float(frame["_truth_total_effect"].mean()), float(frame["_truth_direct_effect"].mean())


def _truth_ratios(frame):
    """Planted truths as ratios, when the generator worked on a log scale."""
    import numpy as np

    if "_truth_total_log_effect" not in frame.columns:
        return None, None
    return (
        float(np.expm1(frame["_truth_total_log_effect"].mean())),
        float(np.expm1(frame["_truth_direct_log_effect"].mean())),
    )


def _print_table(frame) -> None:
    table = Table(show_lines=False)
    for column in frame.columns:
        table.add_column(str(column), justify="right" if frame[column].dtype != object else "left")
    for _, row in frame.iterrows():
        table.add_row(*[f"{v:.4f}" if isinstance(v, float) else str(v) for v in row])
    console.print(table)


def _model_rows(frame, spec, backend: str, n_splits: int, seed: int):
    from adlift.coldstart import evaluate_cold_start, random_baseline
    from adlift.model import CTRModel, resolve_backend

    chosen = resolve_backend(backend)
    results = [random_baseline(frame, spec=spec, seed=seed)]
    results.append(
        evaluate_cold_start(
            frame, CTRModel(backend="baseline", spec=spec), name="GBDT + TF-IDF", n_splits=n_splits
        )
    )
    if chosen != "baseline":
        label = "TabPFN 3.5" if chosen == "client" else "TabPFN (local)"
        results.append(
            evaluate_cold_start(
                frame, CTRModel(backend=chosen, spec=spec), name=label, n_splits=n_splits
            )
        )
    return chosen, results


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@app.command()
def synth(
    out: Path = typer.Option(Path("data/synthetic.csv"), help="Where to write the CSV."),
    spec: str = SPEC_OPTION,
    n_groups: int = typer.Option(120, help="Campaigns (ads) or weeks (threads)."),
    seed: int = typer.Option(7),
    keep_truth: bool = typer.Option(False, help="Keep the hidden _truth_* columns."),
) -> None:
    """Generate a synthetic dataset with a known causal effect."""
    frame = _load_history(None, _spec(spec), seed, n_groups)
    out.parent.mkdir(parents=True, exist_ok=True)
    (frame if keep_truth else frame[[c for c in frame.columns if not c.startswith("_")]]).to_csv(
        out, index=False
    )
    total, direct = _truths(frame, _spec(spec))
    console.print(f"wrote {len(frame)} rows -> {out}")
    if total is not None:
        console.print(
            f"planted total effect {total:+.4g}, direct effect {direct:+.4g} ({_spec(spec).outcome.label})"
        )


# ---------------------------------------------------------------------------
# Evaluation and analysis
# ---------------------------------------------------------------------------


@app.command()
def benchmark(
    data: Path | None = typer.Option(
        None, help="CSV in the spec's schema. Omit for synthetic data."
    ),
    spec: str = SPEC_OPTION,
    outcome: str = OUTCOME_OPTION,
    backend: str = typer.Option("auto", help="auto | client | local | baseline"),
    n_splits: int = typer.Option(4),
    seed: int = typer.Option(7),
    n_groups: int = typer.Option(120),
) -> None:
    """Cold-start benchmark: rank rows in groups the model has never seen."""
    from adlift.coldstart import comparison_table

    dataset = _spec(spec, outcome)
    frame = _load_history(data, dataset, seed, n_groups)
    _, results = _model_rows(frame, dataset, backend, n_splits, seed)
    _print_table(comparison_table(results))


@app.command()
def curve(
    data: Path | None = typer.Option(None),
    spec: str = SPEC_OPTION,
    outcome: str = OUTCOME_OPTION,
    backend: str = typer.Option("auto"),
    start: int = typer.Option(5, help="Rows in context for the first evaluation."),
    step: int = typer.Option(5, help="Rows added per step."),
    seed: int = typer.Option(7),
    n_groups: int = typer.Option(120),
) -> None:
    """Learning curve: how soon the ranking becomes useful as rows accumulate."""
    from adlift.coldstart import learning_curve
    from adlift.model import CTRModel, resolve_backend

    dataset = _spec(spec, outcome)
    frame = _load_history(data, dataset, seed, n_groups)
    chosen = resolve_backend(backend)
    _print_table(
        learning_curve(frame, CTRModel(backend=chosen, spec=dataset), start=start, step=step).round(
            4
        )
    )


@app.command()
def analyze(
    data: Path | None = typer.Option(None),
    spec: str = SPEC_OPTION,
    outcome: str = OUTCOME_OPTION,
    backend: str = typer.Option("auto"),
    n_boot: int = typer.Option(400, help="Cluster-bootstrap replicates."),
    refit_boot: int = typer.Option(30, help="Bootstrap replicates that refit the headline model."),
    segment_by: str = typer.Option("", help="Column for the segment breakdown (default per spec)."),
    seed: int = typer.Option(7),
    n_groups: int = typer.Option(120),
    dopfn: bool = typer.Option(False, help="Also run Do-PFN if ADLIFT_DOPFN_PATH is set."),
) -> None:
    """Estimate the causal effect of LLM-written copy, with checks."""
    from adlift.causal import dopfn_ate, full_analysis
    from adlift.model import CTRModel

    dataset = _spec(spec, outcome)
    frame = _load_history(data, dataset, seed, n_groups)
    analysis = full_analysis(
        frame,
        model_factory=lambda: CTRModel(backend=backend, spec=dataset),
        spec=dataset,
        n_boot=n_boot,
        segment_by=segment_by or None,
        refit_boot=refit_boot,
    )
    if dopfn:
        try:
            analysis["dopfn"] = dopfn_ate(frame, spec=dataset)
        except RuntimeError as error:
            console.print(f"[yellow]{error}[/yellow]")

    total, direct = analysis["total_effect"], analysis["direct_effect"]
    fmt = dataset.outcome.format_effect
    console.rule(f"Effect of LLM authorship on {dataset.outcome.label}")
    console.print(
        f"naive difference       {fmt(analysis['naive_difference'], ratio=analysis['naive_ratio'])}"
    )
    console.print(
        f"T-learner total        {fmt(total.ate, ratio=total.ratio)}  [{total.interval()}]  <- headline"
    )
    console.print(
        f"S-learner total        {fmt(analysis['cross_check'].ate, ratio=analysis['cross_check'].ratio)}"
    )
    console.print(
        f"S-learner direct       {fmt(direct.ate, ratio=direct.ratio)}  (mediator-adjusted)"
    )
    if "dopfn" in analysis:
        console.print(
            f"Do-PFN total           {fmt(analysis['dopfn'].ate, ratio=analysis['dopfn'].ratio)}"
        )
    truth_total, truth_direct = _truths(frame, dataset)
    truth_total_ratio, truth_direct_ratio = _truth_ratios(frame)
    if truth_total is not None:
        console.print(
            f"[bold]planted truth total    {fmt(truth_total, ratio=truth_total_ratio)}[/bold]"
        )
        console.print(
            f"[bold]planted truth direct   {fmt(truth_direct, ratio=truth_direct_ratio)}[/bold]"
        )
    console.print()
    console.print(
        f"estimators agree: {analysis['estimators_agree']} | placebo ratio: {analysis['placebo']['ratio']:.1f}x | "
        f"overlap AUC: {analysis['overlap']['auc']:.2f} | off support: {analysis['overlap']['share_off_support'] * 100:.1f}% | "
        f"detectable: {fmt(analysis['detectable_effect'])}"
        + (
            f" ({analysis['detectable_ratio'] * 100:+.0f}%)"
            if analysis.get("detectable_ratio") is not None
            else ""
        )
    )
    console.print(
        f"indirect share (via visible copy attributes): {analysis['mediation']['indirect_share'] * 100:.0f}%"
    )
    console.print()
    _print_table(analysis["segments"].round(5))


@app.command()
def levers(
    data: Path | None = typer.Option(None),
    spec: str = typer.Option("threads", help="ads | threads"),
    outcome: str = OUTCOME_OPTION,
    backend: str = typer.Option("auto"),
    n_boot: int = typer.Option(200),
    seed: int = typer.Option(7),
    n_groups: int = typer.Option(16),
) -> None:
    """Which levers move the outcome, and by how much, relative to the account's habits."""
    from adlift.levers import lever_analysis
    from adlift.model import CTRModel

    dataset = _spec(spec, outcome)
    frame = _load_history(data, dataset, seed, n_groups)
    table = lever_analysis(
        frame,
        spec=dataset,
        model_factory=lambda: CTRModel(backend=backend, spec=dataset),
        n_boot=n_boot,
        seed=seed,
    )
    show = table[
        ["lever", "from", "to", "effect", "ci_low", "ci_high", "support", "significant"]
    ].copy()
    if "ratio" in table.columns and table["ratio"].notna().any():
        show.insert(4, "change_pct", (table["ratio"] * 100).round(1))
    _print_table(show.round(3))


@app.command()
def demo(
    out: Path = typer.Option(Path("reports"), help="Output directory."),
    data: Path | None = typer.Option(None),
    spec: str = SPEC_OPTION,
    outcome: str = OUTCOME_OPTION,
    backend: str = typer.Option("auto"),
    n_boot: int = typer.Option(400),
    refit_boot: int = typer.Option(30),
    n_splits: int = typer.Option(4),
    seed: int = typer.Option(7),
    n_groups: int = typer.Option(120),
    dopfn: bool = typer.Option(False),
) -> None:
    """Run everything and write a Markdown report with figures and JSON."""
    from adlift.causal import dopfn_ate, full_analysis
    from adlift.coldstart import learning_curve
    from adlift.levers import lever_analysis
    from adlift.model import CTRModel
    from adlift.report import (
        plot_cold_start,
        plot_effects,
        plot_learning_curve,
        plot_levers,
        plot_segments,
        render_markdown,
        results_json,
    )

    dataset = _spec(spec, outcome)
    out.mkdir(parents=True, exist_ok=True)
    frame = _load_history(data, dataset, seed, n_groups)
    n_groups_actual = int(frame[dataset.group_column].nunique())

    with console.status("cold-start benchmark"):
        chosen, cold = _model_rows(frame, dataset, backend, n_splits, seed)
    console.print(
        f"backend: {chosen} | rows: {len(frame)} | {dataset.group_column}s: {n_groups_actual}"
    )

    curves: dict = {}
    if dataset.name == "threads":
        with console.status("learning curve"):
            curves["GBDT + TF-IDF"] = learning_curve(
                frame, CTRModel(backend="baseline", spec=dataset), start=10, step=10
            )
            if chosen != "baseline":
                label = "TabPFN 3.5" if chosen == "client" else "TabPFN (local)"
                curves[label] = learning_curve(
                    frame, CTRModel(backend=chosen, spec=dataset), start=10, step=10
                )

    with console.status("causal analysis"):
        analysis = full_analysis(
            frame,
            model_factory=lambda: CTRModel(backend=chosen, spec=dataset),
            spec=dataset,
            n_boot=n_boot,
            refit_boot=refit_boot,
        )
        if dopfn:
            try:
                analysis["dopfn"] = dopfn_ate(frame, spec=dataset)
            except RuntimeError as error:
                console.print(f"[yellow]{error}[/yellow]")

    levers_table = None
    if dataset.name == "threads":
        with console.status("lever analysis"):
            levers_table = lever_analysis(
                frame,
                spec=dataset,
                model_factory=lambda: CTRModel(backend=chosen, spec=dataset),
                n_boot=max(50, n_boot // 2),
                seed=seed,
            )

    truth_total, truth_direct = _truths(frame, dataset)
    truth_total_ratio, truth_direct_ratio = _truth_ratios(frame)
    figures = {
        "cold_start": plot_cold_start(cold, out / "cold_start.png"),
        "effects": plot_effects(
            analysis,
            out / "effects.png",
            truth_total=truth_total,
            truth_direct=truth_direct,
            truth_ratios=(truth_total_ratio, truth_direct_ratio),
        ),
        "segments": plot_segments(analysis, out / "segments.png"),
    }
    if curves:
        figures["learning_curve"] = plot_learning_curve(curves, out / "learning_curve.png")
    if levers_table is not None and not levers_table.empty:
        figures["levers"] = plot_levers(levers_table, out / "levers.png", spec=dataset)

    (out / "report.md").write_text(
        render_markdown(
            cold_start=cold,
            analysis=analysis,
            figures=figures,
            truth_total=truth_total,
            truth_direct=truth_direct,
            n_rows=len(frame),
            n_groups=n_groups_actual,
            backend=chosen,
            levers=levers_table,
            curves=curves or None,
            truth_ratios=(truth_total_ratio, truth_direct_ratio),
        )
    )
    (out / "results.json").write_text(
        results_json(
            cold_start=cold,
            analysis=analysis,
            truth_total=truth_total,
            truth_direct=truth_direct,
            levers=levers_table,
            curves=curves or None,
        )
    )
    console.print(f"report -> {out / 'report.md'}")
    console.print(f"figures -> {', '.join(str(p) for p in figures.values())}")


# ---------------------------------------------------------------------------
# Ads: score, revise, ingest a banner
# ---------------------------------------------------------------------------


def _ad_context(vertical, placement, device, audience, budget, week) -> dict:
    return {
        "vertical": vertical,
        "placement": placement,
        "device": device,
        "audience": audience,
        "daily_budget_usd": budget,
        "week_index": week,
    }


@app.command()
def score(
    headline: str = typer.Option(..., help="Creative headline."),
    body: str = typer.Option(""),
    cta: str = typer.Option(""),
    author: str = typer.Option("human", help="human | llm"),
    data: Path | None = typer.Option(None, help="History CSV (ads schema). Omit for synthetic."),
    backend: str = typer.Option("auto"),
    vertical: str = typer.Option("saas"),
    placement: str = typer.Option("social_feed"),
    device: str = typer.Option("mobile"),
    audience: str = typer.Option("prospecting"),
    budget: float = typer.Option(300.0),
    week: int = typer.Option(20),
    seed: int = typer.Option(7),
) -> None:
    """Predict CTR for an ad creative that has never served."""
    from adlift.loop import CopyCoach
    from adlift.model import CTRModel
    from adlift.schema import AD_SPEC, AdCreative

    frame = _load_history(data, AD_SPEC, seed, 120)
    creative = AdCreative(
        headline=headline,
        body=body,
        cta_text=cta,
        author=author,  # type: ignore[arg-type]
        campaign_id="new",
        **_ad_context(vertical, placement, device, audience, budget, week),
    )
    coach = CopyCoach(frame, spec=AD_SPEC, model=CTRModel(backend=backend, spec=AD_SPEC)).fit()
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
    author: str = typer.Option("human"),
    n: int = typer.Option(6),
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
    """Have an LLM rewrite an ad draft under data-derived limits; TabPFN ranks the results."""
    from adlift.llm import get_llm
    from adlift.loop import CopyCoach
    from adlift.model import CTRModel
    from adlift.schema import AD_SPEC, AdCreative

    frame = _load_history(data, AD_SPEC, seed, 120)
    draft = AdCreative(
        headline=headline,
        body=body,
        cta_text=cta,
        author=author,  # type: ignore[arg-type]
        campaign_id="new",
        **_ad_context(vertical, placement, device, audience, budget, week),
    )
    coach = CopyCoach(
        frame, spec=AD_SPEC, model=CTRModel(backend=backend, spec=AD_SPEC), llm=get_llm(llm)
    ).fit()
    _print_revision(
        coach.revise(draft, n_variants=n),
        draft_text=f"{headline} / {body} / {cta}".strip(" /"),
        unit="%",
    )


def _print_revision(result, *, draft_text: str, unit: str) -> None:
    def fmt(v: float) -> str:
        return f"{v * 100:.2f}%" if unit == "%" else f"{v:,.0f}"

    table = Table(
        title=f"Draft {fmt(result.draft.predicted)} -> best {fmt(result.best.predicted)}  "
        f"(fit {result.fit_seconds:.1f}s, score {result.score_seconds:.2f}s)"
    )
    for column, justify in (
        ("#", "right"),
        (result.outcome_label, "right"),
        ("lift", "right"),
        ("words", "right"),
        ("copy", "left"),
    ):
        table.add_column(column, justify=justify)
    table.add_row(
        "draft", fmt(result.draft.predicted), "", str(result.draft.creative.word_count), draft_text
    )
    for v in result.variants:
        c = v.creative
        text = f"{c.headline} / {getattr(c, 'body', '')} / {getattr(c, 'cta_text', '')}".strip(" /")
        lift = f"{v.lift_vs_draft * 100:+.2f}" if unit == "%" else f"{v.lift_vs_draft:+,.0f}"
        table.add_row(str(v.rank), fmt(v.predicted), lift, str(c.word_count), text)
    console.print(table)
    console.print(
        f"constraints: <= {result.constraints.max_words} words, tone {result.constraints.tone}, "
        f"keep number: {result.constraints.keep_numeric_claim}, max hashtags: {result.constraints.max_hashtags}"
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
    """Read a banner image into the canonical ad creative record (JSON)."""
    from adlift.ingest.banner import banner_to_creative
    from adlift.llm import get_llm

    context = _ad_context(vertical or "unknown", placement, device, audience, budget, week)
    if not vertical:
        context.pop("vertical")
    creative = banner_to_creative(
        image, llm=get_llm(llm), campaign_id=campaign, author=author, context=context
    )
    console.print(json.dumps({**creative.to_row(), "extra": creative.extra}, indent=2, default=str))


# ---------------------------------------------------------------------------
# Threads
# ---------------------------------------------------------------------------


def _post_context(hour, weekday, media, reply, topic, week) -> dict:
    return {
        "posted_hour": hour,
        "weekday": weekday,
        "has_media": int(media),
        "is_reply": int(reply),
        "topic": topic,
        "week_index": week,
    }


@threads_app.command("fetch")
def threads_fetch(
    out: Path = typer.Option(Path("data/threads_posts.csv")),
    token: str = typer.Option("", envvar="THREADS_ACCESS_TOKEN", help="Threads user access token."),
    limit: int = typer.Option(500),
    include_replies: bool = typer.Option(False),
    tz: str = typer.Option("UTC", help="Timezone for posting hour, e.g. Asia/Seoul."),
) -> None:
    """Pull your own posts and insights through the Threads API into a CSV."""
    from adlift.ingest.threads import fetch_threads_account, write_label_template

    if not token:
        raise typer.BadParameter("pass --token or set THREADS_ACCESS_TOKEN")
    frame = fetch_threads_account(token, limit=limit, include_replies=include_replies, tz=tz)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    template = write_label_template(frame, out.with_name(out.stem + "_labels.csv"))
    console.print(f"wrote {len(frame)} posts -> {out}")
    console.print(
        f"label template -> {template}  (fill 'author' with human/llm, then merge with `threads label`)"
    )


@threads_app.command("template")
def threads_template(
    data: Path = typer.Argument(..., help="Posts CSV (text, timestamp, views, ...)."),
    out: Path = typer.Option(Path("data/threads_labels.csv")),
    tz: str = typer.Option("UTC"),
) -> None:
    """Write a CSV to hand-label author and topic for an existing export."""
    from adlift.ingest.threads import threads_frame_from_csv, write_label_template

    frame = threads_frame_from_csv(data, tz=tz)
    console.print(f"template -> {write_label_template(frame, out)}")


@threads_app.command("label")
def threads_label(
    data: Path = typer.Argument(..., help="Posts CSV."),
    labels: Path = typer.Argument(..., help="Filled label template (post_id, author, topic)."),
    out: Path = typer.Option(Path("data/threads_labeled.csv")),
    tz: str = typer.Option("UTC"),
) -> None:
    """Merge hand labels back into the posts CSV."""
    import pandas as pd

    from adlift.ingest.threads import threads_frame_from_csv

    frame = threads_frame_from_csv(data, tz=tz)
    filled = pd.read_csv(labels, dtype={"post_id": str})
    merged = frame.drop(columns=[c for c in ("author", "topic") if c in frame.columns]).merge(
        filled[["post_id", "author", "topic"]], on="post_id", how="left"
    )
    merged["author"] = merged["author"].fillna("").astype(str).str.lower().replace("", "human")
    merged["topic"] = merged["topic"].fillna("unknown").replace("", "unknown")
    merged.to_csv(out, index=False)
    console.print(
        f"wrote {len(merged)} labelled posts -> {out} "
        f"(llm: {(merged.author == 'llm').sum()}, human: {(merged.author == 'human').sum()})"
    )


@threads_app.command("score")
def threads_score(
    text: str = typer.Option(..., help="Draft post text."),
    data: Path | None = typer.Option(None, help="Labelled posts CSV. Omit for synthetic."),
    backend: str = typer.Option("auto"),
    outcome: str = OUTCOME_OPTION,
    author: str = typer.Option("human"),
    hour: int = typer.Option(20),
    weekday: str = typer.Option("Tue"),
    media: bool = typer.Option(False),
    reply: bool = typer.Option(False),
    topic: str = typer.Option("unknown"),
    week: int = typer.Option(0, help="Weeks since the account's first post; 0 = use the latest."),
    seed: int = typer.Option(7),
    precedents: int = typer.Option(5, help="How many similar past posts to show."),
) -> None:
    """Predict views (or engagement) for a post before it goes out, with a band and precedents."""
    from adlift.loop import CopyCoach, PostDraft
    from adlift.model import CTRModel

    dataset = _spec("threads", outcome)
    frame = _load_history(data, dataset, seed, 16)
    week = week or int(frame["week_index"].max())
    draft = PostDraft(
        text=text, author=author, context=_post_context(hour, weekday, media, reply, topic, week)
    )
    coach = CopyCoach(frame, spec=dataset, model=CTRModel(backend=backend, spec=dataset)).fit()
    [scored] = coach.score([draft])
    console.print(
        json.dumps(
            {
                **scored.to_dict(),
                "outcome": dataset.outcome.label,
                "fit_seconds": round(coach.fit_seconds, 3),
                "backend": coach.model.backend,
            },
            indent=2,
        )
    )
    if precedents:
        console.print("closest past posts:")
        _print_table(coach.precedents(draft, k=precedents))


@threads_app.command("revise")
def threads_revise(
    text: str = typer.Option(..., help="Draft post text."),
    n: int = typer.Option(6),
    data: Path | None = typer.Option(None),
    backend: str = typer.Option("auto"),
    llm: str = typer.Option("auto"),
    outcome: str = OUTCOME_OPTION,
    author: str = typer.Option("human"),
    hour: int = typer.Option(20),
    weekday: str = typer.Option("Tue"),
    media: bool = typer.Option(False),
    reply: bool = typer.Option(False),
    topic: str = typer.Option("unknown"),
    week: int = typer.Option(0),
    seed: int = typer.Option(7),
) -> None:
    """LLM rewrites under the account's own limits; TabPFN ranks them."""
    from adlift.llm import get_llm
    from adlift.loop import CopyCoach, PostDraft
    from adlift.model import CTRModel

    dataset = _spec("threads", outcome)
    frame = _load_history(data, dataset, seed, 16)
    week = week or int(frame["week_index"].max())
    draft = PostDraft(
        text=text, author=author, context=_post_context(hour, weekday, media, reply, topic, week)
    )
    coach = CopyCoach(
        frame, spec=dataset, model=CTRModel(backend=backend, spec=dataset), llm=get_llm(llm)
    ).fit()
    _print_revision(
        coach.revise(draft, n_variants=n),
        draft_text=text,
        unit="%" if dataset.outcome.effect_unit == "pp" else "views",
    )


@threads_app.command("suggest")
def threads_suggest(
    text: str = typer.Option(..., help="Draft post text."),
    data: Path | None = typer.Option(None),
    backend: str = typer.Option("auto"),
    outcome: str = OUTCOME_OPTION,
    author: str = typer.Option("human"),
    hour: int = typer.Option(20),
    weekday: str = typer.Option("Tue"),
    media: bool = typer.Option(False),
    reply: bool = typer.Option(False),
    topic: str = typer.Option("unknown"),
    week: int = typer.Option(0),
    top: int = typer.Option(6),
    seed: int = typer.Option(7),
) -> None:
    """Which single change to this post raises its predicted outcome most."""
    from adlift.levers import suggest_for_post
    from adlift.loop import PostDraft
    from adlift.model import CTRModel

    dataset = _spec("threads", outcome)
    frame = _load_history(data, dataset, seed, 16)
    week = week or int(frame["week_index"].max())
    row = PostDraft(
        text=text, author=author, context=_post_context(hour, weekday, media, reply, topic, week)
    ).to_row(dataset)
    table = suggest_for_post(
        frame,
        row,
        spec=dataset,
        model_factory=lambda: CTRModel(backend=backend, spec=dataset),
        top=top,
    )
    console.print(
        f"as written: {table.attrs.get('as_is', float('nan')):,.0f} {dataset.outcome.label}"
    )
    _print_table(table.round(3))


if __name__ == "__main__":
    app()
