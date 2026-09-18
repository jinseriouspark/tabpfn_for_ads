"""AdLift as an MCP server.

Run it and any MCP client (Claude Desktop, Claude Code, Cursor, an agent
framework) can load an ad account or a Threads timeline, score a draft
before it goes out, ask for constrained rewrites, see which levers move the
outcome and read the causal report, all through tool calls.

    adlift-mcp                      # stdio transport, the default
    python -m mcp_server.server     # same thing

State is one loaded dataset and one fitted coach. Loading is cheap (TabPFN
fits in seconds, and with ``fit_with_cache`` repeated predictions are cheaper
still), so reloading with a different CSV is the normal way to switch
accounts.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd

try:  # mcp >= 2.0 renamed the high-level server
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP

from adlift.causal import full_analysis
from adlift.coldstart import comparison_table, evaluate_cold_start, learning_curve, random_baseline
from adlift.datasets.synth import make_ad_account
from adlift.datasets.synth_threads import make_threads_account
from adlift.ingest.adapter import frame_from_csv
from adlift.ingest.banner import banner_to_creative
from adlift.ingest.threads import fetch_threads_account, threads_frame_from_csv
from adlift.levers import lever_analysis, suggest_for_post
from adlift.llm import get_llm
from adlift.loop import CopyCoach, PostDraft
from adlift.model import CTRModel, resolve_backend
from adlift.schema import AD_SPEC, AdCreative, DatasetSpec, get_spec, validate

mcp = FastMCP(
    "adlift",
    instructions=(
        "AdLift predicts how an ad creative or a social post will perform before it goes out, "
        "rewrites copy under data-derived constraints, says which levers move the outcome, and "
        "estimates whether LLM-written copy causally outperforms human-written copy. "
        "Call load_account (ads) or load_threads (a Threads timeline) first, or rely on the "
        "synthetic default; then score_creative / score_post / revise_* / suggest_post / "
        "lever_report / causal_report."
    ),
)


@dataclass
class _State:
    history: pd.DataFrame | None = None
    coach: CopyCoach | None = None
    source: str = "none"
    spec: DatasetSpec = AD_SPEC
    backend: str = resolve_backend(os.environ.get("ADLIFT_BACKEND", "auto"))  # type: ignore[arg-type]


STATE = _State()


def _set_history(frame: pd.DataFrame, spec: DatasetSpec, source: str) -> CopyCoach:
    validate(frame, spec=spec)
    STATE.history, STATE.spec, STATE.source = frame, spec, source
    STATE.coach = CopyCoach(
        frame, spec=spec, model=CTRModel(backend=STATE.backend, spec=spec), llm=get_llm()
    ).fit()
    return STATE.coach


def _ensure_loaded() -> CopyCoach:
    if STATE.coach is None:
        if STATE.history is None:
            return _set_history(make_ad_account(seed=7), AD_SPEC, "synthetic ads (seed 7)")
        return _set_history(STATE.history, STATE.spec, STATE.source)
    return STATE.coach


def _creative(
    headline: str,
    body: str,
    cta_text: str,
    author: str,
    vertical: str,
    placement: str,
    device: str,
    audience: str,
    daily_budget_usd: float,
    week_index: int,
) -> AdCreative:
    return AdCreative(
        headline=headline,
        body=body,
        cta_text=cta_text,
        author=author,  # type: ignore[arg-type]
        campaign_id="new",
        vertical=vertical,
        placement=placement,
        device=device,
        audience=audience,
        daily_budget_usd=daily_budget_usd,
        week_index=week_index,
    )


def _post_draft(
    text, author, posted_hour, weekday, has_media, is_reply, topic, week_index
) -> PostDraft:
    coach = _ensure_loaded()
    if week_index < 0:
        week_index = (
            int(coach.history["week_index"].max()) if "week_index" in coach.history.columns else 0
        )
    return PostDraft(
        text=text,
        author=author,
        context={
            "posted_hour": posted_hour,
            "weekday": weekday,
            "has_media": int(has_media),
            "is_reply": int(is_reply),
            "topic": topic,
            "week_index": week_index,
        },
    )


def _needs_threads(coach: CopyCoach) -> str | None:
    if coach.spec.name != "threads":
        return json.dumps({"error": "load a Threads timeline first (load_threads)"})
    return None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@mcp.tool()
def load_account(csv_path: str = "", n_campaigns: int = 120, seed: int = 7) -> str:
    """Load an advertiser's creative history (ads schema) and fit the model on it.

    Pass a CSV path in the AdLift ads schema (see README) or leave it empty
    to use the synthetic account with a known planted effect. Fitting is
    in-context, so this returns in seconds.
    """
    if csv_path:
        frame, source = frame_from_csv(csv_path), csv_path
    else:
        frame, source = (
            make_ad_account(n_campaigns=n_campaigns, seed=seed),
            f"synthetic ads (seed {seed})",
        )
    coach = _set_history(frame, AD_SPEC, source)
    return json.dumps(
        {
            "source": source,
            "spec": "ads",
            "rows": int(len(frame)),
            "campaigns": int(frame.campaign_id.nunique()),
            "author_mix": frame.author.value_counts(normalize=True).round(3).to_dict(),
            "model_backend": coach.model.backend,
            "fit_seconds": round(coach.fit_seconds, 3),
        }
    )


@mcp.tool()
def load_threads(
    csv_path: str = "",
    access_token: str = "",
    outcome: str = "views",
    n_weeks: int = 16,
    seed: int = 7,
    tz: str = "UTC",
) -> str:
    """Load a Threads timeline and fit the model on it.

    Pass a labelled posts CSV (text, timestamp, views, author, ...), or a
    Threads user access token to pull your own posts and insights through
    the API, or neither for the synthetic timeline. ``outcome`` is ``views``
    (modelled on a log scale) or ``engagement``.
    """
    spec = get_spec("threads", outcome=outcome)
    if csv_path:
        frame, source = threads_frame_from_csv(csv_path, tz=tz), csv_path
    elif access_token:
        frame, source = fetch_threads_account(access_token, tz=tz), "threads api"
    else:
        frame, source = (
            make_threads_account(n_weeks=n_weeks, seed=seed),
            f"synthetic threads (seed {seed})",
        )
    coach = _set_history(frame, spec, source)
    return json.dumps(
        {
            "source": source,
            "spec": "threads",
            "outcome": spec.outcome.column,
            "rows": int(len(frame)),
            "weeks": int(frame["week"].nunique()),
            "author_mix": frame["author"].value_counts(normalize=True).round(3).to_dict(),
            "model_backend": coach.model.backend,
            "fit_seconds": round(coach.fit_seconds, 3),
        }
    )


# ---------------------------------------------------------------------------
# Ads
# ---------------------------------------------------------------------------


@mcp.tool()
def score_creative(
    headline: str,
    body: str = "",
    cta_text: str = "",
    author: str = "human",
    vertical: str = "saas",
    placement: str = "social_feed",
    device: str = "mobile",
    audience: str = "prospecting",
    daily_budget_usd: float = 300.0,
    week_index: int = 20,
) -> str:
    """Predict click-through rate for an ad creative that has never served.

    Returns the point prediction and an uncertainty band. With hosted TabPFN
    the band is the model's own predictive distribution.
    """
    coach = _ensure_loaded()
    if coach.spec.name != "ads":
        return json.dumps({"error": "load an ads account first (load_account)"})
    creative = _creative(
        headline,
        body,
        cta_text,
        author,
        vertical,
        placement,
        device,
        audience,
        daily_budget_usd,
        week_index,
    )
    [scored] = coach.score([creative])
    return json.dumps({**scored.to_dict(), "model_backend": coach.model.backend})


@mcp.tool()
def revise_creative(
    headline: str,
    body: str = "",
    cta_text: str = "",
    author: str = "human",
    n_variants: int = 6,
    max_words: int = 0,
    tone: str = "",
    vertical: str = "saas",
    placement: str = "social_feed",
    device: str = "mobile",
    audience: str = "prospecting",
    daily_budget_usd: float = 300.0,
    week_index: int = 20,
) -> str:
    """Rewrite an ad draft under data-derived limits and rank the results by predicted CTR.

    Leave max_words and tone empty to derive them from what has worked for
    this account. The response lists every variant with its predicted lift
    over the draft.
    """
    coach = _ensure_loaded()
    if coach.spec.name != "ads":
        return json.dumps({"error": "load an ads account first (load_account)"})
    draft = _creative(
        headline,
        body,
        cta_text,
        author,
        vertical,
        placement,
        device,
        audience,
        daily_budget_usd,
        week_index,
    )
    constraints = None
    if max_words or tone:
        from adlift.loop import constraints_from_history

        constraints = constraints_from_history(coach.history, coach.spec)
        if max_words:
            constraints.max_words = max_words
        if tone:
            constraints.tone = tone
    return json.dumps(coach.revise(draft, n_variants=n_variants, constraints=constraints).to_dict())


@mcp.tool()
def ingest_banner(
    image_path_or_url: str,
    author: str = "human",
    vertical: str = "",
    placement: str = "unknown",
    device: str = "unknown",
    audience: str = "unknown",
    daily_budget_usd: float = 0.0,
    week_index: int = 0,
    score: bool = True,
) -> str:
    """Read a banner image into a structured ad creative, optionally scoring it.

    Uses a vision-capable language model to transcribe the text and describe
    the layout. The placement context is yours to supply; a picture cannot
    tell you where it will run.
    """
    context: dict[str, Any] = {
        "placement": placement,
        "device": device,
        "audience": audience,
        "daily_budget_usd": daily_budget_usd,
        "week_index": week_index,
    }
    if vertical:
        context["vertical"] = vertical
    creative = banner_to_creative(image_path_or_url, llm=get_llm(), author=author, context=context)
    payload: dict[str, Any] = {"creative": creative.to_row(), "extra": creative.extra}
    if score:
        coach = _ensure_loaded()
        if coach.spec.name == "ads":
            [scored] = coach.score([creative])
            payload["prediction"] = scored.to_dict()
    return json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# Threads
# ---------------------------------------------------------------------------


@mcp.tool()
def score_post(
    text: str,
    author: str = "human",
    posted_hour: int = 20,
    weekday: str = "Tue",
    has_media: bool = False,
    is_reply: bool = False,
    topic: str = "unknown",
    week_index: int = -1,
    n_precedents: int = 5,
) -> str:
    """Predict views (or engagement) for a post before it goes out.

    Returns the prediction, its band, and the closest past posts with what
    they got. Requires a Threads timeline to be loaded (load_threads).
    """
    coach = _ensure_loaded()
    if (error := _needs_threads(coach)) is not None:
        return error
    draft = _post_draft(text, author, posted_hour, weekday, has_media, is_reply, topic, week_index)
    [scored] = coach.score([draft])
    payload: dict[str, Any] = {
        **scored.to_dict(),
        "outcome": coach.spec.outcome.label,
        "model_backend": coach.model.backend,
    }
    if n_precedents:
        payload["precedents"] = coach.precedents(draft, k=n_precedents).to_dict(orient="records")
    return json.dumps(payload, default=str)


@mcp.tool()
def revise_post(
    text: str,
    author: str = "human",
    n_variants: int = 6,
    posted_hour: int = 20,
    weekday: str = "Tue",
    has_media: bool = False,
    is_reply: bool = False,
    topic: str = "unknown",
    week_index: int = -1,
) -> str:
    """Rewrite a post under the account's own limits and rank the variants by predicted outcome."""
    coach = _ensure_loaded()
    if (error := _needs_threads(coach)) is not None:
        return error
    draft = _post_draft(text, author, posted_hour, weekday, has_media, is_reply, topic, week_index)
    return json.dumps(coach.revise(draft, n_variants=n_variants).to_dict())


@mcp.tool()
def suggest_post(
    text: str,
    author: str = "human",
    posted_hour: int = 20,
    weekday: str = "Tue",
    has_media: bool = False,
    is_reply: bool = False,
    topic: str = "unknown",
    week_index: int = -1,
    top: int = 6,
) -> str:
    """Which single change (time, media, length, hook, hashtags, ...) raises this post's predicted outcome most."""
    coach = _ensure_loaded()
    if (error := _needs_threads(coach)) is not None:
        return error
    draft = _post_draft(text, author, posted_hour, weekday, has_media, is_reply, topic, week_index)
    table = suggest_for_post(
        coach.history,
        draft.to_row(coach.spec),
        spec=coach.spec,
        model_factory=lambda: CTRModel(backend=coach.model.backend, spec=coach.spec),
        top=top,
    )
    return json.dumps(
        {"as_written": table.attrs.get("as_is"), "suggestions": table.to_dict(orient="records")},
        default=str,
    )


# ---------------------------------------------------------------------------
# Reports (either spec)
# ---------------------------------------------------------------------------


@mcp.tool()
def lever_report(n_boot: int = 100) -> str:
    """Effect of every lever the creator controls, relative to the account's habits, with intervals."""
    coach = _ensure_loaded()
    table = lever_analysis(
        coach.history,
        spec=coach.spec,
        model_factory=lambda: CTRModel(backend=coach.model.backend, spec=coach.spec),
        n_boot=n_boot,
    )
    return json.dumps(
        {
            "source": STATE.source,
            "outcome": coach.spec.outcome.label,
            "levers": table.to_dict(orient="records"),
        },
        default=str,
    )


@mcp.tool()
def learning_curve_report(start: int = 5, step: int = 5) -> str:
    """How soon the ranking becomes useful as the account accumulates rows (time order)."""
    coach = _ensure_loaded()
    table = learning_curve(
        coach.history,
        CTRModel(backend=coach.model.backend, spec=coach.spec),
        start=start,
        step=step,
    )
    return json.dumps(
        {"source": STATE.source, "curve": table.round(4).to_dict(orient="records")}, default=str
    )


@mcp.tool()
def causal_report(n_boot: int = 200, segment_by: str = "", refit_boot: int = 20) -> str:
    """Estimate whether LLM-written copy causally out- or under-performs human copy.

    Runs S- and T-learners on the confounder set with a cluster bootstrap,
    plus a placebo test, an overlap diagnostic and a mediation split. Every
    number comes with its check. Log-scale outcomes also report a ratio.
    """
    coach = _ensure_loaded()
    frame, spec = coach.history, coach.spec
    analysis = full_analysis(
        frame,
        model_factory=lambda: CTRModel(backend=coach.model.backend, spec=spec),
        spec=spec,
        n_boot=n_boot,
        segment_by=segment_by or None,
        refit_boot=refit_boot,
    )
    total, direct, cross = (
        analysis["total_effect"],
        analysis["direct_effect"],
        analysis["cross_check"],
    )
    payload = {
        "source": STATE.source,
        "outcome": spec.outcome.label,
        "effect_unit": spec.outcome.effect_unit,
        "naive_difference": analysis["naive_difference"],
        "naive_ratio": analysis["naive_ratio"],
        "total_effect": {
            "ate": total.ate,
            "ci_low": total.ci_low,
            "ci_high": total.ci_high,
            "ratio": total.ratio,
            "ratio_low": total.ratio_low,
            "ratio_high": total.ratio_high,
            "interval": total.interval(),
            "significant": total.significant,
            "summary": total.describe(),
        },
        "s_learner_total": cross.ate,
        "median_effect": analysis["median_effect"],
        "direct_effect": direct.ate,
        "mediation": {k: round(v, 5) for k, v in analysis["mediation"].items()},
        "checks": {
            "estimators_agree": analysis["estimators_agree"],
            "placebo_ratio": round(analysis["placebo"]["ratio"], 2),
            "overlap_auc": round(analysis["overlap"]["auc"], 3),
            "share_off_support": round(analysis["overlap"]["share_off_support"], 4),
            "detectable_effect": analysis["detectable_effect"],
            "detectable_ratio": analysis.get("detectable_ratio"),
            "n_llm": analysis["n_llm"],
            "n_human": analysis["n_human"],
        },
        "segments": analysis["segments"].round(5).to_dict(orient="records"),
    }
    if "_truth_total_effect" in frame.columns:
        payload["planted_truth"] = {
            "total": float(frame["_truth_total_effect"].mean()),
            "direct": float(frame["_truth_direct_effect"].mean()),
        }
    return json.dumps(payload, default=str)


@mcp.tool()
def cold_start_benchmark(n_splits: int = 4) -> str:
    """Rank rows in held-out groups and report regret, recall and budget saved."""
    coach = _ensure_loaded()
    frame, spec = coach.history, coach.spec
    results = [random_baseline(frame, spec=spec)]
    results.append(
        evaluate_cold_start(
            frame, CTRModel(backend="baseline", spec=spec), name="GBDT + TF-IDF", n_splits=n_splits
        )
    )
    if coach.model.backend != "baseline":
        label = "TabPFN 3.5" if coach.model.backend == "client" else "TabPFN (local)"
        results.append(
            evaluate_cold_start(
                frame,
                CTRModel(backend=coach.model.backend, spec=spec),
                name=label,
                n_splits=n_splits,
            )
        )
    return comparison_table(results).to_json(orient="records")


@mcp.tool()
def account_status() -> str:
    """What is loaded, which backends are active, and whether keys are present."""
    return json.dumps(
        {
            "source": STATE.source,
            "spec": STATE.spec.name,
            "outcome": STATE.spec.outcome.column,
            "rows": int(len(STATE.history)) if STATE.history is not None else 0,
            "model_backend": STATE.backend,
            "tabpfn_token_present": bool(os.environ.get("TABPFN_TOKEN")),
            "anthropic_key_present": bool(
                os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
            ),
            "llm_backend": get_llm().name,
        }
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
