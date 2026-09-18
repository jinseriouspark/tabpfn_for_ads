"""AdLift as an MCP server.

Run it and any MCP client (Claude Desktop, Claude Code, Cursor, an agent
framework) can load an ad account, score a creative before it serves, ask for
constrained rewrites and read the causal report, all through tool calls.

    adlift-mcp                      # stdio transport, the default
    python -m mcp_server.server     # same thing

State is one loaded account and one fitted coach. Loading is cheap (TabPFN
fits in seconds), so reloading with a different CSV is the normal way to
switch advertisers.
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
from adlift.coldstart import comparison_table, evaluate_cold_start, random_baseline
from adlift.datasets.synth import make_ad_account, true_ate
from adlift.ingest.adapter import frame_from_csv
from adlift.ingest.banner import banner_to_creative
from adlift.llm import get_llm
from adlift.loop import CopyCoach
from adlift.model import CTRModel, resolve_backend
from adlift.schema import AdCreative, validate

mcp = FastMCP(
    "adlift",
    instructions=(
        "AdLift predicts ad-creative click-through rate before a creative serves, "
        "rewrites copy under data-derived constraints, and estimates whether LLM-written "
        "copy causally outperforms human-written copy. Call load_account first "
        "(or rely on the synthetic default), then score_creative / revise_creative / causal_report."
    ),
)


@dataclass
class _State:
    history: pd.DataFrame | None = None
    coach: CopyCoach | None = None
    source: str = "none"
    backend: str = resolve_backend(os.environ.get("ADLIFT_BACKEND", "auto"))  # type: ignore[arg-type]


STATE = _State()


def _ensure_loaded() -> CopyCoach:
    if STATE.coach is None:
        if STATE.history is None:
            STATE.history = make_ad_account(seed=7)
            STATE.source = "synthetic (seed 7)"
        STATE.coach = CopyCoach(
            STATE.history, model=CTRModel(backend=STATE.backend), llm=get_llm()
        ).fit()
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


@mcp.tool()
def load_account(csv_path: str = "", n_campaigns: int = 120, seed: int = 7) -> str:
    """Load an advertiser's creative history and fit the model on it.

    Pass a CSV path in the AdLift schema (see README) or leave it empty to use
    the synthetic account with a known planted effect. Fitting is in-context,
    so this returns in seconds.
    """
    if csv_path:
        frame = frame_from_csv(csv_path)
        STATE.source = csv_path
    else:
        frame = make_ad_account(n_campaigns=n_campaigns, seed=seed)
        STATE.source = f"synthetic (seed {seed})"
    validate(frame)
    STATE.history = frame
    STATE.coach = CopyCoach(frame, model=CTRModel(backend=STATE.backend), llm=get_llm()).fit()
    return json.dumps(
        {
            "source": STATE.source,
            "rows": int(len(frame)),
            "campaigns": int(frame.campaign_id.nunique()),
            "author_mix": frame.author.value_counts(normalize=True).round(3).to_dict(),
            "model_backend": STATE.coach.model.backend,
            "fit_seconds": round(STATE.coach.fit_seconds, 3),
        }
    )


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
    """Predict click-through rate for a creative that has never served.

    Returns the point prediction and an uncertainty band. With hosted TabPFN
    the band is the model's own predictive distribution.
    """
    coach = _ensure_loaded()
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
    """Rewrite a draft under data-derived limits and rank the results by predicted CTR.

    Leave max_words and tone empty to derive them from what has worked for
    this account. The response lists every variant with its predicted lift
    over the draft.
    """
    coach = _ensure_loaded()
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

        constraints = constraints_from_history(coach.history)
        if max_words:
            constraints.max_words = max_words
        if tone:
            constraints.tone = tone
    result = coach.revise(draft, n_variants=n_variants, constraints=constraints)
    return json.dumps(result.to_dict())


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
    """Read a banner image into a structured creative, optionally scoring it.

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
        [scored] = coach.score([creative])
        payload["prediction"] = scored.to_dict()
    return json.dumps(payload, default=str)


@mcp.tool()
def causal_report(n_boot: int = 200, segment_by: str = "device") -> str:
    """Estimate whether LLM-written copy causally out- or under-performs human copy.

    Runs S- and T-learners on the confounder set with a cluster bootstrap,
    plus a placebo test, an overlap diagnostic and a mediation split. Every
    number comes with its check.
    """
    coach = _ensure_loaded()
    frame = coach.history
    analysis = full_analysis(
        frame,
        model_factory=lambda: CTRModel(backend=coach.model.backend),
        n_boot=n_boot,
        segment_by=segment_by,
    )
    total, direct, cross = (
        analysis["total_effect"],
        analysis["direct_effect"],
        analysis["cross_check"],
    )
    payload = {
        "source": STATE.source,
        "naive_difference_pp": round(analysis["naive_difference"] * 100, 4),
        "total_effect": {
            "ate_pp": round(total.ate * 100, 4),
            "ci_low_pp": round(total.ci_low * 100, 4),
            "ci_high_pp": round(total.ci_high * 100, 4),
            "significant": total.significant,
            "summary": total.describe(),
        },
        "t_learner_total_pp": round(cross.ate * 100, 4),
        "direct_effect_pp": round(direct.ate * 100, 4),
        "mediation": {k: round(v, 5) for k, v in analysis["mediation"].items()},
        "checks": {
            "estimators_agree": analysis["estimators_agree"],
            "placebo_ratio": round(analysis["placebo"]["ratio"], 2),
            "overlap_auc": round(analysis["overlap"]["auc"], 3),
            "share_off_support": round(analysis["overlap"]["share_off_support"], 4),
        },
        "segments": analysis["segments"].round(5).to_dict(orient="records"),
    }
    if "_truth_total_effect" in frame.columns:
        payload["planted_truth"] = {
            "total_pp": round(true_ate(frame, "total") * 100, 4),
            "direct_pp": round(true_ate(frame, "direct") * 100, 4),
        }
    return json.dumps(payload)


@mcp.tool()
def cold_start_benchmark(n_splits: int = 4) -> str:
    """Rank creatives in held-out campaigns and report regret, recall and budget saved."""
    coach = _ensure_loaded()
    frame = coach.history
    results = [random_baseline(frame)]
    results.append(
        evaluate_cold_start(
            frame, CTRModel(backend="baseline"), name="GBDT + TF-IDF", n_splits=n_splits
        )
    )
    if coach.model.backend != "baseline":
        label = "TabPFN 3.5" if coach.model.backend == "client" else "TabPFN (local)"
        results.append(
            evaluate_cold_start(
                frame, CTRModel(backend=coach.model.backend), name=label, n_splits=n_splits
            )
        )
    return comparison_table(results).to_json(orient="records")


@mcp.tool()
def account_status() -> str:
    """What is loaded, which backends are active, and whether keys are present."""
    return json.dumps(
        {
            "source": STATE.source,
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
