"""A synthetic ad account with a known causal effect.

Real ad-performance data with creative text attached is proprietary, so AdLift
ships a generator instead. The point is not realism for its own sake. It is
that the generator writes down both potential outcomes for every creative --
the click-through rate it would get if a human wrote it, and the rate it would
get if an LLM wrote it -- so the true average treatment effect is known exactly
and every estimator in this package can be scored against it.

The generator deliberately plants the three problems that make this hard:

Confounding
    LLM adoption rises over time and concentrates in a few verticals and
    placements. Those same factors move click-through rate on their own, so a
    naive human-versus-LLM average is biased.

Mediation
    The author changes how the copy is written -- length, tone, whether it
    carries a number -- and those attributes move performance too. Adjusting
    for them would remove part of the very effect we want, so they are kept out
    of the confounder set.

Grouping
    Creatives are nested in campaigns that share a budget, an audience and a
    random performance offset. Rows inside a campaign are not independent,
    which is what makes an ungrouped split leak.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from adlift.schema import ALL_COLUMNS, GROUP_COLUMN

VERTICALS = ["ecommerce", "saas", "fintech", "travel", "healthtech", "gaming"]
PLACEMENTS = ["social_feed", "search", "display", "video_preroll"]
DEVICES = ["mobile", "desktop", "tablet"]
AUDIENCES = ["prospecting", "retargeting", "lookalike"]
CLAIM_TYPES = ["performance", "price", "outcome", "capability", "none"]
TONES = ["direct", "aspirational", "urgent", "playful", "technical"]
BACKGROUNDS = ["photo", "solid", "gradient", "illustration"]
COLORS = ["blue", "purple", "green", "red", "monochrome", "orange"]

# Copy templates. Human copy is terser and leans on concrete numbers; LLM copy
# is smoother, longer and reaches for aspirational verbs. Both carry real
# signal, so a text-aware model has something to find.
HUMAN_HEADLINES = [
    "Cut {metric} by {pct}%.",
    "{pct}% lower {metric}. No setup.",
    "{product} ships in {days} days.",
    "Stop paying for {pain}.",
    "{product}, from ${price} a month.",
    "Built for {audience}. Priced for {audience}.",
    "{pct}% of teams switch within a month.",
    "Fix {pain} this week.",
]
LLM_HEADLINES = [
    "Discover how {product} can transform the way you handle {pain}.",
    "Unlock smarter {metric} with {product}, designed for modern {audience}.",
    "Say goodbye to {pain} and hello to effortless {metric}.",
    "Transform your {metric} with a solution built around {audience}.",
    "Experience {product}: where {metric} meets simplicity.",
    "Elevate your workflow and leave {pain} behind for good.",
    "Reimagine {metric} with {product}, trusted by teams everywhere.",
]
HUMAN_BODIES = [
    "Free for 14 days. Cancel anytime.",
    "No card needed.",
    "Used by {n} teams.",
    "Setup takes {days} minutes.",
    "",
]
LLM_BODIES = [
    "Join thousands of forward-thinking teams who have already made the switch.",
    "Our intuitive platform adapts seamlessly to the way your team already works.",
    "Start your journey toward better {metric} today, with no commitment required.",
    "Empower your team with tools that grow alongside your ambitions.",
    "",
]
HUMAN_CTAS = ["Start free", "See pricing", "Get the demo", "Try it", ""]
LLM_CTAS = ["Begin your journey", "Discover more today", "Unlock your potential", "Learn more", ""]

PRODUCTS = ["Northwind", "Atlas", "Cobalt", "Meridian", "Vector", "Harbor"]
METRICS = ["support costs", "churn", "ad spend", "load times", "onboarding", "claim review"]
PAINS = ["manual reviews", "spreadsheet chaos", "slow approvals", "ticket backlogs", "data silos"]


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def _write_copy(rng: np.random.Generator, author: str) -> tuple[str, str, str]:
    """Render one creative's text in the house style of ``author``."""
    fills = {
        "product": rng.choice(PRODUCTS),
        "metric": rng.choice(METRICS),
        "pain": rng.choice(PAINS),
        "audience": rng.choice(["startups", "enterprises", "agencies", "ops teams"]),
        "pct": int(rng.integers(15, 70)),
        "days": int(rng.integers(2, 21)),
        "price": int(rng.integers(9, 99)),
        "n": f"{int(rng.integers(2, 40)) * 500:,}",
    }
    heads, bodies, ctas = (
        (HUMAN_HEADLINES, HUMAN_BODIES, HUMAN_CTAS)
        if author == "human"
        else (LLM_HEADLINES, LLM_BODIES, LLM_CTAS)
    )
    headline = str(rng.choice(heads)).format(**fills)
    body = str(rng.choice(bodies)).format(**fills)
    cta = str(rng.choice(ctas))
    return headline, body, cta


def _creative_attributes(rng: np.random.Generator, author: str) -> dict:
    """Draw the copy and its attributes in the house style of ``author``.

    This is the mediator block. Calling it twice for the same creative slot,
    once per author, is what makes a *total* effect computable: the copy itself
    changes when the author changes, and that change is part of the effect.
    """
    headline, body, cta = _write_copy(rng, author)
    text = " ".join(part for part in (headline, body, cta) if part)
    return {
        "headline": headline,
        "body": body,
        "cta_text": cta,
        "word_count": len(text.split()),
        "char_count": len(text),
        "has_numeric_claim": int(any(ch.isdigit() for ch in text)),
        "tone": str(
            rng.choice(
                TONES,
                p=[0.45, 0.15, 0.2, 0.1, 0.1] if author == "human" else [0.15, 0.45, 0.1, 0.2, 0.1],
            )
        ),
        "claim_type": str(
            rng.choice(
                CLAIM_TYPES,
                p=[0.3, 0.25, 0.2, 0.15, 0.1] if author == "human" else [0.15, 0.1, 0.3, 0.25, 0.2],
            )
        ),
    }


_VERTICAL_LIFT = {
    "ecommerce": 0.30,
    "saas": 0.05,
    "fintech": -0.18,
    "travel": 0.22,
    "healthtech": -0.30,
    "gaming": 0.40,
}
_PLACEMENT_LIFT = {
    "social_feed": 0.18,
    "search": 0.45,
    "display": -0.35,
    "video_preroll": -0.05,
}
_DEVICE_LIFT = {"mobile": 0.12, "desktop": -0.05, "tablet": -0.10}
_AUDIENCE_LIFT = {"prospecting": -0.22, "retargeting": 0.46, "lookalike": 0.0}
_TONE_LIFT = {
    "direct": 0.08,
    "urgent": 0.05,
    "aspirational": -0.04,
    "playful": 0.0,
    "technical": -0.06,
}


def _outcome_logit(context: dict, creative: dict, shared: dict) -> float:
    """Click-through rate on the logit scale, before the treatment effect.

    Copy length helps up to a point and then hurts: the inverted U that a
    linear model flattens and a flexible one recovers.
    """
    length_term = -0.0022 * (creative["word_count"] - 14) ** 2
    return (
        -4.05
        + shared["campaign_effect"]
        + shared["noise"]
        + 0.020 * context["week_index"]
        + _VERTICAL_LIFT[context["vertical"]]
        + _PLACEMENT_LIFT[context["placement"]]
        + _DEVICE_LIFT[context["device"]]
        + _AUDIENCE_LIFT[context["audience"]]
        + 0.17 * creative["has_numeric_claim"]
        + 0.11 * shared["has_brand_logo"]
        + 0.06 * shared["has_human_face"]
        + 0.35 * (shared["text_contrast"] - 0.5)
        + length_term
        + _TONE_LIFT[creative["tone"]]
    )


def make_ad_account(
    n_campaigns: int = 120,
    creatives_per_campaign: tuple[int, int] = (4, 12),
    direct_effect: float = 0.0035,
    heterogeneity: float = 0.0030,
    seed: int = 7,
) -> pd.DataFrame:
    """Generate one advertiser's creative history with known potential outcomes.

    For every creative the generator writes the copy twice -- once the way a
    person would write it, once the way an LLM would -- and computes the
    click-through rate under each. Only one of the two is observed; both are
    kept in hidden ``_truth_*`` columns.

    That distinction is the whole point. Two different effects exist here and
    they do not agree:

    ``_truth_direct_effect``
        What ``direct_effect`` plants: the lift from LLM authorship with the
        copy's visible attributes held fixed.

    ``_truth_total_effect``
        What actually happens when an LLM writes the creative, including the
        fact that LLM copy runs longer, carries fewer numbers and leans
        aspirational. Those attributes have their own effect on performance,
        and it can point the other way.

    Args:
        n_campaigns: Number of campaigns; the grouping unit for splits and the
            cluster bootstrap.
        creatives_per_campaign: Inclusive range of creatives per campaign.
        direct_effect: Planted direct effect in absolute click-through points.
        heterogeneity: Device tilt on the direct effect. Positive values favour
            LLM copy on mobile and penalise it on desktop.
        seed: Seed for the generator.

    Returns:
        The canonical creative table plus hidden ``_truth_*`` columns. Any
        column starting with an underscore is ground truth, never a feature.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for campaign in range(n_campaigns):
        campaign_id = f"cmp_{campaign:04d}"
        context_base = {
            "vertical": str(rng.choice(VERTICALS)),
            "placement": str(rng.choice(PLACEMENTS)),
            "audience": str(rng.choice(AUDIENCES)),
            "week_index": int(rng.integers(0, 26)),
        }
        budget = float(np.round(np.exp(rng.normal(6.2, 0.7)), 2))
        campaign_effect = float(rng.normal(0.0, 0.28))
        n_creatives = int(rng.integers(creatives_per_campaign[0], creatives_per_campaign[1] + 1))

        for index in range(n_creatives):
            context = dict(context_base, device=str(rng.choice(DEVICES, p=[0.6, 0.3, 0.1])))

            # Treatment assignment, confounded on purpose: adoption climbs over
            # time and clusters in particular verticals and placements.
            llm_logit = (
                -0.45
                + 2.1 * ((context["week_index"] - 12.5) / 12.5)
                + 0.9 * (context["vertical"] in {"ecommerce", "saas"})
                + 0.6 * (context["placement"] == "social_feed")
                - 0.4 * (context["vertical"] == "healthtech")
            )
            author = "llm" if rng.random() < _sigmoid(llm_logit) else "human"

            # Whatever is unobserved about this creative slot is the same under
            # both authors, so it is drawn once and shared.
            shared = {
                "campaign_effect": campaign_effect,
                "noise": float(rng.normal(0.0, 0.16)),
                "has_brand_logo": int(rng.random() < 0.85),
                "has_human_face": int(rng.random() < 0.4),
                "text_contrast": float(np.clip(rng.beta(5, 2), 0.05, 0.99)),
            }

            # Both potential copies, and therefore both potential outcomes.
            as_human = _creative_attributes(rng, "human")
            as_llm = _creative_attributes(rng, "llm")

            device_tilt = {"mobile": 1.0, "desktop": -1.0, "tablet": 0.0}[context["device"]]
            row_direct = direct_effect + heterogeneity * device_tilt

            ctr_human = float(_sigmoid(_outcome_logit(context, as_human, shared)))
            ctr_llm = float(
                np.clip(_sigmoid(_outcome_logit(context, as_llm, shared)) + row_direct, 1e-5, 0.999)
            )

            observed_creative = as_llm if author == "llm" else as_human
            observed_ctr = ctr_llm if author == "llm" else ctr_human
            impressions = int(np.clip(rng.normal(budget * 55, budget * 12), 400, None))
            clicks = int(rng.binomial(impressions, observed_ctr))

            rows.append(
                {
                    "creative_id": f"{campaign_id}_cr{index:02d}",
                    GROUP_COLUMN: campaign_id,
                    **{
                        k: observed_creative[k]
                        for k in (
                            "headline",
                            "body",
                            "cta_text",
                            "word_count",
                            "char_count",
                            "has_numeric_claim",
                            "claim_type",
                            "tone",
                        )
                    },
                    "has_brand_logo": shared["has_brand_logo"],
                    "has_human_face": shared["has_human_face"],
                    "background_kind": str(rng.choice(BACKGROUNDS)),
                    "dominant_color": str(rng.choice(COLORS)),
                    "text_contrast": round(shared["text_contrast"], 4),
                    **context,
                    "daily_budget_usd": budget,
                    "author": author,
                    "impressions": impressions,
                    "clicks": clicks,
                    "ctr": clicks / impressions,
                    # Ground truth. Never a feature.
                    "_truth_ctr_human": ctr_human,
                    "_truth_ctr_llm": ctr_llm,
                    "_truth_total_effect": ctr_llm - ctr_human,
                    "_truth_direct_effect": row_direct,
                    "_truth_ctr_noiseless": observed_ctr,
                }
            )

    frame = pd.DataFrame(rows)
    truth = [c for c in frame.columns if c.startswith("_truth_")]
    return frame[ALL_COLUMNS + truth]


def true_ate(frame: pd.DataFrame, estimand: str = "total") -> float:
    """Read a planted average treatment effect back out of a generated frame.

    Args:
        frame: A frame from :func:`make_ad_account`.
        estimand: ``"total"`` for the effect of handing the brief to an LLM,
            copy changes included; ``"direct"`` for the effect with visible
            creative attributes held fixed.
    """
    column = {"total": "_truth_total_effect", "direct": "_truth_direct_effect"}.get(estimand)
    if column is None:
        raise ValueError(f"estimand must be 'total' or 'direct', got {estimand!r}")
    if column not in frame.columns:
        raise ValueError("frame carries no ground truth; it was not produced by make_ad_account")
    return float(frame[column].mean())
