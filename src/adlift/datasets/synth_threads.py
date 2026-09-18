"""A synthetic Threads timeline with known lever effects.

One account, a few months of posts. Every lever a person can pull when they
publish has a planted effect on the log of views: posting hour, weekday,
attaching media, replying versus posting, length (an inverted U), a question
hook, a number, hashtags, links, emoji, the topic, and the author. Adoption of
LLM-written posts climbs over the period and the account's reach drifts up
with it, so the naive author comparison is confounded exactly the way a real
timeline is. Views are heavy-tailed: a small share of posts catch the
algorithm and run.

As with the ads generator, both potential texts are written for every post,
so the true author effect is known. Columns starting with ``_truth_`` are
ground truth and must never be given to a model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from adlift.schema import THREADS_SPEC
from adlift.text_features import text_features

TOPICS = ["ai_tools", "career", "productivity", "marketing", "personal", "product_launch"]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

HUMAN_POSTS = [
    "Cut my {metric} by {pct}% with one change: {change}. That's it.",
    "{pct}% of the {audience} I talk to still {mistake}. Stop.",
    "Shipped {product} in {days} days. Here's what I'd skip next time: {change}.",
    "Unpopular opinion: {change} beats {pain} every single time.",
    "Tried {product} for {days} days. Verdict: {verdict}.",
    "Question for {audience}: does {change} actually fix {pain}, or is it just me?",
    "{n} people asked about {change}. Short answer: {verdict}.",
    "The only {metric} tip that survived contact with reality: {change}.",
]
LLM_POSTS = [
    "Excited to share how {product} is transforming the way {audience} approach {pain}! 🚀 "
    "Discover the game-changing potential of {change}. #{tag1} #{tag2} #{tag3}",
    "Unlock the secret to better {metric}: {change}. It's time to reimagine what's possible "
    "for {audience} everywhere. ✨ #{tag1} #{tag2} #{tag3} #{tag4}",
    "Say goodbye to {pain} and hello to effortless {metric}! Join thousands of forward-thinking "
    "{audience} embracing {change} today. 💡 #{tag1} #{tag2}",
    "Thrilled to announce a journey toward seamless {metric} with {product}. Empower your "
    "workflow and elevate your results! 🌟 #{tag1} #{tag2} #{tag3}",
    "In today's fast-paced world, {audience} deserve tools that just work. That's why "
    "{change} matters more than ever. Let's dive in! 🔥 #{tag1} #{tag2} #{tag3}",
]
TAGS = ["ai", "productivity", "buildinpublic", "startup", "marketing", "growth", "tech", "founder"]
PRODUCTS = ["Northwind", "Atlas", "Cobalt", "Meridian", "Vector"]
METRICS = ["support load", "churn", "ad spend", "onboarding time", "reply rate", "review time"]
PAINS = ["manual reviews", "spreadsheet chaos", "slow approvals", "ticket backlogs", "silos"]
CHANGES = ["a weekly review", "one dashboard", "shorter posts", "batching replies", "saying no"]
AUDIENCES = ["founders", "marketers", "ops teams", "solo devs", "agencies"]


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _write_post(rng: np.random.Generator, author: str) -> str:
    fills = {
        "product": rng.choice(PRODUCTS),
        "metric": rng.choice(METRICS),
        "pain": rng.choice(PAINS),
        "change": rng.choice(CHANGES),
        "audience": rng.choice(AUDIENCES),
        "pct": int(rng.integers(15, 70)),
        "days": int(rng.integers(3, 30)),
        "n": int(rng.integers(5, 80)),
        "verdict": rng.choice(["worth it", "not yet", "only for small teams", "skip it"]),
        "mistake": rng.choice(["post at 3am", "write for everyone", "skip the number"]),
    }
    tags = rng.choice(TAGS, size=4, replace=False)
    fills.update({f"tag{i + 1}": tags[i] for i in range(4)})
    template = rng.choice(HUMAN_POSTS if author == "human" else LLM_POSTS)
    text = str(template).format(**fills)
    if author == "human" and rng.random() < 0.15:
        text += " https://example.com/notes"
    if author == "llm" and rng.random() < 0.35:
        text += " Learn more: https://example.com/launch"
    return text


_HOUR_LIFT = {
    h: v
    for h, v in zip(
        range(24),
        [
            -0.55,
            -0.7,
            -0.8,
            -0.85,
            -0.8,
            -0.6,
            -0.3,
            -0.05,
            0.1,
            0.15,
            0.1,
            0.05,
            0.15,
            0.1,
            0.0,
            -0.05,
            0.05,
            0.15,
            0.3,
            0.4,
            0.5,
            0.45,
            0.25,
            -0.1,
        ],
        strict=True,
    )
}
_WEEKDAY_LIFT = {
    "Mon": 0.05,
    "Tue": 0.15,
    "Wed": 0.12,
    "Thu": 0.1,
    "Fri": -0.05,
    "Sat": -0.25,
    "Sun": -0.2,
}
_TOPIC_LIFT = {
    "ai_tools": 0.35,
    "career": 0.15,
    "productivity": 0.0,
    "marketing": -0.1,
    "personal": 0.2,
    "product_launch": -0.25,
}


def _log_views(context: dict, features: dict, shared: dict) -> float:
    length_term = -0.00045 * (features["word_count"] - 32) ** 2
    return (
        5.6
        + 0.045 * context["week_index"]  # the account grows
        + _HOUR_LIFT[context["posted_hour"]]
        + _WEEKDAY_LIFT[context["weekday"]]
        + 0.25 * context["has_media"]
        - 0.9 * context["is_reply"]
        + _TOPIC_LIFT[context["topic"]]
        + length_term
        + 0.15 * features["has_question"]
        + 0.12 * features["has_numeric_claim"]
        - 0.08 * max(0, features["n_hashtags"] - 1)
        - 0.35 * features["has_link"]
        + 0.05 * features["has_emoji"]
        - 0.05 * features["has_cta"]
        + shared["week_effect"]
        + shared["noise"]
        + shared["viral"]
    )


def make_threads_account(
    n_weeks: int = 16,
    posts_per_week: tuple[int, int] = (8, 14),
    direct_effect_log: float = -0.06,
    seed: int = 7,
) -> pd.DataFrame:
    """Generate one account's timeline with both potential texts per post.

    Args:
        n_weeks: Length of the timeline. Weeks are the grouping unit.
        posts_per_week: Inclusive range of posts per week.
        direct_effect_log: Planted direct effect of LLM authorship on log
            views, holding the post's visible attributes fixed. Negative means
            readers respond slightly worse to model wording even when it looks
            the same on paper.
        seed: Seed for the generator.

    Returns:
        The canonical Threads table plus hidden ``_truth_*`` columns.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    post_number = 0

    for week in range(n_weeks):
        week_effect = float(rng.normal(0.0, 0.2))  # algorithm mood, follower spikes
        n_posts = int(rng.integers(posts_per_week[0], posts_per_week[1] + 1))
        for _ in range(n_posts):
            weekday = str(rng.choice(WEEKDAYS, p=[0.17, 0.18, 0.17, 0.16, 0.14, 0.09, 0.09]))
            hour = int(rng.choice(24, p=_HOUR_CHOICE_P))
            topic = str(rng.choice(TOPICS))
            context = {
                "posted_hour": hour,
                "weekday": weekday,
                "has_media": int(rng.random() < 0.35),
                "is_reply": int(rng.random() < 0.2),
                "topic": topic,
                "week_index": week,
            }

            # Adoption climbs over the period and clusters in a few topics.
            llm_logit = (
                -0.6
                + 2.0 * ((week - n_weeks / 2) / (n_weeks / 2))
                + 0.8 * (topic in {"ai_tools", "product_launch"})
                - 0.5 * (topic == "personal")
            )
            author = "llm" if rng.random() < _sigmoid(llm_logit) else "human"

            shared = {
                "week_effect": week_effect,
                "noise": float(rng.standard_t(df=5) * 0.45),
                "viral": float(2.2 + rng.exponential(0.5)) if rng.random() < 0.04 else 0.0,
            }

            human_text = _write_post(rng, "human")
            llm_text = _write_post(rng, "llm")
            human_features = text_features(human_text)
            llm_features = text_features(llm_text)

            log_human = _log_views(context, human_features, shared)
            log_llm = _log_views(context, llm_features, shared) + direct_effect_log
            views_human = float(np.exp(log_human))
            views_llm = float(np.exp(log_llm))

            text = llm_text if author == "llm" else human_text
            features = llm_features if author == "llm" else human_features
            expected = views_llm if author == "llm" else views_human
            views = int(max(1, rng.poisson(expected)))

            # The structural expectation: what the post's own attributes and
            # context earn it, with the week shock, the noise and any viral
            # run removed. This is the "true quality" a ranker can be judged
            # against; nothing in the features predicts the shocks.
            quiet = {"week_effect": 0.0, "noise": 0.0, "viral": 0.0}
            structural = _log_views(context, features, quiet) + (
                direct_effect_log if author == "llm" else 0.0
            )

            like_rate = float(
                np.clip(0.03 + 0.01 * features["has_question"] + rng.normal(0, 0.008), 0.005, 0.15)
            )
            likes = int(rng.binomial(views, like_rate))
            replies = int(rng.binomial(views, like_rate * 0.18))
            reposts = int(rng.binomial(views, like_rate * 0.08))
            quotes = int(rng.binomial(views, like_rate * 0.03))
            engagement_rate = (likes + replies + reposts + quotes) / views

            hour_offset = pd.Timedelta(hours=hour, minutes=int(rng.integers(0, 60)))
            day = pd.Timestamp("2026-05-04") + pd.Timedelta(
                weeks=week, days=WEEKDAYS.index(weekday)
            )
            rows.append(
                {
                    "post_id": f"post_{post_number:05d}",
                    "week": f"W{week:02d}",
                    "text": text,
                    **features,
                    **context,
                    "author": author,
                    "views": views,
                    "likes": likes,
                    "replies": replies,
                    "reposts": reposts,
                    "quotes": quotes,
                    "engagement_rate": engagement_rate,
                    "timestamp": (day + hour_offset).isoformat(),
                    # Ground truth. Never a feature.
                    "_truth_views_human": views_human,
                    "_truth_views_llm": views_llm,
                    "_truth_total_effect": views_llm - views_human,
                    "_truth_total_log_effect": log_llm - log_human,
                    "_truth_direct_effect": views_human * float(np.expm1(direct_effect_log)),
                    "_truth_direct_log_effect": direct_effect_log,
                    "_truth_views_noiseless": float(np.exp(structural)),
                }
            )
            post_number += 1

    frame = pd.DataFrame(rows)
    truth = [c for c in frame.columns if c.startswith("_truth_")]
    return frame[THREADS_SPEC.all_columns + ["timestamp"] + truth]


_HOUR_CHOICE_P = np.array(
    [1, 1, 1, 1, 1, 1, 2, 4, 6, 7, 6, 5, 6, 5, 4, 4, 5, 6, 7, 8, 8, 7, 4, 2], dtype=float
)
_HOUR_CHOICE_P /= _HOUR_CHOICE_P.sum()


def true_ate(frame: pd.DataFrame, estimand: str = "total") -> float:
    """Read the planted average author effect back out, on the views scale."""
    column = {"total": "_truth_total_effect", "direct": "_truth_direct_effect"}.get(estimand)
    if column is None or column not in frame.columns:
        raise ValueError("frame carries no ground truth for that estimand")
    return float(frame[column].mean())


def true_log_ate(frame: pd.DataFrame, estimand: str = "total") -> float:
    """The planted author effect on the log scale, as a ratio minus one."""
    column = {"total": "_truth_total_log_effect", "direct": "_truth_direct_log_effect"}.get(
        estimand
    )
    if column is None or column not in frame.columns:
        raise ValueError("frame carries no ground truth for that estimand")
    return float(np.expm1(frame[column].mean()))
