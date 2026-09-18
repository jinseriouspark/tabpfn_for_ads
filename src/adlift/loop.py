"""The loop: a language model proposes copy, TabPFN scores it, instantly.

This is where TabPFN's two properties stop being benchmark numbers and become
a product. Because fitting is in-context, the advertiser's history is loaded
once and every subsequent creative, human-drafted or model-rewritten, is a
single predict call. There is no retraining step between "write a variant" and
"know how it will do", so the whole cycle runs at conversational speed.

The rewrite is *constrained*. The causal analysis in :mod:`adlift.causal`
says what LLM copy tends to do that costs clicks -- run long, drop the number,
reach for aspirational verbs. Those findings become guard-rails on the prompt.
The model is asked to polish wording inside limits the data set, which is what
"a human revising with AI help" should mean: the person keeps the facts and
the shape, the model keeps the phrasing sharp.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from adlift.llm import CopyLLM, RewriteConstraints, get_llm
from adlift.model import CTRModel
from adlift.schema import FEATURE_COLUMNS, GROUP_COLUMN, AdCreative, creatives_to_frame


@dataclass(slots=True)
class ScoredCreative:
    """One creative with its predicted click-through rate and band."""

    creative: AdCreative
    predicted_ctr: float
    low: float
    high: float
    lift_vs_draft: float = 0.0
    rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        row = {
            "headline": self.creative.headline,
            "body": self.creative.body,
            "cta_text": self.creative.cta_text,
            "author": self.creative.author,
            "word_count": self.creative.word_count,
            "has_numeric_claim": self.creative.has_numeric_claim,
            "predicted_ctr": round(self.predicted_ctr, 5),
            "low": round(self.low, 5),
            "high": round(self.high, 5),
            "lift_vs_draft": round(self.lift_vs_draft, 5),
            "rank": self.rank,
        }
        return row


@dataclass(slots=True)
class RevisionResult:
    """A draft, its rewrites, and how each is expected to perform."""

    draft: ScoredCreative
    variants: list[ScoredCreative]
    constraints: RewriteConstraints
    model_backend: str
    llm_backend: str
    fit_seconds: float
    score_seconds: float
    notes: list[str] = field(default_factory=list)

    @property
    def best(self) -> ScoredCreative:
        return self.variants[0] if self.variants else self.draft

    @property
    def improved(self) -> bool:
        return bool(self.variants) and self.best.predicted_ctr > self.draft.predicted_ctr

    def to_dict(self) -> dict[str, Any]:
        return {
            "draft": self.draft.to_dict(),
            "variants": [v.to_dict() for v in self.variants],
            "best": self.best.to_dict(),
            "improved": self.improved,
            "constraints": asdict(self.constraints),
            "model_backend": self.model_backend,
            "llm_backend": self.llm_backend,
            "fit_seconds": round(self.fit_seconds, 3),
            "score_seconds": round(self.score_seconds, 3),
            "notes": self.notes,
        }


def constraints_from_history(history: pd.DataFrame) -> RewriteConstraints:
    """Read rewrite guard-rails off what has actually worked for this account.

    Looks at the top quartile of creatives by click-through rate and copies
    their shape: how long they run, whether they carry a number, which tone
    they use. This is deliberately simple. It is not the causal estimate; it
    is a description of the winners, which is what a copywriter would look at
    before starting.
    """
    if "ctr" not in history.columns or history["ctr"].isna().all():
        return RewriteConstraints()

    top = history[history["ctr"] >= history["ctr"].quantile(0.75)]
    if top.empty:
        return RewriteConstraints()

    max_words = int(np.clip(np.percentile(top["word_count"], 75), 6, 30))

    # Keep the number if creatives that carry one do better on average.
    by_number = history.groupby("has_numeric_claim")["ctr"].mean()
    keep_number = bool(by_number.get(1, 0.0) >= by_number.get(0, 0.0))

    # Prefer the tone that wins, but only where there is enough of it to say so.
    tone = "direct"
    if "tone" in history.columns:
        counts = history["tone"].value_counts()
        eligible = [t for t, n in counts.items() if n >= 20 and t != "unknown"]
        if eligible:
            tone_ctr = history[history["tone"].isin(eligible)].groupby("tone")["ctr"].mean()
            tone = str(tone_ctr.idxmax())

    return RewriteConstraints(max_words=max_words, keep_numeric_claim=keep_number, tone=tone)


class CopyCoach:
    """Fit once on an advertiser's history, then score and rewrite on demand.

    Args:
        history: Canonical creative table with outcomes.
        model: Unfitted click-through model. Defaults to the auto backend.
        llm: Language-model backend for rewrites. Defaults to auto.
    """

    def __init__(
        self,
        history: pd.DataFrame,
        *,
        model: CTRModel | None = None,
        llm: CopyLLM | None = None,
    ) -> None:
        self.history = history.reset_index(drop=True)
        self.model = model or CTRModel()
        self.llm = llm or get_llm()
        self.fit_seconds = 0.0
        self._fitted = False

    def fit(self) -> CopyCoach:
        """Load the history as context. Seconds, not a training run."""
        started = time.perf_counter()
        self.model.fit(
            self.history[FEATURE_COLUMNS],
            self.history["ctr"].to_numpy(dtype=float),
            groups=self.history[GROUP_COLUMN],
        )
        self.fit_seconds = time.perf_counter() - started
        self._fitted = True
        return self

    def score(self, creatives: list[AdCreative]) -> list[ScoredCreative]:
        """Predict click-through rate for creatives that have never run."""
        if not self._fitted:
            self.fit()
        frame = creatives_to_frame(creatives)
        mean, low, high = self.model.predict_interval(frame[FEATURE_COLUMNS])
        return [
            ScoredCreative(creative=c, predicted_ctr=float(m), low=float(lo), high=float(hi))
            for c, m, lo, hi in zip(creatives, mean, low, high, strict=True)
        ]

    def revise(
        self,
        draft: AdCreative,
        *,
        n_variants: int = 6,
        constraints: RewriteConstraints | None = None,
        variant_author: str = "llm",
    ) -> RevisionResult:
        """Rewrite a draft under data-derived constraints and rank the results.

        Args:
            draft: The creative to improve. Its placement fields are copied
                onto every variant so the comparison is like for like.
            n_variants: How many rewrites to request.
            constraints: Guard-rails for the rewrite. Defaults to whatever the
                account's own winners look like.
            variant_author: Author label stamped on the variants. They were
                written by a model, so ``llm`` is the honest default.
        """
        if not self._fitted:
            self.fit()
        constraints = constraints or constraints_from_history(self.history)

        started = time.perf_counter()
        proposals = self.llm.generate_variants(draft, n_variants, constraints)
        variants = [_variant_from(draft, proposal, author=variant_author) for proposal in proposals]

        scored = self.score([draft, *variants])
        score_seconds = time.perf_counter() - started

        scored_draft, scored_variants = scored[0], scored[1:]
        for item in scored_variants:
            item.lift_vs_draft = item.predicted_ctr - scored_draft.predicted_ctr
        scored_variants.sort(key=lambda s: s.predicted_ctr, reverse=True)
        for rank, item in enumerate(scored_variants, start=1):
            item.rank = rank

        notes = []
        if self.model.backend != "client":
            notes.append(
                f"Scored with the '{self.model.backend}' backend; the uncertainty band is "
                "approximate. Hosted TabPFN returns a real predictive distribution."
            )
        if getattr(self.llm, "name", "") == "stub":
            notes.append("Variants came from the offline stub, not a language model.")

        return RevisionResult(
            draft=scored_draft,
            variants=scored_variants,
            constraints=constraints,
            model_backend=self.model.backend,
            llm_backend=getattr(self.llm, "name", "unknown"),
            fit_seconds=self.fit_seconds,
            score_seconds=score_seconds,
            notes=notes,
        )


def _variant_from(draft: AdCreative, proposal: dict[str, str], *, author: str) -> AdCreative:
    """Build a variant that inherits everything about the draft except its words."""
    text = " ".join(
        part
        for part in (
            proposal.get("headline", ""),
            proposal.get("body", ""),
            proposal.get("cta_text", ""),
        )
        if part
    )
    return AdCreative(
        headline=proposal.get("headline", "").strip(),
        body=proposal.get("body", "").strip(),
        cta_text=proposal.get("cta_text", "").strip(),
        campaign_id=draft.campaign_id,
        author=author,  # type: ignore[arg-type]
        claim_type=draft.claim_type,
        tone=draft.tone,
        has_numeric_claim=int(any(ch.isdigit() for ch in text)),
        has_brand_logo=draft.has_brand_logo,
        has_human_face=draft.has_human_face,
        background_kind=draft.background_kind,
        dominant_color=draft.dominant_color,
        text_contrast=draft.text_contrast,
        vertical=draft.vertical,
        placement=draft.placement,
        device=draft.device,
        audience=draft.audience,
        daily_budget_usd=draft.daily_budget_usd,
        week_index=draft.week_index,
    )
