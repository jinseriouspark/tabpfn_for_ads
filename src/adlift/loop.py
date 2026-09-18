"""The loop: a language model proposes copy, TabPFN scores it, instantly.

This is where TabPFN's two properties stop being benchmark numbers and become
a product. Because fitting is in-context, the account's history is loaded once
and every subsequent creative, human-drafted or model-rewritten, is a single
predict call. There is no retraining step between "write a variant" and "know
how it will do", so the whole cycle runs at conversational speed.

The rewrite is *constrained*. The causal analysis in :mod:`adlift.causal`
says what LLM copy tends to do that costs performance -- run long, drop the
number, reach for aspirational verbs, pile on hashtags. Those findings become
guard-rails on the prompt. The model is asked to polish wording inside limits
the data set, which is what "a human revising with AI help" should mean: the
person keeps the facts and the shape, the model keeps the phrasing sharp.

Two kinds of draft go through the same loop: an ``AdCreative`` (headline, body,
call to action) and a ``PostDraft`` (one text field plus posting context).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from adlift.llm import CopyLLM, RewriteConstraints, get_llm
from adlift.model import CTRModel
from adlift.schema import AD_SPEC, AdCreative, DatasetSpec, creatives_to_frame
from adlift.text_features import text_features


@dataclass(slots=True)
class PostDraft:
    """One social post: a single text field plus where and when it will run.

    ``context`` holds the spec's context columns (``posted_hour``, ``weekday``,
    ``has_media``, ``is_reply``, ``topic``, ``week_index``). Anything missing
    falls back to a neutral value. Mediator columns are derived from the text.
    """

    text: str
    author: str = "human"
    group: str = "new"
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def headline(self) -> str:  # lets the LLM rewrite prompt treat it like a creative
        return self.text

    @property
    def body(self) -> str:
        return ""

    @property
    def cta_text(self) -> str:
        return ""

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def has_numeric_claim(self) -> int:
        return int(any(ch.isdigit() for ch in self.text))

    def to_row(self, spec: DatasetSpec) -> dict[str, Any]:
        row: dict[str, Any] = {spec.id_column: "", spec.group_column: self.group}
        row[spec.text_columns[0]] = self.text
        row.update(text_features(self.text))
        defaults = {
            "posted_hour": 12,
            "weekday": "Tue",
            "has_media": 0,
            "is_reply": 0,
            "topic": "unknown",
            "week_index": 0,
        }
        for column in spec.context_columns:
            row[column] = self.context.get(column, defaults.get(column, 0))
        row[spec.treatment_column] = self.author
        row[spec.outcome.column] = np.nan
        return row


def _draft_headline(draft: Any) -> str:
    return getattr(draft, "headline", getattr(draft, "text", ""))


@dataclass(slots=True)
class ScoredCreative:
    """One creative with its predicted outcome and band."""

    creative: Any  # AdCreative or PostDraft
    predicted: float
    low: float
    high: float
    lift_vs_draft: float = 0.0
    rank: int = 0

    @property
    def predicted_ctr(self) -> float:  # backwards-compatible name
        return self.predicted

    def to_dict(self) -> dict[str, Any]:
        creative = self.creative
        return {
            "headline": _draft_headline(creative),
            "body": getattr(creative, "body", ""),
            "cta_text": getattr(creative, "cta_text", ""),
            "author": getattr(creative, "author", "human"),
            "word_count": getattr(creative, "word_count", len(_draft_headline(creative).split())),
            "has_numeric_claim": getattr(creative, "has_numeric_claim", 0),
            "predicted": round(self.predicted, 5),
            "predicted_ctr": round(self.predicted, 5),
            "low": round(self.low, 5),
            "high": round(self.high, 5),
            "lift_vs_draft": round(self.lift_vs_draft, 5),
            "rank": self.rank,
        }


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
    outcome_label: str = "outcome"
    notes: list[str] = field(default_factory=list)

    @property
    def best(self) -> ScoredCreative:
        return self.variants[0] if self.variants else self.draft

    @property
    def improved(self) -> bool:
        return bool(self.variants) and self.best.predicted > self.draft.predicted

    def to_dict(self) -> dict[str, Any]:
        return {
            "draft": self.draft.to_dict(),
            "variants": [v.to_dict() for v in self.variants],
            "best": self.best.to_dict(),
            "improved": self.improved,
            "constraints": asdict(self.constraints),
            "model_backend": self.model_backend,
            "llm_backend": self.llm_backend,
            "outcome": self.outcome_label,
            "fit_seconds": round(self.fit_seconds, 3),
            "score_seconds": round(self.score_seconds, 3),
            "notes": self.notes,
        }


def constraints_from_history(
    history: pd.DataFrame, spec: DatasetSpec = AD_SPEC
) -> RewriteConstraints:
    """Read rewrite guard-rails off what has actually worked for this account.

    Looks at the top quartile by outcome and copies its shape: how long the
    winners run, whether they carry a number, which tone they use, how many
    hashtags they allow. This is deliberately simple. It is not the causal
    estimate; it is a description of the winners, which is what a copywriter
    would look at before starting.
    """
    outcome = spec.outcome.column
    medium = "post" if spec.name == "threads" else "ad"
    if outcome not in history.columns or history[outcome].isna().all():
        return RewriteConstraints(medium=medium)

    top = history[history[outcome] >= history[outcome].quantile(0.75)]
    if top.empty:
        return RewriteConstraints(medium=medium)

    max_words = int(np.clip(np.percentile(top["word_count"], 75), 6, 60))

    keep_number = True
    if "has_numeric_claim" in history.columns:
        by_number = history.groupby("has_numeric_claim")[outcome].mean()
        keep_number = bool(by_number.get(1, 0.0) >= by_number.get(0, 0.0))

    tone = "direct"
    if "tone" in history.columns:
        counts = history["tone"].value_counts()
        eligible = [t for t, n in counts.items() if n >= 20 and t != "unknown"]
        if eligible:
            tone_outcome = history[history["tone"].isin(eligible)].groupby("tone")[outcome].mean()
            tone = str(tone_outcome.idxmax())

    max_hashtags = None
    if "n_hashtags" in top.columns:
        max_hashtags = int(np.percentile(top["n_hashtags"], 75))

    return RewriteConstraints(
        max_words=max_words,
        keep_numeric_claim=keep_number,
        tone=tone,
        max_hashtags=max_hashtags,
        medium=medium,
    )


class CopyCoach:
    """Fit once on an account's history, then score and rewrite on demand.

    Args:
        history: Canonical table with outcomes, in the spec's columns.
        spec: Column roles and outcome scale. Defaults to the ads spec.
        model: Unfitted model. Defaults to the auto backend with this spec.
        llm: Language-model backend for rewrites. Defaults to auto.
    """

    def __init__(
        self,
        history: pd.DataFrame,
        *,
        spec: DatasetSpec = AD_SPEC,
        model: CTRModel | None = None,
        llm: CopyLLM | None = None,
    ) -> None:
        self.spec = spec
        self.history = history.reset_index(drop=True)
        self.model = model or CTRModel(spec=spec)
        if self.model.spec is not spec:
            self.model.spec = spec
        self.llm = llm or get_llm()
        self.fit_seconds = 0.0
        self._fitted = False

    def fit(self) -> CopyCoach:
        """Load the history as context. Seconds, not a training run."""
        spec = self.spec
        started = time.perf_counter()
        self.model.fit(
            self.history[spec.feature_columns],
            self.history[spec.outcome.column].to_numpy(dtype=float),
            groups=self.history[spec.group_column],
        )
        self.fit_seconds = time.perf_counter() - started
        self._fitted = True
        return self

    # -- scoring -----------------------------------------------------------

    def score_frame(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict outcome and band for rows already in the spec's columns."""
        if not self._fitted:
            self.fit()
        return self.model.predict_interval(frame[self.spec.feature_columns])

    def score(self, creatives: list[Any]) -> list[ScoredCreative]:
        """Predict the outcome for creatives that have never run.

        Accepts ``AdCreative`` objects (ads spec) or ``PostDraft`` objects
        (single-text specs such as Threads).
        """
        frame = self._to_frame(creatives)
        mean, low, high = self.score_frame(frame)
        return [
            ScoredCreative(creative=c, predicted=float(m), low=float(lo), high=float(hi))
            for c, m, lo, hi in zip(creatives, mean, low, high, strict=True)
        ]

    def _to_frame(self, creatives: list[Any]) -> pd.DataFrame:
        if creatives and isinstance(creatives[0], PostDraft):
            return pd.DataFrame([c.to_row(self.spec) for c in creatives])
        return creatives_to_frame(list(creatives))

    # -- rewriting ---------------------------------------------------------

    def revise(
        self,
        draft: Any,
        *,
        n_variants: int = 6,
        constraints: RewriteConstraints | None = None,
        variant_author: str = "llm",
    ) -> RevisionResult:
        """Rewrite a draft under data-derived constraints and rank the results.

        Args:
            draft: The creative to improve, an ``AdCreative`` or a ``PostDraft``.
                Its context is copied onto every variant so the comparison is
                like for like.
            n_variants: How many rewrites to request.
            constraints: Guard-rails for the rewrite. Defaults to whatever the
                account's own winners look like.
            variant_author: Author label stamped on the variants. They were
                written by a model, so ``llm`` is the honest default.
        """
        if not self._fitted:
            self.fit()
        constraints = constraints or constraints_from_history(self.history, self.spec)

        started = time.perf_counter()
        proposals = self.llm.generate_variants(draft, n_variants, constraints)
        if isinstance(draft, PostDraft):
            variants: list[Any] = [
                _post_variant_from(draft, proposal, author=variant_author) for proposal in proposals
            ]
        else:
            variants = [
                _variant_from(draft, proposal, author=variant_author) for proposal in proposals
            ]

        scored = self.score([draft, *variants])
        score_seconds = time.perf_counter() - started

        scored_draft, scored_variants = scored[0], scored[1:]
        for item in scored_variants:
            item.lift_vs_draft = item.predicted - scored_draft.predicted
        scored_variants.sort(key=lambda s: s.predicted, reverse=True)
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
            outcome_label=self.spec.outcome.label,
            notes=notes,
        )


def _variant_from(draft: AdCreative, proposal: dict[str, str], *, author: str) -> AdCreative:
    """Build an ad variant that inherits everything about the draft except its words."""
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


def _post_variant_from(draft: PostDraft, proposal: dict[str, str], *, author: str) -> PostDraft:
    """Build a post variant: new text, same posting context."""
    parts = [proposal.get("headline", ""), proposal.get("body", "")]
    text = "\n".join(p.strip() for p in parts if p and p.strip())
    return PostDraft(text=text, author=author, group=draft.group, context=dict(draft.context))


# ---------------------------------------------------------------------------
# Why: the closest precedents in the account's own history
# ---------------------------------------------------------------------------


def precedents(
    history: pd.DataFrame,
    draft: Any,
    *,
    spec: DatasetSpec = AD_SPEC,
    k: int = 5,
) -> pd.DataFrame:
    """The past creatives a draft most resembles, with how they did.

    TabPFN's prediction is an attention-weighted read of its context rows,
    and the open-source model exposes that readout directly (see the
    "decoder readout" cookbook). The hosted API does not, so this is the
    honest stand-in: cosine similarity over the text plus the structured
    attributes, restricted to the same history the model was fitted on. It
    tells a writer "your draft looks like these five, and here is what they
    got", which is the question they were asking.

    Returns:
        Up to ``k`` rows with ``similarity``, the outcome, the author and a
        text snippet, most similar first.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler

    from adlift.model import _text_blob

    if isinstance(draft, PostDraft):
        draft_frame = pd.DataFrame([draft.to_row(spec)])
    else:
        draft_frame = creatives_to_frame([draft])

    texts = pd.concat([_text_blob(history, spec), _text_blob(draft_frame, spec)], ignore_index=True)
    vectoriser = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    text_matrix = vectoriser.fit_transform(texts)

    numeric_columns = [
        c
        for c in spec.creative_columns + tuple(spec.context_columns)
        if c in history.columns and pd.api.types.is_numeric_dtype(history[c])
    ]
    combined = pd.concat(
        [history[numeric_columns], draft_frame[numeric_columns]], ignore_index=True
    )
    scaler = StandardScaler()
    numeric_matrix = scaler.fit_transform(combined.astype(float).fillna(0.0))

    text_sim = cosine_similarity(text_matrix[-1], text_matrix[:-1]).ravel()
    numeric_sim = cosine_similarity(numeric_matrix[-1:], numeric_matrix[:-1]).ravel()
    similarity = 0.6 * text_sim + 0.4 * np.nan_to_num(numeric_sim)

    order = np.argsort(similarity)[::-1][:k]
    out = history.iloc[order][[spec.id_column, spec.treatment_column, spec.outcome.column]].copy()
    out.insert(1, "similarity", np.round(similarity[order], 3))
    out["text"] = _text_blob(history.iloc[order], spec).str.slice(0, 120).to_numpy()
    return out.reset_index(drop=True)


def _coach_precedents(self: CopyCoach, draft: Any, *, k: int = 5) -> pd.DataFrame:
    """See :func:`precedents`; uses the coach's own history and spec."""
    return precedents(self.history, draft, spec=self.spec, k=k)


CopyCoach.precedents = _coach_precedents  # type: ignore[attr-defined]
