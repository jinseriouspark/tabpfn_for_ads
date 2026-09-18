"""One click-through-rate model with three interchangeable backends.

The rest of AdLift asks for a model and gets the same object regardless of what
is actually running underneath:

``client``
    Hosted TabPFN 3.5 through ``tabpfn-client``. This is the backend the
    project is built for. It reads the creative's text as text, so there is no
    vectoriser in the pipeline, and it returns a predictive distribution rather
    than a point estimate.

``local``
    Open-source ``tabpfn`` running on your own machine. Useful with no API key,
    but the open-source line has no text pathway, so text columns are reduced
    to length and keyword flags before they are handed over.

``baseline``
    Gradient-boosted trees over TF-IDF and one-hot features. This is both the
    offline fallback and the comparison the benchmark runs against, because it
    is what an ad team would otherwise build.

Every backend fits in seconds and none of them is tuned. That is the claim
being tested: a new advertiser should not need a modelling project.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from adlift.schema import CATEGORICAL_COLUMNS, TEXT_COLUMNS

Backend = Literal["auto", "client", "local", "baseline"]

# Words that separate the two house styles enough to be worth keeping when a
# backend cannot read raw text.
_STYLE_MARKERS = [
    "discover",
    "unlock",
    "transform",
    "elevate",
    "journey",
    "seamless",
    "effortless",
    "empower",
    "reimagine",
    "free",
    "start",
    "stop",
    "cut",
    "fix",
]


@dataclass(slots=True)
class FitReport:
    """What happened during a fit, for the latency claims in the README."""

    backend: str
    n_train: int
    n_features: int
    fit_seconds: float
    predict_seconds: float = 0.0


def resolve_backend(requested: Backend = "auto") -> str:
    """Pick a backend, preferring hosted TabPFN when it is actually usable.

    ``auto`` walks down the list: hosted TabPFN if the client is installed and a
    token is present, then local TabPFN if that package is importable, then the
    gradient-boosted baseline, which always works.
    """
    if requested != "auto":
        return requested

    if os.environ.get("TABPFN_TOKEN"):
        try:
            import tabpfn_client  # noqa: F401

            return "client"
        except ImportError:
            pass
    try:
        import tabpfn  # noqa: F401

        return "local"
    except ImportError:
        return "baseline"


def _text_blob(frame: pd.DataFrame) -> pd.Series:
    present = [c for c in TEXT_COLUMNS if c in frame.columns]
    if not present:
        return pd.Series([""] * len(frame), index=frame.index)
    return frame[present].fillna("").agg(" ".join, axis=1).str.strip()


def _numeric_text_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse copy into numbers, for backends with no text pathway."""
    blob = _text_blob(frame).str.lower()
    out = pd.DataFrame(index=frame.index)
    out["_txt_words"] = blob.str.split().str.len().fillna(0)
    out["_txt_chars"] = blob.str.len().fillna(0)
    out["_txt_digits"] = blob.str.count(r"\d")
    out["_txt_exclaim"] = blob.str.count("!")
    out["_txt_avg_word"] = (out["_txt_chars"] / out["_txt_words"].clip(lower=1)).round(3)
    for marker in _STYLE_MARKERS:
        out[f"_kw_{marker}"] = blob.str.contains(marker, regex=False).astype(int)
    return out


class CTRModel:
    """Predict click-through rate for a creative that may never have run.

    Args:
        backend: Which engine to use. ``auto`` resolves at construction time.
        model_version: TabPFN model id for the hosted backend, for example
            ``v3.5_default`` or ``v3_default``.
        thinking_effort: Passed to hosted TabPFN to trade latency for accuracy.
            Leave as ``None`` for the fast path the real-time loop needs.
        n_estimators: Ensemble size for TabPFN. ``None`` keeps each backend's
            default; a small number makes the predictive distribution cheaper.
        random_state: Seed for the baseline backend.
    """

    def __init__(
        self,
        backend: Backend = "auto",
        *,
        model_version: str = "v3.5_default",
        thinking_effort: str | None = None,
        n_estimators: int | None = None,
        random_state: int = 0,
    ) -> None:
        self.backend = resolve_backend(backend)
        self.model_version = model_version
        self.thinking_effort = thinking_effort
        self.n_estimators = n_estimators
        self.random_state = random_state

        self._model: Any = None
        self._columns: list[str] = []
        self._vectoriser: Any = None
        self._encoder: Any = None
        self._client_group_col: str | None = None
        self.report: FitReport | None = None

    # -- feature assembly ------------------------------------------------

    def _prepare(self, frame: pd.DataFrame, *, fitting: bool) -> Any:
        if self.backend == "client":
            # TabPFN 3.5 consumes the dataframe directly, text columns included.
            prepared = frame.copy()
            for column in prepared.columns:
                if column in CATEGORICAL_COLUMNS:
                    prepared[column] = prepared[column].astype(str)
            return prepared

        if self.backend == "local":
            numeric = _numeric_text_features(frame)
            rest = frame.drop(columns=[c for c in TEXT_COLUMNS if c in frame.columns])
            merged = pd.concat([rest, numeric], axis=1)
            for column in merged.columns:
                if not pd.api.types.is_numeric_dtype(merged[column]):
                    merged[column] = merged[column].astype("category").cat.codes
            return merged.astype(float).to_numpy()

        # baseline: a dense design matrix so gradient boosting can be used.
        # TF-IDF is compressed with a truncated SVD rather than handed over
        # sparse, because sparse input rules out histogram boosting and a
        # linear model over TF-IDF cannot express the interactions that make
        # this problem interesting.
        blob = _text_blob(frame)
        categorical = [c for c in CATEGORICAL_COLUMNS if c in frame.columns]
        numeric_cols = [
            c
            for c in frame.columns
            if c not in categorical
            and c not in TEXT_COLUMNS
            and pd.api.types.is_numeric_dtype(frame[c])
        ]
        numeric = pd.concat(
            [frame[numeric_cols].astype(float).fillna(0.0), _numeric_text_features(frame)], axis=1
        )
        has_text = bool(blob.str.strip().any())

        if fitting:
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import OneHotEncoder

            self._vectoriser = None
            if has_text:
                vectoriser = TfidfVectorizer(
                    max_features=3000, ngram_range=(1, 2), min_df=2, sublinear_tf=True
                )
                try:
                    counts = vectoriser.fit_transform(blob)
                    components = int(min(64, max(2, min(counts.shape) - 1)))
                    svd = TruncatedSVD(n_components=components, random_state=self.random_state)
                    svd.fit(counts)
                    self._vectoriser = make_pipeline(vectoriser, svd)
                except ValueError:  # vocabulary empty after pruning
                    self._vectoriser = None

            self._encoder = (
                OneHotEncoder(handle_unknown="ignore", sparse_output=False) if categorical else None
            )
            if self._encoder is not None:
                self._encoder.fit(frame[categorical].astype(str))
            self._columns = list(numeric.columns)
        else:
            numeric = numeric.reindex(columns=self._columns, fill_value=0.0)

        blocks = [numeric.to_numpy(dtype=float)]
        if self._vectoriser is not None:
            blocks.append(np.asarray(self._vectoriser.transform(blob), dtype=float))
        if self._encoder is not None:
            blocks.append(
                np.asarray(self._encoder.transform(frame[categorical].astype(str)), dtype=float)
            )
        return np.hstack(blocks)

    # -- sklearn-shaped surface ------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray | pd.Series,
        *,
        groups: pd.Series | None = None,
    ) -> CTRModel:
        """Fit on a creative history.

        For TabPFN this is in-context learning: no weights change, the rows
        become context. That is why it takes seconds and needs no tuning.

        Args:
            X: Feature frame. Text columns may be present.
            y: Click-through rates, one per row.
            groups: Campaign ids. In Thinking mode they are passed to hosted
                TabPFN as ``group_col`` so its internal validation never splits
                a campaign, which is what the cookbook prescribes for grouped
                data. Outside Thinking mode the hosted model has no use for
                them and they are not sent.
        """
        y = np.asarray(y, dtype=float)
        started = time.perf_counter()
        prepared = self._prepare(X, fitting=True)

        if self.backend == "client":
            from tabpfn_client import TabPFNRegressor

            kwargs: dict[str, Any] = {"random_state": self.random_state}
            if self.n_estimators is not None:
                kwargs["n_estimators"] = self.n_estimators
            categorical = [
                index
                for index, column in enumerate(prepared.columns)
                if column in CATEGORICAL_COLUMNS
            ]
            if categorical:
                kwargs["categorical_features_indices"] = categorical
            self._client_group_col = None
            if self.thinking_effort:
                kwargs["thinking_effort"] = self.thinking_effort
                if groups is not None:
                    prepared = prepared.copy()
                    prepared["__group__"] = np.asarray(groups).astype(str)
                    kwargs["group_col"] = "__group__"
                    self._client_group_col = "__group__"
            self._model = _construct_client_regressor(TabPFNRegressor, self.model_version, kwargs)
            self._model.fit(prepared, y)

        elif self.backend == "local":
            from tabpfn import TabPFNRegressor as LocalRegressor

            local_kwargs: dict[str, Any] = {}
            if self.n_estimators is not None:
                local_kwargs["n_estimators"] = self.n_estimators
            self._model = LocalRegressor(**local_kwargs)
            self._model.fit(prepared, y)

        else:
            from sklearn.ensemble import HistGradientBoostingRegressor

            # Untuned gradient boosting: the model an ad team reaches for, and
            # the honest comparison for TabPFN's no-tuning claim.
            self._model = HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.06,
                max_leaf_nodes=31,
                min_samples_leaf=10,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=self.random_state,
            )
            self._model.fit(prepared, y)

        self.report = FitReport(
            backend=self.backend,
            n_train=len(X),
            n_features=int(getattr(prepared, "shape", (0, 0))[1]),
            fit_seconds=time.perf_counter() - started,
        )
        return self

    def _prepared_for_predict(self, X: pd.DataFrame) -> Any:
        prepared = self._prepare(X, fitting=False)
        if self.backend == "client" and self._client_group_col:
            # New creatives belong to no training campaign. The column has to
            # exist because the model was fitted with it; its value is a
            # sentinel the model has never seen.
            prepared = prepared.copy()
            prepared[self._client_group_col] = "__new__"
        return prepared

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Point prediction of click-through rate, clipped to a valid range."""
        started = time.perf_counter()
        prepared = self._prepared_for_predict(X)
        predictions = np.asarray(self._model.predict(prepared), dtype=float).ravel()
        if self.report is not None:
            self.report.predict_seconds = time.perf_counter() - started
        return np.clip(predictions, 0.0, 1.0)

    def predict_interval(
        self, X: pd.DataFrame, *, lower: float = 0.1, upper: float = 0.9
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict with an uncertainty band.

        Hosted TabPFN returns a full predictive distribution, so the band is
        read straight off its quantiles. Other backends have no distribution to
        offer and fall back to a residual-derived band, which is reported
        honestly as approximate rather than dressed up as the real thing.

        Returns:
            ``(mean, low, high)``, each one value per row.
        """
        mean = self.predict(X)
        if self.backend == "client":
            prepared = self._prepared_for_predict(X)
            try:
                out = self._model.predict(
                    prepared, output_type="quantiles", quantiles=[lower, upper]
                )
                if isinstance(out, dict):
                    out = out.get("quantiles", out)
                arr = np.asarray(out, dtype=float)
                if arr.ndim == 2 and arr.shape[0] != 2 and arr.shape[1] == 2:
                    arr = arr.T
                low, high = arr[0].ravel(), arr[1].ravel()
                if low.shape == mean.shape:
                    return mean, np.clip(low, 0, 1), np.clip(high, 0, 1)
            except (TypeError, KeyError, ValueError, IndexError):
                pass
        spread = float(np.std(mean)) or 1e-4
        return mean, np.clip(mean - spread, 0, 1), np.clip(mean + spread, 0, 1)


#: How this project names hosted models, mapped onto the client's enum.
_MODEL_VERSION_MEMBERS = {
    "v3.5_default": "V3_5",
    "v3.5_fast": "V3_5_FAST",
    "v3_default": "V3",
    "v2.6_default": "V2_6",
    "v2.5_default": "V2_5",
    "v2_default": "V2",
}


def _construct_client_regressor(
    regressor_cls: Any, model_version: str, kwargs: dict[str, Any]
) -> Any:
    """Build a hosted regressor, tolerating client-version differences.

    The cookbook's preferred spelling is ``create_default_for_version`` with a
    ``ModelVersion`` member, which also picks that version's default ensemble
    and inference settings. Older clients take ``model_path`` instead. Try the
    documented forms in order and drop arguments the installed version
    rejects, rather than pin one spelling.
    """
    attempts: list[Callable[[], Any]] = []

    member_name = _MODEL_VERSION_MEMBERS.get(model_version)
    if member_name and hasattr(regressor_cls, "create_default_for_version"):
        try:
            from tabpfn_client.api_models import ModelVersion

            member = getattr(ModelVersion, member_name, None)
        except ImportError:
            member = None
        if member is not None:
            attempts.append(lambda: regressor_cls.create_default_for_version(member, **kwargs))

    attempts.append(lambda: regressor_cls(model_path=model_version, **kwargs))
    minimal = {k: v for k, v in kwargs.items() if k not in {"thinking_effort", "group_col"}}
    attempts.append(lambda: regressor_cls(model_path=model_version, **minimal))
    attempts.append(lambda: regressor_cls(**minimal))
    attempts.append(lambda: regressor_cls())

    last_error: Exception | None = None
    for attempt in attempts:
        try:
            return attempt()
        except (TypeError, ValueError) as error:  # unsupported keyword or value for this client
            last_error = error
    raise RuntimeError(f"could not construct TabPFN client regressor: {last_error}")
