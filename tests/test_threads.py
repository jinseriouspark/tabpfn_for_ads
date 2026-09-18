import json

import numpy as np
import pandas as pd
import pytest

from adlift.datasets.synth_threads import make_threads_account, true_ate, true_log_ate
from adlift.schema import THREADS_SPEC, get_spec, validate


@pytest.fixture(scope="module")
def timeline():
    return make_threads_account(n_weeks=14, posts_per_week=(9, 13), seed=3)


def test_synth_timeline_shape(timeline):
    validate(timeline, spec=THREADS_SPEC)
    assert set(THREADS_SPEC.all_columns) <= set(timeline.columns)
    assert timeline["week"].nunique() == 14
    assert (timeline["views"] >= 1).all()
    assert timeline["engagement_rate"].between(0, 1).all()


def test_synth_planted_effects(timeline):
    # The LLM house style is longer, tags more and links more; total effect < direct effect.
    assert true_log_ate(timeline, "total") < true_log_ate(timeline, "direct") < 0
    assert abs(true_ate(timeline, "total")) > 0


def test_get_spec_outcomes():
    assert get_spec("threads").outcome.column == "views"
    assert get_spec("threads", outcome="engagement").outcome.column == "engagement_rate"
    with pytest.raises(ValueError):
        get_spec("threads", outcome="nope")


def test_posts_to_frame_from_api_shape():
    from adlift.ingest.threads import posts_to_frame

    posts = [
        {
            "id": "1",
            "text": "Cut churn 30%? Yes. #saas",
            "timestamp": "2026-09-01T21:15:00+0000",
            "media_type": "IMAGE",
            "views": 1200,
            "likes": 40,
            "replies": 3,
            "reposts": 2,
            "quotes": 0,
        },
        {
            "id": "2",
            "text": "Excited to share our journey!",
            "timestamp": "2026-09-08T09:00:00+0000",
            "media_type": "TEXT_POST",
            "views": 300,
            "likes": 5,
            "replies": 0,
            "reposts": 0,
            "quotes": 0,
            "author": "llm",
        },
    ]
    frame = posts_to_frame(posts, tz="Asia/Seoul")
    assert list(frame["post_id"]) == ["1", "2"]
    assert frame.loc[0, "posted_hour"] == 6  # 21:15 UTC is 06:15 in Seoul
    assert frame.loc[0, "has_media"] == 1 and frame.loc[1, "has_media"] == 0
    assert frame.loc[0, "n_hashtags"] == 1 and frame.loc[0, "has_question"] == 1
    assert frame.loc[1, "author"] == "llm" and frame.loc[0, "author"] == "human"
    assert frame.loc[0, "week"].startswith("2026-W")
    assert frame.loc[1, "week_index"] == 1
    assert frame.loc[0, "engagement_rate"] == pytest.approx(45 / 1200)


def test_csv_roundtrip_and_label_template(tmp_path, timeline):
    from adlift.ingest.threads import threads_frame_from_csv, write_label_template

    path = tmp_path / "posts.csv"
    timeline[
        [
            "post_id",
            "text",
            "timestamp",
            "views",
            "likes",
            "replies",
            "reposts",
            "quotes",
            "has_media",
            "is_reply",
            "topic",
            "author",
        ]
    ].to_csv(path, index=False)
    frame = threads_frame_from_csv(path)
    validate(frame, spec=THREADS_SPEC)
    assert len(frame) == len(timeline)
    template = write_label_template(frame, tmp_path / "labels.csv")
    labels = pd.read_csv(template)
    assert {"post_id", "author", "topic", "text", "views"} <= set(labels.columns)


def test_model_log_scale_roundtrip(timeline):
    from adlift.model import CTRModel

    model = CTRModel(backend="baseline", spec=THREADS_SPEC).fit(
        timeline[THREADS_SPEC.feature_columns], timeline["views"]
    )
    natural = model.predict(timeline[THREADS_SPEC.feature_columns].head(5))
    on_model = model.predict(timeline[THREADS_SPEC.feature_columns].head(5), scale="model")
    np.testing.assert_allclose(np.expm1(on_model), natural, rtol=1e-6)
    assert (natural >= 0).all()


def test_temporal_cold_start_beats_random(timeline):
    from adlift.coldstart import evaluate_cold_start, random_baseline
    from adlift.model import CTRModel

    result = evaluate_cold_start(
        timeline, CTRModel(backend="baseline", spec=THREADS_SPEC), name="gbdt", n_splits=3
    )
    random = random_baseline(timeline, spec=THREADS_SPEC)
    assert result.split == "temporal"
    assert result.spearman > random.spearman
    assert result.top1_regret < random.top1_regret


def test_learning_curve_shape(timeline):
    from adlift.coldstart import learning_curve
    from adlift.model import CTRModel

    curve = learning_curve(
        timeline, CTRModel(backend="baseline", spec=THREADS_SPEC), start=10, step=20
    )
    assert list(curve.columns) == [
        "n_context",
        "block_size",
        "spearman",
        "top1_regret",
        "mae",
        "fit_seconds",
    ]
    assert curve["n_context"].iloc[0] == 10
    assert len(curve) >= 3


def test_causal_recovers_sign_on_log_outcome(timeline):
    from adlift.causal import full_analysis
    from adlift.model import CTRModel

    out = full_analysis(
        timeline,
        model_factory=lambda: CTRModel(backend="baseline", spec=THREADS_SPEC),
        spec=THREADS_SPEC,
        n_boot=40,
    )
    # Heavy-tailed views: the headline is the doubly-robust effect on the log
    # scale, read as a ratio. The natural-scale mean is outlier-driven and is
    # not what the sign test should look at.
    total = out["total_effect"]
    assert total.estimator == "t-learner"
    assert total.ratio is not None
    assert np.sign(total.ratio) == np.sign(true_log_ate(timeline, "total"))
    assert "s_learner" in out and "median_effect" in out
    assert out["detectable_effect"] > 0
    assert set(out["segments"].columns) == {"topic", "effect", "sd", "n"}


def test_lever_analysis_signs(timeline):
    from adlift.levers import lever_analysis
    from adlift.model import CTRModel

    table = lever_analysis(
        timeline,
        spec=THREADS_SPEC,
        model_factory=lambda: CTRModel(backend="baseline", spec=THREADS_SPEC),
        n_boot=30,
    )
    by = table.set_index(["column", "to"])
    assert by.loc[("has_link", 1), "effect"] < 0
    assert by.loc[("has_media", 1), "effect"] > 0
    assert by.loc[("is_reply", 1), "effect"] < 0
    assert {
        "lever",
        "from",
        "to",
        "effect",
        "ci_low",
        "ci_high",
        "ratio",
        "support",
        "significant",
    } <= set(table.columns)


def test_suggest_for_post(timeline):
    from adlift.levers import suggest_for_post
    from adlift.loop import PostDraft
    from adlift.model import CTRModel

    row = PostDraft(
        text="Excited to share our journey! https://x.co #a #b #c",
        author="llm",
        context={
            "posted_hour": 3,
            "weekday": "Sat",
            "has_media": 0,
            "is_reply": 1,
            "topic": "product_launch",
            "week_index": 13,
        },
    ).to_row(THREADS_SPEC)
    table = suggest_for_post(
        timeline,
        row,
        spec=THREADS_SPEC,
        model_factory=lambda: CTRModel(backend="baseline", spec=THREADS_SPEC),
        top=5,
    )
    assert len(table) == 5
    assert table["lift"].iloc[0] >= table["lift"].iloc[-1]
    assert table.attrs["as_is"] > 0


def test_post_revise_and_precedents(timeline):
    from adlift.llm import StubCopyLLM
    from adlift.loop import CopyCoach, PostDraft
    from adlift.model import CTRModel

    coach = CopyCoach(
        timeline,
        spec=THREADS_SPEC,
        model=CTRModel(backend="baseline", spec=THREADS_SPEC),
        llm=StubCopyLLM(),
    ).fit()
    draft = PostDraft(
        text="Tried Atlas for 12 days. Verdict: worth it.",
        context={"posted_hour": 20, "week_index": 13},
    )
    result = coach.revise(draft, n_variants=3)
    assert len(result.variants) == 3 and result.constraints.medium == "post"
    assert all(isinstance(v.creative, PostDraft) for v in result.variants)
    similar = coach.precedents(draft, k=4)
    assert len(similar) == 4 and "similarity" in similar.columns


def test_cli_threads_demo_and_commands(tmp_path):
    from typer.testing import CliRunner

    from adlift.cli import app

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "demo",
            "--spec",
            "threads",
            "--out",
            str(tmp_path),
            "--backend",
            "baseline",
            "--n-boot",
            "20",
            "--n-splits",
            "3",
            "--n-groups",
            "10",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "levers.png").exists() and (tmp_path / "learning_curve.png").exists()
    payload = json.loads((tmp_path / "results.json").read_text())
    assert payload["spec"] == "threads" and "levers" in payload

    result = runner.invoke(
        app,
        [
            "threads",
            "score",
            "--text",
            "Cut churn by 30%. Here's how.",
            "--backend",
            "baseline",
            "--n-groups",
            "10",
        ]
        if False
        else [
            "threads",
            "score",
            "--text",
            "Cut churn by 30%. Here's how.",
            "--backend",
            "baseline",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "predicted" in result.output

    result = runner.invoke(
        app, ["threads", "suggest", "--text", "Excited to share! #a #b #c", "--backend", "baseline"]
    )
    assert result.exit_code == 0, result.output


def test_mcp_threads_tools():
    import os

    os.environ["ADLIFT_BACKEND"] = "baseline"
    from mcp_server import server as srv

    srv.STATE.backend = "baseline"
    out = json.loads(srv.load_threads(n_weeks=8, seed=2))
    assert out["spec"] == "threads" and out["rows"] > 0
    scored = json.loads(srv.score_post(text="Cut churn by 30%.", n_precedents=2))
    assert scored["predicted"] > 0 and len(scored["precedents"]) == 2
    suggestions = json.loads(srv.suggest_post(text="Excited to share our journey!", top=3))
    assert len(suggestions["suggestions"]) == 3
    levers = json.loads(srv.lever_report(n_boot=10))
    assert levers["outcome"] == "views" and len(levers["levers"]) > 5
