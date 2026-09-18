import numpy as np

from adlift.causal import (
    full_analysis,
    naive_difference,
    overlap_diagnostic,
    placebo_test,
    s_learner_ate,
    t_learner_ate,
)
from adlift.datasets.synth import true_ate


def test_s_learner_recovers_sign_and_magnitude(account, baseline_factory):
    truth = true_ate(account, "total")
    result = s_learner_ate(account, model_factory=baseline_factory, n_boot=100)
    assert np.sign(result.ate) == np.sign(truth)
    assert abs(result.ate - truth) < 0.4 * abs(truth) + 0.001
    assert result.ci_low <= result.ate <= result.ci_high
    assert len(result.per_row_effect) == len(account)


def test_naive_is_worse_than_adjusted(account, baseline_factory):
    truth = true_ate(account, "total")
    naive = naive_difference(account)
    adjusted = s_learner_ate(account, model_factory=baseline_factory, n_boot=20).ate
    assert abs(adjusted - truth) < abs(naive - truth)


def test_t_learner_agrees(account, baseline_factory):
    s = s_learner_ate(account, model_factory=baseline_factory, n_boot=20)
    t = t_learner_ate(account, model_factory=baseline_factory, n_boot=20)
    assert np.sign(s.ate) == np.sign(t.ate)


def test_placebo_collapses(account, baseline_factory):
    out = placebo_test(account, model_factory=baseline_factory, n_rounds=2)
    assert out["ratio"] > 3


def test_overlap_diagnostic_keys(account):
    out = overlap_diagnostic(account)
    assert 0.5 <= out["auc"] <= 1.0
    assert 0 <= out["share_off_support"] <= 1


def test_full_analysis_shape(account, baseline_factory):
    out = full_analysis(account, model_factory=baseline_factory, n_boot=20)
    for key in (
        "naive_difference",
        "total_effect",
        "direct_effect",
        "cross_check",
        "mediation",
        "segments",
        "overlap",
        "placebo",
        "estimators_agree",
    ):
        assert key in out
    assert set(out["segments"].columns) == {"device", "effect", "sd", "n"}
