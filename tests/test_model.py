import numpy as np

from adlift.model import CTRModel, resolve_backend
from adlift.schema import CONFOUNDER_COLUMNS, FEATURE_COLUMNS, TREATMENT_COLUMN


def test_resolve_backend_explicit():
    assert resolve_backend("baseline") == "baseline"


def test_fit_predict_with_text(account):
    model = CTRModel(backend="baseline").fit(account[FEATURE_COLUMNS], account.ctr)
    preds = model.predict(account[FEATURE_COLUMNS].head(10))
    assert preds.shape == (10,)
    assert np.all((preds >= 0) & (preds <= 1))
    assert model.report is not None and model.report.fit_seconds > 0


def test_fit_predict_without_text(account):
    columns = CONFOUNDER_COLUMNS + [TREATMENT_COLUMN]
    model = CTRModel(backend="baseline").fit(account[columns], account.ctr)
    preds = model.predict(account[columns].head(5))
    assert preds.shape == (5,)


def test_interval_wraps_mean(account):
    model = CTRModel(backend="baseline").fit(account[FEATURE_COLUMNS], account.ctr)
    mean, low, high = model.predict_interval(account[FEATURE_COLUMNS].head(8))
    assert np.all(low <= mean) and np.all(mean <= high)


def test_learns_signal(account):
    from scipy.stats import spearmanr

    train = account[account.campaign_id < "cmp_0045"]
    test = account[account.campaign_id >= "cmp_0045"]
    model = CTRModel(backend="baseline").fit(train[FEATURE_COLUMNS], train.ctr)
    rho = spearmanr(model.predict(test[FEATURE_COLUMNS]), test["_truth_ctr_noiseless"]).statistic
    assert rho > 0.4
