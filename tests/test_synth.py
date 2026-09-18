import numpy as np

from adlift.datasets.synth import make_ad_account, true_ate
from adlift.schema import ALL_COLUMNS, validate


def test_shape_and_schema(account):
    validate(account)
    assert set(ALL_COLUMNS) <= set(account.columns)
    assert account.campaign_id.nunique() == 60
    assert account.ctr.between(0, 1).all()


def test_ground_truth_columns_exist(account):
    for column in (
        "_truth_ctr_human",
        "_truth_ctr_llm",
        "_truth_total_effect",
        "_truth_direct_effect",
    ):
        assert column in account.columns
    np.testing.assert_allclose(
        account["_truth_total_effect"], account["_truth_ctr_llm"] - account["_truth_ctr_human"]
    )


def test_total_and_direct_effects_differ(account):
    # The point of the generator: the copy changes with the author, so the
    # total effect is not the planted direct effect.
    assert abs(true_ate(account, "total") - true_ate(account, "direct")) > 0.002


def test_confounding_is_present(account):
    # LLM adoption climbs over time.
    early = account[account.week_index < 8].author.eq("llm").mean()
    late = account[account.week_index > 18].author.eq("llm").mean()
    assert late > early + 0.2


def test_deterministic():
    a = make_ad_account(n_campaigns=10, seed=11)
    b = make_ad_account(n_campaigns=10, seed=11)
    assert a.equals(b)


def test_both_authors_in_some_campaign(account):
    mixed = account.groupby("campaign_id").author.nunique()
    assert (mixed > 1).sum() > 10
