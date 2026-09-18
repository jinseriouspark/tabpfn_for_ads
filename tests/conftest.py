import pytest

from adlift.datasets.synth import make_ad_account


@pytest.fixture(scope="session")
def account():
    """A small synthetic account shared across tests."""
    return make_ad_account(n_campaigns=60, creatives_per_campaign=(4, 8), seed=3)


@pytest.fixture
def baseline_factory():
    from adlift.model import CTRModel

    return lambda: CTRModel(backend="baseline")
