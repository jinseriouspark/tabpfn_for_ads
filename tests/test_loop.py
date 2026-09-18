from adlift.llm import RewriteConstraints, StubCopyLLM, get_llm
from adlift.loop import CopyCoach, constraints_from_history
from adlift.model import CTRModel
from adlift.schema import AdCreative


def _draft():
    return AdCreative(
        headline="Discover how Atlas can transform the way you handle ticket backlogs.",
        body="Join thousands of forward-thinking teams.",
        cta_text="Begin your journey",
        campaign_id="new",
        author="llm",
        vertical="saas",
        placement="social_feed",
        device="mobile",
        audience="prospecting",
        daily_budget_usd=400,
        week_index=20,
    )


def test_constraints_from_history(account):
    c = constraints_from_history(account)
    assert 6 <= c.max_words <= 30
    assert c.tone != "unknown"


def test_revise_ranks_variants(account):
    coach = CopyCoach(account, model=CTRModel(backend="baseline"), llm=StubCopyLLM()).fit()
    result = coach.revise(_draft(), n_variants=4)
    assert len(result.variants) == 4
    assert [v.rank for v in result.variants] == [1, 2, 3, 4]
    ctrs = [v.predicted_ctr for v in result.variants]
    assert ctrs == sorted(ctrs, reverse=True)
    assert result.best.lift_vs_draft == result.best.predicted_ctr - result.draft.predicted_ctr
    assert result.fit_seconds > 0 and result.score_seconds >= 0


def test_variants_inherit_context(account):
    coach = CopyCoach(account, model=CTRModel(backend="baseline"), llm=StubCopyLLM()).fit()
    result = coach.revise(_draft(), n_variants=2)
    for v in result.variants:
        assert v.creative.device == "mobile" and v.creative.author == "llm"
        assert v.creative.word_count <= result.constraints.max_words + 6  # body/cta add a few


def test_stub_honours_constraints():
    stub = StubCopyLLM()
    out = stub.generate_variants(_draft(), 3, RewriteConstraints(max_words=5))
    assert len(out) == 3
    assert all(len(v["headline"].split()) <= 5 for v in out)


def test_get_llm_stub(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert get_llm().name == "stub"
    assert get_llm("stub").name == "stub"


def test_result_to_dict(account):
    coach = CopyCoach(account, model=CTRModel(backend="baseline"), llm=StubCopyLLM()).fit()
    payload = coach.revise(_draft(), n_variants=2).to_dict()
    assert {"draft", "variants", "best", "improved", "constraints"} <= set(payload)
