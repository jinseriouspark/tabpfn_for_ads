import json

import pytest


@pytest.fixture(scope="module")
def server():
    import os

    os.environ["ADLIFT_BACKEND"] = "baseline"
    os.environ.pop("ANTHROPIC_API_KEY", None)
    from mcp_server import server as srv

    srv.STATE.backend = "baseline"
    return srv


def _tool_names(mcp):
    import asyncio
    import inspect

    listed = mcp.list_tools()
    if inspect.isawaitable(listed):
        listed = asyncio.run(listed)
    return {t.name for t in listed}


def test_tools_registered(server):
    names = _tool_names(server.mcp)
    assert {
        "load_account",
        "score_creative",
        "revise_creative",
        "ingest_banner",
        "causal_report",
        "cold_start_benchmark",
        "account_status",
    } <= names


def test_load_and_score(server):
    out = json.loads(server.load_account(n_campaigns=30, seed=2))
    assert out["rows"] > 0 and out["model_backend"] == "baseline"
    scored = json.loads(server.score_creative(headline="Cut churn by 30%."))
    assert 0 <= scored["predicted_ctr"] <= 1


def test_revise_and_status(server):
    out = json.loads(
        server.revise_creative(headline="Discover how Atlas transforms onboarding.", n_variants=2)
    )
    assert len(out["variants"]) == 2
    status = json.loads(server.account_status())
    assert status["rows"] > 0


def test_causal_report(server):
    out = json.loads(server.causal_report(n_boot=10))
    assert "total_effect" in out and "checks" in out
