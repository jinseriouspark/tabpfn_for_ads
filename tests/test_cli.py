from typer.testing import CliRunner

from adlift.cli import app

runner = CliRunner()


def test_synth_writes_csv(tmp_path):
    out = tmp_path / "acct.csv"
    result = runner.invoke(app, ["synth", "--out", str(out), "--n-groups", "12"])
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_score_runs_offline():
    result = runner.invoke(
        app, ["score", "--headline", "Cut churn by 30%.", "--backend", "baseline"]
    )
    assert result.exit_code == 0, result.output
    assert "predicted_ctr" in result.output


def test_revise_runs_offline():
    result = runner.invoke(
        app,
        [
            "revise",
            "--headline",
            "Discover how Atlas transforms onboarding.",
            "--backend",
            "baseline",
            "--llm",
            "stub",
            "--n",
            "3",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "constraints" in result.output


def test_demo_writes_report(tmp_path):
    result = runner.invoke(
        app,
        [
            "demo",
            "--out",
            str(tmp_path),
            "--backend",
            "baseline",
            "--n-boot",
            "20",
            "--n-splits",
            "3",
            "--n-groups",
            "40",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "effects.png").exists()
