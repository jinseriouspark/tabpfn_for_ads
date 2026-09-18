from adlift.coldstart import comparison_table, evaluate_cold_start, random_baseline
from adlift.model import CTRModel


def test_model_beats_random(account):
    model = evaluate_cold_start(account, CTRModel(backend="baseline"), name="gbdt", n_splits=3)
    random = random_baseline(account)
    assert model.top1_regret < random.top1_regret
    assert model.spearman > random.spearman + 0.3
    # recall@k moves in steps of 1/60 on this fixture; only guard against a collapse.
    assert model.recall_at_k[3] >= random.recall_at_k[3] - 0.05
    assert 0 < model.exploration_saved[2] < 1


def test_comparison_table_sorted(account):
    rows = [
        random_baseline(account),
        evaluate_cold_start(account, CTRModel(backend="baseline"), name="gbdt", n_splits=3),
    ]
    table = comparison_table(rows)
    assert list(table["Top-1 regret"]) == sorted(table["Top-1 regret"])
    assert table.iloc[0]["Model"] == "gbdt"
