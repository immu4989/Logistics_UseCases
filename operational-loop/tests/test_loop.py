"""Smoke test for the wired loop, small enough for CI, real enough to mean something.

The load-bearing assertion is the economics ladder on REALIZED outcomes:
EV-greedy on model scores must beat top-K risk flagging, which must beat
doing nothing, and none of them may beat the same allocator fed the true
risk. Everything is seeded, so these are deterministic checks, not
flaky-threshold hopes.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_loop  # noqa: E402
from delivery_commit.train import TrainConfig  # noqa: E402

N_DAY = 4_000
N_TRAIN = 8_000
BUDGET = 1_200.0  # scaled with n: same $/shipment as the full run


@pytest.fixture(scope="session")
def result(tmp_path_factory):
    out = tmp_path_factory.mktemp("loop")
    return run_loop.run(
        n_day=N_DAY,
        n_train=N_TRAIN,
        budget=BUDGET,
        train_config=TrainConfig(n_estimators=60, early_stopping_rounds=10),
        out_dir=out,
        chart_path=None,
    )


def test_pipeline_runs_and_reports_all_policies(result):
    comparison = result["comparison"]
    assert set(comparison["policy"]) == set(run_loop.POLICY_LABELS.values())
    assert result["summary"]["n_shipments"] > 0


def test_every_policy_respects_budget(result):
    for name, dec in result["decisions"].items():
        assert dec["spent"].sum() <= BUDGET + 1e-9, f"{name} overspent"


def test_economics_ladder_on_realized_outcomes(result):
    net = result["comparison"].set_index("policy")["net_savings_usd"]
    labels = run_loop.POLICY_LABELS
    none = net[labels["none"]]
    top_k = net[labels["top_k_risk"]]
    greedy = net[labels["expected_value_greedy"]]
    oracle = net[labels["oracle_true_risk"]]
    assert none == 0.0
    assert greedy > top_k > none, f"ladder broken: greedy={greedy}, top_k={top_k}"
    assert oracle >= greedy, f"oracle beaten: oracle={oracle}, greedy={greedy}"


def test_scores_are_model_output_not_true_risk():
    """p_hat must be informative about p_true but not identical to it.

    If a refactor ever leaked p_miss_true into the model matrix, p_hat would
    collapse onto p_true and the oracle gap would silently vanish — the exact
    failure the loop exists to price. (delivery-commit's own suite asserts
    the whitelist; this asserts the observable consequence end to end.)
    """
    import numpy as np
    from delivery_commit.train import TrainConfig

    models = run_loop.train_commit_model(
        N_TRAIN, TrainConfig(n_estimators=60, early_stopping_rounds=10)
    )
    day = run_loop.make_scoring_day(N_DAY)
    p_hat = run_loop.score_day(models, day)
    p_true = day["p_miss_true"].to_numpy()
    rho = np.corrcoef(p_hat, p_true)[0, 1]
    assert 0.4 < rho < 0.995, f"score/truth correlation out of range: {rho}"
