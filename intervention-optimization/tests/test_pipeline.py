"""End-to-end tests for the intervention optimizer.

The load-bearing test is the policy ordering under paired counterfactual
evaluation: on the same day, same draws, and same budget, the expected-value
policy must beat top-K risk flagging, which must beat untargeted spend, and
none of them may beat the oracle. If a refactor breaks the economics (a sign
flip in EV, a policy overspending, greedy funding negative-EV actions), these
fail.
"""

import numpy as np
import pandas as pd
import pytest

from intervention_opt import evaluate, interventions, policy, synthetic

N = 20_000
SEED = 7
BUDGET = policy.DEFAULT_BUDGET_USD


@pytest.fixture(scope="session")
def day():
    return synthetic.make_day(n=N, seed=SEED)


@pytest.fixture(scope="session")
def results(day, tmp_path_factory):
    out = tmp_path_factory.mktemp("reports")
    comparison, decisions = evaluate.evaluate_all(day, budget=BUDGET, seed=SEED, out_dir=out)
    return comparison.set_index("policy"), decisions


def test_generator_is_deterministic():
    a = synthetic.make_day(n=800, seed=3)
    b = synthetic.make_day(n=800, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_generator_shape_and_risk(day):
    assert list(day.columns) == [
        "shipment_id", "customer_tier", "declared_value_usd", "p_true", "p_hat",
    ]
    assert day["shipment_id"].is_unique
    # Right-skewed risk around ~10%: median well below the mean.
    assert 0.07 < day["p_true"].mean() < 0.14
    assert day["p_true"].median() < day["p_true"].mean()
    assert day[["p_true", "p_hat"]].min().min() > 0
    assert day[["p_true", "p_hat"]].max().max() < 1
    # The score must be informative but NOT perfect (it's the whole premise).
    rho = np.corrcoef(day["p_true"], day["p_hat"])[0, 1]
    assert 0.6 < rho < 0.99


def test_every_policy_respects_budget(day):
    for name, fn in policy.POLICIES.items():
        dec = fn(day, BUDGET, seed=SEED)
        assert dec["spent"].sum() <= BUDGET + 1e-9, f"{name} overspent"


def test_decision_frame_covers_every_shipment_once(day):
    for name, fn in policy.POLICIES.items():
        dec = fn(day, BUDGET, seed=SEED)
        assert len(dec) == len(day), name
        assert set(dec["shipment_id"]) == set(day["shipment_id"]), name
        assert dec["shipment_id"].is_unique, name


def test_greedy_never_funds_negative_ev(day):
    dec = policy.policy_ev_greedy(day, BUDGET, seed=SEED)
    ev = policy.ev_matrix(day)
    acted = dec["action"] != "do_nothing"
    chosen_ev = ev.to_numpy()[
        np.arange(len(day))[acted],
        [ev.columns.get_loc(a) for a in dec.loc[acted, "action"]],
    ]
    assert (chosen_ev > 0).all()
    assert (dec.loc[acted, "expected_savings"] > 0).all()


def test_policy_ordering(results):
    comparison, _ = results
    net = comparison["net_savings_usd"]
    # Verified margins at n=20k/seed=7: greedy clears top_k by >2x and random
    # by a wide gap, so the strict ordering below is robust, not a coin flip.
    assert net["expected_value_greedy"] > net["top_k_risk"]
    assert net["expected_value_greedy"] > net["random"]
    assert net["top_k_risk"] > net["random"]
    assert net["expected_value_greedy"] > 0
    # Untargeted upgrades lose money outright, so doing NOTHING beats random
    # spend. That inversion is the point of the use case, and it is asserted.
    assert net["none"] == 0
    assert net["random"] < net["none"]
    # Healthy economics at the default budget.
    assert comparison.loc["expected_value_greedy", "roi"] > 1


def test_oracle_is_the_upper_bound(results):
    comparison, _ = results
    net = comparison["net_savings_usd"]
    assert net["oracle"] >= net["expected_value_greedy"]
    assert comparison.loc["expected_value_greedy", "pct_of_oracle"] <= 100.0


def test_paired_evaluation_is_reproducible(day, tmp_path):
    a, _ = evaluate.evaluate_all(day, budget=BUDGET, seed=SEED, out_dir=tmp_path / "a")
    b, _ = evaluate.evaluate_all(day, budget=BUDGET, seed=SEED, out_dir=tmp_path / "b")
    pd.testing.assert_frame_equal(a, b)


def test_miss_cost_model(day):
    cost = interventions.miss_cost(day["customer_tier"], day["declared_value_usd"])
    # Floors: base cost per tier; ceiling: tier base + capped value at risk.
    floor = day["customer_tier"].map(interventions.TIER_COST_MULTIPLIER) * 30.0
    assert (cost >= floor - 1e-9).all()
    assert (cost <= floor + interventions.VALUE_AT_RISK_CAP_USD + 1e-9).all()


def test_rationale_written(day, results, tmp_path):
    from intervention_opt import explain

    _, decisions = results
    table = explain.write_rationale(day, decisions["expected_value_greedy"], tmp_path)
    assert (tmp_path / "rationale.md").exists()
    assert (tmp_path / "decision_policy.csv").exists()
    assert len(table) >= 2
    matrix = pd.read_csv(tmp_path / "decision_policy.csv")
    # Full EV matrix: every example shipment appears with every action priced.
    assert set(matrix["action"]) == {a.name for a in interventions.CATALOG}
    assert matrix.groupby("shipment_id")["chosen"].sum().eq(1).all()
