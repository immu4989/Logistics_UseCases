"""End-to-end tests for the dynamic pricing pipeline.

The load-bearing tests are the policy ordering under exact counterfactual
evaluation (model pricing must beat the tuned flat markup, which must beat
cost-plus, and none may beat the oracle) and the elasticity-sign test: the
trained model's implied demand curve must slope down everywhere and slope
down more steeply for spot than for contract freight. That second test is
this use case's grounded-explainability check — if a refactor makes the
model's demand curves nonsense, the pretty charts lie and CI fails.
"""

import numpy as np
import pandas as pd
import pytest

from freight_pricing import cleaning, evaluate, price, synthetic
from freight_pricing import train as train_mod

N = 40_000
SEED = 7


@pytest.fixture(scope="session")
def raw():
    return synthetic.make_quotes(n=N, seed=SEED, messy=True)


@pytest.fixture(scope="session")
def clean_df(raw):
    df, _ = cleaning.clean(raw)
    return df


@pytest.fixture(scope="session")
def trained(clean_df):
    models, splits = train_mod.train(clean_df, train_mod.TrainConfig(seed=SEED))
    return models, splits


@pytest.fixture(scope="session")
def results(trained, tmp_path_factory):
    models, splits = trained
    out = tmp_path_factory.mktemp("reports")
    comparison, segment, prices, info = evaluate.evaluate_policies(
        models, splits, seed=SEED, out_dir=out
    )
    return comparison.set_index("policy"), segment, prices, info, out


def _ordering_assertions(comparison, prices, test_df):
    em = comparison["expected_margin_usd"]
    # Verified margins at the default setup: model clears flat by >8% and
    # flat clears cost-plus by >4%, so the strict ordering is robust, not a
    # coin flip on the seed.
    assert em["model_pricing"] > em["flat_optimal"] > em["cost_plus"]
    assert em["oracle"] >= em["model_pricing"]
    assert comparison.loc["model_pricing", "pct_of_oracle"] <= 100.0
    # Guardrails: optimized quotes stay inside [floor, cap] x cost, always.
    cost = test_df["our_cost_usd"].to_numpy()
    for name in ["flat_optimal", "model_pricing", "oracle"]:
        mult = prices[name] / cost
        assert mult.min() >= price.GUARDRAIL_FLOOR - 1e-9, name
        assert mult.max() <= price.GUARDRAIL_CAP + 1e-9, name


# ---------------------------------------------------------------------------
# generator + cleaning
# ---------------------------------------------------------------------------


def test_generator_is_deterministic():
    a = synthetic.make_quotes(n=1_000, seed=3, messy=True)
    b = synthetic.make_quotes(n=1_000, seed=3, messy=True)
    pd.testing.assert_frame_equal(a, b)


def test_generator_has_price_variation_and_sane_acceptance(raw):
    ok = raw[raw["quoted_price_usd"] > 0]
    # Without rep-level price noise, elasticity would be unidentifiable.
    ratio = ok["quoted_price_usd"] / ok["our_cost_usd"]
    assert ratio.groupby(ok["customer_segment"].str.strip().str.lower()).std().min() > 0.03
    assert 0.30 < ok["accepted"].mean() < 0.75


def test_cleaning_fixes_each_mess_class(raw, clean_df):
    # Raw really is messy (the test would be vacuous otherwise) ...
    assert raw["quote_id"].duplicated().any()
    assert (raw["weight_lb"] < 0).any()
    assert (raw["quoted_price_usd"] == 0).any()
    assert raw["customer_segment"].str.strip().str.lower().ne(raw["customer_segment"]).any()
    # ... and cleaning undoes every class of it.
    assert clean_df["quote_id"].is_unique
    assert (clean_df["weight_lb"] > 0).all()
    assert (clean_df["quoted_price_usd"] >= cleaning.MIN_PLAUSIBLE_PRICE).all()
    assert set(clean_df["customer_segment"]) == set(synthetic.TRUE_ELASTICITY)
    assert set(clean_df["urgency"]) == {"standard", "express"}
    assert clean_df["weight_lb__was_missing"].sum() > 0  # imputation left its flag


# ---------------------------------------------------------------------------
# acceptance model
# ---------------------------------------------------------------------------


def test_acceptance_model_auc_on_time_split(trained):
    models, splits = trained
    aucs = evaluate._model_quality(models, splits["test"])
    assert aucs["xgboost"] > 0.65
    assert aucs["logistic"] > 0.60


def test_model_implied_elasticity_signs(trained):
    """Win probability must fall with price everywhere, and fall more
    steeply for spot than for contract. This is the grounded-explainability
    test: the demand curves the README shows are asserted, not eyeballed."""
    models, splits = trained
    grid = price.multiplier_grid()
    drops = {}
    for seg in ["spot", "contract"]:
        seg_df = splits["test"][splits["test"]["customer_segment"] == seg].head(200)
        cost = seg_df["our_cost_usd"].to_numpy()
        curves = np.column_stack(
            [train_mod.predict_accept(models, seg_df, cost * m) for m in grid]
        )
        # Monotone constraint: non-increasing in price for EVERY quote.
        assert (np.diff(curves, axis=1) <= 1e-9).all(), seg
        drops[seg] = float((curves[:, 0] - curves[:, -1]).mean())
    assert drops["spot"] > drops["contract"] > 0.05


# ---------------------------------------------------------------------------
# policies + counterfactual evaluation
# ---------------------------------------------------------------------------


def test_policy_ordering_and_guardrails(results, trained):
    comparison, _, prices, _, _ = results
    _, splits = trained
    _ordering_assertions(comparison, prices, splits["test"])


def test_policy_ordering_on_alternate_seed():
    """The ordering is a property of the method, not of seed 7."""
    raw = synthetic.make_quotes(n=16_000, seed=11, messy=True)
    df, _ = cleaning.clean(raw)
    models, splits = train_mod.train(df, train_mod.TrainConfig(seed=11))
    p, _ = evaluate.price_policies(models, splits["train"], splits["test"])
    u = np.random.default_rng(11 + 1_000_003).random(len(splits["test"]))
    em = {
        name: evaluate._policy_metrics(splits["test"], pr, u)["expected_margin_usd"]
        for name, pr in p.items()
    }
    assert em["model_pricing"] > em["flat_optimal"] > em["cost_plus"]
    assert em["oracle"] >= em["model_pricing"]


def test_model_pricing_never_quotes_below_cost(results, trained):
    comparison, _, prices, _, _ = results
    _, splits = trained
    # No loss-leaders: expected margin is non-negative on every single quote.
    cost = splits["test"]["our_cost_usd"].to_numpy()
    assert (prices["model_pricing"] - cost).min() > 0


def test_common_random_number_evaluation_is_reproducible(trained, tmp_path):
    models, splits = trained
    a, seg_a, _, _ = evaluate.evaluate_policies(models, splits, seed=SEED, out_dir=tmp_path / "a")
    b, seg_b, _, _ = evaluate.evaluate_policies(models, splits, seed=SEED, out_dir=tmp_path / "b")
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(seg_a, seg_b)


def test_segment_uplift_concentrates_in_spot(results):
    _, segment, _, _, _ = results
    seg = segment.set_index("segment")
    # The desk overprices its most price-elastic segment, so that is where
    # the model finds the money.
    assert seg.loc["spot", "share_of_total_uplift_pct"] == seg["share_of_total_uplift_pct"].max()
    # And the direction of each correction matches the true elasticities:
    # spot priced down from 1.35x, premium priced up from 1.40x.
    assert seg.loc["spot", "avg_model_multiplier"] < synthetic.COST_PLUS_MARKUP["spot"]
    assert seg.loc["premium", "avg_model_multiplier"] > synthetic.COST_PLUS_MARKUP["premium"]


def test_reports_written(results):
    _, _, _, _, out = results
    for name in [
        "metrics.json",
        "policy_comparison.csv",
        "segment_uplift.csv",
        "policy_comparison.png",
        "margin_volume_frontier.png",
        "elasticity_curves.png",
    ]:
        assert (out / name).exists(), name


def test_rationale_written(results, trained, tmp_path):
    from freight_pricing import explain

    _, _, prices, _, _ = results
    models, splits = trained
    table = explain.write_rationale(splits["test"], prices, models, tmp_path)
    assert (tmp_path / "rationale.md").exists()
    assert len(table) == 3
    assert {"case", "quote_id", "model_price_usd", "why"} <= set(table.columns)
