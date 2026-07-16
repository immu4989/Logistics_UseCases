"""End-to-end tests on a small synthetic sample.

Three tests here carry the whole use case:

- `test_customer_history_is_causal` pins the leakage rule for the history
  features: prior_return_rate at order t must be computed from orders
  strictly before t, verified on a constructed example where a global
  computation would give a different answer.
- `test_shap_recovers_true_drivers` asserts the explainability layer
  surfaces the generator's real drivers and buries the planted noise.
- `test_expected_cost_targeting_wins` asserts the product claim: ranking by
  p * cost captures more return dollars, and funds a better intervention,
  than ranking by p.
"""

import numpy as np
import pandas as pd
import pytest

from returns_prediction import cleaning, evaluate, explain, features, schema, synthetic
from returns_prediction import train as train_mod

N_SMALL = 25_000


@pytest.fixture(scope="session")
def raw():
    return synthetic.make_dataset(n=N_SMALL, seed=11, messy=True)


@pytest.fixture(scope="session")
def clean_df(raw):
    df, _ = cleaning.clean(raw)
    return df


@pytest.fixture(scope="session")
def trained(clean_df):
    return train_mod.train(clean_df, train_mod.TrainConfig(seed=11))


def _fit_and_simulate(seed: int):
    raw = synthetic.make_dataset(n=N_SMALL, seed=seed, messy=True)
    df, _ = cleaning.clean(raw)
    models, splits = train_mod.train(df, train_mod.TrainConfig(seed=seed))
    p = models.xgb.predict_proba(splits["X_test"])[:, 1]
    cost = evaluate.unit_return_cost(splits["test"])
    policies = evaluate.simulate_intervention(splits["test"], p, cost, k=0.10)
    return policies.set_index("policy")["net_savings_usd"]


def test_generator_is_deterministic():
    a = synthetic.make_dataset(n=2_000, seed=3, messy=True)
    b = synthetic.make_dataset(n=2_000, seed=3, messy=True)
    pd.testing.assert_frame_equal(a, b)


def test_generator_schema_and_rate():
    df = synthetic.make_dataset(n=8_000, seed=1)
    schema.validate(df)
    assert 0.12 < df[schema.LABEL_COL].mean() < 0.24
    rates = df.groupby("product_category")[schema.LABEL_COL].mean()
    assert rates["apparel"] > 0.2 and rates["shoes"] > 0.2  # fashion band
    assert rates["electronics"] < 0.12                       # low-return category


def test_cleaning_removes_each_mess_class(raw, clean_df):
    # duplicates
    assert clean_df[schema.ID_COL].is_unique
    # negative prices
    assert (clean_df["unit_price_usd"] > 0).all()
    # impossible discounts
    assert clean_df["discount_pct"].between(0, 100).all()
    # inconsistent category casing
    assert set(clean_df["product_category"].unique()) <= set(schema.CATEGORIES)
    # nothing left unimputed
    assert clean_df[schema.NUMERIC_FEATURES].notna().all().all()


def test_customer_history_is_causal():
    # One customer, four orders, alternating labels. If prior_return_rate at
    # order t saw anything from t onward, every expected value below breaks.
    df = pd.DataFrame(
        {
            schema.ID_COL: ["A", "B", "C", "D"],
            schema.DATE_COL: pd.to_datetime(
                ["2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01"]
            ),
            schema.CUSTOMER_COL: ["c1"] * 4,
            schema.LABEL_COL: [1, 0, 1, 0],
        }
    )
    hist = synthetic.compute_customer_history(df)
    assert hist["prior_orders"].tolist() == [0, 1, 2, 3]
    # order 1 sees {}, order 2 sees {1}, order 3 sees {1,0}, order 4 sees {1,0,1}
    np.testing.assert_allclose(hist["prior_return_rate"], [0.0, 1.0, 0.5, 2 / 3])
    # A global (leaky) computation would give every order 0.5.


def test_generator_history_matches_recomputation():
    df = synthetic.make_dataset(n=5_000, seed=5)
    hist = synthetic.compute_customer_history(df)
    assert (df["prior_orders"].to_numpy() == hist["prior_orders"].to_numpy()).all()
    np.testing.assert_allclose(
        df["prior_return_rate"].to_numpy(), hist["prior_return_rate"].round(4).to_numpy()
    )


def test_time_split_no_leakage(trained):
    _, splits = trained
    # The test period sits strictly after training: customer-history features
    # only ever look backwards, and so must the split.
    assert splits["train"][schema.DATE_COL].max() <= splits["test"][schema.DATE_COL].min()
    # Post-ship observations and the label never enter the model matrix.
    for col in schema.POST_SHIP_COLS + [schema.LABEL_COL]:
        assert col not in splits["X_train"].columns


def test_xgboost_beats_logistic_on_pr_auc(trained):
    models, splits = trained
    y = splits["y_test"]
    xgb_metrics = evaluate.summarize(y, models.xgb.predict_proba(splits["X_test"])[:, 1])
    base_metrics = evaluate.summarize(y, models.baseline.predict_proba(splits["X_test"])[:, 1])
    assert xgb_metrics["roc_auc"] > 0.72
    assert xgb_metrics["pr_auc"] > 2 * y.mean()  # at least 2x random precision
    # The return process is thresholds and interactions (see synthetic.py),
    # which is exactly the regime where the GBM must beat a linear model.
    assert xgb_metrics["pr_auc"] > base_metrics["pr_auc"]


def test_expected_cost_targeting_wins(trained):
    models, splits = trained
    y = splits["y_test"]
    p = models.xgb.predict_proba(splits["X_test"])[:, 1]
    cost = evaluate.unit_return_cost(splits["test"])
    captured_ec = evaluate.dollars_captured_at(y, p * cost, cost, k=0.10)
    captured_p = evaluate.dollars_captured_at(y, p, cost, k=0.10)
    assert captured_ec > captured_p


def test_intervention_savings_positive_and_ordered_on_two_seeds(trained):
    models, splits = trained
    p = models.xgb.predict_proba(splits["X_test"])[:, 1]
    cost = evaluate.unit_return_cost(splits["test"])
    first = evaluate.simulate_intervention(splits["test"], p, cost, k=0.10)
    first = first.set_index("policy")["net_savings_usd"]
    second = _fit_and_simulate(seed=13)
    for net in (first, second):
        assert net["expected_cost"] > 0
        assert net["expected_cost"] > net["raw_probability"] > net["random"]


def test_shap_recovers_true_drivers_and_buries_noise(trained, tmp_path):
    models, splits = trained
    ranking = explain.explain(models, splits, tmp_path, max_background=3000)
    top5 = set(ranking.head(5)["driver"])

    assert "bracket_buying" in top5, f"bracket buying missing from top drivers: {top5}"
    assert "product_category" in top5, f"category missing from top drivers: {top5}"
    assert "prior_return_rate" in top5, f"return history missing from top drivers: {top5}"

    # Planted noise must NOT outrank the real signal.
    ranked = ranking.reset_index(drop=True)
    for noise in synthetic.NOISE_FEATURES:
        pos = ranked.index[ranked["driver"] == noise]
        assert len(pos) == 0 or pos[0] >= 7, f"{noise} ranked too high"
        share = ranked.loc[ranked["driver"] == noise, "share_of_explanation"]
        assert share.empty or float(share.iloc[0]) < 0.02

    # The order card exists and names the intervention.
    card = (tmp_path / "example_order.md").read_text()
    assert "fit assistant" in card


def test_score_roundtrip(trained):
    models, _ = trained
    new = synthetic.make_dataset(n=400, seed=99, messy=True)
    clean_new, _ = cleaning.clean(new)
    X = features.to_matrix(features.engineer(clean_new))
    X = X.reindex(columns=models.feature_columns, fill_value=0.0)
    probs = models.xgb.predict_proba(X)[:, 1]
    assert np.all((probs >= 0) & (probs <= 1))
