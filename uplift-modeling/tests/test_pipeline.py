"""End-to-end tests on a smaller synthetic pilot.

The load-bearing assertions are causal, not predictive: the estimators must
recover the SHAPE of the planted treatment effect — big on routing-driven
risk, ~zero on weather-driven risk (despite weather being the riskiest
segment), slightly negative on overnight — and targeting by the DR-learner
must beat targeting by risk, in ranking quality (AUUC) and in dollars.
"""

import numpy as np
import pandas as pd
import pytest

from uplift_modeling import cleaning, evaluate, models, synthetic

N_SMALL = 24_000


def _pipeline(seed: int):
    raw = synthetic.make_dataset(n=N_SMALL, seed=seed, messy=True)
    df, _ = cleaning.clean(raw)
    train, test = models.hash_split(df)
    bundle = models.fit_all(train, seed=seed)
    scores = models.predict_scores(bundle, test)
    return train, test, scores


@pytest.fixture(scope="session")
def pipe():
    return _pipeline(seed=11)


@pytest.fixture(scope="session")
def metrics(pipe, tmp_path_factory):
    _, test, scores = pipe
    out = tmp_path_factory.mktemp("reports")
    return evaluate.evaluate_all(test, scores, seed=11, out_dir=out)


# --------------------------------------------------------------------- generator


def test_generator_is_deterministic():
    a = synthetic.make_dataset(n=800, seed=3, messy=True)
    b = synthetic.make_dataset(n=800, seed=3, messy=True)
    pd.testing.assert_frame_equal(a, b)


def test_generator_rates_and_truth_columns():
    df = synthetic.make_dataset(n=6000, seed=1)
    control = df[df[synthetic.TREATMENT_COL] == 0]
    assert 0.06 < control[synthetic.LABEL_COL].mean() < 0.22
    for col in synthetic.TRUTH_COLS:
        assert col in df.columns
    # true_cate must equal p0 - p1 by construction: bounded by p0 above.
    assert (df["true_cate"] <= df["p0_true"] + 1e-9).all()


def test_rct_balance():
    """Randomization sanity: ~25% treated, and features balanced across arms."""
    df = synthetic.make_dataset(n=20_000, seed=5)
    treated_frac = df[synthetic.TREATMENT_COL].mean()
    assert abs(treated_frac - synthetic.PROPENSITY) < 0.02
    for col in ["distance_miles", "origin_congestion", "dest_weather_severity"]:
        a = df.loc[df[synthetic.TREATMENT_COL] == 1, col]
        b = df.loc[df[synthetic.TREATMENT_COL] == 0, col]
        smd = abs(a.mean() - b.mean()) / df[col].std()
        assert smd < 0.05, f"{col} unbalanced across arms (smd={smd:.3f})"


# ----------------------------------------------------------------------- cleaning


def test_cleaning_removes_each_mess_class(pipe):
    train, test, _ = pipe
    df = pd.concat([train, test])
    assert df[synthetic.ID_COL].is_unique
    assert (df["distance_miles"] > 0).all()
    assert set(df["service_level"].unique()) <= set(synthetic.SERVICE_LEVELS)
    assert set(df["customer_tier"].unique()) <= set(synthetic.CUSTOMER_TIERS)
    assert df[cleaning.NUMERIC_FEATURES].notna().all().all()


# -------------------------------------------------------------- leakage protection


def test_truth_columns_never_reach_the_model():
    """The whitelist must exclude true_cate/p0_true/segment_true, label, treatment."""
    raw = synthetic.make_dataset(n=2000, seed=9, messy=True)
    df, _ = cleaning.clean(raw)
    X = models.to_matrix(df)
    banned = set(synthetic.TRUTH_COLS) | {synthetic.LABEL_COL, synthetic.TREATMENT_COL}
    for col in X.columns:
        assert not any(b in col for b in banned), f"leaked column: {col}"

    # And the matrix is IDENTICAL to one built after dropping the truth columns:
    # their presence may not perturb modeling in any way.
    X_plain = models.to_matrix(df.drop(columns=synthetic.TRUTH_COLS))
    pd.testing.assert_frame_equal(X, X_plain)


# ------------------------------------------------------------------- estimator quality


def _auuc_ratios(test_df, scores, seed):
    tc = test_df["true_cate"].to_numpy()
    oracle = evaluate.auuc(evaluate.qini_exact(tc, tc))
    out = {}
    for m in scores.columns:
        out[m] = evaluate.auuc(evaluate.qini_exact(scores[m].to_numpy(), tc)) / oracle
    rng = np.random.default_rng(seed + 404)
    out["random"] = (
        evaluate.auuc(evaluate.qini_exact(rng.random(len(tc)), tc)) / oracle
    )
    return out

def test_dr_beats_risk_and_random_two_seeds(pipe):
    _, test, scores = pipe
    ratios = _auuc_ratios(test, scores, seed=11)
    assert ratios["dr_learner"] > ratios["risk_targeting"] + 0.08
    assert ratios["dr_learner"] > ratios["random"] + 0.3

    _, test2, scores2 = _pipeline(seed=23)
    ratios2 = _auuc_ratios(test2, scores2, seed=23)
    assert ratios2["dr_learner"] > ratios2["risk_targeting"] + 0.08
    assert ratios2["dr_learner"] > ratios2["random"] + 0.3


def test_oracle_bounds_every_method(metrics):
    auuc = pd.DataFrame(metrics["auuc"]).set_index("method")
    assert auuc.loc["oracle", "auuc_vs_oracle"] == 1.0
    assert (auuc["auuc_vs_oracle"] <= 1.0 + 1e-9).all()
    policy = pd.DataFrame(metrics["policy_value"])
    for k in evaluate.POLICY_KS:
        sub = policy[policy["k"] == k].set_index("method")["net_usd"]
        assert sub["oracle"] >= sub.drop("oracle").max() - 1e-6


def test_dr_policy_value_beats_risk_at_20pct(metrics):
    policy = pd.DataFrame(metrics["policy_value"])
    sub = policy[policy["k"] == 0.20].set_index("method")["net_usd"]
    assert sub["dr_learner"] > sub["risk_targeting"]


# --------------------------------------------------------------- the money insights


def test_dr_learns_weather_is_not_fixable(pipe):
    """Weather segment: highest RISK, near-zero uplift — and dr must see it."""
    _, test, scores = pipe
    seg = test["segment_true"]
    dr = scores["dr_learner"]
    routing_mean = dr[seg == "routing_driven"].mean()
    weather_mean = dr[seg == "weather_driven"].mean()
    assert routing_mean > 0.05  # the fixable segment is found
    assert weather_mean < 0.4 * routing_mean
    # meanwhile risk targeting scores weather HIGHEST of all segments:
    risk = scores["risk_targeting"]
    assert risk[seg == "weather_driven"].mean() > risk[seg == "routing_driven"].mean()


def test_dr_learns_overnight_harm_or_at_least_no_gain(pipe):
    """Overnight segment: true effect is -2pp. Don't overclaim the sign — the
    assertion is that dr predicts at most negligible positive uplift there."""
    _, test, scores = pipe
    seg = test["segment_true"]
    overnight_mean = scores["dr_learner"][seg == "overnight"].mean()
    assert overnight_mean < 0.01
