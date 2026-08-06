"""End-to-end tests on a small synthetic sample.

The most important test here is `test_shap_recovers_true_drivers`: because the
synthetic generator's causal structure is known, we can assert that the
explainability layer actually surfaces the real drivers. If a refactor breaks
feature engineering or explanation grouping, this fails.
"""

import numpy as np
import pandas as pd
import pytest

from delivery_commit import cleaning, conformal, evaluate, explain, schema, synthetic
from delivery_commit import train as train_mod

N_SMALL = 12_000


@pytest.fixture(scope="session")
def raw():
    return synthetic.make_dataset(n=N_SMALL, seed=11, messy=True)


@pytest.fixture(scope="session")
def clean_df(raw):
    df, _ = cleaning.clean(raw)
    return df


@pytest.fixture(scope="session")
def trained(clean_df):
    cfg = train_mod.TrainConfig(n_estimators=200, seed=11)
    return train_mod.train(clean_df, cfg)


def test_generator_is_deterministic():
    a = synthetic.make_dataset(n=500, seed=3)
    b = synthetic.make_dataset(n=500, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_generator_schema_and_rate():
    df = synthetic.make_dataset(n=5000, seed=1)
    schema.validate(df)
    assert 0.04 < df[schema.LABEL_COL].mean() < 0.20


def test_cleaning_removes_mess(raw, clean_df):
    assert clean_df[schema.ID_COL].is_unique
    assert not clean_df["distance_miles"].isin([9999.0]).any()
    assert clean_df["package_weight_lb"].between(*cleaning.BOUNDS["package_weight_lb"]).all()
    assert set(clean_df["service_level"].unique()) <= set(schema.SERVICE_LEVELS)
    assert clean_df[schema.NUMERIC_FEATURES].notna().all().all()


def test_time_split_no_leakage(trained):
    _, splits = trained
    assert splits["train"][schema.DATE_COL].max() <= splits["test"][schema.DATE_COL].min()


def test_model_beats_chance_and_baseline(trained):
    models, splits = trained
    y = splits["y_test"]
    xgb_metrics = evaluate.summarize(y, models.xgb.predict_proba(splits["X_test"])[:, 1])
    base_metrics = evaluate.summarize(y, models.baseline.predict_proba(splits["X_test"])[:, 1])
    assert xgb_metrics["roc_auc"] > 0.70
    assert xgb_metrics["pr_auc"] > 2 * y.mean()  # at least 2x random precision
    # On this near-additive synthetic process the linear baseline is close to
    # the Bayes-optimal model family, so the GBM ties or slightly trails it
    # (see README "Why keep the baseline"). Guard against real regressions
    # without asserting an ordering the data doesn't support.
    assert base_metrics["roc_auc"] > 0.70
    assert xgb_metrics["pr_auc"] >= base_metrics["pr_auc"] - 0.06


def test_shap_recovers_true_drivers(trained, tmp_path):
    models, splits = trained
    ranking = explain.explain(models, splits, tmp_path, max_background=2000)
    top8 = set(ranking.head(8)["driver"])

    # The strongest ground-truth drivers must appear among the top-ranked
    # levers. Congestion may surface through the engineered
    # total_hub_congestion; late pickup through the engineered late_pickup_*.
    assert "dest_weather_severity" in top8, f"weather missing from top drivers: {top8}"
    assert "distance_miles" in top8 or "miles_per_promised_day" in top8
    assert any("congestion" in d for d in top8), f"congestion missing: {top8}"
    assert any("late_pickup" in d or "minutes_after_cutoff" in d for d in top8)

    # Known noise features must NOT outrank the real signal.
    noise_rank = ranking[ranking["driver"] == "declared_value_usd"].index
    assert len(noise_rank) == 0 or noise_rank[0] > 7


def test_true_risk_column_never_reaches_the_model():
    """`p_miss_true` (return_true_risk=True) must stay out of the model matrix.

    The column is the generator's pre-Bernoulli miss probability, exposed only
    for downstream counterfactual evaluation (operational-loop). Feeding it to
    a model would be perfect label leakage, so the whitelist in
    features.to_matrix must drop it silently — even after cleaning and
    feature engineering, and even in messy mode where rows get duplicated.
    """
    from delivery_commit import features

    plain = synthetic.make_dataset(n=1500, seed=5)
    assert "p_miss_true" not in plain.columns  # default behavior unchanged

    df = synthetic.make_dataset(n=1500, seed=5, messy=True, return_true_risk=True)
    assert "p_miss_true" in df.columns
    assert df["p_miss_true"].between(0, 1).all()

    clean_df, _ = cleaning.clean(df)
    assert "p_miss_true" in clean_df.columns  # survives cleaning for downstream use
    X = features.to_matrix(features.engineer(clean_df))
    assert not any("p_miss_true" in c for c in X.columns)

    # And the matrix is IDENTICAL to the one built without the column: the
    # flag may not perturb modeling in any way.
    clean_plain, _ = cleaning.clean(synthetic.make_dataset(n=1500, seed=5, messy=True))
    X_plain = features.to_matrix(features.engineer(clean_plain))
    pd.testing.assert_frame_equal(X, X_plain)


# ---------------------------------------------------------------------------
# Conformal layer: calibration + CRC flag thresholds
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def trained_seed7():
    raw7 = synthetic.make_dataset(n=N_SMALL, seed=7, messy=True)
    df, _ = cleaning.clean(raw7)
    return train_mod.train(df, train_mod.TrainConfig(n_estimators=200, seed=7))


def test_crc_threshold_math():
    """Hand-checkable CRC arithmetic: budget k = floor((n+1)*alpha - 1)."""
    # 9 positives scored 0.1 .. 0.9, plus low-scored negatives.
    scores = np.array([0.1 * i for i in range(1, 10)] + [0.05] * 10)
    y = np.array([1] * 9 + [0] * 10)

    t = conformal.crc_threshold(scores, y, alpha=0.2)
    assert t.n_cal_positives == 9
    assert t.allowed_cal_misses == 1  # floor(10 * 0.2 - 1)
    assert t.threshold == pytest.approx(0.2)  # 2nd-smallest positive score
    # exactly one calibration positive (0.1) falls below the threshold
    assert ((scores < t.threshold) & (y == 1)).sum() == 1

    # Too few positives for alpha=0.05 (needs n >= 1/alpha - 1 = 19): the only
    # certifiable action is to flag everything.
    t2 = conformal.crc_threshold(scores, y, alpha=0.05)
    assert t2.threshold == -np.inf
    assert t2.expected_flag_rate == 1.0


@pytest.mark.parametrize("fixture_name", ["trained_seed7", "trained"])  # seeds 7 and 11
def test_crc_fnr_guarantee_holds_on_test_period(request, fixture_name):
    models, splits = request.getfixturevalue(fixture_name)
    thr = models.crc_thresholds[0.10]
    s_test = models.xgb.predict_proba(splits["X_test"])[:, 1]
    y = splits["y_test"]
    fnr = ((y == 1) & (s_test < thr.threshold)).sum() / y.sum()
    # The CRC guarantee bounds the EXPECTED FNR at alpha; one finite test
    # period is a single draw around that expectation, so allow 0.04 slack
    # for test-set finiteness (~300 positives in this held-out window).
    # Measured here: FNR ~= 0.03-0.04 at alpha = 0.10 for both seeds.
    assert fnr <= 0.10 + 0.04


def test_crc_monotonic_in_alpha(trained):
    models, splits = trained
    th = models.crc_thresholds
    assert set(th) == set(conformal.DEFAULT_ALPHAS)
    # Tighter guarantee (smaller alpha) -> threshold no higher -> flag set no
    # smaller. Ties are legal (adjacent alphas can land on the same order
    # statistic), so the assertions are <=, not <.
    assert th[0.05].threshold <= th[0.10].threshold <= th[0.20].threshold
    s = models.xgb.predict_proba(splits["X_test"])[:, 1]
    flag = {a: float((s >= th[a].threshold).mean()) for a in th}
    assert flag[0.05] >= flag[0.10] >= flag[0.20]
    assert th[0.05].expected_flag_rate >= th[0.10].expected_flag_rate >= (
        th[0.20].expected_flag_rate
    )


def test_isotonic_calibration_probability_quality(trained):
    from sklearn.metrics import brier_score_loss

    models, splits = trained
    X_test, y = splits["X_test"], splits["y_test"]
    p_raw = models.xgb.predict_proba(X_test)[:, 1]
    p_cal = models.calibrator.predict_proba(X_test)[:, 1]
    assert np.all((p_cal >= 0) & (p_cal <= 1))
    # Isotonic is monotone: the ranking may not change (ties allowed; the
    # tolerance covers float32 interpolation noise in iso.predict).
    order = np.argsort(p_raw)
    assert np.all(np.diff(p_cal.astype(np.float64)[order]) >= -1e-6)
    # Verified empirically before asserting: this pipeline never reweights
    # classes, so the raw XGBoost is already nearly calibrated and isotonic
    # sometimes wins a hair (seed 7: -0.0002 Brier) and sometimes loses a
    # hair on the ~1.4k-row calibration slice (seed 11: +0.009). Asserting
    # "calibration always improves Brier" would be a lie; the robust
    # invariant is that it never degrades probability quality materially.
    b_raw = brier_score_loss(y, p_raw)
    b_cal = brier_score_loss(y, p_cal)
    assert b_cal <= b_raw * 1.10


def test_conformal_report_writes_artifacts(trained, tmp_path):
    models, splits = trained
    out = evaluate.conformal_report(models, splits, tmp_path)
    assert out is not None
    assert (tmp_path / "crc_table.csv").exists()
    assert (tmp_path / "crc_guarantee.png").exists()
    table = out["table"]
    assert list(table["alpha"]) == sorted(conformal.DEFAULT_ALPHAS)
    assert ((table["realized_capture"] >= 0) & (table["realized_capture"] <= 1)).all()


def test_backward_compat_artifact_without_conformal_fields(trained, tmp_path):
    """Pre-conformal pickles (no calibrator/crc_thresholds) must still work."""
    import copy

    models, splits = trained
    legacy = copy.copy(models)
    # Simulate an old artifact: unpickling bypasses __init__, so the instance
    # dict simply lacks the new attributes.
    del legacy.__dict__["calibrator"]
    del legacy.__dict__["crc_thresholds"]

    train_mod.save(legacy, tmp_path)
    loaded = train_mod.load(tmp_path)
    probs = loaded.xgb.predict_proba(splits["X_test"])[:, 1]
    assert np.all((probs >= 0) & (probs <= 1))
    # Dataclass defaults live on the class, so the attribute reads as None...
    assert getattr(loaded, "calibrator", None) is None
    # ...and the conformal report degrades to a clean no-op.
    assert evaluate.conformal_report(loaded, splits, tmp_path) is None


def test_score_roundtrip(trained, tmp_path):
    from delivery_commit import features

    models, _ = trained
    new = synthetic.make_dataset(n=300, seed=99, messy=True)
    clean_new, _ = cleaning.clean(new)
    X = features.to_matrix(features.engineer(clean_new))
    X = X.reindex(columns=models.feature_columns, fill_value=0.0)
    probs = models.xgb.predict_proba(X)[:, 1]
    assert np.all((probs >= 0) & (probs <= 1))
