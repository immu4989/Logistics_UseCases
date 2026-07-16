"""End-to-end tests on a small synthetic sample.

The two tests that carry the most weight:

- `test_interval_coverage_honest` — the product claim is the P10-P90 range,
  so the range has to hold empirically on data the models never saw.
- `test_shap_recovers_true_drivers` — the synthetic generator's causal
  structure is known, so we can assert the explainability layer surfaces the
  real drivers and buries the planted noise. If a refactor breaks feature
  engineering or explanation grouping, this fails.
"""

import numpy as np
import pandas as pd
import pytest

from eta_regression import cleaning, evaluate, explain, features, schema, synthetic
from eta_regression import train as train_mod

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
    cfg = train_mod.TrainConfig(n_estimators=250, seed=11)
    return train_mod.train(clean_df, cfg)


def test_generator_is_deterministic():
    a = synthetic.make_dataset(n=500, seed=3)
    b = synthetic.make_dataset(n=500, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_generator_schema_and_label_range():
    df = synthetic.make_dataset(n=5000, seed=1)
    schema.validate(df)
    y = df[schema.LABEL_COL]
    assert (y > 0).all()
    assert 2.0 < y.mean() < 6.0  # realistic network-wide mean transit
    # Overnight should actually be overnight-ish and ground should scale up.
    assert y[df["service_level"] == "overnight"].mean() < 2.5
    assert y[df["service_level"] == "ground"].mean() > 3.0


def test_cleaning_removes_mess(raw, clean_df):
    assert clean_df[schema.ID_COL].is_unique
    assert not clean_df["distance_miles"].isin([9999.0]).any()
    assert clean_df["package_weight_lb"].between(*cleaning.BOUNDS["package_weight_lb"]).all()
    assert set(clean_df["service_level"].unique()) <= set(schema.SERVICE_LEVELS)
    assert clean_df[schema.NUMERIC_FEATURES].notna().all().all()
    # The regression-specific defect: impossible transit times must be gone.
    assert (raw[schema.LABEL_COL] <= 0).any()  # the mess was actually injected
    assert (clean_df[schema.LABEL_COL] > 0).all()
    assert (clean_df[schema.LABEL_COL] <= cleaning.LABEL_MAX_DAYS).all()


def test_time_split_no_leakage(trained):
    _, splits = trained
    assert splits["train"][schema.DATE_COL].max() <= splits["test"][schema.DATE_COL].min()


def test_median_model_beats_global_median(trained):
    models, splits = trained
    y_test = splits["y_test"]
    p50 = train_mod.predict_quantiles(models, splits["X_test"])[0.5].to_numpy()
    model_mae = np.mean(np.abs(p50 - y_test))
    naive_mae = np.mean(np.abs(np.median(splits["y_train"]) - y_test))
    # "Quote the network-wide median for everything" is the no-model policy;
    # the P50 model must beat it by a healthy margin, not a rounding error.
    assert model_mae < 0.6 * naive_mae, f"model MAE {model_mae:.3f} vs naive {naive_mae:.3f}"


def test_interval_coverage_honest(trained):
    models, splits = trained
    y_test = splits["y_test"]
    q = train_mod.predict_quantiles(models, splits["X_test"])
    cov = evaluate.interval_metrics(y_test, q[0.1].to_numpy(), q[0.9].to_numpy())["coverage"]
    assert 0.70 <= cov <= 0.90, f"P10-P90 coverage {cov:.3f} outside [0.70, 0.90]"


def test_shap_recovers_true_drivers(trained, tmp_path):
    models, splits = trained
    ranking = explain.explain(models, splits, tmp_path, max_background=2000)
    top8 = set(ranking.head(8)["driver"])

    # The strongest ground-truth drivers must appear among the top-ranked
    # levers. Distance may surface through the engineered leg count, and
    # congestion through the engineered total_hub_congestion.
    assert any("distance" in d or "linehaul_legs" in d for d in top8), f"distance missing: {top8}"
    assert "service_level" in top8, f"service level missing from top drivers: {top8}"
    assert any("congestion" in d for d in top8), f"congestion missing: {top8}"
    assert any("weather" in d for d in top8), f"weather missing: {top8}"

    # Known noise features must NOT outrank the real signal.
    noise_rank = ranking[ranking["driver"] == "declared_value_usd"].index
    assert len(noise_rank) == 0 or noise_rank[0] > 7


def test_score_roundtrip_quantiles_sane(trained):
    models, _ = trained
    new = synthetic.make_dataset(n=300, seed=99, messy=True)
    clean_new, _ = cleaning.clean(new)
    X = features.to_matrix(features.engineer(clean_new))
    X = X.reindex(columns=models.feature_columns, fill_value=0.0)
    q = train_mod.predict_quantiles(models, X, alphas=(0.1, 0.5, 0.9))
    assert np.isfinite(q.to_numpy()).all()
    assert (q[0.1] > 0).all()
    # The quoted range must be ordered for every shipment, no exceptions —
    # a crossed quantile pair on an ops screen discredits all three numbers.
    assert (q[0.1] <= q[0.5]).all()
    assert (q[0.5] <= q[0.9]).all()
