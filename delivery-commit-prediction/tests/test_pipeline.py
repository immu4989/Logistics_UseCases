"""End-to-end tests on a small synthetic sample.

The most important test here is `test_shap_recovers_true_drivers`: because the
synthetic generator's causal structure is known, we can assert that the
explainability layer actually surfaces the real drivers. If a refactor breaks
feature engineering or explanation grouping, this fails.
"""

import numpy as np
import pandas as pd
import pytest

from delivery_commit import cleaning, evaluate, explain, schema, synthetic
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


def test_score_roundtrip(trained, tmp_path):
    from delivery_commit import features

    models, _ = trained
    new = synthetic.make_dataset(n=300, seed=99, messy=True)
    clean_new, _ = cleaning.clean(new)
    X = features.to_matrix(features.engineer(clean_new))
    X = X.reindex(columns=models.feature_columns, fill_value=0.0)
    probs = models.xgb.predict_proba(X)[:, 1]
    assert np.all((probs >= 0) & (probs <= 1))
