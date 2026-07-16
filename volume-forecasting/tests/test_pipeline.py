"""End-to-end tests on the full synthetic network.

The two most important tests here are `test_model_beats_seasonal_naive` (a
volume model that cannot clearly beat "same weekday last week" does not earn a
staffing change) and `test_shap_recovers_true_drivers` (because the
generator's components are exposed, we can assert the explainability layer
surfaces the real drivers and buries the planted noise). If a refactor breaks
feature engineering, the split, or the explanation grouping, these fail.
"""

import pandas as pd
import pytest

from volume_forecasting import cleaning, evaluate, explain, features, schema, synthetic
from volume_forecasting import train as train_mod


@pytest.fixture(scope="session")
def raw():
    return synthetic.make_dataset(seed=11, messy=True)


@pytest.fixture(scope="session")
def clean_df(raw):
    df, _ = cleaning.clean(raw)
    return df


@pytest.fixture(scope="session")
def trained(clean_df):
    cfg = train_mod.TrainConfig(n_estimators=400, seed=11)
    return train_mod.train(clean_df, cfg)


@pytest.fixture(scope="session")
def results(trained, tmp_path_factory):
    models, splits = trained
    return evaluate.evaluate_models(models, splits, tmp_path_factory.mktemp("reports"))


def test_generator_is_deterministic():
    a = synthetic.make_dataset(seed=3, messy=True)
    b = synthetic.make_dataset(seed=3, messy=True)
    pd.testing.assert_frame_equal(a, b)


def test_generator_schema_and_scale():
    df = synthetic.make_dataset(seed=1)
    schema.validate(df)
    assert df[schema.HUB_COL].nunique() == 15
    assert df[schema.DATE_COL].nunique() == 912
    assert (df[schema.TARGET_COL] >= 0).all()
    # The hub-size spread the terciles exist for: superhub >> smallest sort.
    means = df.groupby(schema.HUB_COL)[schema.TARGET_COL].mean()
    assert means.max() / means.min() > 20


def test_cleaning_fixes_each_mess_class(raw, clean_df):
    _, report = cleaning.clean(raw)
    steps = {s["step"]: s["rows_affected"] for s in report.steps}

    # Each injected mess class was actually seen and handled.
    assert steps["drop_duplicate_hub_days"] > 0
    assert steps["drop_negative_volumes"] > 0
    assert steps["drop_corrupted_volumes"] > 0
    assert steps["calendar_gaps_left_as_gaps"] > 0

    # And the output is actually clean: unique keys, no negatives, no 10x spikes.
    assert not clean_df.duplicated([schema.HUB_COL, schema.DATE_COL]).any()
    assert (clean_df[schema.TARGET_COL] >= 0).all()
    truth = synthetic.make_dataset(seed=11, messy=False)
    merged = clean_df.merge(
        truth, on=[schema.HUB_COL, schema.DATE_COL], suffixes=("_clean", "_true")
    )
    ratio = merged[f"{schema.TARGET_COL}_clean"] / merged[f"{schema.TARGET_COL}_true"].clip(lower=1)
    assert (ratio < 5).all(), "a 10x-corrupted volume survived cleaning"


def test_features_tolerate_gaps_and_lags_are_lags(clean_df):
    feat = features.build_features(clean_df)
    # Gap days are not rows; no invented history.
    assert feat[schema.TARGET_COL].notna().all()
    assert feat["lag_7"].notna().all()

    # lag_7 is the volume 7 *calendar* days earlier, even across gaps.
    lookup = clean_df.set_index([schema.HUB_COL, schema.DATE_COL])[schema.TARGET_COL]
    sample = feat.sample(300, random_state=0)
    for _, row in sample.iterrows():
        key = (row[schema.HUB_COL], row[schema.DATE_COL] - pd.Timedelta(days=7))
        assert key in lookup.index
        assert row["lag_7"] == lookup.loc[key]


def test_time_split_no_leakage(trained):
    _, splits = trained
    assert splits["train"][schema.DATE_COL].max() <= splits["test"][schema.DATE_COL].min()
    # The test window must contain the December peak, or the evaluation lies.
    test_dates = splits["test"][schema.DATE_COL]
    assert ((test_dates.dt.month == 12) & (test_dates.dt.day == 18)).any()
    # No same-day information in the matrix: every feature is a lag or calendar.
    assert schema.TARGET_COL not in splits["X_test"].columns


def test_model_beats_seasonal_naive(trained, results):
    models, splits = trained
    acc = results["accuracy"]
    naive_wape = acc["seasonal_naive"]["overall"]["wape"]
    model_wape = acc["xgboost_point"]["overall"]["wape"]
    assert model_wape < 0.70 * naive_wape, (
        f"XGBoost WAPE {model_wape:.3f} is not comfortably below naive {naive_wape:.3f}"
    )
    assert acc["xgboost_point"]["peak_season"]["wape"] < acc["seasonal_naive"]["peak_season"]["wape"]
    # Peak-season WAPE is reported and sane (staffing-usable, not a coin flip).
    assert 0.0 < acc["xgboost_point"]["peak_season"]["wape"] < 0.20
    # No chronic under-forecast: bias within a couple of points either way.
    assert abs(acc["xgboost_point"]["overall"]["bias"]) < 0.03


def test_quantile_coverage_honest(results):
    assert 0.70 <= results["quantiles"]["coverage_p10_p90"] <= 0.90


def test_quantiles_do_not_cross(trained):
    models, splits = trained
    q = train_mod.predict_quantiles(models, splits["X_test"])
    assert (q["p10"] <= q["p50"]).all()
    assert (q["p50"] <= q["p80"]).all()
    assert (q["p80"] <= q["p90"]).all()


def test_staffing_to_p80_beats_point_forecast(results):
    staffing = {r["policy"]: r for r in results["staffing"]}
    assert (
        staffing["staff_to_p80"]["understaffed_days_pct"]
        < staffing["point_forecast"]["understaffed_days_pct"]
    )
    # The point forecast leaves the floor short roughly half the time by
    # construction (it is a median); P80 should be well under that.
    assert staffing["staff_to_p80"]["understaffed_days_pct"] < 0.30


def test_shap_recovers_true_drivers(trained, tmp_path):
    models, splits = trained
    ranking = explain.explain(models, splits, tmp_path, max_background=2500)
    top6 = list(ranking.head(6)["feature"])

    # The true generative drivers must surface: weekly rhythm, recent history,
    # and holiday/peak proximity.
    assert "day_of_week" in top6, f"day_of_week missing from top drivers: {top6}"
    assert any(f in top6 for f in ["lag_7", "trailing_mean_7", "trailing_mean_28", "lag_14"])
    assert any(
        f in top6 for f in ["days_to_nearest_holiday", "is_peak_ramp", "is_holiday"]
    ), f"holiday proximity missing from top drivers: {top6}"

    # Planted noise stays buried.
    for noise in schema.NOISE_FEATURES:
        share = ranking.loc[ranking["feature"] == noise, "share_of_explanation"].iloc[0]
        assert share < 0.01, f"planted noise feature {noise} got {share:.1%} of the explanation"
