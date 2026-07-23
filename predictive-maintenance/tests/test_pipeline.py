"""End-to-end tests on a reduced fleet.

The two tests that carry the most weight:

- `test_models_beat_the_mileage_rule` — the whole business case. If the
  learned models cannot out-rank "service whatever has the most miles on it"
  at the same daily workshop capacity, the pipeline has no reason to exist.
- `test_shap_ranks_real_sensors_over_planted_noise` — the generator documents
  which channels carry wear signal and which are noise, so the explanation
  layer is asserted against ground truth instead of eyeballed.
"""

import numpy as np
import pandas as pd
import pytest

from fleet_maintenance import ai4i, cleaning, evaluate, explain, features, synthetic
from fleet_maintenance import train as train_mod

N_VEHICLES = 300
N_DAYS = 400
# Start in January so the held-out final 90 days include late autumn: the
# battery x cold interaction planted in the generator is then live in test.
START = "2025-01-05"
SEED = 23


@pytest.fixture(scope="session")
def raw():
    return synthetic.make_fleet(
        n_vehicles=N_VEHICLES, n_days=N_DAYS, seed=SEED, start_date=START, messy=True
    )


@pytest.fixture(scope="session")
def clean_df(raw):
    df, _ = cleaning.clean(raw)
    return df


@pytest.fixture(scope="session")
def trained(clean_df):
    cfg = train_mod.TrainConfig(seed=SEED)
    return train_mod.train(clean_df, cfg)


@pytest.fixture(scope="session")
def results(trained, tmp_path_factory):
    models, splits = trained
    return evaluate.evaluate_models(models, splits, tmp_path_factory.mktemp("reports"))


def test_generator_is_deterministic():
    a = synthetic.make_fleet(n_vehicles=40, n_days=80, seed=3, messy=True)
    b = synthetic.make_fleet(n_vehicles=40, n_days=80, seed=3, messy=True)
    pd.testing.assert_frame_equal(a, b)


def test_generator_label_rate_and_bookkeeping(raw):
    # Positives should be rare but present: roughly 2-4% of vehicle-days.
    assert 0.015 < raw[synthetic.LABEL_COL].mean() < 0.06
    # Every failure event names a component; non-failures name none.
    failures = raw[raw[synthetic.EVENT_COL] == 1]
    assert (failures[synthetic.COMPONENT_COL].isin(synthetic.COMPONENTS)).all()
    assert (raw.loc[raw[synthetic.EVENT_COL] == 0, synthetic.COMPONENT_COL] == "").all()


def test_cleaning_dedupes_vehicle_days(raw, clean_df):
    assert raw.duplicated(subset=[synthetic.ID_COL, synthetic.DATE_COL]).any()
    assert not clean_df.duplicated(subset=[synthetic.ID_COL, synthetic.DATE_COL]).any()


def test_cleaning_fixes_negative_mileage(raw, clean_df):
    assert (raw["daily_miles"] < 0).any()
    assert (clean_df["daily_miles"] >= 0).all()
    assert clean_df["daily_miles"].notna().all()


def test_cleaning_detects_frozen_sensor():
    # A transmitter stuck for 10 days must be caught by the zero-variance scan.
    dates = pd.date_range("2025-03-01", periods=30)
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            synthetic.ID_COL: "VEH0001",
            synthetic.DATE_COL: dates,
            **{c: rng.normal(10, 2, 30) for c in cleaning.NUMERIC_COLS},
            "cabin_temp_setting": rng.normal(21, 1, 30),
            "radio_volume_avg": rng.normal(15, 3, 30),
            synthetic.LABEL_COL: 0,
        }
    )
    df.loc[10:19, "vibration_index"] = 2.345  # frozen stretch
    cleaned, report = cleaning.clean(df)
    frozen_step = report.to_frame().set_index("step").loc["frozen_sensor_to_nan[vibration_index]"]
    assert frozen_step["rows_affected"] >= 10
    # The frozen run was replaced (interpolated), so the verbatim repeats are gone.
    assert (cleaned["vibration_index"] == 2.345).sum() <= 2
    assert cleaned["vibration_index__was_missing"].sum() >= 10


def test_cleaning_flags_dropout_as_missingness(raw, clean_df):
    # Dropout days exist in the raw feed and survive as flags, not as NaN.
    assert raw["avg_engine_temp"].isna().any()
    assert clean_df["avg_engine_temp"].notna().all()
    assert clean_df["avg_engine_temp__was_missing"].sum() > 0


def test_time_split_no_leakage(trained):
    _, splits = trained
    assert splits["train"][synthetic.DATE_COL].max() <= splits["test"][synthetic.DATE_COL].min()
    # Ground-truth bookkeeping must never appear in the model matrix.
    assert not set(splits["X_train"].columns) & features.FORBIDDEN


def test_models_beat_the_mileage_rule(results):
    rule, logit, xgb = (results[p] for p in ["mileage_rule", "logistic", "xgboost"])
    # XGBoost must beat the linear baseline on ranking quality: the hazard is
    # exponential in wear and the sensor signal is seasonal-conditional, so a
    # linear model should be genuinely handicapped here.
    assert xgb["pr_auc"] > logit["pr_auc"]
    # Both learned models must beat the status-quo rule at the same daily
    # capacity by a healthy margin, or the project has negative value.
    assert logit["precision_at_capacity"] > 1.5 * rule["precision_at_capacity"]
    assert xgb["precision_at_capacity"] > 2.0 * rule["precision_at_capacity"]
    assert xgb["recall_at_capacity"] > rule["recall_at_capacity"]
    assert xgb["pr_auc"] > 2 * results["base_rate"]


def test_lead_time_is_actionable(results):
    # A warning the shop cannot schedule around is a tow truck with extra steps.
    assert results["xgboost"]["median_lead_days"] >= 3
    assert results["xgboost"]["failures_caught_frac"] > 0.35


def test_money_table_favors_the_model(results):
    assert results["xgboost"]["net_value_vs_mileage_rule_usd"] > 0
    assert results["xgboost"]["net_value_usd"] > results["logistic"]["net_value_usd"] * 0.5


def test_shap_ranks_real_sensors_over_planted_noise(trained, tmp_path):
    models, splits = trained
    ranking = explain.explain(models, splits, tmp_path, max_background=2500)
    order = ranking["driver"].tolist()
    top6 = set(order[:6])

    # At least three of the genuine wear channels must rank in the top six.
    assert len(top6 & set(explain.SIGNAL_GROUPS)) >= 3, f"top drivers were: {order[:6]}"
    # The planted noise channels must sit in the bottom half with trivial share.
    for noise in explain.NOISE_GROUPS:
        pos = order.index(noise)
        assert pos >= len(order) // 2, f"{noise} ranked too high: {order}"
        share = ranking.loc[ranking["driver"] == noise, "share_of_explanation"].iloc[0]
        assert share < 0.02


def test_work_order_card(trained, tmp_path):
    models, splits = trained
    path = explain.write_work_order(models, splits, tmp_path / "work_order_card.md")
    text = path.read_text()
    assert text.startswith("# Work order: VEH")
    assert "Predicted 14-day breakdown risk" in text
    assert "Schedule bay time" in text
    assert "consistent with" in text


# ---------------------------------------------------------------------------
# Real data: UCI AI4I 2020 (public_data/ai4i2020.csv is committed, CC BY 4.0,
# so these run everywhere CI runs — no download, no skipif).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def ai4i_df():
    return ai4i.load()


@pytest.fixture(scope="session")
def ai4i_fitted(ai4i_df):
    train_df, test_df = ai4i.stratified_split(ai4i_df, seed=SEED)
    X_train, y_train = ai4i.to_xy(train_df)
    X_test, y_test = ai4i.to_xy(test_df)
    _, xgb = ai4i.train_models(X_train, y_train, seed=SEED)
    return X_test, y_test, xgb


def test_ai4i_loader_shape_and_label_rate(ai4i_df):
    assert len(ai4i_df) == 10_000
    # Published base rate: 339 failures out of 10,000 (~3.4%).
    assert ai4i_df[ai4i.LABEL_COL].sum() == 339
    assert set(ai4i.FEATURE_COLS) <= set(ai4i_df.columns)
    # One-hot covers every record exactly once.
    assert (ai4i_df[ai4i.TYPE_DUMMIES].sum(axis=1) == 1).all()


def test_ai4i_failure_mode_columns_never_reach_the_matrix(ai4i_df):
    # THE AI4I trap: TWF/HDF/PWF/OSF/RNF are components of the label. Any
    # matrix containing them scores ~99% by reading the answer key.
    assert not set(ai4i_df.columns) & set(ai4i.FAILURE_MODE_COLS)
    X, _ = ai4i.to_xy(ai4i_df)
    assert not set(X.columns) & set(ai4i.FAILURE_MODE_COLS)
    assert ai4i.LABEL_COL not in X.columns
    # Belt and braces: even if someone concatenates the raw columns back on,
    # to_xy must not emit them.
    polluted = ai4i_df.copy()
    for c in ai4i.FAILURE_MODE_COLS:
        polluted[c] = 1
    X2, _ = ai4i.to_xy(polluted)
    assert not set(X2.columns) & set(ai4i.FAILURE_MODE_COLS)


def test_ai4i_split_is_stratified_and_disjoint(ai4i_df):
    train_df, test_df = ai4i.stratified_split(ai4i_df, test_frac=0.25, seed=SEED)
    assert len(train_df) + len(test_df) == len(ai4i_df)
    assert not set(train_df.index) & set(test_df.index)
    # Stratification keeps the rare-positive rate stable on both sides.
    base = ai4i_df[ai4i.LABEL_COL].mean()
    assert abs(test_df[ai4i.LABEL_COL].mean() - base) < 0.005


def test_ai4i_xgboost_pr_auc_beats_5x_base_rate(ai4i_fitted):
    from sklearn.metrics import average_precision_score

    X_test, y_test, xgb = ai4i_fitted
    pr_auc = average_precision_score(y_test, xgb.predict_proba(X_test)[:, 1])
    base_rate = y_test.mean()
    # Honest features only. With the leaked mode columns this would be ~1.0;
    # the bar here is a real ranking win, not the answer key.
    assert pr_auc > 5 * base_rate, f"PR-AUC {pr_auc:.3f} vs base rate {base_rate:.3%}"


def test_ai4i_shap_top3_matches_documented_physics(ai4i_fitted):
    import shap

    X_test, _, xgb = ai4i_fitted
    shap_values = shap.TreeExplainer(xgb).shap_values(X_test)
    mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=X_test.columns)
    grouped = mean_abs.groupby(mean_abs.index.map(lambda c: ai4i.GROUPS.get(c, c))).sum()
    top3 = set(grouped.sort_values(ascending=False).head(3).index)
    # The documented failure modes run on torque (PWF, OSF), tool wear
    # (TWF, OSF) and the air/process temperature pair (HDF), so at least one
    # of those physical drivers must sit in the SHAP top 3.
    physics_drivers = {"torque", "tool wear", "air temperature", "process temperature"}
    assert top3 & physics_drivers, f"top-3 drivers were: {sorted(top3)}"
