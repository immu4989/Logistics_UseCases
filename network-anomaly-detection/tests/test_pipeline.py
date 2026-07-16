"""End-to-end tests against the generator's injected ground truth.

The load-bearing tests are the detection ones: because INJECTED_ANOMALIES is a
documented module constant, we can assert the CUSUM catches step drifts fast,
beats the monthly report on delay, stays under the false-alarm budget, and
does not page the network during a global surge. If a refactor quietly breaks
the shrinkage or the global-effect removal, these fail.
"""

import numpy as np
import pandas as pd
import pytest

from network_anomaly import cleaning, detect, evaluate, schema, synthetic


@pytest.fixture(scope="session")
def raw():
    return synthetic.make_dataset(seed=7, messy=True)


@pytest.fixture(scope="session")
def clean_df(raw):
    df, _ = cleaning.clean(raw)
    return df


@pytest.fixture(scope="session")
def result(clean_df):
    return detect.detect(clean_df)


@pytest.fixture(scope="session")
def scored(result):
    return evaluate.score(result)


def test_generator_is_deterministic():
    a = synthetic.make_dataset(seed=3, messy=True)
    b = synthetic.make_dataset(seed=3, messy=True)
    pd.testing.assert_frame_equal(a, b)


def test_generator_schema_and_network_shape():
    df = synthetic.make_dataset(seed=1)
    schema.validate(df)
    assert df[schema.LANE_COL].nunique() == synthetic.N_LANES
    assert len(df) == synthetic.N_LANES * synthetic.N_DAYS
    vols = df.groupby(schema.LANE_COL, observed=True)[schema.VOLUME_COL].mean()
    assert vols.max() > 1000, "network should have trunk lanes"
    assert (vols < 30).sum() > 40, "network should have many thin lanes"


def test_injected_step_is_really_in_the_data():
    """The anomaly list must match the data: post-onset rate visibly elevated."""
    df = synthetic.make_dataset(seed=7)
    df["day"] = (df[schema.DATE_COL] - df[schema.DATE_COL].min()).dt.days
    step = max(
        (a for a in synthetic.INJECTED_ANOMALIES if a["type"] == "step"),
        key=lambda a: a["magnitude"],
    )
    lane = df[df[schema.LANE_COL] == step["lane"]]
    pre = lane[lane["day"] < step["start_day"]]
    post = lane[lane["day"] >= step["start_day"]]
    rate = lambda part: part[schema.MISSES_COL].sum() / part[schema.VOLUME_COL].sum()  # noqa: E731
    assert rate(post) - rate(pre) > step["magnitude"] * 0.6


def test_cleaning_fixes_each_mess_class(raw, clean_df):
    # duplicates gone
    assert not clean_df.duplicated(subset=[schema.LANE_COL, schema.DATE_COL]).any()
    # impossible counts gone (raw had them)
    assert (raw[schema.MISSES_COL] > raw[schema.VOLUME_COL]).any()
    assert (clean_df[schema.MISSES_COL] <= clean_df[schema.VOLUME_COL]).all()
    # gaps preserved, not filled
    n_days = clean_df[schema.DATE_COL].nunique()
    assert len(clean_df) < clean_df[schema.LANE_COL].nunique() * n_days


def test_cusum_detects_steps_fast_and_beats_monthly(scored):
    metrics, _, delays = scored
    s = metrics["steps"]
    assert s["detection_rate"] >= 0.80, f"step detection too low: {s}"
    assert s["mean_delay_days"] < 21, f"step delay too high: {s}"
    assert s["mean_delay_days"] < s["monthly_mean_delay_days"], (
        "CUSUM must beat the monthly report on mean step delay"
    )


def test_ramps_are_caught(scored):
    metrics, _, _ = scored
    assert metrics["ramps"]["detection_rate"] >= 0.75


def test_false_alarm_budget(scored):
    metrics, _, _ = scored
    assert metrics["false_alarms"]["per_clean_lane_year"] < 1.5


def test_global_surge_does_not_page_the_network(scored):
    metrics, _, _ = scored
    assert metrics["surge_check"]["fraction_of_clean_lanes"] <= 0.05


def test_alarms_schema_sane(result, scored, tmp_path):
    metrics, annotated, _ = evaluate.evaluate(result, tmp_path)
    alarms = pd.read_csv(tmp_path / "alarms.csv")
    assert list(alarms.columns) == [
        "lane", "day", "date", "cusum", "expected_rate",
        "status", "anomaly_type", "days_since_onset",
    ]
    assert (alarms["cusum"] >= result.config.h).all()
    assert alarms["day"].between(result.config.baseline_days, len(result.dates) - 1).all()
    assert set(alarms["status"]) <= {"detection", "spike_window", "false_alarm"}
    assert (tmp_path / "metrics.json").exists()
    for png in ["example_step_lane.png", "example_ramp_lane.png",
                "example_clean_lane.png", "detection_delay_comparison.png",
                "cusum_heatmap.png"]:
        assert (tmp_path / png).exists()


def test_nothing_is_knife_edge_on_another_seed():
    """Re-roll the stochastic realization (same network, same anomalies)."""
    df, _ = cleaning.clean(synthetic.make_dataset(seed=23, messy=True))
    metrics, _, _ = evaluate.score(detect.detect(df))
    assert metrics["steps"]["detection_rate"] >= 0.75
    assert metrics["steps"]["mean_delay_days"] < metrics["steps"]["monthly_mean_delay_days"]
    assert metrics["false_alarms"]["per_clean_lane_year"] < 1.5
    assert metrics["surge_check"]["fraction_of_clean_lanes"] <= 0.05


def test_detector_tolerates_gaps(clean_df):
    """Punch a 3-week hole in a trunk lane; the detector must not crash or alarm on it."""
    lane = max(synthetic.clean_lanes(), key=lambda ln: synthetic.LANE_BASE_VOLUME[ln])
    df = clean_df.copy()
    df["day"] = (df[schema.DATE_COL] - df[schema.DATE_COL].min()).dt.days
    hole = (df[schema.LANE_COL] == lane) & df["day"].between(150, 170)
    df = df[~hole].drop(columns="day")
    res = detect.detect(df)
    assert lane not in set(res.alarms["lane"])


def test_beta_prior_moments():
    rng = np.random.default_rng(0)
    rates = rng.beta(4, 46, 500)
    a, b = detect.fit_beta_prior(rates)
    assert 2.5 < a < 6.5
    assert 30 < b < 65
