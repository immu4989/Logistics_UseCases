"""End-to-end tests on the full synthetic pipeline, run on two seeds.

The most important tests here are the cost-ordering ones: because the
generator's demand distribution is known, every booking policy can be costed
against ground truth, and the whole point of the use case — the critical
fractile beats both the planner habit and the book-the-mean reading of the
same forecast — is asserted in CI rather than eyeballed. Agent-style demos
love a single lucky run; these assertions hold on multiple seeds because the
evaluation is paired (common random numbers), not because the dice were kind.
"""

import numpy as np
import pandas as pd
import pytest

from capacity_planning import cleaning, decide, evaluate, forecast, synthetic

N_REPS = 200


@pytest.fixture(scope="session", params=[7, 11])
def seed(request):
    return request.param


@pytest.fixture(scope="session")
def raw(seed):
    return synthetic.make_dataset(seed=seed, messy=True)


@pytest.fixture(scope="session")
def clean_df(raw):
    df, _ = cleaning.clean(raw)
    return df


@pytest.fixture(scope="session")
def trained(clean_df, seed):
    return forecast.train(clean_df, forecast.TrainConfig(seed=seed))


@pytest.fixture(scope="session")
def bookings(trained):
    models, splits = trained
    return evaluate.build_bookings(models, splits)


@pytest.fixture(scope="session")
def demand(bookings, seed):
    return synthetic.simulate_demand(bookings, seed, N_REPS)


def _mean_cost(bookings, demand, col, spot=decide.SPOT_COST_USD):
    comps = decide.cost_components(bookings[col].to_numpy(), demand, spot_cost=spot)
    return float(np.mean(comps["total_cost"]))


# ---------------------------------------------------------------------------
# Generator + cleaning
# ---------------------------------------------------------------------------

def test_generator_is_deterministic():
    a = synthetic.make_dataset(n_weeks=60, seed=3, messy=True)
    b = synthetic.make_dataset(n_weeks=60, seed=3, messy=True)
    pd.testing.assert_frame_equal(a, b)


def test_mess_is_injected_after_the_truth(raw, seed):
    """messy=False is the exact uncorrupted truth behind the messy feed."""
    truth = synthetic.make_dataset(seed=seed, messy=False)
    merged = raw[raw[synthetic.TARGET_COL] >= 0].merge(
        truth, on=[synthetic.LANE_COL, synthetic.WEEK_COL], suffixes=("_messy", "_true")
    )
    assert np.allclose(
        merged[f"{synthetic.TARGET_COL}_messy"], merged[f"{synthetic.TARGET_COL}_true"]
    )


def test_cleaning_handles_every_mess_class(raw, clean_df):
    _, report = cleaning.clean(raw)
    touched = {s["step"]: s["rows_affected"] for s in report.steps}
    assert touched["drop_duplicate_lane_weeks"] > 0
    assert touched["drop_negative_demand"] > 0
    assert touched["calendar_gaps_left_as_gaps"] > 0
    assert not clean_df.duplicated([synthetic.LANE_COL, synthetic.WEEK_COL]).any()
    assert (clean_df[synthetic.TARGET_COL] >= 0).all()


# ---------------------------------------------------------------------------
# Forecast layer
# ---------------------------------------------------------------------------

def test_time_split_and_peak_in_test(trained):
    _, splits = trained
    assert splits["train"][synthetic.WEEK_COL].max() < splits["test"][synthetic.WEEK_COL].min()
    test_woy = set(
        pd.DatetimeIndex(splits["test"][synthetic.WEEK_COL]).isocalendar().week.astype(int)
    )
    assert set(synthetic.PEAK_WOY) <= test_woy, "the year-end peak must be in the test window"


def test_quantile_forecast_beats_naive_at_qstar(trained):
    models, splits = trained
    alpha = forecast.QUANTILE_ROLES["q_base"]
    q = forecast.predict_quantiles(models, splits["X_test"])
    pb_model = forecast.pinball_loss(splits["y_test"], q["q_base"].to_numpy(), alpha)
    pb_naive = forecast.pinball_loss(splits["y_test"], models.naive.predict(splits["test"]), alpha)
    assert pb_model < 0.9 * pb_naive


def test_quantiles_are_monotone(trained):
    models, splits = trained
    q = forecast.predict_quantiles(models, splits["X_test"])
    assert (q["q_base"] <= q["p50"]).all()
    assert (q["p50"] <= q["q_tight"]).all()


def test_model_roundtrip(trained, tmp_path):
    models, splits = trained
    forecast.save(models, tmp_path)
    loaded = forecast.load(tmp_path)
    a = forecast.predict_quantiles(models, splits["X_test"])
    b = forecast.predict_quantiles(loaded, splits["X_test"])
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------------------
# The decision: newsvendor economics
# ---------------------------------------------------------------------------

def test_critical_fractile_arithmetic():
    # Cu = 2300 - 1400 = 900; Co = 1400 - 350 = 1050; q* = 900 / 1950.
    assert decide.critical_fractile() == pytest.approx(900 / 1950)
    assert decide.critical_fractile(3_200) == pytest.approx(1800 / 2850)


def test_policy_cost_ordering(bookings, demand):
    """The star assertion: fractile < mean < habit, and the oracle floors it."""
    costs = {
        name: _mean_cost(bookings, demand, col) for name, col in decide.POLICY_COLUMNS.items()
    }
    assert costs["oracle"] <= costs["newsvendor_model"]
    assert costs["newsvendor_model"] < costs["book_mean"]
    assert costs["book_mean"] < costs["book_last_year"]
    # A healthy, deployment-worthy margin over the habit, not a rounding win.
    assert costs["newsvendor_model"] < 0.985 * costs["book_last_year"]


def test_service_level_sane_band(bookings, demand):
    """The savings must not come from quietly buying service via overbooking."""
    nv = decide.cost_components(bookings["booked_newsvendor"].to_numpy(), demand)
    mean = decide.cost_components(bookings["booked_mean"].to_numpy(), demand)
    sl_nv = float(np.mean(nv["service_level"]))
    assert 0.85 < sl_nv < 0.97
    # Booking below the median must show up as slightly LESS committed service
    # than book-the-mean, not more — the fractile trades service for cost.
    assert sl_nv <= float(np.mean(mean["service_level"]))
    assert nv["booked_trailers"] <= mean["booked_trailers"]


def test_sensitivity_tight_market(bookings, demand):
    """Reprice the spot market and the fractile — not the habit — adapts."""
    assert decide.critical_fractile(decide.SPOT_COST_TIGHT_USD) > decide.critical_fractile()
    assert bookings["booked_newsvendor_tight"].sum() > bookings["booked_newsvendor"].sum()
    stale = _mean_cost(bookings, demand, "booked_newsvendor", spot=decide.SPOT_COST_TIGHT_USD)
    retuned = _mean_cost(
        bookings, demand, "booked_newsvendor_tight", spot=decide.SPOT_COST_TIGHT_USD
    )
    assert retuned < stale


def test_crn_reproducibility(bookings, seed, tmp_path):
    """Same seed, same draws, same tables — the comparison is paired by design."""
    d1 = synthetic.simulate_demand(bookings, seed, 50)
    d2 = synthetic.simulate_demand(bookings, seed, 50)
    np.testing.assert_array_equal(d1, d2)
    c1, s1 = evaluate.evaluate_all(bookings, seed=seed, out_dir=tmp_path / "a", n_reps=50)
    c2, s2 = evaluate.evaluate_all(bookings, seed=seed, out_dir=tmp_path / "b", n_reps=50)
    pd.testing.assert_frame_equal(c1, c2)
    pd.testing.assert_frame_equal(s1, s2)
