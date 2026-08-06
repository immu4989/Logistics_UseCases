"""End-to-end tests for the route optimizer.

The load-bearing assertions are the constraint checks (every policy serves
every stop exactly once, never overloads a truck, never runs past the shift)
and the policy ordering: on the same cleaned day, savings_2opt must beat both
the fixed-zone status quo and global nearest-neighbor by a real margin, and
savings_ls (the star: + or-opt/2-opt*/swap local search + seeded ILS) must
never lose to savings_2opt, on the default seed and an alternate. NN vs
zones is deliberately NOT ordered —
the balanced-zone baseline and greedy NN trade places seed to seed (verified
across seeds 3/7/11/42/99), and that closeness is part of the story.
"""

import pandas as pd
import pytest

from route_opt import cleaning, evaluate, explain, solve, synthetic

N = 600
SEED = 7
ALT_SEED = 11


def _solve_day(seed):
    stops, _ = cleaning.clean(synthetic.make_day(n=N, seed=seed, messy=True))
    dist = solve.distance_matrix(stops)
    solutions = {name: fn(stops, dist) for name, fn in solve.POLICIES.items()}
    return stops, dist, solutions


@pytest.fixture(scope="session")
def day():
    stops, dist, solutions = _solve_day(SEED)
    return stops, dist, solutions


# --------------------------------------------------------------------------
# Generator + cleaning
# --------------------------------------------------------------------------
def test_generator_is_deterministic():
    a = synthetic.make_day(n=400, seed=3)
    b = synthetic.make_day(n=400, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_generator_plants_each_mess_class():
    raw = synthetic.make_day(n=N, seed=SEED, messy=True)
    assert raw["stop_id"].duplicated().any()
    assert ((raw["x_mi"] == 0.0) & (raw["y_mi"] == 0.0)).any()
    assert (raw["x_mi"].abs() > synthetic.METRO_HALF_MI).any()
    assert (raw["packages"] < 0).any()


def test_cleaning_fixes_each_mess_class():
    raw = synthetic.make_day(n=N, seed=SEED, messy=True)
    stops, report = cleaning.clean(raw)
    assert stops["stop_id"].is_unique
    assert not ((stops["x_mi"] == 0.0) & (stops["y_mi"] == 0.0)).any()
    assert stops["x_mi"].abs().max() <= synthetic.METRO_HALF_MI
    assert stops["y_mi"].abs().max() <= synthetic.METRO_HALF_MI
    assert stops["packages"].between(synthetic.MIN_PACKAGES, synthetic.MAX_PACKAGES).all()
    # Every fix is audited, and each planted fault actually exercised its step.
    steps = report.to_frame().set_index("step")["rows_affected"]
    for step in [
        "drop_duplicate_stop_ids",
        "drop_geocode_null_island",
        "drop_geocode_out_of_metro",
        "clip_impossible_package_counts",
    ]:
        assert steps[step] > 0, step


def test_clean_input_passes_through_untouched():
    raw = synthetic.make_day(n=300, seed=5, messy=False)
    stops, report = cleaning.clean(raw)
    assert len(stops) == len(raw)
    assert report.to_frame()["rows_affected"].sum() == 0


# --------------------------------------------------------------------------
# Constraints: every policy, every route
# --------------------------------------------------------------------------
def test_every_stop_served_exactly_once(day):
    stops, _, solutions = day
    for name, sol in solutions.items():
        visited = sorted(i for r in sol.routes for i in r)
        assert visited == list(range(len(stops))), name


def test_capacity_never_exceeded(day):
    stops, _, solutions = day
    packages = stops["packages"].to_numpy()
    for name, sol in solutions.items():
        for route in sol.routes:
            assert packages[route].sum() <= solve.CAPACITY_PKGS, name


def test_shift_duration_respected(day):
    stops, dist, solutions = day
    service = stops["service_min"].to_numpy()
    for name, sol in solutions.items():
        for route in sol.routes:
            assert solve.route_minutes(route, dist, service) <= solve.MAX_ROUTE_MIN + 1e-6, name


def test_solvers_are_deterministic():
    _, _, a = _solve_day(SEED)
    _, _, b = _solve_day(SEED)
    for name in solve.POLICIES:
        assert a[name].routes == b[name].routes, name


# --------------------------------------------------------------------------
# Policy ordering and the algorithm's internals
# --------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [SEED, ALT_SEED])
def test_savings_2opt_beats_both_baselines_by_margin(seed):
    # NN-global and the balanced-zone baseline trade places across seeds, so
    # only the optimizer's win over BOTH is asserted, with a >5% margin
    # (verified margins: >=8% vs zones, >=14% vs NN on seeds 3/7/11/42/99).
    _, dist, solutions = _solve_day(seed)
    miles = {
        name: sum(solve.route_miles(r, dist) for r in sol.routes)
        for name, sol in solutions.items()
    }
    assert miles["savings_2opt"] < 0.95 * miles["zone_fixed"]
    assert miles["savings_2opt"] < 0.95 * miles["nearest_neighbor_global"]


@pytest.mark.parametrize("seed", [SEED, ALT_SEED])
def test_savings_ls_never_worse_than_savings_2opt(seed):
    # The local-search stack + ILS starts FROM the savings_2opt plan and only
    # ever accepts strict improvements, so it can never lose to it.
    _, dist, solutions = _solve_day(seed)
    miles = {
        name: sum(solve.route_miles(r, dist) for r in sol.routes)
        for name, sol in solutions.items()
    }
    assert miles["savings_ls"] <= miles["savings_2opt"] + 1e-6
    # and it inherits (at least) the old policy's win over both baselines
    assert miles["savings_ls"] < 0.95 * miles["zone_fixed"]
    assert miles["savings_ls"] < 0.95 * miles["nearest_neighbor_global"]


def test_local_search_preserves_feasibility_and_miles(day):
    # Every operator in the stack (or-opt, 2-opt*, swap, intra 2-opt) must
    # keep the plan legal: each stop exactly once, capacity and shift
    # respected on every route — and total miles must never increase.
    stops, dist, _ = day
    packages = stops["packages"].to_numpy(dtype=float)
    service = stops["service_min"].to_numpy(dtype=float)
    before = solve.clarke_wright(stops, dist)
    before_miles = sum(solve.route_miles(r, dist) for r in before)
    after = solve.local_search(before, dist, packages, service)
    visited = sorted(i for r in after for i in r)
    assert visited == list(range(len(stops)))
    for route in after:
        assert packages[route].sum() <= solve.CAPACITY_PKGS
        assert solve.route_minutes(route, dist, service) <= solve.MAX_ROUTE_MIN + 1e-6
    assert sum(solve.route_miles(r, dist) for r in after) <= before_miles + 1e-9


def test_savings_ls_stage_miles_recorded_and_monotone(day):
    # Each stage may only improve on the last: construction >= 2-opt >=
    # local-search stack >= ILS best == the returned plan.
    _, dist, solutions = day
    star = solutions["savings_ls"]
    s = star.stage_miles
    assert s is not None
    assert s["construction"] >= s["two_opt"] >= s["local_search"] >= s["ils"]
    final = sum(solve.route_miles(r, dist) for r in star.routes)
    assert final == pytest.approx(s["ils"], abs=0.01)


def test_two_opt_never_increases_route_length(day):
    stops, dist, _ = day
    for route in solve.clarke_wright(stops, dist):
        before = solve.route_miles(route, dist)
        after_route = solve.two_opt(route, dist)
        assert solve.route_miles(after_route, dist) <= before + 1e-9
        assert sorted(after_route) == sorted(route)  # a reorder, never a re-assignment


def test_lower_bound_is_below_best_heuristic(day):
    stops, dist, solutions = day
    best = sum(solve.route_miles(r, dist) for r in solutions["savings_ls"].routes)
    lb = solve.lower_bound_miles(stops, dist)
    assert 0 < lb <= best


def test_construction_miles_recorded_and_2opt_helped(day):
    _, dist, solutions = day
    star = solutions["savings_2opt"]
    final = sum(solve.route_miles(r, dist) for r in star.routes)
    assert star.construction_miles is not None
    assert final <= star.construction_miles + 1e-6


# --------------------------------------------------------------------------
# Evaluation + explanation artifacts
# --------------------------------------------------------------------------
def test_evaluate_writes_artifacts_and_sane_economics(day, tmp_path):
    stops, dist, solutions = day
    table = evaluate.evaluate_all(stops, solutions, dist, out_dir=tmp_path)
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "routes.csv").exists()
    for png in ["route_maps.png", "miles_by_policy.png", "stops_per_truck_hour.png"]:
        assert (tmp_path / png).exists(), png

    t = table.set_index("policy")
    assert t.loc["zone_fixed", "daily_savings_vs_zone_usd"] == 0.0
    assert t.loc["savings_2opt", "daily_savings_vs_zone_usd"] > 0
    # the star must be at least as good as the policy it extends
    assert (
        t.loc["savings_ls", "daily_savings_vs_zone_usd"]
        >= t.loc["savings_2opt", "daily_savings_vs_zone_usd"]
    )
    assert t.loc["savings_ls", "annual_savings_vs_zone_usd"] == pytest.approx(
        t.loc["savings_ls", "daily_savings_vs_zone_usd"] * evaluate.OPERATING_DAYS_PER_YEAR,
        abs=0.5,  # the annual figure is rounded to whole dollars
    )
    assert (t["max_route_hours"] <= solve.MAX_ROUTE_MIN / 60.0 + 1e-6).all()

    routes = pd.read_csv(tmp_path / "routes.csv")
    for name in solve.POLICIES:  # every policy's full plan is in the CSV
        assert (routes["policy"] == name).sum() == len(stops)


def test_rationale_written_and_checkable(day, tmp_path):
    stops, dist, solutions = day
    attribution = explain.write_rationale(stops, solutions, dist, tmp_path)
    text = (tmp_path / "rationale.md").read_text()
    assert "Cum mi" in text and "DEPOT" in text
    assert "ILS" in text  # the attribution now covers the full solver stack
    # The attribution must reconcile: stage savings sum to zone minus final.
    zone = attribution["total_miles"].iloc[0]
    final = attribution["total_miles"].iloc[-1]
    assert attribution["miles_saved"].sum() == pytest.approx(zone - final, abs=0.2)


def test_dispatch_sheet_arithmetic(day):
    stops, dist, solutions = day
    route = solutions["savings_ls"].routes[0]
    sheet = explain.dispatch_sheet(route, stops, dist)
    assert len(sheet) == len(route) + 1  # + the ride home
    assert sheet["cum_load"].iloc[-1] == stops["packages"].to_numpy()[route].sum()
    assert sheet["cum_mi"].iloc[-1] == pytest.approx(solve.route_miles(route, dist), abs=0.2)
    assert sheet["cum_load"].is_monotonic_increasing
