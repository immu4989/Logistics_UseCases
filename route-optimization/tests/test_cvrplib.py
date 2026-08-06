"""CVRPLIB benchmark plumbing, tested offline on one committed instance.

tests/fixtures/{A-n32-k5,A-n65-k9}.{vrp,sol} come from CVRPLIB (see the
attribution headers in the files). Everything here runs without network; the
full multi-instance benchmark is exercised only when the download cache
already exists.
"""

from pathlib import Path

import numpy as np
import pytest

from route_opt import bench, cvrplib

DATA = Path(__file__).parent / "fixtures"
BKS_A32 = cvrplib.REGISTRY["A-n32-k5"]["bks"]  # 784, proven optimal


@pytest.fixture(scope="module")
def a32():
    return cvrplib.parse_vrp((DATA / "A-n32-k5.vrp").read_text())


# --------------------------------------------------------------------------
# Parser round-trip on the committed instance
# --------------------------------------------------------------------------
def test_parse_committed_instance(a32):
    assert a32.name == "A-n32-k5"
    assert a32.capacity == 100
    assert a32.n_customers == 31
    assert a32.coords.shape == (32, 2)
    assert a32.demands.shape == (32,)
    assert a32.demands[0] == 0  # depot
    assert a32.coords[0].tolist() == [82.0, 76.0]  # depot is .vrp node 1
    assert a32.demands[1:].min() > 0


def test_tsplib_euc2d_rounding(a32):
    # TSPLIB EUC_2D = nint(sqrt(.)): hand-computed distances, not floats.
    # Depot (82,76) -> node 2 (96,44): sqrt(14^2 + 32^2) = sqrt(1220) = 34.928... -> 35
    # Depot (82,76) -> node 3 (50,5):  sqrt(32^2 + 71^2) = sqrt(6065) = 77.878... -> 78
    dist = cvrplib.euc_2d_matrix(a32)
    assert dist[0, 1] == 35
    assert dist[0, 2] == 78
    assert dist[1, 2] == round(np.hypot(96 - 50, 44 - 5))  # 60.407... -> 60
    assert np.all(dist == dist.T)
    assert np.all(dist.diagonal() == 0)
    # integer-valued everywhere: BKS costs assume integer edge lengths
    assert np.all(dist == np.floor(dist))


# --------------------------------------------------------------------------
# Best-known solution: valid, and its cost reproduces the published BKS
# --------------------------------------------------------------------------
def test_bks_solution_valid_and_cost_reproduced(a32):
    routes, stated_cost = cvrplib.parse_sol((DATA / "A-n32-k5.sol").read_text())
    assert stated_cost == BKS_A32
    # every customer exactly once
    visited = sorted(stop for route in routes for stop in route)
    assert visited == list(range(a32.n_customers))
    # capacity respected on every route
    demands = a32.demands[1:]
    assert all(demands[route].sum() <= a32.capacity for route in routes)
    # recomputing the cost under our rounding must hit the BKS exactly —
    # this pins the .sol numbering convention AND the distance convention
    dist = cvrplib.euc_2d_matrix(a32)
    assert cvrplib.solution_cost(routes, dist) == BKS_A32


# --------------------------------------------------------------------------
# Our solver on the committed instance
# --------------------------------------------------------------------------
def test_solver_valid_and_within_15pct_of_bks(a32):
    routes, cost = cvrplib.solve_instance(a32)
    visited = sorted(stop for route in routes for stop in route)
    assert visited == list(range(a32.n_customers))
    demands = a32.demands[1:]
    assert all(demands[route].sum() <= a32.capacity for route in routes)
    assert cost >= BKS_A32  # nobody beats a proven optimum
    assert (cost - BKS_A32) / BKS_A32 <= 0.15


def test_solver_is_deterministic_on_instance(a32):
    routes_a, cost_a = cvrplib.solve_instance(a32)
    routes_b, cost_b = cvrplib.solve_instance(a32)
    assert routes_a == routes_b
    assert cost_a == cost_b


# --------------------------------------------------------------------------
# savings_ls (local-search stack + ILS) on the committed instance
# --------------------------------------------------------------------------
def test_savings_ls_valid_and_within_4pct_of_bks(a32):
    routes, cost = cvrplib.solve_instance_ls(a32)
    visited = sorted(stop for route in routes for stop in route)
    assert visited == list(range(a32.n_customers))
    demands = a32.demands[1:]
    assert all(demands[route].sum() <= a32.capacity for route in routes)
    assert cost >= BKS_A32  # nobody beats a proven optimum
    assert (cost - BKS_A32) / BKS_A32 <= 0.04
    # and it can never lose to the plan it starts from
    _, cost_2opt = cvrplib.solve_instance(a32)
    assert cost <= cost_2opt


def test_savings_ls_is_byte_deterministic(a32):
    # Seeded RNG + fixed scan orders + fixed budgets: two runs must produce
    # the identical plan, route for route, stop for stop.
    routes_a, cost_a = cvrplib.solve_instance_ls(a32)
    routes_b, cost_b = cvrplib.solve_instance_ls(a32)
    assert routes_a == routes_b
    assert cost_a == cost_b


def test_mean_gap_on_committed_fixtures():
    # Network-free gap floor: the two committed instances (easiest and the
    # old worst case) must average under the README's claimed 3% mean.
    results = bench.run_bench(["A-n32-k5", "A-n65-k9"], cache_dir=DATA)
    assert (results["ls_gap_pct"] >= 0).all()
    assert (results["ls_gap_pct"] <= results["gap_pct"]).all()
    assert results["ls_gap_pct"].mean() <= 3.0


# --------------------------------------------------------------------------
# Full bench — only when the network cache is already populated
# --------------------------------------------------------------------------
@pytest.mark.skipif(
    not all(
        (cvrplib.DEFAULT_CACHE_DIR / f"{name}{ext}").exists()
        for name in cvrplib.REGISTRY
        for ext in (".vrp", ".sol")
    ),
    reason="CVRPLIB cache not downloaded (run: route-opt bench)",
)
def test_full_bench_on_cached_instances():
    results = bench.run_bench()  # run_bench validity-checks every plan
    assert set(results["instance"]) == set(cvrplib.REGISTRY)
    assert (results["gap_pct"] >= 0).all()  # BKS are proven optima for these sets
    assert (results["gap_pct"] <= 15).all()
    # the README's savings_ls claims, asserted: mean <= 3%, worst <= 5%,
    # never worse than the savings_2opt plan it starts from
    assert (results["ls_gap_pct"] >= 0).all()
    assert (results["ls_gap_pct"] <= results["gap_pct"]).all()
    assert results["ls_gap_pct"].mean() <= 3.0
    assert results["ls_gap_pct"].max() <= 5.0
