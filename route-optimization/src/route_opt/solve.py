"""Three routing policies and a lower bound, in plain numpy. This file is meant to be read.

All three policies answer the same question — "assign every stop to a truck
and order the visits" — under the same two hard constraints:

- capacity: a truck carries at most ``CAPACITY_PKGS`` packages,
- duration: drive time plus service time, depot back to depot, fits in
  ``MAX_ROUTE_MIN`` (a driver shift).

The ladder, from how depots route today to how a solver would:

- ``zone_fixed``              — the status quo. The metro is pre-cut into
  equal angular wedges, one truck per wedge, stops visited nearest-neighbor
  within the wedge. This is genuinely how many depots run: fixed zones keep
  drivers on familiar turf and make dispatch trivial, and nobody re-cuts the
  wedges when demand shifts.
- ``nearest_neighbor_global`` — the obvious "just be greedy" upgrade: ignore
  zones, send each truck to its nearest unserved stop until full. Included
  because every team tries it first, and because its failure mode (early
  trucks cherry-pick, the last truck inherits scattered leftovers) is worth
  seeing in miles.
- ``savings_2opt``            — the point of this use case. Clarke-Wright
  savings construction, then 2-opt improvement within each route. Both
  pieces are decades old, fit in a page of numpy, and land within a few
  percent of commercial solvers at this scale.

No OR-Tools, no MIP solver, on purpose: the algorithms here are transparent
enough to audit line by line, deterministic, and dependency-free. Distances
are euclidean; see the README for what changes (and what does not) when you
swap in road-network travel times.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Fleet physics. Business constants — recalibrate from your fleet profile.
# ---------------------------------------------------------------------------
CAPACITY_PKGS = 180        # package positions per truck
MAX_ROUTE_MIN = 9 * 60.0   # driver shift: drive + service, depot to depot
SPEED_MPH = 25.0           # blended urban average, stop-to-stop

TWO_OPT_MAX_PASSES = 60    # cap improvement sweeps per route: determinism + bounded runtime


@dataclass
class Solution:
    """Routes for one policy. Each route is a list of stop row-indices, visit order."""

    policy: str
    routes: list[list[int]]
    # savings_2opt also records total miles after Clarke-Wright construction
    # but before 2-opt, so the savings can be attributed between the two.
    construction_miles: float | None = field(default=None)


# ---------------------------------------------------------------------------
# Geometry helpers. Node 0 is the depot (origin); stop i is node i + 1.
# ---------------------------------------------------------------------------
def distance_matrix(stops: pd.DataFrame) -> np.ndarray:
    """Full euclidean matrix over depot + stops, in miles."""
    pts = np.vstack([[0.0, 0.0], stops[["x_mi", "y_mi"]].to_numpy()])
    diff = pts[:, None, :] - pts[None, :, :]
    return np.hypot(diff[..., 0], diff[..., 1])


def route_miles(route: list[int], dist: np.ndarray) -> float:
    """Depot -> stops in order -> depot, in miles."""
    if not route:
        return 0.0
    nodes = np.asarray(route) + 1
    legs = dist[0, nodes[0]] + dist[nodes[-1], 0]
    legs += dist[nodes[:-1], nodes[1:]].sum()
    return float(legs)


def route_minutes(route: list[int], dist: np.ndarray, service_min: np.ndarray) -> float:
    """Total route duration: drive time at SPEED_MPH plus per-stop service."""
    drive = route_miles(route, dist) / SPEED_MPH * 60.0
    return drive + float(service_min[route].sum())


def trucks_needed(stops: pd.DataFrame) -> int:
    """Minimum feasible fleet size, from capacity alone."""
    return ceil(int(stops["packages"].sum()) / CAPACITY_PKGS)


def _feasible(route: list[int], dist, packages, service_min) -> bool:
    return (
        packages[route].sum() <= CAPACITY_PKGS
        and route_minutes(route, dist, service_min) <= MAX_ROUTE_MIN
    )


def _nearest_neighbor_order(idx: np.ndarray, dist: np.ndarray) -> list[int]:
    """Order the given stops by repeated nearest-neighbor, starting from the depot."""
    remaining = list(idx)
    route: list[int] = []
    node = 0  # depot
    while remaining:
        nxt = min(remaining, key=lambda i: dist[node, i + 1])
        route.append(nxt)
        remaining.remove(nxt)
        node = nxt + 1
    return route


# ---------------------------------------------------------------------------
# Policy (a): fixed angular zones — the status quo.
# ---------------------------------------------------------------------------
def zone_fixed(stops: pd.DataFrame, dist: np.ndarray) -> Solution:
    """Fixed angular zones around the depot, one truck per zone, NN inside.

    How the zones are drawn matters for fairness. A naive equal-ANGLE cut is
    a strawman: a wedge that catches a dense subdivision bursts, the depot
    adds trucks, and the optimizer wins by an unbelievable margin. Real zone
    planners are not stupid — they draw zone boundaries once, balanced on
    package volume. So this baseline sweeps the metro by bearing from the
    depot and cuts a new zone each time the running package count reaches an
    equal share, starting from the capacity-minimum truck count and adding a
    truck only if a zone still breaks a constraint.

    What stays deliberately dumb is exactly what stays dumb in real depots:
    the boundaries are FIXED lines on a map. They slice through clusters,
    chain rural stragglers to whatever wedge they fall in, and never react
    to what today's demand actually looks like. That blindness — not the
    zone sizing — is what the optimizer gets to eat.
    """
    packages = stops["packages"].to_numpy()
    service_min = stops["service_min"].to_numpy()
    angles = np.arctan2(stops["y_mi"], stops["x_mi"]).to_numpy() % (2 * np.pi)
    by_angle = np.argsort(angles, kind="stable")  # sweep order around the depot

    k = trucks_needed(stops)
    while k <= len(stops):
        # Cut the angular sweep into k contiguous zones of ~equal packages:
        # a stop joins zone z while the packages BEFORE it are under z+1 shares.
        cum_before = np.cumsum(packages[by_angle]) - packages[by_angle]
        share = packages.sum() / k
        zone = np.minimum(cum_before // share, k - 1).astype(int)
        routes = [
            _nearest_neighbor_order(by_angle[zone == z], dist)
            for z in range(k)
        ]
        routes = [r for r in routes if r]
        if all(_feasible(r, dist, packages, service_min) for r in routes):
            return Solution("zone_fixed", routes)
        k += 1  # a zone broke capacity or the shift: add a truck, redraw
    raise RuntimeError("no feasible fixed-zone plan (constraints too tight for this day)")


# ---------------------------------------------------------------------------
# Policy (b): global nearest-neighbor — greedy construction.
# ---------------------------------------------------------------------------
def nearest_neighbor_global(stops: pd.DataFrame, dist: np.ndarray) -> Solution:
    """Fill trucks nearest-first, ignoring zones.

    A truck leaves the depot, repeatedly drives to the nearest unserved stop
    it can still legally take (capacity, and duration INCLUDING the ride
    home), and returns when nothing fits. Early trucks look great; the miles
    hide in the last trucks, which inherit whatever the greedy sweep left
    scattered across the metro.
    """
    packages = stops["packages"].to_numpy()
    service_min = stops["service_min"].to_numpy()
    unserved = set(range(len(stops)))
    routes: list[list[int]] = []

    while unserved:
        route: list[int] = []
        load, elapsed = 0.0, 0.0
        node = 0  # depot
        while True:
            best, best_d = -1, np.inf
            for i in unserved:
                d = dist[node, i + 1]
                if d >= best_d or load + packages[i] > CAPACITY_PKGS:
                    continue
                # Feasibility must include the ride home, or the last stop
                # of the day strands the driver past the end of shift.
                drive_min = (d + dist[i + 1, 0]) / SPEED_MPH * 60.0
                if elapsed + drive_min + service_min[i] <= MAX_ROUTE_MIN:
                    best, best_d = i, d
            if best < 0:
                break
            route.append(best)
            unserved.remove(best)
            load += packages[best]
            elapsed += best_d / SPEED_MPH * 60.0 + service_min[best]
            node = best + 1
        if not route:
            raise RuntimeError("a lone stop violates the constraints by itself")
        routes.append(route)
    return Solution("nearest_neighbor_global", routes)


# ---------------------------------------------------------------------------
# Policy (c): Clarke-Wright savings + 2-opt — the star.
# ---------------------------------------------------------------------------
def clarke_wright(stops: pd.DataFrame, dist: np.ndarray) -> list[list[int]]:
    """Clarke-Wright parallel savings construction (1964, still the one to beat).

    Start with the worst legal plan: one dedicated round trip per stop.
    Serving i and j on one truck instead of two saves

        s(i, j) = d(depot, i) + d(depot, j) - d(i, j)

    — the two return legs replaced by one direct hop. Sort all pairs by
    savings, descending, and greedily merge: a pair (i, j) links its two
    routes end-to-end iff i and j are both route ENDPOINTS of different
    routes (so each route stays one unbroken path) and the merged route
    still respects capacity and the shift. Big savings get locked in first;
    pairs that would bend a route through the depot's far side never merge.
    """
    n = len(stops)
    packages = stops["packages"].to_numpy()
    service_min = stops["service_min"].to_numpy()

    d0 = dist[0, 1:]  # depot -> each stop
    savings = d0[:, None] + d0[None, :] - dist[1:, 1:]
    iu, ju = np.triu_indices(n, k=1)
    order = np.argsort(-savings[iu, ju], kind="stable")  # stable: deterministic ties
    pairs = np.column_stack([iu[order], ju[order]])

    routes: dict[int, list[int]] = {i: [i] for i in range(n)}  # route-id -> stops
    route_of = np.arange(n)     # stop -> route-id
    load = packages.astype(float).copy()  # per route-id

    for i, j in pairs:
        ri, rj = route_of[i], route_of[j]
        if ri == rj:
            continue
        a, b = routes[ri], routes[rj]
        # Merging keeps each route a single path, so i and j must sit at the
        # ends that get joined: orient a to END with i and b to START with j.
        if a[0] == i:
            a = a[::-1]
        elif a[-1] != i:
            continue  # i is interior; this pair can no longer be linked
        if b[-1] == j:
            b = b[::-1]
        elif b[0] != j:
            continue
        if load[ri] + load[rj] > CAPACITY_PKGS:
            continue
        merged = a + b
        if route_minutes(merged, dist, service_min) > MAX_ROUTE_MIN:
            continue
        routes[ri] = merged
        load[ri] += load[rj]
        route_of[np.asarray(b)] = ri
        del routes[rj]

    return sorted(routes.values(), key=lambda r: r[0])  # stable output order


def two_opt(route: list[int], dist: np.ndarray, max_passes: int = TWO_OPT_MAX_PASSES) -> list[int]:
    """Uncross one route: reverse any segment whose reversal shortens the tour.

    A 2-opt move deletes two edges and reconnects the tour with the segment
    between them reversed. On euclidean instances the classic picture is two
    crossing legs becoming parallel. Only the four touched edges change, so
    each candidate move is an O(1) delta check:

        delta = d(p, B) + d(A, q) - d(p, A) - d(B, q)

    for edges (p, A) and (B, q) with A..B the segment to reverse. Sweeps
    repeat until a full pass finds no improving move, capped at
    ``max_passes`` for bounded, deterministic runtime. First-improvement
    order is fixed, so the result is reproducible. Service time is
    unaffected by visit order, so a shorter tour is never infeasible.
    """
    if len(route) < 3:
        return list(route)
    tour = np.concatenate([[0], np.asarray(route) + 1, [0]])  # depot-padded nodes
    for _ in range(max_passes):
        improved = False
        # a indexes edge (tour[a], tour[a+1]); b indexes edge (tour[b], tour[b+1]).
        for a in range(len(tour) - 3):
            for b in range(a + 2, len(tour) - 1):
                delta = (
                    dist[tour[a], tour[b]]
                    + dist[tour[a + 1], tour[b + 1]]
                    - dist[tour[a], tour[a + 1]]
                    - dist[tour[b], tour[b + 1]]
                )
                if delta < -1e-10:
                    tour[a + 1 : b + 1] = tour[a + 1 : b + 1][::-1]
                    improved = True
        if not improved:
            break
    return list(tour[1:-1] - 1)


def savings_2opt(stops: pd.DataFrame, dist: np.ndarray) -> Solution:
    """Clarke-Wright construction, then 2-opt polish per route."""
    constructed = clarke_wright(stops, dist)
    construction_miles = sum(route_miles(r, dist) for r in constructed)
    routes = [two_opt(r, dist) for r in constructed]
    return Solution("savings_2opt", routes, construction_miles=round(construction_miles, 2))


# ---------------------------------------------------------------------------
# Lower bound — context, not an optimum.
# ---------------------------------------------------------------------------
def lower_bound_miles(stops: pd.DataFrame, dist: np.ndarray) -> float:
    """A degree-based lower bound on total miles for ANY feasible plan.

    In any set of closed routes covering all stops, every stop touches
    exactly 2 route edges, and the depot touches 2 edges per truck. Since
    total length = half the sum, over nodes, of their incident edge lengths,
    replacing each node's actual incident edges with its cheapest possible
    ones can only shrink the total:

        LB = 0.5 * [ sum over stops of (two cheapest links from that stop)
                     + sum of the 2K cheapest depot links ],   K = capacity minimum.

    Honesty note: this is a LOOSE bound. It ignores that the cheap links must
    assemble into consistent tours, ignores the duration constraint entirely,
    and uses the capacity-minimum truck count. The true optimum sits somewhere
    between this number and the best heuristic; the bound's job is to show the
    heuristic is in the right neighborhood, not to certify optimality.
    """
    k = trucks_needed(stops)
    stop_rows = dist[1:].copy()
    np.fill_diagonal(stop_rows[:, 1:], np.inf)  # a stop can't link to itself
    two_cheapest = np.sort(stop_rows, axis=1)[:, :2].sum()
    depot_links = np.sort(dist[0, 1:])[: 2 * k].sum()
    return float(0.5 * (two_cheapest + depot_links))


# Registry, in the order results should be reported.
POLICIES = {
    "zone_fixed": zone_fixed,
    "nearest_neighbor_global": nearest_neighbor_global,
    "savings_2opt": savings_2opt,
}
