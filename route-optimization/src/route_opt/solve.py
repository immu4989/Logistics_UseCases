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
- ``savings_2opt``            — the classic pairing. Clarke-Wright
  savings construction, then 2-opt improvement within each route. Both
  pieces are decades old, fit in a page of numpy, and land within a few
  percent of commercial solvers at this scale.
- ``savings_ls``              — the star. The same Clarke-Wright + 2-opt
  start, then a proper local-search stack (inter-route 2-opt*, or-opt
  segment relocation, inter-route swap) driven to a local optimum, then a
  seeded iterated local search (ILS) that perturbs and re-optimizes for a
  fixed round budget. This is the same family of moves that state-of-the-art
  heuristics (LKH, HGS) are built from, implemented small and readable.

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

# Local-search stack (savings_ls). All budgets are fixed COUNTS, never wall
# clock: identical inputs must produce byte-identical plans on any machine.
LS_NEIGHBORS = 20          # granular neighborhood: moves only link stops that are
                           # among each other's 20 nearest (Toth & Vigo's trick;
                           # prunes the O(n^2) move space to O(n·K) with almost
                           # no quality loss, because long new edges never help)
OR_OPT_MAX_SEG = 3         # or-opt relocates segments of 1..3 consecutive stops
ILS_ROUNDS = 40            # fixed perturbation budget for the iterated local search
ILS_KICKS = 3              # random segment relocations per perturbation round
ILS_SEED = 0               # the seed is part of the policy definition: same
                           # seed, same plan, byte for byte
_EPS = 1e-9                # a move must beat this to count as an improvement


@dataclass
class Solution:
    """Routes for one policy. Each route is a list of stop row-indices, visit order."""

    policy: str
    routes: list[list[int]]
    # savings_2opt also records total miles after Clarke-Wright construction
    # but before 2-opt, so the savings can be attributed between the two.
    construction_miles: float | None = field(default=None)
    # savings_ls records miles after every stage — construction, intra-route
    # 2-opt, the or-opt/2-opt*/swap local optimum, and the ILS best — so the
    # rationale can attribute the saved miles stage by stage.
    stage_miles: dict[str, float] | None = field(default=None)


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


def _feasible(
    route: list[int],
    dist,
    packages,
    service_min,
    capacity: float = CAPACITY_PKGS,
    max_route_min: float = MAX_ROUTE_MIN,
) -> bool:
    return (
        packages[route].sum() <= capacity
        and route_minutes(route, dist, service_min) <= max_route_min
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
def nearest_neighbor_global(
    stops: pd.DataFrame,
    dist: np.ndarray,
    capacity: float = CAPACITY_PKGS,
    max_route_min: float = MAX_ROUTE_MIN,
) -> Solution:
    """Fill trucks nearest-first, ignoring zones.

    A truck leaves the depot, repeatedly drives to the nearest unserved stop
    it can still legally take (capacity, and duration INCLUDING the ride
    home), and returns when nothing fits. Early trucks look great; the miles
    hide in the last trucks, which inherit whatever the greedy sweep left
    scattered across the metro.

    ``capacity``/``max_route_min`` default to the fleet constants; the
    CVRPLIB benchmark passes the instance capacity and an infinite shift.
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
                if d >= best_d or load + packages[i] > capacity:
                    continue
                # Feasibility must include the ride home, or the last stop
                # of the day strands the driver past the end of shift.
                drive_min = (d + dist[i + 1, 0]) / SPEED_MPH * 60.0
                if elapsed + drive_min + service_min[i] <= max_route_min:
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
def clarke_wright(
    stops: pd.DataFrame,
    dist: np.ndarray,
    capacity: float = CAPACITY_PKGS,
    max_route_min: float = MAX_ROUTE_MIN,
) -> list[list[int]]:
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
        if load[ri] + load[rj] > capacity:
            continue
        merged = a + b
        if route_minutes(merged, dist, service_min) > max_route_min:
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


def savings_2opt(
    stops: pd.DataFrame,
    dist: np.ndarray,
    capacity: float = CAPACITY_PKGS,
    max_route_min: float = MAX_ROUTE_MIN,
) -> Solution:
    """Clarke-Wright construction, then 2-opt polish per route.

    2-opt only reorders stops within a route, so it can never break
    capacity, and a shorter tour can never break the duration cap either.
    The same two calls serve both the synthetic depot day (fleet-constant
    defaults) and the CVRPLIB benchmark (instance capacity, no shift).
    """
    constructed = clarke_wright(stops, dist, capacity=capacity, max_route_min=max_route_min)
    construction_miles = sum(route_miles(r, dist) for r in constructed)
    routes = [two_opt(r, dist) for r in constructed]
    return Solution("savings_2opt", routes, construction_miles=round(construction_miles, 2))


# ---------------------------------------------------------------------------
# Policy (d): savings_ls — the same start, then a real local-search stack.
#
# This is how serious heuristic solvers (the LKH / HGS lineage) earn their
# gaps: not one clever move, but a STACK of cheap moves driven to a mutual
# local optimum, then perturb-and-reoptimize to escape it. Everything below
# is deterministic: fixed scan orders, fixed budgets, a seeded RNG.
# ---------------------------------------------------------------------------
def _neighbor_lists(dist: np.ndarray, k: int = LS_NEIGHBORS) -> np.ndarray:
    """For each stop, its k nearest OTHER stops, nearest first.

    Local-search moves below only ever try to create an edge between a stop
    and one of its k nearest neighbors. That is the granular-neighborhood
    idea: an improving move must add at least one short edge, so candidates
    whose new edges are all long are never worth scanning. Stable argsort
    keeps tie order (and therefore the whole search) deterministic.
    """
    n = dist.shape[0] - 1
    order = np.argsort(dist[1:, 1:], axis=1, kind="stable")
    rows = []
    for i in range(n):
        row = order[i]
        rows.append(row[row != i][: min(k, n - 1)])  # drop self even on 0-distance ties
    return np.asarray(rows)


class _Plan:
    """Mutable working state for the local-search stack.

    Keeps, per route: the stop list, total load, total service minutes and
    total miles, plus two per-stop lookup arrays (which route, which
    position). Move evaluation is O(1) arithmetic on four to six distance
    matrix entries; only an ACCEPTED move pays to rebuild its two routes'
    bookkeeping, and accepted moves are rare relative to scans.
    """

    def __init__(self, routes, dist, packages, service_min, capacity, max_route_min):
        self.dist = dist
        self.packages = packages
        self.service_min = service_min
        self.capacity = capacity
        self.max_route_min = max_route_min
        self.n = len(packages)
        self.routes: list[list[int]] = [list(r) for r in routes if r]
        self.route_of = np.zeros(self.n, dtype=np.int64)
        self.pos = np.zeros(self.n, dtype=np.int64)
        self.load = [0.0] * len(self.routes)
        self.svc = [0.0] * len(self.routes)
        self.miles = [0.0] * len(self.routes)
        for rid in range(len(self.routes)):
            self._rebuild(rid)

    def _rebuild(self, rid: int) -> None:
        """Recompute one route's bookkeeping after a move touched it."""
        r = self.routes[rid]
        for p, stop in enumerate(r):
            self.route_of[stop] = rid
            self.pos[stop] = p
        self.load[rid] = float(self.packages[r].sum()) if r else 0.0
        self.svc[rid] = float(self.service_min[r].sum()) if r else 0.0
        self.miles[rid] = route_miles(r, self.dist)

    def total_miles(self) -> float:
        return float(sum(self.miles))

    def _time_ok(self, new_miles: float, new_svc: float) -> bool:
        """Shift check from tracked totals — no list walk needed."""
        if self.max_route_min == float("inf"):
            return True  # CVRPLIB rules: capacity only, no shift clock
        return new_miles / SPEED_MPH * 60.0 + new_svc <= self.max_route_min + 1e-9

    # -- the sweep ----------------------------------------------------------
    def improve(self, neighbors: np.ndarray, active: np.ndarray | None = None) -> None:
        """First-improvement sweeps to a local optimum of the full move stack.

        ``active`` is the classic don't-look-bit array: a stop that offered
        no improving move is switched off and only switched back on when a
        move touches its route (or lands near it). Scan order is stop index
        0..n-1 — fixed, so the local optimum is deterministic. The ILS below
        passes a mostly-off ``active`` so each round only re-optimizes the
        region the perturbation disturbed.
        """
        if active is None:
            active = np.ones(self.n, dtype=bool)
        while True:
            improved_any = False
            for i in range(self.n):
                if not active[i]:
                    continue
                if self._improve_stop(i, neighbors, active):
                    improved_any = True
                else:
                    active[i] = False  # don't look again until something moves nearby
            if not improved_any:
                return

    def _activate(self, active, rid_a, rid_b, neighbors, i, j) -> None:
        """Wake the stops whose best move may have changed: both touched
        routes, plus the moved stops' own neighborhoods (their neighbors may
        sit in routes that were NOT touched)."""
        active[self.routes[rid_a]] = True
        active[self.routes[rid_b]] = True
        active[neighbors[i]] = True
        active[neighbors[j]] = True

    def _improve_stop(self, i: int, neighbors: np.ndarray, active: np.ndarray) -> bool:
        """Try every move that would create a short edge at stop ``i``.

        Fixed operator order per neighbor j — intra 2-opt, or-opt relocation
        (after j, then before j, segment length 1..3), then 2-opt* and swap
        when i and j ride different trucks. The first improving feasible
        move is applied immediately (first-improvement keeps sweeps cheap
        and, with the fixed scan order, deterministic).
        """
        d, P, S = self.dist, self.packages, self.service_min
        for j in neighbors[i]:
            a_id = int(self.route_of[i])
            b_id = int(self.route_of[j])
            ra, rb = self.routes[a_id], self.routes[b_id]
            pi, pj = int(self.pos[i]), int(self.pos[j])
            ni, nj = i + 1, j + 1  # distance-matrix nodes

            # ---- intra-route 2-opt: reverse the span between i and j -----
            # Removes edges (a, a+1) and (b, b+1), adds (a, b) and (a+1, b+1)
            # with the segment a+1..b reversed. Miles can only drop and the
            # load is untouched, so no feasibility check is needed.
            if a_id == b_id:
                lo, hi = (pi, pj) if pi < pj else (pj, pi)
                if hi > lo + 1:
                    s_lo, s_hi = ra[lo] + 1, ra[hi] + 1
                    after_lo = ra[lo + 1] + 1
                    after_hi = ra[hi + 1] + 1 if hi + 1 < len(ra) else 0
                    delta = (
                        d[s_lo, s_hi] + d[after_lo, after_hi]
                        - d[s_lo, after_lo] - d[s_hi, after_hi]
                    )
                    if delta < -_EPS:
                        ra[lo + 1 : hi + 1] = ra[lo + 1 : hi + 1][::-1]
                        self._rebuild(a_id)
                        self._activate(active, a_id, a_id, neighbors, i, j)
                        return True

            # ---- or-opt: relocate the 1..3-stop segment starting at i ----
            # Removal gain is fixed per segment; each insertion point near j
            # (right after j, right before j) is one O(1) delta on top.
            for seg_len in range(1, OR_OPT_MAX_SEG + 1):
                if pi + seg_len > len(ra):
                    break
                if a_id == b_id and pi <= pj < pi + seg_len:
                    continue  # j is inside the segment being moved
                s0 = ra[pi] + 1
                s1 = ra[pi + seg_len - 1] + 1
                prev_s = ra[pi - 1] + 1 if pi > 0 else 0
                next_s = ra[pi + seg_len] + 1 if pi + seg_len < len(ra) else 0
                gain = d[prev_s, next_s] - d[prev_s, s0] - d[s1, next_s]
                seg_load = float(P[ra[pi : pi + seg_len]].sum())
                seg_svc = float(S[ra[pi : pi + seg_len]].sum())

                # Two insertion slots: (j, here, next_j) and (prev_j, here, j).
                # Skip the slot that would just rebuild the removed edges.
                nxt_j = rb[pj + 1] + 1 if pj + 1 < len(rb) else 0
                prv_j = rb[pj - 1] + 1 if pj > 0 else 0
                slots = []
                if not (a_id == b_id and pj == pi - 1):  # after-j = no-op then
                    slots.append((d[nj, s0] + d[s1, nxt_j] - d[nj, nxt_j], pj + 1))
                if not (a_id == b_id and pj == pi + seg_len):  # before-j = no-op then
                    slots.append((d[prv_j, s0] + d[s1, nj] - d[prv_j, nj], pj))

                for cost, insert_at in slots:
                    if gain + cost >= -_EPS:
                        continue
                    if a_id != b_id:  # capacity + shift only change across trucks
                        if self.load[b_id] + seg_load > self.capacity:
                            continue
                        if not self._time_ok(self.miles[b_id] + cost, self.svc[b_id] + seg_svc):
                            continue
                        # rounded distances can break the triangle inequality,
                        # so the shrunk donor route is checked too (cheap)
                        if not self._time_ok(self.miles[a_id] + gain, self.svc[a_id] - seg_svc):
                            continue
                    seg = ra[pi : pi + seg_len]
                    del ra[pi : pi + seg_len]
                    at = insert_at - seg_len if (a_id == b_id and pj > pi) else insert_at
                    rb[at:at] = seg
                    self._rebuild(a_id)
                    if b_id != a_id:
                        self._rebuild(b_id)
                    self._activate(active, a_id, b_id, neighbors, i, j)
                    return True

            if a_id == b_id:
                continue

            # ---- 2-opt*: swap route tails after i and after j -------------
            # New route A = head of A up to i, then B's tail; new route B =
            # head of B up to j, then A's tail. Exactly two edges change, so
            # the miles delta is O(1); loads need the head/tail split sums,
            # paid only when the delta already looks like a win.
            after_i = ra[pi + 1] + 1 if pi + 1 < len(ra) else 0
            after_j = rb[pj + 1] + 1 if pj + 1 < len(rb) else 0
            delta = d[ni, after_j] + d[nj, after_i] - d[ni, after_i] - d[nj, after_j]
            if delta < -_EPS:
                head_a_load = float(P[ra[: pi + 1]].sum())
                head_b_load = float(P[rb[: pj + 1]].sum())
                tail_a_load = self.load[a_id] - head_a_load
                tail_b_load = self.load[b_id] - head_b_load
                if head_a_load + tail_b_load <= self.capacity and \
                        head_b_load + tail_a_load <= self.capacity:
                    new_a = ra[: pi + 1] + rb[pj + 1 :]
                    new_b = rb[: pj + 1] + ra[pi + 1 :]
                    ok = True
                    if self.max_route_min != float("inf"):
                        for r in (new_a, new_b):
                            if not self._time_ok(route_miles(r, d), float(S[r].sum())):
                                ok = False
                                break
                    if ok:
                        self.routes[a_id] = new_a
                        self.routes[b_id] = new_b
                        self._rebuild(a_id)
                        self._rebuild(b_id)
                        self._activate(active, a_id, b_id, neighbors, i, j)
                        return True

            # ---- swap: exchange stops i and j between their routes --------
            prev_i = ra[pi - 1] + 1 if pi > 0 else 0
            next_i = ra[pi + 1] + 1 if pi + 1 < len(ra) else 0
            prev_j = rb[pj - 1] + 1 if pj > 0 else 0
            next_j = rb[pj + 1] + 1 if pj + 1 < len(rb) else 0
            delta_a = d[prev_i, nj] + d[nj, next_i] - d[prev_i, ni] - d[ni, next_i]
            delta_b = d[prev_j, ni] + d[ni, next_j] - d[prev_j, nj] - d[nj, next_j]
            if delta_a + delta_b < -_EPS:
                if self.load[a_id] - P[i] + P[j] <= self.capacity and \
                        self.load[b_id] - P[j] + P[i] <= self.capacity and \
                        self._time_ok(self.miles[a_id] + delta_a,
                                      self.svc[a_id] - S[i] + S[j]) and \
                        self._time_ok(self.miles[b_id] + delta_b,
                                      self.svc[b_id] - S[j] + S[i]):
                    ra[pi], rb[pj] = j, i
                    self._rebuild(a_id)
                    self._rebuild(b_id)
                    self._activate(active, a_id, b_id, neighbors, i, j)
                    return True
        return False


def local_search(
    routes: list[list[int]],
    dist: np.ndarray,
    packages: np.ndarray,
    service_min: np.ndarray,
    capacity: float = CAPACITY_PKGS,
    max_route_min: float = MAX_ROUTE_MIN,
    neighbors: np.ndarray | None = None,
) -> list[list[int]]:
    """Drive a plan to a local optimum of the full move stack.

    Operators: intra-route 2-opt, or-opt segment relocation (intra and
    inter, segments of 1..{OR_OPT_MAX_SEG}), inter-route 2-opt* tail swaps,
    and inter-route stop swaps — all capacity- and shift-checked, all
    first-improvement in a fixed scan order, so the result is deterministic
    and total miles never increase. Routes emptied by relocation are dropped.
    """
    plan = _Plan(routes, dist, packages, service_min, capacity, max_route_min)
    if neighbors is None:
        neighbors = _neighbor_lists(dist)
    plan.improve(neighbors)
    return [r for r in plan.routes if r]


def _perturb(
    routes: list[list[int]],
    rng: np.random.Generator,
    dist: np.ndarray,
    packages: np.ndarray,
    service_min: np.ndarray,
    capacity: float,
    max_route_min: float,
) -> tuple[list[list[int]], set[int]]:
    """One ILS kick: a few random (seeded) segment relocations.

    Each kick tears a random 1..3-stop segment out of one route and splices
    it into a random position in another (feasibility-checked; infeasible
    draws are undone and retried a bounded number of times). The damage is
    deliberately non-improving — its job is to knock the plan off its local
    optimum so the next local-search pass can find a different one. Returns
    the perturbed plan plus the set of stops worth re-scanning.
    """
    routes = [list(r) for r in routes]
    touched: set[int] = set()
    for _ in range(ILS_KICKS):
        for _attempt in range(20):
            nonempty = [k for k, r in enumerate(routes) if r]
            src = nonempty[int(rng.integers(len(nonempty)))]
            ra = routes[src]
            seg_len = min(int(rng.integers(1, OR_OPT_MAX_SEG + 1)), len(ra))
            p = int(rng.integers(0, len(ra) - seg_len + 1))
            dst = nonempty[int(rng.integers(len(nonempty)))]
            if dst == src and len(ra) == seg_len:
                continue  # would just rebuild the same single-segment route
            seg = ra[p : p + seg_len]
            del ra[p : p + seg_len]
            rb = routes[dst]
            q = int(rng.integers(0, len(rb) + 1))
            rb[q:q] = seg
            if (
                packages[rb].sum() <= capacity
                and route_minutes(rb, dist, service_min) <= max_route_min
            ):
                touched.update(routes[src])
                touched.update(rb)
                break
            del rb[q : q + seg_len]  # undo and redraw
            ra[p:p] = seg
    return routes, touched


def savings_ls(
    stops: pd.DataFrame,
    dist: np.ndarray,
    capacity: float = CAPACITY_PKGS,
    max_route_min: float = MAX_ROUTE_MIN,
    ils_rounds: int = ILS_ROUNDS,
    seed: int = ILS_SEED,
) -> Solution:
    """Clarke-Wright + 2-opt + the local-search stack + iterated local search.

    The pipeline, with miles recorded at each stage for the rationale:

    1. Clarke-Wright construction (decides which stops share a truck),
    2. per-route 2-opt (uncrosses each tour),
    3. the move stack — or-opt, 2-opt*, swap, intra 2-opt — to a mutual
       local optimum (fixes ASSIGNMENT mistakes construction locked in),
    4. ILS: ``ils_rounds`` rounds of seeded perturbation + re-optimization,
       keeping the best plan ever seen (escapes the local optimum itself).

    Every budget is a fixed count and the RNG is seeded, so the same inputs
    give byte-identical routes on every run and every machine.
    """
    packages = stops["packages"].to_numpy(dtype=float)
    service_min = stops["service_min"].to_numpy(dtype=float)
    neighbors = _neighbor_lists(dist)

    constructed = clarke_wright(stops, dist, capacity=capacity, max_route_min=max_route_min)
    construction_miles = sum(route_miles(r, dist) for r in constructed)
    polished = [two_opt(r, dist) for r in constructed]
    two_opt_miles = sum(route_miles(r, dist) for r in polished)

    plan = _Plan(polished, dist, packages, service_min, capacity, max_route_min)
    plan.improve(neighbors)
    best = [list(r) for r in plan.routes if r]
    ls_miles = best_miles = plan.total_miles()

    # Iterated local search: perturb the BEST plan, re-optimize only the
    # disturbed region (don't-look bits start off everywhere else), keep the
    # result only if it beats the record. Rejected rounds restart from the
    # record, so one bad kick never compounds.
    rng = np.random.default_rng(seed)
    for _ in range(ils_rounds):
        kicked, touched = _perturb(
            best, rng, dist, packages, service_min, capacity, max_route_min
        )
        plan = _Plan(kicked, dist, packages, service_min, capacity, max_route_min)
        active = np.zeros(len(packages), dtype=bool)
        if touched:
            active[list(touched)] = True
        plan.improve(neighbors, active=active)
        cand_miles = plan.total_miles()
        if cand_miles < best_miles - _EPS:
            best = [list(r) for r in plan.routes if r]
            best_miles = cand_miles

    return Solution(
        "savings_ls",
        best,
        construction_miles=round(construction_miles, 2),
        stage_miles={
            "construction": round(construction_miles, 2),
            "two_opt": round(two_opt_miles, 2),
            "local_search": round(ls_miles, 2),
            "ils": round(best_miles, 2),
        },
    )


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
    "savings_ls": savings_ls,
}
