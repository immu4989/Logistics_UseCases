"""CVRPLIB instances: parse, fetch, and solve with the exact same optimizer.

CVRPLIB (https://galgos.inf.puc-rio.br/cvrplib, Uchoa et al.) is the standard
benchmark library for the capacitated VRP, with best-known solutions (BKS)
for every instance — for the classic sets below the BKS are proven optima.
Benchmarking against it turns "our optimizer beats our own baselines" into
"our optimizer is X% above the best solution anyone has ever found".

Three pieces live here:

- a TSPLIB-format parser for ``.vrp`` instances (EUC_2D only) and the
  matching ``.sol`` best-known-solution files,
- a small registry of instances spanning 32-153 customers, with BKS values
  verified against the downloaded ``.sol`` files,
- a thin adapter that runs the UNCHANGED Clarke-Wright + 2-opt from
  :mod:`route_opt.solve` on a parsed instance: capacity constraint only
  (CVRPLIB instances have no shift clock), unbounded trucks, minimizing
  distance with depot round trips counted exactly as CVRPLIB counts them.

One convention matters and is easy to get wrong: TSPLIB ``EUC_2D`` distances
are ROUNDED TO THE NEAREST INTEGER, ``nint(sqrt(dx^2 + dy^2))``. Every BKS
assumes that rounding; scoring euclidean floats against integer-valued BKS
would misstate the gap.
"""

from __future__ import annotations

import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import solve

# ---------------------------------------------------------------------------
# Registry. BKS values verified 2026-07-22 against the `Cost` line of each
# instance's .sol file downloaded from CVRPLIB (galgos.inf.puc-rio.br/cvrplib);
# for these sets the BKS are proven optima. The numeric id is CVRPLIB's
# download id for both the instance file and its best-known solution.
# Sets: A/B = Augerat et al. (1995), X = Uchoa et al. (2017).
# ---------------------------------------------------------------------------
DOWNLOAD_URL = "https://galgos.inf.puc-rio.br/cvrplib/en/download/{kind}/{id}"

REGISTRY: dict[str, dict] = {
    "A-n32-k5": {"id": 4, "bks": 784},
    "A-n45-k6": {"id": 16, "bks": 944},
    "A-n65-k9": {"id": 29, "bks": 1174},
    "A-n80-k10": {"id": 31, "bks": 1763},
    "B-n50-k7": {"id": 41, "bks": 741},
    "B-n78-k10": {"id": 53, "bks": 1221},
    "X-n101-k25": {"id": 158, "bks": 27591},
    "X-n153-k22": {"id": 169, "bks": 21220},
}

DEFAULT_CACHE_DIR = Path("data/cvrplib")  # covered by the repo root .gitignore (data/)


@dataclass
class Instance:
    """One parsed CVRPLIB instance, depot at row 0 of ``coords``/``demands``."""

    name: str
    capacity: int
    coords: np.ndarray   # (n, 2) floats, node 0 = depot
    demands: np.ndarray  # (n,) ints, demands[0] = 0

    @property
    def n_customers(self) -> int:
        return len(self.coords) - 1


# ---------------------------------------------------------------------------
# Parsing. TSPLIB files are 1-indexed; CVRPLIB instances put the depot at
# node 1, which maps onto solve.py's convention (node 0 = depot) directly.
# X-set files use tabs and CRLF line endings, so tokenize on any whitespace.
# ---------------------------------------------------------------------------
def parse_vrp(text: str) -> Instance:
    """Parse a TSPLIB-format .vrp file (EUC_2D coordinates only)."""
    header: dict[str, str] = {}
    coords: dict[int, tuple[float, float]] = {}
    demands: dict[int, int] = {}
    depot: int | None = None
    section: str | None = None

    for raw in text.splitlines():
        line = raw.replace("\r", "").strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith(("NODE_COORD_SECTION", "DEMAND_SECTION", "DEPOT_SECTION", "EOF")):
            section = upper.split()[0]
            continue
        if section is None:
            if ":" in line:
                key, value = line.split(":", 1)
                header[key.strip().upper()] = value.strip()
            continue
        tokens = line.split()
        if section == "NODE_COORD_SECTION":
            coords[int(tokens[0])] = (float(tokens[1]), float(tokens[2]))
        elif section == "DEMAND_SECTION":
            demands[int(tokens[0])] = int(float(tokens[1]))
        elif section == "DEPOT_SECTION":
            node = int(float(tokens[0]))
            if node != -1 and depot is None:
                depot = node

    n = int(header["DIMENSION"])
    weight_type = header.get("EDGE_WEIGHT_TYPE", "").upper()
    if weight_type != "EUC_2D":
        raise ValueError(f"only EUC_2D instances supported, got {weight_type!r}")
    if sorted(coords) != list(range(1, n + 1)) or sorted(demands) != list(range(1, n + 1)):
        raise ValueError("coordinate/demand sections do not cover nodes 1..DIMENSION")
    if depot != 1:
        # True for every CVRPLIB A/B/X instance; keeping the assumption
        # explicit keeps the .sol customer numbering below trivially right.
        raise ValueError(f"expected the depot at node 1, got {depot}")
    if demands[1] != 0:
        raise ValueError("depot demand must be 0")

    return Instance(
        name=header.get("NAME", "?"),
        capacity=int(header["CAPACITY"]),
        coords=np.array([coords[i] for i in range(1, n + 1)], dtype=float),
        demands=np.array([demands[i] for i in range(1, n + 1)], dtype=int),
    )


def parse_sol(text: str) -> tuple[list[list[int]], int]:
    """Parse a CVRPLIB .sol file into (routes, cost).

    Solution files number customers 1..n-1 in instance order, depot excluded
    — i.e. customer ``c`` is .vrp node ``c + 1``. Routes are returned in
    solve.py's stop-index convention (stop ``i`` = distance-matrix node
    ``i + 1``), which for depot-at-node-1 instances means ``c - 1``.
    Lines that are neither ``Route #k: ...`` nor ``Cost ...`` are ignored,
    so attribution comments survive parsing.
    """
    routes: list[list[int]] = []
    cost: int | None = None
    for line in text.splitlines():
        route_match = re.match(r"Route\s*#\d+\s*:\s*(.+)", line.strip(), re.IGNORECASE)
        if route_match:
            routes.append([int(tok) - 1 for tok in route_match.group(1).split()])
            continue
        cost_match = re.match(r"Cost\s+(\d+)", line.strip(), re.IGNORECASE)
        if cost_match:
            cost = int(cost_match.group(1))
    if not routes or cost is None:
        raise ValueError("no routes / no Cost line found in solution file")
    return routes, cost


def euc_2d_matrix(instance: Instance) -> np.ndarray:
    """TSPLIB EUC_2D distance matrix: nint(sqrt(dx^2 + dy^2)).

    ``nint`` is round-half-up, i.e. floor(x + 0.5) — NOT numpy's default
    round-half-to-even. BKS values assume this exact rounding.
    """
    diff = instance.coords[:, None, :] - instance.coords[None, :, :]
    return np.floor(np.hypot(diff[..., 0], diff[..., 1]) + 0.5)


def solution_cost(routes: list[list[int]], dist: np.ndarray) -> float:
    """Total distance of a set of routes, depot round trips included."""
    return sum(solve.route_miles(route, dist) for route in routes)


# ---------------------------------------------------------------------------
# Fetching.
# ---------------------------------------------------------------------------
def fetch(name: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> tuple[Path, Path]:
    """Download a registry instance's .vrp and .sol into ``cache_dir`` (cached)."""
    if name not in REGISTRY:
        raise KeyError(f"{name!r} not in registry; known: {', '.join(REGISTRY)}")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    entry = REGISTRY[name]
    paths = {}
    for kind, suffix in [("instance", ".vrp"), ("bks", ".sol")]:
        path = cache_dir / f"{name}{suffix}"
        if not path.exists():
            url = DOWNLOAD_URL.format(kind=kind, id=entry["id"])
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            path.write_bytes(data)
        paths[suffix] = path
    return paths[".vrp"], paths[".sol"]


def load(name: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> Instance:
    """Fetch (if needed) and parse a registry instance."""
    vrp_path, _ = fetch(name, cache_dir)
    return parse_vrp(vrp_path.read_text())


# ---------------------------------------------------------------------------
# Adapter: same optimizer, CVRPLIB rules.
# ---------------------------------------------------------------------------
def _as_stops_frame(instance: Instance) -> pd.DataFrame:
    """Customers as the stop frame solve.py expects: demand in, no service time."""
    return pd.DataFrame(
        {
            "packages": instance.demands[1:],
            "service_min": 0.0,  # CVRPLIB cost is pure distance
        }
    )


def solve_instance(instance: Instance) -> tuple[list[list[int]], float]:
    """Run the repo's Clarke-Wright + 2-opt on a CVRPLIB instance.

    Same implementation as the synthetic pipeline — only the constraints
    differ: instance capacity, no duration cap (CVRPLIB has no shift clock),
    trucks unbounded. Returns (routes in stop-index convention, total cost).
    """
    dist = euc_2d_matrix(instance)
    stops = _as_stops_frame(instance)
    result = solve.savings_2opt(
        stops, dist, capacity=instance.capacity, max_route_min=float("inf")
    )
    return result.routes, solution_cost(result.routes, dist)


def solve_instance_ls(instance: Instance) -> tuple[list[list[int]], float]:
    """Run the repo's full local-search solver (savings_ls) on an instance.

    Same code path as the synthetic pipeline's star policy: Clarke-Wright +
    2-opt start, then the or-opt / 2-opt* / swap stack and the seeded ILS.
    Deterministic: same instance, same routes, byte for byte, every run.
    """
    dist = euc_2d_matrix(instance)
    stops = _as_stops_frame(instance)
    result = solve.savings_ls(
        stops, dist, capacity=instance.capacity, max_route_min=float("inf")
    )
    return result.routes, solution_cost(result.routes, dist)


def solve_instance_nn(instance: Instance) -> tuple[list[list[int]], float]:
    """Plain nearest-neighbor on a CVRPLIB instance, for context in the bench."""
    dist = euc_2d_matrix(instance)
    stops = _as_stops_frame(instance)
    result = solve.nearest_neighbor_global(
        stops, dist, capacity=instance.capacity, max_route_min=float("inf")
    )
    return result.routes, solution_cost(result.routes, dist)
