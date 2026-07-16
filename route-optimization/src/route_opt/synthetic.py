"""One delivery day for one depot, in a synthetic metro with a documented geography.

Route optimizers get benchmarked on Solomon instances and deployed on cities.
The gap matters: real delivery demand is not uniform, it is lumpy — subdivisions
where every third house gets a parcel, a commercial strip where one truck can
serve forty doors in a mile, and rural stragglers that eat an hour for six
stops. The fixed-zone routing this repo uses as its status quo baseline is
only beatable BECAUSE demand is lumpy (equal wedges cut through clusters and
strand rural stops with dense ones), so the generator plants that lumpiness
explicitly rather than sampling uniform points.

The metro, in depot-centered miles (depot at the origin):

- ``residential`` (~62% of stops) — six subdivision clusters, centers drawn
  3-9 miles out at random bearings, stops Gaussian around each center
  (sigma ~0.9 mi). Mostly 1-2 packages, short curbside service.
- ``commercial`` (~22%) — a strip along an east-west arterial just north of
  the depot (x in [-7, 7], y ~ 1.5). More packages per door, longer service
  (dock, signature, freight elevator).
- ``rural``      (~16%) — scattered points 8.5-11.5 miles out at any bearing.
  Same parcel mix as residential, slightly longer service (long driveways),
  and the real cost is the driving between them.

Everything is deterministic given ``seed``. Fleet and physics constants live
in ``solve.py`` (capacity 180 packages, 9h duration cap, 25 mph); this module
only makes the demand.

``messy=True`` plants the three data faults every dispatch extract actually
has, so ``cleaning.py`` has real work to audit:

- duplicate stop rows (the WMS exported the stop once per package line),
- (0, 0) coordinates — the geocoder's "null island" failure code — plus a
  couple of stops geocoded into the next county,
- negative package counts from a returns line-item joined the wrong way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Metro geometry (miles, depot at origin). The square bound is what cleaning
# uses to reject geocode failures: no real stop is outside it.
METRO_HALF_MI = 12.0

N_CLUSTERS = 6
CLUSTER_RADIUS_MI = (3.0, 9.0)  # how far out subdivision centers sit
CLUSTER_SIGMA_MI = 0.9

STRIP_X_MI = (-7.0, 7.0)  # the commercial arterial
STRIP_Y_MI, STRIP_SIGMA_MI = 1.5, 0.35

RURAL_RADIUS_MI = (8.5, 11.5)

STOP_MIX = {"residential": 0.62, "commercial": 0.22, "rural": 0.16}

# Packages per stop (1-4) and curbside service minutes, by stop type.
# Commercial doors take longer (dock, signature) but carry more packages;
# rural adds driveway time. Calibrate from your stop-level scan data.
PACKAGE_PROBS = {
    "residential": [0.55, 0.30, 0.10, 0.05],
    "commercial": [0.20, 0.30, 0.30, 0.20],
    "rural": [0.55, 0.30, 0.10, 0.05],
}
SERVICE_MIN = {"residential": 3.0, "commercial": 5.5, "rural": 4.0}

MIN_PACKAGES, MAX_PACKAGES = 1, 4


def make_day(n: int = 600, seed: int = 7, messy: bool = True) -> pd.DataFrame:
    """Generate one delivery day of ``n`` stops. Deterministic given ``seed``."""
    rng = np.random.default_rng(seed)

    types = rng.choice(list(STOP_MIX), n, p=list(STOP_MIX.values()))
    x = np.empty(n)
    y = np.empty(n)

    # Residential: pick a cluster, then a Gaussian offset around its center.
    centers_r = rng.uniform(*CLUSTER_RADIUS_MI, N_CLUSTERS)
    centers_a = rng.uniform(0, 2 * np.pi, N_CLUSTERS)
    cx, cy = centers_r * np.cos(centers_a), centers_r * np.sin(centers_a)
    res = types == "residential"
    which = rng.integers(0, N_CLUSTERS, res.sum())
    x[res] = cx[which] + rng.normal(0, CLUSTER_SIGMA_MI, res.sum())
    y[res] = cy[which] + rng.normal(0, CLUSTER_SIGMA_MI, res.sum())

    # Commercial: along the arterial.
    com = types == "commercial"
    x[com] = rng.uniform(*STRIP_X_MI, com.sum())
    y[com] = rng.normal(STRIP_Y_MI, STRIP_SIGMA_MI, com.sum())

    # Rural: an outer annulus, any bearing.
    rur = types == "rural"
    radius = rng.uniform(*RURAL_RADIUS_MI, rur.sum())
    bearing = rng.uniform(0, 2 * np.pi, rur.sum())
    x[rur] = radius * np.cos(bearing)
    y[rur] = radius * np.sin(bearing)

    # Keep every real stop strictly inside the metro bound so cleaning can
    # treat "outside the bound" as a hard geocode failure, not a judgment call.
    lim = METRO_HALF_MI - 0.2
    x, y = x.clip(-lim, lim), y.clip(-lim, lim)

    packages = np.empty(n, dtype=int)
    for t, probs in PACKAGE_PROBS.items():
        mask = types == t
        packages[mask] = rng.choice([1, 2, 3, 4], mask.sum(), p=probs)

    df = pd.DataFrame(
        {
            "stop_id": [f"STP{seed:02d}{i:05d}" for i in range(n)],
            "stop_type": types,
            "x_mi": x.round(4),
            "y_mi": y.round(4),
            "packages": packages,
            "service_min": pd.Series(types).map(SERVICE_MIN).to_numpy(),
        }
    )
    if messy:
        df = _mess_up(df, rng)
    return df.reset_index(drop=True)


def _mess_up(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Plant the dispatch-extract faults that cleaning must find and fix."""
    n = len(df)

    # WMS exports the stop once per package line: literal duplicate rows.
    dup_idx = rng.choice(n, size=max(3, n // 50), replace=False)
    df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)

    # Geocoder failures: "null island" (0, 0) ...
    fail_idx = rng.choice(n, size=max(2, n // 120), replace=False)
    df.loc[fail_idx, ["x_mi", "y_mi"]] = 0.0
    # ... and a couple of stops geocoded into the next county.
    county_idx = rng.choice(np.setdiff1d(np.arange(n), fail_idx), size=3, replace=False)
    df.loc[county_idx, "x_mi"] = rng.uniform(30.0, 60.0, 3).round(4)

    # A returns line-item joined the wrong way: negative package counts.
    neg_idx = rng.choice(n, size=max(2, n // 150), replace=False)
    df.loc[neg_idx, "packages"] = -df.loc[neg_idx, "packages"]

    return df
