"""Synthetic parcel-shipment generator with a documented ground-truth transit process.

Why synthetic data? Carrier operational data is proprietary, so this repo ships
with a generator whose *causal structure is known*. That gives you two things:

1. The pipeline runs end-to-end with zero external downloads.
2. The SHAP analysis can be checked against ground truth: the drivers the model
   "discovers" should match the process below, and that check lives in the test
   suite. Keep this harness when you adapt the pipeline to your own data — it is
   the regression test for the entire explanation stack.

The generative process (actual transit time, in days):

    mu = SERVICE_PARAMS[service].base                       (induction + delivery overhead)
       + distance / SERVICE_PARAMS[service].miles_per_day   (linehaul; dominant for ground)
       + 0.25 * floor(distance / 800)   for ground only     (each extra sort leg costs time)
       + 0.90 * origin_hub_congestion                       (0-1)
       + 0.70 * dest_hub_congestion                         (0-1)
       + 0.30 * dest_weather_severity                       (0-3)
       + 0.35 * is_peak_season
       + 0.35 * is_rural_dest
       + 0.45 * weekend ship * ground service               (parcel waits for Monday linehaul)
       + 0.05 * route_stop_density

    actual_transit_days = max(mu + noise, 0.2)

**The noise is heteroscedastic by design.** Its standard deviation grows with
distance and congestion:

    sigma = (0.18 + 0.30 * distance/1000 + 0.50 * mean_hub_congestion)
            * SERVICE_PARAMS[service].sigma_scale

A 3,000-mile ground lane through congested hubs is roughly 3x noisier than a
short intra-region hop. That is the product story of this use case: the point
ETA is easy, but a single number is a lie on long lanes — honest uncertainty
bounds are what a customer promise actually needs, which is why the pipeline
trains quantile models, not just a point regressor.

Declared value, signature flag, package weight and volume carry zero true
signal; they exist so the explainability step has genuine negatives to rank
below the real drivers.

`make_dataset(..., messy=True)` additionally injects the data-quality problems
every real extract has (duplicates, missing values, sentinel codes,
inconsistent category strings) plus one regression-specific defect: a small
fraction of impossible zero/negative transit times, the kind produced by a
delivery scan mis-keyed before the induction scan. `cleaning.py` must drop
those rows — a regressor trained on them learns that some parcels arrive
before they ship.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema

# Ground-truth coefficients (days), exposed so tests and the explain step can
# compare the model's SHAP ranking against reality.
SERVICE_PARAMS = {
    # base = fixed induction + last-mile overhead; miles_per_day = effective
    # linehaul speed; sigma_scale = how much of the network's noise the
    # service absorbs (dedicated air is tightly scheduled, ground is not).
    "overnight": {"base": 1.05, "miles_per_day": 18_000, "sigma_scale": 0.35},
    "two_day": {"base": 1.85, "miles_per_day": 9_000, "sigma_scale": 0.55},
    "ground": {"base": 1.35, "miles_per_day": 700, "sigma_scale": 1.0},
}

TRUE_DRIVERS = {
    "distance_miles": 1 / 700,       # days per mile on ground linehaul, plus 0.25/sort leg
    "service_level": 4.0,            # ground-vs-overnight spread at the mean cross-region lane
    "origin_hub_congestion": 0.90,
    "dest_hub_congestion": 0.70,
    "dest_weather_severity": 0.30,
    "is_peak_season": 0.35,
    "is_rural_dest": 0.35,
    "day_of_week": 0.45,             # via the weekend-induction hold on ground service
    "route_stop_density": 0.05,
}
NOISE_FEATURES = [
    "package_weight_lb",
    "package_volume_cuft",
    "declared_value_usd",
    "signature_required",
]

# Heteroscedastic noise: sigma = SIGMA_BASE + SIGMA_DIST*distance/1000 + SIGMA_CONG*mean_cong.
SIGMA_BASE = 0.18
SIGMA_DIST = 0.30
SIGMA_CONG = 0.50

GROUND_LEG_MILES = 800   # one extra sort leg per this many ground miles
GROUND_LEG_DAYS = 0.25


def make_dataset(
    n: int = 60_000,
    seed: int = 7,
    start_date: str = "2025-01-06",
    n_days: int = 180,
    messy: bool = False,
) -> pd.DataFrame:
    """Generate `n` shipments over `n_days`, returning the canonical schema."""
    rng = np.random.default_rng(seed)

    day_offsets = rng.integers(0, n_days, n)
    ship_date = pd.Timestamp(start_date) + pd.to_timedelta(day_offsets, unit="D")
    day_of_week = ship_date.dayofweek.to_numpy()

    # Two promo/surge windows (think spring sale + summer prime-day style
    # event) placed so that BOTH the training period and the held-out test
    # period contain surge conditions — if peak only existed at the end of the
    # window, the time-based split would leave it entirely out of training.
    w1, w2 = int(n_days * 0.33), int(n_days * 0.83)
    is_peak = (
        ((day_offsets >= w1) & (day_offsets < w1 + 15))
        | ((day_offsets >= w2) & (day_offsets < w2 + 15))
    ).astype(int)

    service_level = rng.choice(schema.SERVICE_LEVELS, n, p=[0.15, 0.25, 0.60])
    origin_region = rng.choice(schema.REGIONS, n)
    dest_region = rng.choice(schema.REGIONS, n)

    same_region = origin_region == dest_region
    distance = np.where(
        same_region,
        rng.gamma(2.0, 90, n),          # intra-region lanes
        400 + rng.gamma(2.5, 320, n),   # cross-region lanes
    ).clip(5, 3200)

    dest_type = rng.choice(schema.DEST_TYPES, n, p=[0.72, 0.28])
    is_rural = rng.binomial(1, 0.22, n)
    signature = rng.binomial(1, 0.12, n)

    weight = rng.lognormal(0.9, 0.9, n).clip(0.1, 150)
    volume = (weight * rng.uniform(0.05, 0.25, n)).clip(0.01, 60)
    declared_value = rng.lognormal(3.4, 1.1, n).clip(1, 20_000)

    # Hub congestion: baseline beta, pushed up in peak season.
    origin_cong = (rng.beta(2.2, 4.5, n) + 0.22 * is_peak).clip(0, 1)
    dest_cong = (rng.beta(2.2, 4.5, n) + 0.18 * is_peak).clip(0, 1)

    # Weather severity at destination: mostly clear, winter-weighted.
    month = ship_date.month.to_numpy()
    winter = np.isin(month, [1, 2, 12]).astype(int)
    weather = rng.choice([0, 1, 2, 3], n, p=[0.70, 0.18, 0.09, 0.03]) * (1 + 0.4 * winter)
    weather = np.minimum(np.round(weather), 3).astype(int)

    stop_density = np.where(
        dest_type == "residential",
        rng.gamma(3.0, 1.2, n),
        rng.gamma(2.0, 0.9, n),
    ).clip(0.2, 15)

    # --- the transit-time process (documented in the module docstring) ------
    base = np.array([SERVICE_PARAMS[s]["base"] for s in service_level])
    speed = np.array([SERVICE_PARAMS[s]["miles_per_day"] for s in service_level])
    sigma_scale = np.array([SERVICE_PARAMS[s]["sigma_scale"] for s in service_level])
    is_ground = (service_level == "ground").astype(int)
    is_weekend = np.isin(day_of_week, [5, 6]).astype(int)

    mu = (
        base
        + distance / speed
        + GROUND_LEG_DAYS * np.floor(distance / GROUND_LEG_MILES) * is_ground
        + 0.90 * origin_cong
        + 0.70 * dest_cong
        + 0.30 * weather
        + 0.35 * is_peak
        + 0.35 * is_rural
        + 0.45 * is_weekend * is_ground
        + 0.05 * stop_density
    )
    sigma = (SIGMA_BASE + SIGMA_DIST * distance / 1000 + SIGMA_CONG * (origin_cong + dest_cong) / 2)
    sigma = sigma * sigma_scale
    transit = np.maximum(mu + rng.normal(0, 1, n) * sigma, 0.2)

    df = pd.DataFrame(
        {
            schema.ID_COL: [f"SHP{seed:02d}{i:08d}" for i in range(n)],
            schema.DATE_COL: ship_date,
            "distance_miles": distance.round(1),
            "package_weight_lb": weight.round(2),
            "package_volume_cuft": volume.round(3),
            "declared_value_usd": declared_value.round(2),
            "origin_hub_congestion": origin_cong.round(4),
            "dest_hub_congestion": dest_cong.round(4),
            "dest_weather_severity": weather,
            "route_stop_density": stop_density.round(3),
            "service_level": service_level,
            "origin_region": origin_region,
            "dest_region": dest_region,
            "dest_type": dest_type,
            "day_of_week": day_of_week,
            "is_peak_season": is_peak,
            "is_rural_dest": is_rural,
            "signature_required": signature,
            schema.LABEL_COL: transit.round(2),
        }
    )

    if messy:
        df = _inject_mess(df, rng)
    return df


def _inject_mess(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add the defects real shipment extracts have. cleaning.py must undo all of this."""
    df = df.copy()

    # 1. Exact duplicate rows (double-scanned events exported twice).
    dupes = df.sample(frac=0.01, random_state=int(rng.integers(1e6)))
    df = pd.concat([df, dupes], ignore_index=True)

    # 2. Missing values in operational fields.
    for col, frac in [
        ("origin_hub_congestion", 0.03),
        ("dest_weather_severity", 0.05),
        ("route_stop_density", 0.02),
        ("package_weight_lb", 0.01),
    ]:
        idx = df.sample(frac=frac, random_state=int(rng.integers(1e6))).index
        df.loc[idx, col] = np.nan

    # 3. Sentinel codes instead of NaN (classic mainframe export artifact).
    idx = df.sample(frac=0.008, random_state=int(rng.integers(1e6))).index
    df.loc[idx, "distance_miles"] = 9999.0
    idx = df.sample(frac=0.006, random_state=int(rng.integers(1e6))).index
    df.loc[idx, "package_weight_lb"] = -1.0

    # 4. Inconsistent categorical casing / whitespace.
    idx = df.sample(frac=0.04, random_state=int(rng.integers(1e6))).index
    df.loc[idx, "service_level"] = df.loc[idx, "service_level"].str.upper()
    idx = df.sample(frac=0.03, random_state=int(rng.integers(1e6))).index
    df.loc[idx, "dest_type"] = " " + df.loc[idx, "dest_type"] + " "

    # 5. Impossible package weights.
    idx = df.sample(frac=0.004, random_state=int(rng.integers(1e6))).index
    df.loc[idx, "package_weight_lb"] = rng.uniform(500, 2000, len(idx)).round(1)

    # 6. Regression-specific: impossible zero/negative transit times, e.g. a
    #    delivery scan mis-keyed before the induction scan. Left in training
    #    data, these teach a regressor that some parcels arrive before they
    #    ship; cleaning must drop the rows, not clip them.
    idx = df.sample(frac=0.005, random_state=int(rng.integers(1e6))).index
    df.loc[idx, schema.LABEL_COL] = rng.uniform(-2.0, 0.0, len(idx)).round(2)

    return df.sample(frac=1, random_state=int(rng.integers(1e6))).reset_index(drop=True)
