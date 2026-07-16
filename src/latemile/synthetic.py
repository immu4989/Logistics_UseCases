"""Synthetic parcel-shipment generator with a documented ground-truth process.

Why synthetic data? Carrier operational data is proprietary, so this repo ships
with a generator whose *causal structure is known*. That gives you two things:

1. The pipeline runs end-to-end with zero external downloads.
2. The SHAP analysis can be checked against ground truth: the drivers the model
   "discovers" should match the coefficients below. This is a much stronger
   demo than SHAP on a black-box dataset, and a useful sanity harness when you
   adapt the pipeline to your own data.

The generative process (log-odds of missing the delivery commitment):

    logit = BASE
          + 0.9  * late_pickup            (picked up after facility cutoff)
          + 1.6  * origin_hub_congestion   (0-1)
          + 1.1  * dest_hub_congestion     (0-1)
          + 0.55 * dest_weather_severity   (0-3)
          + 0.0012 * distance_miles        (long lanes -> more legs to slip)
          + 0.5  * is_peak_season
          + 0.45 * is_rural_dest
          + 0.35 * residential             (denser stop sequencing risk)
          - 0.9  * ground service          (widest commit window = most slack)
          - 0.4  * two_day service
          + 0.25 * route_stop_density / 3
          + interaction: weather hits ground service hardest (+0.3 * sev * ground)
          + interaction: peak * origin congestion (+0.8 * both)
          + noise ~ Normal(0, 0.35)

Weight, volume, declared value, signature and day-of-week carry ~zero true
signal; they exist so the explainability step has genuine negatives to rank
below the real drivers.

`make_dataset(..., messy=True)` additionally injects the data-quality problems
every real extract has (duplicates, missing values, sentinel codes,
inconsistent category strings, impossible values) so `cleaning.py` has real
work to do.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema

# Ground-truth coefficients, exposed so tests and the explain step can compare
# the model's SHAP ranking against reality.
TRUE_DRIVERS = {
    "origin_hub_congestion": 1.6,
    "dest_hub_congestion": 1.1,
    "minutes_after_cutoff": 0.9,       # via the late_pickup indicator
    "dest_weather_severity": 0.55,
    "service_level": 0.9,              # ground vs overnight spread
    "is_peak_season": 0.5,
    "is_rural_dest": 0.45,
    "distance_miles": 0.0012,          # per mile; ~1.2 logits at the mean lane
    "dest_type": 0.35,
    "route_stop_density": 0.25,
}
NOISE_FEATURES = [
    "package_weight_lb",
    "package_volume_cuft",
    "declared_value_usd",
    "signature_required",
    "day_of_week",
]

BASE_LOGIT = -5.55  # calibrated to ~9-11% overall miss rate


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

    # Pickup time relative to cutoff: most on time, a tail arrives late.
    minutes_after_cutoff = rng.normal(-45, 40, n)
    late_tail = rng.random(n) < 0.12
    minutes_after_cutoff[late_tail] = rng.uniform(1, 120, late_tail.sum())
    late_pickup = (minutes_after_cutoff > 0).astype(int)

    # Hub congestion: baseline beta, pushed up in peak season.
    origin_cong = rng.beta(2.2, 4.5, n) + 0.22 * is_peak
    dest_cong = rng.beta(2.2, 4.5, n) + 0.18 * is_peak
    origin_cong = origin_cong.clip(0, 1)
    dest_cong = dest_cong.clip(0, 1)

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

    promised_days = np.select(
        [service_level == "overnight", service_level == "two_day"],
        [1, 2],
        default=np.ceil(1 + distance / 600).clip(3, 7),
    ).astype(int)

    is_ground = (service_level == "ground").astype(int)
    is_two_day = (service_level == "two_day").astype(int)
    residential = (dest_type == "residential").astype(int)

    logit = (
        BASE_LOGIT
        + 0.9 * late_pickup
        + 1.6 * origin_cong
        + 1.1 * dest_cong
        + 0.55 * weather
        + 0.0012 * distance
        + 0.5 * is_peak
        + 0.45 * is_rural
        + 0.35 * residential
        - 0.9 * is_ground
        - 0.4 * is_two_day
        + 0.25 * stop_density / 3
        + 0.3 * weather * is_ground
        + 0.8 * is_peak * origin_cong
        + rng.normal(0, 0.35, n)
    )
    p_miss = 1 / (1 + np.exp(-logit))
    missed = rng.binomial(1, p_miss)

    df = pd.DataFrame(
        {
            schema.ID_COL: [f"SHP{seed:02d}{i:08d}" for i in range(n)],
            schema.DATE_COL: ship_date,
            "distance_miles": distance.round(1),
            "package_weight_lb": weight.round(2),
            "package_volume_cuft": volume.round(3),
            "declared_value_usd": declared_value.round(2),
            "minutes_after_cutoff": minutes_after_cutoff.round(1),
            "origin_hub_congestion": origin_cong.round(4),
            "dest_hub_congestion": dest_cong.round(4),
            "dest_weather_severity": weather,
            "route_stop_density": stop_density.round(3),
            "promised_transit_days": promised_days,
            "service_level": service_level,
            "origin_region": origin_region,
            "dest_region": dest_region,
            "dest_type": dest_type,
            "day_of_week": day_of_week,
            "is_peak_season": is_peak,
            "is_rural_dest": is_rural,
            "signature_required": signature,
            schema.LABEL_COL: missed,
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

    # 5. Impossible values.
    idx = df.sample(frac=0.004, random_state=int(rng.integers(1e6))).index
    df.loc[idx, "package_weight_lb"] = rng.uniform(500, 2000, len(idx)).round(1)

    return df.sample(frac=1, random_state=int(rng.integers(1e6))).reset_index(drop=True)
