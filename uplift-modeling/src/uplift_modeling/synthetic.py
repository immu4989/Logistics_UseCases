"""Synthetic randomized-pilot log with a documented heterogeneous treatment effect.

The scenario: for one quarter, a parcel network ran a pilot of "reroute assist"
(an ops analyst re-plans the linehaul path of a flagged shipment, $4 of labor
and API cost per shipment). Crucially the pilot was RANDOMIZED: 25% of
shipments got the assist, chosen by coin flip, independent of every feature.

Why randomized, and why that matters more than any model choice downstream:
you cannot learn uplift from a targeted log for the same reason you cannot
learn price elasticity from a disciplined cost-plus quote log (see the
dynamic-pricing use case in this repo). If historical reroutes only ever went
to shipments an old rule flagged, treatment assignment is a deterministic
function of the features, treated and untreated shipments never overlap on the
strata you care about, and no estimator — however fancy — can separate "the
reroute helped" from "these shipments were different". Randomization (or at
least injected exploration) is what makes the counterfactual identifiable.

Ground truth, all exposed as constants + ``true_effect`` so tests and the
evaluation can compare estimates against reality:

Control miss probability p0 (logistic, base rate ~11%, delivery-commit flavor):

    logit0 = BASE
           + 1.6  * origin_congestion      (0-1)
           + 1.0  * dest_congestion        (0-1)
           + 1.15 * dest_weather_severity  (0-3)  <- the biggest single driver
           + 0.0009 * distance_miles
           + 0.45 * is_peak
           + 0.40 * is_rural
           - 0.25 * ground service         (widest window, some slack)
           + 0.30 * overnight service      (tightest window)
           + 1.6  * routing_gate           (congested-origin ground long-haul:
                                            missed linehaul connections compound)
           + noise ~ Normal(0, 0.30)

Treatment effect on the miss probability (absolute reduction; the point of
this use case is that it is NOT proportional to risk):

    * ROUTING-DRIVEN segment (ground service x congested origin x long lane):
      rerouting genuinely fixes the problem — up to ~60% relative reduction
      of p0 (REL_REDUCTION_MAX), phased in smoothly by congestion and distance.
    * WEATHER-DRIVEN segment (dest_weather_severity >= 2): effect ~0. You can
      re-plan a linehaul path; you cannot reroute around a storm sitting on
      the destination. These shipments carry the HIGHEST risk in the whole
      log and close to ZERO uplift — the money insight.
    * OVERNIGHT segment: reroute assist inserts an extra handling leg into a
      window with no slack, RAISING miss probability by ~2pp. Intervention
      harm is a real phenomenon, and it is why "treat everyone risky" is not
      a safe default.

    p1 = clip(p0 - effect, ...);  true_cate = p0 - p1  (positive = misses
    prevented per treated shipment).

Realized outcome: miss ~ Bernoulli(p1 if treated else p0). The per-row
``true_cate`` (plus ``p0_true`` and ``segment_true``) ride along in columns
that the model-matrix whitelist excludes — same pattern as ``p_miss_true`` in
delivery-commit-prediction, asserted by a pollution test.

``messy=True`` injects duplicate ids, negative distances and inconsistent
categorical casing so cleaning.py has real work to do. Deterministic per seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ID_COL = "shipment_id"
DATE_COL = "ship_date"
LABEL_COL = "missed_commit"
TREATMENT_COL = "treated"

SERVICE_LEVELS = ["overnight", "two_day", "ground"]
CUSTOMER_TIERS = ["standard", "premium", "contract"]

# Ground-truth columns: exposed for evaluation, banned from the model matrix.
TRUTH_COLS = ["true_cate", "p0_true", "segment_true"]

PROPENSITY = 0.25          # randomized pilot: P(treated) for every shipment
BASE_LOGIT = -5.05         # calibrated to a ~12% control miss rate
ROUTING_RISK_COEF = 1.6    # extra log-odds of missing on gated routing lanes

# --- treatment-effect constants (the documented ground truth) ---------------
REL_REDUCTION_MAX = 0.60   # max relative reduction of p0 in the routing segment
OVERNIGHT_HARM = 0.02      # absolute miss-prob INCREASE when overnight is rerouted
CONG_RAMP = (0.35, 0.45)   # origin congestion range over which the effect phases in
DIST_RAMP = (450.0, 650.0)  # lane distance range over which the effect phases in
WEATHER_RAMP = (0.5, 2.0)  # severity range over which reroute usefulness dies


def routing_gate(
    is_ground: np.ndarray, origin_congestion: np.ndarray, distance_miles: np.ndarray
) -> np.ndarray:
    """0-1 intensity of 'routing-driven' risk: ground x congested origin x long lane.

    The SAME gate drives both the extra control risk (missed linehaul
    connections) and the treatment effect (a reroute fixes exactly that), which
    is what makes this risk genuinely fixable — unlike weather risk, which
    inflates p0 without opening any effect.
    """
    return (
        is_ground
        * _ramp(origin_congestion, *CONG_RAMP)
        * _ramp(distance_miles, *DIST_RAMP)
    )


def _ramp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Piecewise-linear 0->1 ramp between lo and hi."""
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def true_effect(
    p0: np.ndarray,
    is_ground: np.ndarray,
    origin_congestion: np.ndarray,
    distance_miles: np.ndarray,
    weather: np.ndarray,
    is_overnight: np.ndarray,
) -> np.ndarray:
    """Absolute reduction in miss probability caused by reroute assist.

    Positive = the intervention helps; negative = it hurts. This function IS
    the ground truth the learners are graded against.
    """
    gate = routing_gate(is_ground, origin_congestion, distance_miles)
    # A reroute fixes routing problems, not weather: usefulness fades to zero
    # as destination weather severity approaches 2 (real storm).
    weather_block = 1.0 - _ramp(weather.astype(float), *WEATHER_RAMP)
    effect = REL_REDUCTION_MAX * p0 * gate * weather_block
    return effect - OVERNIGHT_HARM * is_overnight


def segment_of(df: pd.DataFrame) -> pd.Series:
    """Ground-truth segment labels for the autopsy table (mutually exclusive).

    Weather wins ties on purpose: a congested long ground lane INTO a storm is
    still a shipment the reroute cannot save.
    """
    weather = df["dest_weather_severity"].to_numpy() >= 2
    overnight = (df["service_level"] == "overnight").to_numpy() & ~weather
    routing = (
        (df["service_level"] == "ground").to_numpy()
        & (df["origin_congestion"].to_numpy() >= 0.45)
        & (df["distance_miles"].to_numpy() >= 650)
        & ~weather
    )
    seg = np.full(len(df), "other", dtype=object)
    seg[routing] = "routing_driven"
    seg[overnight] = "overnight"
    seg[weather] = "weather_driven"
    return pd.Series(seg, index=df.index, name="segment_true")


def make_dataset(
    n: int = 40_000,
    seed: int = 7,
    start_date: str = "2025-03-03",
    n_days: int = 90,
    messy: bool = False,
) -> pd.DataFrame:
    """Generate one quarter of randomized-pilot shipments."""
    rng = np.random.default_rng(seed)

    day_offsets = rng.integers(0, n_days, n)
    ship_date = pd.Timestamp(start_date) + pd.to_timedelta(day_offsets, unit="D")

    service_level = rng.choice(SERVICE_LEVELS, n, p=[0.15, 0.25, 0.60])
    customer_tier = rng.choice(CUSTOMER_TIERS, n, p=[0.62, 0.24, 0.14])

    distance = rng.gamma(2.2, 330, n).clip(10, 3200)
    origin_congestion = rng.beta(2.2, 3.2, n).round(4)
    dest_congestion = rng.beta(2.2, 4.0, n).round(4)
    weather = rng.choice([0, 1, 2, 3], n, p=[0.62, 0.26, 0.09, 0.03])
    is_peak = rng.binomial(1, 0.15, n)
    is_rural = rng.binomial(1, 0.22, n)
    declared_value = rng.lognormal(3.4, 1.1, n).clip(1, 20_000).round(2)

    is_ground = (service_level == "ground").astype(float)
    is_overnight = (service_level == "overnight").astype(float)

    logit0 = (
        BASE_LOGIT
        + 1.6 * origin_congestion
        + 1.0 * dest_congestion
        + 1.15 * weather
        + 0.0009 * distance
        + 0.45 * is_peak
        + 0.40 * is_rural
        - 0.25 * is_ground
        + 0.30 * is_overnight
        + ROUTING_RISK_COEF * routing_gate(is_ground, origin_congestion, distance)
        + rng.normal(0, 0.30, n)
    )
    p0 = 1 / (1 + np.exp(-logit0))

    effect = true_effect(p0, is_ground, origin_congestion, distance, weather, is_overnight)
    p1 = np.clip(p0 - effect, 0.002, 0.98)
    true_cate = p0 - p1

    # THE design decision of this dataset: assignment is a pure coin flip,
    # independent of every feature. A targeted log (treat whatever the old
    # risk rule flagged) makes treated and control incomparable and the CATE
    # unidentifiable — the exact analogue of trying to learn price elasticity
    # from a disciplined cost-plus quote log (dynamic-pricing use case).
    treated = rng.binomial(1, PROPENSITY, n)

    p_realized = np.where(treated == 1, p1, p0)
    missed = rng.binomial(1, p_realized)

    df = pd.DataFrame(
        {
            ID_COL: [f"SHP{seed:02d}{i:08d}" for i in range(n)],
            DATE_COL: ship_date,
            "distance_miles": distance.round(1),
            "origin_congestion": origin_congestion,
            "dest_congestion": dest_congestion,
            "dest_weather_severity": weather,
            "is_peak": is_peak,
            "is_rural": is_rural,
            "declared_value_usd": declared_value,
            "service_level": service_level,
            "customer_tier": customer_tier,
            TREATMENT_COL: treated,
            LABEL_COL: missed,
            # ground truth for evaluation only; excluded from the model matrix
            # by the whitelist in models.to_matrix (asserted in tests).
            "true_cate": true_cate.round(6),
            "p0_true": p0.round(6),
        }
    )
    df["segment_true"] = segment_of(df)

    if messy:
        df = _inject_mess(df, rng)
    return df


def _inject_mess(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add the defects a real pilot extract has. cleaning.py must undo all of this."""
    df = df.copy()

    # 1. Duplicate rows (the pilot flag was logged by two systems).
    dupes = df.sample(frac=0.012, random_state=int(rng.integers(1e6)))
    df = pd.concat([df, dupes], ignore_index=True)

    # 2. Negative distances (a geocoding job that returns -1 on failure).
    idx = df.sample(frac=0.007, random_state=int(rng.integers(1e6))).index
    df.loc[idx, "distance_miles"] = -1.0

    # 3. Inconsistent categorical casing / whitespace.
    idx = df.sample(frac=0.04, random_state=int(rng.integers(1e6))).index
    df.loc[idx, "service_level"] = df.loc[idx, "service_level"].str.upper()
    idx = df.sample(frac=0.03, random_state=int(rng.integers(1e6))).index
    df.loc[idx, "customer_tier"] = " " + df.loc[idx, "customer_tier"].str.title() + " "

    return df.sample(frac=1, random_state=int(rng.integers(1e6))).reset_index(drop=True)
