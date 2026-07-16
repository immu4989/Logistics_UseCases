"""Synthetic weekly lane-demand generator with a documented ground-truth process.

Why synthetic data? Lane-level linehaul demand is one of the most closely held
numbers in a trucking network (it is the input to every carrier's rate
negotiation), so this repo ships with a generator whose *components are known
and exposed*. That gives you two things:

1. The pipeline runs end-to-end with zero external downloads.
2. The booking policies can be judged against a true **oracle**: because the
   demand distribution of every (lane, week) is known exactly, the best
   possible newsvendor booking is computable, and every policy's cost gap to
   it is a real number rather than a guess. The test suite leans on this.

The generative process, per (lane, week) — multiplicative, because that is how
freight demand composes (a peak week is +40% on a 60-trailer lane and on a
4-trailer lane alike):

    demand = lane_base                 fixed per-lane weekly level; big
                                       corridor lanes and thin spokes coexist
           * trend                     per-lane linear growth or decline
                                       (growing e-commerce lanes, a couple of
                                       fading industrial ones)
           * seasonal                  gentle annual sinusoid, autumn-peaking
           * peak_mult[week-of-year]   fixed table: the Nov/Dec ramp up to
                                       +38%, the Christmas-week shutdown, the
                                       January lull
           * promo_mult                a few network promo weeks at 1.1-1.3x,
                                       dates and multipliers exposed below
           * exp(Normal(0, sigma))     lane-specific lognormal noise; thin
                                       lanes are relatively noisier
           * shock                     rare demand shocks: a surge (a
                                       competitor failure, a customer win) or
                                       a slump (a plant shutdown)

Demand is a *continuous* number of trailer-equivalents (23.6 means the last
trailer cubes out at 60%); bookings are discrete trailers, which is why every
policy in decide.py rounds up.

`make_dataset(..., messy=True)` additionally injects the defects every real
demand feed has (duplicated lane-weeks, negative corruptions, missing weeks)
so cleaning.py has real work to do. Mess is injected *after* all demand draws,
so `make_dataset(seed, messy=False)` is the exact uncorrupted truth behind the
messy feed with the same seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Canonical columns. The unit of prediction is a (lane, week) pair.
# ---------------------------------------------------------------------------
LANE_COL = "lane_id"
WEEK_COL = "week_start"  # Monday of the demand week
TARGET_COL = "demand_teq"  # trailer-equivalents of freight tendered that week

# ---------------------------------------------------------------------------
# Ground-truth components, exposed so tests, the oracle policy and the explain
# step can compare every decision against reality.
# ---------------------------------------------------------------------------

# Per-lane base weekly demand (trailer-equivalents). Frozen so every seed
# shares the same network shape: a few dense corridors, a long thin tail.
LANE_BASE = {
    "MEM-ORD": 62, "MEM-DFW": 55, "MEM-ATL": 48, "ORD-EWR": 44, "DFW-LAX": 41,
    "ATL-MIA": 36, "ORD-MSP": 33, "MEM-EWR": 30, "LAX-OAK": 28, "ATL-CLT": 26,
    "DFW-HOU": 24, "ORD-CVG": 22, "EWR-BOS": 21, "LAX-PHX": 19, "ATL-MCO": 18,
    "MEM-IND": 16, "ORD-DTW": 15, "DFW-SAT": 14, "OAK-SEA": 13, "EWR-PHL": 12,
    "CVG-CMH": 11, "MIA-TPA": 10, "MSP-FAR": 9, "CLT-RDU": 8, "IND-SDF": 7,
    "PHX-ABQ": 6, "SEA-PDX": 6, "BOS-PWM": 5, "SDF-BNA": 4, "TPA-JAX": 3,
}
LANES = list(LANE_BASE)

# Per-lane linear trend, per year. Sun-belt e-commerce lanes grow; two
# industrial lanes fade. Every other lane is flat.
TREND_PER_YEAR = {
    "LAX-PHX": 0.20, "ATL-MCO": 0.16, "OAK-SEA": 0.14, "EWR-BOS": 0.12,
    "ORD-CVG": -0.10, "MSP-FAR": -0.13,
}

# Trend is anchored to a fixed epoch, not to the requested window, so the same
# calendar week always has the same expected demand regardless of the caller.
EPOCH = pd.Timestamp("2024-01-01")

SEASONAL_AMPLITUDE = 0.05  # annual sinusoid, peaking around ISO week 40

# Year-end peak table by ISO week: the pre-Christmas ramp, the Christmas-week
# shutdown, the January lull. Fixed so the shape repeats every year.
PEAK_WEEK_MULT = {46: 1.08, 47: 1.15, 48: 1.24, 49: 1.32, 50: 1.38, 51: 1.30, 52: 0.82, 1: 0.88}
PEAK_WOY = tuple(range(46, 53))  # the weeks the peak breakout in evaluate.py reports on

# Network-wide promo weeks (marketing publishes this calendar weeks ahead,
# which is why forecast.py exposes it to the model as is_promo_week).
PROMO_EVENTS = {
    "2024-03-11": 1.12,
    "2024-07-08": 1.22,
    "2024-10-07": 1.28,
    "2025-03-10": 1.14,
    "2025-07-07": 1.24,
    "2025-10-06": 1.30,  # falls in the held-out test window on purpose
}

# Lane noise: thin lanes are relatively noisier (one customer's tender swing
# moves a 4-trailer lane far more than a 60-trailer corridor).
NOISE_SIGMA_FLOOR = 0.10
NOISE_SIGMA_SCALE = 0.30  # sigma = FLOOR + SCALE / sqrt(lane_base)

# Rare demand shocks, per lane-week. Not knowable in advance — the honest
# reason no booking policy reaches 100% service on committed capacity.
SHOCK_UP_PROB, SHOCK_UP_RANGE = 0.012, (1.35, 1.90)
SHOCK_DOWN_PROB, SHOCK_DOWN_RANGE = 0.006, (0.45, 0.75)

TRUE_COMPONENTS = {
    "lane_base": LANE_BASE,
    "trend_per_year": TREND_PER_YEAR,
    "seasonal_amplitude": SEASONAL_AMPLITUDE,
    "peak_week_mult": PEAK_WEEK_MULT,
    "promo_events": PROMO_EVENTS,
    "noise_sigma": (NOISE_SIGMA_FLOOR, NOISE_SIGMA_SCALE),
    "shock_up": (SHOCK_UP_PROB, SHOCK_UP_RANGE),
    "shock_down": (SHOCK_DOWN_PROB, SHOCK_DOWN_RANGE),
}


def lane_sigma(lane: str) -> float:
    """Lognormal noise sigma for a lane; thin lanes are relatively noisier."""
    return NOISE_SIGMA_FLOOR + NOISE_SIGMA_SCALE / np.sqrt(LANE_BASE[lane])


def demand_level(weeks: pd.DatetimeIndex) -> pd.DataFrame:
    """Deterministic demand level per (lane, week): every component except noise.

    The level is the *median* of the no-shock demand distribution; the oracle
    and the evaluation simulator both build on it.
    """
    woy = weeks.isocalendar().week.to_numpy().astype(int)
    seasonal = 1.0 + SEASONAL_AMPLITUDE * np.sin(2 * np.pi * (woy - 27) / 52.18)
    peak = np.array([PEAK_WEEK_MULT.get(w, 1.0) for w in woy])
    promo = np.ones(len(weeks))
    for d, m in PROMO_EVENTS.items():
        promo[weeks == pd.Timestamp(d)] = m
    years = (weeks - EPOCH).days.to_numpy() / 365.25

    frames = []
    for lane in LANES:
        trend = 1.0 + TREND_PER_YEAR.get(lane, 0.0) * years
        frames.append(
            pd.DataFrame(
                {
                    LANE_COL: lane,
                    WEEK_COL: weeks,
                    "level": LANE_BASE[lane] * trend * seasonal * peak * promo,
                    "sigma": lane_sigma(lane),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _draw_multipliers(sigma: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw noise-times-shock multipliers, one per sigma entry."""
    mult = np.exp(sigma * rng.normal(0.0, 1.0, len(sigma)))
    u = rng.random(len(sigma))
    up = u < SHOCK_UP_PROB
    down = (u >= SHOCK_UP_PROB) & (u < SHOCK_UP_PROB + SHOCK_DOWN_PROB)
    mult[up] *= rng.uniform(*SHOCK_UP_RANGE, up.sum())
    mult[down] *= rng.uniform(*SHOCK_DOWN_RANGE, down.sum())
    return mult


def make_dataset(
    start_week: str = "2024-01-01",
    n_weeks: int = 108,
    seed: int = 7,
    messy: bool = False,
) -> pd.DataFrame:
    """Generate weekly demand for every lane over ``n_weeks`` (~2 years).

    Default window: 2024-01-01 (a Monday) through the week of 2026-01-19, so
    the final ~16 weeks — the natural held-out test period — contain a full
    year-end peak. Same seed, same frame, byte for byte.
    """
    weeks = pd.date_range(start_week, periods=n_weeks, freq="7D")
    rng = np.random.default_rng(seed)

    level = demand_level(weeks)
    demand = level["level"].to_numpy() * _draw_multipliers(level["sigma"].to_numpy(), rng)
    df = pd.DataFrame(
        {
            LANE_COL: level[LANE_COL],
            WEEK_COL: level[WEEK_COL],
            TARGET_COL: demand.round(2),
        }
    )
    if messy:
        df = _inject_mess(df, rng)
    return df.reset_index(drop=True)


def _inject_mess(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add the defects real demand feeds have. cleaning.py must handle all of this."""
    df = df.copy()

    # 1. Missing weeks: random feed drops, plus one contiguous 5-week outage
    #    for a single lane (a TMS integration going dark, early in the window
    #    so the outage sits in training history, not the test period).
    drop_idx = df.sample(frac=0.015, random_state=int(rng.integers(1e6))).index
    df = df.drop(drop_idx)
    outage_start = df[WEEK_COL].min() + pd.Timedelta(weeks=30)
    outage = (df[LANE_COL] == "ATL-CLT") & df[WEEK_COL].between(
        outage_start, outage_start + pd.Timedelta(weeks=4)
    )
    df = df[~outage]

    # 2. Duplicated (lane, week) rows: the upstream job re-ran and appended.
    dupes = df.sample(frac=0.015, random_state=int(rng.integers(1e6)))
    df = pd.concat([df, dupes], ignore_index=True)

    # 3. Negative demand: tender-reversal records netted into the wrong week.
    idx = df.sample(frac=0.008, random_state=int(rng.integers(1e6))).index
    df.loc[idx, TARGET_COL] = -df.loc[idx, TARGET_COL]

    return df.sample(frac=1, random_state=int(rng.integers(1e6))).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Ground-truth demand distribution: quantiles for the oracle, draws for the
# counterfactual cost evaluation. The multiplier distribution depends only on
# the lane (sigma and the shock mixture), so one big Monte Carlo sample per
# sigma — under a FIXED seed, independent of any dataset seed — prices every
# week of that lane.
# ---------------------------------------------------------------------------
_MC_SEED = 20_260_701
_MC_SIZE = 40_000
_MC_CACHE: dict[float, np.ndarray] = {}


def _multiplier_sample(sigma: float) -> np.ndarray:
    key = round(float(sigma), 6)
    if key not in _MC_CACHE:
        rng = np.random.default_rng(_MC_SEED)
        _MC_CACHE[key] = np.sort(_draw_multipliers(np.full(_MC_SIZE, key), rng))
    return _MC_CACHE[key]


def true_demand_quantile(lanes: pd.Series, weeks: pd.Series, q: float) -> np.ndarray:
    """Exact demand quantile ``q`` per (lane, week) under the true process."""
    level = demand_level(pd.DatetimeIndex(sorted(weeks.unique())))
    keyed = level.set_index([LANE_COL, WEEK_COL])["level"]
    lv = keyed.loc[list(zip(lanes, pd.to_datetime(weeks)))].to_numpy()
    mult_q = np.array([np.quantile(_multiplier_sample(lane_sigma(ln)), q) for ln in lanes])
    return lv * mult_q


def simulate_demand(frame: pd.DataFrame, seed: int, n_reps: int) -> np.ndarray:
    """Draw ``n_reps`` demand realizations per (lane, week) row of ``frame``.

    Returns an (n_rows, n_reps) matrix. The seed stream is offset so these
    draws never collide with generator randomness, and evaluate.py calls this
    exactly once — every policy is costed against the SAME matrix (common
    random numbers; see evaluate.py for why that matters).
    """
    level = demand_level(pd.DatetimeIndex(sorted(frame[WEEK_COL].unique())))
    keyed = level.set_index([LANE_COL, WEEK_COL])
    rows = keyed.loc[list(zip(frame[LANE_COL], pd.to_datetime(frame[WEEK_COL])))]
    lv = rows["level"].to_numpy()[:, None]
    sigma = np.repeat(rows["sigma"].to_numpy(), n_reps)
    rng = np.random.default_rng(seed + 1_000_003)
    return lv * _draw_multipliers(sigma, rng).reshape(len(frame), n_reps)
