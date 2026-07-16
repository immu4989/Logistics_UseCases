"""Synthetic hub-volume generator with a documented ground-truth process.

Why synthetic data? Daily inbound volume per sort facility is one of the most
tightly guarded numbers in the parcel business, so this repo ships with a
generator whose *components are known and exposed*. That gives you two things:

1. The pipeline runs end-to-end with zero external downloads.
2. The SHAP analysis can be checked against ground truth: the drivers the
   model discovers should match the components below, and the deliberately
   planted noise columns should stay buried. That check lives in the test
   suite, so a refactor that silently breaks explanations fails CI.

The generative process, per (hub, day) — multiplicative, because that is how
volume actually composes (a Sunday is ~6% of a Monday at *every* hub size):

    volume = hub_base                  fixed per-hub scale, lognormal spread
                                       across hubs (a MEM-style superhub and
                                       small regional sorts coexist)
           * trend                     +9%/year linear e-commerce growth
           * dow_profile[weekday]      Mon-heavy, Sunday near-dark
           * annual                    1 + 0.08 * sin(day of year), Q4-peaking
           * holiday_mult              fixed table: pre-holiday surge, holiday
                                       collapse, Nov 20 - Dec 22 peak ramp,
                                       day-after-Christmas crash
           * promo_mult                a few flash-sale days at 1.3-1.8x,
                                       dates and multipliers exposed below
           * weather_mult              2-3 day regional shutdowns on subsets
                                       of hubs, dates exposed below
           * exp(Normal(0, 0.07))      multiplicative noise

`hub_paint_color_code` and `moon_phase` carry exactly zero signal; they exist
so the explainability step has genuine negatives to rank below real drivers.

`make_dataset(..., messy=True)` additionally injects the defects every real
volume feed has (missing days, duplicated hub-day rows, negative and
10x-corrupted counts) so cleaning.py has real work to do.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema

# ---------------------------------------------------------------------------
# Ground-truth components, exposed so tests and the explain step can compare
# the model's SHAP ranking against reality.
# ---------------------------------------------------------------------------

# Per-hub base daily volume. Drawn once from a lognormal and frozen here so
# every seed shares the same network shape: one superhub, a mid tier, and
# small regional sorts. This spread is why evaluation reports hub-size
# terciles — a WAPE averaged across MEM and CLT hides both.
HUB_BASE = {
    "MEM": 52_000, "IND": 31_000, "ORD": 24_000, "DFW": 19_500, "ATL": 16_000,
    "EWR": 12_500, "LAX": 11_000, "OAK": 8_200, "CVG": 6_800, "PHL": 5_600,
    "SEA": 4_700, "DEN": 3_900, "MIA": 3_100, "MSP": 2_400, "CLT": 1_800,
}

TREND_PER_YEAR = 0.09  # e-commerce growth, linear

# Monday is the week's wave (weekend orders hit the network); Sunday is
# near-dark. This is the single biggest source of variation and the reason
# MAPE is the wrong headline metric here (see evaluate.py).
DOW_PROFILE = {0: 1.30, 1: 1.18, 2: 1.10, 3: 1.06, 4: 0.98, 5: 0.55, 6: 0.06}

ANNUAL_AMPLITUDE = 0.08  # sinusoid peaking in early November (Q4 build)
ANNUAL_PEAK_DOY = 305

# Fixed holiday table. Day-of-holiday multiplier reflects how hard the network
# actually shuts down; pre-holiday surge days are shoppers beating the closure.
HOLIDAYS = {
    "2023-07-04": ("independence_day", 0.35),
    "2023-09-04": ("labor_day", 0.45),
    "2023-11-23": ("thanksgiving", 0.10),
    "2023-12-25": ("christmas", 0.05),
    "2024-01-01": ("new_year", 0.30),
    "2024-05-27": ("memorial_day", 0.45),
    "2024-07-04": ("independence_day", 0.35),
    "2024-09-02": ("labor_day", 0.45),
    "2024-11-28": ("thanksgiving", 0.10),
    "2024-12-25": ("christmas", 0.05),
    "2025-01-01": ("new_year", 0.30),
    "2025-05-26": ("memorial_day", 0.45),
    "2025-07-04": ("independence_day", 0.35),
    "2025-09-01": ("labor_day", 0.45),
    "2025-11-27": ("thanksgiving", 0.10),
    "2025-12-25": ("christmas", 0.05),
    "2026-01-01": ("new_year", 0.30),  # just past the data window; kept so
                                       # days-to-holiday is defined at the edge
}
PRE_HOLIDAY_SURGE = {1: 1.22, 2: 1.15, 3: 1.08}  # days before any holiday
DAY_AFTER_CHRISTMAS = 0.45  # the famous crash; Dec 27 recovers partway
DEC_27_RECOVERY = 0.75

# Peak ramp: volume climbs from Nov 20, tops out Dec 18-22, at up to +55%.
PEAK_RAMP_START = (11, 20)
PEAK_RAMP_FULL = (12, 18)
PEAK_RAMP_END = (12, 22)
PEAK_RAMP_MAX = 0.55

# Flash-sale days (marketing publishes this calendar weeks ahead, which is why
# features.py exposes it to the model — see is_promo_day). Network-wide.
PROMO_EVENTS = {
    "2023-10-10": 1.45,
    "2024-03-19": 1.30,
    "2024-07-16": 1.65,
    "2024-10-08": 1.50,
    "2025-03-25": 1.35,
    "2025-07-15": 1.70,
    "2025-10-07": 1.55,  # falls in the held-out test window on purpose
}

# Regional weather shutdowns: (start, end inclusive, affected hubs, multiplier).
# Unlike promos these are NOT visible to the model in advance — they are the
# honest reason P10-P90 coverage cannot hit 100%.
WEATHER_EVENTS = [
    ("2024-01-15", "2024-01-17", ["ORD", "IND", "MSP", "CVG", "DEN"], 0.35),
    ("2024-12-10", "2024-12-11", ["EWR", "PHL", "CLT"], 0.50),
    ("2025-02-03", "2025-02-05", ["ORD", "MSP", "DEN", "IND", "CVG", "SEA"], 0.40),
    ("2025-11-30", "2025-12-01", ["EWR", "PHL", "MIA", "CLT"], 0.55),  # in test
]

NOISE_SIGMA = 0.07

TRUE_COMPONENTS = {
    "hub_base": HUB_BASE,
    "trend_per_year": TREND_PER_YEAR,
    "dow_profile": DOW_PROFILE,
    "annual_amplitude": ANNUAL_AMPLITUDE,
    "holidays": HOLIDAYS,
    "pre_holiday_surge": PRE_HOLIDAY_SURGE,
    "peak_ramp_max": PEAK_RAMP_MAX,
    "promo_events": PROMO_EVENTS,
    "weather_events": WEATHER_EVENTS,
    "noise_sigma": NOISE_SIGMA,
    "noise_features": schema.NOISE_FEATURES,
}

# Static per-hub nonsense, so the "noise stays buried" test has a categorical-
# looking negative as well as a smooth one (moon_phase).
HUB_PAINT_COLOR = {h: (i * 7) % 9 + 1 for i, h in enumerate(schema.HUBS)}


def holiday_multiplier(dates: pd.DatetimeIndex) -> np.ndarray:
    """Combined holiday-table effect for a date index (vectorised)."""
    mult = np.ones(len(dates))
    holiday_ts = {pd.Timestamp(d): m for d, (_, m) in HOLIDAYS.items()}

    for i, d in enumerate(dates):
        if d in holiday_ts:
            mult[i] *= holiday_ts[d]
        else:
            for lead, surge in PRE_HOLIDAY_SURGE.items():
                if (d + pd.Timedelta(days=lead)) in holiday_ts:
                    mult[i] *= surge
                    break
        if (d.month, d.day) == (12, 26):
            mult[i] *= DAY_AFTER_CHRISTMAS
        elif (d.month, d.day) == (12, 27):
            mult[i] *= DEC_27_RECOVERY

    # Peak ramp, same shape every year: linear climb Nov 20 -> Dec 18, hold
    # through Dec 22. Compounds with Thanksgiving (which falls inside it).
    start = pd.Series([pd.Timestamp(d.year, *PEAK_RAMP_START) for d in dates])
    full = pd.Series([pd.Timestamp(d.year, *PEAK_RAMP_FULL) for d in dates])
    end = pd.Series([pd.Timestamp(d.year, *PEAK_RAMP_END) for d in dates])
    days_in = (pd.Series(dates) - start).dt.days.to_numpy().astype(float)
    climb_len = (full - start).dt.days.to_numpy().astype(float)
    in_ramp = (pd.Series(dates) >= start) & (pd.Series(dates) <= end)
    doy_pos = np.clip(days_in / climb_len, 0.0, 1.0) * in_ramp.to_numpy()
    return mult * (1.0 + PEAK_RAMP_MAX * doy_pos)


def make_dataset(
    start_date: str = "2023-07-03",
    n_days: int = 912,
    seed: int = 7,
    messy: bool = False,
) -> pd.DataFrame:
    """Generate daily volume for every hub over ``n_days`` (~2.5 years).

    Default window: 2023-07-03 (a Monday) through 2025-12-31, so the final
    four months — the natural held-out test period — contain a full December
    peak. Same seed, same frame, byte for byte.
    """
    dates = pd.date_range(start_date, periods=n_days, freq="D")
    rng = np.random.default_rng(seed)

    # Date-level components (shared across hubs).
    years_elapsed = np.arange(n_days) / 365.25
    trend = 1.0 + TREND_PER_YEAR * years_elapsed
    dow = dates.dayofweek.to_numpy()
    dow_mult = np.array([DOW_PROFILE[d] for d in dow])
    doy = dates.dayofyear.to_numpy()
    annual = 1.0 + ANNUAL_AMPLITUDE * np.sin(
        2 * np.pi * (doy - (ANNUAL_PEAK_DOY - 365.25 / 4)) / 365.25
    )
    hol_mult = holiday_multiplier(dates)
    promo_mult = np.ones(n_days)
    for d, m in PROMO_EVENTS.items():
        promo_mult[dates == pd.Timestamp(d)] = m
    date_mult = trend * dow_mult * annual * hol_mult * promo_mult

    frames = []
    for hub in schema.HUBS:
        vol = HUB_BASE[hub] * date_mult * np.exp(rng.normal(0, NOISE_SIGMA, n_days))
        frames.append(
            pd.DataFrame(
                {
                    schema.HUB_COL: hub,
                    schema.DATE_COL: dates,
                    schema.TARGET_COL: vol,
                    "hub_paint_color_code": HUB_PAINT_COLOR[hub],
                    # ~29.5-day lunar cycle, smooth in [0, 1]. Zero true signal.
                    "moon_phase": (0.5 + 0.5 * np.sin(2 * np.pi * np.arange(n_days) / 29.53)),
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)

    # Weather shutdowns hit subsets of hubs for 2-3 days.
    for start, end, hubs, m in WEATHER_EVENTS:
        mask = (
            df[schema.HUB_COL].isin(hubs)
            & (df[schema.DATE_COL] >= pd.Timestamp(start))
            & (df[schema.DATE_COL] <= pd.Timestamp(end))
        )
        df.loc[mask, schema.TARGET_COL] *= m

    df[schema.TARGET_COL] = df[schema.TARGET_COL].round().astype(int)
    df["moon_phase"] = df["moon_phase"].round(4)

    if messy:
        df = _inject_mess(df, rng)
    return df.reset_index(drop=True)


def _inject_mess(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add the defects real volume feeds have. cleaning.py must handle all of this."""
    df = df.copy()

    # 1. Missing days: random feed drops, plus one contiguous 4-day outage for
    #    a single hub (an integration going dark is how gaps arrive in practice).
    drop_idx = df.sample(frac=0.012, random_state=int(rng.integers(1e6))).index
    df = df.drop(drop_idx)
    outage_start = df[schema.DATE_COL].min() + pd.Timedelta(days=200)
    outage = (df[schema.HUB_COL] == "OAK") & df[schema.DATE_COL].between(
        outage_start, outage_start + pd.Timedelta(days=3)
    )
    df = df[~outage]

    # 2. Duplicated (hub, day) rows: the upstream job re-ran and appended.
    dupes = df.sample(frac=0.006, random_state=int(rng.integers(1e6)))
    df = pd.concat([df, dupes], ignore_index=True)

    # 3. Impossible values: negatives (reversal records) and 10x fat-finger
    #    counts (a unit mix-up, pieces vs containers).
    idx = df.sample(frac=0.0015, random_state=int(rng.integers(1e6))).index
    df.loc[idx, schema.TARGET_COL] = -df.loc[idx, schema.TARGET_COL]
    idx = df.sample(frac=0.0015, random_state=int(rng.integers(1e6))).index
    df.loc[idx, schema.TARGET_COL] = df.loc[idx, schema.TARGET_COL] * 10

    return df.sample(frac=1, random_state=int(rng.integers(1e6))).reset_index(drop=True)
