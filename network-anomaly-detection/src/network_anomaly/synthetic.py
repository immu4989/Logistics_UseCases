"""Synthetic daily lane-level OTP data with documented ground-truth anomalies.

Why synthetic? The same reason as the delivery-commit use case next door:
carrier network data is proprietary, and a generator with *known* injected
anomalies lets the test suite assert that the detector catches real drift and
stays quiet on clean lanes. A monitoring system you can't score against ground
truth is a monitoring system you'll only evaluate during an outage.

The network:

- 12 hubs, 120 directed lanes (hub pairs).
- Each lane has a stable base miss rate drawn from Beta(4, 46) — mean ~8%,
  realistic spread (roughly 3% to 16% across lanes).
- Daily volume is lognormal around a per-lane base. The base volumes are a
  two-tier mixture: a handful of trunk lanes moving thousands of shipments a
  day, and a long tail of thin lanes moving 5-30. The thin lanes are the heart
  of this use case — at 10 shipments/day, one bad afternoon is a 20-point
  rate swing, and any detector that treats that like a trunk-lane signal will
  page ops into numbness.
- Daily misses ~ Binomial(volume, rate).
- A shared day-of-week effect (weekends slightly worse) and two network-wide
  surge windows. The windows are placed so that BOTH the detector's baseline
  period and the monitoring period contain surge conditions — if surges only
  existed after the baseline, every detector would learn a calm network and
  alarm on the first storm. (Same placement trick as the peak windows in
  delivery-commit-prediction's generator.) These effects are GLOBAL: a good
  detector must not fire 120 lane alarms for a bad-weather week.

Injected ground-truth anomalies (INJECTED_ANOMALIES, one dict per anomaly):

- "step":  a sudden persistent jump of +7 to +15 percentage points on 8 lanes,
  starting at various dates in the second half of the year. The classic
  silent failure: a linehaul schedule change, a sort re-plan gone wrong.
  Magnitudes are paired inversely with lane volume — the thinnest step lanes
  get the biggest jumps, because a +6pp step on a 6-shipment/day lane is
  statistically invisible in any three-week window no matter the method.
  Detecting it at all is what the shrinkage + CUSUM machinery buys.
- "ramp":  rate climbing ~1-1.5pp per week for RAMP_WEEKS weeks on 4 lanes,
  then holding. Slow decay (staffing erosion, creeping congestion) that
  monthly aggregates average away until it's a quarter-long problem.
- "spike": a single bad day (+20 to +35pp) on 5 lanes — a storm. A good
  detector should NOT page for one bad day that self-recovers; alarms on a
  spike lane within a few days of the spike are treated as acceptable but
  tracked, and the tradeoff is documented in the README.

Everything not listed stays clean all year and is the false-alarm denominator.

The network topology and anomaly placement come from a FIXED internal seed, so
INJECTED_ANOMALIES is a module-level constant the tests can import. The `seed`
argument to make_dataset controls only the stochastic realization (volumes,
binomial draws, mess), which is exactly what "check on another seed" should
re-roll.

`messy=True` adds the defects a real daily feed has: missing days for some
lanes, duplicated (lane, day) rows, and a few rows where misses > volume.
cleaning.py must fix each, audited.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema

START_DATE = "2025-01-01"
N_DAYS = 365
HUBS = ["ATL", "CLT", "DEN", "DFW", "EWR", "LAX", "MEM", "MIA", "MSP", "ORD", "PHX", "SEA"]
N_LANES = 120
RAMP_WEEKS = 10

# Additive percentage-point day-of-week effect, Mon..Sun. Weekends run worse
# network-wide (thinner staffing, compressed sort windows).
DOW_EFFECT_PP = np.array([0.002, 0.0, 0.0, 0.0, 0.004, 0.011, 0.014])

# Two network-wide surge windows (spring promo + fall peak ramp), +3pp on every
# lane and +25% volume. One sits inside the detector's 90-day baseline, one in
# the monitoring period — see module docstring for why both matter.
SURGE_WINDOWS = [(55, 69), (235, 249)]
SURGE_EFFECT_PP = 0.030
SURGE_VOLUME_MULT = 1.25


def _build_network():
    """Fixed network topology + anomaly placement (independent of data seed)."""
    rng = np.random.default_rng(20250101)

    pairs = [(o, d) for o in HUBS for d in HUBS if o != d]  # 132 candidates
    chosen = rng.choice(len(pairs), size=N_LANES, replace=False)
    lanes = sorted(f"{pairs[i][0]}-{pairs[i][1]}" for i in chosen)

    base_rate = rng.beta(4.0, 46.0, N_LANES)  # mean 8%, sd ~3.8pp

    # Two-tier volume mixture: ~10 trunk lanes in the thousands, the rest a
    # lognormal tail centered near 18/day (many land in the 5-30 band).
    is_trunk = rng.random(N_LANES) < 0.09
    trunk_vol = np.exp(rng.normal(np.log(1800), 0.45, N_LANES))
    thin_vol = np.exp(rng.normal(np.log(18), 0.85, N_LANES))
    base_volume = np.where(is_trunk, trunk_vol, thin_vol).clip(5, 6000)

    # --- anomaly placement -------------------------------------------------
    anom_idx = rng.choice(N_LANES, size=17, replace=False)
    step_idx, ramp_idx, spike_idx = anom_idx[:8], anom_idx[8:12], anom_idx[12:]

    anomalies: list[dict] = []
    step_mags = np.sort(rng.uniform(0.07, 0.15, 8))          # ascending
    vol_order = np.argsort(base_volume[step_idx])[::-1]       # biggest lane first
    for rank, j in enumerate(vol_order):                      # biggest lane -> smallest mag
        anomalies.append(
            {
                "lane": lanes[step_idx[j]],
                "type": "step",
                "start_day": int(rng.integers(190, 320)),
                "magnitude": float(step_mags[rank]),
            }
        )
    for i in ramp_idx:
        anomalies.append(
            {
                "lane": lanes[i],
                "type": "ramp",
                "start_day": int(rng.integers(150, 230)),
                "magnitude": float(rng.uniform(0.010, 0.016)),  # pp per week
            }
        )
    for i in spike_idx:
        anomalies.append(
            {
                "lane": lanes[i],
                "type": "spike",
                "start_day": int(rng.integers(100, 350)),
                "magnitude": float(rng.uniform(0.20, 0.35)),
            }
        )

    rates = dict(zip(lanes, base_rate))
    volumes = dict(zip(lanes, base_volume))
    return lanes, rates, volumes, anomalies


LANES, LANE_BASE_RATE, LANE_BASE_VOLUME, INJECTED_ANOMALIES = _build_network()


def anomaly_lanes(kinds: tuple[str, ...] = ("step", "ramp", "spike")) -> set[str]:
    return {a["lane"] for a in INJECTED_ANOMALIES if a["type"] in kinds}


def clean_lanes() -> list[str]:
    """Lanes with no injected anomaly: the false-alarm denominator."""
    bad = anomaly_lanes()
    return [ln for ln in LANES if ln not in bad]


def _anomaly_pp(day: np.ndarray) -> np.ndarray:
    """(n_lanes, n_days) additive percentage points from injected anomalies."""
    add = np.zeros((N_LANES, len(day)))
    lane_row = {ln: i for i, ln in enumerate(LANES)}
    for a in INJECTED_ANOMALIES:
        row, start, mag = lane_row[a["lane"]], a["start_day"], a["magnitude"]
        if a["type"] == "step":
            add[row, day >= start] += mag
        elif a["type"] == "ramp":
            weeks = np.clip((day - start) / 7.0, 0, RAMP_WEEKS)
            add[row] += mag * weeks * (day >= start)
        elif a["type"] == "spike":
            add[row, day == start] += mag
    return add


def make_dataset(seed: int = 7, n_days: int = N_DAYS, messy: bool = False) -> pd.DataFrame:
    """Generate one year of daily lane-level OTP rows in the canonical schema."""
    rng = np.random.default_rng(seed)

    day = np.arange(n_days)
    dates = pd.Timestamp(START_DATE) + pd.to_timedelta(day, unit="D")
    dow = dates.dayofweek.to_numpy()

    surge = np.zeros(n_days, dtype=bool)
    for lo, hi in SURGE_WINDOWS:
        surge |= (day >= lo) & (day <= hi)

    base_rate = np.array([LANE_BASE_RATE[ln] for ln in LANES])[:, None]
    base_vol = np.array([LANE_BASE_VOLUME[ln] for ln in LANES])[:, None]

    # Daily volume: lognormal noise around the lane base, surge multiplier.
    vol_mult = np.where(surge, SURGE_VOLUME_MULT, 1.0)[None, :]
    volume = np.maximum(
        1, np.round(base_vol * vol_mult * np.exp(rng.normal(0.0, 0.25, (N_LANES, n_days))))
    ).astype(int)

    rate = np.clip(
        base_rate
        + DOW_EFFECT_PP[dow][None, :]
        + np.where(surge, SURGE_EFFECT_PP, 0.0)[None, :]
        + _anomaly_pp(day),
        0.002,
        0.90,
    )
    misses = rng.binomial(volume, rate)

    df = pd.DataFrame(
        {
            schema.LANE_COL: np.repeat(LANES, n_days),
            schema.DATE_COL: np.tile(dates, N_LANES),
            schema.VOLUME_COL: volume.ravel(),
            schema.MISSES_COL: misses.ravel(),
        }
    )

    if messy:
        df = _inject_mess(df, rng)
    return df.reset_index(drop=True)


def _inject_mess(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add the defects a real daily feed has. cleaning.py must undo each one."""
    df = df.copy()

    # 1. Missing days: some lanes simply don't report for a stretch (feed
    #    outage at the origin hub). The detector must tolerate gaps, so
    #    cleaning logs these but does NOT invent rows for them.
    lane_pick = rng.choice(LANES, size=12, replace=False)
    drop_idx = []
    for ln in lane_pick:
        rows = df.index[df[schema.LANE_COL] == ln].to_numpy()
        start = int(rng.integers(0, len(rows) - 10))
        drop_idx.extend(rows[start : start + int(rng.integers(3, 10))])
    df = df.drop(index=drop_idx)

    # 2. Duplicated (lane, day) rows: the extract job ran twice.
    dupes = df.sample(n=40, random_state=int(rng.integers(1e6)))
    df = pd.concat([df, dupes], ignore_index=True)

    # 3. misses > volume: a join defect upstream double-counts exceptions.
    idx = df.sample(n=25, random_state=int(rng.integers(1e6))).index
    df.loc[idx, schema.MISSES_COL] = df.loc[idx, schema.VOLUME_COL] + rng.integers(
        1, 15, len(idx)
    )

    return df.sample(frac=1, random_state=int(rng.integers(1e6))).reset_index(drop=True)
