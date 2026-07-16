"""The drift detector: empirical-Bayes shrinkage + global-effect removal + CUSUM.

Three stages, each earning its place operationally:

1. **Empirical-Bayes shrinkage** for each lane's expected rate. Fit a Beta
   prior to the distribution of lane rates over the first `baseline_days`
   days (method of moments), then set each lane's expected rate to its
   posterior mean: (alpha + misses) / (alpha + beta + volume). A trunk lane
   with 150k baseline shipments keeps its own rate almost exactly; a
   10-shipment/day lane gets pulled toward the network prior. This is what
   makes thin lanes monitorable at all — their raw 90-day rate can easily be
   off by several points, and a detector standardized against a wrong
   expectation either alarms constantly or never. Small lanes lie; the prior
   is how you stop believing them uncritically.

2. **Global daily effect removal.** Each day, compute the volume-weighted
   network-wide deviation from expectation and subtract it from every lane.
   A bad-weather Tuesday or a peak-season surge moves the whole network; that
   is a capacity conversation, not 120 separate lane incidents, and a
   detector that pages every lane during a surge gets muted within a month.
   Two robustness guards keep a genuinely-broken trunk lane from laundering
   its own step change into "network weather" and hiding from itself: a lane
   whose CUSUM is already elevated (> h/2) loses its vote on what "network
   normal" is until it recovers, and per-lane deviations are clipped
   (± `robust_clip_pp`) so the first pre-suspicion days of a large break
   can't move the estimate much either.

3. **CUSUM** on the standardized daily residual
   z = (observed - expected) / binomial std of the daily rate at the expected
   level. We track both one-sided CUSUMs but only ALARM on the upper one:
   ops pages when lanes get worse, and improvement is a report, not a page.
   S_t = max(0, S_{t-1} + z_t - k); alarm when S_t >= h, then reset. The
   slack k ignores drift smaller than ~k standard errors per day; the
   threshold h is the alarm budget. Unlike a threshold on the daily rate,
   the CUSUM has memory: a persistent +1 sigma shift accumulates ~0.5/day
   and crosses h in ~10 days, while single bad days decay away. Missing days
   simply don't update the statistic.

The `baseline_monthly` reference detector is the status quo being replaced:
at each calendar month end, flag lanes whose monthly aggregate rate exceeds
expected + 2 sigma. Its detection day is the month end, because that is when
a monthly OTP report exists. It even gets the same global correction (a
seasonally-adjusted monthly report is a generous strawman); the CUSUM has to
beat it on detection delay, not on a handicap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import schema


@dataclass
class DetectorConfig:
    baseline_days: int = 90   # days used to fit the prior + lane expectations
    k: float = 0.5            # CUSUM slack, in daily standard errors
    h: float = 5.5            # CUSUM alarm threshold (the alarm budget knob)
    robust_clip_pp: float = 0.06  # clip per-lane deviations entering the global estimate


@dataclass
class DetectionResult:
    config: DetectorConfig
    lanes: list[str]
    dates: pd.DatetimeIndex          # full daily grid, day 0 = first date
    prior_alpha: float
    prior_beta: float
    expected: pd.Series              # per-lane expected rate (posterior mean)
    volume: np.ndarray               # (lanes, days), NaN where missing
    obs_rate: np.ndarray             # (lanes, days), NaN where missing
    p0: np.ndarray                   # (lanes, days) expected rate incl. global effect
    z: np.ndarray                    # (lanes, days) standardized residual
    cusum: np.ndarray                # (lanes, days) upper CUSUM statistic
    alarms: pd.DataFrame = field(default_factory=pd.DataFrame)
    monthly_alarms: pd.DataFrame = field(default_factory=pd.DataFrame)

    def day_index(self, date: pd.Timestamp) -> int:
        return int((pd.Timestamp(date) - self.dates[0]).days)


def _pivot(df: pd.DataFrame, dates: pd.DatetimeIndex, lanes: list[str]):
    """(lanes, days) matrices of volume and misses, NaN on missing days."""
    vol = (
        df.pivot(index=schema.LANE_COL, columns=schema.DATE_COL, values=schema.VOLUME_COL)
        .reindex(index=lanes, columns=dates)
        .to_numpy(dtype=float)
    )
    mis = (
        df.pivot(index=schema.LANE_COL, columns=schema.DATE_COL, values=schema.MISSES_COL)
        .reindex(index=lanes, columns=dates)
        .to_numpy(dtype=float)
    )
    return vol, mis


def fit_beta_prior(rates: np.ndarray) -> tuple[float, float]:
    """Method-of-moments Beta fit to the cross-lane distribution of base rates."""
    m, v = float(np.mean(rates)), float(np.var(rates))
    v = max(v, 1e-6)
    ab = max(m * (1 - m) / v - 1.0, 2.0)  # floor keeps the prior proper if lanes are near-equal
    return m * ab, (1 - m) * ab


def detect(df: pd.DataFrame, config: DetectorConfig | None = None) -> DetectionResult:
    """Run the full detector on a cleaned daily lane feed."""
    cfg = config or DetectorConfig()
    lanes = sorted(df[schema.LANE_COL].unique())
    dates = pd.date_range(df[schema.DATE_COL].min(), df[schema.DATE_COL].max(), freq="D")
    vol, mis = _pivot(df, dates, lanes)
    n_days = len(dates)
    base = slice(0, cfg.baseline_days)

    # --- stage 1: empirical-Bayes expected rate per lane --------------------
    base_vol = np.nansum(vol[:, base], axis=1)
    base_mis = np.nansum(mis[:, base], axis=1)
    raw_rates = base_mis / np.maximum(base_vol, 1.0)
    alpha, beta = fit_beta_prior(raw_rates)
    expected = (alpha + base_mis) / (alpha + beta + base_vol)

    # --- stages 2 + 3: global-effect removal, residuals, CUSUM --------------
    # One pass over days: the global estimate for day d excludes lanes whose
    # CUSUM was already elevated *entering* the day, so the loop is causal.
    obs_rate = mis / vol  # NaN propagates through missing days
    dev = obs_rate - expected[:, None]
    dev_clipped = np.clip(dev, -cfg.robust_clip_pp, cfg.robust_clip_pp)

    p0 = np.empty((len(lanes), n_days))
    z = np.full((len(lanes), n_days), np.nan)
    cusum = np.zeros((len(lanes), n_days))
    s_up = np.zeros(len(lanes))
    alarm_rows = []
    for d in range(n_days):
        present = ~np.isnan(obs_rate[:, d])
        voters = present & (s_up <= cfg.h / 2)  # suspects don't define "normal"
        w = vol[voters, d]
        g = float((w * dev_clipped[voters, d]).sum() / w.sum()) if voters.any() else 0.0
        p0[:, d] = np.clip(expected + g, 1e-4, 0.95)
        sd = np.sqrt(p0[:, d] * (1 - p0[:, d]) / np.maximum(np.nan_to_num(vol[:, d]), 1.0))
        z[present, d] = (obs_rate[present, d] - p0[present, d]) / sd[present]

        if d < cfg.baseline_days:
            continue  # baseline days fit the prior; monitoring starts after
        s_up[present] = np.maximum(0.0, s_up[present] + z[present, d] - cfg.k)
        cusum[:, d] = s_up
        fired = np.where(s_up >= cfg.h)[0]
        for i in fired:
            alarm_rows.append(
                {
                    "lane": lanes[i],
                    "day": d,
                    "date": dates[d],
                    "cusum": float(s_up[i]),
                    "expected_rate": float(expected[i]),
                }
            )
        s_up[fired] = 0.0  # reset after alarm; a still-broken lane re-alarms on its own

    alarms = pd.DataFrame(alarm_rows, columns=["lane", "day", "date", "cusum", "expected_rate"])

    result = DetectionResult(
        config=cfg,
        lanes=lanes,
        dates=dates,
        prior_alpha=alpha,
        prior_beta=beta,
        expected=pd.Series(expected, index=lanes),
        volume=vol,
        obs_rate=obs_rate,
        p0=p0,
        z=z,
        cusum=cusum,
        alarms=alarms,
    )
    result.monthly_alarms = baseline_monthly(result)
    return result


def baseline_monthly(res: DetectionResult) -> pd.DataFrame:
    """The status-quo detector: month-end report, flag rate > expected + 2 sigma.

    Detection day is the month END — that is the entire point. Even when the
    drift is obvious in the aggregate, nobody sees it until the report runs.
    """
    cfg = res.config
    day = np.arange(len(res.dates))
    months = res.dates.to_period("M")
    rows = []
    for m in months.unique():
        in_month = np.asarray(months == m) & (day >= cfg.baseline_days)
        if not in_month.any() or in_month.sum() < 20:  # skip partial months
            continue
        v = np.nansum(np.where(np.isnan(res.volume[:, in_month]), 0, res.volume[:, in_month]),
                      axis=1)
        mis = np.nansum(
            np.where(np.isnan(res.obs_rate[:, in_month]), 0,
                     res.obs_rate[:, in_month] * res.volume[:, in_month]),
            axis=1,
        )
        rate = mis / np.maximum(v, 1.0)
        dev = rate - res.expected.to_numpy()
        # Same global correction the CUSUM gets: a seasonally-adjusted report.
        w = np.maximum(v, 0.0)
        g = float(np.sum(w * np.clip(dev, -cfg.robust_clip_pp, cfg.robust_clip_pp)) / w.sum())
        p0 = np.clip(res.expected.to_numpy() + g, 1e-4, 0.95)
        sigma = np.sqrt(p0 * (1 - p0) / np.maximum(v, 1.0))
        month_end = int(day[in_month].max())
        for i in np.where((rate - p0 > 2 * sigma) & (v > 0))[0]:
            rows.append(
                {
                    "lane": res.lanes[i],
                    "day": month_end,
                    "date": res.dates[month_end],
                    "month": str(m),
                    "monthly_rate": float(rate[i]),
                    "expected_rate": float(p0[i]),
                }
            )
    return pd.DataFrame(
        rows, columns=["lane", "day", "date", "month", "monthly_rate", "expected_rate"]
    )
