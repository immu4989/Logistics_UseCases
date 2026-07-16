"""Score the detector against the generator's injected ground truth.

Metrics chosen for how a monitoring system is actually judged by an ops team:

- **Detection rate** for steps and ramps, separately. A detector that catches
  cliff-edge failures but sleeps through slow rot (or vice versa) needs to
  say so out loud.
- **Detection delay in days**, CUSUM vs the monthly report. Delay is the
  product: every day earlier is a day of misses not shipped into.
- **False alarms per clean-lane-year**, target < 1. Ops trust is a budget;
  alarm-fatigue is how monitoring systems die. Denominator is the clean-lane
  exposure actually monitored (plus pre-onset stretches of anomaly lanes).
- **Spike-lane behavior, reported separately.** A one-day storm that
  self-recovers should not page anyone; alarms within a few days of a spike
  are acceptable-but-tracked rather than counted as detections or failures.

Only days at or after each anomaly's start are eligible to count as its
detection; anything a detector fires before onset is a false alarm, no credit.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .detect import DetectionResult
from .synthetic import INJECTED_ANOMALIES, clean_lanes

SPIKE_GRACE_DAYS = 7  # alarms on a spike lane within this window: tracked, not punished

BLUE = "#2b6cb0"    # detector / data series
GRAY = "#718096"    # expectation / status-quo baseline
ORANGE = "#dd6b20"  # true anomaly onset (ground truth marker)
RED = "#c53030"     # alarm (status color, never a series)


# ---------------------------------------------------------------------------
# Matching + metrics
# ---------------------------------------------------------------------------

def _first_detection(alarms: pd.DataFrame, lane: str, start_day: int) -> int | None:
    hits = alarms[(alarms["lane"] == lane) & (alarms["day"] >= start_day)]
    return int(hits["day"].min()) if len(hits) else None


def annotate_alarms(res: DetectionResult) -> pd.DataFrame:
    """Label every CUSUM alarm: real drift, spike aftermath, or false alarm."""
    by_lane = {a["lane"]: a for a in INJECTED_ANOMALIES}
    rows = []
    for _, al in res.alarms.iterrows():
        a = by_lane.get(al["lane"])
        if a is None:
            status, kind, since = "false_alarm", "clean_lane", None
        elif a["type"] == "spike":
            in_window = a["start_day"] <= al["day"] <= a["start_day"] + SPIKE_GRACE_DAYS
            status = "spike_window" if in_window else "false_alarm"
            kind, since = "spike", int(al["day"] - a["start_day"])
        elif al["day"] >= a["start_day"]:
            status, kind, since = "detection", a["type"], int(al["day"] - a["start_day"])
        else:
            status, kind, since = "false_alarm", f"pre_onset_{a['type']}", None
        rows.append({**al.to_dict(), "status": status, "anomaly_type": kind,
                     "days_since_onset": since})
    return pd.DataFrame(rows)


def _delay_table(res: DetectionResult) -> pd.DataFrame:
    """Per injected step/ramp anomaly: first CUSUM and monthly detection + delays."""
    rows = []
    for a in INJECTED_ANOMALIES:
        if a["type"] == "spike":
            continue
        cus = _first_detection(res.alarms, a["lane"], a["start_day"])
        mon = _first_detection(res.monthly_alarms, a["lane"], a["start_day"])
        rows.append(
            {
                "lane": a["lane"],
                "type": a["type"],
                "start_day": a["start_day"],
                "magnitude": a["magnitude"],
                "cusum_day": cus,
                "cusum_delay": None if cus is None else cus - a["start_day"],
                "monthly_day": mon,
                "monthly_delay": None if mon is None else mon - a["start_day"],
            }
        )
    return pd.DataFrame(rows)


def _false_alarm_stats(res: DetectionResult, annotated: pd.DataFrame,
                       monthly: bool = False) -> dict:
    """False alarms per clean-lane-year, with honest exposure accounting."""
    cfg = res.config
    monitor_days = len(res.dates) - cfg.baseline_days
    by_lane = {a["lane"]: a for a in INJECTED_ANOMALIES}

    exposure_days = len(clean_lanes()) * monitor_days
    for a in INJECTED_ANOMALIES:
        if a["type"] == "spike":
            exposure_days += monitor_days - (SPIKE_GRACE_DAYS + 1)
        else:
            exposure_days += max(a["start_day"] - cfg.baseline_days, 0)

    if monthly:
        n_false = 0
        for _, al in res.monthly_alarms.iterrows():
            a = by_lane.get(al["lane"])
            if a is None or (a["type"] == "spike") or al["day"] < a["start_day"]:
                # month-end flags on clean lanes, spike lanes, or pre-onset
                n_false += 1
    else:
        n_false = int((annotated["status"] == "false_alarm").sum())

    years = exposure_days / 365.0
    return {"count": n_false, "clean_lane_years": round(years, 2),
            "per_clean_lane_year": round(n_false / years, 3)}


def _rate_block(delays: pd.DataFrame, kind: str) -> dict:
    sub = delays[delays["type"] == kind]
    det = sub["cusum_delay"].dropna()
    mon = sub["monthly_delay"].dropna()
    return {
        "n": int(len(sub)),
        "detected": int(det.size),
        "detection_rate": round(det.size / len(sub), 3),
        "mean_delay_days": None if det.empty else round(float(det.mean()), 1),
        "median_delay_days": None if det.empty else round(float(det.median()), 1),
        "monthly_detected": int(mon.size),
        "monthly_detection_rate": round(mon.size / len(sub), 3),
        "monthly_mean_delay_days": None if mon.empty else round(float(mon.mean()), 1),
    }


def score(res: DetectionResult) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Returns (metrics dict, annotated alarms, per-anomaly delay table)."""
    annotated = annotate_alarms(res)
    delays = _delay_table(res)

    spikes = [a for a in INJECTED_ANOMALIES if a["type"] == "spike"]
    spike_hits = annotated[annotated["status"] == "spike_window"]

    # Surge sanity check: the monitoring-period surge window must not turn
    # into a page-storm across clean lanes.
    from .synthetic import SURGE_WINDOWS
    lo, hi = SURGE_WINDOWS[-1]
    clean = set(clean_lanes())
    surge_alarms = annotated[
        (annotated["lane"].isin(clean))
        & (annotated["day"] >= lo)
        & (annotated["day"] <= hi + 7)
    ]

    metrics = {
        "detector_config": asdict(res.config),
        "beta_prior": {"alpha": round(res.prior_alpha, 2), "beta": round(res.prior_beta, 2)},
        "n_lanes": len(res.lanes),
        "n_clean_lanes": len(clean),
        "steps": _rate_block(delays, "step"),
        "ramps": _rate_block(delays, "ramp"),
        "spikes": {
            "n_lanes": len(spikes),
            "lanes_with_spike_window_alarm": int(spike_hits["lane"].nunique()),
            "note": "alarms within %d days of a one-day spike: tracked, not counted "
                    "as detections or false alarms" % SPIKE_GRACE_DAYS,
        },
        "false_alarms": _false_alarm_stats(res, annotated),
        "monthly_false_alarms": _false_alarm_stats(res, annotated, monthly=True),
        "surge_check": {
            "window_days": [int(lo), int(hi)],
            "clean_lanes_alarming": int(surge_alarms["lane"].nunique()),
            "fraction_of_clean_lanes": round(surge_alarms["lane"].nunique() / len(clean), 4),
        },
    }
    return metrics, annotated, delays


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_lane(res: DetectionResult, lane: str, path: Path, title: str,
              anomaly: dict | None = None) -> None:
    """Two-panel lane figure: observed rate vs expected band, CUSUM below."""
    i = res.lanes.index(lane)
    x = res.dates
    vol = np.nan_to_num(res.volume[i], nan=1.0)
    # Exact binomial band, not a normal approximation: at 10 shipments/day the
    # normal band goes negative and understates how wild clean days can look.
    band_lo = stats.binom.ppf(0.025, vol, res.p0[i]) / vol
    band_hi = stats.binom.ppf(0.975, vol, res.p0[i]) / vol

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 5.6), sharex=True, height_ratios=[2, 1]
    )
    ax1.fill_between(x, band_lo, band_hi, color=GRAY, alpha=0.25, lw=0,
                     label="expected band (95%, binomial)")
    ax1.plot(x, res.obs_rate[i], color=BLUE, lw=0.8, label="observed daily miss rate")
    ax2.plot(x, np.where(np.arange(len(x)) >= res.config.baseline_days, res.cusum[i], np.nan),
             color=BLUE, lw=1.2)
    ax2.axhline(res.config.h, color=GRAY, ls="--", lw=1)
    ax2.text(x[2], res.config.h * 1.04, f"alarm threshold h = {res.config.h:g}",
             color=GRAY, fontsize=8, va="bottom")

    for ax in (ax1, ax2):
        ax.axvspan(x[0], x[res.config.baseline_days - 1], color=GRAY, alpha=0.10, lw=0)
        if anomaly is not None:
            ax.axvline(x[anomaly["start_day"]], color=ORANGE, ls="--", lw=1.4)
    if anomaly is not None:
        ax1.text(x[anomaly["start_day"]], ax1.get_ylim()[1] * 0.97,
                 f"  true {anomaly['type']} onset", color=ORANGE, fontsize=8, va="top")
    ax1.text(x[10], ax1.get_ylim()[1] * 0.97, "baseline\n(prior fit)", color=GRAY,
             fontsize=8, va="top")

    lane_alarms = res.alarms[res.alarms["lane"] == lane]
    for _, al in lane_alarms.iterrows():
        ax2.plot(al["date"], al["cusum"], "o", color=RED, ms=7, zorder=5)
    if len(lane_alarms):
        first = lane_alarms.iloc[0]
        ax2.annotate(f"first alarm {first['date']:%b %d}", (first["date"], first["cusum"]),
                     xytext=(-90, 14), textcoords="offset points", color=RED, fontsize=8,
                     bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1),
                     arrowprops=dict(arrowstyle="-", color=RED, lw=0.8))

    ax1.set_ylabel("daily miss rate")
    ax1.set_ylim(bottom=0)
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax2.set_ylabel("CUSUM")
    ax2.set_ylim(bottom=0)
    ax1.set_title(title, fontsize=10)
    for ax in (ax1, ax2):
        ax.grid(alpha=0.2, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_delay_comparison(delays: pd.DataFrame, path: Path) -> None:
    """Paired dots per anomaly: CUSUM detection day vs monthly-report day."""
    d = delays.dropna(subset=["cusum_delay"]).sort_values("cusum_delay").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(d) + 1.6))
    xmax = float(max(d["monthly_delay"].dropna().max() if d["monthly_delay"].notna().any()
                     else 0, d["cusum_delay"].max())) + 8
    for y, row in d.iterrows():
        label = f"{row['lane']} ({row['type']})"
        if pd.notna(row["monthly_delay"]):
            ax.plot([row["cusum_delay"], row["monthly_delay"]], [y, y],
                    color=GRAY, lw=1, zorder=1)
            ax.plot(row["monthly_delay"], y, "o", color=GRAY, ms=8, zorder=2)
        else:
            ax.text(xmax - 1, y, "monthly report: never flagged", color=GRAY,
                    fontsize=8, va="center", ha="right")
        ax.plot(row["cusum_delay"], y, "o", color=BLUE, ms=8, zorder=3)
        ax.text(-2, y, label, ha="right", va="center", fontsize=8)
    ax.plot([], [], "o", color=BLUE, label="CUSUM")
    ax.plot([], [], "o", color=GRAY, label="monthly report")
    ax.set_yticks([])
    ax.set_xlim(-30, xmax)
    ax.set_xlabel("days from drift onset to detection")
    ax.set_title("Detection delay per injected anomaly", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.2, lw=0.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_heatmap(res: DetectionResult, path: Path) -> None:
    """Lanes-by-time CUSUM heatmap: the network health dashboard shot."""
    by_lane = {a["lane"]: a for a in INJECTED_ANOMALIES}
    order = (
        [a["lane"] for a in INJECTED_ANOMALIES if a["type"] == "step"]
        + [a["lane"] for a in INJECTED_ANOMALIES if a["type"] == "ramp"]
        + [a["lane"] for a in INJECTED_ANOMALIES if a["type"] == "spike"]
        + [ln for ln in res.lanes if ln not in by_lane]
    )
    idx = [res.lanes.index(ln) for ln in order]
    mat = np.clip(res.cusum[idx], 0, res.config.h * 1.5)

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(mat, aspect="auto", cmap="Blues", interpolation="nearest",
                   vmin=0, vmax=res.config.h * 1.5)
    for row, ln in enumerate(order):
        a = by_lane.get(ln)
        if a is not None:
            ax.plot(a["start_day"], row, ">", color=ORANGE, ms=5, zorder=5)
    n_step = sum(a["type"] == "step" for a in INJECTED_ANOMALIES)
    n_ramp = sum(a["type"] == "ramp" for a in INJECTED_ANOMALIES)
    n_anom = len(by_lane)
    for boundary in (n_step - 0.5, n_step + n_ramp - 0.5, n_anom - 0.5):
        ax.axhline(boundary, color=GRAY, lw=0.8)
    ax.axvline(res.config.baseline_days - 0.5, color=GRAY, lw=0.8, ls="--")
    ax.text(res.config.baseline_days + 3, len(order) - 3, "monitoring starts",
            color=GRAY, fontsize=8)
    groups = [
        (f"{n_step} step", n_step / 2 - 0.5),
        (f"{n_ramp} ramp", n_step + n_ramp / 2 - 0.5),
        (f"{n_anom - n_step - n_ramp} spike", (n_step + n_ramp + n_anom) / 2 - 0.5),
        (f"{len(order) - n_anom} clean lanes", (n_anom + len(order)) / 2 - 0.5),
    ]
    ax.set_yticks([y for _, y in groups])
    ax.set_yticklabels([g for g, _ in groups], fontsize=8)
    ax.set_xlabel("day of year")
    ax.set_title("CUSUM statistic per lane (darker = closer to alarm); "
                 "orange = true anomaly onset", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.6, label="CUSUM (clipped at 1.5h)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def evaluate(res: DetectionResult, out_dir: str | Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Score, plot, and write reports. Returns (metrics, annotated alarms, delays)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    metrics, annotated, delays = score(res)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    annotated.to_csv(out / "alarms.csv", index=False)
    res.monthly_alarms.to_csv(out / "monthly_alarms.csv", index=False)
    delays.to_csv(out / "detection_delays.csv", index=False)

    # Example lanes: a mid-pack detected step, a detected ramp, a busy clean lane.
    det_steps = delays[(delays["type"] == "step") & delays["cusum_delay"].notna()]
    det_ramps = delays[(delays["type"] == "ramp") & delays["cusum_delay"].notna()]
    by_lane = {a["lane"]: a for a in INJECTED_ANOMALIES}
    if len(det_steps):
        lane = det_steps.sort_values("cusum_delay").iloc[len(det_steps) // 2]["lane"]
        plot_lane(res, lane, out / "example_step_lane.png",
                  f"Step-drift lane {lane}: +{by_lane[lane]['magnitude']:.0%} points overnight",
                  by_lane[lane])
    if len(det_ramps):
        lane = det_ramps.sort_values("cusum_delay").iloc[len(det_ramps) // 2]["lane"]
        plot_lane(res, lane, out / "example_ramp_lane.png",
                  f"Ramp lane {lane}: +{by_lane[lane]['magnitude'] * 100:.1f}pp per week, "
                  "no single bad day", by_lane[lane])
    fired = set(res.alarms["lane"])
    quiet_clean = [ln for ln in clean_lanes() if ln not in fired]
    busiest = max(quiet_clean, key=lambda ln: np.nansum(res.volume[res.lanes.index(ln)]))
    plot_lane(res, busiest, out / "example_clean_lane.png",
              f"Clean lane {busiest}: stays quiet through both network-wide surges")

    plot_delay_comparison(delays, out / "detection_delay_comparison.png")
    plot_heatmap(res, out / "cusum_heatmap.png")
    return metrics, annotated, delays
