"""Per-lane booking rationale: the audit trail a linehaul planner signs off on.

No SHAP here, on purpose. The question a planner asks is not "which feature
moved the quantile" — it is "why 23 trailers and not 25", and for a newsvendor
policy the honest explanation is the economics itself: the quantile forecast
says where demand is likely to land, the critical fractile says where on that
distribution the cost structure tells you to sit, and the expected cost
decomposition prices the choice. Every number below is checkable by hand.

Three lanes are written up, chosen to cover the decision modes an audit asks
about: the biggest stable corridor (where the fractile, not the forecast, does
the work), the fastest-growing lane in a peak week (where the habit is a year
and a ramp behind), and a declining lane (where the habit quietly pays to move
air every single week).

The expected-cost columns use the generator's true demand distribution — a
grounding that is only possible because the data is synthetic. On real data,
replace them with backtest averages; the structure of the card stays the same.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import decide, synthetic


def _pick_lanes() -> dict[str, str]:
    """Choose the three archetype lanes from the generator's exposed tables."""
    flat = {ln: b for ln, b in synthetic.LANE_BASE.items() if ln not in synthetic.TREND_PER_YEAR}
    return {
        "big stable corridor": max(flat, key=flat.get),
        "growing lane at peak": max(synthetic.TREND_PER_YEAR, key=synthetic.TREND_PER_YEAR.get),
        "declining lane": min(synthetic.TREND_PER_YEAR, key=synthetic.TREND_PER_YEAR.get),
    }


def _pick_week(bookings: pd.DataFrame, lane: str, want_peak: bool) -> pd.Series:
    """A representative test row for the lane: a peak week or an ordinary one."""
    df = bookings[bookings[synthetic.LANE_COL] == lane].sort_values(synthetic.WEEK_COL)
    woy = pd.DatetimeIndex(df[synthetic.WEEK_COL]).isocalendar().week.to_numpy()
    in_peak = np.isin(woy, synthetic.PEAK_WOY)
    pick = df[in_peak] if want_peak and in_peak.any() else df[~in_peak]
    return pick.iloc[len(pick) // 2]


def _expected_components(lane: str, week: pd.Timestamp, booked: int) -> dict[str, float]:
    """Expected weekly cost decomposition at a booking level, from ground truth."""
    level = synthetic.demand_level(pd.DatetimeIndex([week]))
    lv = float(level.loc[level[synthetic.LANE_COL] == lane, "level"].iloc[0])
    sample = lv * synthetic._multiplier_sample(synthetic.lane_sigma(lane))
    short = np.maximum(sample - booked, 0.0).mean()
    empty = np.maximum(booked - sample, 0.0).mean()
    return {
        "expected_spot_teq": float(short),
        "expected_empty_teq": float(empty),
        "expected_cost_usd": float(
            decide.COMMITTED_COST_USD * booked
            + decide.SPOT_COST_USD * short
            - decide.SALVAGE_USD * empty
        ),
    }


def _why(role: str, row: pd.Series, penalty: float) -> str:
    if role == "big stable corridor":
        return (
            f"booking P{decide.critical_fractile() * 100:.0f} not P50: an empty trailer costs "
            f"$1,050 and a spot cover $900 this quarter, so the right seat is just below the "
            f"median — one trailer of restraint on a lane this size."
        )
    if role == "growing lane at peak":
        return (
            f"last year's number ({row['booked_last_year']} trailers) is a year of growth and a "
            f"peak ramp out of date; the quantile forecast carries both, and the habit hands "
            f"~${penalty:,.0f}/week to the spot market here."
        )
    return (
        f"the habit re-books a lane that shrank; ~${penalty:,.0f}/week of that booking now "
        f"moves air, and the model steps the commitment down instead."
    )


def write_rationale(bookings: pd.DataFrame, out_dir: str | Path) -> pd.DataFrame:
    """Write rationale.md with three per-lane booking cards; return the table."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    q_star = decide.critical_fractile()

    rows = []
    for role, lane in _pick_lanes().items():
        row = _pick_week(bookings, lane, want_peak=(role == "growing lane at peak"))
        booked = int(row["booked_newsvendor"])
        habit = int(row["booked_last_year"])
        ours = _expected_components(lane, row[synthetic.WEEK_COL], booked)
        theirs = _expected_components(lane, row[synthetic.WEEK_COL], habit)
        penalty = theirs["expected_cost_usd"] - ours["expected_cost_usd"]
        rows.append(
            {
                "case": role,
                "lane": lane,
                "week": str(pd.Timestamp(row[synthetic.WEEK_COL]).date()),
                "forecast_q_base": round(float(row["q_base"]), 1),
                "forecast_p50": round(float(row["p50"]), 1),
                "booked_newsvendor": booked,
                "booked_last_year": habit,
                "expected_spot_teq": round(ours["expected_spot_teq"], 2),
                "expected_empty_teq": round(ours["expected_empty_teq"], 2),
                "expected_cost_usd": round(ours["expected_cost_usd"], 0),
                "habit_expected_cost_usd": round(theirs["expected_cost_usd"], 0),
                "habit_penalty_usd_per_week": round(penalty, 0),
                "why": _why(role, row, penalty),
            }
        )
    table = pd.DataFrame(rows)

    lines = [
        "# Why the desk booked these numbers",
        "",
        "The booking rule is one line of arithmetic on top of the quantile forecast:",
        "",
        f"    Cu = spot - committed = ${decide.SPOT_COST_USD:,.0f} - "
        f"${decide.COMMITTED_COST_USD:,.0f} = ${decide.SPOT_COST_USD - decide.COMMITTED_COST_USD:,.0f}",
        f"    Co = committed - salvage = ${decide.COMMITTED_COST_USD:,.0f} - "
        f"${decide.SALVAGE_USD:,.0f} = ${decide.COMMITTED_COST_USD - decide.SALVAGE_USD:,.0f}",
        f"    q* = Cu / (Cu + Co) = {q_star:.3f}   ->   book = ceil(demand quantile at q*)",
        "",
        "| Case | Lane | Week | Q(q*) fcst | P50 fcst | Booked | Habit booked | "
        "E[spot teq] | E[empty teq] | E[cost] | Habit E[cost] | Habit penalty/wk | Why |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['case']} | {r['lane']} | {r['week']} | {r['forecast_q_base']} "
            f"| {r['forecast_p50']} | {r['booked_newsvendor']} | {r['booked_last_year']} "
            f"| {r['expected_spot_teq']} | {r['expected_empty_teq']} "
            f"| ${r['expected_cost_usd']:,.0f} | ${r['habit_expected_cost_usd']:,.0f} "
            f"| ${r['habit_penalty_usd_per_week']:,.0f} | {r['why']} |"
        )
    lines += [
        "",
        "_Expected columns are computed from the generator's true demand distribution (the "
        "grounding only synthetic data allows); on real data, substitute backtest averages._",
    ]
    (out_dir / "rationale.md").write_text("\n".join(lines))
    return table
