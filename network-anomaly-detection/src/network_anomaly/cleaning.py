"""Audited cleaning for the daily lane feed.

Same CleaningReport pattern as delivery-commit-prediction: every step logs how
many rows it touched, so in production you can alert when a step's touch-count
jumps. For a monitoring pipeline this matters double — a broken upstream feed
looks exactly like a network anomaly, and the cleaning report is how you tell
the two apart before paging anyone.

Design choices specific to this feed:

- Duplicate (lane, day) rows are dropped, keep-first. Both copies carry the
  same day's counts; keeping both would double that day's evidence.
- misses > volume rows are DROPPED, not clipped. Clipping to volume fabricates
  a 100%-miss day, which is precisely the kind of day a drift detector exists
  to react to. Better a gap than a fabricated catastrophe.
- Missing (lane, day) rows are left missing but counted. The detector is built
  to tolerate gaps (the CUSUM simply doesn't update on a missing day).
  Filling gaps with zero misses would fabricate perfect service instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import schema


@dataclass
class CleaningReport:
    steps: list[dict] = field(default_factory=list)

    def add(self, step: str, rows_affected: int, detail: str = "") -> None:
        self.steps.append({"step": step, "rows_affected": int(rows_affected), "detail": detail})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)

    def __str__(self) -> str:
        lines = [
            f"  {s['step']:<34} {s['rows_affected']:>7,} rows  {s['detail']}" for s in self.steps
        ]
        return "CleaningReport:\n" + "\n".join(lines)


def clean(
    df: pd.DataFrame, report: CleaningReport | None = None
) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean a raw daily lane feed. Returns (clean_df, report)."""
    report = report or CleaningReport()
    df = df.copy()
    df[schema.DATE_COL] = pd.to_datetime(df[schema.DATE_COL])

    # --- rows missing essentials ------------------------------------------
    before = len(df)
    df = df.dropna(subset=schema.ALL_COLS)
    report.add("drop_rows_missing_essentials", before - len(df))

    # --- duplicate (lane, day) rows ----------------------------------------
    before = len(df)
    df = df.drop_duplicates(subset=[schema.LANE_COL, schema.DATE_COL], keep="first")
    report.add("drop_duplicate_lane_days", before - len(df), "extract job ran twice")

    # --- impossible counts --------------------------------------------------
    bad = (df[schema.MISSES_COL] > df[schema.VOLUME_COL]) | (df[schema.MISSES_COL] < 0)
    report.add(
        "drop_misses_gt_volume", int(bad.sum()), "dropped, not clipped: see module docstring"
    )
    df = df[~bad]

    bad_vol = df[schema.VOLUME_COL] < 1
    report.add("drop_nonpositive_volume", int(bad_vol.sum()))
    df = df[~bad_vol]

    # --- gap audit (log only, never fill) -----------------------------------
    n_lanes = df[schema.LANE_COL].nunique()
    n_days = df[schema.DATE_COL].nunique()
    missing_lane_days = n_lanes * n_days - len(df)
    lane_counts = df.groupby(schema.LANE_COL, observed=True).size()
    lanes_with_gaps = int((lane_counts < n_days).sum())
    report.add(
        "missing_lane_days_left_as_gaps",
        missing_lane_days,
        f"{lanes_with_gaps} lanes have gaps; detector tolerates them",
    )

    df = df.sort_values([schema.LANE_COL, schema.DATE_COL]).reset_index(drop=True)
    return df, report
