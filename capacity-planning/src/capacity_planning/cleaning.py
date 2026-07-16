"""Data cleaning: turn a raw lane-demand feed into a modeling-ready table.

Every step is logged into a CleaningReport so the transformation is auditable —
in a production pipeline you would emit this report to your data-quality
monitoring and alert when a step suddenly starts touching far more rows than
usual (an upstream TMS change is the #1 silent forecast killer).

Two decisions specific to demand data:

- **Conflicting duplicates keep the smaller value.** When a (lane, week)
  appears twice, the survivor must be deterministic; keeping the smaller
  demand is the conservative choice for a booking pipeline, and an inflated
  survivor would be caught by monitoring the report, not by luck.
- **Missing weeks stay missing.** Interpolating a gap would invent history
  that the lag features then treat as fact. Gaps are counted and reported
  here, and forecast.py is built to tolerate them (lags that reach into a gap
  are imputed from the lane's own trailing mean, with the imputation flagged).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import synthetic


@dataclass
class CleaningReport:
    steps: list[dict] = field(default_factory=list)

    def add(self, step: str, rows_affected: int, detail: str = "") -> None:
        self.steps.append({"step": step, "rows_affected": int(rows_affected), "detail": detail})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)

    def __str__(self) -> str:
        lines = [
            f"  {s['step']:<38} {s['rows_affected']:>8,} rows  {s['detail']}" for s in self.steps
        ]
        return "CleaningReport:\n" + "\n".join(lines)


def clean(
    df: pd.DataFrame, report: CleaningReport | None = None
) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean a raw lane-week demand feed. Returns (clean_df, report)."""
    report = report or CleaningReport()
    df = df.copy()
    df[synthetic.WEEK_COL] = pd.to_datetime(df[synthetic.WEEK_COL])

    # --- duplicated (lane, week) rows ---------------------------------------
    # Sort by demand first so the survivor of a conflicting pair is
    # deterministic (keep the smaller value; see module docstring).
    before = len(df)
    df = df.sort_values([synthetic.LANE_COL, synthetic.WEEK_COL, synthetic.TARGET_COL])
    df = df.drop_duplicates(subset=[synthetic.LANE_COL, synthetic.WEEK_COL], keep="first")
    report.add("drop_duplicate_lane_weeks", before - len(df))

    # --- impossible negatives -> gap -----------------------------------------
    neg = df[synthetic.TARGET_COL] < 0
    report.add("drop_negative_demand", int(neg.sum()), "reversal records; week becomes a gap")
    df = df[~neg]

    # --- calendar gaps: counted, NOT filled -----------------------------------
    n_weeks = df[synthetic.WEEK_COL].nunique()
    span = (df[synthetic.WEEK_COL].max() - df[synthetic.WEEK_COL].min()).days // 7 + 1
    expected = span * df[synthetic.LANE_COL].nunique()
    report.add(
        "calendar_gaps_left_as_gaps",
        expected - len(df),
        f"{n_weeks} distinct weeks; forecast.py imputes lags through gaps and flags it",
    )

    return df.sort_values([synthetic.LANE_COL, synthetic.WEEK_COL]).reset_index(drop=True), report
