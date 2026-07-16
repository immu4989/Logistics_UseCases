"""Data cleaning: turn a raw volume feed into a modeling-ready table.

Every step is logged into a CleaningReport so the transformation is auditable —
in a production pipeline you would emit this report to your data-quality
monitoring and alert when a step suddenly starts touching far more rows than
usual (an upstream feed change is the #1 silent forecast killer).

Two decisions specific to volume data:

- **Corruption is judged against the same weekday.** Daily volume swings 20x
  between Monday and Sunday by design, so "5x the rolling median" over raw
  days would either miss a 10x-corrupted Sunday or flag every honest Monday.
  The rule compares each value to the rolling median of the *same weekday at
  the same hub*, which moves slowly enough that even the December ramp stays
  under 2x of it while a fat-fingered 10x count stands out.
- **Missing days stay missing.** Interpolating a gap would invent history that
  the lag features then treat as fact. The gaps are counted and reported here,
  and features.py is built to tolerate them (lags simply come back NaN, which
  XGBoost handles natively).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import schema

# A value more than this many times the same-weekday rolling median is a data
# error, not a surge: the biggest honest spike in the network (flash-sale day
# on top of the peak ramp) stays under 2x its weekday neighbours.
CORRUPTION_FACTOR = 5.0
CORRUPTION_WINDOW = 7  # same-weekday observations, i.e. ~7 weeks, centered


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
    """Clean a raw hub-day volume feed. Returns (clean_df, report)."""
    report = report or CleaningReport()
    schema.validate(df)
    df = df.copy()
    df[schema.DATE_COL] = pd.to_datetime(df[schema.DATE_COL])

    # --- duplicated (hub, day) rows ----------------------------------------
    # Sort by volume first so the survivor of a conflicting duplicate pair is
    # deterministic (we keep the smaller count; if that one is itself corrupt,
    # the rules below catch it).
    before = len(df)
    df = df.sort_values([schema.HUB_COL, schema.DATE_COL, schema.TARGET_COL])
    df = df.drop_duplicates(subset=[schema.HUB_COL, schema.DATE_COL], keep="first")
    report.add("drop_duplicate_hub_days", before - len(df))

    # --- impossible negatives -> gap ---------------------------------------
    neg = df[schema.TARGET_COL] < 0
    report.add("drop_negative_volumes", int(neg.sum()), "reversal records; day becomes a gap")
    df = df[~neg]

    # --- corrupted counts -> gap -------------------------------------------
    # Compare to the rolling median of the same weekday at the same hub.
    df = df.sort_values([schema.HUB_COL, schema.DATE_COL])
    dow = df[schema.DATE_COL].dt.dayofweek
    ref = (
        df.groupby([schema.HUB_COL, dow], sort=False)[schema.TARGET_COL]
        .transform(
            lambda s: s.rolling(CORRUPTION_WINDOW, center=True, min_periods=3).median()
        )
    )
    corrupt = df[schema.TARGET_COL] > CORRUPTION_FACTOR * ref
    report.add(
        "drop_corrupted_volumes",
        int(corrupt.sum()),
        f">{CORRUPTION_FACTOR:.0f}x same-weekday rolling median",
    )
    df = df[~corrupt]

    # --- calendar gaps: counted, NOT filled --------------------------------
    span = (df[schema.DATE_COL].max() - df[schema.DATE_COL].min()).days + 1
    expected = span * df[schema.HUB_COL].nunique()
    report.add(
        "calendar_gaps_left_as_gaps",
        expected - len(df),
        "features.py tolerates gaps; interpolation would invent history",
    )

    return df.reset_index(drop=True), report
