"""Audited cleaning for a raw dispatch extract.

A routing algorithm is brutally literal about bad data: one stop geocoded to
(0, 0) drags a truck to "null island", a duplicated stop gets visited twice,
and a negative package count quietly inflates remaining capacity. Every fix
below is logged into a CleaningReport — in production you would ship this
report to data-quality monitoring and alert when a step suddenly touches far
more rows than usual, because upstream schema drift is the number-one silent
route killer.
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
        lines = [f"  {s['step']:<34} {s['rows_affected']:>6,} rows  {s['detail']}" for s in self.steps]
        return "CleaningReport:\n" + "\n".join(lines)


def clean(df: pd.DataFrame, report: CleaningReport | None = None) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean a raw stop extract. Returns (clean_df, report)."""
    report = report or CleaningReport()
    df = df.copy()

    # --- duplicate stop rows ------------------------------------------------
    # The WMS exports one row per package line; keep the first occurrence.
    before = len(df)
    df = df.drop_duplicates(subset=["stop_id"], keep="first")
    report.add("drop_duplicate_stop_ids", before - len(df))

    # --- geocode failures ----------------------------------------------------
    # (0, 0) is the geocoder's null result, and nothing outside the metro
    # bound is a real stop for this depot. These rows are DROPPED, not
    # imputed: routing a truck to an invented coordinate is worse than
    # sending the stop back to the geocoding queue. In production this
    # bucket goes to a re-geocode/manual-fix worklist, not the trash.
    null_island = (df["x_mi"] == 0.0) & (df["y_mi"] == 0.0)
    df = df[~null_island]
    report.add("drop_geocode_null_island", int(null_island.sum()), "coords exactly (0, 0)")

    bound = synthetic.METRO_HALF_MI
    outside = (df["x_mi"].abs() > bound) | (df["y_mi"].abs() > bound)
    df = df[~outside]
    report.add("drop_geocode_out_of_metro", int(outside.sum()), f"|x| or |y| > {bound} mi")

    # --- impossible package counts -------------------------------------------
    # Negative counts are a returns line-item joined the wrong way; the stop
    # itself is real (a driver still goes there), so clip rather than drop.
    lo, hi = synthetic.MIN_PACKAGES, synthetic.MAX_PACKAGES
    bad = (df["packages"] < lo) | (df["packages"] > hi)
    df.loc[bad, "packages"] = df.loc[bad, "packages"].clip(lo, hi)
    report.add("clip_impossible_package_counts", int(bad.sum()), f"clipped into [{lo}, {hi}]")

    return df.reset_index(drop=True), report
