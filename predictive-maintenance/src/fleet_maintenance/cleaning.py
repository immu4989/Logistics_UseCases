"""Data cleaning: turn a raw telematics feed into a modeling-ready panel.

Every step is logged into a CleaningReport so the transformation is auditable —
in production you would emit this report to data-quality monitoring and alert
when a step suddenly touches far more rows than usual (a telematics firmware
update that changes units upstream is exactly the kind of silent model killer
this catches).

Two steps deserve a close read:

- **Frozen-sensor detection.** A stuck transmitter repeats its last value
  verbatim for days. Those readings are not "calm and healthy", they are
  absent — so a rolling window of zero variance on a continuous channel gets
  NaN'd and then handled like any other missing data.
- **Imputation keeps `__was_missing` flags.** In telematics, missingness is
  signal: the generator makes dropout probability rise with hidden wear, the
  way a vehicle shaking itself apart also shakes its transmitter loose. The
  flags let the model learn that, and the SHAP tests confirm it does.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import synthetic

# Physical bounds for a delivery vehicle; anything outside is a data error.
BOUNDS = {
    "daily_miles": (0.0, 700.0),
    "engine_hours": (0.0, 24.0),
    "avg_engine_temp": (40.0, 150.0),
    "oil_pressure": (5.0, 80.0),
    "vibration_index": (0.0, 12.0),
    "battery_voltage": (9.0, 15.0),
    "hard_braking_count": (0.0, 200.0),
    "fault_code_count": (0.0, 50.0),
    "vehicle_age_years": (0.0, 30.0),
    "days_since_maint": (0.0, 2000.0),
    "miles_since_maint": (0.0, 100_000.0),
}

# Continuous channels where an exact repeat across days is physically
# implausible (a float read from a real sensor never lands twice).
FROZEN_CHECK_COLS = [
    "avg_engine_temp",
    "oil_pressure",
    "vibration_index",
    "battery_voltage",
]
FROZEN_WINDOW = 5  # this many identical consecutive readings = stuck transmitter

# Channels imputed by per-vehicle interpolation (a sensor trace is a time
# series; the neighborhood is far more informative than a fleet median).
INTERPOLATE_COLS = synthetic.SENSOR_COLS

NUMERIC_COLS = list(BOUNDS)


@dataclass
class CleaningReport:
    steps: list[dict] = field(default_factory=list)

    def add(self, step: str, rows_affected: int, detail: str = "") -> None:
        self.steps.append({"step": step, "rows_affected": int(rows_affected), "detail": detail})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)

    def __str__(self) -> str:
        lines = [
            f"  {s['step']:<40} {s['rows_affected']:>8,} rows  {s['detail']}"
            for s in self.steps
        ]
        return "CleaningReport:\n" + "\n".join(lines)


def _flag_frozen(df: pd.DataFrame, col: str) -> pd.Series:
    """True where `col` sits inside a zero-variance run of >= FROZEN_WINDOW days."""
    grp = df.groupby(synthetic.ID_COL, sort=False)[col]
    # Rolling std over the trailing window; exactly zero means the transmitter
    # repeated its value verbatim. NaNs inside the window make std NaN, which
    # correctly refuses to call a gappy stretch "frozen".
    tail_std = grp.rolling(FROZEN_WINDOW, min_periods=FROZEN_WINDOW).std().reset_index(
        level=0, drop=True
    )
    frozen_end = tail_std.eq(0.0).fillna(False)
    # A zero-variance window ending at day t implicates days t-4..t, so smear
    # the flag back over the whole window (per vehicle, so runs never bleed
    # across vehicle boundaries).
    frozen = frozen_end.copy()
    for k in range(1, FROZEN_WINDOW):
        shifted = frozen_end.groupby(df[synthetic.ID_COL], sort=False).shift(-k)
        frozen |= shifted.fillna(False).astype(bool)
    return frozen


def clean(
    df: pd.DataFrame, report: CleaningReport | None = None
) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean a raw telematics panel. Returns (clean_df, report)."""
    report = report or CleaningReport()
    df = df.copy()

    # --- duplicated (vehicle, day) rows ------------------------------------
    before = len(df)
    df = df.sort_values([synthetic.ID_COL, synthetic.DATE_COL], kind="stable")
    df = df.drop_duplicates(subset=[synthetic.ID_COL, synthetic.DATE_COL], keep="first")
    report.add("drop_duplicate_vehicle_days", before - len(df))
    df = df.reset_index(drop=True)

    # --- impossible values -> NaN -------------------------------------------
    neg = (df["daily_miles"] < 0).fillna(False)
    df.loc[neg, "daily_miles"] = np.nan
    report.add("negative_mileage_to_nan", int(neg.sum()))

    for col, (lo, hi) in BOUNDS.items():
        if col in df.columns:
            mask = ((df[col] < lo) | (df[col] > hi)) & df[col].notna()
            if mask.any():
                df.loc[mask, col] = np.nan
            report.add(f"out_of_bounds_to_nan[{col}]", int(mask.sum()), f"bounds=({lo}, {hi})")

    # --- frozen sensors -> NaN ------------------------------------------------
    for col in FROZEN_CHECK_COLS:
        frozen = _flag_frozen(df, col)
        df.loc[frozen, col] = np.nan
        report.add(
            f"frozen_sensor_to_nan[{col}]",
            int(frozen.sum()),
            f"zero variance over {FROZEN_WINDOW}+ days",
        )

    # --- impute, keeping missingness flags ------------------------------------
    for col in INTERPOLATE_COLS + ["daily_miles"]:
        n_missing = int(df[col].isna().sum())
        df[f"{col}__was_missing"] = df[col].isna().astype(int)
        if n_missing:
            df[col] = (
                df.groupby(synthetic.ID_COL, sort=False)[col]
                .transform(lambda s: s.interpolate(limit_direction="both"))
            )
            med = df[col].median()  # vehicles missing an entire channel
            if pd.notna(med):
                df[col] = df[col].fillna(med)
            report.add(f"impute_interpolate[{col}]", n_missing, "added __was_missing flag")

    for col in NUMERIC_COLS:
        n_missing = int(df[col].isna().sum())
        if n_missing:
            med = df[col].median()
            if pd.notna(med):
                df[col] = df[col].fillna(med)
            report.add(f"impute_median[{col}]", n_missing)

    # --- rows we cannot save ---------------------------------------------------
    essential = [synthetic.ID_COL, synthetic.DATE_COL]
    if synthetic.LABEL_COL in df.columns:
        essential.append(synthetic.LABEL_COL)
    before = len(df)
    df = df.dropna(subset=[c for c in essential if c in df.columns])
    report.add("drop_rows_missing_id_date_or_label", before - len(df))

    return df.reset_index(drop=True), report
