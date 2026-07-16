"""Data cleaning: turn a raw shipment extract into a modeling-ready table.

Every step is logged into a CleaningReport so the transformation is auditable —
in a production pipeline you would emit this report to your data-quality
monitoring, and alert when a step suddenly starts touching far more rows than
usual (schema drift upstream is the #1 silent model killer).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import schema

# Physical bounds for a parcel network; anything outside is treated as a data
# error, not a real shipment.
BOUNDS = {
    "distance_miles": (1.0, 3500.0),
    "package_weight_lb": (0.05, 160.0),
    "package_volume_cuft": (0.001, 80.0),
    "declared_value_usd": (0.0, 50_000.0),
    "minutes_after_cutoff": (-24 * 60.0, 24 * 60.0),
    "origin_hub_congestion": (0.0, 1.0),
    "dest_hub_congestion": (0.0, 1.0),
    "dest_weather_severity": (0.0, 3.0),
    "route_stop_density": (0.05, 30.0),
    "promised_transit_days": (1, 10),
}

SENTINELS = {
    "distance_miles": [9999.0, -1.0],
    "package_weight_lb": [-1.0, 9999.0],
}


@dataclass
class CleaningReport:
    steps: list[dict] = field(default_factory=list)

    def add(self, step: str, rows_affected: int, detail: str = "") -> None:
        self.steps.append({"step": step, "rows_affected": int(rows_affected), "detail": detail})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)

    def __str__(self) -> str:
        lines = [f"  {s['step']:<38} {s['rows_affected']:>8,} rows  {s['detail']}" for s in self.steps]
        return "CleaningReport:\n" + "\n".join(lines)


def clean(df: pd.DataFrame, report: CleaningReport | None = None) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean a raw shipment extract. Returns (clean_df, report)."""
    report = report or CleaningReport()
    df = df.copy()

    # --- duplicates -------------------------------------------------------
    before = len(df)
    df = df.drop_duplicates(subset=[schema.ID_COL], keep="first")
    report.add("drop_duplicate_shipment_ids", before - len(df))

    # --- normalise categoricals ------------------------------------------
    # NB: check string-ness with the pandas API, not `dtype == object` —
    # pandas 3.0 stores strings as a dedicated `str` dtype and the object
    # check silently skips normalisation there.
    for col in ["service_level", "origin_region", "dest_region", "dest_type"]:
        if col in df.columns and pd.api.types.is_string_dtype(df[col]):
            normalised = df[col].str.strip().str.lower()
            changed = (normalised != df[col]).sum()
            df[col] = normalised
            report.add(f"normalise_categorical[{col}]", changed)

    # --- sentinel codes -> NaN -------------------------------------------
    for col, sentinels in SENTINELS.items():
        if col in df.columns:
            mask = df[col].isin(sentinels)
            df.loc[mask, col] = np.nan
            report.add(f"sentinel_to_nan[{col}]", mask.sum(), f"sentinels={sentinels}")

    # --- out-of-bounds -> NaN --------------------------------------------
    for col, (lo, hi) in BOUNDS.items():
        if col in df.columns:
            mask = (df[col] < lo) | (df[col] > hi)
            mask &= df[col].notna()
            df.loc[mask, col] = np.nan
            report.add(f"out_of_bounds_to_nan[{col}]", mask.sum(), f"bounds=({lo}, {hi})")

    # --- impute -----------------------------------------------------------
    # Median imputation with a missingness indicator: the *fact* that an
    # operational field was missing is often itself predictive (a hub too
    # overwhelmed to report congestion is probably congested).
    for col in schema.NUMERIC_FEATURES:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing:
                df[f"{col}__was_missing"] = df[col].isna().astype(int)
                df[col] = df[col].fillna(df[col].median())
                report.add(f"impute_median[{col}]", n_missing, "added __was_missing flag")

    # --- rows we cannot save ----------------------------------------------
    essential = [schema.ID_COL, schema.DATE_COL]
    if schema.LABEL_COL in df.columns:
        essential.append(schema.LABEL_COL)
    before = len(df)
    df = df.dropna(subset=[c for c in essential if c in df.columns])
    report.add("drop_rows_missing_id_date_or_label", before - len(df))

    return df.reset_index(drop=True), report
