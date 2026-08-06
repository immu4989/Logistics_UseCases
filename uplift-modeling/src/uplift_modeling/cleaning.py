"""Data cleaning: turn a raw pilot extract into an analysis-ready table.

Every step is logged into a CleaningReport so the transformation is auditable.
In production you would emit this report to data-quality monitoring and alert
when a step suddenly touches far more rows than usual — upstream schema drift
is the number-one silent killer of models, and it kills uplift models twice
over because it can also silently break the randomization bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import synthetic

BOUNDS = {
    "distance_miles": (1.0, 3500.0),
    "origin_congestion": (0.0, 1.0),
    "dest_congestion": (0.0, 1.0),
    "dest_weather_severity": (0.0, 3.0),
    "declared_value_usd": (0.0, 50_000.0),
}

NUMERIC_FEATURES = [
    "distance_miles",
    "origin_congestion",
    "dest_congestion",
    "dest_weather_severity",
    "declared_value_usd",
]

CATEGORICAL_COLS = ["service_level", "customer_tier"]


@dataclass
class CleaningReport:
    steps: list[dict] = field(default_factory=list)

    def add(self, step: str, rows_affected: int, detail: str = "") -> None:
        self.steps.append({"step": step, "rows_affected": int(rows_affected), "detail": detail})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)

    def __str__(self) -> str:
        lines = [
            f"  {s['step']:<38} {s['rows_affected']:>8,} rows  {s['detail']}"
            for s in self.steps
        ]
        return "CleaningReport:\n" + "\n".join(lines)


def clean(
    df: pd.DataFrame, report: CleaningReport | None = None
) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean a raw pilot extract. Returns (clean_df, report)."""
    report = report or CleaningReport()
    df = df.copy()

    # --- duplicates ---------------------------------------------------------
    before = len(df)
    df = df.drop_duplicates(subset=[synthetic.ID_COL], keep="first")
    report.add("drop_duplicate_shipment_ids", before - len(df))

    # --- normalise categoricals ----------------------------------------------
    # NB: check string-ness with the pandas API, not `dtype == object` —
    # pandas 3.x stores strings as a dedicated `str` dtype and the object
    # check silently skips normalisation there.
    for col in CATEGORICAL_COLS:
        if col in df.columns and pd.api.types.is_string_dtype(df[col]):
            normalised = df[col].str.strip().str.lower()
            changed = (normalised != df[col]).sum()
            df[col] = normalised
            report.add(f"normalise_categorical[{col}]", changed)

    # --- out-of-bounds -> NaN (covers the -1 geocoder distances) -------------
    for col, (lo, hi) in BOUNDS.items():
        if col in df.columns:
            mask = (df[col] < lo) | (df[col] > hi)
            mask &= df[col].notna()
            df.loc[mask, col] = np.nan
            report.add(f"out_of_bounds_to_nan[{col}]", mask.sum(), f"bounds=({lo}, {hi})")

    # --- impute ---------------------------------------------------------------
    # Median imputation with a missingness indicator: the fact that a field was
    # missing is often itself predictive.
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing:
                df[f"{col}__was_missing"] = df[col].isna().astype(int)
                df[col] = df[col].fillna(df[col].median())
                report.add(f"impute_median[{col}]", n_missing, "added __was_missing flag")

    # --- rows we cannot save ---------------------------------------------------
    # Uplift needs the outcome AND the treatment flag; a row missing either is
    # unusable, and imputing a treatment flag would corrupt the randomization.
    essential = [
        c
        for c in [synthetic.ID_COL, synthetic.LABEL_COL, synthetic.TREATMENT_COL]
        if c in df.columns
    ]
    before = len(df)
    df = df.dropna(subset=essential)
    report.add("drop_rows_missing_id_label_or_treatment", before - len(df))

    return df.reset_index(drop=True), report
