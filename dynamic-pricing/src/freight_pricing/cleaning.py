"""Data cleaning: turn a raw quote extract into a modeling-ready table.

Every step is logged into a CleaningReport so the transformation is auditable
— in production you would ship this report to data-quality monitoring and
alert when a step suddenly touches far more rows than usual (upstream schema
drift is the number-one silent model killer, and a pricing model trained on
a drifted extract quotes real money wrong).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ID_COL = "quote_id"
DATE_COL = "quote_date"
LABEL_COL = "accepted"
PRICE_COL = "quoted_price_usd"

# Physical / economic bounds for a freight quote; anything outside is a data
# error, not a real shipment.
BOUNDS = {
    "distance_miles": (10.0, 3500.0),
    "weight_lb": (1.0, 60_000.0),
    "volume_cuft": (0.1, 4000.0),
    "market_rate_index": (0.5, 1.6),
    "fuel_index": (0.5, 1.6),
    "competitor_pressure": (0.0, 1.0),
    "our_cost_usd": (50.0, 50_000.0),
    "reference_price_usd": (50.0, 80_000.0),
}

# Columns whose NaNs are imputed with a median + __was_missing flag.
IMPUTE_COLS = [
    "distance_miles",
    "weight_lb",
    "volume_cuft",
    "market_rate_index",
    "fuel_index",
    "competitor_pressure",
]

MIN_PLAUSIBLE_PRICE = 10.0  # below this a "quote" is an abandoned form, not an offer


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


def clean(df: pd.DataFrame, report: CleaningReport | None = None) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean a raw quote extract. Returns (clean_df, report)."""
    report = report or CleaningReport()
    df = df.copy()

    # --- duplicates ---------------------------------------------------------
    before = len(df)
    df = df.drop_duplicates(subset=[ID_COL], keep="first")
    report.add("drop_duplicate_quote_ids", before - len(df))

    # --- normalise categoricals ---------------------------------------------
    # NB: check string-ness with the pandas API, not `dtype == object` —
    # pandas 3.x stores strings as a dedicated `str` dtype and the object
    # check silently skips normalisation there.
    for col in ["customer_segment", "urgency"]:
        if col in df.columns and pd.api.types.is_string_dtype(df[col]):
            normalised = df[col].str.strip().str.lower()
            changed = (normalised != df[col]).sum()
            df[col] = normalised
            report.add(f"normalise_categorical[{col}]", changed)

    # --- out-of-bounds -> NaN -------------------------------------------------
    for col, (lo, hi) in BOUNDS.items():
        if col in df.columns:
            mask = ((df[col] < lo) | (df[col] > hi)) & df[col].notna()
            df.loc[mask, col] = np.nan
            report.add(f"out_of_bounds_to_nan[{col}]", mask.sum(), f"bounds=({lo}, {hi})")

    # --- impute ---------------------------------------------------------------
    # Median imputation with a missingness indicator: the *fact* a field was
    # missing is often itself informative (a lane with no benchmark rate is
    # probably a thin, odd lane).
    for col in IMPUTE_COLS:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing:
                df[f"{col}__was_missing"] = df[col].isna().astype(int)
                df[col] = df[col].fillna(df[col].median())
                report.add(f"impute_median[{col}]", n_missing, "added __was_missing flag")

    # --- impossible prices: drop, never impute ---------------------------------
    # The historical price is the treatment variable the model learns
    # elasticity from. A zero price is unrecoverable: imputing a median price
    # would fabricate a (price, accepted) pair that never happened and inject
    # fake demand signal exactly where the model is most sensitive.
    before = len(df)
    bad_price = df[PRICE_COL].isna() | (df[PRICE_COL] < MIN_PLAUSIBLE_PRICE)
    df = df[~bad_price]
    report.add("drop_rows_impossible_price", before - len(df), f"price < {MIN_PLAUSIBLE_PRICE}")

    # --- rows we cannot save ----------------------------------------------------
    essential = [c for c in [ID_COL, DATE_COL, LABEL_COL, "our_cost_usd", "reference_price_usd"]
                 if c in df.columns]
    before = len(df)
    df = df.dropna(subset=essential)
    report.add("drop_rows_missing_essentials", before - len(df))

    return df.reset_index(drop=True), report
