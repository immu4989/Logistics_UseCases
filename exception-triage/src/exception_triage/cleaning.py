"""Data cleaning: turn a raw ticket extract into a modeling-ready table.

Every step is logged into a CleaningReport so the transformation is auditable —
in production you would emit this report to data-quality monitoring and alert
when a step suddenly starts touching far more rows than usual (schema drift
upstream is the #1 silent model killer).

The step with the most leverage here is label normalization. The historical
`resolution_queue` labels come from three generations of CRM: one title-cased
the queue names with spaces, one SHOUTED them, one padded whitespace. Train on
those raw strings and the model happily learns "Address Correction" and
"address_correction" as different classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import schema

# Physical bounds for an exception ticket; anything outside is a data error,
# not a real ticket.
BOUNDS = {
    "scan_gap_hours": (0.0, 400.0),
    "prior_exceptions": (0, 20),
    "declared_value_usd": (0.0, 50_000.0),
    "delivery_attempt_count": (0, 6),
}

# Flag columns where a missing value means "the upstream feed didn't answer",
# not "unknown category". Imputed to 0 with a missingness indicator.
FLAG_COLS_IMPUTE_ZERO = [
    "weather_event_at_location",
    "address_validation_failed",
    "damage_scan_flag",
    "return_to_sender_flag",
    "is_international",
]


@dataclass
class CleaningReport:
    steps: list[dict] = field(default_factory=list)

    def add(self, step: str, rows_affected: int, detail: str = "") -> None:
        self.steps.append({"step": step, "rows_affected": int(rows_affected), "detail": detail})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)

    def __str__(self) -> str:
        lines = [f"  {s['step']:<40} {s['rows_affected']:>8,} rows  {s['detail']}" for s in self.steps]
        return "CleaningReport:\n" + "\n".join(lines)


def normalize_queue_label(labels: pd.Series) -> pd.Series:
    """Canonicalize queue names: strip, lowercase, spaces -> underscores."""
    return labels.str.strip().str.lower().str.replace(" ", "_", regex=False)


def clean(df: pd.DataFrame, report: CleaningReport | None = None) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean a raw exception-ticket extract. Returns (clean_df, report)."""
    report = report or CleaningReport()
    df = df.copy()

    # --- duplicates ---------------------------------------------------------
    # The CRM re-emits the whole ticket row on every status touch.
    before = len(df)
    df = df.drop_duplicates(subset=[schema.ID_COL], keep="first")
    report.add("drop_duplicate_ticket_ids", before - len(df))

    # --- normalise the label ------------------------------------------------
    # NB: check string-ness with the pandas API, not `dtype == object` —
    # pandas 3.0 stores strings as a dedicated `str` dtype and the object
    # check silently skips normalisation there.
    if schema.LABEL_COL in df.columns and pd.api.types.is_string_dtype(df[schema.LABEL_COL]):
        normalised = normalize_queue_label(df[schema.LABEL_COL])
        changed = (normalised != df[schema.LABEL_COL]).sum()
        df[schema.LABEL_COL] = normalised
        report.add("normalise_queue_label_casing", changed)
        unknown = ~df[schema.LABEL_COL].isin(schema.QUEUES)
        df = df[~unknown]
        report.add("drop_unknown_queue_labels", unknown.sum())

    # --- normalise categorical features --------------------------------------
    for col in schema.CATEGORICAL_FEATURES:
        if col in df.columns and pd.api.types.is_string_dtype(df[col]):
            normalised = df[col].str.strip().str.lower()
            changed = (normalised != df[col]).sum()
            df[col] = normalised
            report.add(f"normalise_categorical[{col}]", changed)

    # --- impossible values -> NaN --------------------------------------------
    # Negative scan gaps are clock skew between the scan feed and the CRM.
    mask = (df["scan_gap_hours"] < 0) & df["scan_gap_hours"].notna()
    df.loc[mask, "scan_gap_hours"] = np.nan
    report.add("negative_scan_gap_to_nan", mask.sum())

    for col, (lo, hi) in BOUNDS.items():
        if col in df.columns:
            mask = ((df[col] < lo) | (df[col] > hi)) & df[col].notna()
            df.loc[mask, col] = np.nan
            report.add(f"out_of_bounds_to_nan[{col}]", mask.sum(), f"bounds=({lo}, {hi})")

    # --- impute ---------------------------------------------------------------
    # Numerics: median with a missingness indicator (a scan gap the feed could
    # not compute is itself a signal that the shipment went dark).
    for col in schema.NUMERIC_FEATURES:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing:
                df[f"{col}__was_missing"] = df[col].isna().astype(int)
                df[col] = df[col].fillna(df[col].median())
                report.add(f"impute_median[{col}]", n_missing, "added __was_missing flag")

    # Flags: a NULL from the weather feed or address validator means the check
    # never ran; impute 0 (no event observed) and keep the indicator.
    for col in FLAG_COLS_IMPUTE_ZERO:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing:
                df[f"{col}__was_missing"] = df[col].isna().astype(int)
                df[col] = df[col].fillna(0)
                report.add(f"impute_zero_flag[{col}]", n_missing, "added __was_missing flag")
            df[col] = df[col].astype(int)

    # --- rows we cannot save ----------------------------------------------------
    essential = [schema.ID_COL, schema.DATE_COL]
    if schema.LABEL_COL in df.columns:
        essential.append(schema.LABEL_COL)
    before = len(df)
    df = df.dropna(subset=[c for c in essential if c in df.columns])
    report.add("drop_rows_missing_id_date_or_label", before - len(df))

    return df.reset_index(drop=True), report
