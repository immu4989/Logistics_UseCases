"""Canonical daily lane-level OTP table.

One row per (lane, date): how many shipments moved on that lane that day, and
how many of them missed their delivery commitment. This is the coarsest table
that still supports early drift detection — anything aggregated further
(weekly, monthly) throws away exactly the resolution a fast detector needs,
and anything finer (per-shipment) belongs to the delivery-commit-prediction
use case next door.
"""

from __future__ import annotations

LANE_COL = "lane"        # "ORIGIN-DEST" hub pair, e.g. "MEM-ORD"
DATE_COL = "date"        # calendar date of the delivery commitments scored
VOLUME_COL = "volume"    # shipments with a commitment scored that day
MISSES_COL = "misses"    # of those, how many missed the commitment

ALL_COLS = [LANE_COL, DATE_COL, VOLUME_COL, MISSES_COL]


def validate(df) -> None:
    """Raise ValueError if the DataFrame is missing schema columns."""
    missing = set(ALL_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {sorted(missing)}")
