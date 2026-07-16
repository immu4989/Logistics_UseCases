"""Feature engineering and the train/test split.

Two deliberate choices worth copying into any on-time-performance model:

1. **Time-based split, never random.** Shipments from the same lane/day are
   heavily correlated; a random split leaks future operating conditions into
   training and inflates offline metrics that then don't survive deployment.
   We train on the first ~80% of ship dates and test on the last ~20%.

2. **Interpretable encodings.** One-hot for low-cardinality categoricals keeps
   every SHAP value attached to a human-readable operational lever ("ground
   service", "rural destination") instead of an opaque embedding dimension.
"""

from __future__ import annotations

import pandas as pd

from . import schema


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features. Input must already be cleaned."""
    df = df.copy()

    # Pickup timing is far more informative as "late at all + how late" than raw minutes.
    df["late_pickup"] = (df["minutes_after_cutoff"] > 0).astype(int)
    df["late_pickup_minutes"] = df["minutes_after_cutoff"].clip(lower=0)

    # Slack: promised days relative to how far the parcel must travel.
    df["miles_per_promised_day"] = df["distance_miles"] / df["promised_transit_days"].clip(lower=1)

    # Network stress: both endpoints congested is worse than either alone.
    df["total_hub_congestion"] = df["origin_hub_congestion"] + df["dest_hub_congestion"]

    df["is_cross_region"] = (df["origin_region"] != df["dest_region"]).astype(int)
    df["is_weekend_ship"] = df["day_of_week"].astype(int).isin([5, 6]).astype(int)

    return df


ENGINEERED_NUMERIC = [
    "late_pickup",
    "late_pickup_minutes",
    "miles_per_promised_day",
    "total_hub_congestion",
    "is_cross_region",
    "is_weekend_ship",
]

ONE_HOT_COLS = ["service_level", "origin_region", "dest_region", "dest_type"]


def to_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build the model matrix: numeric features + one-hot categoricals."""
    numeric = [c for c in schema.NUMERIC_FEATURES if c in df.columns]
    numeric += [c for c in ENGINEERED_NUMERIC if c in df.columns]
    numeric += [c for c in df.columns if c.endswith("__was_missing")]
    passthrough = [
        c
        for c in ["day_of_week", "is_peak_season", "is_rural_dest", "signature_required"]
        if c in df.columns
    ]

    X = df[numeric + passthrough].astype(float)
    dummies = pd.get_dummies(df[ONE_HOT_COLS], prefix=ONE_HOT_COLS, dtype=float)
    return pd.concat([X, dummies], axis=1)


def time_split(
    df: pd.DataFrame, test_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split on ship date: train on the past, test on the most recent period."""
    cutoff = df[schema.DATE_COL].quantile(1 - test_frac)
    train = df[df[schema.DATE_COL] <= cutoff]
    test = df[df[schema.DATE_COL] > cutoff]
    return train, test, cutoff


def align_columns(X_train: pd.DataFrame, X_other: pd.DataFrame) -> pd.DataFrame:
    """Reindex another matrix to the training columns (unseen dummies -> 0)."""
    return X_other.reindex(columns=X_train.columns, fill_value=0.0)
