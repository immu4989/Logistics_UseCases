"""Model matrix construction and the train/test split.

Two deliberate choices worth copying into any triage model:

1. **Time-based split, never random.** Tickets from the same weather event or
   hub meltdown are heavily correlated; a random split leaks those episodes
   into training and inflates offline metrics. We train on the first ~80% of
   ticket dates and hold out the rest.

2. **Raw features in, thresholds learned.** The generator's ground truth uses
   scan-gap regimes (8h, 36h), but the model matrix carries the raw hours and
   lets the trees find the cut points. Hand-engineering the true thresholds
   would flatter the linear baseline and hide exactly the nonlinearity that
   earns the GBM its keep.
"""

from __future__ import annotations

import pandas as pd

from . import schema

ONE_HOT_COLS = ["last_scan_location_type", "service_level", "customer_tier"]


def to_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build the model matrix: numeric + flags + noise columns + one-hots."""
    numeric = [c for c in schema.NUMERIC_FEATURES if c in df.columns]
    numeric += [c for c in schema.FLAG_FEATURES if c in df.columns]
    numeric += [c for c in schema.NOISE_FEATURES if c in df.columns]
    numeric += [c for c in df.columns if c.endswith("__was_missing")]

    X = df[numeric].astype(float)
    dummies = pd.get_dummies(df[ONE_HOT_COLS], prefix=ONE_HOT_COLS, dtype=float)
    return pd.concat([X, dummies], axis=1)


def time_split(
    df: pd.DataFrame, test_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split on ticket date: train on the past, test on the most recent period."""
    cutoff = df[schema.DATE_COL].quantile(1 - test_frac)
    train = df[df[schema.DATE_COL] <= cutoff]
    test = df[df[schema.DATE_COL] > cutoff]
    return train, test, cutoff


def align_columns(X_train: pd.DataFrame, X_other: pd.DataFrame) -> pd.DataFrame:
    """Reindex another matrix to the training columns (unseen dummies -> 0)."""
    return X_other.reindex(columns=X_train.columns, fill_value=0.0)
