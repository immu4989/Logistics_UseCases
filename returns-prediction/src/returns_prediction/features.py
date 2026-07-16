"""Feature engineering and the train/test split.

Two leakage rules, both enforced (not just documented):

1. **Time-based split, never random.** The customer-history features
   (`prior_orders`, `prior_return_rate`) are built causally — each order sees
   only that customer's PAST orders — so the split must respect the same
   arrow of time: the test period sits strictly after the training period.
   Split randomly and a customer's July orders train a model that is then
   "tested" on their March orders, with July history baked into March rows'
   downstream neighbours. A test asserts train.max(date) <= test.min(date).

2. **Order-time features only.** `delivery_days_late` and the label never
   enter the model matrix (`to_matrix` builds from the schema whitelist, and
   a test asserts the post-ship column is absent). Post-delivery signals
   predict returns brilliantly and uselessly: by the time you observe them,
   the pre-ship intervention window is gone.
"""

from __future__ import annotations

import pandas as pd

from . import schema


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features. Input must already be cleaned."""
    df = df.copy()

    # What the customer actually paid: the return-economics features downstream
    # care about price, but the behavioural signal is the paid amount.
    df["paid_price_usd"] = df["unit_price_usd"] * (1 - df["discount_pct"] / 100.0)

    # History depth matters separately from the rate: a 50% return rate over
    # 2 orders and over 20 orders are different customers.
    df["prior_returns"] = (df["prior_orders"] * df["prior_return_rate"]).round(0)

    return df


ENGINEERED_NUMERIC = [
    "paid_price_usd",
    "prior_returns",
]

ONE_HOT_COLS = ["product_category", "channel"]

PASSTHROUGH_FLAGS = [
    "size_limited",
    "first_time_buyer",
    "is_gift",
    "express_shipping",
    "is_bracket_buy",
]


def to_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build the model matrix: numeric features + one-hot categoricals.

    Built from the schema whitelist, so post-ship columns and the label can
    never leak in by accident.
    """
    numeric = [c for c in schema.NUMERIC_FEATURES if c in df.columns]
    numeric += [c for c in ENGINEERED_NUMERIC if c in df.columns]
    numeric += [c for c in df.columns if c.endswith("__was_missing")]
    passthrough = [c for c in PASSTHROUGH_FLAGS if c in df.columns]

    X = df[numeric + passthrough].astype(float)
    dummies = pd.get_dummies(df[ONE_HOT_COLS], prefix=ONE_HOT_COLS, dtype=float)
    return pd.concat([X, dummies], axis=1)


def time_split(
    df: pd.DataFrame, test_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split on order date: train on the past, test on the most recent period."""
    cutoff = df[schema.DATE_COL].quantile(1 - test_frac)
    train = df[df[schema.DATE_COL] <= cutoff]
    test = df[df[schema.DATE_COL] > cutoff]
    return train, test, cutoff


def align_columns(X_train: pd.DataFrame, X_other: pd.DataFrame) -> pd.DataFrame:
    """Reindex another matrix to the training columns (unseen dummies -> 0)."""
    return X_other.reindex(columns=X_train.columns, fill_value=0.0)
