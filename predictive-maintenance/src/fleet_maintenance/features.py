"""Rolling-window features per vehicle, and the time-based split.

The leakage rules for a look-ahead maintenance label are strict, and both are
encoded here rather than left to discipline:

1. **Features at day t use days <= t only.** Every rolling statistic is a
   trailing window (pandas' default), so a feature row never peeks at readings
   from the future. The label, by contrast, looks *forward* 14 days — which is
   exactly why rule 2 exists.
2. **The split must be by TIME, not by row or by vehicle.** Any test day must
   come after every training day. A random row split would put day t in test
   while days t-3 and t+2 of the same vehicle sit in train — the model would
   effectively be shown the failure it is asked to predict. Training uses all
   but the last ~3 months; the final ~90 days are held out untouched.

Feature families per sensor channel: trailing 7d / 28d means and 7d / 28d
slopes (an OLS slope over the window — trend is where the warning lives; a
vibration level of 2.1 is ambiguous, a vibration level *climbing 4% a day*
is not), plus the service-clock features (days / miles since maintenance) and
the 7d sensor-dropout rate, because a unit that reports less often is itself
a symptom.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import synthetic

# Channels that get rolling means + slopes. The two noise channels are
# deliberately included: the model is free to use them, and the SHAP tests
# assert it learns not to.
ROLLING_COLS = [
    "avg_engine_temp",
    "oil_pressure",
    "vibration_index",
    "battery_voltage",
    "fault_code_count",
    "hard_braking_count",
    "daily_miles",
    "cabin_temp_setting",
    "radio_volume_avg",
]

PASSTHROUGH_COLS = [
    "vehicle_age_years",
    "days_since_maint",
    "miles_since_maint",
    "engine_hours",
]

WINDOWS = (7, 28)
TEST_DAYS = 90  # held-out final ~3 months

# Columns that exist for evaluation only and must never enter the matrix.
FORBIDDEN = {synthetic.EVENT_COL, synthetic.COMPONENT_COL, synthetic.LABEL_COL}


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add trailing-window features. Input must be cleaned and deduplicated.

    Slopes use the identity slope = Cov(t, x) / Var(t); within a full window
    of consecutive days Var(t) is the constant (w^2 - 1) / 12, so each slope
    costs three rolling means instead of a per-window regression.
    """
    df = df.sort_values([synthetic.ID_COL, synthetic.DATE_COL], kind="stable").copy()
    df["__day_idx"] = df.groupby(synthetic.ID_COL, sort=False).cumcount().astype(float)

    def roll_mean(col: str, w: int) -> pd.Series:
        return df.groupby(synthetic.ID_COL, sort=False)[col].transform(
            lambda s: s.rolling(w, min_periods=w).mean()
        )

    mean_t = {w: roll_mean("__day_idx", w) for w in WINDOWS}
    for col in ROLLING_COLS:
        df["__tx"] = df[col] * df["__day_idx"]
        for w in WINDOWS:
            mean_x = roll_mean(col, w)
            df[f"{col}_mean_{w}d"] = mean_x
            var_t = (w**2 - 1) / 12.0
            df[f"{col}_slope_{w}d"] = (roll_mean("__tx", w) - mean_t[w] * mean_x) / var_t
    df = df.drop(columns="__tx")

    # Dropout rate: fraction of the last 7 days the sensor suite went dark.
    missing_flags = [c for c in df.columns if c.endswith("__was_missing")]
    if missing_flags:
        df["__any_missing"] = df[missing_flags].max(axis=1).astype(float)
        df["sensor_dropout_rate_7d"] = df.groupby(synthetic.ID_COL, sort=False)[
            "__any_missing"
        ].transform(lambda s: s.rolling(7, min_periods=7).mean())
        df = df.drop(columns="__any_missing")

    # Season, so the model can separate "hot because July" from "hot because
    # the engine is dying" and "low voltage because January" from battery wear.
    doy = df[synthetic.DATE_COL].dt.dayofyear.astype(float)
    df["season_sin"] = np.sin(2 * np.pi * doy / 365.0)
    df["season_cos"] = np.cos(2 * np.pi * doy / 365.0)

    return df.drop(columns="__day_idx")


def to_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build the model matrix from an engineered panel (drops warm-up rows upstream)."""
    rolling_derived = [
        c
        for c in df.columns
        if not c.endswith("__was_missing")
        and any(c.startswith(r + "_") for r in ROLLING_COLS)
    ]
    feature_cols = list(
        dict.fromkeys(
            rolling_derived
            + [c for c in PASSTHROUGH_COLS if c in df.columns]
            + [c for c in ROLLING_COLS if c in df.columns]
            + [c for c in df.columns if c.endswith("__was_missing")]
            + [c for c in ["sensor_dropout_rate_7d", "season_sin", "season_cos"] if c in df.columns]
        )
    )
    feature_cols = [c for c in feature_cols if c not in FORBIDDEN]
    return df[feature_cols].astype(float)


def drop_warmup(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose largest rolling window is not yet full (start of each vehicle)."""
    probe = f"{ROLLING_COLS[0]}_mean_{max(WINDOWS)}d"
    return df[df[probe].notna()].reset_index(drop=True)


def time_split(
    df: pd.DataFrame, test_days: int = TEST_DAYS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split on calendar date: train on the past, test on the final `test_days`."""
    cutoff = df[synthetic.DATE_COL].max() - pd.Timedelta(days=test_days)
    train = df[df[synthetic.DATE_COL] <= cutoff]
    test = df[df[synthetic.DATE_COL] > cutoff]
    return train, test, cutoff


def align_columns(X_train: pd.DataFrame, X_other: pd.DataFrame) -> pd.DataFrame:
    """Reindex another matrix to the training columns (absent columns -> 0)."""
    return X_other.reindex(columns=X_train.columns, fill_value=0.0)
