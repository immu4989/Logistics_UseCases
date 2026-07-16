"""Feature engineering and the train/test split.

The one rule that keeps a volume forecast honest: **lags only, never same-day
information**. A staffing forecast is consumed before the day starts, so the
model may see the calendar (known years ahead), the marketing promo calendar
(published weeks ahead), and volume history up to *yesterday* — and nothing
else. Same-day scan counts or door telemetry would predict today's volume
brilliantly and uselessly; that is how volume models leak.

Gaps are first-class citizens. The cleaned feed has missing days (feed
outages, dropped corrupt values), so lags are computed on a per-hub daily
calendar where a missing day is NaN. A lag that reaches back into a gap comes
back NaN and XGBoost routes it through its default branch — no interpolation,
no invented history.

The split is time-based and the test window is pinned to the final ~4 months
so it *must* contain a December peak. A random split — or a test window that
ends in October — would grade the model on the easy 10 months and certify a
forecast that falls over exactly when staffing matters most.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema, synthetic

LAGS = (7, 14, 28)

# Fixed calendars, derived from the generator's exposed tables. When you adapt
# this to your network, these come from your holiday file and your marketing
# team's promo calendar — both known well before the forecast is made.
HOLIDAY_DATES = pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in synthetic.HOLIDAYS))
PROMO_DATES = pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in synthetic.PROMO_EVENTS))


def _days_to_nearest_holiday(dates: pd.DatetimeIndex) -> np.ndarray:
    """Signed days to the nearest holiday (positive = holiday ahead), clipped.

    The clip at +/-30 matters: outside a month of any holiday the exact value
    carries no signal, and an unclipped feature hands the model hundreds of
    meaningless split points.
    """
    deltas = (HOLIDAY_DATES.values[None, :] - dates.values[:, None]).astype(
        "timedelta64[D]"
    ).astype(int)
    nearest = deltas[np.arange(len(dates)), np.abs(deltas).argmin(axis=1)]
    return np.clip(nearest, -30, 30)


def _in_peak_ramp(dates: pd.DatetimeIndex) -> np.ndarray:
    md = list(zip(dates.month, dates.day))
    return np.array(
        [(synthetic.PEAK_RAMP_START <= x <= synthetic.PEAK_RAMP_END) for x in md], dtype=int
    )


def build_features(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Hub-day feature frame: calendar + lag features, gap-tolerant.

    Rows whose target is missing (calendar gaps) are dropped — they are not
    observations. Rows missing ``lag_7`` are also dropped, because the
    seasonal-naive baseline cannot quote a number there and the model/baseline
    comparison must run on identical rows.
    """
    df = df_clean.sort_values([schema.HUB_COL, schema.DATE_COL])
    full_idx = pd.date_range(df[schema.DATE_COL].min(), df[schema.DATE_COL].max(), freq="D")

    # Wide hub x day matrix on the complete calendar: gaps become NaN, so a
    # shift(7) below is always "7 calendar days ago", never "7 rows ago" —
    # shifting rows across a gap silently turns lag-7 into lag-9.
    wide = df.pivot(
        index=schema.DATE_COL, columns=schema.HUB_COL, values=schema.TARGET_COL
    ).reindex(full_idx)

    pieces = {f"lag_{k}": wide.shift(k) for k in LAGS}
    # Trailing means end at yesterday (shift(1)): t-1..t-7 and t-1..t-28.
    pieces["trailing_mean_7"] = wide.shift(1).rolling(7, min_periods=5).mean()
    pieces["trailing_mean_28"] = wide.shift(1).rolling(28, min_periods=20).mean()

    long = wide.rename_axis(index=schema.DATE_COL).reset_index().melt(
        id_vars=schema.DATE_COL, var_name=schema.HUB_COL, value_name=schema.TARGET_COL
    )
    for name, piece in pieces.items():
        melted = piece.rename_axis(index=schema.DATE_COL).reset_index().melt(
            id_vars=schema.DATE_COL, var_name=schema.HUB_COL, value_name=name
        )
        long = long.merge(melted, on=[schema.DATE_COL, schema.HUB_COL], how="left")

    long = long.dropna(subset=[schema.TARGET_COL, "lag_7"]).reset_index(drop=True)

    # Calendar features: all knowable arbitrarily far in advance.
    dates = pd.DatetimeIndex(long[schema.DATE_COL])
    long["day_of_week"] = dates.dayofweek
    long["month"] = dates.month
    doy = dates.dayofyear.to_numpy()
    long["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    long["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    long["is_holiday"] = dates.isin(HOLIDAY_DATES).astype(int)
    long["days_to_nearest_holiday"] = _days_to_nearest_holiday(dates)
    long["is_peak_ramp"] = _in_peak_ramp(dates)
    long["is_promo_day"] = dates.isin(PROMO_DATES).astype(int)

    # Planted noise columns ride along so SHAP has negatives to bury.
    noise = df_clean[[schema.HUB_COL, *schema.NOISE_FEATURES]].drop_duplicates(schema.HUB_COL)
    long = long.merge(noise[[schema.HUB_COL, "hub_paint_color_code"]], on=schema.HUB_COL)
    long["moon_phase"] = (
        0.5 + 0.5 * np.sin(2 * np.pi * (dates - dates.min()).days.to_numpy() / 29.53)
    ).round(4)

    return long


HISTORY_FEATURES = ["lag_7", "lag_14", "lag_28", "trailing_mean_7", "trailing_mean_28"]
CALENDAR_FEATURES = [
    "day_of_week", "month", "doy_sin", "doy_cos",
    "is_holiday", "days_to_nearest_holiday", "is_peak_ramp", "is_promo_day",
]
NUMERIC_FEATURES = HISTORY_FEATURES + CALENDAR_FEATURES + schema.NOISE_FEATURES


def to_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Model matrix: numeric features + one-hot hub identity.

    Two encoding choices that matter:

    - History features enter as ``log1p``, matching the log-scale target. The
      volume process is multiplicative, so on the log scale "tomorrow = last
      week times a calendar factor" is *additive* — without the transform the
      trees burn depth re-deriving the log curve separately for a 50k-parcel
      superhub and a 2k-parcel regional sort, and the leftovers show up as
      bias in exactly the highest-volume weeks.
    - One-hot hub identity (not a single integer code) keeps every SHAP value
      attached to a named facility — "the model leans on MEM's identity" is a
      sentence an ops manager can act on; "hub_code contributed 0.3" is not.
    """
    X = df[NUMERIC_FEATURES].astype(float)
    X[HISTORY_FEATURES] = np.log1p(X[HISTORY_FEATURES])
    dummies = pd.get_dummies(df[schema.HUB_COL], prefix=schema.HUB_COL, dtype=float)
    return pd.concat([X, dummies], axis=1)


def time_split(
    df: pd.DataFrame, test_days: int = 120
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Train on everything up to the cutoff; test on the final ``test_days``.

    120 days back from a late-December end date puts the entire Nov-Dec peak
    in the test window — the season the forecast exists for.
    """
    cutoff = df[schema.DATE_COL].max() - pd.Timedelta(days=test_days)
    train = df[df[schema.DATE_COL] <= cutoff]
    test = df[df[schema.DATE_COL] > cutoff]
    return train, test, cutoff


def align_columns(X_train: pd.DataFrame, X_other: pd.DataFrame) -> pd.DataFrame:
    """Reindex another matrix to the training columns (unseen dummies -> 0)."""
    return X_other.reindex(columns=X_train.columns, fill_value=0.0)
