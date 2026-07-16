"""Quantile demand forecasting: the planner-habit baseline plus GBM quantiles.

Two model families, on purpose:

- **Seasonal naive** — book what this week did last year (``lag_52``), falling
  back to the last four observed weeks when last year is a gap. This is not a
  strawman: it is how most linehaul desks actually plan, and it doubles as the
  ``book_last_year`` policy in decide.py.
- **GradientBoostingRegressor quantile models** (``loss="quantile"``) — one
  model per fractile the economics needs. Which fractiles? decide.py owns
  that answer: the P50 (for the book-the-mean comparison policy) plus the
  critical fractile under each cost scenario. The forecast layer deliberately
  asks the decision layer what to predict, not the other way round.

There is no SHAP in this use case, and that is a design decision rather than
a shortcut: the explanation a capacity planner needs is not "which feature
moved the quantile" but "why book 23 trailers and not 25", and that answer is
the critical-fractile arithmetic in decide.py plus the per-lane rationale in
explain.py. Plain sklearn is enough to carry the distribution; the economics
carries the explanation.

Modeling choices that matter:

- Features are lags only, never same-week information: bookings are placed a
  week ahead, so the model sees demand history through last week (lags 1, 2,
  4 and 52), the calendar (week-of-year sine/cosine, the published promo
  calendar, the peak-ramp weeks) and the lane identity as one-hots. Lags are
  computed on a per-lane weekly calendar so a gap never silently turns lag-1
  into lag-2; a lag that reaches into a gap is imputed from the lane's own
  trailing mean and flagged.
- The target is ``log1p(demand)``. Demand composes multiplicatively (peak is
  +38% on every lane size), so on the log scale the lane spread stops soaking
  up the loss; and because ``expm1`` is monotone, the q-th quantile of the log
  maps back to exactly the q-th quantile of demand.
- Quantile models are trained independently, so per row the three predictions
  are sorted into fractile order before use (monotone rearrangement). One
  crossed pair on a planner's screen discredits all three numbers.
- The split is time-based and the test window is pinned to the final 16 weeks
  so it must contain a full year-end peak — the season a booking desk earns
  or loses its budget.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from . import decide, synthetic

LAGS = (1, 2, 4, 52)
TEST_WEEKS = 16

# The fractiles worth a model: decide.py's economics, plus the median for the
# book-the-mean comparison. Keys are the column names used everywhere.
QUANTILE_ROLES = {
    "q_base": decide.critical_fractile(decide.SPOT_COST_USD),
    "p50": 0.5,
    "q_tight": decide.critical_fractile(decide.SPOT_COST_TIGHT_USD),
}

PROMO_WEEKS = pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in synthetic.PROMO_EVENTS))

HISTORY_FEATURES = ["lag_1", "lag_2", "lag_4", "lag_52", "trailing_mean_4"]
CALENDAR_FEATURES = ["woy_sin", "woy_cos", "is_promo_week", "is_peak_ramp"]
FLAG_FEATURES = ["lag_52_was_missing"]
NUMERIC_FEATURES = HISTORY_FEATURES + CALENDAR_FEATURES + FLAG_FEATURES


class SeasonalNaive:
    """Same week last year, fallback last-4-week mean. The planner habit."""

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["naive_seasonal"].to_numpy()


@dataclass
class TrainConfig:
    test_weeks: int = TEST_WEEKS
    seed: int = 7
    n_estimators: int = 300
    # Shallow and heavily leaf-weighted: the calendar structure is a compact
    # function shared across lanes, and a deeper tree mostly memorises lane
    # noise — which quietly biases the quantiles toward the training draws.
    max_depth: int = 3
    learning_rate: float = 0.06
    min_samples_leaf: int = 25
    subsample: float = 0.9

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainedModels:
    naive: SeasonalNaive
    quantile_models: dict[str, GradientBoostingRegressor] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)
    cutoff_week: str = ""
    config: TrainConfig | None = None


def build_features(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Lane-week feature frame: history lags + calendar, gap-tolerant.

    Lags are shifted on a complete per-lane weekly calendar (gaps become NaN,
    so ``shift(1)`` is always "one calendar week ago", never "one row ago").
    Rows whose own demand is missing are dropped — they are not observations.
    A missing ``lag_52`` (first year of history, or last year was a gap) is
    imputed with the lane's trailing 4-week mean and flagged; short gaps in
    the recent lags are imputed the same way, unflagged.
    """
    df = clean_df.sort_values([synthetic.LANE_COL, synthetic.WEEK_COL])
    full_idx = pd.date_range(
        df[synthetic.WEEK_COL].min(), df[synthetic.WEEK_COL].max(), freq="7D"
    )
    wide = df.pivot(
        index=synthetic.WEEK_COL, columns=synthetic.LANE_COL, values=synthetic.TARGET_COL
    ).reindex(full_idx)

    pieces = {f"lag_{k}": wide.shift(k) for k in LAGS}
    pieces["trailing_mean_4"] = wide.shift(1).rolling(4, min_periods=2).mean()

    long = wide.rename_axis(index=synthetic.WEEK_COL).reset_index().melt(
        id_vars=synthetic.WEEK_COL, var_name=synthetic.LANE_COL, value_name=synthetic.TARGET_COL
    )
    for name, piece in pieces.items():
        melted = piece.rename_axis(index=synthetic.WEEK_COL).reset_index().melt(
            id_vars=synthetic.WEEK_COL, var_name=synthetic.LANE_COL, value_name=name
        )
        long = long.merge(melted, on=[synthetic.WEEK_COL, synthetic.LANE_COL], how="left")
    long = long.dropna(subset=[synthetic.TARGET_COL, "trailing_mean_4"]).reset_index(drop=True)

    # The habit's number: same week last year, fallback the last 4 weeks.
    long["naive_seasonal"] = long["lag_52"].fillna(long["trailing_mean_4"])
    long["lag_52_was_missing"] = long["lag_52"].isna().astype(int)
    long["lag_52"] = long["lag_52"].fillna(long["trailing_mean_4"])
    for col in ["lag_1", "lag_2", "lag_4"]:
        long[col] = long[col].fillna(long["trailing_mean_4"])

    # Calendar features: all knowable arbitrarily far ahead of the booking.
    weeks = pd.DatetimeIndex(long[synthetic.WEEK_COL])
    woy = weeks.isocalendar().week.to_numpy().astype(int)
    long["woy_sin"] = np.sin(2 * np.pi * woy / 52.18)
    long["woy_cos"] = np.cos(2 * np.pi * woy / 52.18)
    long["is_promo_week"] = weeks.isin(PROMO_WEEKS).astype(int)
    long["is_peak_ramp"] = np.isin(woy, synthetic.PEAK_WOY).astype(int)
    return long


def to_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Model matrix: log1p history features + calendar + one-hot lane identity."""
    X = df[NUMERIC_FEATURES].astype(float)
    X[HISTORY_FEATURES] = np.log1p(X[HISTORY_FEATURES])
    dummies = pd.get_dummies(df[synthetic.LANE_COL], prefix="lane", dtype=float)
    return pd.concat([X, dummies], axis=1)


def time_split(
    df: pd.DataFrame, test_weeks: int = TEST_WEEKS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Train on everything up to the cutoff; test on the final ``test_weeks``.

    16 weeks back from a mid-January end date puts the entire Nov/Dec peak in
    the test window — the weeks a booking policy exists for.
    """
    cutoff = df[synthetic.WEEK_COL].max() - pd.Timedelta(weeks=test_weeks)
    return df[df[synthetic.WEEK_COL] <= cutoff], df[df[synthetic.WEEK_COL] > cutoff], cutoff


def align_columns(X_train: pd.DataFrame, X_other: pd.DataFrame) -> pd.DataFrame:
    return X_other.reindex(columns=X_train.columns, fill_value=0.0)


def train(clean_df: pd.DataFrame, config: TrainConfig | None = None) -> tuple[TrainedModels, dict]:
    """Train the naive baseline + one GBM per needed fractile.

    Returns the models and a splits dict: {"train", "test", "X_train",
    "X_test", "y_train", "y_test"}.
    """
    config = config or TrainConfig()
    df = build_features(clean_df)
    train_df, test_df, cutoff = time_split(df, config.test_weeks)

    X_train = to_matrix(train_df)
    X_test = align_columns(X_train, to_matrix(test_df))
    y_train = np.log1p(train_df[synthetic.TARGET_COL].to_numpy())

    quantile_models = {}
    for role, alpha in QUANTILE_ROLES.items():
        gbm = GradientBoostingRegressor(
            loss="quantile",
            alpha=alpha,
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            min_samples_leaf=config.min_samples_leaf,
            subsample=config.subsample,
            random_state=config.seed,
        )
        gbm.fit(X_train, y_train)
        quantile_models[role] = gbm

    models = TrainedModels(
        naive=SeasonalNaive(),
        quantile_models=quantile_models,
        feature_columns=list(X_train.columns),
        cutoff_week=str(cutoff.date()),
        config=config,
    )
    splits = {
        "train": train_df,
        "test": test_df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": train_df[synthetic.TARGET_COL].to_numpy(),
        "y_test": test_df[synthetic.TARGET_COL].to_numpy(),
    }
    return models, splits


def predict_quantiles(models: TrainedModels, X: pd.DataFrame) -> pd.DataFrame:
    """Quantile forecasts in trailer-equivalents, monotone-rearranged per row.

    ``expm1`` is monotone, so each column really is the demand quantile at its
    fractile; the per-row sort guarantees q_base <= p50 <= q_tight on screen.
    """
    roles = sorted(QUANTILE_ROLES, key=QUANTILE_ROLES.get)
    preds = np.column_stack(
        [np.expm1(models.quantile_models[r].predict(X)).clip(min=0.0) for r in roles]
    )
    preds.sort(axis=1)
    return pd.DataFrame(preds, columns=roles, index=X.index)


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Mean pinball loss: the proper score for a quantile forecast."""
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


def save(models: TrainedModels, model_dir: str | Path) -> Path:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / "capacity_planning_models.joblib"
    joblib.dump(models, path)
    return path


def load(model_dir: str | Path) -> TrainedModels:
    return joblib.load(Path(model_dir) / "capacity_planning_models.joblib")
