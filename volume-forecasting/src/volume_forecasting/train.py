"""Model training: the ops-floor baseline plus point and quantile forecasters.

Three model families, on purpose:

- **Seasonal naive** — tomorrow's volume equals the same weekday last week
  (``lag_7``). This is the forecast every ops floor already runs on, usually
  as a cell in a spreadsheet, so it is the number a model must beat *by a
  margin that survives deployment friction* before anyone changes a staffing
  process for it.
- **XGBoost point model** — the conventional one-number forecast, kept for the
  headline WAPE comparison and the overlay chart.
- **XGBoost quantile models** (``reg:quantileerror`` with ``quantile_alpha``,
  available since xgboost 2.0) — the product. Staffing is an asymmetric-cost
  decision: an understaffed peak Monday costs service failures and overtime, a
  quiet overstaffed Tuesday costs a few idle hours. P80 is the capacity you
  staff to; P10-P90 is the honest range you show the planner.

All XGBoost models fit on a **log-ratio target**: ``log1p(volume)`` minus a
per-row anchor, where the anchor is ``log1p(trailing_mean_7)`` (falling back
to ``log1p(lag_7)`` when a gap thins the trailing window). This one decision
does most of the work in this module, for three reasons. First, the volume
process is multiplicative (a 20x bigger hub has 20x bigger absolute swings),
so on the log scale the noise is additive and the superhub stops soaking up
the entire loss. Second, differencing against recent history cancels hub scale
and trend entirely: the model only has to learn the calendar structure (the
weekly rhythm, holiday collapses, the December ramp, promo days) that makes
tomorrow different from last week — a compact, hub-shared function instead of
fifteen level curves. Trees cannot extrapolate a trend, but a ratio model
never has to. The trailing *mean* is the anchor rather than the naive's single
``lag_7`` day because averaging seven days cuts the anchor's own noise by
~sqrt(7) — anchoring on one noisy Monday would bake that Monday's luck into
every quantile. Third, quantiles survive the trip: adding the anchor back and
``expm1`` are monotone per row, so the P80 of the ratio maps back to exactly
the P80 of the volume.

Two guards keep the quantiles honest:

- **Split-conformal calibration (CQR).** In-sample quantile fits are always a
  little too sure of themselves: the model's own estimation error is invisible
  to it. Each quantile's calibration margin is measured on the early-stopping
  validation slice — the last 90 days of training time, where the probe
  model's predictions are out-of-sample — and added at prediction time. It is
  a two-line fix that moves P10-P90 coverage from ~72% to the nominal 80%
  without touching the models.
- **Monotone rearrangement.** Independently trained quantile models can cross
  (a P10 above the P50 for some row). One crossed pair on a planner's screen
  discredits all three numbers, so ``predict_quantiles`` sorts each row across
  the quantile grid — it never hurts pinball loss and guarantees
  p10 <= p50 <= p80 <= p90.

Early stopping validates on the last slice of *training* time only — the
held-out test window stays untouched — and then each model is **refit on the
full training window** with the tree count early stopping selected, so the
final models are not blind to the most recent 90 days of history.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from . import features, schema

# 0.1 / 0.5 / 0.9 are the range shown to the planner; 0.8 is the staffing
# quantile the evaluation's understaffed-days table is built on.
QUANTILES = (0.1, 0.5, 0.8, 0.9)


class SeasonalNaive:
    """Same weekday last week. The spreadsheet the model has to beat.

    Reads ``lag_7`` from the model matrix, where history features are stored
    as ``log1p`` (see features.to_matrix), hence the expm1 on the way out.
    """

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.expm1(X["lag_7"].to_numpy())


@dataclass
class TrainConfig:
    test_days: int = 120
    seed: int = 7
    n_estimators: int = 1000
    # Deliberately shallow and heavily minimum-weighted: the ratio target is a
    # compact calendar function shared across hubs, and a deeper tree mostly
    # memorises training noise — which quietly narrows the quantile spread
    # in-sample and costs interval coverage out-of-sample.
    max_depth: int = 4
    learning_rate: float = 0.05
    min_child_weight: int = 20
    early_stopping_rounds: int = 40
    quantiles: tuple[float, ...] = QUANTILES

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainedModels:
    naive: SeasonalNaive
    xgb_point: XGBRegressor
    xgb_quantiles: dict[float, XGBRegressor] = field(default_factory=dict)
    # CQR calibration margins per quantile, on the log-ratio scale (see module
    # docstring). Added to raw predictions in predict_quantiles.
    quantile_margins: dict[float, float] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)
    cutoff_date: str = ""
    config: TrainConfig | None = None


def _anchor(X: pd.DataFrame) -> np.ndarray:
    """Per-row log-scale anchor: trailing 7-day mean, lag_7 where gaps thin it.

    The matrix stores history features as log1p (features.to_matrix), so this
    is already on the target's scale.
    """
    tm7 = X["trailing_mean_7"].to_numpy()
    lag7 = X["lag_7"].to_numpy()
    return np.where(np.isnan(tm7), lag7, tm7)


def _xgb_params(config: TrainConfig) -> dict:
    return dict(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=config.min_child_weight,
        early_stopping_rounds=config.early_stopping_rounds,
        random_state=config.seed,
        n_jobs=-1,
    )


def train(df_clean: pd.DataFrame, config: TrainConfig | None = None) -> tuple[TrainedModels, dict]:
    """Train naive + point + quantile models on a cleaned volume table.

    Returns the trained models and a dict of split frames for evaluation:
    {"train", "test", "X_train", "X_test", "y_train", "y_test"}.
    """
    config = config or TrainConfig()
    schema.validate(df_clean)

    df = features.build_features(df_clean)
    train_df, test_df, cutoff = features.time_split(df, config.test_days)

    X_train = features.to_matrix(train_df)
    X_test = features.align_columns(X_train, features.to_matrix(test_df))
    y_train = train_df[schema.TARGET_COL].to_numpy()
    y_test = test_df[schema.TARGET_COL].to_numpy()

    # The regression target: log-ratio to the anchor (see module docstring).
    def _ratio_target(frame: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
        return np.log1p(frame[schema.TARGET_COL].to_numpy()) - _anchor(X)

    # Hold out the final slice of *training* time for early stopping and CQR
    # calibration; the test window stays untouched.
    es_train, es_valid, _ = features.time_split(train_df, test_days=90)
    X_es_train = features.align_columns(X_train, features.to_matrix(es_train))
    X_es_valid = features.align_columns(X_train, features.to_matrix(es_valid))
    y_es_train = _ratio_target(es_train, X_es_train)
    y_es_valid = _ratio_target(es_valid, X_es_valid)
    eval_set = [(X_es_valid, y_es_valid)]
    y_train_ratio = _ratio_target(train_df, X_train)

    def _fit_with_refit(objective: str, **extra) -> tuple[XGBRegressor, np.ndarray]:
        """Early-stop on the validation slice, then refit on all training data.

        Returns the refit model plus the probe's out-of-sample predictions on
        the validation slice (the raw material for CQR calibration).
        """
        probe = XGBRegressor(objective=objective, **extra, **_xgb_params(config))
        probe.fit(X_es_train, y_es_train, eval_set=eval_set, verbose=False)
        params = _xgb_params(config)
        params.pop("early_stopping_rounds")
        params["n_estimators"] = int(probe.best_iteration) + 1
        final = XGBRegressor(objective=objective, **extra, **params)
        final.fit(X_train, y_train_ratio, verbose=False)
        return final, probe.predict(X_es_valid)

    xgb_point, _ = _fit_with_refit("reg:squarederror")
    xgb_quantiles, valid_preds = {}, {}
    for alpha in config.quantiles:
        xgb_quantiles[alpha], valid_preds[alpha] = _fit_with_refit(
            "reg:quantileerror", quantile_alpha=alpha
        )

    # CQR margins from the validation slice. The 10/90 pair shares one
    # two-sided margin (the standard CQR score); interior quantiles get a
    # one-sided margin at their own level. P50 is left alone — a median
    # correction of a fraction of a percent is not worth the moving part.
    band_score = np.maximum(valid_preds[0.1] - y_es_valid, y_es_valid - valid_preds[0.9])
    band_margin = float(np.quantile(band_score, 0.80))
    quantile_margins = {0.1: -band_margin, 0.9: band_margin, 0.5: 0.0}
    for alpha in config.quantiles:
        if alpha not in quantile_margins:
            quantile_margins[alpha] = float(
                np.quantile(y_es_valid - valid_preds[alpha], alpha)
            )

    models = TrainedModels(
        naive=SeasonalNaive(),
        xgb_point=xgb_point,
        xgb_quantiles=xgb_quantiles,
        quantile_margins=quantile_margins,
        feature_columns=list(X_train.columns),
        cutoff_date=str(cutoff.date()),
        config=config,
    )
    splits = {
        "train": train_df,
        "test": test_df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }
    return models, splits


def predict_point(models: TrainedModels, X: pd.DataFrame) -> np.ndarray:
    """Point forecast in parcels: predicted log-ratio + anchor, inverted."""
    return np.expm1(models.xgb_point.predict(X) + _anchor(X)).clip(min=0.0)


def predict_quantiles(models: TrainedModels, X: pd.DataFrame) -> pd.DataFrame:
    """Calibrated quantile forecasts in parcels, monotone-rearranged per row.

    Adding back the (per-row constant) anchor and applying expm1 are both
    monotone, so each column really is the corresponding volume quantile.
    """
    alphas = sorted(models.xgb_quantiles)
    anchor = _anchor(X)
    preds = np.column_stack(
        [
            np.expm1(
                models.xgb_quantiles[a].predict(X) + models.quantile_margins[a] + anchor
            ).clip(min=0.0)
            for a in alphas
        ]
    )
    preds.sort(axis=1)  # monotone rearrangement: no crossed quantiles on screen
    return pd.DataFrame(preds, columns=[f"p{int(a * 100)}" for a in alphas], index=X.index)


def save(models: TrainedModels, model_dir: str | Path) -> Path:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / "volume_forecasting_models.joblib"
    joblib.dump(models, path)
    return path


def load(model_dir: str | Path) -> TrainedModels:
    return joblib.load(Path(model_dir) / "volume_forecasting_models.joblib")
