"""Model training: a point-ETA baseline plus the quantile models that are the product.

Three model families, on purpose:

- **Linear regression** — the honest baseline on scaled features. If a
  gradient-boosted model can't clearly beat it, the transit process is mostly
  additive and you should ship the simple model.
- **XGBoost point model** (``reg:squarederror``) — the conventional "one number"
  ETA, kept for comparison and for the pred-vs-actual chart.
- **XGBoost quantile models** (``reg:quantileerror`` with ``quantile_alpha``,
  available since xgboost 2.0) — one model per quantile. P50 is the ETA you
  display; P10-P90 is the honest range; ceil(P90) is the transit time you can
  *promise* a customer and keep ~9 times out of 10. The generator's noise
  grows with distance and congestion, so a single point estimate understates
  risk exactly where promises matter most: long congested lanes.

Beyond the product quantiles (0.1, 0.5, 0.9) we fit a few extra alphas so the
evaluation step can draw the promise curve (promised-by rate vs quoted
quantile) from real models instead of interpolation. Each extra model is a few
seconds of training; the curve is the chart the revenue team argues over.

Independently trained quantile models can cross (a P10 above the P50 for some
row) — rare, but one crossed pair on an ops screen destroys trust in all three
numbers. ``predict_quantiles`` therefore sorts each row's predictions across
the quantile grid (monotone rearrangement), which never hurts pinball loss and
guarantees p10 <= p50 <= p90.

Early stopping validates on the last slice of *training* time only, so the
held-out test period stays untouched.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from . import features, schema

# The quantiles the product quotes ...
PRODUCT_QUANTILES = (0.1, 0.5, 0.9)
# ... plus extra alphas used only to trace the promise curve in evaluate.py.
CURVE_QUANTILES = (0.3, 0.7, 0.8, 0.95)
ALL_QUANTILES = tuple(sorted(set(PRODUCT_QUANTILES) | set(CURVE_QUANTILES)))


@dataclass
class TrainConfig:
    test_frac: float = 0.2
    seed: int = 7
    n_estimators: int = 600
    max_depth: int = 5
    learning_rate: float = 0.05
    min_child_weight: int = 20
    early_stopping_rounds: int = 40
    quantiles: tuple[float, ...] = ALL_QUANTILES

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainedModels:
    baseline: Pipeline
    xgb_point: XGBRegressor
    xgb_quantiles: dict[float, XGBRegressor] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)
    cutoff_date: str = ""
    config: TrainConfig | None = None


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
    """Train baseline + point + quantile models on a cleaned shipment table.

    Returns the trained models and a dict of the split frames for evaluation:
    {"train": ..., "test": ..., "X_train": ..., "X_test": ..., "y_train": ..., "y_test": ...}
    """
    config = config or TrainConfig()
    schema.validate(df_clean)

    df = features.engineer(df_clean)
    train_df, test_df, cutoff = features.time_split(df, config.test_frac)

    X_train = features.to_matrix(train_df)
    X_test = features.align_columns(X_train, features.to_matrix(test_df))
    y_train = train_df[schema.LABEL_COL].to_numpy()
    y_test = test_df[schema.LABEL_COL].to_numpy()

    baseline = Pipeline([("scale", StandardScaler()), ("linreg", LinearRegression())])
    baseline.fit(X_train, y_train)

    # Hold out the final slice of *training* time for early stopping, so the
    # test period stays untouched.
    es_train, es_valid, _ = features.time_split(train_df, 0.15)
    X_es_train = features.align_columns(X_train, features.to_matrix(es_train))
    X_es_valid = features.align_columns(X_train, features.to_matrix(es_valid))
    y_es_train = es_train[schema.LABEL_COL].to_numpy()
    y_es_valid = es_valid[schema.LABEL_COL].to_numpy()
    eval_set = [(X_es_valid, y_es_valid)]

    xgb_point = XGBRegressor(objective="reg:squarederror", **_xgb_params(config))
    xgb_point.fit(X_es_train, y_es_train, eval_set=eval_set, verbose=False)

    xgb_quantiles: dict[float, XGBRegressor] = {}
    for alpha in config.quantiles:
        m = XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=alpha, **_xgb_params(config)
        )
        m.fit(X_es_train, y_es_train, eval_set=eval_set, verbose=False)
        xgb_quantiles[alpha] = m

    models = TrainedModels(
        baseline=baseline,
        xgb_point=xgb_point,
        xgb_quantiles=xgb_quantiles,
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


def predict_quantiles(
    models: TrainedModels, X: pd.DataFrame, alphas: tuple[float, ...] | None = None
) -> pd.DataFrame:
    """Predict quantiles for X, monotone-rearranged so rows never cross.

    Returns a DataFrame with one column per alpha (ascending), same index as X.
    """
    alphas = tuple(sorted(alphas or models.config.quantiles))
    preds = np.column_stack([models.xgb_quantiles[a].predict(X) for a in alphas])
    preds = np.sort(preds, axis=1)  # monotone rearrangement across the quantile grid
    preds = np.maximum(preds, 0.05)  # a quoted transit time below ~1 hour is a data bug
    return pd.DataFrame(preds, columns=list(alphas), index=X.index)


def save(models: TrainedModels, model_dir: str | Path) -> Path:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / "eta_regression_models.joblib"
    joblib.dump(models, path)
    return path


def load(model_dir: str | Path) -> TrainedModels:
    return joblib.load(Path(model_dir) / "eta_regression_models.joblib")
