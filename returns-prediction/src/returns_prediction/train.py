"""Model training.

Two models, on purpose:

- **Logistic regression** — the honest baseline. If a gradient-boosted model
  can't clearly beat it, your signal is mostly linear and you should ship the
  simple model.
- **XGBoost** — the workhorse for tabular commerce data. Returns behaviour is
  full of thresholds and interactions a linear model cannot express: deep
  discounts flip risky only past ~40% and mostly in fashion, a 60% prior
  return rate means something different over 2 orders than over 20, and
  bracket buying only exists inside two categories. This is where the GBM
  earns its keep (and on this dataset it does — see the README).

Class imbalance: at ~18% positives we deliberately do NOT reweight or
resample. Reweighting buys nothing for ranking metrics here and it wrecks
probability calibration — and this pipeline's entire product surface is
p * cost arithmetic, which is only as honest as p. If your return rate is far
rarer, reweight for trainability and recalibrate on a held-out slice before
anyone multiplies the probabilities by dollars.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import features, schema


@dataclass
class TrainConfig:
    test_frac: float = 0.2
    seed: int = 7
    n_estimators: int = 1200
    max_depth: int = 5
    learning_rate: float = 0.05
    min_child_weight: int = 40
    early_stopping_rounds: int = 60

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainedModels:
    baseline: Pipeline
    xgb: XGBClassifier
    feature_columns: list[str]
    cutoff_date: str
    config: TrainConfig


def train(df_clean: pd.DataFrame, config: TrainConfig | None = None) -> tuple[TrainedModels, dict]:
    """Train baseline + XGBoost on a cleaned order table.

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

    baseline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=2000, random_state=config.seed)),
        ]
    )
    baseline.fit(X_train, y_train)

    # Two-stage fit: hold out the final slice of *training* time to early-stop
    # a probe model (the test period stays untouched), then refit at the
    # chosen tree count on the FULL training window. Without the refit the
    # GBM never sees the most recent 15% of training history -- the slice
    # that matters most for the customer-history features -- and it gives up
    # a real chunk of held-out PR-AUC to the baseline.
    es_train, es_valid, _ = features.time_split(train_df, 0.15)
    X_es_train = features.align_columns(X_train, features.to_matrix(es_train))
    X_es_valid = features.align_columns(X_train, features.to_matrix(es_valid))

    xgb_params = dict(
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=config.min_child_weight,
        eval_metric="aucpr",
        random_state=config.seed,
        n_jobs=-1,
    )
    probe = XGBClassifier(
        n_estimators=config.n_estimators,
        early_stopping_rounds=config.early_stopping_rounds,
        **xgb_params,
    )
    probe.fit(
        X_es_train,
        es_train[schema.LABEL_COL].to_numpy(),
        eval_set=[(X_es_valid, es_valid[schema.LABEL_COL].to_numpy())],
        verbose=False,
    )
    xgb = XGBClassifier(n_estimators=probe.best_iteration + 1, **xgb_params)
    xgb.fit(X_train, y_train, verbose=False)

    models = TrainedModels(
        baseline=baseline,
        xgb=xgb,
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


def save(models: TrainedModels, model_dir: str | Path) -> Path:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / "returns_prediction_models.joblib"
    joblib.dump(models, path)
    return path


def load(model_dir: str | Path) -> TrainedModels:
    return joblib.load(Path(model_dir) / "returns_prediction_models.joblib")
