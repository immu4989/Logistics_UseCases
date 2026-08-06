"""Model training.

Two models, on purpose:

- **Logistic regression** — the honest baseline. If a gradient-boosted model
  can't clearly beat it, your signal is mostly linear and you should ship the
  simple model.
- **XGBoost** — the workhorse for tabular operational data; captures the
  interactions that actually matter in a parcel network (weather x service
  level, peak x congestion) and pairs with exact TreeSHAP for explanation.

Class imbalance: at ~10% positives we deliberately do NOT reweight or
resample. Reweighting buys nothing for ranking metrics here and it wrecks
probability calibration — the model starts telling ops "80% risk" for
shipments that miss 30% of the time. If your miss rate is far rarer (<1-2%),
reweight for trainability and then recalibrate (e.g. isotonic on a held-out
slice) before anyone consumes the probabilities.
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

from . import conformal, features, schema


@dataclass
class TrainConfig:
    test_frac: float = 0.2
    seed: int = 7
    n_estimators: int = 800
    max_depth: int = 4
    learning_rate: float = 0.05
    min_child_weight: int = 20
    early_stopping_rounds: int = 40

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainedModels:
    baseline: Pipeline
    xgb: XGBClassifier
    feature_columns: list[str]
    cutoff_date: str
    config: TrainConfig
    # Conformal layer (None-defaults keep pre-conformal pickles loadable;
    # consumers must use getattr(..., None) because unpickling a dataclass
    # bypasses __init__ and old artifacts simply lack these attributes).
    calibrator: conformal.CalibratedScorer | None = None
    crc_thresholds: dict[float, conformal.CRCThreshold] | None = None


def train(df_clean: pd.DataFrame, config: TrainConfig | None = None) -> tuple[TrainedModels, dict]:
    """Train baseline + XGBoost on a cleaned shipment table.

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

    # Hold out the final slice of *training* time for early stopping, so the
    # test period stays untouched.
    es_train, es_valid, _ = features.time_split(train_df, 0.15)
    X_es_train = features.align_columns(X_train, features.to_matrix(es_train))
    X_es_valid = features.align_columns(X_train, features.to_matrix(es_valid))

    xgb = XGBClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=config.min_child_weight,
        eval_metric="aucpr",
        early_stopping_rounds=config.early_stopping_rounds,
        random_state=config.seed,
        n_jobs=-1,
    )
    xgb.fit(
        X_es_train,
        es_train[schema.LABEL_COL].to_numpy(),
        eval_set=[(X_es_valid, es_valid[schema.LABEL_COL].to_numpy())],
        verbose=False,
    )

    # Conformal layer: isotonic calibration + CRC flag thresholds, fitted on
    # the SAME final-slice-of-training-time used for early stopping above.
    # Honest caveat on the reuse: early stopping already peeked at this slice
    # to pick the tree count, so the calibrated probabilities and thresholds
    # inherit a mild optimism. We accept that in exchange for not burning a
    # third time window (the same trade eta-regression makes in
    # conformal_qhat); with more history, give calibration its own slice.
    # CRC runs on the RAW score, not the calibrated one: isotonic is piecewise
    # constant, and thresholding inside one of its plateaus forces the whole
    # plateau into the flag set (measured cost at alpha=0.20: ~77% flagged vs
    # ~62% on raw scores, same guarantee). Division of labor: isotonic fixes
    # probability levels, CRC certifies the flag set on the fine-grained
    # ranking. The guarantee holds for any fixed score function.
    y_cal = es_valid[schema.LABEL_COL].to_numpy()
    calibrator = conformal.calibrate_probabilities(xgb, X_es_valid, y_cal)
    s_cal = xgb.predict_proba(X_es_valid)[:, 1]
    crc_thresholds = {
        alpha: conformal.crc_threshold(s_cal, y_cal, alpha) for alpha in conformal.DEFAULT_ALPHAS
    }

    models = TrainedModels(
        baseline=baseline,
        xgb=xgb,
        feature_columns=list(X_train.columns),
        cutoff_date=str(cutoff.date()),
        config=config,
        calibrator=calibrator,
        crc_thresholds=crc_thresholds,
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
    path = model_dir / "delivery_commit_models.joblib"
    joblib.dump(models, path)
    return path


def load(model_dir: str | Path) -> TrainedModels:
    return joblib.load(Path(model_dir) / "delivery_commit_models.joblib")
