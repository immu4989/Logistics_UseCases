"""Model training: a rules router, a multinomial logistic, and XGBoost.

Three policies, on purpose:

- **Rules baseline** — the if/elif router every exception desk already runs,
  written honestly (flag precedence, a scan-gap cutoff). This is the
  comparator that earns the model its job: beating a strawman proves nothing.
- **Multinomial logistic** — the linear reference. The gap between it and the
  GBM measures how much of triage is threshold-and-interaction shaped.
- **XGBoost (multi:softprob)** — the workhorse. Its calibrated class
  probabilities are what make the confidence gate possible; an argmax-only
  model cannot tell you which tickets it is *sure* about.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import features, schema

# The scan-gap cutoff ops desks actually use ("silent for a day = chase it").
# The generator's true regime boundary is 36h; the deployed rule of thumb is
# 24h. That miscalibration is realistic and left in deliberately.
RULES_SCAN_GAP_CUTOFF_H = 24.0


def rules_route(df: pd.DataFrame) -> np.ndarray:
    """The hand-written router an exception desk deploys on day one.

    Precedence mirrors the SOP wall chart: hard evidence first (a damage scan
    settles the question), then paperwork, then the address, then weather,
    and only then the scan-gap judgement call. Each rule is individually
    sensible; the whole is what a strong non-ML comparator looks like.
    """
    conditions = [
        # A recorded damage scan trumps everything: claims need photos fast.
        df["damage_scan_flag"] == 1,
        # International + last seen at customs: it's a paperwork problem.
        (df["is_international"] == 1) & (df["last_scan_location_type"] == "customs"),
        # The validator already said the address is bad.
        df["address_validation_failed"] == 1,
        # Active weather at the last scan: sit tight, it self-resolves.
        df["weather_event_at_location"] == 1,
        # Silent past the cutoff: assume misroute, send a dispatcher.
        df["scan_gap_hours"] > RULES_SCAN_GAP_CUTOFF_H,
    ]
    choices = [
        "damage_claims",
        "customs_docs",
        "address_correction",
        "hold_and_monitor",
        "reroute",
    ]
    # Everything else: scans are current, so the blocker is on the customer
    # side (failed attempts, refusals, gate codes). Call them.
    return np.select(conditions, choices, default="customer_callback")


@dataclass
class TrainConfig:
    test_frac: float = 0.2
    seed: int = 7
    n_estimators: int = 600
    max_depth: int = 5
    learning_rate: float = 0.08
    min_child_weight: int = 20
    early_stopping_rounds: int = 30

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainedModels:
    baseline: Pipeline
    xgb: XGBClassifier
    classes: list[str]
    feature_columns: list[str]
    cutoff_date: str
    config: TrainConfig


def train(df_clean: pd.DataFrame, config: TrainConfig | None = None) -> tuple[TrainedModels, dict]:
    """Train logistic + XGBoost on a cleaned ticket table.

    Returns the trained models and the split frames for evaluation:
    {"train", "test", "X_train", "X_test", "y_train", "y_test"} where y_* are
    integer-encoded against the sorted class list.
    """
    config = config or TrainConfig()
    schema.validate(df_clean)

    train_df, test_df, cutoff = features.time_split(df_clean, config.test_frac)
    classes = sorted(df_clean[schema.LABEL_COL].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    X_train = features.to_matrix(train_df)
    X_test = features.align_columns(X_train, features.to_matrix(test_df))
    y_train = train_df[schema.LABEL_COL].map(class_to_idx).to_numpy()
    y_test = test_df[schema.LABEL_COL].map(class_to_idx).to_numpy()

    baseline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=3000, random_state=config.seed)),
        ]
    )
    baseline.fit(X_train, y_train)

    # Hold out the final slice of *training* time for early stopping, so the
    # test period stays untouched.
    es_train, es_valid, _ = features.time_split(train_df, 0.15)
    X_es_train = features.align_columns(X_train, features.to_matrix(es_train))
    X_es_valid = features.align_columns(X_train, features.to_matrix(es_valid))
    y_es_train = es_train[schema.LABEL_COL].map(class_to_idx).to_numpy()
    y_es_valid = es_valid[schema.LABEL_COL].map(class_to_idx).to_numpy()

    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=len(classes),
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=config.min_child_weight,
        eval_metric="mlogloss",
        early_stopping_rounds=config.early_stopping_rounds,
        random_state=config.seed,
        n_jobs=-1,
    )
    xgb.fit(X_es_train, y_es_train, eval_set=[(X_es_valid, y_es_valid)], verbose=False)

    models = TrainedModels(
        baseline=baseline,
        xgb=xgb,
        classes=classes,
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
    path = model_dir / "exception_triage_models.joblib"
    joblib.dump(models, path)
    return path


def load(model_dir: str | Path) -> TrainedModels:
    return joblib.load(Path(model_dir) / "exception_triage_models.joblib")
