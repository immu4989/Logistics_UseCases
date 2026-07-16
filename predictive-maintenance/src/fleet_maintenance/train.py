"""Model training: the rule the fleet already runs, plus two learned challengers.

Three policies, on purpose:

- **Mileage rule** — the status quo in most fleets: service whatever has gone
  longest since its last service, measured in miles. It costs nothing to run
  and it is the bar any model must clear *at the same workshop capacity*, or
  the ML project has negative value.
- **Logistic regression** — the honest learned baseline on scaled features.
- **XGBoost** — the workhorse for this shape of problem. Breakdown hazard is
  exponential in wear and the sensors read wear linearly, so the signal is
  strongly nonlinear, and it is conditional (low battery voltage in January
  means winter; low voltage in July means a dying battery). Trees capture
  both.

Class imbalance: at ~2-4% positives we still do NOT reweight. Ranking is what
the workshop consumes (top-k per day at fixed bay capacity), reweighting does
not change the ranking of a well-fit model, and it wrecks the probabilities.
Early stopping holds out the final slice of *training* time, so the test
period stays untouched.
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

from . import features, synthetic


@dataclass
class TrainConfig:
    test_days: int = features.TEST_DAYS
    seed: int = 7
    n_estimators: int = 1500
    max_depth: int = 4
    learning_rate: float = 0.03
    min_child_weight: int = 10
    colsample_bytree: float = 0.6
    reg_lambda: float = 5.0
    early_stopping_rounds: int = 60

    def to_dict(self) -> dict:
        return asdict(self)


class MileageRule:
    """The status-quo policy: rank vehicles by miles since last service.

    Flagging everything past X miles and ranking by miles-since-service are
    the same policy once a daily capacity is imposed — capacity picks the
    effective X each morning.
    """

    score_col = "miles_since_maint"

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.score_col].to_numpy(dtype=float)


@dataclass
class TrainedModels:
    mileage_rule: MileageRule
    baseline: Pipeline
    xgb: XGBClassifier
    feature_columns: list[str]
    cutoff_date: str
    config: TrainConfig


def train(df_clean: pd.DataFrame, config: TrainConfig | None = None) -> tuple[TrainedModels, dict]:
    """Train all policies on a cleaned panel.

    Returns the trained models plus the split frames for evaluation:
    {"train": ..., "test": ..., "X_train": ..., "X_test": ..., "y_train": ..., "y_test": ...}
    """
    config = config or TrainConfig()

    df = features.drop_warmup(features.engineer(df_clean))
    train_df, test_df, cutoff = features.time_split(df, config.test_days)

    X_train = features.to_matrix(train_df)
    X_test = features.align_columns(X_train, features.to_matrix(test_df))
    y_train = train_df[synthetic.LABEL_COL].to_numpy()
    y_test = test_df[synthetic.LABEL_COL].to_numpy()

    baseline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=3000, random_state=config.seed)),
        ]
    )
    baseline.fit(X_train, y_train)

    # Tree-count selection: early-stop against the final slice of *training*
    # time (never the test period), then refit at that size on the full
    # training window. Without the refit, XGBoost would train on ~6 fewer
    # weeks of the most recent data than the logistic baseline — a systematic
    # handicap that has nothing to do with model quality.
    es_days = max(30, int(config.test_days * 0.5))
    es_train, es_valid, _ = features.time_split(train_df, es_days)
    X_es_train = features.align_columns(X_train, features.to_matrix(es_train))
    X_es_valid = features.align_columns(X_train, features.to_matrix(es_valid))

    xgb_params = dict(
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=0.9,
        colsample_bytree=config.colsample_bytree,
        min_child_weight=config.min_child_weight,
        reg_lambda=config.reg_lambda,
        eval_metric="logloss",
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
        es_train[synthetic.LABEL_COL].to_numpy(),
        eval_set=[(X_es_valid, es_valid[synthetic.LABEL_COL].to_numpy())],
        verbose=False,
    )
    n_trees = max(int(probe.best_iteration) + 1, 50)

    xgb = XGBClassifier(n_estimators=n_trees, **xgb_params)
    xgb.fit(X_train, y_train, verbose=False)

    models = TrainedModels(
        mileage_rule=MileageRule(),
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
    path = model_dir / "fleet_maintenance_models.joblib"
    joblib.dump(models, path)
    return path


def load(model_dir: str | Path) -> TrainedModels:
    return joblib.load(Path(model_dir) / "fleet_maintenance_models.joblib")
