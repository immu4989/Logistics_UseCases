"""Acceptance-model training: learn P(accept | quote features, PRICE).

Two models, on purpose:

- **Logistic regression** — the honest baseline, with the price/reference
  ratio as an explicit coefficient anyone can read. Its known blind spot
  here: without hand-built interaction terms it must fit ONE price slope for
  every segment, and per-segment slopes are the whole game.
- **XGBoost** — learns the segment x price interaction on its own, which is
  exactly the elasticity structure the pricing sweep needs.

Price is a model INPUT, not a leak. The historical price is the treatment
the desk chose, known before the accept/reject outcome, and the entire
pricing step (``price.py``) works by sweeping candidate prices through this
model. A model that cannot answer "what if we had quoted $200 less" cannot
price anything.

Monotone constraint: P(accept) is constrained to be non-increasing in the
price features. Demand curves slope down; an unconstrained tree model fits
small upward-sloping pockets from noise, and a revenue optimizer will find
and exploit every one of them ("quote $80 more here, the model says they
accept MORE"). Baking the economics in as a constraint is standard practice
on real pricing desks, and it is what makes the elasticity-sign test in CI
meaningful rather than lucky.

Time-based split, never random. The market rate index random-walks across
the year, so quotes from the same week share a market regime. A random
split would scatter each regime across train and test and leak tomorrow's
market conditions into training; the honest question is "priced with last
year's model, how do this quarter's quotes go".
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

from . import cleaning

NUMERIC_FEATURES = [
    "distance_miles",
    "weight_lb",
    "volume_cuft",
    "market_rate_index",
    "fuel_index",
    "competitor_pressure",
]
PRICE_FEATURES = ["price_to_reference", "price_to_cost"]
SEGMENT_COL = "customer_segment"

# Never a feature: the generator's unobservable willingness term (the whole
# point is that the desk can't see it) and the outcome itself.
EXCLUDED = ["latent_willingness", "accepted", "quoted_price_usd"]


def to_matrix(df: pd.DataFrame, prices: np.ndarray) -> pd.DataFrame:
    """Model matrix for `df` priced at `prices` (any candidate prices, not
    just the historical ones — this is what makes the model sweepable)."""
    prices = np.asarray(prices, dtype=float)
    X = df[[c for c in NUMERIC_FEATURES if c in df.columns]].astype(float).copy()
    for c in df.columns:
        if c.endswith("__was_missing"):
            X[c] = df[c].astype(float)
    X["price_to_reference"] = prices / df["reference_price_usd"].to_numpy()
    X["price_to_cost"] = prices / df["our_cost_usd"].to_numpy()
    X["is_express"] = (df["urgency"] == "express").astype(float)
    dummies = pd.get_dummies(df[SEGMENT_COL], prefix="segment", dtype=float)
    return pd.concat([X, dummies], axis=1)


def align_columns(X_train: pd.DataFrame, X_other: pd.DataFrame) -> pd.DataFrame:
    """Reindex another matrix to the training columns (unseen dummies -> 0)."""
    return X_other.reindex(columns=X_train.columns, fill_value=0.0)


def time_split(df: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split on quote date: train on the past, test on the most recent period."""
    cutoff = df[cleaning.DATE_COL].quantile(1 - test_frac)
    train_df = df[df[cleaning.DATE_COL] <= cutoff]
    test_df = df[df[cleaning.DATE_COL] > cutoff]
    return train_df, test_df, cutoff


@dataclass
class TrainConfig:
    test_frac: float = 0.2
    seed: int = 7
    n_estimators: int = 600
    max_depth: int = 5
    learning_rate: float = 0.05
    min_child_weight: int = 30
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


def train(df_clean: pd.DataFrame, config: TrainConfig | None = None) -> tuple[TrainedModels, dict]:
    """Train baseline + XGBoost on a cleaned quote log.

    Returns the trained models and the split frames for evaluation:
    {"train": ..., "test": ...}.
    """
    config = config or TrainConfig()
    train_df, test_df, cutoff = time_split(df_clean, config.test_frac)

    X_train = to_matrix(train_df, train_df[cleaning.PRICE_COL].to_numpy())
    y_train = train_df[cleaning.LABEL_COL].to_numpy()

    baseline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=2000, random_state=config.seed)),
        ]
    )
    baseline.fit(X_train, y_train)

    # Hold out the final slice of *training* time for early stopping, so the
    # test period stays untouched.
    es_train, es_valid, _ = time_split(train_df, 0.15)
    X_es_train = align_columns(X_train, to_matrix(es_train, es_train[cleaning.PRICE_COL].to_numpy()))
    X_es_valid = align_columns(X_train, to_matrix(es_valid, es_valid[cleaning.PRICE_COL].to_numpy()))

    # Non-increasing in both price features, unconstrained elsewhere.
    monotone = tuple(-1 if c in PRICE_FEATURES else 0 for c in X_train.columns)
    xgb = XGBClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=config.min_child_weight,
        monotone_constraints=monotone,
        eval_metric="auc",
        early_stopping_rounds=config.early_stopping_rounds,
        random_state=config.seed,
        n_jobs=-1,
    )
    xgb.fit(
        X_es_train,
        es_train[cleaning.LABEL_COL].to_numpy(),
        eval_set=[(X_es_valid, es_valid[cleaning.LABEL_COL].to_numpy())],
        verbose=False,
    )

    models = TrainedModels(
        baseline=baseline,
        xgb=xgb,
        feature_columns=list(X_train.columns),
        cutoff_date=str(cutoff.date()),
        config=config,
    )
    return models, {"train": train_df.reset_index(drop=True), "test": test_df.reset_index(drop=True)}


def predict_accept(
    models: TrainedModels, df: pd.DataFrame, prices: np.ndarray, model: str = "xgb"
) -> np.ndarray:
    """P_hat(accept) for `df` at candidate `prices`, from the chosen model."""
    X = to_matrix(df, prices).reindex(columns=models.feature_columns, fill_value=0.0)
    est = models.xgb if model == "xgb" else models.baseline
    return est.predict_proba(X)[:, 1]


def save(models: TrainedModels, model_dir: str | Path) -> Path:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / "freight_pricing_models.joblib"
    joblib.dump(models, path)
    return path


def load(model_dir: str | Path) -> TrainedModels:
    return joblib.load(Path(model_dir) / "freight_pricing_models.joblib")
