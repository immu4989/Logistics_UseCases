"""Uplift estimators, from the ops status quo to the doubly-robust learner.

Every method produces one score per shipment, and every score means "treat the
highest-scored shipments first". The sign convention throughout matches the
generator: CATE = p0 - p1, positive = misses prevented per treated shipment.

The ladder:

- ``risk_targeting`` — the rule every ops desk runs today: model P(miss) on
  CONTROL shipments and treat the riskiest. Not an uplift method at all; it
  ranks by how likely a shipment is to fail, not by whether the intervention
  changes that. It is the baseline the causal methods must beat.
- ``s_learner``  — one classifier with the treatment flag as a feature;
  CATE_hat = f(x, t=0) - f(x, t=1). Simple, but the tree ensemble is free to
  ignore a weak treatment feature, which shrinks effects toward zero.
- ``t_learner``  — separate treated/control models; CATE_hat = mu0(x) - mu1(x).
  No shared regularization, so with a 25% treated arm the treated model is
  noisier than the control model and their difference inherits both errors.
- ``dr_learner`` — the star. AIPW pseudo-outcomes built from cross-fitted
  nuisance models, then a regression of the pseudo-outcomes on features.
  Debiased against nuisance error and the standard tool for this job.

Propensity note: this is a RANDOMIZED pilot, so the propensity is the KNOWN
constant 0.25 and is used as such — no propensity model, no overlap worries.
On observational data you would have to estimate e(x), check overlap (no
stratum with e(x) near 0 or 1), and the doubly-robust construction is what
keeps you honest when either nuisance model is wrong. That machinery is the
price of skipping the experiment; the README says more.

Split: 70/30 by a deterministic hash of shipment id. A randomized pilot has
exchangeable rows (no time leakage to defend against, unlike the repo's
observational use cases), and a hash split keeps membership stable if the log
is re-extracted or grows.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from . import cleaning, synthetic

# Modest fixed parameters, deliberately untuned: the comparison is between
# estimation STRATEGIES, and tuning one arm would muddy that.
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.08,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "n_jobs": 4,
}

# The DR final stage regresses PSEUDO-outcomes whose noise is amplified by the
# inverse-propensity weights (1/e = 4 on the treated arm), so it needs far
# heavier regularization than a model fitted to real labels. Fitting it with
# the classifier params above costs ~0.2 of normalized AUUC — the single
# biggest implementation pitfall in this whole use case.
DR_FINAL_PARAMS = {
    "n_estimators": 600,
    "max_depth": 3,
    "learning_rate": 0.02,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 20.0,
    "min_child_weight": 100,
    "n_jobs": 4,
}
DR_FOLDS = 4

ONE_HOT_COLS = ["service_level", "customer_tier"]
PASSTHROUGH = ["is_peak", "is_rural"]

METHODS = ["risk_targeting", "s_learner", "t_learner", "dr_learner"]


def to_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Model matrix via WHITELIST: numeric + one-hot categoricals, nothing else.

    The ground-truth columns (``true_cate``, ``p0_true``, ``segment_true``),
    the label and the treatment flag can never leak in, because they are not
    on the list — the same defensive pattern as ``p_miss_true`` in
    delivery-commit-prediction, and tests assert it.
    """
    numeric = [c for c in cleaning.NUMERIC_FEATURES if c in df.columns]
    numeric += [c for c in df.columns if c.endswith("__was_missing")]
    passthrough = [c for c in PASSTHROUGH if c in df.columns]
    X = df[numeric + passthrough].astype(float)
    dummies = pd.get_dummies(df[ONE_HOT_COLS], prefix=ONE_HOT_COLS, dtype=float)
    return pd.concat([X, dummies], axis=1)


def hash_split(df: pd.DataFrame, train_frac: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic 70/30 split on a CRC32 hash of the shipment id."""
    buckets = df[synthetic.ID_COL].map(lambda s: zlib.crc32(s.encode()) % 100)
    train = df[buckets < int(train_frac * 100)].reset_index(drop=True)
    test = df[buckets >= int(train_frac * 100)].reset_index(drop=True)
    return train, test


@dataclass
class Bundle:
    """All fitted estimators plus the training feature columns."""

    risk: XGBClassifier
    s: XGBClassifier
    t0: XGBClassifier
    t1: XGBClassifier
    dr: XGBRegressor
    feature_columns: list[str]


def _clf(seed: int) -> XGBClassifier:
    return XGBClassifier(random_state=seed, eval_metric="logloss", **XGB_PARAMS)


def fit_all(train_df: pd.DataFrame, seed: int = 7) -> Bundle:
    """Fit every estimator on the training split."""
    X = to_matrix(train_df)
    y = train_df[synthetic.LABEL_COL].to_numpy()
    t = train_df[synthetic.TREATMENT_COL].to_numpy()

    # --- risk targeting: P(miss | x) on CONTROL rows only --------------------
    # (Control-only, so the pilot's own treatment effect does not contaminate
    # the risk score — this is the strongest fair version of the status quo.)
    risk = _clf(seed).fit(X[t == 0], y[t == 0])

    # --- S-learner ------------------------------------------------------------
    Xs = X.copy()
    Xs["treated"] = t.astype(float)
    s = _clf(seed).fit(Xs, y)

    # --- T-learner --------------------------------------------------------------
    t0 = _clf(seed).fit(X[t == 0], y[t == 0])
    t1 = _clf(seed + 1).fit(X[t == 1], y[t == 1])

    # --- DR-learner (AIPW pseudo-outcomes, cross-fitted nuisances) -------------
    # e = 0.25 is the KNOWN randomization probability. Observational data would
    # need e(x) estimated and overlap-checked; here plugging in the design
    # constant is both simpler and exactly right.
    e = synthetic.PROPENSITY
    psi = np.zeros(len(train_df))
    fold = np.arange(len(train_df)) % DR_FOLDS  # deterministic cross-fitting
    for f in range(DR_FOLDS):
        fit_idx, out_idx = fold != f, fold == f
        mu0 = _clf(seed + 10 + f).fit(X[fit_idx & (t == 0)], y[fit_idx & (t == 0)])
        mu1 = _clf(seed + 20 + f).fit(X[fit_idx & (t == 1)], y[fit_idx & (t == 1)])
        m0 = mu0.predict_proba(X[out_idx])[:, 1]
        m1 = mu1.predict_proba(X[out_idx])[:, 1]
        yo, to = y[out_idx], t[out_idx]
        # AIPW pseudo-outcome for tau = E[Y0 - Y1 | x] (positive = helps):
        psi[out_idx] = (m0 - m1) + (1 - to) / (1 - e) * (yo - m0) - to / e * (yo - m1)
    dr = XGBRegressor(random_state=seed, **DR_FINAL_PARAMS).fit(X, psi)

    return Bundle(risk=risk, s=s, t0=t0, t1=t1, dr=dr, feature_columns=list(X.columns))


def predict_scores(bundle: Bundle, df: pd.DataFrame) -> pd.DataFrame:
    """One targeting score per method per shipment (higher = treat first).

    For the three uplift methods the score is CATE_hat = p0_hat - p1_hat.
    For risk_targeting it is P(miss): a probability, not an effect — the two
    live on different scales on purpose, because ranking is all a top-k
    policy consumes.
    """
    X = to_matrix(df).reindex(columns=bundle.feature_columns, fill_value=0.0)

    risk = bundle.risk.predict_proba(X)[:, 1]

    X0, X1 = X.copy(), X.copy()
    X0["treated"], X1["treated"] = 0.0, 1.0
    cols = list(bundle.feature_columns) + ["treated"]
    s_cate = (
        bundle.s.predict_proba(X0[cols])[:, 1] - bundle.s.predict_proba(X1[cols])[:, 1]
    )

    t_cate = bundle.t0.predict_proba(X)[:, 1] - bundle.t1.predict_proba(X)[:, 1]
    dr_cate = bundle.dr.predict(X)

    return pd.DataFrame(
        {
            "risk_targeting": risk,
            "s_learner": s_cate,
            "t_learner": t_cate,
            "dr_learner": dr_cate,
        },
        index=df.index,
    )
