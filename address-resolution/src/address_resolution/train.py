"""Train the pair scorer and pick the operating threshold.

Three decisions worth reading before you swap in your own data:

**Split by label, share the delivery points.** Labels are hashed on their id
into train/test, and both sides match against the same delivery-point
database. That is not leakage — it is the production setup. The database is a
fixture: tomorrow's labels resolve against the same canonical rows today's
did, so a point appearing on both sides of the split is exactly what deploy
looks like. What WOULD leak is the same *label* on both sides, because the
model would have seen that exact corruption instance; the id hash makes that
impossible, and it is stable across runs (no dependence on row order).

**Logistic regression, on purpose.** A GBM would squeeze out another half
point of AUC and cost the thing this product actually ships: every match
probability decomposes into per-feature coefficient contributions that a
review-queue screen can display next to the label ("street name agrees +3.1,
unit conflicts -2.9"). The reviewer either trusts the rationale or fixes the
feature; nobody arbitrates a SHAP plot at 6am in a sort facility. The features
were engineered monotone (agreements up, conflicts down) precisely so a linear
scorer is not leaving much behind. See explain.py for the cards.

**The threshold is chosen for a precision target, not accuracy.** Positives
are ~1-of-150 blocked candidates and we do not reweight; raw probabilities
skew low and that is fine, because the decision rule only needs a monotone
score plus a threshold. The default threshold is the lowest one that holds
auto-match precision at ``target_precision`` on the train side — coverage is
then whatever precision leaves on the table, which is the honest direction of
that trade.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import resolve, synthetic


@dataclass
class ResolverConfig:
    seed: int = 7
    train_frac: float = 0.7
    neg_per_label: int = 8
    target_precision: float = 0.995
    C: float = 1.0


@dataclass
class TrainedResolver:
    pipeline: Pipeline
    feature_names: list[str]
    threshold: float
    target_precision: float
    config: ResolverConfig


@dataclass
class MatchRun:
    """Everything downstream stages need, computed once."""

    points: pd.DataFrame
    labels: pd.DataFrame
    lnorm: pd.DataFrame
    pnorm: pd.DataFrame
    li: np.ndarray
    pi: np.ndarray
    X: np.ndarray
    y: np.ndarray
    probs: np.ndarray
    is_train: np.ndarray
    resolver: TrainedResolver
    decisions: pd.DataFrame
    blocking_recall: float


def hash_split(label_ids: pd.Series, train_frac: float = 0.7) -> np.ndarray:
    """Stable train membership from a hash of the label id."""
    buckets = np.array(
        [int(hashlib.md5(s.encode()).hexdigest()[:8], 16) % 1000 for s in label_ids]
    )
    return buckets < int(train_frac * 1000)


def pair_labels(labels: pd.DataFrame, points: pd.DataFrame, li, pi) -> np.ndarray:
    """y for every blocked pair: is this candidate the true delivery point?"""
    idx_of = {pid: i for i, pid in enumerate(points["point_id"])}
    true_pi = np.array(
        [idx_of.get(t, -1) for t in labels["true_point_id"].fillna("")], dtype=np.int64
    )
    return (pi == true_pi[li]).astype(np.float32)


def blocking_recall(labels: pd.DataFrame, li, y) -> float:
    """Share of matchable labels whose true point survived blocking. This is
    the ceiling on coverage; everything blocking drops is lost forever."""
    matchable = (labels["true_point_id"].fillna("") != "").to_numpy()
    found = np.zeros(len(labels), dtype=bool)
    found[li[y > 0]] = True
    return float(found[matchable].mean())


def sample_training_pairs(
    li: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    is_train_label: np.ndarray,
    neg_per_label: int,
    seed: int,
) -> np.ndarray:
    """All positives, every HARD negative, up to ``neg_per_label`` random ones.

    A hard negative is a candidate that agrees on the street number or touches
    unit logic — same building different unit, unit-less label against a
    multi-unit building, an imposter street with the same number. Random
    subsampling would dilute exactly these, and the coefficients that decide
    wrong-door-vs-review would come out mushy (an early version of this
    pipeline auto-delivered unit-less apartment labels for that reason). Easy
    negatives are capped per label purely to keep the training set small.
    """
    rng = np.random.default_rng(seed)
    in_train = is_train_label[li]
    pos = np.where(in_train & (y > 0))[0]
    num_exact = resolve.FEATURES.index("number_exact")
    num_trans = resolve.FEATURES.index("number_transposed")
    conflict = resolve.FEATURES.index("unit_conflict")
    unres = resolve.FEATURES.index("unit_unresolvable")
    hard_mask = (X[:, [num_exact, num_trans, conflict, unres]] > 0.5).any(axis=1)
    hard = np.where(in_train & (y == 0) & hard_mask)[0]
    easy = np.where(in_train & (y == 0) & ~hard_mask)[0]
    keys = rng.random(len(easy))
    order = np.lexsort((keys, li[easy]))
    sorted_li = li[easy][order]
    group_start = np.searchsorted(sorted_li, sorted_li, side="left")
    rank = np.arange(len(sorted_li)) - group_start
    keep_easy = easy[order[rank < neg_per_label]]
    return np.sort(np.concatenate([pos, hard, keep_easy]))


def fit_scorer(X: np.ndarray, y: np.ndarray, config: ResolverConfig) -> Pipeline:
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("logreg", LogisticRegression(C=config.C, max_iter=1000, random_state=config.seed)),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


def pick_threshold(p_best: np.ndarray, hit_best: np.ndarray, target: float) -> float:
    """Lowest threshold whose auto-match precision holds ``target``.

    Evaluated only at tie-group boundaries: identical feature rows produce
    identical probabilities (every unit of an apartment building, say), and a
    threshold admits such a cluster whole or not at all. Cutting mid-cluster
    during the sweep would report a precision the deployed rule can't achieve.
    Chosen on the train side only.
    """
    has_cand = p_best >= 0
    p = p_best[has_cand]
    h = hit_best[has_cand]
    order = np.argsort(-p)
    p_s, h_s = p[order], h[order]
    k = np.arange(1, len(p_s) + 1)
    boundary = np.r_[p_s[1:] != p_s[:-1], True]
    prec = np.cumsum(h_s)[boundary] / k[boundary]
    pb = p_s[boundary]
    feasible = np.where(prec >= target)[0]
    if len(feasible) == 0:
        return 0.99
    return float(pb[feasible[-1]])


def save(resolver: TrainedResolver, model_dir: str | Path) -> Path:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / "resolver.joblib"
    joblib.dump(resolver, path)
    return path


def load(model_dir: str | Path) -> TrainedResolver:
    return joblib.load(Path(model_dir) / "resolver.joblib")


def run(
    points: pd.DataFrame, labels: pd.DataFrame, config: ResolverConfig | None = None
) -> MatchRun:
    """The full matching pipeline on prepared tables. Used by the CLI and tests."""
    config = config or ResolverConfig()

    lnorm = resolve.normalize_labels(labels)
    pnorm = resolve.normalize_points(points)
    blocker = resolve.Blocker(pnorm)
    li, pi = resolve.build_pairs(lnorm, blocker)
    X = resolve.featurize(lnorm, pnorm, li, pi)
    y = pair_labels(labels, points, li, pi)
    recall = blocking_recall(labels, li, y)

    is_train = hash_split(labels["label_id"], config.train_frac)
    train_idx = sample_training_pairs(li, X, y, is_train, config.neg_per_label, config.seed)
    pipeline = fit_scorer(X[train_idx], y[train_idx], config)
    probs = pipeline.predict_proba(X)[:, 1]

    prelim = resolve.decide(labels, points, li, pi, probs, threshold=2.0)  # nothing accepted
    train_rows = is_train
    threshold = pick_threshold(
        prelim.loc[train_rows, "p_best"].to_numpy(),
        prelim.loc[train_rows, "hit_best"].to_numpy(),
        config.target_precision,
    )
    resolver = TrainedResolver(
        pipeline=pipeline,
        feature_names=list(resolve.FEATURES),
        threshold=threshold,
        target_precision=config.target_precision,
        config=config,
    )
    decisions = resolve.decide(labels, points, li, pi, probs, threshold)
    decisions["is_train"] = is_train

    return MatchRun(
        points=points,
        labels=labels,
        lnorm=lnorm,
        pnorm=pnorm,
        li=li,
        pi=pi,
        X=X,
        y=y,
        probs=probs,
        is_train=is_train,
        resolver=resolver,
        decisions=decisions,
        blocking_recall=recall,
    )


__all__ = [
    "ResolverConfig",
    "TrainedResolver",
    "MatchRun",
    "hash_split",
    "pair_labels",
    "blocking_recall",
    "sample_training_pairs",
    "fit_scorer",
    "pick_threshold",
    "save",
    "load",
    "run",
    "synthetic",
]
