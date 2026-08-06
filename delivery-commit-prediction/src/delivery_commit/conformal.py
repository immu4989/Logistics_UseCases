"""Distribution-free uncertainty for the daily flag list.

Two pieces, both plain numpy + scikit-learn:

1. **Isotonic probability calibration** (`calibrate_probabilities`) — maps raw
   XGBoost scores to calibrated probabilities using the most recent slice of
   training time.
2. **Conformal Risk Control for the false-negative rate** (`crc_threshold`) —
   turns the model's ranking into a contract: "flag a set of shipments that
   captures at least 1 - alpha of tomorrow's misses, in expectation, with a
   finite-sample guarantee" (Angelopoulos, Bates, Fisch, Lei, Schuster 2022,
   *Conformal Risk Control*).

Guarantee statement, stated once and precisely so nobody oversells it in a
meeting: the CRC threshold controls the **expected** false-negative rate at
level alpha, where the expectation is over the joint draw of calibration and
deployment shipments, **under exchangeability** between the calibration
positives and the deployment positives. It is not a per-day guarantee (any
single morning's realized FNR fluctuates around alpha), and it is void when
exchangeability breaks — a regime shift like the Olist truckers' strike (see
the README's Olist section), where the score distribution of late shipments
moves, is exactly such a violation. Recalibrate on recent data after any
regime change; the thresholds are cheap to refit and should be refit often.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

# Alphas the pipeline fits by default: 95% / 90% / 80% expected miss capture.
DEFAULT_ALPHAS = (0.05, 0.10, 0.20)


# ---------------------------------------------------------------------------
# 1. Isotonic probability calibration
# ---------------------------------------------------------------------------


@dataclass
class CalibratedScorer:
    """Picklable wrapper: raw model scores -> isotonic-calibrated probabilities.

    Exposes the same ``predict_proba(X) -> (n, 2)`` API as the underlying
    classifier so it can drop into any code path that consumes probabilities.
    Both members are plain sklearn/xgboost objects, so joblib round-trips the
    wrapper unchanged (and pickling shares the model object with
    ``TrainedModels.xgb`` rather than duplicating it).
    """

    model: object  # anything with predict_proba
    iso: IsotonicRegression

    def predict_proba(self, X) -> np.ndarray:
        raw = np.asarray(self.model.predict_proba(X))[:, 1]
        p = np.clip(self.iso.predict(raw), 0.0, 1.0)
        return np.column_stack([1.0 - p, p])


def calibrate_probabilities(model, X_cal, y_cal) -> CalibratedScorer:
    """Fit an isotonic map from raw scores to calibrated miss probabilities.

    Why isotonic, and why on a TIME slice rather than a random one: isotonic
    regression is monotone, so it fixes probability *levels* without touching
    the *ranking* — exactly the failure split we care about operationally.
    Calibration is the first casualty of drift: a shift in the base miss rate
    or in promise policy moves every probability while often leaving the
    ordering of shipments intact (the Olist regime shift in the README is the
    canonical example — the late rate collapsed from 21% to 1.4% and level-
    based outputs went stale long before the baseline's ranking did). So the
    calibration data must be the most *recent* data available, i.e. the last
    slice of training time, not a random sample that averages over stale
    months. The same logic says these maps should be refit on a frequent
    cadence in production.
    """
    raw = np.asarray(model.predict_proba(X_cal))[:, 1]
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(raw, np.asarray(y_cal, dtype=float))
    return CalibratedScorer(model=model, iso=iso)


# ---------------------------------------------------------------------------
# 2. Conformal Risk Control for the false-negative rate
# ---------------------------------------------------------------------------


@dataclass
class CRCThreshold:
    """A CRC-certified flag threshold: score >= threshold means 'flag it'."""

    alpha: float  # target expected FNR (1 - alpha = target miss capture)
    threshold: float  # flag rule: calibrated score >= threshold
    expected_flag_rate: float  # fraction of the calibration set flagged at threshold
    n_cal_positives: int  # calibration positives backing the guarantee
    allowed_cal_misses: int  # the CRC budget k derived below


def crc_threshold(scores_cal: np.ndarray, y_cal: np.ndarray, alpha: float) -> CRCThreshold:
    """Largest threshold whose CRC-adjusted calibration FNR is <= alpha.

    Flag rule: flag shipment i iff score_i >= t. The risk is the
    false-negative rate — the fraction of actual misses NOT flagged.

    Derivation of the finite-sample correction (Angelopoulos et al. 2022):

    1. Per-example loss on a *positive* (missed-commit) shipment i:
       L_i(t) = 1{score_i < t}. It is bounded by B = 1 and non-decreasing in
       t, so the empirical risk over the n calibration positives,
       FNR_hat(t) = (1/n) * sum_i L_i(t), is a monotone risk in CRC's sense.
    2. CRC's theorem: if calibration and deployment positives are
       exchangeable, choosing the largest t with
           (n / (n+1)) * FNR_hat(t) + B / (n+1) <= alpha
       guarantees E[FNR at t] <= alpha on deployment data. The B/(n+1) term
       is the price of the unseen (n+1)-th point: in the worst case it is a
       positive that the threshold misses, contributing loss B = 1 with
       weight 1/(n+1).
    3. Substitute FNR_hat(t) = k/n, where k = number of calibration positives
       with score < t:
           (n/(n+1)) * (k/n) + 1/(n+1) <= alpha
           (k + 1) / (n + 1)           <= alpha
           k                           <= (n + 1) * alpha - 1
       so the calibration budget is k_max = floor((n+1) * alpha - 1) missed
       calibration positives.
    4. The largest t missing at most k_max positives is the (k_max + 1)-th
       smallest positive score (that positive itself is still flagged, since
       the rule is >=; ties below it only make the choice conservative).

    If k_max < 0 (too few calibration positives: n < 1/alpha - 1), no finite
    threshold can honor the bound and the certified action is to flag
    everything (threshold = -inf); collect more calibration data.

    The guarantee is on the EXPECTED FNR under exchangeability, as spelled
    out in the module docstring: in expectation across days, not per-day, and
    void under regime shift.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    scores_cal = np.asarray(scores_cal, dtype=float)
    y_cal = np.asarray(y_cal).astype(int)
    pos_scores = np.sort(scores_cal[y_cal == 1])
    n = len(pos_scores)
    if n == 0:
        raise ValueError("no positive examples in the calibration slice")

    k_max = int(np.floor((n + 1) * alpha - 1 + 1e-9))
    threshold = -np.inf if k_max < 0 else float(pos_scores[k_max])
    return CRCThreshold(
        alpha=float(alpha),
        threshold=threshold,
        expected_flag_rate=float(np.mean(scores_cal >= threshold)),
        n_cal_positives=n,
        allowed_cal_misses=max(k_max, 0),
    )


def crc_report(
    thresholds: dict[float, CRCThreshold], scores_test: np.ndarray, y_test: np.ndarray
) -> pd.DataFrame:
    """Realized FNR / miss capture / flag rate of each CRC threshold on a test set.

    One period's realized capture is a single draw of a quantity guaranteed
    only in expectation; read it as a sanity check on the machinery, not as
    the guarantee itself.
    """
    scores_test = np.asarray(scores_test, dtype=float)
    y_test = np.asarray(y_test).astype(int)
    n_pos = int(y_test.sum())
    rows = []
    for alpha in sorted(thresholds):
        t = thresholds[alpha]
        flagged = scores_test >= t.threshold
        fnr = float(((y_test == 1) & ~flagged).sum() / n_pos) if n_pos else float("nan")
        rows.append(
            {
                "alpha": t.alpha,
                "threshold": t.threshold,
                "target_capture": 1.0 - t.alpha,
                "realized_capture": 1.0 - fnr,
                "realized_fnr": fnr,
                "flag_rate": float(flagged.mean()),
                "n_flagged": int(flagged.sum()),
                "n_test_misses": n_pos,
                "n_cal_positives": t.n_cal_positives,
            }
        )
    return pd.DataFrame(rows)
