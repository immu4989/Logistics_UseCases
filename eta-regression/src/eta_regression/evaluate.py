"""Evaluation chosen for how ETAs are actually consumed.

A point ETA feeds a tracking page; a quantile pair feeds a *promise*. The
metrics split the same way:

- **MAE / RMSE / median APE** — point accuracy for the baseline, the
  squared-error model, and the P50 quantile model. MAE is the headline: "our
  displayed ETA is off by X days on average" is a sentence an ops director
  can act on, RMSE is not.
- **Pinball loss per quantile** — the only proper score for a quantile model;
  a P90 that wins on MAE is broken by construction.
- **P10-P90 coverage and mean width** — coverage says the interval is honest
  (target ~80%), width says it is useful. Either alone is gameable: quote
  0-30 days and coverage is perfect.
- **Coverage by distance tercile** — the generator's noise grows with
  distance, so a model that only got the *average* right would over-cover
  short lanes and under-cover long ones. Segmented coverage is the check that
  the intervals adapt where it matters.
- **The promise table** — the product metric. If you commit ceil(P90) as the
  promised transit days, what fraction of shipments arrive by the promise,
  and how many promise-days does that cost versus quoting ceil(P50)? That is
  the trade the revenue and ops teams argue over; the promise curve plots it
  across every trained quantile.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_pinball_loss, mean_squared_error

from . import features, schema
from . import train as train_mod

TERCILE_LABELS = ["short", "medium", "long"]


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Point-forecast quality: MAE, RMSE, median absolute percentage error."""
    ape = np.abs(y_pred - y_true) / np.maximum(y_true, 0.25)
    return {
        "mae_days": float(mean_absolute_error(y_true, y_pred)),
        "rmse_days": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "median_ape": float(np.median(ape)),
    }


def interval_metrics(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict:
    """Honesty (coverage) and usefulness (width) of a prediction interval."""
    covered = (y_true >= lo) & (y_true <= hi)
    return {
        "coverage": float(covered.mean()),
        "mean_width_days": float(np.mean(hi - lo)),
    }


def coverage_by_distance(
    y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray, distance: np.ndarray
) -> pd.DataFrame:
    """P10-P90 coverage and width split by distance tercile (short/medium/long lanes)."""
    tercile = pd.qcut(distance, 3, labels=TERCILE_LABELS)
    df = pd.DataFrame(
        {
            "tercile": tercile,
            "covered": ((y_true >= lo) & (y_true <= hi)).astype(float),
            "width": hi - lo,
            "distance": distance,
        }
    )
    out = (
        df.groupby("tercile", observed=True)
        .agg(
            shipments=("covered", "size"),
            coverage=("covered", "mean"),
            mean_width_days=("width", "mean"),
            mean_distance_miles=("distance", "mean"),
        )
        .reset_index()
    )
    return out


def conformal_qhat(models, splits, nominal: float = 0.8, cal_frac: float = 0.15) -> float:
    """Split-conformal (CQR) width correction for the P10-P90 interval.

    Quantile models fitted by gradient boosting have no coverage guarantee —
    on data whose noise the trees never saw (a real network instead of the
    generator), the raw interval can run systematically narrow or wide.
    Conformalized quantile regression fixes the *level* without retraining:
    on a calibration slice, measure how far outside the interval the truth
    lands (score = max(p10 - y, y - p90)), take the ``nominal`` quantile of
    those scores with the finite-sample correction, and widen (or, if
    negative, shrink) both ends by that constant.

    The calibration slice is the last ``cal_frac`` of *training* time — the
    most recent data the test period never sees. It is the same slice early
    stopping validated on, which makes qhat slightly optimistic in exchange
    for not burning a third time window; on a regime shift the test month
    will drift from calibration anyway, which is exactly what the
    coverage-by-month breakout is there to show.
    """
    _, cal_df, _ = features.time_split(splits["train"], cal_frac)
    X_cal = features.to_matrix(cal_df).reindex(columns=models.feature_columns, fill_value=0.0)
    q = train_mod.predict_quantiles(models, X_cal, alphas=(0.1, 0.9))
    y = cal_df[schema.LABEL_COL].to_numpy()
    scores = np.maximum(q[0.1].to_numpy() - y, y - q[0.9].to_numpy())
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * nominal) / n)
    return float(np.quantile(scores, level, method="higher"))


def coverage_by_month(
    dates: pd.Series, y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray,
    lo_c: np.ndarray, hi_c: np.ndarray,
) -> pd.DataFrame:
    """Raw vs conformal P10-P90 coverage per test calendar month.

    Aggregate coverage over a multi-month test window can average a good
    month against a broken one; if operating conditions shift mid-window
    (strikes, carrier changes, promise-policy edits), this is the table that
    shows it.
    """
    df = pd.DataFrame(
        {
            "month": dates.dt.to_period("M").astype(str).to_numpy(),
            "covered_raw": ((y_true >= lo) & (y_true <= hi)).astype(float),
            "covered_conformal": ((y_true >= lo_c) & (y_true <= hi_c)).astype(float),
            "actual": y_true,
        }
    )
    return (
        df.groupby("month")
        .agg(
            shipments=("actual", "size"),
            coverage_raw=("covered_raw", "mean"),
            coverage_conformal=("covered_conformal", "mean"),
            mean_actual_transit_days=("actual", "mean"),
        )
        .reset_index()
        .sort_values("month")
        .reset_index(drop=True)
    )


def promise_table(y_true: np.ndarray, qpreds: pd.DataFrame) -> pd.DataFrame:
    """The ops tradeoff: quote ceil(quantile) as the committed transit days.

    For each trained quantile: what fraction of shipments arrive by the
    promise, and what does the promise cost in quoted days?
    """
    rows = []
    for alpha in qpreds.columns:
        promised = np.ceil(qpreds[alpha].to_numpy())
        rows.append(
            {
                "quoted_quantile": float(alpha),
                "promised_by_rate": float((y_true <= promised).mean()),
                "mean_promised_days": float(promised.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("quoted_quantile").reset_index(drop=True)


def evaluate_models(models, splits, out_dir: str | Path) -> dict:
    """Score all models on the held-out period; write metrics.json + plots."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_test, y_test = splits["X_test"], splits["y_test"]
    distance = splits["test"]["distance_miles"].to_numpy()

    qpreds = train_mod.predict_quantiles(models, X_test)
    p10 = qpreds[0.1].to_numpy()
    p50 = qpreds[0.5].to_numpy()
    p90 = qpreds[0.9].to_numpy()

    results: dict = {
        "linear_baseline": point_metrics(y_test, models.baseline.predict(X_test)),
        "xgboost_point": point_metrics(y_test, models.xgb_point.predict(X_test)),
        "xgboost_p50": point_metrics(y_test, p50),
        "pinball_loss": {
            f"q{int(a * 100)}": float(mean_pinball_loss(y_test, qpreds[a], alpha=a))
            for a in train_mod.PRODUCT_QUANTILES
        },
        "interval_p10_p90": {"nominal_coverage": 0.8, **interval_metrics(y_test, p10, p90)},
    }

    # Conformal correction: fixes the interval *level* on networks where the
    # raw quantile models run miscalibrated (see conformal_qhat docstring).
    qhat = conformal_qhat(models, splits)
    p10_c = np.maximum(p10 - qhat, 0.05)
    p90_c = p90 + qhat
    results["interval_p10_p90_conformal"] = {
        "nominal_coverage": 0.8,
        "qhat_days": qhat,
        **interval_metrics(y_test, p10_c, p90_c),
    }

    cov = coverage_by_distance(y_test, p10, p90, distance)
    results["coverage_by_distance_tercile"] = cov.to_dict(orient="records")

    monthly = coverage_by_month(
        splits["test"][schema.DATE_COL], y_test, p10, p90, p10_c, p90_c
    )
    results["coverage_by_month"] = monthly.to_dict(orient="records")

    promise = promise_table(y_test, qpreds)
    results["promise_table"] = promise.to_dict(orient="records")
    p50_row = promise[promise["quoted_quantile"] == 0.5].iloc[0]
    p90_row = promise[promise["quoted_quantile"] == 0.9].iloc[0]
    results["promise_tradeoff"] = {
        "quote_ceil_p50": {
            "promised_by_rate": float(p50_row["promised_by_rate"]),
            "mean_promised_days": float(p50_row["mean_promised_days"]),
        },
        "quote_ceil_p90": {
            "promised_by_rate": float(p90_row["promised_by_rate"]),
            "mean_promised_days": float(p90_row["mean_promised_days"]),
        },
        "extra_days_for_p90_promise": float(
            p90_row["mean_promised_days"] - p50_row["mean_promised_days"]
        ),
    }
    results["test_period_start"] = models.cutoff_date
    results["n_test"] = int(len(y_test))
    results["mean_actual_transit_days"] = float(np.mean(y_test))

    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    cov.to_csv(out_dir / "coverage_by_distance.csv", index=False)
    monthly.to_csv(out_dir / "coverage_by_month.csv", index=False)
    promise.to_csv(out_dir / "promise_table.csv", index=False)

    _plot_pred_vs_actual(y_test, p50, out_dir / "pred_vs_actual.png")
    _plot_coverage_by_tercile(cov, out_dir / "coverage_by_distance.png")
    if monthly["month"].nunique() >= 2:
        _plot_coverage_by_month(monthly, out_dir / "coverage_by_month.png")
    _plot_promise_curve(promise, out_dir / "promise_curve.png")
    return results


def _plot_pred_vs_actual(y_test, p50, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    lim = float(np.quantile(y_test, 0.999)) + 1
    hb = ax.hexbin(p50, y_test, gridsize=45, cmap="Blues", mincnt=1, extent=(0, lim, 0, lim))
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="perfect ETA")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Predicted P50 transit days")
    ax.set_ylabel("Actual transit days")
    ax.set_title("Median ETA vs actual (held-out period)")
    fig.colorbar(hb, ax=ax, label="shipments")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_coverage_by_tercile(cov: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(cov))
    ax.bar(x, cov["coverage"], color="#2b6cb0", width=0.6)
    ax.axhline(0.8, color="k", ls="--", lw=1, label="nominal 80%")
    for i, row in cov.iterrows():
        ax.text(i, row["coverage"] + 0.01, f"{row['coverage']:.1%}", ha="center", fontsize=9)
        ax.text(
            i,
            0.04,
            f"width {row['mean_width_days']:.1f} d\n~{row['mean_distance_miles']:,.0f} mi",
            ha="center",
            fontsize=8,
            color="white",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t} lanes" for t in cov["tercile"]])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("P10-P90 empirical coverage")
    ax.set_title("Interval honesty by lane length (held-out period)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_coverage_by_month(monthly: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(monthly))
    ax.bar(x, monthly["coverage_raw"], width=0.62, color="#2b6cb0", label="raw P10–P90")
    ax.plot(
        x,
        monthly["coverage_conformal"],
        marker="o",
        color="#c05621",
        lw=2,
        label="conformal P10–P90",
    )
    ax.axhline(0.8, color="k", ls="--", lw=1, label="nominal 80%")
    for i, row in monthly.iterrows():
        ax.text(i, row["coverage_raw"] - 0.05, f"{row['coverage_raw']:.0%}",
                ha="center", fontsize=8, color="white", fontweight="bold")
        ax.text(i + 0.08, row["coverage_conformal"] + 0.025,
                f"{row['coverage_conformal']:.0%}", ha="left", fontsize=8, color="#c05621")
        ax.text(i, 0.04, f"{row['shipments']:,}\nshipments", ha="center",
                fontsize=7, color="white")
    ax.set_xticks(x)
    ax.set_xticklabels(monthly["month"], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("P10–P90 empirical coverage")
    ax.set_title("Interval coverage by test month: raw vs conformal")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_promise_curve(promise: pd.DataFrame, path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(
        promise["quoted_quantile"],
        promise["promised_by_rate"],
        marker="o",
        color="#2b6cb0",
        label="promised-by rate",
    )
    ax1.axhline(0.9, color="k", ls="--", lw=1)
    ax1.text(promise["quoted_quantile"].min(), 0.905, "90% kept-promise target", fontsize=8)
    ax1.set_xlabel("Quoted quantile (promise = ceil of its prediction)")
    ax1.set_ylabel("Shipments arriving by the promise", color="#2b6cb0")
    ax1.set_ylim(0.5, 1.02)

    ax2 = ax1.twinx()
    ax2.plot(
        promise["quoted_quantile"],
        promise["mean_promised_days"],
        marker="s",
        color="#c05621",
        label="mean promised days",
    )
    ax2.set_ylabel("Mean promised transit days", color="#c05621")

    ax1.set_title("The promise curve: reliability vs quoted days")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
