"""Evaluation, chosen for how a staffing forecast is actually consumed.

- **WAPE** (weighted absolute percentage error, sum|err| / sum actual) is the
  headline. It is the number a capacity planner recognises: "we mis-planned 7%
  of the parcels". Plain MAPE is unusable on this data — Sundays run at ~6% of
  a Monday, so a 40-parcel miss on a 60-parcel Sunday explodes into a 67%
  "error" and the metric ends up ranking models by their Sunday luck.
- **sMAPE** is reported for comparability with forecasting literature, never
  as the decision metric.
- **Bias** (signed error / sum actual) gets its own column because the two
  directions cost differently: chronic under-forecast means missed sorts and
  emergency overtime every week, while noise averages out. A model with 6%
  WAPE and -4% bias is worse for staffing than one with 7% WAPE and 0% bias.
- Everything is broken out **overall / peak season / hub-size tercile**. The
  December ramp is the season the forecast exists for, and an aggregate WAPE
  lets ten easy months bury the six weeks that matter.
- **The product metric**: staff each hub-day to a capacity and count the days
  the wave overtops it. Staffing to the P80 quantile versus the point forecast
  is the whole argument for shipping quantile models, in one table.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import schema, synthetic, train as train_mod


def wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.abs(forecast - actual).sum() / actual.sum())


def smape(actual: np.ndarray, forecast: np.ndarray) -> float:
    denom = (np.abs(actual) + np.abs(forecast)).clip(min=1e-9)
    return float(np.mean(2.0 * np.abs(forecast - actual) / denom))


def bias(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Signed. Negative = chronic under-forecast, the expensive direction."""
    return float((forecast - actual).sum() / actual.sum())


def pinball(actual: np.ndarray, forecast: np.ndarray, alpha: float) -> float:
    diff = actual - forecast
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


def _metric_row(actual: np.ndarray, forecast: np.ndarray) -> dict:
    return {
        "wape": wape(actual, forecast),
        "smape": smape(actual, forecast),
        "bias": bias(actual, forecast),
    }


def hub_size_terciles(train_df: pd.DataFrame) -> dict[str, str]:
    """large / mid / small, from *training* mean volume only (no test peeking)."""
    means = train_df.groupby(schema.HUB_COL)[schema.TARGET_COL].mean().sort_values()
    labels = {}
    n = len(means)
    for i, hub in enumerate(means.index):
        labels[hub] = "small" if i < n / 3 else ("mid" if i < 2 * n / 3 else "large")
    return labels


def staffing_table(actual: np.ndarray, capacities: dict[str, np.ndarray]) -> pd.DataFrame:
    """Understaffed-days % and idle-capacity cost per staffing policy.

    A day is understaffed when actual volume exceeds the capacity the policy
    staffed for. Idle capacity is the flip side (planned minus actual, on days
    with slack) so the table shows what the P80 cushion costs, not just what
    it buys.
    """
    rows = []
    for policy, cap in capacities.items():
        rows.append(
            {
                "policy": policy,
                "understaffed_days_pct": float(np.mean(actual > cap)),
                "parcels_uncovered_pct": float(np.maximum(actual - cap, 0).sum() / actual.sum()),
                "idle_capacity_pct": float(np.maximum(cap - actual, 0).sum() / actual.sum()),
            }
        )
    return pd.DataFrame(rows)


def evaluate_models(models, splits, out_dir: str | Path) -> dict:
    """Score all models on the held-out window; write metrics + plots."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_df = splits["test"].reset_index(drop=True)
    X_test = splits["X_test"].reset_index(drop=True)
    actual = test_df[schema.TARGET_COL].to_numpy()

    forecasts = {
        "seasonal_naive": models.naive.predict(X_test),
        "xgboost_point": train_mod.predict_point(models, X_test),
    }
    q = train_mod.predict_quantiles(models, X_test)
    forecasts["xgboost_p50"] = q["p50"].to_numpy()

    # --- WAPE / sMAPE / bias, overall + peak + hub-size terciles ------------
    peak_mask = test_df["is_peak_ramp"].to_numpy().astype(bool)
    terciles = hub_size_terciles(splits["train"])
    tercile_col = test_df[schema.HUB_COL].map(terciles).to_numpy()

    accuracy: dict[str, dict] = {}
    for name, f in forecasts.items():
        accuracy[name] = {
            "overall": _metric_row(actual, f),
            "peak_season": _metric_row(actual[peak_mask], f[peak_mask]),
        }
        for t in ["large", "mid", "small"]:
            m = tercile_col == t
            accuracy[name][f"hubs_{t}"] = _metric_row(actual[m], f[m])

    # --- quantile quality ----------------------------------------------------
    covered = (actual >= q["p10"].to_numpy()) & (actual <= q["p90"].to_numpy())
    quantile_metrics = {
        "pinball": {c: pinball(actual, q[c].to_numpy(), int(c[1:]) / 100) for c in q.columns},
        "coverage_p10_p90": float(covered.mean()),
        "coverage_p10_p90_peak": float(covered[peak_mask].mean()),
        "mean_interval_width_pct": float((q["p90"] - q["p10"]).sum() / actual.sum()),
    }

    # --- the product metric: understaffed days --------------------------------
    staffing = staffing_table(
        actual,
        {
            "seasonal_naive": forecasts["seasonal_naive"],
            "point_forecast": forecasts["xgboost_point"],
            "staff_to_p80": q["p80"].to_numpy(),
        },
    )
    staffing.to_csv(out_dir / "staffing_table.csv", index=False)

    # --- per-hub WAPE ---------------------------------------------------------
    per_hub = []
    for hub in sorted(test_df[schema.HUB_COL].unique(), key=lambda h: -synthetic.HUB_BASE[h]):
        m = (test_df[schema.HUB_COL] == hub).to_numpy()
        per_hub.append(
            {
                "hub": hub,
                "tercile": terciles[hub],
                "wape_naive": wape(actual[m], forecasts["seasonal_naive"][m]),
                "wape_xgboost": wape(actual[m], forecasts["xgboost_point"][m]),
            }
        )
    per_hub = pd.DataFrame(per_hub)
    per_hub.to_csv(out_dir / "wape_by_hub.csv", index=False)

    results = {
        "accuracy": accuracy,
        "quantiles": quantile_metrics,
        "staffing": staffing.to_dict(orient="records"),
        "test_window_start": models.cutoff_date,
        "n_test_hub_days": int(len(test_df)),
    }
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))

    _plot_overlay(test_df, q, forecasts, out_dir / "december_overlay.png")
    _plot_wape_by_hub(per_hub, out_dir / "wape_by_hub.png")
    _plot_coverage(test_df, q, actual, out_dir / "coverage_by_month.png")
    return results


def _plot_overlay(test_df, q, forecasts, path: Path) -> None:
    """The money chart: the biggest hub through the December peak."""
    hub = max(synthetic.HUB_BASE, key=synthetic.HUB_BASE.get)
    m = (test_df[schema.HUB_COL] == hub).to_numpy()
    sub = test_df[m]
    dates = sub[schema.DATE_COL]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.fill_between(
        dates, q.loc[m, "p10"], q.loc[m, "p90"],
        alpha=0.25, color="#2b6cb0", label="P10-P90 range", lw=0,
    )
    ax.plot(dates, sub[schema.TARGET_COL], color="black", lw=1.2, label="actual")
    ax.plot(dates, q.loc[m, "p50"], color="#2b6cb0", lw=1.2, label="forecast (P50)")
    ax.plot(
        dates, forecasts["seasonal_naive"][m],
        color="#c53030", lw=0.9, ls="--", alpha=0.7, label="seasonal naive",
    )
    ax.set_title(f"{hub}: forecast vs actual through the December peak (held-out window)")
    ax.set_ylabel("inbound parcels / day")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_wape_by_hub(per_hub: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(per_hub))
    ax.bar(x - 0.2, per_hub["wape_naive"] * 100, width=0.4, color="#c53030", label="seasonal naive")
    ax.bar(x + 0.2, per_hub["wape_xgboost"] * 100, width=0.4, color="#2b6cb0", label="XGBoost")
    ax.set_xticks(x)
    ax.set_xticklabels(per_hub["hub"], rotation=45, ha="right")
    ax.set_ylabel("WAPE, %  (held-out window)")
    ax.set_title("Forecast error by hub, largest to smallest")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_coverage(test_df, q, actual, path: Path) -> None:
    covered = (actual >= q["p10"].to_numpy()) & (actual <= q["p90"].to_numpy())
    month = test_df[schema.DATE_COL].dt.to_period("M").astype(str)
    cov = pd.Series(covered).groupby(month.to_numpy()).mean()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(cov.index, cov.values * 100, color="#2b6cb0", width=0.6)
    ax.axhline(80, color="black", ls="--", lw=1, label="nominal 80%")
    ax.set_ylim(0, 100)
    ax.set_ylabel("P10-P90 coverage, %")
    ax.set_title("Interval coverage by test month")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
