"""Zero-shot foundation-model benchmark: Chronos-Bolt vs the tuned pipeline.

The 2025-2026 forecasting argument, run on this repo's own held-out window
instead of settled by blog post. Amazon's Chronos-Bolt is a pretrained
time-series foundation model: hand it raw history, get calibrated quantiles
back, no feature engineering, no training. The question a network planner
actually has is whether that beats the tuned XGBoost — and this module answers
it on the identical final ~4 months (December peak included) that
``evaluate.py`` scores everything else on.

The protocol is the honest one — **rolling-origin, day-ahead**:

- For every test day ``t`` and every hub, the context is the *actual* cleaned
  history up to ``t-1`` (capped at the model's 2048-day limit), and the model
  predicts exactly one step. No forecast is ever fed back as history, so every
  prediction is the same day-ahead task the XGBoost row is scored on. Feed
  gaps stay NaN in the context; Chronos-Bolt masks missing values natively,
  the same "gaps stay gaps" contract as the rest of the pipeline.
- All 15 hubs are batched into one model call per day, so the full window is
  ~120 calls per model and runs in minutes on CPU.

The fairness asymmetry, stated both ways because it *is* the tradeoff:

- Chronos is **univariate and zero-shot**. It never sees the promo calendar,
  the holiday table, or a peak-ramp flag — the covariates XGBoost gets as
  features. A promo spike is unforecastable from history alone.
- XGBoost needed the whole apparatus Chronos skips: a feature pipeline, a
  documented holiday/promo calendar, a log-ratio target, tuning, conformal
  calibration. Chronos got a matrix of raw counts.

Metrics are computed by the same functions ``evaluate.py`` uses (WAPE, sMAPE,
bias, pinball, P10-P90 coverage; overall and peak-only), so the tables line up
number for number. Model invocation and metric computation are deliberately
separate functions: the metric path has no torch dependency and is tested even
where the ``fm`` extra is not installed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import cleaning, evaluate, schema, synthetic, train as train_mod

CHRONOS_MODELS = {
    "tiny": "amazon/chronos-bolt-tiny",
    "small": "amazon/chronos-bolt-small",
}
QUANTILE_LEVELS = (0.1, 0.5, 0.8, 0.9)
CONTEXT_CAP = 2048  # chronos-bolt context limit; our ~900-day history fits under it

INSTALL_HINT = (
    "the foundation-model benchmark needs the optional 'fm' extra "
    "(torch is a ~200MB wheel, which is why it is not a default dependency):\n"
    '    pip install -e ".[fm]"'
)


# ---------------------------------------------------------------------------
# Model invocation (needs torch + chronos; imported lazily)
# ---------------------------------------------------------------------------
def load_pipeline(model_id: str):
    """Load a Chronos-Bolt pipeline on CPU, with a clear hint if fm isn't installed."""
    try:
        import torch
        from chronos import BaseChronosPipeline
    except ImportError as e:
        raise ImportError(INSTALL_HINT) from e
    # torch and xgboost each bundle their own OpenMP runtime; on macOS the two
    # can deadlock at an OMP barrier when both run parallel regions in one
    # process (fm-bench trains XGBoost, then runs Chronos in the same run).
    # Single-threaded torch sidesteps the clash, and Bolt inference on a
    # 15-hub batch stays well under a second per day.
    torch.set_num_threads(1)
    return BaseChronosPipeline.from_pretrained(
        model_id, device_map="cpu", torch_dtype=torch.float32
    )


def rolling_forecast(
    pipeline,
    wide: pd.DataFrame,
    test_dates: pd.DatetimeIndex,
    context_cap: int = CONTEXT_CAP,
    progress: bool = True,
) -> pd.DataFrame:
    """Day-ahead rolling-origin quantile forecasts for every hub and test day.

    ``wide`` is the actuals matrix (rows = the complete daily calendar,
    columns = hubs, gaps = NaN). For each test day the context is everything
    strictly before it — actual history only, never the model's own output —
    and all hubs go through the model as one batch.

    Returns a long frame: date, hub, p10/p50/p80/p90 (monotone per row).
    """
    import torch

    hubs = list(wide.columns)
    rows = []
    t0 = time.time()
    for i, day in enumerate(test_dates):
        history = wide.loc[wide.index < day].tail(context_cap)
        context = torch.tensor(history.to_numpy().T, dtype=torch.float32)
        q, _ = pipeline.predict_quantiles(
            context, prediction_length=1, quantile_levels=list(QUANTILE_LEVELS)
        )
        # Monotone rearrangement + clip at zero parcels, exactly as train.py
        # does for the XGBoost quantiles.
        q = np.sort(q[:, 0, :].numpy(), axis=1).clip(min=0.0)
        for j, hub in enumerate(hubs):
            rows.append({schema.DATE_COL: day, schema.HUB_COL: hub, **{
                f"p{int(a * 100)}": float(q[j, k]) for k, a in enumerate(QUANTILE_LEVELS)
            }})
        if progress and (i + 1) % 20 == 0:
            print(f"        day {i + 1}/{len(test_dates)} ({time.time() - t0:.0f}s elapsed)")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metrics (torch-free on purpose: testable wherever the fm extra is absent)
# ---------------------------------------------------------------------------
def compute_fm_metrics(actual: np.ndarray, q: pd.DataFrame, peak_mask: np.ndarray) -> dict:
    """Accuracy + quantile metrics in evaluate.py's exact shape, from P50/quantiles."""
    p50 = q["p50"].to_numpy()
    covered = (actual >= q["p10"].to_numpy()) & (actual <= q["p90"].to_numpy())
    return {
        "overall": {
            "wape": evaluate.wape(actual, p50),
            "smape": evaluate.smape(actual, p50),
            "bias": evaluate.bias(actual, p50),
        },
        "peak_season": {
            "wape": evaluate.wape(actual[peak_mask], p50[peak_mask]),
            "smape": evaluate.smape(actual[peak_mask], p50[peak_mask]),
            "bias": evaluate.bias(actual[peak_mask], p50[peak_mask]),
        },
        "pinball": {
            c: evaluate.pinball(actual, q[c].to_numpy(), int(c[1:]) / 100)
            for c in ["p10", "p50", "p90"]
        },
        "coverage_p10_p90": float(covered.mean()),
        "coverage_p10_p90_peak": float(covered[peak_mask].mean()),
        "mean_interval_width_pct": float((q["p90"] - q["p10"]).sum() / actual.sum()),
    }


def per_hub_winloss(
    test_df: pd.DataFrame,
    actual: np.ndarray,
    xgb_point: np.ndarray,
    fm_p50s: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Per-hub WAPE for XGBoost and each Chronos model, largest hub first."""
    rows = []
    for hub in sorted(test_df[schema.HUB_COL].unique(), key=lambda h: -synthetic.HUB_BASE[h]):
        m = (test_df[schema.HUB_COL] == hub).to_numpy()
        row = {"hub": hub, "wape_xgboost": evaluate.wape(actual[m], xgb_point[m])}
        for name, p50 in fm_p50s.items():
            row[f"wape_{name}"] = evaluate.wape(actual[m], p50[m])
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The benchmark run
# ---------------------------------------------------------------------------
def run_benchmark(
    model_keys: tuple[str, ...] = ("tiny", "small"),
    artifacts: str | Path = "artifacts",
    seed: int = 7,
    docs_img: str | Path | None = "docs/img",
) -> dict:
    """Full FM-vs-XGBoost benchmark on the standard pipeline's held-out window."""
    report_dir = Path(artifacts) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] rebuilding the standard pipeline (same seed, same split) ...")
    raw = synthetic.make_dataset(seed=seed, messy=True)
    clean_df, _ = cleaning.clean(raw)
    models, splits = train_mod.train(clean_df, train_mod.TrainConfig(seed=seed))
    test_df = splits["test"].reset_index(drop=True)
    X_test = splits["X_test"].reset_index(drop=True)
    actual = test_df[schema.TARGET_COL].to_numpy()
    peak_mask = test_df["is_peak_ramp"].to_numpy().astype(bool)
    print(f"      held-out window starts after {models.cutoff_date}; "
          f"{len(test_df):,} hub-days, {peak_mask.sum()} in peak")

    naive = models.naive.predict(X_test)
    xgb_point = train_mod.predict_point(models, X_test)
    xgb_q = train_mod.predict_quantiles(models, X_test)

    # Actuals matrix on the complete calendar: the context Chronos rolls over.
    wide = clean_df.pivot(
        index=schema.DATE_COL, columns=schema.HUB_COL, values=schema.TARGET_COL
    ).reindex(pd.date_range(clean_df[schema.DATE_COL].min(), clean_df[schema.DATE_COL].max()))
    test_dates = pd.DatetimeIndex(sorted(test_df[schema.DATE_COL].unique()))

    results: dict = {
        "protocol": {
            "type": "rolling_origin_day_ahead",
            "context": "actual history through t-1, capped at 2048 days, gaps as NaN",
            "device": "cpu",
            "covariates_visible_to_chronos": "none (univariate zero-shot)",
        },
        "accuracy": {}, "runtime_seconds": {},
    }
    fm_q_frames: dict[str, pd.DataFrame] = {}
    fm_p50s: dict[str, np.ndarray] = {}

    for step, key in enumerate(model_keys):
        model_id = CHRONOS_MODELS[key]
        name = f"chronos_bolt_{key}"
        print(f"[{step + 2}/4] {model_id}: rolling day-ahead over "
              f"{len(test_dates)} days x {wide.shape[1]} hubs ...")
        t0 = time.time()
        pipeline = load_pipeline(model_id)
        fm = rolling_forecast(pipeline, wide, test_dates)
        elapsed = time.time() - t0
        results["runtime_seconds"][name] = round(elapsed, 1)

        # Align to the exact rows every other model is scored on.
        merged = test_df[[schema.DATE_COL, schema.HUB_COL]].merge(
            fm, on=[schema.DATE_COL, schema.HUB_COL], how="left", validate="one_to_one"
        )
        q_cols = [f"p{int(a * 100)}" for a in QUANTILE_LEVELS]
        assert merged[q_cols].notna().all().all(), "Chronos left test rows unforecast"
        fm_q_frames[name] = merged[q_cols]
        fm_p50s[name] = merged["p50"].to_numpy()
        results["accuracy"][name] = compute_fm_metrics(actual, merged[q_cols], peak_mask)
        print(f"      done in {elapsed:.0f}s -> WAPE "
              f"{results['accuracy'][name]['overall']['wape']:.1%} overall, "
              f"{results['accuracy'][name]['peak_season']['wape']:.1%} peak")

    # Incumbents, through the identical metric path.
    results["accuracy"]["seasonal_naive"] = {
        "overall": evaluate._metric_row(actual, naive),
        "peak_season": evaluate._metric_row(actual[peak_mask], naive[peak_mask]),
    }
    results["accuracy"]["xgboost"] = compute_fm_metrics(actual, xgb_q, peak_mask)

    print("[4/4] scoring + writing artifacts ...")
    per_hub = per_hub_winloss(test_df, actual, xgb_point, fm_p50s)
    results["per_hub"] = per_hub.to_dict(orient="records")
    results["per_hub_wins_vs_xgboost"] = {
        name: {
            "chronos_wins": int((per_hub[f"wape_{name}"] < per_hub["wape_xgboost"]).sum()),
            "xgboost_wins": int((per_hub[f"wape_{name}"] >= per_hub["wape_xgboost"]).sum()),
        }
        for name in fm_p50s
    }
    results["test_window_start"] = models.cutoff_date
    results["n_test_hub_days"] = int(len(test_df))

    comparison = _comparison_table(results["accuracy"])
    comparison.to_csv(report_dir / "fm_comparison.csv", index=False)
    (report_dir / "fm_benchmark.json").write_text(json.dumps(results, indent=2))

    best = min(fm_p50s, key=lambda n: results["accuracy"][n]["overall"]["wape"])
    fig_path = report_dir / "fm_vs_xgb.png"
    _plot_fm_overlay(test_df, xgb_q, fm_q_frames[best], best, fig_path)
    if docs_img is not None and Path(docs_img).is_dir():
        _plot_fm_overlay(test_df, xgb_q, fm_q_frames[best], best, Path(docs_img) / "fm_vs_xgb.png")

    print(f"      wrote {report_dir / 'fm_benchmark.json'}, fm_comparison.csv, {fig_path}")
    return results


def _comparison_table(accuracy: dict) -> pd.DataFrame:
    """One row per method, evaluate.py-style columns; naive has no quantiles."""
    rows = []
    for name, acc in accuracy.items():
        rows.append({
            "method": name,
            "wape_overall": acc["overall"]["wape"],
            "wape_peak": acc["peak_season"]["wape"],
            "smape_overall": acc["overall"]["smape"],
            "bias_overall": acc["overall"]["bias"],
            "bias_peak": acc["peak_season"]["bias"],
            "pinball_p10": acc.get("pinball", {}).get("p10"),
            "pinball_p50": acc.get("pinball", {}).get("p50"),
            "pinball_p90": acc.get("pinball", {}).get("p90"),
            "coverage_p10_p90": acc.get("coverage_p10_p90"),
            "coverage_p10_p90_peak": acc.get("coverage_p10_p90_peak"),
        })
    return pd.DataFrame(rows)


def _plot_fm_overlay(test_df, xgb_q, fm_q, fm_name: str, path: Path) -> None:
    """December overlay for the biggest hub: actual vs XGBoost P50 vs Chronos P50+band."""
    hub = max(synthetic.HUB_BASE, key=synthetic.HUB_BASE.get)
    m = (test_df[schema.HUB_COL] == hub).to_numpy()
    sub = test_df[m]
    dates = sub[schema.DATE_COL]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.fill_between(
        dates, fm_q.loc[m, "p10"], fm_q.loc[m, "p90"],
        alpha=0.25, color="#dd6b20", label=f"{fm_name} P10-P90", lw=0,
    )
    ax.plot(dates, sub[schema.TARGET_COL], color="black", lw=1.2, label="actual")
    ax.plot(dates, xgb_q.loc[m, "p50"], color="#2b6cb0", lw=1.2, label="XGBoost (P50)")
    ax.plot(dates, fm_q.loc[m, "p50"], color="#dd6b20", lw=1.2, label=f"{fm_name} (P50)")
    ax.set_title(
        f"{hub}: covariate-aware XGBoost vs zero-shot {fm_name} (held-out window)"
    )
    ax.set_ylabel("inbound parcels / day")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
