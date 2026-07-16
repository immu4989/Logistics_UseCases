"""SHAP driver analysis: from "this vehicle is risky" to a work order a mechanic trusts.

Outputs:

- ``shap_summary.png``        — beeswarm: which readings push risk up or down
- ``shap_dependence_*.png``   — functional form of the top two features
- ``driver_ranking.csv``      — global ranking, with every rolling feature
  grouped back to the physical sensor it came from (``vibration_index_mean_7d``
  and ``vibration_index_slope_28d`` both roll up to "vibration"), because a
  mechanic reasons about sensors, not window statistics
- ``work_order_card.md``      — for the highest-risk vehicle in the test
  period: top contributing sensors, their recent trend, and a plain-language
  note of what the pattern is consistent with

On the synthetic fleet the ranking is checkable against ground truth: the
generator documents which channels carry wear signal and which are planted
noise (cabin temperature setting, radio volume). The test suite asserts SHAP
puts the real sensors on top and buries the noise — so a refactor that breaks
explanations fails CI, instead of shipping confident nonsense to a workshop.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from . import synthetic

# Map any model-matrix column back to the physical thing a mechanic can check.
SENSOR_GROUPS = {
    "avg_engine_temp": "engine temp",
    "oil_pressure": "oil pressure",
    "vibration_index": "vibration",
    "battery_voltage": "battery voltage",
    "fault_code_count": "fault codes",
    "hard_braking_count": "hard braking",
    "daily_miles": "daily mileage",
    "engine_hours": "engine hours",
    "days_since_maint": "service interval",
    "miles_since_maint": "service interval",
    "vehicle_age_years": "vehicle age",
    "cabin_temp_setting": "cabin temp setting (noise)",
    "radio_volume_avg": "radio volume (noise)",
    "season_sin": "season",
    "season_cos": "season",
    "sensor_dropout_rate_7d": "sensor dropout",
}
NOISE_GROUPS = ["cabin temp setting (noise)", "radio volume (noise)"]
SIGNAL_GROUPS = ["engine temp", "oil pressure", "vibration", "battery voltage", "fault codes"]


def group_feature(col: str) -> str:
    """Roll a model-matrix column back up to its sensor group."""
    if col.endswith("__was_missing"):
        return "sensor dropout"
    for prefix, group in SENSOR_GROUPS.items():
        if col == prefix or col.startswith(prefix + "_"):
            return group
    return col


def explain(models, splits, out_dir: str | Path, max_background: int = 4000) -> pd.DataFrame:
    """Compute SHAP values on the held-out period; write plots, ranking and work order."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_test = splits["X_test"]
    X_sample = X_test.sample(min(max_background, len(X_test)), random_state=0)

    explainer = shap.TreeExplainer(models.xgb)
    shap_values = explainer.shap_values(X_sample)

    # --- global beeswarm ------------------------------------------------------
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False, max_display=14)
    plt.title("What drives 14-day breakdown risk", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # --- ranking, grouped back to physical sensors ----------------------------
    mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=X_sample.columns)
    ranking = (
        mean_abs.groupby(mean_abs.index.map(group_feature))
        .sum()
        .sort_values(ascending=False)
        .rename("mean_abs_shap")
        .reset_index()
        .rename(columns={"index": "driver"})
    )
    ranking["share_of_explanation"] = ranking["mean_abs_shap"] / ranking["mean_abs_shap"].sum()
    ranking.to_csv(out_dir / "driver_ranking.csv", index=False)

    # --- dependence plots for the two strongest raw features -------------------
    for col in mean_abs.sort_values(ascending=False).head(2).index:
        plt.figure()
        shap.dependence_plot(col, shap_values, X_sample, interaction_index=None, show=False)
        plt.tight_layout()
        plt.savefig(out_dir / f"shap_dependence_{col}.png", dpi=150, bbox_inches="tight")
        plt.close("all")

    # --- work-order card for the riskiest vehicle ------------------------------
    write_work_order(models, splits, out_dir / "work_order_card.md")
    return ranking


def _trend_pct(panel_row: pd.Series, col: str) -> float | None:
    """Recent-vs-baseline change: 7d mean relative to 28d mean, as a percentage."""
    m7, m28 = panel_row.get(f"{col}_mean_7d"), panel_row.get(f"{col}_mean_28d")
    if m7 is None or m28 is None or not np.isfinite(m7) or not np.isfinite(m28) or m28 == 0:
        return None
    return 100.0 * (m7 - m28) / abs(m28)


# What a sustained move on each sensor is consistent with, mechanically.
_DIAGNOSIS = {
    "vibration": ("rising", "driveline or brake wear"),
    "engine temp": ("rising", "engine wear or cooling degradation"),
    "oil pressure": ("falling", "engine wear"),
    "battery voltage": ("falling", "battery end-of-life"),
    "fault codes": ("rising", "accumulating ECU faults"),
}


def write_work_order(models, splits, path: Path) -> Path:
    """Markdown work-order card for the highest-risk vehicle-day in the test period."""
    X_test = splits["X_test"]
    panel = splits["test"]
    probs = models.xgb.predict_proba(X_test)[:, 1]
    idx = X_test.index[int(np.argmax(probs))]

    explainer = shap.TreeExplainer(models.xgb)
    row_shap = pd.Series(
        explainer.shap_values(X_test.loc[[idx]])[0], index=X_test.columns
    )
    grouped = row_shap.groupby(row_shap.index.map(group_feature)).sum()
    top = grouped.reindex(grouped.abs().sort_values(ascending=False).head(6).index)

    prow = panel.loc[idx]
    veh, day = prow[synthetic.ID_COL], prow[synthetic.DATE_COL]

    lines = [
        f"# Work order: {veh}",
        "",
        f"Date: {pd.Timestamp(day).date()}  |  Predicted 14-day breakdown risk: "
        f"**{probs.max():.0%}**  |  Miles since service: "
        f"{prow['miles_since_maint']:,.0f}",
        "",
        "## Top contributing readings",
        "",
        "| Sensor group | Contribution (log-odds) | Recent trend (7d vs 28d) |",
        "|---|---:|---:|",
    ]
    notes = []
    raw_col = {v: k for k, v in SENSOR_GROUPS.items() if v in _DIAGNOSIS}
    for group, val in top.items():
        trend = _trend_pct(prow, raw_col[group]) if group in raw_col else None
        trend_txt = f"{trend:+.0f}%" if trend is not None else "n/a"
        lines.append(f"| {group} | {val:+.2f} | {trend_txt} |")
        if group in _DIAGNOSIS and val > 0 and trend is not None and abs(trend) >= 3:
            direction, meaning = _DIAGNOSIS[group]
            notes.append(f"{group} {direction} ({trend:+.0f}% over 28d)")
    diag = [g for g, v in top.items() if g in _DIAGNOSIS and v > 0]
    summary = "; ".join(notes) if notes else "multiple readings drifting from baseline"
    consistent = _DIAGNOSIS[diag[0]][1] if diag else "component wear"
    lines += [
        "",
        "## Recommendation",
        "",
        f"{summary}: consistent with {consistent}. Schedule bay time within the "
        "next few days rather than waiting for the service interval.",
        "",
        "_Contributions are SHAP values from the XGBoost model; positive pushes toward "
        "breakdown. Trend compares the last 7 days against the trailing 28-day baseline._",
    ]
    path.write_text("\n".join(lines))
    return path
