"""SHAP driver analysis on the P50 model: what actually moves tomorrow's wave.

Outputs:

- ``shap_summary.png``                    — beeswarm: global drivers with direction
- ``shap_dependence_day_of_week.png``     — the weekly rhythm, as the model learned it
- ``shap_dependence_days_to_nearest_holiday.png`` — surge-then-collapse around holidays
- ``driver_ranking.csv``                  — per-feature ranking with each feature
  mapped to its operational lever group (volume history / calendar / holiday
  & peak / hub identity / planted noise), plus group share subtotals

The explanation runs on the P50 quantile model because that is the number the
planner reads first. SHAP values are in log-parcels (the training scale); the
*ranking* is what the tests pin down. On the synthetic dataset the ranking
must recover the generator's exposed components (``synthetic.TRUE_COMPONENTS``)
and bury the planted noise columns — that check lives in the test suite, so a
refactor that silently breaks explanations fails CI.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from . import schema

# Feature -> operational lever group. "Which knob does an ops manager turn"
# is the level the morning call talks at, not individual matrix columns.
GROUPS = {
    "lag_7": "volume history",
    "lag_14": "volume history",
    "lag_28": "volume history",
    "trailing_mean_7": "volume history",
    "trailing_mean_28": "volume history",
    "day_of_week": "calendar",
    "month": "calendar",
    "doy_sin": "calendar",
    "doy_cos": "calendar",
    "is_holiday": "holiday & peak",
    "days_to_nearest_holiday": "holiday & peak",
    "is_peak_ramp": "holiday & peak",
    "is_promo_day": "holiday & peak",
    "hub_paint_color_code": "planted noise",
    "moon_phase": "planted noise",
}


def _group(col: str) -> str:
    if col.startswith(schema.HUB_COL + "_"):
        return "hub identity"
    return GROUPS.get(col, col)


def explain(models, splits, out_dir: str | Path, max_background: int = 4000) -> pd.DataFrame:
    """Compute SHAP on the held-out window; write plots + ranking CSV."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_test = splits["X_test"]
    X_sample = (
        X_test.sample(max_background, random_state=0) if len(X_test) > max_background else X_test
    )

    explainer = shap.TreeExplainer(models.xgb_quantiles[0.5])
    shap_values = explainer.shap_values(X_sample)

    # --- beeswarm -----------------------------------------------------------
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False, max_display=14)
    plt.title("What drives tomorrow's inbound volume (P50 model)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # --- per-feature ranking with lever groups --------------------------------
    mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=X_sample.columns)
    ranking = (
        mean_abs.rename("mean_abs_shap")
        .reset_index()
        .rename(columns={"index": "feature"})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    ranking["lever_group"] = ranking["feature"].map(_group)
    ranking["share_of_explanation"] = ranking["mean_abs_shap"] / ranking["mean_abs_shap"].sum()
    ranking["group_share"] = ranking.groupby("lever_group")["share_of_explanation"].transform(
        "sum"
    )
    ranking.to_csv(out_dir / "driver_ranking.csv", index=False)

    # --- dependence plots for the two drivers ops argues about ----------------
    for col in ["day_of_week", "days_to_nearest_holiday"]:
        plt.figure()
        shap.dependence_plot(col, shap_values, X_sample, interaction_index=None, show=False)
        plt.tight_layout()
        plt.savefig(out_dir / f"shap_dependence_{col}.png", dpi=150, bbox_inches="tight")
        plt.close("all")

    return ranking
