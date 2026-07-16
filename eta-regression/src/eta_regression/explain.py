"""SHAP driver analysis on the median (P50) quantile model.

Outputs:

- ``shap_summary.png``       — beeswarm: global drivers with direction of effect
- ``shap_importance.png``    — mean |SHAP| bar chart
- ``shap_dependence_*.png``  — the top drivers vs. their SHAP contribution
- ``driver_ranking.csv``     — global ranking (grouped back to operational
  levers, i.e. one-hot columns re-aggregated to their source feature)
- ``example_shipment.md``    — a local explanation for one long-lane shipment,
  the format a customer-service screen would show next to a quoted window

Why explain the P50 model rather than the point model: the P50 is the ETA the
product displays, and because the target is transit days, every SHAP value
reads directly in days ("this lane's distance adds 2.1 days") — no log-odds
translation needed for the ops call.

On the synthetic dataset the ranking should recover the generator's
ground-truth process (see ``synthetic.TRUE_DRIVERS``) — that check lives in
the test suite, so a refactor that silently breaks explanations fails CI.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from . import features
from . import train as train_mod


def _group_feature(col: str) -> str:
    """Map a model-matrix column back to its operational lever."""
    for prefix in features.ONE_HOT_COLS:
        if col.startswith(prefix + "_"):
            return prefix
    if col.endswith("__was_missing"):
        return col.replace("__was_missing", "") + " (missingness)"
    return col


def explain(models, splits, out_dir: str | Path, max_background: int = 5000) -> pd.DataFrame:
    """Compute SHAP values on the held-out period; write plots + ranking."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_test = splits["X_test"]
    if len(X_test) > max_background:
        X_sample = X_test.sample(max_background, random_state=0)
    else:
        X_sample = X_test

    median_model = models.xgb_quantiles[0.5]
    explainer = shap.TreeExplainer(median_model)
    shap_values = explainer.shap_values(X_sample)

    # --- global plots ------------------------------------------------------
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
    plt.title("What drives transit time (SHAP on the P50 model, in days)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # --- driver ranking, grouped to operational levers ----------------------
    mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=X_sample.columns)
    ranking = (
        mean_abs.groupby(mean_abs.index.map(_group_feature))
        .sum()
        .sort_values(ascending=False)
        .rename("mean_abs_shap_days")
        .reset_index()
        .rename(columns={"index": "driver"})
    )
    ranking["share_of_explanation"] = (
        ranking["mean_abs_shap_days"] / ranking["mean_abs_shap_days"].sum()
    )
    ranking.to_csv(out_dir / "driver_ranking.csv", index=False)

    # --- dependence plots for the top raw features --------------------------
    top_raw = mean_abs.sort_values(ascending=False).head(2).index
    for col in top_raw:
        plt.figure()
        shap.dependence_plot(col, shap_values, X_sample, interaction_index=None, show=False)
        plt.ylabel(f"SHAP value for {col} (days)")
        plt.tight_layout()
        safe = col.replace("/", "_")
        plt.savefig(out_dir / f"shap_dependence_{safe}.png", dpi=150, bbox_inches="tight")
        plt.close("all")

    # --- one local explanation ----------------------------------------------
    _write_example(models, X_sample, shap_values, out_dir)
    return ranking


def _write_example(models, X_sample, shap_values, out_dir: Path) -> None:
    """Explain one long-lane shipment: the case where the interval earns its keep."""
    long_lanes = X_sample["distance_miles"] >= X_sample["distance_miles"].quantile(0.9)
    qpreds = train_mod.predict_quantiles(models, X_sample[long_lanes], alphas=(0.1, 0.5, 0.9))
    pos = int(np.argmax(qpreds[0.9].to_numpy()))  # widest-risk shipment on a long lane
    idx = qpreds.index[pos]
    iloc = X_sample.index.get_loc(idx)

    row_shap = pd.Series(shap_values[iloc], index=X_sample.columns)
    top = row_shap.reindex(row_shap.abs().sort_values(ascending=False).head(8).index)

    lines = [
        "# Example: how one long-lane ETA was built",
        "",
        f"Quoted window: P50 **{qpreds[0.5].iloc[pos]:.1f} days**, "
        f"P90 **{qpreds[0.9].iloc[pos]:.1f} days** "
        f"(promise ceil(P90) = **{int(np.ceil(qpreds[0.9].iloc[pos]))} days**, "
        f"P10 {qpreds[0.1].iloc[pos]:.1f})",
        "",
        "| Driver | Value | Contribution to P50 ETA (days) |",
        "|---|---|---|",
    ]
    for col, val in top.items():
        arrow = "adds" if val > 0 else "saves"
        lines.append(f"| {col} | {X_sample.loc[idx, col]:.3g} | {val:+.2f} ({arrow} time) |")
    lines += [
        "",
        "_Positive contributions push the ETA later; negative pull it earlier._",
    ]
    (out_dir / "example_shipment.md").write_text("\n".join(lines))
