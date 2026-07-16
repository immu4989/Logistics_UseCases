"""SHAP driver analysis: from "the model predicts X" to "here is why, and what to fix".

Outputs:

- ``shap_summary.png``       — beeswarm: global drivers with direction of effect
- ``shap_importance.png``    — mean |SHAP| bar chart
- ``shap_dependence_*.png``  — the top drivers vs. their SHAP contribution
- ``driver_ranking.csv``     — global ranking (grouped back to operational
  levers, i.e. one-hot columns re-aggregated to their source feature)
- ``example_shipment.md``    — a local explanation for one high-risk shipment,
  the format an ops team would see next to a flagged package

On the synthetic dataset the ranking should recover the generator's
ground-truth coefficients (see ``synthetic.TRUE_DRIVERS``) — that check lives
in the test suite, so a refactor that silently breaks explanations fails CI.
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

    explainer = shap.TreeExplainer(models.xgb)
    shap_values = explainer.shap_values(X_sample)

    # --- global plots ------------------------------------------------------
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
    plt.title("What drives missed delivery commitments", fontsize=12)
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
        .rename("mean_abs_shap")
        .reset_index()
        .rename(columns={"index": "driver"})
    )
    ranking["share_of_explanation"] = ranking["mean_abs_shap"] / ranking["mean_abs_shap"].sum()
    ranking.to_csv(out_dir / "driver_ranking.csv", index=False)

    # --- dependence plots for the top raw features ---------------------------
    top_raw = mean_abs.sort_values(ascending=False).head(4).index
    for col in top_raw:
        plt.figure()
        shap.dependence_plot(
            col, shap_values, X_sample, interaction_index=None, show=False
        )
        plt.tight_layout()
        safe = col.replace("/", "_")
        plt.savefig(out_dir / f"shap_dependence_{safe}.png", dpi=150, bbox_inches="tight")
        plt.close("all")

    # --- one local explanation ----------------------------------------------
    _write_example(models, X_sample, shap_values, out_dir)
    return ranking


def _write_example(models, X_sample, shap_values, out_dir: Path) -> None:
    probs = models.xgb.predict_proba(X_sample)[:, 1]
    idx = int(np.argmax(probs))
    row_shap = pd.Series(shap_values[idx], index=X_sample.columns)
    top = row_shap.reindex(row_shap.abs().sort_values(ascending=False).head(8).index)

    lines = [
        "# Example: why this shipment was flagged",
        "",
        f"Predicted miss probability: **{probs[idx]:.0%}**",
        "",
        "| Driver | Value | Contribution to risk (log-odds) |",
        "|---|---|---|",
    ]
    for col, val in top.items():
        arrow = "raises" if val > 0 else "lowers"
        lines.append(f"| {col} | {X_sample.iloc[idx][col]:.3g} | {val:+.2f} ({arrow} risk) |")
    lines += [
        "",
        "_Positive contributions push toward a missed commitment; negative pull away._",
    ]
    (out_dir / "example_shipment.md").write_text("\n".join(lines))
