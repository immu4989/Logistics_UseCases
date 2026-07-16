"""SHAP driver analysis: from "this order will come back" to "here is why, and what to do".

Outputs:

- ``shap_summary.png``       — beeswarm: global drivers with direction of effect
- ``shap_importance.png``    — mean |SHAP| bar chart
- ``shap_dependence_*.png``  — discount depth and prior return rate vs. their
  SHAP contribution (the two drivers whose *shape* matters for policy: where
  does discounting flip risky, and how fast does a returner's history bite?)
- ``driver_ranking.csv``     — global ranking, grouped back to merchandising
  levers (one-hot columns re-aggregated, the bracket pair merged)
- ``example_order.md``       — a local explanation for one high-risk bracket
  order, the card a CX / returns desk would see before the parcel ships

On the synthetic dataset the ranking should recover the generator's
ground-truth process (see ``synthetic.TRUE_DRIVERS``) and bury the planted
noise (``page_dwell_seconds``, ``ad_campaign_id``) — both checks live in the
test suite, so a refactor that silently breaks explanations fails CI.
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

# Model-matrix columns that are one lever wearing two coats.
GROUPS = {
    "is_bracket_buy": "bracket_buying",
    "num_sizes_ordered": "bracket_buying",
    "prior_return_rate": "prior_return_rate",
    "prior_returns": "prior_return_rate",
}


def _group_feature(col: str) -> str:
    """Map a model-matrix column back to its merchandising lever."""
    for prefix in features.ONE_HOT_COLS:
        if col.startswith(prefix + "_"):
            return prefix
    if col.endswith("__was_missing"):
        return col.replace("__was_missing", "") + " (missingness)"
    return GROUPS.get(col, col)


def explain(models, splits, out_dir: str | Path, max_background: int = 5000) -> pd.DataFrame:
    """Compute SHAP values on the held-out period; write plots + ranking + order card."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_test = splits["X_test"]
    if len(X_test) > max_background:
        pos = np.sort(np.random.default_rng(0).choice(len(X_test), max_background, replace=False))
    else:
        pos = np.arange(len(X_test))
    X_sample = X_test.iloc[pos]
    test_sample = splits["test"].iloc[pos]

    explainer = shap.TreeExplainer(models.xgb)
    shap_values = explainer.shap_values(X_sample)

    # --- global plots ------------------------------------------------------
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
    plt.title("What drives product returns", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # --- driver ranking, grouped to merchandising levers --------------------
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

    # --- dependence plots: the shape of the two policy-relevant drivers ------
    for col in ["discount_pct", "prior_return_rate"]:
        plt.figure()
        shap.dependence_plot(col, shap_values, X_sample, interaction_index=None, show=False)
        plt.tight_layout()
        plt.savefig(out_dir / f"shap_dependence_{col}.png", dpi=150, bbox_inches="tight")
        plt.close("all")

    # --- one local explanation: a high-risk bracket order --------------------
    _write_example(models, X_sample, test_sample, shap_values, out_dir)
    return ranking


def _write_example(models, X_sample, test_sample, shap_values, out_dir: Path) -> None:
    """The pre-ship card: the riskiest bracket-buy order, explained in plain language."""
    probs = models.xgb.predict_proba(X_sample)[:, 1]
    # Prefer a bracket order from an ordinary shopper (not a power buyer with
    # dozens of prior orders) so the card reads like the everyday case.
    mask = (X_sample["is_bracket_buy"].to_numpy() == 1) & (
        X_sample["prior_orders"].to_numpy() <= 30
    )
    if not mask.any():
        mask = X_sample["is_bracket_buy"].to_numpy() == 1
    candidates = np.where(mask)[0] if mask.any() else np.arange(len(X_sample))
    idx = int(candidates[np.argmax(probs[candidates])])

    row = test_sample.iloc[idx]
    row_shap = pd.Series(shap_values[idx], index=X_sample.columns)
    top = row_shap.reindex(row_shap.abs().sort_values(ascending=False).head(8).index)

    lines = [
        "# Example: why this order was flagged before shipping",
        "",
        f"Order `{row['order_id']}` — {row['product_category']}, "
        f"${row['unit_price_usd']:.2f}, predicted return probability "
        f"**{probs[idx]:.0%}**",
        "",
        "| Driver | Value | Contribution to risk (log-odds) |",
        "|---|---|---|",
    ]
    for col, val in top.items():
        arrow = "raises" if val > 0 else "lowers"
        lines.append(f"| {col} | {X_sample.iloc[idx][col]:.3g} | {val:+.2f} ({arrow} risk) |")
    note = (
        f"**Suggested action:** {int(row['num_sizes_ordered'])} sizes of the same item in one "
        f"order and a {row['prior_return_rate']:.0%} historical return rate over "
        f"{int(row['prior_orders'])} prior orders. Offer the fit assistant before the parcel "
        "ships, and hold the bonus sample until the keep is confirmed."
    )
    lines += ["", note, "", "_Positive contributions push toward a return; negative pull away._"]
    (out_dir / "example_order.md").write_text("\n".join(lines))
