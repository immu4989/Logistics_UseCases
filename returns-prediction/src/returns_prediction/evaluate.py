"""Evaluation: from classifier metrics to reverse-logistics dollars.

The ranking metrics (PR-AUC, ROC-AUC) are table stakes. The numbers that
decide whether this model ships are the dollar views:

- **Expected return cost, not raw probability, is the sort key.** The cost of
  a return is roughly fixed logistics (reverse shipping + processing) plus a
  category-dependent slice of the price that never comes back. So a 60%
  return risk on a $200 bracket order is worth far more attention than an 80%
  risk on a $15 tee: $34 of expected loss versus $12. Ranking by p alone
  optimises a KPI nobody pays; ranking by p * cost optimises the invoice.
- **Lift by expected-cost decile, in dollars.** "The top decile of orders by
  expected cost carries X% of all return dollars" is the sentence that gets a
  returns program funded.
- **Calibration by category segment.** Segment-level honesty is what the
  p * cost arithmetic actually consumes; a model that is calibrated on
  average but overshoots beauty and undershoots shoes misprices every order.
- **The intervention simulation** — the product metric. Target the top k% of
  orders by expected cost with a pre-ship fit-assistant / size-nudge flow and
  compare the net savings against random targeting and against raw-probability
  targeting.

Every dollar constant below is a business input, not a modelling choice.
Recalibrate REVERSE_SHIPPING_USD and PROCESSING_USD from your 3PL invoices,
and RESTOCK_LOSS_FRAC from your returns-disposition data, before believing
any dollar figure this pipeline prints.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from . import schema

# --- the cost model (recalibrate from your own data) -----------------------
REVERSE_SHIPPING_USD = 8.0   # carrier pickup / label cost per returned parcel
PROCESSING_USD = 4.0         # inspection, grading, repackaging labour
RESTOCK_LOSS_FRAC = {
    # Share of item price lost even when the return goes smoothly: markdowns,
    # damage, liquidation. Beauty is 1.0 because opened hygiene product cannot
    # legally be restocked -- the whole item is written off.
    "apparel": 0.15,
    "shoes": 0.15,
    "electronics": 0.08,
    "home": 0.10,
    "beauty": 1.00,
}

# --- the intervention (recalibrate from your own A/B tests) -----------------
INTERVENTION_COST_USD = 0.30      # fit-assistant / size-nudge flow, per targeted order
INTERVENTION_EFFECT = 0.25        # relative cut in return probability, sized goods only
INTERVENTION_CATEGORIES = ("apparel", "shoes")  # a size nudge means nothing on a blender


def unit_return_cost(df: pd.DataFrame) -> np.ndarray:
    """Dollar cost IF this order comes back: fixed logistics + restock loss."""
    frac = df["product_category"].map(RESTOCK_LOSS_FRAC).astype(float).to_numpy()
    return REVERSE_SHIPPING_USD + PROCESSING_USD + frac * df["unit_price_usd"].to_numpy()


def summarize(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Core ranking-metric dict for one model."""
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "base_return_rate": float(np.mean(y_true)),
    }


def cost_decile_table(y_true: np.ndarray, y_prob: np.ndarray, cost: np.ndarray) -> pd.DataFrame:
    """Decile table sorted by EXPECTED return cost (p * cost), reported in dollars."""
    df = pd.DataFrame({"y": y_true, "expected_cost": y_prob * cost, "cost": cost})
    df["realised_cost"] = df["y"] * df["cost"]
    df = df.sort_values("expected_cost", ascending=False)
    df["decile"] = np.ceil((np.arange(len(df)) + 1) / len(df) * 10).astype(int)
    out = (
        df.groupby("decile")
        .agg(
            orders=("y", "size"),
            returns=("y", "sum"),
            return_rate=("y", "mean"),
            return_cost_usd=("realised_cost", "sum"),
        )
        .reset_index()
    )
    out["cum_cost_capture"] = out["return_cost_usd"].cumsum() / out["return_cost_usd"].sum()
    return out


def dollars_captured_at(
    y_true: np.ndarray, score: np.ndarray, cost: np.ndarray, k: float = 0.10
) -> float:
    """Realised return dollars sitting inside the top-k% of orders by `score`."""
    n_top = max(1, int(round(k * len(score))))
    top = np.argsort(-score)[:n_top]
    return float((y_true[top] * cost[top]).sum())


def simulate_intervention(
    test_df: pd.DataFrame,
    y_prob: np.ndarray,
    cost: np.ndarray,
    k: float = 0.10,
    seed: int = 0,
) -> pd.DataFrame:
    """Compare targeting policies for a pre-ship fit-assistant intervention.

    All policies get the same budget: k% of the day's orders, chosen among
    ELIGIBLE orders (apparel/shoes -- a size nudge cannot avert an electronics
    return), each costing INTERVENTION_COST_USD. The intervention cuts the
    return probability of a targeted order by INTERVENTION_EFFECT, so the
    expected saving on a targeted order is effect * (realised return cost) --
    evaluating on realised outcomes keeps the estimate unbiased without
    needing counterfactual draws.

    Policies:
    - random           pick eligible orders at random (the no-model program)
    - raw_probability  pick the eligible orders most LIKELY to return
    - expected_cost    pick the eligible orders whose returns COST the most

    Expected-cost targeting should win: savings are proportional to p * cost,
    which is exactly what it sorts by, while probability targeting happily
    burns budget on near-certain returns of $15 tees.
    """
    y = test_df[schema.LABEL_COL].to_numpy()
    eligible = test_df["product_category"].isin(INTERVENTION_CATEGORIES).to_numpy()
    n_target = max(1, int(round(k * len(test_df))))
    rng = np.random.default_rng(seed)

    scores = {
        "random": rng.random(len(test_df)),
        "raw_probability": y_prob,
        "expected_cost": y_prob * cost,
    }
    rows = []
    for policy, score in scores.items():
        masked = np.where(eligible, score, -np.inf)
        target = np.argsort(-masked)[:n_target]
        target = target[eligible[target]]  # never spend on ineligible orders
        savings = INTERVENTION_EFFECT * float((y[target] * cost[target]).sum())
        spend = INTERVENTION_COST_USD * len(target)
        rows.append(
            {
                "policy": policy,
                "orders_targeted": int(len(target)),
                "spend_usd": round(spend, 2),
                "expected_savings_usd": round(savings, 2),
                "net_savings_usd": round(savings - spend, 2),
                "roi": round((savings - spend) / spend, 2) if spend else 0.0,
            }
        )
    return pd.DataFrame(rows)


def evaluate_models(models, splits, out_dir: str | Path, k: float = 0.10) -> dict:
    """Score baseline + XGBoost on the held-out period; write metrics + plots."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_df, X_test, y_test = splits["test"], splits["X_test"], splits["y_test"]
    probs = {
        "logistic_baseline": models.baseline.predict_proba(X_test)[:, 1],
        "xgboost": models.xgb.predict_proba(X_test)[:, 1],
    }
    cost = unit_return_cost(test_df)
    p = probs["xgboost"]

    results: dict = {name: summarize(y_test, pr) for name, pr in probs.items()}
    results["test_period_start"] = models.cutoff_date
    results["n_test"] = int(len(y_test))

    # Dollar view: what fraction of return spend does the top decile carry,
    # and how much does the p * cost sort key buy over p alone?
    decile = cost_decile_table(y_test, p, cost)
    total_cost = float((y_test * cost).sum())
    rng = np.random.default_rng(0)
    results["cost_view"] = {
        "total_return_cost_usd": round(total_cost, 2),
        "flag_frac": k,
        "captured_usd_expected_cost": round(dollars_captured_at(y_test, p * cost, cost, k), 2),
        "captured_usd_raw_probability": round(dollars_captured_at(y_test, p, cost, k), 2),
        "captured_usd_random": round(
            dollars_captured_at(y_test, rng.random(len(y_test)), cost, k), 2
        ),
        "top_decile_cost_share": round(float(decile.loc[0, "cum_cost_capture"]), 4),
        "top_3_decile_cost_share": round(float(decile.loc[2, "cum_cost_capture"]), 4),
    }

    intervention = simulate_intervention(test_df, p, cost, k)
    results["intervention"] = intervention.to_dict(orient="records")

    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    decile.to_csv(out_dir / "cost_decile_table.csv", index=False)
    intervention.to_csv(out_dir / "intervention_policies.csv", index=False)

    _plot_cost_deciles(decile, out_dir / "lift_by_expected_cost_decile.png")
    _plot_category_calibration(test_df, y_test, p, out_dir / "category_return_rates.png")
    _plot_intervention(intervention, out_dir / "intervention_policy_comparison.png")
    return results


def _plot_cost_deciles(decile: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(decile["decile"], decile["return_cost_usd"], color="#2b6cb0")
    ax.set_xticks(decile["decile"])
    ax.set_xlabel("Expected-cost decile (1 = highest p x cost)")
    ax.set_ylabel("Realised return cost ($)")
    ax.set_title("Where the return dollars live (held-out period)")
    for _, row in decile.iterrows():
        ax.text(
            row["decile"],
            row["return_cost_usd"],
            f"${row['return_cost_usd'] / 1000:.1f}k",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_category_calibration(test_df, y_test, y_prob, path: Path) -> None:
    seg = pd.DataFrame(
        {"category": test_df["product_category"].to_numpy(), "y": y_test, "p": y_prob}
    )
    agg = seg.groupby("category").agg(actual=("y", "mean"), predicted=("p", "mean"))
    agg = agg.sort_values("actual", ascending=False)
    x = np.arange(len(agg))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - 0.2, agg["actual"], width=0.4, label="actual return rate", color="#2b6cb0")
    ax.bar(x + 0.2, agg["predicted"], width=0.4, label="mean predicted", color="#ed8936")
    ax.set_xticks(x)
    ax.set_xticklabels(agg.index)
    ax.set_ylabel("Return rate")
    ax.set_title("Calibration by category (held-out period)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_intervention(intervention: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"random": "#a0aec0", "raw_probability": "#ed8936", "expected_cost": "#2b6cb0"}
    ax.bar(
        intervention["policy"],
        intervention["net_savings_usd"],
        color=[colors[p] for p in intervention["policy"]],
    )
    ax.axhline(0, color="k", lw=1)
    ax.set_ylabel("Net savings ($, held-out period)")
    ax.set_title("Pre-ship fit-assistant: net savings by targeting policy")
    for _, row in intervention.iterrows():
        ax.text(
            row["policy"],
            row["net_savings_usd"],
            f"${row['net_savings_usd']:,.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
