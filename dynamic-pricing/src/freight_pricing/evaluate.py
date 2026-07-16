"""Counterfactual evaluation: every pricing policy judged on the same quotes.

Because the generator exposes the TRUE acceptance model, each policy's
prices can be scored exactly, on the held-out test period:

- expected margin  = sum over quotes of (price - cost) x P_true(accept|price)
  — exact, no simulation noise; this is the headline metric.
- realized margin  — one simulated pass of accept/reject decisions, drawn
  with COMMON RANDOM NUMBERS: one uniform ``u`` per quote, reused across
  every policy. A quote is won iff ``u < P_true(accept | that policy's
  price)``. Reusing the draw makes comparisons paired (the same hard-to-win
  shippers are hard to win under every policy) and monotone (any quote won
  at a high price is also won at any lower price), which removes the
  between-run luck that would otherwise swamp real policy differences.
  Evaluate each policy on fresh draws instead and the ranking can flip run
  to run for no reason at all.

The margin-vs-volume frontier reruns the model policy with its prices scaled
up and down, tracing expected margin against expected win rate. That curve
is what a pricing desk actually negotiates over: sales wants volume, finance
wants margin, and the model's job is to draw the frontier so the argument is
about WHERE on it to sit, not about whose spreadsheet is right.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import cleaning, price, synthetic, train

FRONTIER_SCALES = np.linspace(0.85, 1.20, 15)

SEGMENT_ORDER = ["spot", "contract", "premium"]
POLICY_COLORS = {
    "cost_plus": "#8d99ae",
    "flat_optimal": "#e8a33d",
    "model_pricing": "#2b6cb0",
}


def price_policies(
    models: train.TrainedModels, train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[dict[str, np.ndarray], float]:
    """Every policy's prices for the test-period quotes.

    Returns ({policy name -> prices}, flat multiplier). The flat multiplier
    is fitted on the training period only.
    """
    flat_m = price.choose_flat_multiplier(models, train_df)
    prices = {
        "cost_plus": price.policy_cost_plus(test_df),
        "flat_optimal": price.policy_flat(test_df, flat_m),
        "model_pricing": price.policy_model_pricing(test_df, models),
        "oracle": price.policy_oracle(test_df),
    }
    return prices, flat_m


def _policy_metrics(df: pd.DataFrame, prices: np.ndarray, u: np.ndarray) -> dict:
    """Exact expected metrics plus one paired simulated realization."""
    cost = df["our_cost_usd"].to_numpy()
    p_true = synthetic.true_accept_prob(df, prices)
    won = u < p_true
    return {
        "expected_margin_usd": float(((prices - cost) * p_true).sum()),
        "realized_margin_usd": float(((prices - cost) * won).sum()),
        "expected_win_rate": float(p_true.mean()),
        "realized_win_rate": float(won.mean()),
        "avg_price_multiplier": float((prices / cost).mean()),
    }


def evaluate_policies(
    models: train.TrainedModels,
    splits: dict,
    seed: int = 7,
    out_dir: str | Path = "artifacts/reports",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], dict]:
    """Score every policy on the held-out test period; write metrics + plots.

    Returns (policy comparison, segment uplift table, {policy -> prices},
    info dict with AUCs and the flat multiplier).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df, test_df = splits["train"], splits["test"]

    prices, flat_m = price_policies(models, train_df, test_df)

    # One uniform per test quote, shared by every policy and every frontier
    # point. Offset the seed stream so the outcome draws never collide with
    # generator randomness — the quotes' luck must not depend on the policy.
    u = np.random.default_rng(seed + 1_000_003).random(len(test_df))

    rows = []
    for name, p in prices.items():
        m = _policy_metrics(test_df, p, u)
        rows.append({"policy": name, **{k: round(v, 4) for k, v in m.items()}})
    comparison = pd.DataFrame(rows)
    oracle_margin = comparison.loc[
        comparison["policy"] == "oracle", "expected_margin_usd"
    ].iloc[0]
    cp_margin = comparison.loc[comparison["policy"] == "cost_plus", "expected_margin_usd"].iloc[0]
    comparison["margin_per_quote_usd"] = (
        comparison["expected_margin_usd"] / len(test_df)
    ).round(2)
    comparison["uplift_vs_cost_plus_pct"] = (
        (comparison["expected_margin_usd"] / cp_margin - 1) * 100
    ).round(1)
    comparison["pct_of_oracle"] = (
        comparison["expected_margin_usd"] / oracle_margin * 100
    ).round(1)

    segment = _segment_table(test_df, prices)
    frontier = _frontier(test_df, prices["model_pricing"])
    aucs = _model_quality(models, test_df)

    # ---- persist -------------------------------------------------------------
    comparison.to_csv(out_dir / "policy_comparison.csv", index=False)
    segment.to_csv(out_dir / "segment_uplift.csv", index=False)
    metrics = {
        "n_test_quotes": int(len(test_df)),
        "seed": int(seed),
        "cutoff_date": models.cutoff_date,
        "auc_logistic": aucs["logistic"],
        "auc_xgboost": aucs["xgboost"],
        "flat_optimal_multiplier": round(flat_m, 4),
        "share_of_model_quotes_at_cap": round(
            float(
                (
                    prices["model_pricing"] / test_df["our_cost_usd"].to_numpy()
                    > price.GUARDRAIL_CAP - 0.01
                ).mean()
            ),
            4,
        ),
        "policies": comparison.to_dict(orient="records"),
        "segments": segment.to_dict(orient="records"),
        "frontier": frontier.to_dict(orient="records"),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    _plot_comparison(comparison, out_dir / "policy_comparison.png")
    _plot_frontier(frontier, comparison, out_dir / "margin_volume_frontier.png")
    _plot_elasticity(models, test_df, out_dir / "elasticity_curves.png")

    info = {"aucs": aucs, "flat_multiplier": flat_m}
    return comparison, segment, prices, info


def _segment_table(test_df: pd.DataFrame, prices: dict[str, np.ndarray]) -> pd.DataFrame:
    """Where the uplift comes from: cost-plus vs model pricing, by segment."""
    cost = test_df["our_cost_usd"].to_numpy()
    em = {
        name: (p - cost) * synthetic.true_accept_prob(test_df, p)
        for name, p in prices.items()
    }
    total_uplift = (em["model_pricing"] - em["cost_plus"]).sum()
    rows = []
    for seg in SEGMENT_ORDER:
        mask = (test_df["customer_segment"] == seg).to_numpy()
        cp, mp = em["cost_plus"][mask], em["model_pricing"][mask]
        rows.append(
            {
                "segment": seg,
                "n_quotes": int(mask.sum()),
                "cost_plus_margin_per_quote_usd": round(cp.mean(), 2),
                "model_margin_per_quote_usd": round(mp.mean(), 2),
                "uplift_pct": round((mp.sum() / cp.sum() - 1) * 100, 1),
                "share_of_total_uplift_pct": round(
                    (mp.sum() - cp.sum()) / total_uplift * 100, 1
                ),
                "avg_model_multiplier": round(
                    float((prices["model_pricing"][mask] / cost[mask]).mean()), 3
                ),
                "cost_plus_multiplier": synthetic.COST_PLUS_MARKUP[seg],
            }
        )
    return pd.DataFrame(rows)


def _frontier(test_df: pd.DataFrame, model_prices: np.ndarray) -> pd.DataFrame:
    """Scale the model policy's prices up and down (inside the guardrails)
    and trace expected margin against expected win rate."""
    cost = test_df["our_cost_usd"].to_numpy()
    rows = []
    for s in FRONTIER_SCALES:
        p = np.clip(model_prices * s, cost * price.GUARDRAIL_FLOOR, cost * price.GUARDRAIL_CAP)
        p_true = synthetic.true_accept_prob(test_df, p)
        rows.append(
            {
                "scale": round(float(s), 3),
                "expected_win_rate": round(float(p_true.mean()), 4),
                "expected_margin_usd": round(float(((p - cost) * p_true).sum()), 2),
            }
        )
    return pd.DataFrame(rows)


def _model_quality(models: train.TrainedModels, test_df: pd.DataFrame) -> dict:
    """ROC-AUC of both acceptance models on the held-out period, at the
    historical prices (the only prices with real outcomes attached)."""
    hist = test_df[cleaning.PRICE_COL].to_numpy()
    y = test_df[cleaning.LABEL_COL].to_numpy()
    return {
        "logistic": round(
            float(roc_auc_score(y, train.predict_accept(models, test_df, hist, "baseline"))), 4
        ),
        "xgboost": round(
            float(roc_auc_score(y, train.predict_accept(models, test_df, hist, "xgb"))), 4
        ),
    }


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------


def _plot_comparison(comparison: pd.DataFrame, path: Path) -> None:
    """Expected margin and win rate per policy; the oracle as a ceiling line."""
    plot_df = comparison[comparison["policy"] != "oracle"]
    oracle = comparison[comparison["policy"] == "oracle"].iloc[0]
    colors = [POLICY_COLORS[p] for p in plot_df["policy"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    ax1.bar(plot_df["policy"], plot_df["expected_margin_usd"] / 1000, color=colors)
    ax1.axhline(
        oracle["expected_margin_usd"] / 1000,
        color="#c0392b",
        ls="--",
        lw=1.2,
        label=f"oracle: ${oracle['expected_margin_usd'] / 1000:,.0f}k",
    )
    for i, row in plot_df.reset_index().iterrows():
        ax1.text(
            i,
            row["expected_margin_usd"] / 1000,
            f"${row['expected_margin_usd'] / 1000:,.0f}k",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax1.set_ylabel("Expected margin (USD thousands)")
    ax1.set_title("Same quotes, only the pricing differs")
    ax1.legend(fontsize=8)

    ax2.bar(plot_df["policy"], plot_df["expected_win_rate"] * 100, color=colors)
    ax2.axhline(
        oracle["expected_win_rate"] * 100,
        color="#c0392b",
        ls="--",
        lw=1.2,
        label=f"oracle: {oracle['expected_win_rate']:.0%}",
    )
    for i, row in plot_df.reset_index().iterrows():
        ax2.text(
            i,
            row["expected_win_rate"] * 100,
            f"{row['expected_win_rate']:.0%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax2.set_ylabel("Expected win rate (%)")
    ax2.set_title("Margin is not bought with volume")
    ax2.legend(fontsize=8)
    for ax in (ax1, ax2):
        ax.tick_params(axis="x", labelrotation=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_frontier(frontier: pd.DataFrame, comparison: pd.DataFrame, path: Path) -> None:
    """The curve the pricing desk negotiates over: margin vs volume."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        frontier["expected_win_rate"] * 100,
        frontier["expected_margin_usd"] / 1000,
        marker="o",
        ms=3.5,
        color="#2b6cb0",
        label="model policy, prices scaled",
    )
    for name, marker in [("cost_plus", "s"), ("model_pricing", "*")]:
        row = comparison[comparison["policy"] == name].iloc[0]
        ax.scatter(
            row["expected_win_rate"] * 100,
            row["expected_margin_usd"] / 1000,
            marker=marker,
            s=140 if marker == "*" else 60,
            color=POLICY_COLORS[name],
            zorder=5,
            label=name,
        )
    ax.set_xlabel("Expected win rate (%)")
    ax.set_ylabel("Expected margin (USD thousands)")
    ax.set_title("Margin vs volume: the model draws the frontier")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _representative_quote(test_df: pd.DataFrame, segment: str) -> pd.DataFrame:
    """A one-row frame for the median test lane: neutral market, neutral
    competition, standard urgency, latent term at its mean (0)."""
    med = test_df[["distance_miles", "weight_lb", "volume_cuft"]].median()
    base = synthetic.base_rate(med["distance_miles"], med["weight_lb"], med["volume_cuft"])
    return pd.DataFrame(
        {
            "distance_miles": [med["distance_miles"]],
            "weight_lb": [med["weight_lb"]],
            "volume_cuft": [med["volume_cuft"]],
            "customer_segment": [segment],
            "urgency": ["standard"],
            "market_rate_index": [1.0],
            "fuel_index": [1.0],
            "competitor_pressure": [0.5],
            "our_cost_usd": [float(base)],
            "reference_price_usd": [float(base) * synthetic.REF_MARKUP],
            "latent_willingness": [0.0],
        }
    )


def _plot_elasticity(models: train.TrainedModels, test_df: pd.DataFrame, path: Path) -> None:
    """The model-validation money chart: TRUE demand curve vs MODEL-implied
    curve per segment, on one representative lane. If these disagree in
    slope, every price the optimizer picks is built on sand."""
    grid = price.multiplier_grid()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    for ax, seg in zip(axes, SEGMENT_ORDER):
        row = _representative_quote(test_df, seg)
        cost = row["our_cost_usd"].iloc[0]
        rep = pd.concat([row] * len(grid), ignore_index=True)
        prices = cost * grid
        p_true = synthetic.true_accept_prob(rep, prices)
        p_model = train.predict_accept(models, rep, prices)
        x = prices / row["reference_price_usd"].iloc[0] * 100
        ax.plot(x, p_true * 100, color="#2b6cb0", lw=2, label="true")
        ax.plot(x, p_model * 100, color="#e8a33d", lw=2, ls="--", label="model-implied")
        cp = synthetic.COST_PLUS_MARKUP[seg] * cost / row["reference_price_usd"].iloc[0] * 100
        ax.axvline(cp, color="#8d99ae", ls=":", lw=1.2, label="cost-plus price")
        b = synthetic.TRUE_ELASTICITY[seg]["slope"]
        ax.set_title(f"{seg} (true slope b={b:g})", fontsize=10)
        ax.set_xlabel("price as % of market rate")
    axes[0].set_ylabel("P(accept) %")
    axes[0].legend(fontsize=8)
    fig.suptitle("Demand curves: ground truth vs what the model learned", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
