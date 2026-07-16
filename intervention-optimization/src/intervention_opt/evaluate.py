"""Counterfactual evaluation: every policy judged on the exact same day.

Because the generator exposes ``p_true`` and every intervention effect is a
known constant, outcomes can be simulated EXACTLY rather than estimated:

- Draw ONE uniform ``u`` per shipment and reuse it for every policy (common
  random numbers). A shipment misses under an action iff
  ``u < p_true * (1 - p_reduction)``. Reusing the draw does two things:
  comparisons are PAIRED, which removes the between-run luck that would
  otherwise dominate small policy differences, and effects are monotone —
  any miss a weak action prevents, a stronger action on the same shipment
  also prevents. Evaluate each policy on fresh draws instead and the
  policy ranking can flip run to run for no real reason.

- Every metric is measured against the ``do nothing`` baseline on the same
  draws: misses prevented, miss cost avoided, net savings (avoided minus
  spend), ROI, and share of the oracle's net savings.

The budget sweep reruns the EV-greedy policy across budgets on the same
draws, tracing the diminishing-returns curve that answers the CFO question
"what is the RIGHT daily budget", not just "was today's budget well spent".
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import interventions, policy

# Budgets for the diminishing-returns sweep (EV-greedy only).
SWEEP_BUDGETS = [0, 1_000, 2_000, 3_000, 4_000, 6_000, 8_000, 12_000, 16_000, 20_000]

ACTION_COLORS = {
    "do_nothing": "#c3cbd6",
    "notify_customer": "#5aa9d6",
    "reroute": "#e8a33d",
    "upgrade_service": "#c0392b",
}


def simulate(df: pd.DataFrame, decisions: pd.DataFrame, u: np.ndarray) -> dict:
    """Realize one day under a policy's decisions, on shared uniform draws `u`.

    Returns totals: spend, misses, realized miss cost. The caller differences
    these against the do-nothing baseline computed on the SAME `u`.
    """
    p_red = decisions["action"].map(
        {a.name: a.p_reduction for a in interventions.CATALOG}
    ).to_numpy()
    c_red = decisions["action"].map(
        {a.name: a.cost_reduction for a in interventions.CATALOG}
    ).to_numpy()
    cost = interventions.miss_cost(df["customer_tier"], df["declared_value_usd"])

    missed = u < df["p_true"].to_numpy() * (1 - p_red)
    realized_cost = float((missed * cost * (1 - c_red)).sum())
    return {
        "spend": float(decisions["spent"].sum()),
        "misses": int(missed.sum()),
        "realized_miss_cost": realized_cost,
    }


def evaluate_all(
    df: pd.DataFrame,
    budget: float = policy.DEFAULT_BUDGET_USD,
    seed: int = 7,
    out_dir: str | Path = "artifacts/reports",
) -> tuple[pd.DataFrame, dict]:
    """Run every policy, simulate paired outcomes, write metrics + plots.

    Returns (comparison frame, {policy name -> decision frame}).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # One uniform per shipment, shared by every policy and every sweep point.
    # Offset the seed stream so outcome draws never collide with generator or
    # policy randomness — the day's luck must not depend on the policy tried.
    u = np.random.default_rng(seed + 1_000_003).random(len(df))
    baseline = simulate(df, policy.policy_none(df, 0), u)

    decisions = {
        name: fn(df, budget, seed=seed) for name, fn in policy.POLICIES.items()
    }
    rows = []
    for name, dec in decisions.items():
        sim = simulate(df, dec, u)
        avoided = baseline["realized_miss_cost"] - sim["realized_miss_cost"]
        net = avoided - sim["spend"]
        rows.append(
            {
                "policy": name,
                "spend_usd": round(sim["spend"], 2),
                "misses_prevented": baseline["misses"] - sim["misses"],
                "miss_cost_avoided_usd": round(avoided, 2),
                "net_savings_usd": round(net, 2),
                "roi": round(net / sim["spend"], 3) if sim["spend"] > 0 else None,
            }
        )
    comparison = pd.DataFrame(rows)
    oracle_net = comparison.loc[comparison["policy"] == "oracle", "net_savings_usd"].iloc[0]
    comparison["pct_of_oracle"] = (comparison["net_savings_usd"] / oracle_net * 100).round(1)

    # Budget sweep: same day, same draws, EV-greedy at each budget level.
    sweep_budgets = sorted(set(SWEEP_BUDGETS) | {budget})
    sweep_rows = []
    for b in sweep_budgets:
        dec = policy.policy_ev_greedy(df, b, seed=seed)
        sim = simulate(df, dec, u)
        avoided = baseline["realized_miss_cost"] - sim["realized_miss_cost"]
        sweep_rows.append(
            {
                "budget_usd": b,
                "spend_usd": round(sim["spend"], 2),
                "net_savings_usd": round(avoided - sim["spend"], 2),
            }
        )
    sweep = pd.DataFrame(sweep_rows)

    # ---- persist -----------------------------------------------------------
    comparison.to_csv(out_dir / "policy_comparison.csv", index=False)
    decisions["expected_value_greedy"].to_csv(out_dir / "decisions.csv", index=False)
    metrics = {
        "n_shipments": int(len(df)),
        "budget_usd": float(budget),
        "seed": int(seed),
        "baseline_misses": baseline["misses"],
        "baseline_miss_cost_usd": round(baseline["realized_miss_cost"], 2),
        # NaN (the zero-spend policy's ROI) is not valid JSON; emit null.
        "policies": comparison.astype(object).where(comparison.notna(), None).to_dict(orient="records"),
        "budget_sweep": sweep.to_dict(orient="records"),
        "oracle_gap_usd": round(
            oracle_net
            - comparison.loc[
                comparison["policy"] == "expected_value_greedy", "net_savings_usd"
            ].iloc[0],
            2,
        ),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    _plot_comparison(comparison, out_dir / "policy_comparison.png")
    _plot_budget_sweep(sweep, budget, out_dir / "savings_vs_budget.png")
    _plot_decision_mix(df, decisions["expected_value_greedy"], out_dir / "decision_mix.png")
    return comparison, decisions


def _plot_comparison(comparison: pd.DataFrame, path: Path) -> None:
    """Net savings per policy; the oracle drawn as the ceiling, not a bar."""
    plot_df = comparison[comparison["policy"] != "oracle"]
    oracle_net = comparison.loc[comparison["policy"] == "oracle", "net_savings_usd"].iloc[0]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#8d99ae", "#8d99ae", "#e8a33d", "#2b6cb0"]
    ax.bar(plot_df["policy"], plot_df["net_savings_usd"], color=colors[: len(plot_df)])
    ax.axhline(oracle_net, color="#c0392b", ls="--", lw=1.2,
               label=f"oracle (perfect scores): ${oracle_net:,.0f}")
    ax.axhline(0, color="k", lw=0.8)
    for i, row in plot_df.reset_index().iterrows():
        va = "bottom" if row["net_savings_usd"] >= 0 else "top"
        ax.text(i, row["net_savings_usd"], f"${row['net_savings_usd']:,.0f}",
                ha="center", va=va, fontsize=9)
    ax.set_ylabel("Net savings (USD, one day)")
    ax.set_title("Same budget, same day, same shipments: only the policy differs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_budget_sweep(sweep: pd.DataFrame, default_budget: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sweep["budget_usd"], sweep["net_savings_usd"], marker="o", color="#2b6cb0")
    ax.axvline(default_budget, color="#c0392b", ls="--", lw=1.2,
               label=f"default budget ${default_budget:,.0f}")
    ax.set_xlabel("Daily intervention budget (USD)")
    ax.set_ylabel("Net savings (USD)")
    ax.set_title("EV-greedy: diminishing returns to budget")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_decision_mix(df: pd.DataFrame, decisions: pd.DataFrame, path: Path) -> None:
    """Stacked action counts by model-score decile: the policy's logic made visible.

    Reads left (safest decile) to right (riskiest). The picture to expect:
    do-nothing dominates the safe deciles, notifications blanket the cheap
    middle risk, and the expensive physical actions concentrate where risk
    AND value justify them.
    """
    decile = pd.qcut(df["p_hat"], 10, labels=False) + 1  # 1 = lowest score
    mix = (
        pd.DataFrame({"decile": decile, "action": decisions["action"]})
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=list(ACTION_COLORS), fill_value=0)
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bottom = np.zeros(len(mix))
    for action in ACTION_COLORS:
        ax.bar(mix.index, mix[action], bottom=bottom,
               color=ACTION_COLORS[action], label=action)
        bottom += mix[action].to_numpy()
    ax.set_xticks(mix.index)
    ax.set_xlabel("Model-score decile (10 = highest predicted risk)")
    ax.set_ylabel("Shipments")
    ax.set_title("What the EV-greedy policy chose, by risk decile")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
