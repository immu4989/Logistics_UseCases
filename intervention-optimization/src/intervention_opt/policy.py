"""Budget-constrained decision policies, from naive to oracle.

All policies answer the same question — "spend at most $B today, on which
shipments, doing what" — and all return the same per-shipment decision frame,
so ``evaluate.py`` can compare them on identical counterfactual ground truth.

The ladder, in the order an ops org usually climbs it:

- ``none``       — do nothing. The baseline every dollar must beat.
- ``random``     — spend the budget with no targeting (upgrade random
  shipments until broke). Sounds like a strawman; it is the implicit policy
  of "the supervisor expedites whatever lands on their desk".
- ``top_k_risk`` — the classic ops rule: sort by risk score, upgrade the
  scariest shipments until the budget is gone. Better, but it optimizes the
  wrong quantity: a 90% risk on a $5 envelope is worth less than a 40% risk
  on a $2,000 contract pallet, and it only knows one action.
- ``expected_value_greedy`` — the point of this use case. Price every
  (shipment, action) pair by expected net savings, then fill the budget by
  bang-per-buck. Risk is an input, not the objective.
- ``oracle``     — the same greedy allocator fed the TRUE miss probability
  instead of the model score. Unreachable in production, and exactly the
  point: the oracle-minus-greedy gap is the dollar value of a better
  upstream model, which is how you budget the data-science roadmap.

Everything is plain numpy/pandas and deterministic given a seed. No solver:
at these unit costs the greedy knapsack heuristic is within noise of the
exact optimum, and an ops team can read it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import interventions

DEFAULT_BUDGET_USD = 6_000.0

_ACTION_COSTS = np.array([a.cost_usd for a in interventions.CATALOG])
_ACTION_NAMES = [a.name for a in interventions.CATALOG]


def ev_matrix(df: pd.DataFrame, prob_col: str = "p_hat") -> pd.DataFrame:
    """Expected net savings (USD) of every action for every shipment.

    For action a:  EV = miss_cost * p * effect(a) - cost(a).
    ``do_nothing`` is always exactly 0 — the column every other action must beat.
    """
    cost = interventions.miss_cost(df["customer_tier"], df["declared_value_usd"])
    p = df[prob_col].to_numpy()
    return pd.DataFrame(
        {a.name: cost * p * a.effect - a.cost_usd for a in interventions.CATALOG},
        index=df.index,
    )


def _frame(df: pd.DataFrame, actions, expected_savings, spent) -> pd.DataFrame:
    """Standard per-shipment decision frame; one row per shipment, df order."""
    return pd.DataFrame(
        {
            "shipment_id": df["shipment_id"].to_numpy(),
            "action": actions,
            "expected_savings": np.round(expected_savings, 4),
            "spent": spent,
        }
    )


def policy_none(df: pd.DataFrame, budget: float, seed: int = 0) -> pd.DataFrame:
    """Do nothing. Net savings is zero by construction; everything is measured against this."""
    n = len(df)
    return _frame(df, np.full(n, "do_nothing"), np.zeros(n), np.zeros(n))


def policy_random(df: pd.DataFrame, budget: float, seed: int = 0) -> pd.DataFrame:
    """Upgrade uniformly random shipments until the budget runs out.

    Untargeted spend: the control that prices what targeting itself is worth.
    """
    rng = np.random.default_rng(seed)
    k = min(int(budget // interventions.UPGRADE.cost_usd), len(df))
    chosen = rng.permutation(len(df))[:k]
    return _upgrade_frame(df, chosen)


def policy_top_k_risk(df: pd.DataFrame, budget: float, seed: int = 0) -> pd.DataFrame:
    """Sort by model score, upgrade the riskiest shipments until the budget runs out.

    The rule most networks actually run. Its two blind spots: it ignores what
    a miss costs (risk != value at risk), and it knows only one, most
    expensive, action.
    """
    k = min(int(budget // interventions.UPGRADE.cost_usd), len(df))
    chosen = np.argsort(-df["p_hat"].to_numpy(), kind="stable")[:k]
    return _upgrade_frame(df, chosen)


def _upgrade_frame(df: pd.DataFrame, chosen: np.ndarray) -> pd.DataFrame:
    """Decision frame for upgrade-only policies (random / top-k)."""
    n = len(df)
    upgrade = interventions.UPGRADE
    ev = ev_matrix(df)[upgrade.name].to_numpy()
    actions = np.full(n, "do_nothing", dtype=object)
    spent = np.zeros(n)
    expected = np.zeros(n)
    actions[chosen] = upgrade.name
    spent[chosen] = upgrade.cost_usd
    expected[chosen] = ev[chosen]
    return _frame(df, actions, expected, spent)


def policy_ev_greedy(
    df: pd.DataFrame,
    budget: float,
    seed: int = 0,
    prob_col: str = "p_hat",
) -> pd.DataFrame:
    """Expected-value greedy: fill the budget by savings-per-dollar.

    1. For each shipment, pick its best action: the one with the highest
       expected net savings, and only if that EV is strictly positive. A
       negative-EV action is never taken — even with budget left over,
       spending $15 to protect $8 of expected loss destroys money.
    2. Rank the surviving (shipment, best action) pairs by EV per dollar of
       action cost (the classic bang-per-buck knapsack heuristic) and fund
       them in order until the budget can't cover the cheapest action.

    Why per-dollar and not raw EV: with a binding budget, the scarce resource
    is dollars, and two $0.50 notifications each saving $3 beat one $4
    reroute saving $5. Step 1's max-EV choice per shipment keeps the big
    upgrades competitive when they dominate the shipment's alternatives.

    The oracle policy is this exact function with ``prob_col="p_true"`` —
    same allocator, better information — so the oracle gap isolates model
    quality from everything else.
    """
    ev = ev_matrix(df, prob_col=prob_col).to_numpy()
    best_action = ev.argmax(axis=1)
    best_ev = ev.max(axis=1)
    action_cost = _ACTION_COSTS[best_action]

    # Candidates: strictly positive EV and an action that actually costs money
    # (do_nothing can never be "funded"). Stable sort keeps ties deterministic.
    candidates = np.flatnonzero((best_ev > 0) & (action_cost > 0))
    order = candidates[np.argsort(-(best_ev[candidates] / action_cost[candidates]), kind="stable")]

    n = len(df)
    actions = np.full(n, "do_nothing", dtype=object)
    spent = np.zeros(n)
    expected = np.zeros(n)
    remaining = float(budget)
    for i in order:
        if remaining < interventions.MIN_ACTION_COST:
            break  # can't afford even the cheapest action; stop scanning
        if action_cost[i] <= remaining:
            remaining -= action_cost[i]
            actions[i] = _ACTION_NAMES[best_action[i]]
            spent[i] = action_cost[i]
            expected[i] = best_ev[i]
        # else: skip — a cheaper candidate further down may still fit
    return _frame(df, actions, expected, spent)


def policy_oracle(df: pd.DataFrame, budget: float, seed: int = 0) -> pd.DataFrame:
    """EV-greedy on the TRUE miss probability: the perfect-model upper bound."""
    return policy_ev_greedy(df, budget, seed=seed, prob_col="p_true")


# Registry, in the order results should be reported.
POLICIES = {
    "none": policy_none,
    "random": policy_random,
    "top_k_risk": policy_top_k_risk,
    "expected_value_greedy": policy_ev_greedy,
    "oracle": policy_oracle,
}
