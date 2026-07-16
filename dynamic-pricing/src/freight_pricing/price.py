"""Pricing policies, from the status quo to the perfect-information ceiling.

All policies answer the same question — "what do we quote for THIS request"
— and all return one price per quote, so ``evaluate.py`` can compare them on
identical counterfactual ground truth. The ladder, in the order a pricing
desk usually climbs it:

- ``cost_plus``     — the status quo: cost times a fixed per-segment markup.
  Every freight desk starts here, most never leave.
- ``flat_optimal``  — one global markup, chosen to maximize the model's
  expected margin over the training period. The best a desk can do with a
  single knob; still blind to which customer is on the other end.
- ``model_pricing`` — the point of this use case. For each quote, sweep a
  grid of candidate prices, predict P(accept) at each, and quote the price
  maximizing expected margin = (price - cost) x P_hat(accept). Elastic spot
  freight gets priced down to win; urgency-driven premium gets priced up.
- ``oracle``        — the identical sweep fed the TRUE acceptance
  probabilities (latent willingness and all). Unreachable in production, and
  exactly the point: the oracle-minus-model gap is what acceptance-model
  error costs in dollars, which is how you budget the model roadmap.

Guardrails: every optimized quote is clamped to [1.02x, 1.60x] cost.
Unconstrained revenue optimization against an imperfect model produces
absurd quotes — the model is least trustworthy exactly where the optimizer
wants to go, at prices far from anything in the training data — and every
real pricing desk runs floor/cap guardrails for that reason. The floor also
bans loss-leaders outright: quoting below cost to buy share is a commercial
strategy someone signs off on, never an optimization output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import synthetic, train

GUARDRAIL_FLOOR = 1.02  # never quote below cost + 2%
GUARDRAIL_CAP = 1.60  # never quote above 1.6x cost
N_GRID = 30


def multiplier_grid() -> np.ndarray:
    """Candidate price multipliers over cost. The grid IS the guardrail:
    optimized policies can only choose prices inside it."""
    return np.linspace(GUARDRAIL_FLOOR, GUARDRAIL_CAP, N_GRID)


def policy_cost_plus(df: pd.DataFrame) -> np.ndarray:
    """The status quo, as a clean rule (the historical log adds rep noise on
    top of this; the policy being compared is the rule itself)."""
    markup = df["customer_segment"].map(synthetic.COST_PLUS_MARKUP).to_numpy()
    return df["our_cost_usd"].to_numpy() * markup


def _expected_margin_by_multiplier(
    df: pd.DataFrame, prob_fn, grid: np.ndarray
) -> np.ndarray:
    """(n_quotes, n_grid) expected margin: (price - cost) x P(accept)."""
    cost = df["our_cost_usd"].to_numpy()
    out = np.empty((len(df), len(grid)))
    for j, m in enumerate(grid):
        prices = cost * m
        out[:, j] = (prices - cost) * prob_fn(df, prices)
    return out


def choose_flat_multiplier(models: train.TrainedModels, fit_df: pd.DataFrame) -> float:
    """One global markup maximizing model-expected margin on the TRAINING
    period. Chosen on train, applied to test — the flat policy gets no peek
    at the evaluation period either."""
    grid = multiplier_grid()

    def prob_fn(d, p):
        return train.predict_accept(models, d, p)

    ev = _expected_margin_by_multiplier(fit_df, prob_fn, grid)
    return float(grid[ev.mean(axis=0).argmax()])


def policy_flat(df: pd.DataFrame, multiplier: float) -> np.ndarray:
    """Apply one global markup to every quote."""
    return df["our_cost_usd"].to_numpy() * multiplier


def policy_model_pricing(df: pd.DataFrame, models: train.TrainedModels) -> np.ndarray:
    """Per-quote price sweep against the TRAINED acceptance model.

    argmax over the grid of (price - cost) x P_hat(accept). Because the grid
    starts at 1.02x cost, expected margin is non-negative by construction —
    the policy never buys a win at a loss.
    """
    grid = multiplier_grid()

    def prob_fn(d, p):
        return train.predict_accept(models, d, p)

    ev = _expected_margin_by_multiplier(df, prob_fn, grid)
    best = ev.argmax(axis=1)
    return df["our_cost_usd"].to_numpy() * grid[best]


def policy_oracle(df: pd.DataFrame) -> np.ndarray:
    """The same sweep with the TRUE acceptance probabilities: the ceiling
    that prices model error. Same grid, same guardrails — the only thing the
    oracle has that the model doesn't is perfect information."""
    grid = multiplier_grid()
    ev = _expected_margin_by_multiplier(df, synthetic.true_accept_prob, grid)
    best = ev.argmax(axis=1)
    return df["our_cost_usd"].to_numpy() * grid[best]
