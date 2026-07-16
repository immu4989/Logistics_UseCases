"""The booking decision: newsvendor economics on top of a quantile forecast.

This module is the point of the whole use case. The forecast's job is to be
right about the demand *distribution*; the numbers below decide where on that
distribution to sit. No model choice, no feature, no metric in forecast.py
changes this arithmetic.

The cost structure of a week-ahead linehaul booking:

    COMMITTED_COST  $1,400  a trailer booked a week ahead, contract rate
    SPOT_COST       $2,300  a trailer-equivalent bought day-of when demand
                            exceeds the committed capacity
    SALVAGE           $350  recovered per unused committed trailer
                            (reassignment to another lane, partial
                            cancellation credit)

The newsvendor derivation, step by step:

    1. Consider the marginal trailer: should you book one MORE?
    2. If demand turns out to need it (probability 1 - F(b), where F is the
       demand CDF at the booking level b), that trailer saves you a spot buy:
       you pay $1,400 instead of $2,300. Underage cost of NOT having it:
           Cu = SPOT_COST - COMMITTED_COST = 2,300 - 1,400 = $900
    3. If demand does not need it (probability F(b)), you paid $1,400 to move
       air and recover only the $350 salvage. Overage cost of having it:
           Co = COMMITTED_COST - SALVAGE = 1,400 - 350 = $1,050
    4. Book the marginal trailer while its expected saving beats its expected
       waste:  Cu * (1 - F(b)) > Co * F(b).
    5. The break-even point is the critical fractile:
           F(b*) = Cu / (Cu + Co) = 900 / 1,950 ~= 0.462
    6. So the optimal booking is the demand quantile at q* — and because
       trailers are discrete while demand is continuous,
           book = ceil( Q_demand(q*) )
       is exactly the optimal integer booking (the smallest b with
       F(b) >= q*).

Note where q* landed: *below* the median. This quarter an empty trailer
($1,050) costs more than a spot cover ($900), so the right booking is the
46th percentile, not the P50 "best guess". When the spot market tightens to
$3,200 (the sensitivity scenario), Cu jumps to $1,800, q* rises to 0.632, and
the right booking moves ABOVE the median. The method adapts by re-running one
division; a planner habit does not adapt at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COMMITTED_COST_USD = 1_400.0
SPOT_COST_USD = 2_300.0
SALVAGE_USD = 350.0
SPOT_COST_TIGHT_USD = 3_200.0  # tight-market sensitivity scenario


def critical_fractile(spot_cost: float = SPOT_COST_USD) -> float:
    """q* = Cu / (Cu + Co); see the derivation in the module docstring."""
    cu = spot_cost - COMMITTED_COST_USD
    co = COMMITTED_COST_USD - SALVAGE_USD
    return cu / (cu + co)


# Policy name -> the bookings column it produces. Order matters: this is the
# order every report table prints in, habit first, oracle last.
POLICY_COLUMNS = {
    "book_last_year": "booked_last_year",
    "book_mean": "booked_mean",
    "newsvendor_model": "booked_newsvendor",
    "oracle": "booked_oracle",
}
# Extra columns used only by the tight-spot sensitivity scenario.
TIGHT_COLUMNS = {"newsvendor_model_retuned": "booked_newsvendor_tight",
                 "oracle": "booked_oracle_tight"}


def make_bookings(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn forecasts into integer trailer bookings, one column per policy.

    ``frame`` is the test-period feature frame carrying, per (lane, week):
    ``naive_seasonal`` (same week last year, the planner habit's number),
    the model quantiles ``p50``, ``q_base``, ``q_tight`` from forecast.py,
    and the ground-truth quantiles ``true_q_base``, ``true_q_tight``.

    Every policy books ``ceil`` of its chosen number — trailers are discrete —
    and the policies differ ONLY in which number they choose:

    - book_last_year:   last year's realized demand. The habit. It re-books
      one noisy draw, misses trend, and never heard of the cost structure.
    - book_mean:        the P50 forecast. A good model consumed the naive way.
    - newsvendor_model: the q* quantile forecast. Same model family as
      book_mean; the only difference is WHERE on the distribution it reads.
    - oracle:           the q* quantile of the TRUE demand distribution. The
      cost floor no forecast-based policy can beat in expectation.
    """
    out = frame.copy()
    out["booked_last_year"] = np.ceil(out["naive_seasonal"]).astype(int)
    out["booked_mean"] = np.ceil(out["p50"]).astype(int)
    out["booked_newsvendor"] = np.ceil(out["q_base"]).astype(int)
    out["booked_newsvendor_tight"] = np.ceil(out["q_tight"]).astype(int)
    out["booked_oracle"] = np.ceil(out["true_q_base"]).astype(int)
    out["booked_oracle_tight"] = np.ceil(out["true_q_tight"]).astype(int)
    return out


def cost_components(
    booked: np.ndarray, demand: np.ndarray, spot_cost: float = SPOT_COST_USD
) -> dict[str, np.ndarray | float]:
    """Cost one booking vector against demand realizations.

    ``booked`` is (n,); ``demand`` is (n,) or (n, n_reps). Per replication:

        total = COMMITTED * booked + SPOT * shortfall - SALVAGE * empty

    which decomposes into the three stacked-bar components the report plots:
    committed capacity that moved freight, empty capacity (net of salvage),
    and spot buys.
    """
    b = np.asarray(booked, dtype=float)[:, None]
    d = np.asarray(demand, dtype=float)
    if d.ndim == 1:
        d = d[:, None]
    short = np.maximum(d - b, 0.0)
    empty = np.maximum(b - d, 0.0)
    moved = np.minimum(b, d)
    return {
        "total_cost": (COMMITTED_COST_USD * b + spot_cost * short - SALVAGE_USD * empty).sum(axis=0),
        "committed_moved_cost": (COMMITTED_COST_USD * moved).sum(axis=0),
        "empty_cost": ((COMMITTED_COST_USD - SALVAGE_USD) * empty).sum(axis=0),
        "spot_cost": (spot_cost * short).sum(axis=0),
        "spot_teq": short.sum(axis=0),
        "empty_teq": empty.sum(axis=0),
        "service_level": moved.sum(axis=0) / d.sum(axis=0),
        "booked_trailers": float(b.sum()),
    }
