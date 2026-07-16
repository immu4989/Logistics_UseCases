"""A year of freight-quote requests, with the demand curve a real log never contains.

Dynamic pricing lives or dies on one function nobody observes directly:
P(customer accepts | price). A real quote log records a single (price,
accepted) point per request — never the curve around it. This generator
writes the curve down explicitly (``TRUE_ELASTICITY``, ``true_accept_prob``)
so that:

1. pricing policies can be evaluated EXACTLY against the true acceptance
   probabilities (see ``evaluate.py``) — the counterfactual "what if we had
   quoted $200 less" that no historical log can answer;
2. the model-implied elasticity curves can be tested against ground truth in
   CI, the same trick use case 1 plays with SHAP and planted noise features.

The world, per quote request:

- a lane (``distance_miles``) and a load (``weight_lb``, ``volume_cuft``;
  billable weight is the max of actual and dimensional weight, the industry
  convention that stops feather-light bulky freight riding for free);
- ``our_cost_usd``        — what the desk pays to move it: a documented
  linehaul function (base + per-mile + per-billable-lb) times a fuel index
  that random-walks slowly over the year;
- ``reference_price_usd`` — the going market rate for the lane: the same
  base rate times ``REF_MARKUP``, times a market rate index that also
  random-walks (think DAT / Xeneta benchmark). Customers judge a quote
  against THIS, never against our cost, which they cannot see;
- ``customer_segment``    — spot / contract / premium, each with its own
  price elasticity (the point of the exercise);
- ``urgency``             — standard / express; an urgent shipper needs the
  freight moved more than they need it shopped;
- ``competitor_pressure`` — 0-1 proxy for how many rival quotes the shipper
  is holding (a real desk sees this in quote-request metadata and win/loss
  debriefs).

GROUND TRUTH acceptance model (inspired by the AI-based quote pricing
programs Maersk and DHL have described publicly):

    P(accept | price) = sigmoid( a_seg
                                 + URGENCY_BOOST * is_express
                                 - COMPETITOR_COEF * (competitor_pressure - 0.5)
                                 + latent_willingness
                                 - b_seg * ELASTICITY_SCALE * (price / reference_price - 1) )

- spot     (a=0.6, b=18.0) — pure price shoppers; quoting 10% over market
  costs ~1.4 logits of win probability. Highly elastic.
- contract (a=1.0, b=6.0)  — relationship freight; price matters, mildly.
- premium  (a=1.3, b=3.5)  — urgency-driven; they need it moved, not shopped.

``latent_willingness`` ~ Normal(0, 0.6) is everything the desk cannot see:
the shipper's alternatives, internal deadlines, how their morning went. The
TRUE model uses it; the trained model never sees it. That gives the trained
model an honest, irreducible gap to the oracle instead of a fake 100%.

HISTORICAL prices in the log were set by cost-plus with a per-segment markup
PLUS lognormal noise (reps rounding, haggling, ad-hoc discounts). That noise
is load-bearing: a perfectly disciplined cost-plus desk quotes identical
lanes identically, and a log like that contains no price variation at all —
elasticity is then unidentifiable no matter how many rows you have. The
noise here stands in for what a real desk gets from rep-to-rep variance, or
buys deliberately with a randomized-pricing / A-B program. If your own quote
log has no price variation, run that experiment before trusting any demand
model fitted to it.

``make_quotes(..., messy=True)`` injects the defects real quote extracts
have (duplicate quote ids, negative weights, impossible zero prices,
inconsistent segment casing) so ``cleaning.py`` has real work to do.
Deterministic given a seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- ground-truth acceptance model -----------------------------------------
# Exposed so tests and the elasticity-validation chart can compare the
# trained model's implied demand curves against reality.
TRUE_ELASTICITY = {
    #            a (intercept)  b (price slope)
    "spot": {"intercept": 0.6, "slope": 18.0},
    "contract": {"intercept": 1.0, "slope": 6.0},
    "premium": {"intercept": 1.3, "slope": 3.5},
}
ELASTICITY_SCALE = 1.0  # the global k multiplying every segment slope
URGENCY_BOOST = 0.6  # express shippers accept more readily at any price
COMPETITOR_COEF = 1.5  # each rival quote in hand makes ours easier to refuse
LATENT_SD = 0.6  # sd of the willingness term the desk can never observe

# --- cost and market-rate structure -----------------------------------------
COST_BASE_USD = 150.0  # pickup/delivery fixed handling
COST_PER_MILE = 1.6  # linehaul
COST_PER_BILLABLE_LB = 0.04
DIM_FACTOR_LB_PER_CUFT = 10.0  # dimensional-weight conversion
REF_MARKUP = 1.25  # the market typically clears ~25% over base cost

# The status-quo desk rule: cost times a fixed markup per segment. Note the
# desk charges SPOT the most (one-off shippers, no relationship to protect)
# — exactly backwards for the most price-elastic segment, which is where the
# margin left on the table concentrates.
COST_PLUS_MARKUP = {"spot": 1.35, "contract": 1.28, "premium": 1.40}
# Lognormal sd of rep-level deviation from the rule. Deliberately wide: it
# stands in for rep discretion PLUS a randomized-pricing pilot, and it is
# what makes the log cover the price range the optimizer will later sweep.
# Train on a tightly disciplined log (sd ~0.05) and the model has never seen
# a quote near 1.6x cost — it extrapolates a flat demand curve out there and
# the optimizer happily chases phantom margin into the cap. Coverage of the
# candidate price range is a data requirement, not a nicety.
QUOTE_NOISE_SD = 0.12

SEGMENT_PROBS = {"spot": 0.50, "contract": 0.35, "premium": 0.15}
EXPRESS_FRAC = 0.25


def base_rate(distance_miles, weight_lb, volume_cuft):
    """Lane base rate before fuel/market scaling. Billable weight = max(actual, dim)."""
    billable = np.maximum(weight_lb, volume_cuft * DIM_FACTOR_LB_PER_CUFT)
    return COST_BASE_USD + COST_PER_MILE * distance_miles + COST_PER_BILLABLE_LB * billable


def true_accept_prob(df: pd.DataFrame, prices: np.ndarray) -> np.ndarray:
    """The TRUE P(accept | price) for each quote in `df` at the given prices.

    This is the counterfactual oracle: policies are scored by calling it with
    prices the desk never actually quoted. Expects normalized segment/urgency
    strings (i.e. cleaned data) and the `latent_willingness` column.
    """
    a = df["customer_segment"].map({s: v["intercept"] for s, v in TRUE_ELASTICITY.items()})
    b = df["customer_segment"].map({s: v["slope"] for s, v in TRUE_ELASTICITY.items()})
    ratio = np.asarray(prices, dtype=float) / df["reference_price_usd"].to_numpy()
    logit = (
        a.to_numpy()
        + URGENCY_BOOST * (df["urgency"] == "express").to_numpy()
        - COMPETITOR_COEF * (df["competitor_pressure"].to_numpy() - 0.5)
        + df["latent_willingness"].to_numpy()
        - b.to_numpy() * ELASTICITY_SCALE * (ratio - 1.0)
    )
    return 1.0 / (1.0 + np.exp(-logit))


def make_quotes(
    n: int = 40_000,
    seed: int = 7,
    start_date: str = "2025-01-06",
    n_days: int = 364,
    messy: bool = False,
) -> pd.DataFrame:
    """Generate `n` quote requests over `n_days`. Deterministic given `seed`."""
    rng = np.random.default_rng(seed)

    # Market and fuel indices drift as slow random walks over the year. This
    # is why train/test must split on TIME: a random split would scatter the
    # same market regime across both sides and leak it.
    market_walk = np.cumsum(rng.normal(0, 0.004, n_days)) + 1.0
    fuel_walk = np.cumsum(rng.normal(0, 0.003, n_days)) + 1.0
    market_walk = market_walk.clip(0.80, 1.25)
    fuel_walk = fuel_walk.clip(0.85, 1.20)

    day = rng.integers(0, n_days, n)
    quote_date = pd.Timestamp(start_date) + pd.to_timedelta(day, unit="D")
    market_index = market_walk[day]
    fuel_index = fuel_walk[day]

    distance = rng.gamma(2.2, 220, n).clip(25, 2800)
    weight = np.exp(rng.normal(6.2, 1.0, n)).clip(50, 45_000)
    volume = (weight / rng.uniform(4, 18, n)).clip(1, 3500)  # implied density spread

    segments = list(SEGMENT_PROBS)
    segment = rng.choice(segments, n, p=[SEGMENT_PROBS[s] for s in segments])
    urgency = np.where(rng.random(n) < EXPRESS_FRAC, "express", "standard")
    competitor_pressure = rng.beta(2.0, 2.0, n)
    latent = rng.normal(0, LATENT_SD, n)

    base = base_rate(distance, weight, volume)
    cost = base * fuel_index
    reference = base * REF_MARKUP * market_index

    # Historical cost-plus quotes: rule markup times rep-level noise. The
    # noise is the only reason elasticity is learnable from this log.
    markup = pd.Series(segment).map(COST_PLUS_MARKUP).to_numpy()
    quoted = cost * markup * np.exp(rng.normal(0, QUOTE_NOISE_SD, n))

    df = pd.DataFrame(
        {
            "quote_id": [f"QTE{seed:02d}{i:08d}" for i in range(n)],
            "quote_date": quote_date,
            "distance_miles": distance.round(1),
            "weight_lb": weight.round(1),
            "volume_cuft": volume.round(2),
            "customer_segment": segment,
            "urgency": urgency,
            "market_rate_index": market_index.round(4),
            "fuel_index": fuel_index.round(4),
            "competitor_pressure": competitor_pressure.round(4),
            "our_cost_usd": cost.round(2),
            "reference_price_usd": reference.round(2),
            "latent_willingness": latent.round(4),
            "quoted_price_usd": quoted.round(2),
        }
    )
    p_accept = true_accept_prob(df, df["quoted_price_usd"].to_numpy())
    df["accepted"] = rng.binomial(1, p_accept)

    if messy:
        df = _inject_mess(df, rng)
    return df


def _inject_mess(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add the defects real quote extracts have. cleaning.py must undo all of this."""
    df = df.copy()

    # 1. Duplicate quote ids (the CRM re-exports a quote every time it is touched).
    dupes = df.sample(frac=0.01, random_state=int(rng.integers(1e6)))
    df = pd.concat([df, dupes], ignore_index=True)

    # 2. Negative weights (a tare-weight subtraction bug in the scale export).
    idx = df.sample(frac=0.005, random_state=int(rng.integers(1e6))).index
    df.loc[idx, "weight_lb"] = -rng.uniform(10, 500, len(idx)).round(1)

    # 3. Impossible zero prices (quotes abandoned mid-entry, saved anyway).
    idx = df.sample(frac=0.004, random_state=int(rng.integers(1e6))).index
    df.loc[idx, "quoted_price_usd"] = 0.0

    # 4. Inconsistent categorical casing / whitespace.
    idx = df.sample(frac=0.04, random_state=int(rng.integers(1e6))).index
    df.loc[idx, "customer_segment"] = df.loc[idx, "customer_segment"].str.upper()
    idx = df.sample(frac=0.03, random_state=int(rng.integers(1e6))).index
    df.loc[idx, "urgency"] = " " + df.loc[idx, "urgency"].str.title() + " "

    # 5. Missing operational fields.
    idx = df.sample(frac=0.03, random_state=int(rng.integers(1e6))).index
    df.loc[idx, "market_rate_index"] = np.nan
    idx = df.sample(frac=0.02, random_state=int(rng.integers(1e6))).index
    df.loc[idx, "volume_cuft"] = np.nan

    return df.sample(frac=1, random_state=int(rng.integers(1e6))).reset_index(drop=True)
