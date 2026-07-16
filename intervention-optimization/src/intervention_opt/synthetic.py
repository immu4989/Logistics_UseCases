"""One day of scored shipments, with the ground truth a real dataset can never give you.

This use case starts where a risk model ends. Every shipment already carries a
miss-probability score from an upstream model (use case 1 in this repo builds
exactly that model); the question is no longer "who is at risk" but "given a
finite daily ops budget, what do we DO about each one". Evaluating a decision
policy is harder than evaluating a classifier: you need to know what WOULD
have happened under the action you didn't take. Real logs never contain that
counterfactual, which is why intervention programs get judged on vibes.

Synthetic data fixes this. The generator emits, per shipment:

- ``p_true``     — the TRUE miss probability. Right-skewed mixture: ~90% of
                   shipments draw from Beta(1.2, 14) (mostly safe, median ~6%)
                   and ~10% from Beta(2, 4) (a troubled tail: bad lanes, late
                   pickups, weather). Overall mean ~10%, matching the miss
                   rate in use case 1.
- ``p_hat``      — what the ops team actually sees: a NOISY model score,
                   sigmoid(logit(p_true) + Normal(0, score_noise)). The
                   default ``score_noise=0.6`` yields rank correlation ~0.88
                   with p_true and ROC-AUC ~0.74 against realized misses
                   (even PERFECT scores only reach ~0.77 here — Bernoulli
                   outcome noise is irreducible). That is a decent production
                   OTP model, not an oracle. Turn the knob to see how policy
                   value degrades as the upstream model gets worse.
- ``customer_tier``      — standard / premium / contract. Tier drives the
                   miss-cost multiplier in ``interventions.py``: a missed
                   contract pallet triggers SLA penalties a missed standard
                   parcel does not.
- ``declared_value_usd`` — lognormal, tier-shifted (contract freight is
                   pallets, not envelopes). Feeds the value-at-risk term of
                   the miss cost.

Because ``p_true`` is known and every intervention's effect is a documented
constant (see ``interventions.CATALOG``), policies can be evaluated EXACTLY:
draw one uniform per shipment, and the same draw decides the outcome under
every candidate action (common random numbers, see ``evaluate.py``). No
uplift modeling, no assumptions, no vibes.

Deliberate design choice: ``p_true`` is statistically independent of tier and
value. Risk tells you nothing about worth, and worth tells you nothing about
risk. That independence is precisely what breaks the classic "flag the top-K
riskiest" ops rule and is the reason expected-value allocation earns its keep.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Mixture weights for true risk: most shipments are routine, a minority sit
# on troubled lanes. Calibrated to a ~10% overall miss rate.
TROUBLED_FRAC = 0.10
SAFE_BETA = (1.2, 14.0)  # median ~6% miss risk
TROUBLED_BETA = (2.0, 4.0)  # mean ~33% miss risk

# Customer mix and the lognormal(mu, sigma) of declared value per tier.
# Contract freight is pallets and B2B replenishment, hence the higher mu.
TIER_PROBS = {"standard": 0.70, "premium": 0.22, "contract": 0.08}
TIER_VALUE_MU = {"standard": 3.4, "premium": 4.0, "contract": 5.2}
VALUE_SIGMA = 1.1

DEFAULT_SCORE_NOISE = 0.6  # sd of logit-space noise; ~ROC-AUC 0.83 upstream model


def make_day(n: int = 20_000, seed: int = 7, score_noise: float = DEFAULT_SCORE_NOISE) -> pd.DataFrame:
    """Generate one operating day of `n` scored shipments. Deterministic given `seed`."""
    rng = np.random.default_rng(seed)

    troubled = rng.random(n) < TROUBLED_FRAC
    p_true = np.where(
        troubled,
        rng.beta(*TROUBLED_BETA, n),
        rng.beta(*SAFE_BETA, n),
    ).clip(1e-4, 1 - 1e-4)

    # The imperfect upstream model: noise in logit space keeps scores in (0,1)
    # and makes errors multiplicative in odds, which is how real model error
    # behaves (a model is rarely off by "+0.3 probability" uniformly).
    logit = np.log(p_true / (1 - p_true))
    p_hat = 1 / (1 + np.exp(-(logit + rng.normal(0, score_noise, n))))
    p_hat = p_hat.clip(1e-4, 1 - 1e-4)

    tiers = list(TIER_PROBS)
    tier = rng.choice(tiers, n, p=[TIER_PROBS[t] for t in tiers])
    mu = pd.Series(tier).map(TIER_VALUE_MU).to_numpy()
    declared_value = np.exp(rng.normal(mu, VALUE_SIGMA, n)).clip(1, 50_000)

    return pd.DataFrame(
        {
            "shipment_id": [f"SHP{seed:02d}{i:08d}" for i in range(n)],
            "customer_tier": tier,
            "declared_value_usd": declared_value.round(2),
            "p_true": p_true.round(6),
            "p_hat": p_hat.round(6),
        }
    )
