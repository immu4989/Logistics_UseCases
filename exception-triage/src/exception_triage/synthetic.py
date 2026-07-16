"""Synthetic exception-ticket generator with a documented ground-truth routing process.

Why synthetic data? Carrier exception queues are proprietary, so this repo
ships with a generator whose *causal structure is known*. That gives you two
things:

1. The pipeline runs end-to-end with zero external downloads.
2. The per-queue SHAP analysis can be checked against ground truth: the
   drivers the model "discovers" for each queue should match the weights
   below, and the planted noise (csr_id, ticket-creation hour) should land at
   the bottom. The test suite asserts both, so a refactor that silently
   breaks explanations fails CI.

The generative process is a softmax over six queue scores. Each queue's score
is a base rate (calibrated to realistic queue imbalance) plus the weighted
terms in TRUE_RULES, plus Normal(0, 0.4) noise; the ticket's true queue is
*sampled* from the softmax, not argmaxed. That sampling is the point: a scan
gap during a weather event is usually hold_and_monitor but sometimes a
genuine misroute needing a dispatcher, an RTS label sometimes means a bad
address and sometimes a customer conversation. No model can score 99% here,
and none should — the achievable ceiling is in the mid-to-high 80s, which is
what real triage data looks like.

`make_dataset(..., messy=True)` additionally injects the defects every real
ticket extract has: duplicated ticket ids, impossible negative scan gaps,
inconsistent queue-name casing in the label column (the historical labels
come from three generations of CRM), and missing flag values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema

# ---------------------------------------------------------------------------
# Ground truth, exposed for the explainability tests.
#
# Keys are derived terms computed in _derived_terms(); weights are log-odds
# contributions to that queue's softmax score. Strong flag-driven terms
# (damage, failed address validation, customs scan, weather) create the
# high-purity tickets a confidence gate can safely automate; the moderate
# terms overlap and create the ambiguous middle that needs a human.
# ---------------------------------------------------------------------------
TRUE_RULES: dict[str, dict[str, float]] = {
    "address_correction": {
        "address_validation_failed": 7.6,   # the address is the problem
        "return_to_sender_flag": 1.5,       # RTS often traces back to a bad label
        "last_mile_with_attempt": 1.6,      # driver stood at the wrong door
    },
    "reroute": {
        "scan_gap_over_36h": 5.0,           # silent for 1.5+ days: likely misrouted
        "linehaul_scan": 1.0,               # went dark between facilities
        "hub_scan_moderate_gap": 0.5,       # sitting at a hub longer than a sort cycle
        "repeat_offender": 0.9,             # 2+ prior exceptions: something is off
    },
    "customs_docs": {
        # Gated: this queue is impossible for domestic shipments.
        "customs_scan": 6.6,                # last seen entering customs
        "high_declared_value": 0.7,         # high-value intl draws inspection
    },
    "damage_claims": {
        "damage_scan_flag": 8.8,            # a handler recorded damage; near-deterministic
    },
    "hold_and_monitor": {
        "weather_event_at_location": 8.0,   # weather delays self-resolve; don't touch
        "moderate_scan_gap": 4.2,           # 8-36h quiet: normal congestion, still moving
        "hub_or_linehaul_scan": 0.8,        # mid-network is where congestion lives
        "last_mile_scan": -1.2,             # already out for delivery; nothing to monitor
    },
    "customer_callback": {
        "attempts_exhausted": 3.6,          # 2+ failed attempts: need the customer
        "short_scan_gap": 3.6,              # scans are current -> customer-driven issue
        "premium_or_enterprise": 1.2,       # contract accounts get a call
        "return_to_sender_flag": 1.6,       # RTS also resolves via the customer
        "high_declared_value": 0.5,
        "evening_ticket": 0.03,             # planted near-noise: hour of day
    },
}

# Base scores calibrated (empirically, seed-stable) to the target queue mix:
# address ~24%, reroute ~20%, customs ~8% (intl only), damage ~7%,
# hold ~26%, callback ~15%.
BASE_LOGITS = {
    "address_correction": 0.92,
    "reroute": 0.71,
    "customs_docs": 1.61,
    "damage_claims": -0.69,
    "hold_and_monitor": -2.41,
    "customer_callback": -0.13,
}

LOGIT_NOISE_SD = 0.2


def _derived_terms(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Compute the named terms TRUE_RULES puts weights on, from raw features."""
    gap = df["scan_gap_hours"].to_numpy()
    loc = df["last_scan_location_type"].to_numpy()
    return {
        "address_validation_failed": df["address_validation_failed"].to_numpy(),
        "return_to_sender_flag": df["return_to_sender_flag"].to_numpy(),
        "last_mile_with_attempt": ((loc == "last_mile") & (df["delivery_attempt_count"] >= 1))
        .to_numpy()
        .astype(float),
        "scan_gap_over_36h": (gap > 36).astype(float),
        "moderate_scan_gap": ((gap > 8) & (gap <= 36)).astype(float),
        "short_scan_gap": (gap <= 8).astype(float),
        "linehaul_scan": (loc == "linehaul").astype(float),
        "hub_scan_moderate_gap": ((loc == "hub") & (gap > 8) & (gap <= 36)).astype(float),
        "hub_or_linehaul_scan": np.isin(loc, ["hub", "linehaul"]).astype(float),
        "last_mile_scan": (loc == "last_mile").astype(float),
        "repeat_offender": (df["prior_exceptions"] >= 2).to_numpy().astype(float),
        "customs_scan": (loc == "customs").astype(float),
        "high_declared_value": (df["declared_value_usd"] > 1500).to_numpy().astype(float),
        "damage_scan_flag": df["damage_scan_flag"].to_numpy(),
        "weather_event_at_location": df["weather_event_at_location"].to_numpy(),
        "attempts_exhausted": (df["delivery_attempt_count"] >= 2).to_numpy().astype(float),
        "premium_or_enterprise": df["customer_tier"].isin(["premium", "enterprise"])
        .to_numpy()
        .astype(float),
        "evening_ticket": (df["ticket_created_hour_of_day"] >= 18).to_numpy().astype(float),
    }


def queue_logits(df: pd.DataFrame, rng: np.random.Generator | None = None) -> np.ndarray:
    """True (noisy) queue scores for each ticket; columns follow schema.QUEUES."""
    terms = _derived_terms(df)
    n = len(df)
    logits = np.zeros((n, len(schema.QUEUES)))
    for j, queue in enumerate(schema.QUEUES):
        logits[:, j] = BASE_LOGITS[queue]
        for term, weight in TRUE_RULES[queue].items():
            logits[:, j] += weight * terms[term]
    # customs_docs cannot happen for domestic shipments.
    intl = df["is_international"].to_numpy().astype(bool)
    logits[~intl, schema.QUEUES.index("customs_docs")] = -30.0
    if rng is not None:
        logits += rng.normal(0, LOGIT_NOISE_SD, logits.shape)
    return logits


def make_dataset(
    n: int = 40_000,
    seed: int = 7,
    start_date: str = "2025-01-05",
    n_days: int = 182,
    messy: bool = False,
) -> pd.DataFrame:
    """Generate `n` exception tickets over ~6 months, in the canonical schema."""
    rng = np.random.default_rng(seed)

    day_offsets = rng.integers(0, n_days, n)
    created = pd.Timestamp(start_date) + pd.to_timedelta(day_offsets, unit="D")
    month = created.month.to_numpy()

    # Tickets get keyed during business hours mostly; hour is planted noise.
    hour = rng.choice(24, n, p=_hour_profile())
    csr_id = rng.integers(0, 40, n)

    is_intl = rng.binomial(1, 0.18, n)
    loc = np.where(
        is_intl == 1,
        rng.choice(schema.LOCATION_TYPES, n, p=[0.22, 0.18, 0.12, 0.48]),
        rng.choice(schema.LOCATION_TYPES, n, p=[0.38, 0.27, 0.35, 0.00]),
    )

    winter = np.isin(month, [1, 2, 12]).astype(int)
    weather = rng.binomial(1, 0.15 + 0.09 * winter)

    # Scan gap: routine congestion baseline plus a genuine misroute tail (a
    # mis-sorted parcel goes dark for days). The tail hits weather and
    # non-weather tickets alike, which is what makes a long gap during a
    # weather event genuinely ambiguous between hold and reroute.
    misroute_tail = rng.binomial(1, 0.16, n)
    gap = rng.gamma(2.0, 9.0, n) + misroute_tail * rng.gamma(2.5, 28.0, n)
    gap = gap.clip(0.5, 240.0)

    damage = rng.binomial(1, 0.075, n)
    # Address validation failures skew domestic, and rarely co-occur with a
    # damage scan (a damaged box gets a claims ticket before anyone re-keys
    # the address).
    p_addr = np.where(is_intl == 1, 0.06, 0.22) * np.where(damage == 1, 0.2, 1.0)
    addr_failed = rng.binomial(1, p_addr)
    prior = np.minimum(rng.poisson(0.35, n), 5)
    service = rng.choice(schema.SERVICE_LEVELS, n, p=[0.15, 0.25, 0.60])
    declared = rng.lognormal(3.6, 1.1, n).clip(1, 20_000)
    tier = rng.choice(schema.CUSTOMER_TIERS, n, p=[0.75, 0.18, 0.07])
    attempts = np.where(loc == "last_mile", np.minimum(rng.poisson(1.2, n), 3), 0)
    rts = rng.binomial(1, np.where(attempts >= 3, 0.20, 0.035))

    df = pd.DataFrame(
        {
            schema.ID_COL: [f"EXC{seed:02d}{i:08d}" for i in range(n)],
            schema.DATE_COL: created,
            "scan_gap_hours": gap.round(1),
            "prior_exceptions": prior,
            "declared_value_usd": declared.round(2),
            "delivery_attempt_count": attempts,
            "is_international": is_intl,
            "weather_event_at_location": weather,
            "address_validation_failed": addr_failed,
            "damage_scan_flag": damage,
            "return_to_sender_flag": rts,
            "last_scan_location_type": loc,
            "service_level": service,
            "customer_tier": tier,
            "ticket_created_hour_of_day": hour,
            "csr_id": csr_id,
        }
    )

    # Sample the true queue from the noisy softmax — not argmax. The sampling
    # is the irreducible ambiguity that keeps accuracy honest.
    logits = queue_logits(df, rng)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    draws = rng.random(n)
    queue_idx = (probs.cumsum(axis=1) < draws[:, None]).sum(axis=1)
    df[schema.LABEL_COL] = np.array(schema.QUEUES)[queue_idx]

    if messy:
        df = _inject_mess(df, rng)
    return df


def _hour_profile() -> np.ndarray:
    """Ticket-creation hour distribution: peaks over the business day."""
    weights = np.ones(24)
    weights[7:19] = 4.0
    weights[9:16] = 7.0
    return weights / weights.sum()


def _inject_mess(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add the defects real ticket extracts have. cleaning.py must undo all of this."""
    df = df.copy()

    # 1. Duplicate ticket ids (the CRM re-emits a ticket on every status touch).
    dupes = df.sample(frac=0.012, random_state=int(rng.integers(1e6)))
    df = pd.concat([df, dupes], ignore_index=True)

    # 2. Impossible negative scan gaps (clock skew between scan feed and CRM).
    idx = df.sample(frac=0.006, random_state=int(rng.integers(1e6))).index
    df.loc[idx, "scan_gap_hours"] = -rng.uniform(1, 48, len(idx)).round(1)

    # 3. Inconsistent queue-name casing in the label column. The historical
    #    labels come from three generations of CRM: one title-cased with
    #    spaces, one SHOUTED, one padded with whitespace.
    label = df[schema.LABEL_COL].copy()
    idx = df.sample(frac=0.06, random_state=int(rng.integers(1e6))).index
    label.loc[idx] = label.loc[idx].str.replace("_", " ").str.title()
    idx = df.sample(frac=0.04, random_state=int(rng.integers(1e6))).index
    label.loc[idx] = label.loc[idx].str.upper()
    idx = df.sample(frac=0.03, random_state=int(rng.integers(1e6))).index
    label.loc[idx] = " " + label.loc[idx] + " "
    df[schema.LABEL_COL] = label

    # 4. Missing flag values (the weather feed and address validator both
    #    time out sometimes; the integration writes NULL, not 0).
    for col, frac in [
        ("weather_event_at_location", 0.04),
        ("address_validation_failed", 0.03),
        ("scan_gap_hours", 0.02),
    ]:
        idx = df.sample(frac=frac, random_state=int(rng.integers(1e6))).index
        df.loc[idx, col] = np.nan

    return df.sample(frac=1, random_state=int(rng.integers(1e6))).reset_index(drop=True)
