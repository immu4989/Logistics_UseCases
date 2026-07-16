"""The action catalog and the miss-cost model. Every dollar constant lives here.

These numbers are BUSINESS constants, not modeling choices. Before running
this on your own network, replace each one with a figure defended from your
own claims, refund, WISMO-call and SLA-penalty data — the optimizer is only
as honest as this file. The structure to keep:

- Each action has a unit cost and a multiplicative effect. Two distinct
  effect channels matter operationally:

  * ``p_reduction``    — the action makes the miss LESS LIKELY (reroute,
    service upgrade physically change the shipment's path).
  * ``cost_reduction`` — the action makes the miss CHEAPER when it still
    happens. Proactively notifying the customer does not save the parcel,
    but a warned customer files fewer WISMO calls, fewer chargebacks and
    fewer refund claims. Cheap failures are a real lever and most
    top-K-flagging programs ignore it entirely.

- The cost of a missed commitment is not one number. A miss on a contract
  pallet triggers SLA penalties; a premium customer churns harder than a
  standard one; and part of the pain scales with what's in the box.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Miss-cost model: what one missed commitment costs the business.
#   cost = BASE * tier_multiplier + min(VALUE_FRACTION * declared_value, CAP)
# BASE covers the tier-independent overhead of a miss (support contact,
# goodwill credit, re-delivery attempt). The value term captures
# claims/refund exposure, capped because insurance takes over above the cap.
# ---------------------------------------------------------------------------
BASE_MISS_COST_USD = 30.0
TIER_COST_MULTIPLIER = {"standard": 1.0, "premium": 3.0, "contract": 6.0}
VALUE_AT_RISK_FRACTION = 0.02
VALUE_AT_RISK_CAP_USD = 200.0


@dataclass(frozen=True)
class Intervention:
    """One action the ops desk can take on a shipment, priced per unit."""

    name: str
    cost_usd: float          # marginal cost of applying the action once
    p_reduction: float       # multiplicative cut in miss probability (0..1)
    cost_reduction: float    # multiplicative cut in realized miss cost (0..1)
    capacity_note: str       # operational constraint to model when you go real

    @property
    def effect(self) -> float:
        """Fraction of EXPECTED miss cost the action removes.

        Expected miss cost under the action is
        ``p * (1 - p_reduction) * cost * (1 - cost_reduction)``, so the two
        channels combine into a single number the optimizer can rank on.
        """
        return 1.0 - (1.0 - self.p_reduction) * (1.0 - self.cost_reduction)


CATALOG = [
    Intervention(
        "do_nothing", 0.0, 0.0, 0.0,
        "unlimited; the correct action for most shipments most days",
    ),
    Intervention(
        # $0.50 covers SMS/email delivery plus the amortized cost of the
        # notification service. 40% cost cut is the "warned customer" effect:
        # fewer WISMO calls, fewer refunds — calibrate from your CX data.
        "notify_customer", 0.50, 0.0, 0.40,
        "effectively unlimited, but notification fatigue caps sensible volume",
    ),
    Intervention(
        # Move the parcel to a healthier lane or an earlier linehaul; 45%
        # of misses avoided reflects that rerouting fixes routing problems
        # but not, e.g., destination weather.
        "reroute", 4.0, 0.45, 0.0,
        "limited by alternate-lane capacity; model as a per-lane cap in production",
    ),
    Intervention(
        # Bump to a faster service (ground -> express). Strongest lever,
        # priced at the internal transfer cost of the faster product.
        "upgrade_service", 15.0, 0.80, 0.0,
        "limited by express network capacity, especially in peak season",
    ),
]
BY_NAME = {a.name: a for a in CATALOG}
DO_NOTHING = BY_NAME["do_nothing"]
UPGRADE = BY_NAME["upgrade_service"]
MIN_ACTION_COST = min(a.cost_usd for a in CATALOG if a.cost_usd > 0)


def miss_cost(tier: pd.Series, declared_value_usd: pd.Series) -> np.ndarray:
    """Dollar cost of one missed commitment, per shipment (vectorized)."""
    multiplier = tier.map(TIER_COST_MULTIPLIER).to_numpy(dtype=float)
    value_at_risk = np.minimum(
        VALUE_AT_RISK_FRACTION * declared_value_usd.to_numpy(dtype=float),
        VALUE_AT_RISK_CAP_USD,
    )
    return BASE_MISS_COST_USD * multiplier + value_at_risk
