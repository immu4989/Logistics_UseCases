"""Synthetic e-commerce order generator with a documented ground-truth return process.

Why synthetic data? Retailer order-and-returns data is proprietary, so this
repo ships with a generator whose *causal structure is known*. That gives you
two things:

1. The pipeline runs end-to-end with zero external downloads.
2. The SHAP analysis can be checked against ground truth: the drivers the
   model "discovers" should match the coefficients below, and the planted
   noise features should rank near the bottom. The test suite asserts both.

The generative process (log-odds of the order being returned):

    logit = CATEGORY_BASE                  (apparel/shoes high, electronics low)
          + 1.30 * is_bracket_buy          (~4x odds -- the apparel signature move:
                                            order 3 sizes, keep 1, return 2)
          + 0.50 * is_bracket_buy * shoes  (shoe sizing is the least trustworthy of all)
          + 0.35 * (num_sizes_ordered == 3)
          + 1.40 * deep_discount * fashion ("wasn't sure, but it was 50% off" is a fashion story)
          - 0.80 * deep_discount * electronics  (deep-discounted gadgets read as final-sale)
          + 3.20 * max(0, prior_return_rate - 0.4)  (habitual returners are a regime, not a slope)
          + 0.70 * is_bracket_buy * (prior_return_rate > 0.3)   (the serial bracketer)
          + 0.70 * fashion * max(0, ln(price) - 4.2)   (premium fashion = higher fit stakes)
          - 0.80 * electronics * (ln(price) - 4.95)    (cheap gadgets bounce; researched ones stay)
          + 0.10 * is_gift + 1.20 * is_gift * (electronics or home)  (the unwanted-gadget give-back)
          + 0.30 * size_limited * fashion  (out-of-stock size -> "close enough" substitution)
          + 0.08 * min(delivery_days_late, 10)  (late delivery misses the occasion)
          - 0.15 * express_shipping
          + 0.10 * marketplace channel
          + customer_propensity ~ Normal(0, 0.5)   (stable per-customer trait)
          + noise ~ Normal(0, 0.20)

    Note how much of this is thresholds and interactions rather than slopes:
    that is deliberate, and it is what real returns behaviour looks like
    (a deep discount means opposite things on a dress and on a laptop; price
    runs in opposite directions in fashion and electronics; return history
    only matters once it crosses into habit). It is also why the gradient-
    boosted model beats the linear baseline on this problem, where it tied
    on the near-additive process in delivery-commit-prediction.

`prior_return_rate` is computed CAUSALLY: orders are generated in date order,
and each order's history features see only that customer's strictly earlier
orders. Compute it globally (the classic mistake) and the feature contains the
label's own future -- `compute_customer_history` is exposed so tests can pin
this down on a constructed example.

`page_dwell_seconds` and `ad_campaign_id` carry zero true signal; they exist
so the explainability step has genuine negatives to rank below the real
drivers.

`make_dataset(..., messy=True)` additionally injects the defects every real
order extract has (duplicate order ids, negative prices, impossible >100%
discounts, inconsistent category casing) so `cleaning.py` has real work to do.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema

# Ground-truth log-odds contributions, exposed so tests and the explain step
# can compare the model's SHAP ranking against reality.
TRUE_DRIVERS = {
    "is_bracket_buy": 1.30,          # ~4x odds; the strongest apparel-returns predictor
    "prior_return_rate": 3.20,       # via the hinge above 0.4 (plus 0.7 x bracket)
    "product_category": 1.50,        # apparel/shoes vs electronics base-rate spread
    "discount_pct": 1.40,            # deep-discount (>=40%) step in fashion, -0.8 in electronics
    "is_gift": 1.20,                 # on electronics/home (unwanted gifts), 0.1 elsewhere
    "unit_price_usd": 0.70,          # log-price hinge in fashion, opposite slope in electronics
    "num_sizes_ordered": 0.35,       # the 3-size bracket rides on top of the flag
    "size_limited": 0.30,            # fashion only
    "delivery_days_late": 0.08,      # per day late; post-ship, so a driver but not a feature
    "express_shipping": -0.15,
    "channel": 0.10,                 # marketplace vs app/web
}
NOISE_FEATURES = ["page_dwell_seconds", "ad_campaign_id"]

# Per-category base log-odds, calibrated to ~18% overall return rate with the
# modifiers above: apparel/shoes land in the 25-35% band typical of fashion
# e-commerce, electronics stay in the single digits.
CATEGORY_BASE = {
    "apparel": -2.00,
    "shoes": -1.90,
    "electronics": -3.15,
    "home": -3.00,
    "beauty": -3.35,
}
CATEGORY_MIX = [0.34, 0.16, 0.18, 0.20, 0.12]

# Median price scale per category (lognormal parameters).
PRICE_PARAMS = {
    "apparel": (3.65, 0.55),
    "shoes": (4.25, 0.45),
    "electronics": (4.95, 0.80),
    "home": (3.90, 0.70),
    "beauty": (3.25, 0.50),
}

FASHION = ("apparel", "shoes")


def compute_customer_history(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute (prior_orders, prior_return_rate) causally from a finished table.

    For each order, history = that customer's orders STRICTLY earlier in
    (order_date, row-position) order. Row position breaks same-day ties, so a
    customer's second order of the day sees the first but never itself or
    anything later. This is the function the causality test drills into.
    """
    order = df.reset_index(drop=True)
    seq = order.sort_values(schema.DATE_COL, kind="stable").index.to_numpy()
    counts: dict = {}
    returns: dict = {}
    prior_orders = np.zeros(len(order), dtype=int)
    prior_rate = np.zeros(len(order), dtype=float)
    customers = order[schema.CUSTOMER_COL].to_numpy()
    labels = order[schema.LABEL_COL].to_numpy()
    for i in seq:
        c = customers[i]
        n = counts.get(c, 0)
        prior_orders[i] = n
        prior_rate[i] = returns.get(c, 0) / n if n else 0.0
        counts[c] = n + 1
        returns[c] = returns.get(c, 0) + int(labels[i])
    return pd.DataFrame(
        {"prior_orders": prior_orders, "prior_return_rate": prior_rate}, index=df.index
    )


def make_dataset(
    n: int = 50_000,
    seed: int = 7,
    start_date: str = "2024-11-04",
    n_days: int = 270,
    messy: bool = False,
) -> pd.DataFrame:
    """Generate `n` orders over `n_days` (~9 months), returning the canonical schema."""
    rng = np.random.default_rng(seed)

    day_offsets = rng.integers(0, n_days, n)
    order_date = pd.Timestamp(start_date) + pd.to_timedelta(day_offsets, unit="D")

    # Customers: a skewed base (a few heavy buyers, a long tail of one-timers)
    # so the causal history features have real variance to work with.
    n_customers = max(2_000, n // 4)
    cust_weights = rng.lognormal(0.0, 1.1, n_customers)
    cust_idx = rng.choice(n_customers, n, p=cust_weights / cust_weights.sum())
    customer_id = np.array([f"CUST{c:06d}" for c in cust_idx])
    # A stable per-customer return propensity: this is WHY prior_return_rate
    # predicts the future -- both are readouts of the same latent trait.
    propensity = rng.normal(0.0, 0.5, n_customers)[cust_idx]

    category = rng.choice(schema.CATEGORIES, n, p=CATEGORY_MIX)
    is_fashion = np.isin(category, FASHION).astype(int)

    mu = np.vectorize(lambda c: PRICE_PARAMS[c][0])(category)
    sigma = np.vectorize(lambda c: PRICE_PARAMS[c][1])(category)
    price = np.exp(rng.normal(mu, sigma)).clip(3, 4000)

    discount = np.where(rng.random(n) < 0.55, 0.0, rng.uniform(5, 65, n)).round(0)
    deep_discount = (discount >= 40).astype(int)

    size_limited = (is_fashion * rng.binomial(1, 0.18, n)).astype(int)

    # Bracket buying: same item, multiple sizes, one order. Fashion only, and
    # more likely when the shopper can't trust the size run in stock.
    bracket_p = np.where(size_limited == 1, 0.28, 0.16) * is_fashion
    is_bracket = rng.binomial(1, bracket_p)
    num_sizes = 1 + is_bracket * rng.choice([1, 2], n, p=[0.7, 0.3])

    channel = rng.choice(schema.CHANNELS, n, p=[0.42, 0.38, 0.20])
    is_marketplace = (channel == "marketplace").astype(int)

    # Gifts spike in the holiday window (late Nov - late Dec of the 9 months).
    holiday = (day_offsets >= 21) & (day_offsets < 50)
    is_gift = rng.binomial(1, np.where(holiday, 0.22, 0.05))

    express = rng.binomial(1, 0.22, n)
    promised_days = np.where(express == 1, 2, 4 + rng.integers(0, 3, n))

    # Post-ship reality: some orders arrive late, more so in the holiday crush.
    # This column drives returns below but is NEVER a model feature (see
    # schema.POST_SHIP_COLS); the pre-ship version of this signal is the
    # delivery-commit-prediction model's output.
    late = rng.random(n) < np.where(holiday, 0.28, 0.15)
    days_late = np.where(late, np.ceil(rng.gamma(1.6, 1.7, n)).clip(1, 14), 0).astype(int)

    page_dwell = rng.lognormal(4.0, 0.7, n).clip(3, 3600).round(0)
    ad_campaign = rng.integers(1, 9, n)

    is_electronics = (category == "electronics").astype(int)
    is_shoes = (category == "shoes").astype(int)
    gift_prone = np.isin(category, ("electronics", "home")).astype(int)
    base = np.vectorize(CATEGORY_BASE.get)(category)
    logit_static = (
        base
        + 1.30 * is_bracket
        + 0.50 * is_bracket * is_shoes
        + 0.35 * (num_sizes == 3)
        + 1.40 * deep_discount * is_fashion
        - 0.80 * deep_discount * is_electronics
        + 0.70 * is_fashion * np.maximum(0.0, np.log(price) - 4.2)
        - 0.80 * is_electronics * (np.log(price) - 4.95)
        + 0.10 * is_gift
        + 1.20 * is_gift * gift_prone
        + 0.30 * size_limited * is_fashion
        + 0.08 * np.minimum(days_late, 10)
        - 0.15 * express
        + 0.10 * is_marketplace
        + propensity
        + rng.normal(0, 0.20, n)
    )

    # Sequential pass in date order: each order's history features and label
    # depend only on that customer's strictly earlier orders.
    uniforms = rng.random(n)
    seq = np.lexsort((np.arange(n), day_offsets))
    counts = np.zeros(n_customers, dtype=int)
    ret_counts = np.zeros(n_customers, dtype=int)
    prior_orders = np.zeros(n, dtype=int)
    prior_rate = np.zeros(n, dtype=float)
    returned = np.zeros(n, dtype=int)
    for i in seq:
        c = cust_idx[i]
        prior_orders[i] = counts[c]
        pr = ret_counts[c] / counts[c] if counts[c] else 0.0
        prior_rate[i] = pr
        logit = (
            logit_static[i]
            + 3.20 * max(0.0, pr - 0.4)
            + 0.70 * is_bracket[i] * (pr > 0.3)
        )
        y = int(uniforms[i] < 1.0 / (1.0 + np.exp(-logit)))
        returned[i] = y
        counts[c] += 1
        ret_counts[c] += y

    df = pd.DataFrame(
        {
            schema.ID_COL: [f"ORD{seed:02d}{i:08d}" for i in range(n)],
            schema.DATE_COL: order_date,
            schema.CUSTOMER_COL: customer_id,
            "product_category": category,
            "unit_price_usd": price.round(2),
            "discount_pct": discount,
            "size_limited": size_limited,
            "prior_orders": prior_orders,
            "prior_return_rate": prior_rate.round(4),
            "first_time_buyer": (prior_orders == 0).astype(int),
            "channel": channel,
            "is_gift": is_gift,
            "express_shipping": express,
            "num_sizes_ordered": num_sizes,
            "is_bracket_buy": is_bracket,
            "promised_delivery_days": promised_days,
            "page_dwell_seconds": page_dwell,
            "ad_campaign_id": ad_campaign,
            "delivery_days_late": days_late,
            schema.LABEL_COL: returned,
        }
    )

    if messy:
        df = _inject_mess(df, rng)
    return df


def _inject_mess(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add the defects real order extracts have. cleaning.py must undo all of this."""
    df = df.copy()

    # 1. Duplicate order ids (the returns feed and the order feed both export the row).
    dupes = df.sample(frac=0.012, random_state=int(rng.integers(1e6)))
    df = pd.concat([df, dupes], ignore_index=True)

    # 2. Negative prices (refund lines mis-joined onto the order record).
    idx = df.sample(frac=0.005, random_state=int(rng.integers(1e6))).index
    df.loc[idx, "unit_price_usd"] = -df.loc[idx, "unit_price_usd"]

    # 3. Impossible discounts (>100%: a stacked-promo bug upstream).
    idx = df.sample(frac=0.004, random_state=int(rng.integers(1e6))).index
    df.loc[idx, "discount_pct"] = rng.uniform(110, 260, len(idx)).round(0)

    # 4. Inconsistent categorical casing / whitespace.
    idx = df.sample(frac=0.04, random_state=int(rng.integers(1e6))).index
    df.loc[idx, "product_category"] = df.loc[idx, "product_category"].str.upper()
    idx = df.sample(frac=0.03, random_state=int(rng.integers(1e6))).index
    df.loc[idx, "product_category"] = " " + df.loc[idx, "product_category"].str.title() + " "

    return df.sample(frac=1, random_state=int(rng.integers(1e6))).reset_index(drop=True)
