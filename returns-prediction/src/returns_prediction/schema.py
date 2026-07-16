"""Canonical order schema.

Every data source (the synthetic generator or your company's own order extract)
must produce a DataFrame with these columns before entering the pipeline.

The leakage rule for this use case: **a model feature must be knowable at order
time**, the moment the customer clicks "buy" and the only moment a pre-ship
intervention (fit assistant, size nudge, packaging hold) is still possible.
Anything observed after the ship scan lives in POST_SHIP_COLS: it may drive the
label in the generator's ground truth, but it never enters the model matrix,
and a test enforces that.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Identifiers / bookkeeping (not used as model features)
# ---------------------------------------------------------------------------
ID_COL = "order_id"
DATE_COL = "order_date"  # date the order was placed (the decision point)
CUSTOMER_COL = "customer_id"

# ---------------------------------------------------------------------------
# Label
# ---------------------------------------------------------------------------
LABEL_COL = "returned"  # 1 = the order came back through reverse logistics

# ---------------------------------------------------------------------------
# Features known at order time
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "unit_price_usd",        # item price before discount
    "discount_pct",          # 0-100, depth of the applied promotion
    "prior_orders",          # this customer's completed orders BEFORE this one
    "prior_return_rate",     # share of those prior orders that came back (0 if none)
    "num_sizes_ordered",     # sizes of the same item in this order (bracket buying >= 2)
    "promised_delivery_days",  # the delivery promise shown at checkout
    "page_dwell_seconds",    # time on the product page (planted noise -- see synthetic.py)
    "ad_campaign_id",        # acquisition campaign code (planted noise)
]

CATEGORICAL_FEATURES = [
    "product_category",      # apparel / shoes / electronics / home / beauty
    "channel",               # app / web / marketplace
    "size_limited",          # 1 = size run partially out of stock at order time
    "first_time_buyer",      # 1 = no prior order history
    "is_gift",               # buyer marked the order as a gift
    "express_shipping",      # paid expedited shipping
    "is_bracket_buy",        # same item, multiple sizes, one order
]

# ---------------------------------------------------------------------------
# Observed only after the ship scan -- NEVER model features.
# `delivery_days_late` drives returns in the ground truth (late gifts miss the
# occasion, late apparel misses the event), but at order time it hasn't
# happened yet. The pre-ship signal for it is the *predicted* lateness from
# the delivery-commit-prediction use case in this repo; see the README.
# ---------------------------------------------------------------------------
POST_SHIP_COLS = ["delivery_days_late"]

FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
ALL_COLS = [ID_COL, DATE_COL, CUSTOMER_COL, *FEATURE_COLS, *POST_SHIP_COLS, LABEL_COL]

CATEGORIES = ["apparel", "shoes", "electronics", "home", "beauty"]
CHANNELS = ["app", "web", "marketplace"]


def validate(df, require_label: bool = True) -> None:
    """Raise ValueError if the DataFrame is missing schema columns."""
    required = set(FEATURE_COLS) | {ID_COL, DATE_COL, CUSTOMER_COL}
    if require_label:
        required.add(LABEL_COL)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {sorted(missing)}")
