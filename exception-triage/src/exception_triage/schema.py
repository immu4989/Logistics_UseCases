"""Canonical exception-ticket schema.

Every data source must produce a DataFrame with these columns before entering
the pipeline. All features are restricted to what a triage system can see at
ticket-creation time — scan history up to now, package attributes, customer
attributes. Nothing that only becomes known after a human works the ticket
(resolution notes, callback outcomes) is allowed in, because the whole point
is to route the ticket *before* anyone touches it.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Identifiers / bookkeeping (not used as model features)
# ---------------------------------------------------------------------------
ID_COL = "ticket_id"
DATE_COL = "ticket_created_date"  # date the exception ticket was opened

# ---------------------------------------------------------------------------
# Label: the resolution queue that actually closed the ticket
# ---------------------------------------------------------------------------
LABEL_COL = "resolution_queue"

QUEUES = [
    "address_correction",   # bad/incomplete address; fix and relabel
    "reroute",              # genuinely misrouted; dispatcher builds a new path
    "customs_docs",         # international, stuck on paperwork (intl only)
    "damage_claims",        # damage scan; evidence + claim workflow
    "hold_and_monitor",     # weather/congestion; self-resolves, don't touch it
    "customer_callback",    # needs the customer: attempts exhausted, RTS, VIP
]

# ---------------------------------------------------------------------------
# Features observable at ticket creation
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "scan_gap_hours",           # hours since the last tracking scan
    "prior_exceptions",         # earlier exception tickets on this same shipment
    "declared_value_usd",
    "delivery_attempt_count",   # delivery attempts logged so far
]

FLAG_FEATURES = [
    "is_international",
    "weather_event_at_location",   # active weather event at the last-scan location
    "address_validation_failed",   # label address failed validation at induction
    "damage_scan_flag",            # a handler recorded visible damage
    "return_to_sender_flag",       # label already marked RTS
]

CATEGORICAL_FEATURES = [
    "last_scan_location_type",  # hub / linehaul / last_mile / customs
    "service_level",            # overnight / two_day / ground
    "customer_tier",            # standard / premium / enterprise
]

# Planted noise: present in every real ticket extract, should carry ~zero
# routing signal. The explainability tests assert the model buries them.
NOISE_FEATURES = [
    "ticket_created_hour_of_day",
    "csr_id",                   # agent who keyed the ticket
]

FEATURE_COLS = NUMERIC_FEATURES + FLAG_FEATURES + CATEGORICAL_FEATURES + NOISE_FEATURES
ALL_COLS = [ID_COL, DATE_COL, *FEATURE_COLS, LABEL_COL]

LOCATION_TYPES = ["hub", "linehaul", "last_mile", "customs"]
SERVICE_LEVELS = ["overnight", "two_day", "ground"]
CUSTOMER_TIERS = ["standard", "premium", "enterprise"]


def validate(df, require_label: bool = True) -> None:
    """Raise ValueError if the DataFrame is missing schema columns."""
    required = set(FEATURE_COLS) | {ID_COL, DATE_COL}
    if require_label:
        required.add(LABEL_COL)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {sorted(missing)}")
