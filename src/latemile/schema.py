"""Canonical shipment schema.

Every data source (synthetic generator, Olist adapter, or your company's own
extract) must produce a DataFrame with these columns before entering the
pipeline. All features are restricted to information available at induction
time (when the package enters the network) so the model can score shipments
*before* they are late — see README "Avoiding label leakage".
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Identifiers / bookkeeping (not used as model features)
# ---------------------------------------------------------------------------
ID_COL = "shipment_id"
DATE_COL = "ship_date"  # date the shipment entered the network

# ---------------------------------------------------------------------------
# Label
# ---------------------------------------------------------------------------
LABEL_COL = "missed_commit"  # 1 = delivered after the committed date/time

# ---------------------------------------------------------------------------
# Features known at induction time
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "distance_miles",          # origin -> destination great-circle distance
    "package_weight_lb",
    "package_volume_cuft",
    "declared_value_usd",
    "minutes_after_cutoff",    # pickup time relative to origin-facility cutoff (<=0 means on time)
    "origin_hub_congestion",   # 0-1 utilisation of the origin sort facility that day
    "dest_hub_congestion",     # 0-1 utilisation of the destination sort facility
    "dest_weather_severity",   # 0 (clear) .. 3 (severe) forecast at destination
    "route_stop_density",      # planned stops per mile on the delivery route
    "promised_transit_days",   # commit window length in days
]

CATEGORICAL_FEATURES = [
    "service_level",           # e.g. overnight / two_day / ground
    "origin_region",
    "dest_region",
    "dest_type",               # residential / commercial
    "day_of_week",             # ship-date day of week, 0=Mon
    "is_peak_season",          # holiday-peak indicator
    "is_rural_dest",
    "signature_required",
]

FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
ALL_COLS = [ID_COL, DATE_COL, *FEATURE_COLS, LABEL_COL]

SERVICE_LEVELS = ["overnight", "two_day", "ground"]
REGIONS = ["northeast", "southeast", "midwest", "southwest", "west"]
DEST_TYPES = ["residential", "commercial"]


def validate(df, require_label: bool = True) -> None:
    """Raise ValueError if the DataFrame is missing schema columns."""
    required = set(FEATURE_COLS) | {ID_COL, DATE_COL}
    if require_label:
        required.add(LABEL_COL)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {sorted(missing)}")
