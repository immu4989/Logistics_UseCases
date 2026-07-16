"""Canonical hub-day volume schema.

The unit of prediction is a (hub, day) pair: how many parcels arrive at that
hub on that day. Every data source (the synthetic generator or your own WMS /
sortation extract) must produce a DataFrame with these columns before entering
the pipeline. There is deliberately no same-day operational telemetry in the
schema — a staffing forecast is consumed days in advance, so the model may
only see the calendar and the volume history (see README "lags only").
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------
HUB_COL = "hub_id"
DATE_COL = "date"

# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------
TARGET_COL = "parcel_volume"  # inbound parcels handled by the hub that day

# The 15 hubs of the synthetic network. Codes are airport-style because that
# is how every carrier actually names sort facilities.
HUBS = [
    "MEM", "IND", "ORD", "DFW", "ATL",
    "EWR", "LAX", "OAK", "CVG", "PHL",
    "SEA", "DEN", "MIA", "MSP", "CLT",
]

# Planted noise columns: carried through the whole pipeline so the
# explainability step has genuine negatives to bury (see synthetic.py).
NOISE_FEATURES = ["hub_paint_color_code", "moon_phase"]

ALL_COLS = [HUB_COL, DATE_COL, TARGET_COL, *NOISE_FEATURES]


def validate(df, require_target: bool = True) -> None:
    """Raise ValueError if the DataFrame is missing schema columns."""
    required = {HUB_COL, DATE_COL, *NOISE_FEATURES}
    if require_target:
        required.add(TARGET_COL)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {sorted(missing)}")
    unknown = set(df[HUB_COL].unique()) - set(HUBS)
    if unknown:
        raise ValueError(f"DataFrame contains unknown hubs: {sorted(unknown)}")
