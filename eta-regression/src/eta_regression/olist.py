"""Adapter: Olist Brazilian e-commerce dataset -> canonical eta_regression schema.

Olist is the best public proxy for transit-time regression: ~96k real delivered
orders with a purchase timestamp and a customer-delivery timestamp, so the
``actual_transit_days`` label is real, not simulated. Median transit is ~10
days across a continent-sized network — a very different regime from the
synthetic parcel network, which is exactly the point of running both.

Download (free Kaggle account required):
    kaggle datasets download -d olistbr/brazilian-ecommerce -p data/olist --unzip
or from https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

Then:
    eta-regression all --source olist --olist-dir data/olist

Heads-up when evaluating on this dataset: Brazil's May 2018 truckers' strike
sits near the end of the data, right at the default train/test boundary, and
Olist inflated its promised delivery windows after it. The promise windows are
a *feature* here (they are known at purchase time), and actual transit times
may shift regime too — trucking capacity was disrupted for weeks, then
normalised. Do not tune that away: report whatever the monthly coverage
numbers show. See the README's Olist section.

Mapping notes — this file is the template for adapting *your* company's data:
some canonical columns have no Olist equivalent (hub congestion, weather).
They are filled with neutral constants, and the model simply learns they carry
no signal here. When you write your own adapter, populate whatever you have
and leave the rest neutral; the pipeline does not change.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import schema

FILES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geo": "olist_geolocation_dataset.csv",
}

# Label sanity bounds for this network. Brazilian e-commerce legitimately has
# 40-day deliveries to the North region, so the synthetic network's 30-day cap
# would silently drop ~5% of *real* slow orders — exactly the tail a promise
# model exists for. Pass ``label_max_days=olist.LABEL_MAX_DAYS`` to
# ``cleaning.clean`` when using this source (the CLI does).
LABEL_MIN_DAYS = 0.25  # anything delivered within 6 hours of purchase is a scan error here
LABEL_MAX_DAYS = 60.0

# Brazilian states -> coarse regions (used for origin/dest region features).
STATE_REGION = {
    **dict.fromkeys(["SP", "RJ", "MG", "ES"], "southeast"),
    **dict.fromkeys(["PR", "SC", "RS"], "south"),
    **dict.fromkeys(["DF", "GO", "MT", "MS"], "center_west"),
    **dict.fromkeys(["BA", "PE", "CE", "MA", "PB", "RN", "AL", "SE", "PI"], "northeast"),
    **dict.fromkeys(["AM", "PA", "RO", "RR", "AC", "AP", "TO"], "north"),
}


def _haversine_miles(lat1, lon1, lat2, lon2):
    r_miles = 3958.8
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * r_miles * np.arcsin(np.sqrt(a))


def load(olist_dir: str | Path) -> pd.DataFrame:
    """Build the canonical shipment table from the raw Olist CSVs."""
    d = Path(olist_dir)
    missing = [f for f in FILES.values() if not (d / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Olist files not found in {d}: {missing}. "
            "Download with: kaggle datasets download -d olistbr/brazilian-ecommerce "
            f"-p {d} --unzip"
        )

    orders = pd.read_csv(d / FILES["orders"], parse_dates=[
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ])
    items = pd.read_csv(d / FILES["items"])
    products = pd.read_csv(d / FILES["products"])
    customers = pd.read_csv(d / FILES["customers"])
    sellers = pd.read_csv(d / FILES["sellers"])
    geo = pd.read_csv(d / FILES["geo"])

    # Only delivered orders have a defined transit-time label.
    orders = orders[orders["order_status"] == "delivered"].dropna(
        subset=["order_delivered_customer_date", "order_estimated_delivery_date"]
    )

    # One shipment per order: aggregate items; take the first seller as origin.
    item_agg = (
        items.groupby("order_id")
        .agg(
            n_items=("order_item_id", "size"),
            price=("price", "sum"),
            freight_value=("freight_value", "sum"),
            product_id=("product_id", "first"),
            seller_id=("seller_id", "first"),
        )
        .reset_index()
    )

    df = (
        orders.merge(item_agg, on="order_id", how="inner")
        .merge(
            products[
                ["product_id", "product_weight_g", "product_length_cm",
                 "product_height_cm", "product_width_cm"]
            ],
            on="product_id",
            how="left",
        )
        .merge(customers, on="customer_id", how="left")
        .merge(sellers, on="seller_id", how="left")
    )

    # Zip-prefix centroids for seller -> customer distance.
    centroids = (
        geo.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]]
        .mean()
        .rename(columns={"geolocation_lat": "lat", "geolocation_lng": "lng"})
    )
    df = df.join(
        centroids.rename(columns=lambda c: f"cust_{c}"), on="customer_zip_code_prefix"
    ).join(centroids.rename(columns=lambda c: f"sell_{c}"), on="seller_zip_code_prefix")

    distance = _haversine_miles(df["sell_lat"], df["sell_lng"], df["cust_lat"], df["cust_lng"])

    purchase = df["order_purchase_timestamp"]

    # The label: fractional days from purchase to the customer-delivery scan.
    transit_days = (
        (df["order_delivered_customer_date"] - purchase).dt.total_seconds() / 86400
    )

    # The promise window Olist showed at checkout — known at purchase time, so
    # it is a legitimate feature (and the main carrier of the post-strike
    # regime shift; see module docstring).
    promised_window = (
        (df["order_estimated_delivery_date"] - purchase).dt.total_seconds() / 86400
    ).clip(1, 90)

    weight_lb = (df["product_weight_g"].fillna(0) / 453.6).clip(0.05, 160)
    volume_cuft = (
        df[["product_length_cm", "product_height_cm", "product_width_cm"]]
        .prod(axis=1)
        .fillna(0)
        / 28_316.8
    ).clip(0.001, 80)

    out = pd.DataFrame(
        {
            schema.ID_COL: df["order_id"],
            schema.DATE_COL: purchase.dt.normalize(),
            "distance_miles": distance.fillna(distance.median()).clip(1, 3500),
            "package_weight_lb": weight_lb,
            "package_volume_cuft": volume_cuft,
            "declared_value_usd": df["price"].clip(0, 50_000),  # BRL, but the model only ranks it
            # No hub telemetry or weather feeds in Olist: neutral constants
            # (see module docstring).
            "origin_hub_congestion": 0.5,
            "dest_hub_congestion": 0.5,
            "dest_weather_severity": 0.0,
            # Freight-to-price ratio as a remoteness/awkwardness proxy: heavy,
            # cheap, far-flung orders pay disproportionate freight.
            "route_stop_density": (df["freight_value"] / df["price"].clip(lower=1)).clip(0.05, 30),
            "service_level": "ground",
            "origin_region": df["seller_state"].map(STATE_REGION).fillna("other"),
            "dest_region": df["customer_state"].map(STATE_REGION).fillna("other"),
            "dest_type": "residential",
            "day_of_week": purchase.dt.dayofweek,
            "is_peak_season": purchase.dt.month.isin([11, 12]).astype(int),
            "is_rural_dest": (~df["customer_state"].isin(["SP", "RJ"])).astype(int),
            "signature_required": 0,
            # Adapter-specific extra feature (features.OPTIONAL_NUMERIC picks
            # it up when present; the synthetic source simply doesn't have it).
            "promised_window_days": promised_window,
            schema.LABEL_COL: transit_days,
        }
    )

    # Label sanity at the source: sub-6-hour "deliveries" and >60-day parcels
    # are scan errors / lost-and-found on this network, not lanes to learn.
    out = out[
        (out[schema.LABEL_COL] >= LABEL_MIN_DAYS) & (out[schema.LABEL_COL] <= LABEL_MAX_DAYS)
    ]
    return out.reset_index(drop=True)
