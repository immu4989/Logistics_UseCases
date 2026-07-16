"""Adapter: Olist Brazilian e-commerce dataset -> canonical latemile schema.

Olist is the best public proxy for delivery-commit prediction: ~100k real
orders with both a *promised* date (``order_estimated_delivery_date``) and an
*actual* one (``order_delivered_customer_date``), so the miss label is real,
not simulated.

Download (free Kaggle account required):
    kaggle datasets download -d olistbr/brazilian-ecommerce -p data/olist --unzip
or from https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

Then:
    latemile all --source olist --olist-dir data/olist

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
        "order_approved_at",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ])
    items = pd.read_csv(d / FILES["items"])
    products = pd.read_csv(d / FILES["products"])
    customers = pd.read_csv(d / FILES["customers"])
    sellers = pd.read_csv(d / FILES["sellers"])
    geo = pd.read_csv(d / FILES["geo"])

    # Only delivered orders have a defined label.
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

    # Zip-prefix centroids for distance.
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
    promised_days = (
        (df["order_estimated_delivery_date"] - purchase).dt.total_seconds() / 86400
    ).clip(1, 60)

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
            "declared_value_usd": df["price"].clip(0, 50_000),
            # Approval lag stands in for pickup-after-cutoff: how long the
            # order sat before the fulfilment clock really started.
            "minutes_after_cutoff": (
                (df["order_approved_at"] - purchase).dt.total_seconds() / 60
            ).fillna(0).clip(-1440, 1440),
            # No hub telemetry in Olist: neutral constants (see module docstring).
            "origin_hub_congestion": 0.5,
            "dest_hub_congestion": 0.5,
            "dest_weather_severity": 0.0,
            "route_stop_density": (df["freight_value"] / df["price"].clip(lower=1)).clip(0.05, 30),
            "promised_transit_days": promised_days.round().astype(int).clip(1, 10),
            "service_level": "ground",
            "origin_region": df["seller_state"].map(STATE_REGION).fillna("other"),
            "dest_region": df["customer_state"].map(STATE_REGION).fillna("other"),
            "dest_type": "residential",
            "day_of_week": purchase.dt.dayofweek,
            "is_peak_season": purchase.dt.month.isin([11, 12]).astype(int),
            "is_rural_dest": (~df["customer_state"].isin(["SP", "RJ"])).astype(int),
            "signature_required": 0,
            schema.LABEL_COL: (
                df["order_delivered_customer_date"] > df["order_estimated_delivery_date"]
            ).astype(int),
        }
    )
    return out.reset_index(drop=True)
