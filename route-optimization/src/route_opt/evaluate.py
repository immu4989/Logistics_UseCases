"""Score every policy on the same day, price the miles, draw the maps.

Every policy serves the identical cleaned stop list with the identical fleet
constraints, so the differences below are pure routing. Metrics per policy:

- total miles and miles per stop (the optimizer's own units),
- trucks used, total route hours, max route duration (the constraint check),
- stops per truck-hour (the productivity number ops managers actually track),
- dollars: miles and driver-hours priced per the constants below, differenced
  against ``zone_fixed`` (the status quo) and annualized.

The route map is the artifact that convinces people: the same metro, zone
wedges on the left with their spokes through the depot, and the optimizer's
compact petals on the right.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import solve

# ---------------------------------------------------------------------------
# Economics. Business constants — replace with your fleet's cost accounting.
# COST_PER_MILE covers fuel, maintenance and depreciation (typical mid-size
# delivery van, blended); driver cost is fully loaded wage + benefits.
# ---------------------------------------------------------------------------
COST_PER_MILE_USD = 0.85
DRIVER_HOURLY_USD = 38.0
OPERATING_DAYS_PER_YEAR = 250

BASELINE = "zone_fixed"

# One fixed color per policy across every chart (color follows the entity).
POLICY_COLORS = {
    "zone_fixed": "#8d99ae",
    "nearest_neighbor_global": "#e8a33d",
    "savings_2opt": "#2b6cb0",
    "savings_ls": "#2f855a",
}
# Truck colors on the maps: fixed assignment (color follows the truck, never
# its rank), tab10 while it lasts, tab20's second half beyond that.
def _truck_color(truck: int):
    return plt.get_cmap("tab10")(truck) if truck < 10 else plt.get_cmap("tab20")(truck % 20)


def summarize(sol: solve.Solution, stops: pd.DataFrame, dist: np.ndarray) -> dict:
    """One row of the policy table."""
    service_min = stops["service_min"].to_numpy()
    miles = [solve.route_miles(r, dist) for r in sol.routes]
    minutes = [solve.route_minutes(r, dist, service_min) for r in sol.routes]
    total_hours = sum(minutes) / 60.0
    daily_cost = sum(miles) * COST_PER_MILE_USD + total_hours * DRIVER_HOURLY_USD
    return {
        "policy": sol.policy,
        "total_miles": round(sum(miles), 1),
        "trucks_used": len(sol.routes),
        "total_route_hours": round(total_hours, 2),
        "max_route_hours": round(max(minutes) / 60.0, 2),
        "stops_per_truck_hour": round(len(stops) / total_hours, 2),
        "miles_per_stop": round(sum(miles) / len(stops), 3),
        "daily_cost_usd": round(daily_cost, 2),
    }


def evaluate_all(
    stops: pd.DataFrame,
    solutions: dict[str, solve.Solution],
    dist: np.ndarray,
    out_dir: str | Path = "artifacts/reports",
) -> pd.DataFrame:
    """Build the comparison table, write metrics.json / routes.csv / plots."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table = pd.DataFrame([summarize(s, stops, dist) for s in solutions.values()])
    base_cost = table.loc[table["policy"] == BASELINE, "daily_cost_usd"].iloc[0]
    base_miles = table.loc[table["policy"] == BASELINE, "total_miles"].iloc[0]
    table["daily_savings_vs_zone_usd"] = (base_cost - table["daily_cost_usd"]).round(2)
    table["annual_savings_vs_zone_usd"] = (
        table["daily_savings_vs_zone_usd"] * OPERATING_DAYS_PER_YEAR
    ).round(0)
    table["miles_saved_vs_zone_pct"] = (
        (base_miles - table["total_miles"]) / base_miles * 100
    ).round(1)

    lb = solve.lower_bound_miles(stops, dist)
    metrics = {
        "n_stops": int(len(stops)),
        "total_packages": int(stops["packages"].sum()),
        "capacity_pkgs": solve.CAPACITY_PKGS,
        "max_route_hours": solve.MAX_ROUTE_MIN / 60.0,
        "speed_mph": solve.SPEED_MPH,
        "cost_per_mile_usd": COST_PER_MILE_USD,
        "driver_hourly_usd": DRIVER_HOURLY_USD,
        "operating_days_per_year": OPERATING_DAYS_PER_YEAR,
        "lower_bound_miles": round(lb, 1),
        "construction_miles_savings_2opt": solutions["savings_2opt"].construction_miles,
        # savings_ls stage miles: construction -> 2-opt -> local-search stack -> ILS
        "stage_miles_savings_ls": solutions["savings_ls"].stage_miles,
        "policies": table.to_dict(orient="records"),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    _write_routes_csv(stops, solutions, out_dir / "routes.csv")
    _plot_route_maps(stops, solutions, dist, out_dir / "route_maps.png")
    _plot_miles(table, lb, out_dir / "miles_by_policy.png")
    _plot_productivity(table, out_dir / "stops_per_truck_hour.png")
    return table


def _write_routes_csv(stops, solutions, path: Path) -> None:
    """Every policy's full plan, one row per visit, in drive order."""
    rows = []
    for name, sol in solutions.items():
        for truck, route in enumerate(sol.routes, start=1):
            for seq, i in enumerate(route, start=1):
                rows.append(
                    {
                        "policy": name,
                        "truck": truck,
                        "seq": seq,
                        "stop_id": stops["stop_id"].iloc[i],
                        "x_mi": stops["x_mi"].iloc[i],
                        "y_mi": stops["y_mi"].iloc[i],
                        "packages": int(stops["packages"].iloc[i]),
                    }
                )
    pd.DataFrame(rows).to_csv(path, index=False)


def _draw_solution(ax, stops, sol: solve.Solution, dist) -> None:
    """One panel of the route map: colored tours, starred depot, quiet grid."""
    xy = stops[["x_mi", "y_mi"]].to_numpy()
    for truck, route in enumerate(sol.routes):
        color = _truck_color(truck)
        path = np.vstack([[0.0, 0.0], xy[route], [0.0, 0.0]])
        ax.plot(path[:, 0], path[:, 1], color=color, lw=1.1, zorder=2)
        ax.scatter(xy[route, 0], xy[route, 1], s=7, color=color, zorder=3)
    ax.scatter([0], [0], marker="*", s=340, color="#1a1a2e", zorder=5,
               edgecolors="white", linewidths=0.8, label="depot")
    miles = sum(solve.route_miles(r, dist) for r in sol.routes)
    ax.set_title(f"{sol.policy}: {miles:,.0f} miles, {len(sol.routes)} trucks", fontsize=12)
    ax.set_aspect("equal")
    ax.grid(True, color="#e3e6ea", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("miles east of depot")
    for spine in ax.spines.values():
        spine.set_color("#c9ced4")


def _plot_route_maps(stops, solutions, dist, path: Path) -> None:
    """The money chart: status quo vs optimizer, same metro, side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.3), sharex=True, sharey=True)
    _draw_solution(axes[0], stops, solutions[BASELINE], dist)
    _draw_solution(axes[1], stops, solutions["savings_ls"], dist)
    axes[0].set_ylabel("miles north of depot")
    axes[0].legend(loc="lower left", fontsize=9, framealpha=0.9)
    fig.suptitle("Same stops, same trucks, same constraints — only the routing differs",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_miles(table: pd.DataFrame, lower_bound: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = [POLICY_COLORS[p] for p in table["policy"]]
    ax.bar(table["policy"], table["total_miles"], color=colors, width=0.62)
    ax.axhline(lower_bound, color="#c0392b", ls="--", lw=1.2,
               label=f"lower bound (loose): {lower_bound:,.0f} mi")
    for i, row in table.reset_index().iterrows():
        ax.text(i, row["total_miles"], f"{row['total_miles']:,.0f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Total miles (one delivery day)")
    ax.set_title("Miles to serve the identical stop list")
    ax.grid(axis="y", color="#e3e6ea", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_productivity(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = [POLICY_COLORS[p] for p in table["policy"]]
    ax.bar(table["policy"], table["stops_per_truck_hour"], color=colors, width=0.62)
    for i, row in table.reset_index().iterrows():
        ax.text(i, row["stops_per_truck_hour"], f"{row['stops_per_truck_hour']:.1f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Stops per truck-hour")
    ax.set_title("Driver productivity on the identical stop list")
    ax.grid(axis="y", color="#e3e6ea", lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
