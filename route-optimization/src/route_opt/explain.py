"""The audit trail: a dispatch sheet a driver could run, plus where the savings came from.

No SHAP here; there is no model to attribute. What a routing recommendation
must survive is a different audit: a dispatcher looking at truck 3's manifest
and asking "can my driver actually run this sheet?" So the explanation is the
dispatch sheet itself — stop order with running load and clock time, checked
against capacity and shift length line by line — plus an attribution that
splits the saved miles between the two algorithm stages (Clarke-Wright
construction vs 2-opt polish), so nobody has to take "the optimizer did it"
on faith.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import solve

SHIFT_START = 8 * 60.0  # 08:00 dispatch, minutes since midnight


def dispatch_sheet(route: list[int], stops: pd.DataFrame, dist: np.ndarray) -> pd.DataFrame:
    """One truck's manifest: visit order with cumulative load, miles and clock."""
    service = stops["service_min"].to_numpy()
    packages = stops["packages"].to_numpy()

    rows = []
    clock, cum_mi, cum_load = SHIFT_START, 0.0, 0
    prev = 0  # depot
    for seq, i in enumerate(route, start=1):
        leg = dist[prev, i + 1]
        clock += leg / solve.SPEED_MPH * 60.0
        cum_mi += leg
        cum_load += int(packages[i])
        rows.append(
            {
                "seq": seq,
                "stop_id": stops["stop_id"].iloc[i],
                "type": stops["stop_type"].iloc[i],
                "leg_mi": round(leg, 2),
                "cum_mi": round(cum_mi, 1),
                "packages": int(packages[i]),
                "cum_load": cum_load,
                "arrive": f"{int(clock // 60):02d}:{int(clock % 60):02d}",
            }
        )
        clock += service[i]
        prev = i + 1
    # The ride home closes the duration check.
    leg = dist[prev, 0]
    clock += leg / solve.SPEED_MPH * 60.0
    cum_mi += leg
    rows.append(
        {
            "seq": len(route) + 1, "stop_id": "DEPOT", "type": "return",
            "leg_mi": round(leg, 2), "cum_mi": round(cum_mi, 1),
            "packages": 0, "cum_load": cum_load,
            "arrive": f"{int(clock // 60):02d}:{int(clock % 60):02d}",
        }
    )
    return pd.DataFrame(rows)


def write_rationale(
    stops: pd.DataFrame,
    solutions: dict[str, solve.Solution],
    dist: np.ndarray,
    out_dir: str | Path,
) -> pd.DataFrame:
    """Write rationale.md: two dispatch sheets + the miles-saved attribution.

    Returns the summary attribution table (one row per stage).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    star = solutions["savings_2opt"]
    zone_miles = sum(solve.route_miles(r, dist) for r in solutions["zone_fixed"].routes)
    final_miles = sum(solve.route_miles(r, dist) for r in star.routes)
    cw_miles = star.construction_miles
    attribution = pd.DataFrame(
        [
            {"stage": "zone_fixed (status quo)", "total_miles": round(zone_miles, 1),
             "miles_saved": 0.0},
            {"stage": "after Clarke-Wright construction", "total_miles": round(cw_miles, 1),
             "miles_saved": round(zone_miles - cw_miles, 1)},
            {"stage": "after per-route 2-opt", "total_miles": round(final_miles, 1),
             "miles_saved": round(cw_miles - final_miles, 1)},
        ]
    )

    # Two contrasting example trucks: the fullest (capacity does the work)
    # and the longest (the shift constraint does the work).
    packages = stops["packages"].to_numpy()
    service = stops["service_min"].to_numpy()
    loads = [packages[r].sum() for r in star.routes]
    hours = [solve.route_minutes(r, dist, service) / 60.0 for r in star.routes]
    examples = [
        ("fullest truck", int(np.argmax(loads))),
        ("longest shift", int(np.argmax(hours))),
    ]
    if examples[0][1] == examples[1][1]:  # same truck? take the second-longest shift
        order = np.argsort(hours)
        examples[1] = ("longest shift", int(order[-2]))

    lines = [
        "# Dispatch rationale: can a driver actually run these sheets?",
        "",
        "Every number below is checkable against the fleet constraints: "
        f"capacity {solve.CAPACITY_PKGS} packages, shift "
        f"{solve.MAX_ROUTE_MIN / 60:.0f}h from an 08:00 dispatch, {solve.SPEED_MPH:.0f} mph "
        "between stops plus per-stop service time.",
        "",
        "## Where the saved miles came from",
        "",
        "| Stage | Total miles | Miles saved at this stage |",
        "|---|---:|---:|",
    ]
    for _, r in attribution.iterrows():
        lines.append(f"| {r['stage']} | {r['total_miles']:,.1f} | {r['miles_saved']:,.1f} |")
    lines += [
        "",
        "Construction does the heavy lifting (it decides WHICH stops share a "
        "truck); 2-opt then uncrosses each route's path. Both stages respect "
        "capacity and shift limits at every step, so the final plan needs no "
        "repair pass.",
    ]

    for label, t in examples:
        route = star.routes[t]
        sheet = dispatch_sheet(route, stops, dist)
        end = sheet["arrive"].iloc[-1]
        lines += [
            "",
            f"## Truck {t + 1} ({label}): {len(route)} stops, "
            f"{int(packages[route].sum())}/{solve.CAPACITY_PKGS} packages, "
            f"back at depot {end}",
            "",
            "| # | Stop | Type | Leg mi | Cum mi | Pkgs | Load | Arrive |",
            "|---:|---|---|---:|---:|---:|---:|---|",
        ]
        # Head and tail of the manifest; the middle compresses to one line.
        show = pd.concat([sheet.head(6), sheet.tail(4)]) if len(sheet) > 12 else sheet
        skipped = len(sheet) - len(show)
        for k, (_, r) in enumerate(show.iterrows()):
            if skipped > 0 and k == 6:
                lines.append(f"| ... | _{skipped} more stops_ | | | | | | |")
            lines.append(
                f"| {r['seq']} | {r['stop_id']} | {r['type']} | {r['leg_mi']:.2f} "
                f"| {r['cum_mi']:.1f} | {r['packages']} | {r['cum_load']} | {r['arrive']} |"
            )

    lines += [
        "",
        "_Full plans for every policy, one row per visit in drive order, are in "
        "`routes.csv`._",
    ]
    (out_dir / "rationale.md").write_text("\n".join(lines))
    return attribution
