"""Benchmark the optimizer against CVRPLIB best-known solutions.

For each registry instance: run BOTH solvers the synthetic pipeline runs —
Clarke-Wright + 2-opt (``savings_2opt``) and the full local-search stack
(``savings_ls``: + or-opt, 2-opt*, swap, seeded ILS) — and report each
distance, the BKS, both gaps, and both runtimes; run plain nearest-neighbor
for context. Writes ``bench_results.csv`` and a README-ready
``bench_table.md``.

Needs network on the first run (instances download into the cache dir);
afterwards everything is offline and deterministic.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from . import cvrplib


def run_bench(
    names: list[str] | None = None,
    cache_dir: Path = cvrplib.DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """Run the benchmark, one row per instance, validity-checked."""
    names = list(names) if names else list(cvrplib.REGISTRY)
    rows = []
    for name in names:
        instance = cvrplib.load(name, cache_dir=cache_dir)
        bks = cvrplib.REGISTRY[name]["bks"]

        start = time.perf_counter()
        routes_2opt, cost_2opt = cvrplib.solve_instance(instance)
        runtime_2opt = time.perf_counter() - start
        _check_valid(instance, routes_2opt)

        start = time.perf_counter()
        routes_ls, cost_ls = cvrplib.solve_instance_ls(instance)
        runtime_ls = time.perf_counter() - start
        _check_valid(instance, routes_ls)

        _, nn_cost = cvrplib.solve_instance_nn(instance)

        rows.append(
            {
                "instance": name,
                "customers": instance.n_customers,
                "capacity": instance.capacity,
                "ours": int(cost_2opt),
                "gap_pct": round(100.0 * (cost_2opt - bks) / bks, 2),
                "ls": int(cost_ls),
                "ls_gap_pct": round(100.0 * (cost_ls - bks) / bks, 2),
                "bks": bks,
                "nn": int(nn_cost),
                "nn_gap_pct": round(100.0 * (nn_cost - bks) / bks, 2),
                "routes": len(routes_2opt),
                "ls_routes": len(routes_ls),
                "runtime_s": round(runtime_2opt, 3),
                "ls_runtime_s": round(runtime_ls, 3),
            }
        )
    return pd.DataFrame(rows)


def _check_valid(instance: cvrplib.Instance, routes: list[list[int]]) -> None:
    """A benchmark number is meaningless unless the plan behind it is legal."""
    visited = sorted(stop for route in routes for stop in route)
    if visited != list(range(instance.n_customers)):
        raise AssertionError(f"{instance.name}: customers not served exactly once")
    demands = instance.demands[1:]
    for route in routes:
        if demands[route].sum() > instance.capacity:
            raise AssertionError(f"{instance.name}: a route exceeds capacity")


def to_markdown_table(results: pd.DataFrame) -> str:
    """README-ready table plus the mean-gap summary line."""
    lines = [
        "| Instance | Customers | CW+2-opt | Gap | savings_ls | Gap | BKS "
        "| NN (context) | Runtime |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in results.iterrows():
        lines.append(
            f"| {r['instance']} | {r['customers']} | {r['ours']:,} | {r['gap_pct']:.1f}% "
            f"| {r['ls']:,} | **{r['ls_gap_pct']:.1f}%** | {r['bks']:,} "
            f"| +{r['nn_gap_pct']:.0f}% | {r['ls_runtime_s']:.1f}s |"
        )
    lines.append(
        f"| **mean** | | | **{results['gap_pct'].mean():.1f}%** "
        f"| | **{results['ls_gap_pct'].mean():.1f}%** | "
        f"| +{results['nn_gap_pct'].mean():.0f}% | |"
    )
    return "\n".join(lines)


def write_reports(results: pd.DataFrame, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "bench_results.csv", index=False)
    (out_dir / "bench_table.md").write_text(to_markdown_table(results) + "\n")
