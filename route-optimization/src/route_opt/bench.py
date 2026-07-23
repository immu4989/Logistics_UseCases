"""Benchmark the optimizer against CVRPLIB best-known solutions.

For each registry instance: run Clarke-Wright + 2-opt (the exact code the
synthetic pipeline runs), report our distance, the BKS, the gap, and the
runtime; run plain nearest-neighbor for context. Writes ``bench_results.csv``
and a README-ready ``bench_table.md``.

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
        routes, ours = cvrplib.solve_instance(instance)
        runtime_s = time.perf_counter() - start
        _check_valid(instance, routes)
        _, nn_cost = cvrplib.solve_instance_nn(instance)

        rows.append(
            {
                "instance": name,
                "customers": instance.n_customers,
                "capacity": instance.capacity,
                "ours": int(ours),
                "bks": bks,
                "gap_pct": round(100.0 * (ours - bks) / bks, 2),
                "nn": int(nn_cost),
                "nn_gap_pct": round(100.0 * (nn_cost - bks) / bks, 2),
                "routes": len(routes),
                "runtime_s": round(runtime_s, 3),
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
        "| Instance | Customers | Ours (CW+2-opt) | BKS | Gap | NN (context) | Runtime |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in results.iterrows():
        lines.append(
            f"| {r['instance']} | {r['customers']} | {r['ours']:,} | {r['bks']:,} "
            f"| {r['gap_pct']:.1f}% | +{r['nn_gap_pct']:.0f}% | {r['runtime_s']:.2f}s |"
        )
    lines.append(
        f"| **mean** | | | | **{results['gap_pct'].mean():.1f}%** "
        f"| +{results['nn_gap_pct'].mean():.0f}% | |"
    )
    return "\n".join(lines)


def write_reports(results: pd.DataFrame, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "bench_results.csv", index=False)
    (out_dir / "bench_table.md").write_text(to_markdown_table(results) + "\n")
