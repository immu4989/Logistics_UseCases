"""Command-line interface.

    route-opt generate --out artifacts/data/stops.csv   # one messy delivery day
    route-opt all                                        # full pipeline
    route-opt bench                                      # CVRPLIB benchmark vs BKS

Artifacts land in ./artifacts by default:
    artifacts/data/      the raw (messy) and cleaned stop lists
    artifacts/reports/   metrics.json, routes.csv, rationale.md,
                         cleaning_report.csv, plots, bench_results.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import bench, cleaning, cvrplib, evaluate, explain, solve, synthetic


def cmd_generate(args) -> None:
    df = synthetic.make_day(n=args.n, seed=args.seed, messy=not args.clean)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df):,} stop rows -> {out}")


def cmd_all(args) -> None:
    art = Path(args.artifacts)
    data_dir, report_dir = art / "data", art / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] generating one delivery day (n={args.n:,} stops, messy extract) ...")
    raw = synthetic.make_day(n=args.n, seed=args.seed, messy=True)
    raw.to_csv(data_dir / "stops_raw.csv", index=False)

    print("[2/4] cleaning the dispatch extract ...")
    stops, report = cleaning.clean(raw)
    print(report)
    stops.to_csv(data_dir / "stops_clean.csv", index=False)
    report.to_frame().to_csv(report_dir / "cleaning_report.csv", index=False)
    print(
        f"      {len(stops):,} routable stops, {int(stops['packages'].sum()):,} packages, "
        f"minimum fleet {solve.trucks_needed(stops)} trucks"
    )

    print("[3/4] routing the day four ways (same stops, same fleet constraints) ...")
    dist = solve.distance_matrix(stops)
    solutions = {name: fn(stops, dist) for name, fn in solve.POLICIES.items()}
    table = evaluate.evaluate_all(stops, solutions, dist, out_dir=report_dir)
    header = (
        f"      {'policy':<26} {'miles':>7} {'trucks':>7} {'hours':>7} "
        f"{'max hr':>7} {'$/day':>9} {'saved/yr':>11}"
    )
    print(header)
    for _, r in table.iterrows():
        print(
            f"      {r['policy']:<26} {r['total_miles']:>7,.0f} {r['trucks_used']:>7} "
            f"{r['total_route_hours']:>7.1f} {r['max_route_hours']:>7.2f} "
            f"${r['daily_cost_usd']:>8,.0f} ${r['annual_savings_vs_zone_usd']:>10,.0f}"
        )
    lb = solve.lower_bound_miles(stops, dist)
    print(f"      loose lower bound: {lb:,.0f} miles (no plan can beat this)")

    print("[4/4] writing dispatch-sheet rationale ...")
    attribution = explain.write_rationale(stops, solutions, dist, report_dir)
    for _, r in attribution.iterrows():
        print(f"      {r['stage']:<42} {r['total_miles']:>8,.1f} mi  (saved {r['miles_saved']:,.1f})")

    star = table.set_index("policy").loc["savings_ls"]
    print(
        f"\nsavings_ls vs today's fixed zones: {star['miles_saved_vs_zone_pct']:.1f}% fewer miles, "
        f"${star['daily_savings_vs_zone_usd']:,.0f}/day, "
        f"${star['annual_savings_vs_zone_usd']:,.0f}/year at {evaluate.OPERATING_DAYS_PER_YEAR} "
        f"operating days.\ndone. reports -> {report_dir}"
    )


def cmd_bench(args) -> None:
    names = [n.strip() for n in args.instances.split(",") if n.strip()] if args.instances else None
    report_dir = Path(args.artifacts) / "reports"
    print(
        f"benchmarking savings_2opt and savings_ls against CVRPLIB best-known solutions "
        f"({len(names or cvrplib.REGISTRY)} instances, cache: {args.cache_dir}) ..."
    )
    results = bench.run_bench(names, cache_dir=Path(args.cache_dir))
    print(
        f"{'instance':<12} {'cust':>5} {'2opt':>7} {'gap':>6} {'ls':>7} {'ls gap':>7} "
        f"{'bks':>7} {'nn gap':>7} {'time':>6}"
    )
    for _, r in results.iterrows():
        print(
            f"{r['instance']:<12} {r['customers']:>5} {r['ours']:>7,} {r['gap_pct']:>5.1f}% "
            f"{r['ls']:>7,} {r['ls_gap_pct']:>6.1f}% {r['bks']:>7,} "
            f"{r['nn_gap_pct']:>6.1f}% {r['ls_runtime_s']:>5.1f}s"
        )
    print(f"mean gap to BKS: savings_2opt {results['gap_pct'].mean():.1f}%, "
          f"savings_ls {results['ls_gap_pct'].mean():.1f}% "
          f"(nearest-neighbor context: +{results['nn_gap_pct'].mean():.0f}%)")
    bench.write_reports(results, report_dir)
    print(f"wrote {report_dir}/bench_results.csv and {report_dir}/bench_table.md")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="route-opt", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="write one synthetic delivery day of stops")
    g.add_argument("--out", default="artifacts/data/stops.csv")
    g.add_argument("--n", type=int, default=600)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--clean", action="store_true", help="skip the planted data faults")
    g.set_defaults(func=cmd_generate)

    a = sub.add_parser("all", help="run the full pipeline: generate, clean, route, evaluate, explain")
    a.add_argument("--n", type=int, default=600)
    a.add_argument("--seed", type=int, default=7)
    a.add_argument("--artifacts", default="artifacts")
    a.set_defaults(func=cmd_all)

    b = sub.add_parser("bench", help="benchmark the optimizer against CVRPLIB best-known solutions")
    b.add_argument(
        "--instances",
        default=None,
        help=f"comma-separated instance names (default: all of {', '.join(cvrplib.REGISTRY)})",
    )
    b.add_argument("--cache-dir", default=str(cvrplib.DEFAULT_CACHE_DIR),
                   help="where downloaded .vrp/.sol files are cached")
    b.add_argument("--artifacts", default="artifacts")
    b.set_defaults(func=cmd_bench)

    args = p.parse_args(argv)
    try:
        args.func(args)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
