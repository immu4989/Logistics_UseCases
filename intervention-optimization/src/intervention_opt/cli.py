"""Command-line interface.

    intervention-opt generate --out artifacts/data/shipments.csv   # one scored day
    intervention-opt all                                           # full pipeline

Artifacts land in ./artifacts by default:
    artifacts/data/      the generated day of scored shipments
    artifacts/reports/   metrics.json, policy_comparison.csv, decisions.csv,
                         rationale.md, decision_policy.csv, plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import evaluate, explain, policy, synthetic


def cmd_generate(args) -> None:
    df = synthetic.make_day(n=args.n, seed=args.seed, score_noise=args.score_noise)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df):,} scored shipments -> {out}")


def cmd_all(args) -> None:
    art = Path(args.artifacts)
    data_dir, report_dir = art / "data", art / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] generating one day of scored shipments (n={args.n:,}) ...")
    df = synthetic.make_day(n=args.n, seed=args.seed, score_noise=args.score_noise)
    df.to_csv(data_dir / "shipments.csv", index=False)
    print(
        f"      true mean miss risk {df['p_true'].mean():.1%} | "
        f"model score noise sd {args.score_noise} (logit space)"
    )

    print(f"[2/3] allocating ${args.budget:,.0f} across policies and simulating outcomes ...")
    comparison, decisions = evaluate.evaluate_all(
        df, budget=args.budget, seed=args.seed, out_dir=report_dir
    )
    header = (
        f"      {'policy':<24} {'spend':>9} {'prevented':>10} "
        f"{'avoided':>10} {'net savings':>12} {'ROI':>6} {'% oracle':>9}"
    )
    print(header)
    for _, r in comparison.iterrows():
        roi = f"{r['roi']:.1f}x" if pd.notna(r["roi"]) else "-"
        print(
            f"      {r['policy']:<24} ${r['spend_usd']:>8,.0f} {r['misses_prevented']:>10,} "
            f"${r['miss_cost_avoided_usd']:>9,.0f} ${r['net_savings_usd']:>11,.0f} "
            f"{roi:>6} {r['pct_of_oracle']:>8.1f}%"
        )

    print("[3/3] writing per-decision rationale ...")
    table = explain.write_rationale(df, decisions["expected_value_greedy"], report_dir)
    for _, r in table.iterrows():
        print(
            f"      {r['case']:<36} {r['shipment_id']} -> {r['action']} "
            f"(exp. net ${r['expected_net_savings_usd']:,.2f})"
        )
    print(f"\ndone. reports -> {report_dir}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="intervention-opt", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="write one synthetic day of scored shipments")
    g.add_argument("--out", default="artifacts/data/shipments.csv")
    g.add_argument("--n", type=int, default=20_000)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument(
        "--score-noise", type=float, default=synthetic.DEFAULT_SCORE_NOISE,
        help="sd of logit-space noise on the upstream model score",
    )
    g.set_defaults(func=cmd_generate)

    a = sub.add_parser("all", help="run the full pipeline: generate, allocate, evaluate, explain")
    a.add_argument("--n", type=int, default=20_000)
    a.add_argument("--seed", type=int, default=7)
    a.add_argument("--budget", type=float, default=policy.DEFAULT_BUDGET_USD)
    a.add_argument("--score-noise", type=float, default=synthetic.DEFAULT_SCORE_NOISE)
    a.add_argument("--artifacts", default="artifacts")
    a.set_defaults(func=cmd_all)

    args = p.parse_args(argv)
    try:
        args.func(args)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
