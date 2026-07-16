"""Command-line interface.

    freight-price generate --out artifacts/data/quotes.csv   # synthetic quote log
    freight-price all                                        # full pipeline

Artifacts land in ./artifacts by default:
    artifacts/data/      raw + cleaned quote logs
    artifacts/models/    trained acceptance models (joblib)
    artifacts/reports/   metrics.json, policy_comparison.csv, segment_uplift.csv,
                         rationale.md, plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import cleaning, evaluate, explain, synthetic
from . import train as train_mod


def cmd_generate(args) -> None:
    df = synthetic.make_quotes(n=args.n, seed=args.seed, messy=not args.clean)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df):,} quote requests -> {out}")


def cmd_all(args) -> None:
    art = Path(args.artifacts)
    data_dir, model_dir, report_dir = art / "data", art / "models", art / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] generating the historical quote log (n={args.n:,}) ...")
    raw = synthetic.make_quotes(n=args.n, seed=args.seed, messy=True)
    raw.to_csv(data_dir / "quotes.csv", index=False)
    print(f"      {len(raw):,} rows, historical acceptance rate {raw['accepted'].mean():.1%}")

    print("[2/5] cleaning ...")
    clean_df, report = cleaning.clean(raw)
    clean_df.to_csv(data_dir / "clean.csv", index=False)
    report.to_frame().to_csv(data_dir / "cleaning_report.csv", index=False)
    print(str(report))

    print("[3/5] training acceptance models (logistic baseline + monotone XGBoost) ...")
    models, splits = train_mod.train(clean_df, train_mod.TrainConfig(seed=args.seed))
    path = train_mod.save(models, model_dir)
    print(f"      trained on quote dates <= {models.cutoff_date}; models -> {path}")

    print("[4/5] pricing the held-out quarter under every policy ...")
    comparison, segment, prices, info = evaluate.evaluate_policies(
        models, splits, seed=args.seed, out_dir=report_dir
    )
    print(
        f"      acceptance-model ROC-AUC: logistic {info['aucs']['logistic']:.3f} | "
        f"xgboost {info['aucs']['xgboost']:.3f} | "
        f"flat-optimal markup {info['flat_multiplier']:.2f}x"
    )

    print("[5/5] writing per-quote rationale ...")
    table = explain.write_rationale(splits["test"], prices, models, report_dir)
    for _, r in table.iterrows():
        print(
            f"      {r['case']:<38} {r['quote_id']} "
            f"${r['cost_plus_price_usd']:,.0f} -> ${r['model_price_usd']:,.0f}"
        )

    print(
        f"\n      {'policy':<16} {'exp. margin':>12} {'per quote':>10} "
        f"{'win rate':>9} {'avg mult':>9} {'vs cost-plus':>13} {'% oracle':>9}"
    )
    for _, r in comparison.iterrows():
        print(
            f"      {r['policy']:<16} ${r['expected_margin_usd']:>11,.0f} "
            f"${r['margin_per_quote_usd']:>9,.2f} {r['expected_win_rate']:>8.1%} "
            f"{r['avg_price_multiplier']:>8.2f}x {r['uplift_vs_cost_plus_pct']:>+12.1f}% "
            f"{r['pct_of_oracle']:>8.1f}%"
        )

    print(
        f"\n      {'segment':<10} {'quotes':>7} {'cost-plus $/q':>14} {'model $/q':>10} "
        f"{'uplift':>8} {'share of uplift':>16}"
    )
    for _, r in segment.iterrows():
        print(
            f"      {r['segment']:<10} {r['n_quotes']:>7,} "
            f"${r['cost_plus_margin_per_quote_usd']:>13,.2f} "
            f"${r['model_margin_per_quote_usd']:>9,.2f} {r['uplift_pct']:>+7.1f}% "
            f"{r['share_of_total_uplift_pct']:>15.1f}%"
        )
    print(f"\ndone. reports -> {report_dir}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="freight-price", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="write a synthetic historical quote log")
    g.add_argument("--out", default="artifacts/data/quotes.csv")
    g.add_argument("--n", type=int, default=40_000)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--clean", action="store_true", help="skip mess injection")
    g.set_defaults(func=cmd_generate)

    a = sub.add_parser("all", help="run the full pipeline: generate, clean, train, price, explain")
    a.add_argument("--n", type=int, default=40_000)
    a.add_argument("--seed", type=int, default=7)
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
