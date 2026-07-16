"""Command-line interface.

    exception-triage generate --out artifacts/data/raw.csv   # synthetic raw extract
    exception-triage all                                      # full pipeline

Artifacts land in ./artifacts by default:
    artifacts/data/      raw + cleaned tables, cleaning report
    artifacts/models/    trained models (joblib)
    artifacts/reports/   metrics.json, plots, per-queue drivers, ticket cards
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import cleaning, evaluate, schema, synthetic
from . import explain as explain_mod
from . import train as train_mod


def cmd_generate(args) -> None:
    df = synthetic.make_dataset(n=args.n, seed=args.seed, messy=not args.clean)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df):,} exception tickets -> {out}")


def cmd_all(args) -> None:
    art = Path(args.artifacts)
    data_dir, model_dir, report_dir = art / "data", art / "models", art / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] generating raw ticket extract ...")
    raw = synthetic.make_dataset(n=args.n, seed=args.seed, messy=True)
    raw.to_csv(data_dir / "raw.csv", index=False)
    print(f"      {len(raw):,} tickets over ~6 months, 6 resolution queues")

    print("[2/5] cleaning ...")
    clean_df, report = cleaning.clean(raw)
    clean_df.to_csv(data_dir / "clean.csv", index=False)
    report.to_frame().to_csv(data_dir / "cleaning_report.csv", index=False)
    print(str(report))

    print("[3/5] training (rules comparator + multinomial logistic + XGBoost) ...")
    models, splits = train_mod.train(clean_df, train_mod.TrainConfig(seed=args.seed))
    path = train_mod.save(models, model_dir)
    print(f"      trained on ticket dates <= {models.cutoff_date}; models -> {path}")

    print("[4/5] evaluating on the held-out period ...")
    results = evaluate.evaluate_models(models, splits, report_dir)

    print("[5/5] per-queue SHAP driver analysis ...")
    ranking = explain_mod.explain(models, splits, report_dir)
    print("      top driver per queue:")
    for queue in schema.QUEUES:
        top = ranking[(ranking["queue"] == queue) & (ranking["rank"] == 1)].iloc[0]
        print(f"        {queue:<20} <- {top['driver']}")

    print("\nmodel vs. rules on the held-out period:")
    print(f"      {'policy':<18} {'accuracy':>9} {'macro-F1':>9} {'delay-days':>11}")
    for name in ["rules_baseline", "logistic", "xgboost"]:
        m = results[name]
        print(
            f"      {name:<18} {m['accuracy']:>9.3f} {m['macro_f1']:>9.3f}"
            f" {m['mean_delay_days']:>11.3f}"
        )

    op = results["automation_operating_point"]
    print(
        f"\nautomation operating point (target {op['target_auto_accuracy']:.0%}"
        f" auto-route accuracy):\n"
        f"      auto-route {op['frac_auto']:.1%} of tickets at {op['auto_accuracy']:.1%}"
        f" accuracy (tau={op['tau']:.2f})\n"
        f"      human queue: {op['n_human']:,} of {results['n_test']:,} tickets;"
        f" hybrid delay {op['hybrid_delay_days']:.3f} days/ticket"
        f" vs {results['rules_baseline']['mean_delay_days']:.3f} all-rules"
        f" and 0.000 all-human"
    )
    print(f"\ndone. reports -> {report_dir}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="exception-triage", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="write a synthetic raw ticket extract")
    g.add_argument("--out", default="artifacts/data/raw.csv")
    g.add_argument("--n", type=int, default=40_000)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--clean", action="store_true", help="skip mess injection")
    g.set_defaults(func=cmd_generate)

    a = sub.add_parser("all", help="run the full pipeline: generate, clean, train, evaluate, explain")
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
