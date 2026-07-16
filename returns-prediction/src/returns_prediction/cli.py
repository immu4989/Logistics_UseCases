"""Command-line interface.

    returns-predict generate --out artifacts/data/raw.csv    # synthetic raw order extract
    returns-predict all                                      # full pipeline

Artifacts land in ./artifacts by default:
    artifacts/data/      raw + cleaned tables
    artifacts/models/    trained models (joblib)
    artifacts/reports/   metrics.json, cost/calibration plots, SHAP outputs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import cleaning, evaluate, explain, schema, synthetic
from . import train as train_mod


def cmd_generate(args) -> None:
    df = synthetic.make_dataset(n=args.n, seed=args.seed, messy=not args.clean)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df):,} orders -> {out}")


def cmd_all(args) -> None:
    art = Path(args.artifacts)
    data_dir, model_dir, report_dir = art / "data", art / "models", art / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] generating raw orders (synthetic, messy) ...")
    raw = synthetic.make_dataset(n=args.n, seed=args.seed, messy=True)
    raw.to_csv(data_dir / "raw.csv", index=False)
    print(f"      {len(raw):,} rows, return rate {raw[schema.LABEL_COL].mean():.1%}")

    print("[2/5] cleaning ...")
    clean_df, report = cleaning.clean(raw)
    clean_df.to_csv(data_dir / "clean.csv", index=False)
    report.to_frame().to_csv(data_dir / "cleaning_report.csv", index=False)
    print(str(report))

    print("[3/5] training (logistic baseline + XGBoost, time-based split) ...")
    models, splits = train_mod.train(clean_df, train_mod.TrainConfig(seed=args.seed))
    path = train_mod.save(models, model_dir)
    print(f"      trained on order dates <= {models.cutoff_date}; models -> {path}")

    print("[4/5] evaluating on the held-out period ...")
    results = evaluate.evaluate_models(models, splits, report_dir)
    for name in ["logistic_baseline", "xgboost"]:
        m = results[name]
        print(
            f"      {name:<18} PR-AUC {m['pr_auc']:.3f} | ROC-AUC {m['roc_auc']:.3f} | "
            f"Brier {m['brier']:.3f}"
        )
    cv = results["cost_view"]
    print(
        f"      top 10% by expected cost captures ${cv['captured_usd_expected_cost']:,.0f} "
        f"of return spend vs ${cv['captured_usd_raw_probability']:,.0f} by raw probability "
        f"and ${cv['captured_usd_random']:,.0f} at random"
    )
    print("      pre-ship intervention, net savings by targeting policy:")
    for row in results["intervention"]:
        print(
            f"        {row['policy']:<16} targeted {row['orders_targeted']:>5,} | "
            f"spend ${row['spend_usd']:>8,.0f} | saved ${row['expected_savings_usd']:>9,.0f} | "
            f"net ${row['net_savings_usd']:>9,.0f} ({row['roi']:+.1f}x ROI)"
        )

    print("[5/5] SHAP driver analysis ...")
    ranking = explain.explain(models, splits, report_dir)
    print("      top drivers (share of model explanation):")
    for _, row in ranking.head(8).iterrows():
        print(f"        {row['driver']:<24} {row['share_of_explanation']:.1%}")
    print(f"\ndone. reports -> {report_dir}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="returns-predict", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="write a synthetic raw order extract")
    g.add_argument("--out", default="artifacts/data/raw.csv")
    g.add_argument("--n", type=int, default=50_000)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--clean", action="store_true", help="skip mess injection")
    g.set_defaults(func=cmd_generate)

    a = sub.add_parser("all", help="run the full pipeline: generate, clean, train, evaluate, explain")
    a.add_argument("--n", type=int, default=50_000)
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
