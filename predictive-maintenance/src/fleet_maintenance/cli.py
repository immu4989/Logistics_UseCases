"""Command-line interface.

    fleet-maint generate --out artifacts/data/raw.csv    # synthetic raw telematics feed
    fleet-maint all                                      # full pipeline

Artifacts land in ./artifacts by default:
    artifacts/data/      raw + cleaned panels, cleaning report
    artifacts/models/    trained models (joblib)
    artifacts/reports/   metrics.json, policy table, plots, SHAP outputs, work order
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import cleaning, evaluate, explain, synthetic
from . import train as train_mod

POLICY_LABELS = {
    "mileage_rule": "Mileage rule",
    "logistic": "Logistic",
    "xgboost": "XGBoost",
}


def cmd_generate(args) -> None:
    df = synthetic.make_fleet(
        n_vehicles=args.vehicles, n_days=args.days, seed=args.seed, messy=not args.clean
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(
        f"wrote {len(df):,} vehicle-days ({args.vehicles} vehicles x {args.days} days) -> {out}"
    )


def cmd_all(args) -> None:
    art = Path(args.artifacts)
    data_dir, model_dir, report_dir = art / "data", art / "models", art / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] simulating fleet telematics ({args.vehicles} vehicles, {args.days} days) ...")
    raw = synthetic.make_fleet(
        n_vehicles=args.vehicles, n_days=args.days, seed=args.seed, messy=True
    )
    raw.to_csv(data_dir / "raw.csv", index=False)
    print(
        f"      {len(raw):,} vehicle-days, "
        f"{int(raw[synthetic.EVENT_COL].sum())} breakdowns, "
        f"label rate {raw[synthetic.LABEL_COL].mean():.1%}"
    )

    print("[2/5] cleaning ...")
    clean_df, report = cleaning.clean(raw)
    clean_df.to_csv(data_dir / "clean.csv", index=False)
    report.to_frame().to_csv(data_dir / "cleaning_report.csv", index=False)
    print(str(report))

    print("[3/5] training (mileage rule + logistic + XGBoost, time-based split) ...")
    models, splits = train_mod.train(clean_df, train_mod.TrainConfig(seed=args.seed))
    path = train_mod.save(models, model_dir)
    print(f"      trained on days <= {models.cutoff_date}; models -> {path}")

    print("[4/5] evaluating on the held-out final months ...")
    results = evaluate.evaluate_models(models, splits, report_dir)
    for policy in evaluate.POLICIES:
        m = results[policy]
        print(
            f"      {POLICY_LABELS[policy]:<13} PR-AUC {m['pr_auc']:.3f} | "
            f"ROC-AUC {m['roc_auc']:.3f} | median lead "
            f"{m['median_lead_days'] if m['median_lead_days'] is not None else '-'} days"
        )

    print("[5/5] SHAP driver analysis + work-order card ...")
    ranking = explain.explain(models, splits, report_dir)
    print("      top drivers (share of model explanation):")
    for _, row in ranking.head(6).iterrows():
        print(f"        {row['driver']:<28} {row['share_of_explanation']:.1%}")

    cap = results["capacity_frac"]
    print(f"\nAt workshop capacity ({cap:.0%} of the fleet per day), held-out period:")
    print(f"  {'policy':<14}{'precision':>10}{'recall':>9}{'lead(med)':>11}"
          f"{'actions':>9}{'averted':>9}{'net value':>12}")
    for policy in evaluate.POLICIES:
        m = results[policy]
        lead = f"{m['median_lead_days']:.0f}d" if m["median_lead_days"] is not None else "-"
        print(
            f"  {POLICY_LABELS[policy]:<14}{m['precision_at_capacity']:>10.3f}"
            f"{m['recall_at_capacity']:>9.3f}{lead:>11}{m['service_actions']:>9,}"
            f"{m['breakdowns_averted']:>9,}{m['net_value_usd']:>11,}$"
        )
    delta = results["xgboost"]["net_value_vs_mileage_rule_usd"]
    print(
        f"\nXGBoost vs the mileage rule at the same capacity: "
        f"{'+' if delta >= 0 else ''}{delta:,}$ over {results['n_test_days']} days "
        f"(${delta * 30 // max(results['n_test_days'], 1):,}/month)."
    )
    print(f"done. reports -> {report_dir}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="fleet-maint", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="write a synthetic raw telematics feed")
    g.add_argument("--out", default="artifacts/data/raw.csv")
    g.add_argument("--vehicles", type=int, default=600)
    g.add_argument("--days", type=int, default=545)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--clean", action="store_true", help="skip mess injection")
    g.set_defaults(func=cmd_generate)

    a = sub.add_parser(
        "all", help="run the full pipeline: simulate, clean, train, evaluate, explain"
    )
    a.add_argument("--vehicles", type=int, default=600)
    a.add_argument("--days", type=int, default=545)
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
