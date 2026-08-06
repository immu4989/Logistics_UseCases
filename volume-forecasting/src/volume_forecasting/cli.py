"""Command-line interface.

    volume-forecast generate --out artifacts/data/raw.csv   # synthetic raw feed
    volume-forecast all                                     # full pipeline
    volume-forecast fm-bench                                # Chronos-Bolt vs XGBoost
                                                            # (needs: pip install -e ".[fm]")

Artifacts land in ./artifacts by default:
    artifacts/data/      raw + cleaned tables
    artifacts/models/    trained models (joblib)
    artifacts/reports/   metrics.json, overlay/coverage plots, SHAP outputs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import cleaning, evaluate, explain, schema, synthetic, train as train_mod


def cmd_generate(args) -> None:
    df = synthetic.make_dataset(seed=args.seed, messy=not args.clean)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df):,} hub-day rows -> {out}")


def cmd_all(args) -> None:
    art = Path(args.artifacts)
    data_dir, model_dir, report_dir = art / "data", art / "models", art / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] generating raw volume feed (15 hubs, ~2.5 years, deliberately messy) ...")
    raw = synthetic.make_dataset(seed=args.seed, messy=True)
    raw.to_csv(data_dir / "raw.csv", index=False)
    print(f"      {len(raw):,} hub-day rows, {raw[schema.HUB_COL].nunique()} hubs")

    print("[2/5] cleaning ...")
    clean_df, report = cleaning.clean(raw)
    clean_df.to_csv(data_dir / "clean.csv", index=False)
    report.to_frame().to_csv(data_dir / "cleaning_report.csv", index=False)
    print(str(report))

    print("[3/5] training (seasonal naive + XGBoost point + quantiles, time-based split) ...")
    models, splits = train_mod.train(clean_df, train_mod.TrainConfig(seed=args.seed))
    path = train_mod.save(models, model_dir)
    print(f"      trained on dates <= {models.cutoff_date}; models -> {path}")

    print("[4/5] evaluating on the held-out final ~4 months (December peak included) ...")
    results = evaluate.evaluate_models(models, splits, report_dir)
    acc = results["accuracy"]
    print(f"      {'':<16} {'WAPE':>7} {'peak WAPE':>10} {'bias':>7}")
    for name in ["seasonal_naive", "xgboost_point"]:
        print(
            f"      {name:<16} {acc[name]['overall']['wape']:>6.1%} "
            f"{acc[name]['peak_season']['wape']:>9.1%} "
            f"{acc[name]['overall']['bias']:>+6.1%}"
        )
    qm = results["quantiles"]
    print(
        f"      P10-P90 coverage {qm['coverage_p10_p90']:.1%} overall, "
        f"{qm['coverage_p10_p90_peak']:.1%} in peak (nominal 80%)"
    )

    print("[5/5] SHAP driver analysis (P50 model) ...")
    ranking = explain.explain(models, splits, report_dir)
    print("      top drivers (share of model explanation):")
    for _, row in ranking.head(6).iterrows():
        print(f"        {row['feature']:<26} {row['share_of_explanation']:.1%}")

    print("\nthe staffing decision (held-out window):")
    print(f"      {'policy':<16} {'understaffed days':>18} {'idle capacity':>15}")
    for row in results["staffing"]:
        print(
            f"      {row['policy']:<16} {row['understaffed_days_pct']:>17.1%} "
            f"{row['idle_capacity_pct']:>14.1%}"
        )
    print(f"\ndone. reports -> {report_dir}")


def cmd_fm_bench(args) -> None:
    try:
        import chronos  # noqa: F401
    except ImportError:
        print(
            "error: the foundation-model benchmark needs the optional 'fm' extra "
            "(kept out of the default install because torch is a ~200MB wheel):\n"
            '    pip install -e ".[fm]"',
            file=sys.stderr,
        )
        sys.exit(1)

    from . import fm_benchmark

    keys = tuple(k.strip() for k in args.models.split(",") if k.strip())
    unknown = [k for k in keys if k not in fm_benchmark.CHRONOS_MODELS]
    if unknown:
        print(
            f"error: unknown model(s) {unknown}; "
            f"choose from {sorted(fm_benchmark.CHRONOS_MODELS)}",
            file=sys.stderr,
        )
        sys.exit(1)

    results = fm_benchmark.run_benchmark(keys, artifacts=args.artifacts, seed=args.seed)
    acc = results["accuracy"]
    print(f"\n      {'':<20} {'WAPE':>7} {'peak WAPE':>10} {'bias':>7} {'P10-P90 cov':>12}")
    for name in ["seasonal_naive", "xgboost", *[f"chronos_bolt_{k}" for k in keys]]:
        cov = acc[name].get("coverage_p10_p90")
        cov_str = f"{cov:>11.1%}" if cov is not None else f"{'-':>11}"
        print(
            f"      {name:<20} {acc[name]['overall']['wape']:>6.1%} "
            f"{acc[name]['peak_season']['wape']:>9.1%} "
            f"{acc[name]['overall']['bias']:>+6.1%} {cov_str}"
        )
    for name, wl in results["per_hub_wins_vs_xgboost"].items():
        print(
            f"      per-hub WAPE: {name} wins {wl['chronos_wins']}, "
            f"XGBoost wins {wl['xgboost_wins']} of {len(results['per_hub'])} hubs"
        )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="volume-forecast", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="write a synthetic raw volume feed")
    g.add_argument("--out", default="artifacts/data/raw.csv")
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--clean", action="store_true", help="skip mess injection")
    g.set_defaults(func=cmd_generate)

    a = sub.add_parser("all", help="run the full pipeline: generate, clean, train, evaluate, explain")
    a.add_argument("--seed", type=int, default=7)
    a.add_argument("--artifacts", default="artifacts")
    a.set_defaults(func=cmd_all)

    f = sub.add_parser(
        "fm-bench",
        help='zero-shot Chronos-Bolt vs XGBoost on the held-out window (pip install -e ".[fm]")',
    )
    f.add_argument("--models", default="tiny,small", help="comma-separated: tiny,small")
    f.add_argument("--seed", type=int, default=7)
    f.add_argument("--artifacts", default="artifacts")
    f.set_defaults(func=cmd_fm_bench)

    args = p.parse_args(argv)
    try:
        args.func(args)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
