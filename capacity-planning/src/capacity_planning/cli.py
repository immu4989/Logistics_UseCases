"""Command-line interface.

    capacity-plan generate --out artifacts/data/raw.csv    # synthetic raw demand feed
    capacity-plan all                                      # full pipeline

Artifacts land in ./artifacts by default:
    artifacts/data/      raw + cleaned tables
    artifacts/models/    trained quantile models (joblib)
    artifacts/reports/   metrics.json, bookings.csv, plots, rationale.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import cleaning, decide, evaluate, explain, forecast, synthetic


def cmd_generate(args) -> None:
    df = synthetic.make_dataset(n_weeks=args.n_weeks, seed=args.seed, messy=not args.clean)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df):,} lane-weeks -> {out}")


def cmd_all(args) -> None:
    art = Path(args.artifacts)
    data_dir, model_dir, report_dir = art / "data", art / "models", art / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] generating the raw demand feed (synthetic, messy) ...")
    raw = synthetic.make_dataset(n_weeks=args.n_weeks, seed=args.seed, messy=True)
    raw.to_csv(data_dir / "raw.csv", index=False)
    print(f"      {len(raw):,} lane-week rows, {raw[synthetic.LANE_COL].nunique()} lanes")

    print("[2/6] cleaning ...")
    clean_df, report = cleaning.clean(raw)
    clean_df.to_csv(data_dir / "clean.csv", index=False)
    report.to_frame().to_csv(data_dir / "cleaning_report.csv", index=False)
    print(str(report))

    print("[3/6] training quantile forecasts (seasonal naive + GBM per fractile) ...")
    models, splits = forecast.train(clean_df, forecast.TrainConfig(seed=args.seed))
    path = forecast.save(models, model_dir)
    fr = ", ".join(f"{k}={v:.4f}" for k, v in forecast.QUANTILE_ROLES.items())
    print(f"      trained on weeks <= {models.cutoff_week} at fractiles {fr}")
    print(f"      models -> {path}")

    print("[4/6] booking decisions (newsvendor critical fractile) ...")
    q = decide.critical_fractile()
    print(
        f"      Cu = ${decide.SPOT_COST_USD - decide.COMMITTED_COST_USD:,.0f}, "
        f"Co = ${decide.COMMITTED_COST_USD - decide.SALVAGE_USD:,.0f}, q* = {q:.4f} "
        f"(tight-market scenario: q* = "
        f"{decide.critical_fractile(decide.SPOT_COST_TIGHT_USD):.4f})"
    )
    bookings = evaluate.build_bookings(models, splits)
    print(f"      booked {bookings[synthetic.WEEK_COL].nunique()} test weeks "
          f"x {bookings[synthetic.LANE_COL].nunique()} lanes")

    print("[5/6] counterfactual cost evaluation (common random numbers) ...")
    comparison, sensitivity = evaluate.evaluate_all(
        bookings, seed=args.seed, out_dir=report_dir, n_reps=args.reps
    )
    cols = ["policy", "total_cost_usd", "committed_trailers_per_week", "spot_teq",
            "empty_teq", "service_level", "savings_vs_habit_pct", "excess_vs_oracle_pct"]
    print(comparison[cols].to_string(index=False))
    print("      tight-market sensitivity (spot $3,200):")
    print(sensitivity[["policy", "total_cost_usd", "committed_trailers_per_week",
                       "savings_vs_habit_pct"]].to_string(index=False))

    print("[6/6] per-lane booking rationale ...")
    table = explain.write_rationale(bookings, report_dir)
    for _, r in table.iterrows():
        print(f"      {r['case']:<22} {r['lane']}: booked {r['booked_newsvendor']} "
              f"(habit {r['booked_last_year']}), habit penalty "
              f"${r['habit_penalty_usd_per_week']:,.0f}/wk")

    nv = comparison.loc[comparison["policy"] == "newsvendor_model"].iloc[0]
    print(
        f"\nheadline: newsvendor booking saves ${nv['savings_vs_habit_usd']:,.0f} "
        f"({nv['savings_vs_habit_pct']:.1f}%) vs the book-last-year habit over the "
        f"16 test weeks, within {nv['excess_vs_oracle_pct']:.1f}% of the oracle floor."
    )
    print(f"reports -> {report_dir}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="capacity-plan", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="write a synthetic raw lane-demand feed")
    g.add_argument("--out", default="artifacts/data/raw.csv")
    g.add_argument("--n-weeks", type=int, default=108)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--clean", action="store_true", help="skip mess injection")
    g.set_defaults(func=cmd_generate)

    a = sub.add_parser(
        "all", help="run the full pipeline: generate, clean, forecast, decide, evaluate, explain"
    )
    a.add_argument("--n-weeks", type=int, default=108)
    a.add_argument("--seed", type=int, default=7)
    a.add_argument("--reps", type=int, default=evaluate.N_REPS,
                   help="demand replications for the counterfactual evaluation")
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
