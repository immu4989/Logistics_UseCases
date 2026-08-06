"""Command-line interface.

    uplift-model generate --out artifacts/data/raw.csv   # synthetic pilot extract
    uplift-model all                                     # full pipeline

Artifacts land in ./artifacts by default:
    artifacts/data/      raw + cleaned pilot log
    artifacts/reports/   metrics.json, qini/calibration/policy plots, tables
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import cleaning, evaluate, models, synthetic


def cmd_generate(args) -> None:
    df = synthetic.make_dataset(n=args.n, seed=args.seed, messy=not args.clean)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df):,} pilot shipments -> {out}")


def cmd_all(args) -> None:
    art = Path(args.artifacts)
    data_dir, report_dir = art / "data", art / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] generating randomized pilot log (n={args.n:,}, seed={args.seed}) ...")
    raw = synthetic.make_dataset(n=args.n, seed=args.seed, messy=True)
    raw.to_csv(data_dir / "raw.csv", index=False)
    treated_share = raw[synthetic.TREATMENT_COL].mean()
    print(
        f"      {len(raw):,} rows, {treated_share:.1%} treated (randomized), "
        f"miss rate {raw[synthetic.LABEL_COL].mean():.1%}"
    )

    print("[2/5] cleaning ...")
    clean_df, report = cleaning.clean(raw)
    clean_df.to_csv(data_dir / "clean.csv", index=False)
    report.to_frame().to_csv(data_dir / "cleaning_report.csv", index=False)
    print(str(report))

    print("[3/5] 70/30 hash split + fitting 4 estimators (risk / S / T / DR) ...")
    train_df, test_df = models.hash_split(clean_df)
    bundle = models.fit_all(train_df, seed=args.seed)
    print(f"      train {len(train_df):,} | test {len(test_df):,} shipments")

    print("[4/5] scoring the held-out set + Qini / AUUC ...")
    scores = models.predict_scores(bundle, test_df)
    metrics = evaluate.evaluate_all(test_df, scores, seed=args.seed, out_dir=report_dir)
    auuc = pd.DataFrame(metrics["auuc"])
    print("      AUUC (exact, normalized so oracle = 1.0):")
    for _, row in auuc.iterrows():
        print(f"        {row['method']:<16} {row['auuc_vs_oracle']:>7.3f}")

    print("[5/5] policy value + segment autopsy ...")
    policy = pd.DataFrame(metrics["policy_value"])
    policy["k"] = (policy["k"] * 100).astype(int).astype(str) + "%"
    print(
        policy[["method", "k", "treated", "misses_prevented", "spend_usd", "net_usd"]]
        .to_string(index=False)
    )

    autopsy = pd.DataFrame(metrics["segment_autopsy"])
    weather = autopsy[autopsy["segment"] == "weather_driven"].iloc[0]
    routing = autopsy[autopsy["segment"] == "routing_driven"].iloc[0]
    print(
        f"\n      the money insight: the weather segment is the RISKIEST "
        f"(control miss prob {weather['mean_control_miss_prob']:.0%} vs "
        f"{routing['mean_control_miss_prob']:.0%} for routing-driven) "
        f"but its true uplift is ~zero "
        f"({weather['mean_true_cate']:.4f} vs {routing['mean_true_cate']:.4f}) — "
        f"and dr_learner sees it "
        f"(predicted {weather['mean_dr_predicted_cate']:.4f} vs "
        f"{routing['mean_dr_predicted_cate']:.4f})."
    )
    print(f"\ndone. reports -> {report_dir}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="uplift-model", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="write a synthetic randomized-pilot extract")
    g.add_argument("--out", default="artifacts/data/raw.csv")
    g.add_argument("--n", type=int, default=40_000)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--clean", action="store_true", help="skip mess injection")
    g.set_defaults(func=cmd_generate)

    a = sub.add_parser("all", help="run the full pipeline: generate, clean, fit, evaluate")
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
