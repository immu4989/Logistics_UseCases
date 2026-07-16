"""Command-line interface.

    network-anomaly generate --out artifacts/data/raw.csv   # synthetic daily lane feed
    network-anomaly all                                     # full pipeline

Artifacts land in ./artifacts by default:
    artifacts/data/      raw + cleaned daily lane tables
    artifacts/reports/   metrics.json, alarms.csv, alarms.md, plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import cleaning, detect, evaluate, explain, schema, synthetic


def cmd_generate(args) -> None:
    df = synthetic.make_dataset(seed=args.seed, messy=not args.clean)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df):,} lane-day rows -> {out}")


def cmd_all(args) -> None:
    art = Path(args.artifacts)
    data_dir, report_dir = art / "data", art / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] generating one year of daily lane data (seed={args.seed}) ...")
    raw = synthetic.make_dataset(seed=args.seed, messy=True)
    raw.to_csv(data_dir / "raw.csv", index=False)
    n_anom = len(synthetic.INJECTED_ANOMALIES)
    vols = sorted(synthetic.LANE_BASE_VOLUME.values())
    print(
        f"      {len(raw):,} rows | {raw[schema.LANE_COL].nunique()} lanes | "
        f"volumes {vols[0]:.0f}-{vols[-1]:.0f}/day | "
        f"{n_anom} injected anomalies ({sum(a['type'] == 'step' for a in synthetic.INJECTED_ANOMALIES)} steps, "
        f"{sum(a['type'] == 'ramp' for a in synthetic.INJECTED_ANOMALIES)} ramps, "
        f"{sum(a['type'] == 'spike' for a in synthetic.INJECTED_ANOMALIES)} spikes)"
    )

    print("[2/5] cleaning ...")
    clean_df, report = cleaning.clean(raw)
    clean_df.to_csv(data_dir / "clean.csv", index=False)
    report.to_frame().to_csv(data_dir / "cleaning_report.csv", index=False)
    print(str(report))

    print("[3/5] detecting (EB shrinkage + global-effect removal + CUSUM) ...")
    cfg = detect.DetectorConfig(k=args.k, h=args.h)
    res = detect.detect(clean_df, cfg)
    print(
        f"      config: baseline={cfg.baseline_days}d, k={cfg.k:g}, h={cfg.h:g} | "
        f"Beta prior fit: a={res.prior_alpha:.1f}, b={res.prior_beta:.1f} "
        f"(prior mean {res.prior_alpha / (res.prior_alpha + res.prior_beta):.1%})"
    )
    print(f"      {len(res.alarms)} CUSUM alarms | {len(res.monthly_alarms)} monthly-report flags")

    print("[4/5] scoring against injected ground truth ...")
    metrics, annotated, _ = evaluate.evaluate(res, report_dir)
    s, r, fa = metrics["steps"], metrics["ramps"], metrics["false_alarms"]
    print("      headline results (CUSUM vs monthly report):")
    print(f"        step drifts   : {s['detected']}/{s['n']} detected, "
          f"mean delay {s['mean_delay_days']}d vs monthly {s['monthly_mean_delay_days']}d")
    print(f"        gradual ramps : {r['detected']}/{r['n']} detected, "
          f"mean delay {r['mean_delay_days']}d vs monthly {r['monthly_mean_delay_days']}d")
    print(f"        false alarms  : {fa['count']} in {fa['clean_lane_years']} clean-lane-years "
          f"= {fa['per_clean_lane_year']:.2f}/lane-year (monthly report: "
          f"{metrics['monthly_false_alarms']['per_clean_lane_year']:.2f})")
    sc = metrics["surge_check"]
    print(f"        surge check   : {sc['clean_lanes_alarming']} of {metrics['n_clean_lanes']} "
          f"clean lanes alarmed during the network-wide surge window")

    print("[5/5] writing incident cards ...")
    cards = explain.write_cards(res, annotated, report_dir)
    for card in cards[:3]:
        print()
        print("\n".join("      " + line for line in card.splitlines()))
    print(f"\ndone. reports -> {report_dir}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="network-anomaly", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="write a synthetic daily lane feed")
    g.add_argument("--out", default="artifacts/data/raw.csv")
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--clean", action="store_true", help="skip mess injection")
    g.set_defaults(func=cmd_generate)

    a = sub.add_parser("all", help="full pipeline: generate, clean, detect, score, explain")
    a.add_argument("--seed", type=int, default=7)
    a.add_argument("--k", type=float, default=0.5, help="CUSUM slack (std errors/day)")
    a.add_argument("--h", type=float, default=5.5, help="CUSUM alarm threshold")
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
