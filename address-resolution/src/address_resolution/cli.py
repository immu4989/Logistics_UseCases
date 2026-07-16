"""Command-line interface.

    address-resolve generate                    # write the synthetic city + labels
    address-resolve all                         # full pipeline: generate, match, evaluate, explain

Artifacts land in ./artifacts by default:
    artifacts/data/      delivery points + shipping labels
    artifacts/models/    trained resolver (joblib)
    artifacts/reports/   metrics.json, operating_points.csv, plots, rationale.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import evaluate as evaluate_mod
from . import explain, synthetic
from . import train as train_mod


def cmd_generate(args) -> None:
    points, labels = synthetic.make_dataset(seed=args.seed, n_labels=args.n)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    points.to_csv(out / "points.csv", index=False)
    labels.to_csv(out / "labels.csv", index=False)
    print(f"wrote {len(points):,} delivery points and {len(labels):,} labels -> {out}")


def cmd_all(args) -> None:
    art = Path(args.artifacts)
    data_dir, model_dir, report_dir = art / "data", art / "models", art / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] generating the synthetic city (seed={args.seed}) ...")
    points, labels = synthetic.make_dataset(seed=args.seed, n_labels=args.n)
    points.to_csv(data_dir / "points.csv", index=False)
    labels.to_csv(data_dir / "labels.csv", index=False)
    orphan_rate = (labels["true_point_id"] == "").mean()
    print(
        f"      {len(points):,} delivery points, {len(labels):,} labels "
        f"({orphan_rate:.1%} are true no-matches)"
    )

    print("[2/6] normalizing, blocking, scoring, deciding ...")
    cfg = train_mod.ResolverConfig(seed=args.seed, target_precision=args.target_precision)
    run = train_mod.run(points, labels, cfg)
    n_pairs = len(run.li)
    print(
        f"      {n_pairs:,} candidate pairs survived blocking "
        f"(~{n_pairs / len(labels):.0f} per label vs {len(points):,} points scored naively); "
        f"blocking recall {run.blocking_recall:.2%}"
    )

    print("[3/6] scorer trained on the hashed-out label split ...")
    n_train_pairs = int(run.is_train[run.li].sum())
    path = train_mod.save(run.resolver, model_dir)
    print(
        f"      logistic scorer over {len(run.resolver.feature_names)} features; "
        f"threshold {run.resolver.threshold:.3f} chosen for "
        f"{run.resolver.target_precision:.1%} train-side precision "
        f"({n_train_pairs:,} train-side pairs) -> {path}"
    )

    print("[4/6] resolving held-out labels + both baselines ...")
    print("[5/6] evaluating ...")
    metrics = evaluate_mod.evaluate(run, report_dir)

    print("[6/6] rationale cards from the logistic coefficients ...")
    rat = explain.write_rationale(run, report_dir)
    print(f"      -> {rat}")

    # ---- the numbers that decide whether this ships -------------------------
    d = metrics["default_operating_point"]
    fz = metrics["baselines"]["fuzzy_top1"]
    ex = metrics["baselines"]["exact_match"]
    table = evaluate_mod.operating_table(
        run.decisions[~run.decisions["is_train"]].reset_index(drop=True),
        [0.30, 0.50, 0.70, 0.90, run.resolver.threshold,
         metrics["coverage_at_99_5_precision"]["threshold"],
         metrics["coverage_at_99_9_precision"]["threshold"]],
    )
    print("\noperating points (held-out labels):")
    show = table[["threshold", "coverage", "precision", "false_per_10k", "orphan_recall",
                  "review_rate"]].copy()
    for col, fmt in [("threshold", "{:.3f}"), ("coverage", "{:.1%}"), ("precision", "{:.2%}"),
                     ("false_per_10k", "{:.1f}"), ("orphan_recall", "{:.1%}"),
                     ("review_rate", "{:.1%}")]:
        show[col] = show[col].map(fmt.format)
    print(show.to_string(index=False))

    print(
        f"\nheadline: at the default threshold the resolver auto-matches "
        f"{d['coverage']:.1%} of labels with {d['false_per_10k']:.1f} wrong doors per 10k "
        f"auto-matches and sends {d['orphan_recall']:.0%} of true no-matches to review."
    )
    print(
        f"fuzzy top-1 (no reject option) delivers {fz['false_per_10k']:.0f} wrong doors "
        f"per 10k; exact match is clean but covers only {ex['coverage']:.1%}."
    )
    print(f"\ndone. reports -> {report_dir}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="address-resolve", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="write the synthetic delivery points + labels")
    g.add_argument("--out", default="artifacts/data")
    g.add_argument("--n", type=int, default=20_000, help="number of shipping labels")
    g.add_argument("--seed", type=int, default=7)
    g.set_defaults(func=cmd_generate)

    a = sub.add_parser("all", help="run the full pipeline: generate, match, evaluate, explain")
    a.add_argument("--n", type=int, default=20_000, help="number of shipping labels")
    a.add_argument("--seed", type=int, default=7)
    a.add_argument("--target-precision", type=float, default=0.995,
                   help="train-side auto-match precision the default threshold must hold")
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
