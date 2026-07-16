"""Command-line interface.

    latemile generate --out artifacts/data/raw.csv          # synthetic raw extract
    latemile all                                            # full pipeline, synthetic
    latemile all --source olist --olist-dir data/olist      # full pipeline, real data
    latemile score --input new_shipments.csv                # score with a saved model

Artifacts land in ./artifacts by default:
    artifacts/data/      raw + cleaned tables
    artifacts/models/    trained models (joblib)
    artifacts/reports/   metrics.json, lift/calibration plots, SHAP outputs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import cleaning, evaluate, explain, features, schema, synthetic, train as train_mod


def _load_raw(args) -> pd.DataFrame:
    if args.source == "synthetic":
        return synthetic.make_dataset(n=args.n, seed=args.seed, messy=True)
    if args.source == "olist":
        from . import olist

        return olist.load(args.olist_dir)
    raise ValueError(f"unknown source {args.source}")


def cmd_generate(args) -> None:
    df = synthetic.make_dataset(n=args.n, seed=args.seed, messy=not args.clean)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df):,} shipments -> {out}")


def cmd_all(args) -> None:
    art = Path(args.artifacts)
    data_dir, model_dir, report_dir = art / "data", art / "models", art / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] loading raw data (source={args.source}) ...")
    raw = _load_raw(args)
    raw.to_csv(data_dir / "raw.csv", index=False)
    print(f"      {len(raw):,} rows, miss rate {raw[schema.LABEL_COL].mean():.1%}")

    print("[2/5] cleaning ...")
    clean_df, report = cleaning.clean(raw)
    clean_df.to_csv(data_dir / "clean.csv", index=False)
    report.to_frame().to_csv(data_dir / "cleaning_report.csv", index=False)
    print(str(report))

    print("[3/5] training (logistic baseline + XGBoost, time-based split) ...")
    models, splits = train_mod.train(clean_df, train_mod.TrainConfig(seed=args.seed))
    path = train_mod.save(models, model_dir)
    print(f"      trained on ship dates <= {models.cutoff_date}; models -> {path}")

    print("[4/5] evaluating on the held-out period ...")
    results = evaluate.evaluate_models(models, splits, report_dir)
    for name in ["logistic_baseline", "xgboost"]:
        m = results[name]
        print(
            f"      {name:<18} PR-AUC {m['pr_auc']:.3f} | ROC-AUC {m['roc_auc']:.3f} | "
            f"top-decile lift {m['top_decile_lift']:.1f}x | "
            f"recall@{m['flag_frac']:.0%}-flagged {m['recall_at_flag']:.1%}"
        )

    print("[5/5] SHAP driver analysis ...")
    ranking = explain.explain(models, splits, report_dir)
    print("      top drivers (share of model explanation):")
    for _, row in ranking.head(8).iterrows():
        print(f"        {row['driver']:<28} {row['share_of_explanation']:.1%}")
    print(f"\ndone. reports -> {report_dir}")


def cmd_score(args) -> None:
    models = train_mod.load(Path(args.artifacts) / "models")
    raw = pd.read_csv(args.input, parse_dates=[schema.DATE_COL])
    clean_df, _ = cleaning.clean(raw)
    X = features.to_matrix(features.engineer(clean_df))
    X = X.reindex(columns=models.feature_columns, fill_value=0.0)
    out = clean_df[[schema.ID_COL]].copy()
    out["miss_probability"] = models.xgb.predict_proba(X)[:, 1]
    out = out.sort_values("miss_probability", ascending=False)
    dest = Path(args.out)
    out.to_csv(dest, index=False)
    print(f"scored {len(out):,} shipments -> {dest}")
    print(out.head(10).to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="latemile", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="write a synthetic raw shipment extract")
    g.add_argument("--out", default="artifacts/data/raw.csv")
    g.add_argument("--n", type=int, default=60_000)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--clean", action="store_true", help="skip mess injection")
    g.set_defaults(func=cmd_generate)

    a = sub.add_parser("all", help="run the full pipeline: load, clean, train, evaluate, explain")
    a.add_argument("--source", choices=["synthetic", "olist"], default="synthetic")
    a.add_argument("--olist-dir", default="data/olist")
    a.add_argument("--n", type=int, default=60_000, help="rows for synthetic source")
    a.add_argument("--seed", type=int, default=7)
    a.add_argument("--artifacts", default="artifacts")
    a.set_defaults(func=cmd_all)

    s = sub.add_parser("score", help="score new shipments with saved models")
    s.add_argument("--input", required=True)
    s.add_argument("--artifacts", default="artifacts")
    s.add_argument("--out", default="scored_shipments.csv")
    s.set_defaults(func=cmd_score)

    args = p.parse_args(argv)
    try:
        args.func(args)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
