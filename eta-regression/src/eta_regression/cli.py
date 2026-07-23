"""Command-line interface.

    eta-regression generate --out artifacts/data/raw.csv    # synthetic raw extract
    eta-regression all                                      # full pipeline, synthetic
    eta-regression all --source olist --olist-dir data/olist  # full pipeline, real data
    eta-regression score --input new_shipments.csv          # quote ETAs with a saved model

Artifacts land in ./artifacts by default (./artifacts-olist for the Olist
source, so a real-data run never overwrites the synthetic one):
    <artifacts>/data/      raw + cleaned tables
    <artifacts>/models/    trained models (joblib)
    <artifacts>/reports/   metrics.json, coverage/promise plots, SHAP outputs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import cleaning, evaluate, explain, features, schema, synthetic
from . import train as train_mod


def _load_raw(args) -> pd.DataFrame:
    if args.source == "synthetic":
        return synthetic.make_dataset(n=args.n, seed=args.seed, messy=True)
    if args.source == "olist":
        from . import olist

        return olist.load(args.olist_dir)
    raise ValueError(f"unknown source {args.source}")


def _label_max_days(source: str) -> float:
    if source == "olist":
        from . import olist

        return olist.LABEL_MAX_DAYS
    return cleaning.LABEL_MAX_DAYS


def cmd_generate(args) -> None:
    df = synthetic.make_dataset(n=args.n, seed=args.seed, messy=not args.clean)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df):,} shipments -> {out}")


def cmd_all(args) -> None:
    art = Path(args.artifacts or ("artifacts-olist" if args.source == "olist" else "artifacts"))
    data_dir, model_dir, report_dir = art / "data", art / "models", art / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] loading raw data (source={args.source}) ...")
    raw = _load_raw(args)
    raw.to_csv(data_dir / "raw.csv", index=False)
    print(f"      {len(raw):,} rows, mean transit {raw[schema.LABEL_COL].mean():.2f} days")

    print("[2/5] cleaning ...")
    clean_df, report = cleaning.clean(raw, label_max_days=_label_max_days(args.source))
    clean_df.to_csv(data_dir / "clean.csv", index=False)
    report.to_frame().to_csv(data_dir / "cleaning_report.csv", index=False)
    print(str(report))

    print("[3/5] training (linear baseline + XGBoost point + quantile models) ...")
    models, splits = train_mod.train(clean_df, train_mod.TrainConfig(seed=args.seed))
    path = train_mod.save(models, model_dir)
    print(f"      trained on ship dates <= {models.cutoff_date}; models -> {path}")

    print("[4/5] evaluating on the held-out period ...")
    results = evaluate.evaluate_models(models, splits, report_dir)
    for name in ["linear_baseline", "xgboost_point", "xgboost_p50"]:
        m = results[name]
        print(
            f"      {name:<16} MAE {m['mae_days']:.3f} d | RMSE {m['rmse_days']:.3f} d | "
            f"median APE {m['median_ape']:.1%}"
        )
    iv = results["interval_p10_p90"]
    print(
        f"      P10-P90 interval  coverage {iv['coverage']:.1%} (nominal 80%) | "
        f"mean width {iv['mean_width_days']:.2f} d"
    )
    ivc = results["interval_p10_p90_conformal"]
    print(
        f"      + conformal       coverage {ivc['coverage']:.1%} | "
        f"mean width {ivc['mean_width_days']:.2f} d (qhat {ivc['qhat_days']:+.2f} d)"
    )
    tr = results["promise_tradeoff"]
    print(
        f"      promise ceil(P90): {tr['quote_ceil_p90']['promised_by_rate']:.1%} kept, "
        f"{tr['quote_ceil_p90']['mean_promised_days']:.2f} d quoted "
        f"(+{tr['extra_days_for_p90_promise']:.2f} d vs ceil(P50) at "
        f"{tr['quote_ceil_p50']['promised_by_rate']:.1%} kept)"
    )

    print("[5/5] SHAP driver analysis (P50 model) ...")
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
    qpreds = train_mod.predict_quantiles(models, X, alphas=(0.1, 0.5, 0.9))

    out = clean_df[[schema.ID_COL]].copy()
    out["eta_p10"] = qpreds[0.1].round(2).to_numpy()
    out["eta_p50"] = qpreds[0.5].round(2).to_numpy()
    out["eta_p90"] = qpreds[0.9].round(2).to_numpy()
    out["promised_days"] = np.ceil(qpreds[0.9]).astype(int).to_numpy()
    out = out.sort_values("eta_p90", ascending=False)
    dest = Path(args.out)
    out.to_csv(dest, index=False)
    print(f"quoted {len(out):,} shipments -> {dest}")
    print(out.head(10).to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="eta-regression", description=__doc__)
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
    a.add_argument(
        "--artifacts",
        default=None,
        help="default: ./artifacts (synthetic) or ./artifacts-olist (olist)",
    )
    a.set_defaults(func=cmd_all)

    s = sub.add_parser("score", help="quote ETA quantiles for new shipments with saved models")
    s.add_argument("--input", required=True)
    s.add_argument("--artifacts", default="artifacts")
    s.add_argument("--out", default="quoted_etas.csv")
    s.set_defaults(func=cmd_score)

    args = p.parse_args(argv)
    try:
        args.func(args)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
