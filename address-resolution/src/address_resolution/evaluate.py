"""Evaluation for a matcher whose failure mode is a confident wrong door.

The metrics are the ones a delivery operation actually watches:

- **Auto-match rate (coverage)** — the share of labels that skip the human
  queue. Every point of coverage is money.
- **Auto-match precision** and its operational twin, the **false-match rate
  per 10k auto-matches** — THE headline risk number. A false match is not a
  modeling error, it is a parcel physically driven to the wrong door: a
  redelivery, a claim, sometimes a lost customer. It costs far more than the
  manual review it replaced, which is why the whole system is shaped around
  keeping this number tiny rather than maximizing accuracy.
- **No-match detection recall** — the ~8% of labels that correspond to no
  delivery point at all. The right answer for them is the review queue;
  auto-matching an orphan is always a wrong door.
- **Review-queue size** — the cost side. Humans clear the queue; its size is
  headcount.

The operating-point table sweeps the threshold to show what coverage costs at
99.5% and 99.9% auto-match precision. That exchange rate — coverage bought
with precision — is a business decision, not a modeling one, and the table is
how you hand it to the business.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import resolve
from .train import MatchRun

_BLUE, _RED, _GREEN, _GRAY = "#2b6cb0", "#c53030", "#2f855a", "#718096"


# --------------------------------------------------------------------------- core metrics
def point_metrics(dec: pd.DataFrame, threshold: float) -> dict:
    """Coverage / precision / false-match / orphan-recall at one threshold."""
    accepted = (dec["p_best"] >= threshold) & (dec["best_pair"] >= 0)
    n = len(dec)
    n_acc = int(accepted.sum())
    correct = dec["hit_best"] & accepted
    precision = float(correct.sum() / n_acc) if n_acc else 1.0
    orphans = dec["is_orphan"]
    return {
        "threshold": float(threshold),
        "coverage": n_acc / n,
        "precision": precision,
        "false_per_10k": (1.0 - precision) * 10_000,
        "orphan_recall": float((~accepted[orphans]).mean()) if orphans.any() else 1.0,
        "review_rate": 1.0 - n_acc / n,
        "n_auto": n_acc,
        "n_false": int(n_acc - correct.sum()),
    }


def precision_coverage_curve(dec: pd.DataFrame) -> pd.DataFrame:
    """Precision as coverage grows, best-scored labels first.

    Evaluated only at tie-group boundaries: identical feature rows share one
    probability (every unit in the same building, say), and a threshold rule
    admits such a cluster whole or not at all. Points inside a cluster would
    describe operating points no deployable threshold can reach.
    """
    has = dec["best_pair"].to_numpy() >= 0
    p = dec["p_best"].to_numpy()[has]
    h = dec["hit_best"].to_numpy()[has]
    order = np.argsort(-p)
    p_s, h_s = p[order], h[order]
    k = np.arange(1, len(p_s) + 1)
    boundary = np.r_[p_s[1:] != p_s[:-1], True]
    return pd.DataFrame(
        {
            "threshold": p_s[boundary],
            "coverage": k[boundary] / len(dec),
            "precision": np.cumsum(h_s)[boundary] / k[boundary],
        }
    )


def coverage_at_precision(curve: pd.DataFrame, target: float) -> dict:
    """Max coverage whose (tie-aware) running precision still holds the target."""
    ok = curve[curve["precision"] >= target]
    if ok.empty:
        return {"target": target, "coverage": 0.0, "threshold": 1.0}
    row = ok.iloc[-1]
    return {
        "target": target,
        "coverage": float(row["coverage"]),
        "threshold": float(row["threshold"]),
    }


def operating_table(dec: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows = [point_metrics(dec, t) for t in sorted(set(thresholds), reverse=True)]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- error taxonomy
def error_taxonomy(dec: pd.DataFrame) -> pd.DataFrame:
    """Per corruption type: did labels carrying it auto-match cleanly, land in
    review, or survive to a wrong door? Free because the generator recorded
    the ladder on every label."""
    rows = []
    tags = dec["corruptions"].astype(str).where(dec["corruptions"].astype(str) != "", "none")
    exploded = pd.DataFrame(
        {"tag": tags.str.split("|"), "accepted": dec["accepted"], "correct": dec["auto_correct"]}
    ).explode("tag")
    for tag, grp in exploded.groupby("tag"):
        rows.append(
            {
                "corruption": tag,
                "n_labels": len(grp),
                "auto_correct": int(grp["correct"].sum()),
                "auto_wrong": int((grp["accepted"] & ~grp["correct"]).sum()),
                "review": int((~grp["accepted"]).sum()),
            }
        )
    out = pd.DataFrame(rows).sort_values("n_labels", ascending=False).reset_index(drop=True)
    out["review_rate"] = out["review"] / out["n_labels"]
    return out


def review_reason(row: pd.Series) -> str:
    c = str(row["corruptions"])
    if c.startswith("orphan_"):
        return c
    n = 0 if c == "" else c.count("|") + 1
    if n >= 2:
        return "multiple corruptions"
    if n == 1:
        return c
    return "clean (model unsure)"


# --------------------------------------------------------------------------- evaluation entry
def evaluate(run: MatchRun, out_dir: str | Path) -> dict:
    """Score the held-out labels, sweep operating points, run both baselines,
    write metrics.json + plots. Returns the metrics dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test = ~run.is_train
    dec = run.decisions.loc[test].reset_index(drop=True)
    threshold = run.resolver.threshold

    curve = precision_coverage_curve(dec)
    cov995 = coverage_at_precision(curve, 0.995)
    cov999 = coverage_at_precision(curve, 0.999)
    default = point_metrics(dec, threshold)

    table = operating_table(
        dec, [0.30, 0.50, 0.70, 0.90, threshold, cov995["threshold"], cov999["threshold"]]
    )
    table.to_csv(out_dir / "operating_points.csv", index=False)

    # ---- baselines on the same held-out labels ------------------------------
    test_np = np.asarray(test)
    exact_all = resolve.exact_match_baseline(run.lnorm, run.pnorm)
    fuzzy_all = resolve.fuzzy_top1_baseline(run.lnorm, run.pnorm, run.li, run.pi, test_np)
    idx_of = {pid: i for i, pid in enumerate(run.points["point_id"])}
    true_idx = np.array(
        [idx_of.get(t, -1) for t in run.labels["true_point_id"].fillna("")], dtype=np.int64
    )
    baselines = {}
    for name, assigned in [("exact_match", exact_all[test_np]), ("fuzzy_top1", fuzzy_all[test_np])]:
        acc = assigned >= 0
        correct = acc & (assigned == true_idx[test_np])
        prec = float(correct.sum() / acc.sum()) if acc.any() else 1.0
        baselines[name] = {
            "coverage": float(acc.mean()),
            "precision": prec,
            "false_per_10k": (1.0 - prec) * 10_000,
        }

    taxonomy = error_taxonomy(dec)
    taxonomy.to_csv(out_dir / "error_taxonomy.csv", index=False)

    metrics = {
        "n_test_labels": int(len(dec)),
        "blocking_recall": run.blocking_recall,
        "default_threshold": float(threshold),
        "default_operating_point": default,
        "coverage_at_99_5_precision": cov995,
        "coverage_at_99_9_precision": cov999,
        "baselines": baselines,
        "orphan_share": float(dec["is_orphan"].mean()),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    _plot_precision_coverage(curve, default, baselines, out_dir / "precision_coverage.png")
    _plot_taxonomy(taxonomy, out_dir / "error_taxonomy.png")
    _plot_review_queue(dec, out_dir / "review_queue.png")
    return metrics


# --------------------------------------------------------------------------- plots
def _plot_precision_coverage(curve, default, baselines, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(curve["coverage"], curve["precision"], color=_BLUE, lw=2,
            label="scorer (threshold sweep)")
    ax.scatter([default["coverage"]], [default["precision"]], color=_BLUE, zorder=5, s=60)
    ax.annotate(
        f"  default threshold\n  {default['coverage']:.0%} coverage, "
        f"{default['false_per_10k']:.0f} wrong doors / 10k",
        (default["coverage"], default["precision"]),
        fontsize=8, va="top",
    )
    ex = baselines["exact_match"]
    ax.scatter([ex["coverage"]], [ex["precision"]], color=_GREEN, marker="s", s=70, zorder=5,
               label=f"exact match after normalization ({ex['coverage']:.0%} coverage)")
    fz = baselines["fuzzy_top1"]
    ax.scatter([fz["coverage"]], [fz["precision"]], color=_RED, marker="X", s=110, zorder=5,
               label=f"fuzzy top-1, no reject ({fz['false_per_10k']:.0f} wrong doors / 10k)")
    ax.annotate("no reject option:\nthis is the only point it has",
                (fz["coverage"], fz["precision"]), xytext=(-10, 14),
                textcoords="offset points", fontsize=8, color=_RED,
                ha="right", va="bottom")
    floor = min(0.985, fz["precision"] - 0.01)
    ax.set_ylim(floor, 1.0015)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Coverage (share of labels auto-matched)")
    ax.set_ylabel("Auto-match precision")
    ax.set_title("Coverage is bought with precision; the reject option sets the price")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_taxonomy(taxonomy: pd.DataFrame, path: Path) -> None:
    t = taxonomy.sort_values("review_rate")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    axes[0].barh(t["corruption"], 100 * t["review_rate"], color=_GRAY)
    axes[0].set_xlabel("% of labels with this corruption routed to review")
    axes[0].set_title("What lands in the review queue")
    axes[1].barh(t["corruption"], t["auto_wrong"], color=_RED)
    axes[1].set_xlabel("Auto-matched to the wrong door (count)")
    axes[1].set_title("What survives to a false match")
    for ax in axes:
        ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_review_queue(dec: pd.DataFrame, path: Path) -> None:
    queue = dec[~dec["accepted"]]
    reasons = queue.apply(review_reason, axis=1).value_counts()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    colors = [_RED if str(r).startswith("orphan_") else _BLUE for r in reasons.index]
    ax.barh(reasons.index.astype(str)[::-1], reasons.to_numpy()[::-1],
            color=list(reversed(colors)))
    ax.set_xlabel("Labels in the review queue")
    ax.set_title(
        f"Review-queue composition: {len(queue):,} of {len(dec):,} test labels\n"
        "(red = no valid delivery point exists)",
        fontsize=11,
    )
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
