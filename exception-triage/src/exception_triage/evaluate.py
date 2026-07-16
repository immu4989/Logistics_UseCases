"""Evaluation for a triage router: accuracy is not the product metric.

Three views, in increasing order of how much they matter:

- **Accuracy / macro-F1** — the comparable numbers, reported for the rules
  router, the logistic baseline, and XGBoost alike.
- **Cost-weighted delay** — queues are not symmetric. Sending a damage claim
  to hold-and-monitor burns the evidence window; sending a hold ticket to a
  dispatcher wastes half a touch. The COST_MATRIX prices every misroute in
  re-queue delay days, and policies are compared on mean delay per ticket.
- **The automation curve** — the actual product. Auto-route a ticket when the
  model's top class probability clears a threshold tau; humans take the rest.
  Sweeping tau trades coverage against auto-route accuracy, and the operating
  point (here: 97% auto-route accuracy) decides how many tickets never wait
  for a human at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from . import schema
from .train import rules_route

# ---------------------------------------------------------------------------
# What a misroute costs, in re-queue delay days: COST_MATRIX.loc[true, pred].
#
# The numbers encode how each queue fails when its ticket lands elsewhere:
#
# - damage_claims anywhere else costs 4.0: the package keeps moving, the
#   damage evidence window closes, and the claim ends up disputed.
# - customs_docs anywhere else costs 3.0: the shipment sits in bonded storage
#   accruing demurrage while the ticket bounces back.
# - address_correction misroutes cost 2.0, except to customer_callback (1.0):
#   the call usually surfaces the corrected address anyway.
# - reroute misroutes cost 1.5, except to hold_and_monitor (2.5): a genuine
#   misroute "monitored" is a parcel driving the wrong way with sign-off.
# - hold_and_monitor misroutes cost 0.5 everywhere: the ticket self-resolves
#   regardless; the cost is the wasted specialist touch (worst as a
#   dispatcher touch, hence reroute is the canonical example).
# - customer_callback misroutes cost 1.0, except to hold_and_monitor (2.0):
#   nobody calls, the customer waits, then escalates.
# ---------------------------------------------------------------------------
COST_MATRIX = pd.DataFrame(
    # pred:  addr  reroute customs damage  hold  callback
    [
        [0.0, 2.0, 2.0, 2.0, 2.0, 1.0],   # true address_correction
        [1.5, 0.0, 1.5, 1.5, 2.5, 1.5],   # true reroute
        [3.0, 3.0, 0.0, 3.0, 3.0, 3.0],   # true customs_docs
        [4.0, 4.0, 4.0, 0.0, 4.0, 4.0],   # true damage_claims
        [0.5, 0.5, 0.5, 0.5, 0.0, 0.5],   # true hold_and_monitor
        [1.0, 1.0, 1.0, 1.0, 2.0, 0.0],   # true customer_callback
    ],
    index=[
        "address_correction",
        "reroute",
        "customs_docs",
        "damage_claims",
        "hold_and_monitor",
        "customer_callback",
    ],
    columns=[
        "address_correction",
        "reroute",
        "customs_docs",
        "damage_claims",
        "hold_and_monitor",
        "customer_callback",
    ],
)


def mean_delay_days(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Average re-queue delay per ticket under the cost matrix."""
    costs = COST_MATRIX.to_numpy()
    t = pd.Index(COST_MATRIX.index).get_indexer(y_true)
    p = pd.Index(COST_MATRIX.columns).get_indexer(y_pred)
    return float(costs[t, p].mean())


def summarize(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Core metric dict for one routing policy (label-string arrays)."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "mean_delay_days": mean_delay_days(y_true, y_pred),
    }


def per_queue_table(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Precision / recall / support per resolution queue."""
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=schema.QUEUES, zero_division=0
    )
    return pd.DataFrame(
        {
            "queue": schema.QUEUES,
            "precision": prec.round(4),
            "recall": rec.round(4),
            "f1": f1.round(4),
            "support": support,
        }
    )


def automation_sweep(
    y_true: np.ndarray, probs: np.ndarray, classes: list[str]
) -> pd.DataFrame:
    """Sweep the confidence threshold tau: auto-route when max prob >= tau.

    Humans are assumed to route their share correctly (that is what the
    specialist queues are for), so the hybrid policy's delay comes only from
    auto-routed mistakes, averaged over ALL tickets.
    """
    max_prob = probs.max(axis=1)
    y_pred = np.array(classes)[probs.argmax(axis=1)]
    correct = y_pred == y_true
    costs = COST_MATRIX.to_numpy()
    t_idx = pd.Index(COST_MATRIX.index).get_indexer(y_true)
    p_idx = pd.Index(COST_MATRIX.columns).get_indexer(y_pred)
    ticket_cost = costs[t_idx, p_idx]

    rows = []
    for tau in np.arange(0.30, 0.999, 0.005):
        auto = max_prob >= tau
        n_auto = int(auto.sum())
        if n_auto == 0:
            continue
        rows.append(
            {
                "tau": round(float(tau), 3),
                "frac_auto": n_auto / len(y_true),
                "n_auto": n_auto,
                "n_human": len(y_true) - n_auto,
                "auto_accuracy": float(correct[auto].mean()),
                "hybrid_delay_days": float(ticket_cost[auto].sum() / len(y_true)),
            }
        )
    return pd.DataFrame(rows)


def pick_operating_point(sweep: pd.DataFrame, target_accuracy: float = 0.97) -> dict:
    """Highest-coverage tau whose auto-routed slice hits the accuracy target."""
    ok = sweep[sweep["auto_accuracy"] >= target_accuracy]
    row = ok.iloc[ok["frac_auto"].to_numpy().argmax()] if len(ok) else sweep.iloc[-1]
    return {
        "target_auto_accuracy": target_accuracy,
        "tau": float(row["tau"]),
        "frac_auto": float(row["frac_auto"]),
        "n_auto": int(row["n_auto"]),
        "n_human": int(row["n_human"]),
        "auto_accuracy": float(row["auto_accuracy"]),
        "hybrid_delay_days": float(row["hybrid_delay_days"]),
    }


def evaluate_models(models, splits, out_dir: str | Path, target_auto_accuracy: float = 0.97) -> dict:
    """Score rules + logistic + XGBoost on the held-out period; write reports."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_df, X_test = splits["test"], splits["X_test"]
    classes = models.classes
    y_true = test_df[schema.LABEL_COL].to_numpy()

    xgb_probs = models.xgb.predict_proba(X_test)
    preds = {
        "rules_baseline": rules_route(test_df),
        "logistic": np.array(classes)[models.baseline.predict(X_test)],
        "xgboost": np.array(classes)[xgb_probs.argmax(axis=1)],
    }

    results = {name: summarize(y_true, p) for name, p in preds.items()}

    queue_table = per_queue_table(y_true, preds["xgboost"])
    queue_table.to_csv(out_dir / "per_queue_precision_recall.csv", index=False)

    sweep = automation_sweep(y_true, xgb_probs, classes)
    sweep.to_csv(out_dir / "automation_curve.csv", index=False)
    op = pick_operating_point(sweep, target_auto_accuracy)

    results["automation_operating_point"] = op
    results["test_period_start"] = models.cutoff_date
    results["n_test"] = int(len(y_true))
    results["per_queue_xgboost"] = queue_table.to_dict(orient="records")
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))

    _plot_confusion(y_true, preds["xgboost"], out_dir / "confusion_heatmap.png")
    _plot_automation_curve(
        sweep, op, results["rules_baseline"]["accuracy"], out_dir / "automation_curve.png"
    )
    _plot_cost_comparison(results, op, out_dir / "cost_comparison.png")
    return results


def _plot_confusion(y_true, y_pred, path: Path) -> None:
    cm = pd.crosstab(
        pd.Series(y_true, name="true queue"),
        pd.Series(y_pred, name="predicted queue"),
        normalize="index",
    ).reindex(index=schema.QUEUES, columns=schema.QUEUES, fill_value=0.0)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(cm.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(schema.QUEUES)), schema.QUEUES, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(schema.QUEUES)), schema.QUEUES, fontsize=8)
    for i in range(len(schema.QUEUES)):
        for j in range(len(schema.QUEUES)):
            v = cm.iloc[i, j]
            ax.text(
                j, i, f"{v:.02f}", ha="center", va="center", fontsize=8,
                color="white" if v > 0.5 else "#1a202c",
            )
    ax.set_xlabel("Predicted queue")
    ax.set_ylabel("True queue")
    ax.set_title("XGBoost routing, row-normalized (held-out period)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="share of true queue")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_automation_curve(sweep: pd.DataFrame, op: dict, rules_acc: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(
        sweep["frac_auto"] * 100, sweep["auto_accuracy"] * 100,
        color="#2b6cb0", lw=2, label="auto-routed slice accuracy",
    )
    ax.axhline(
        rules_acc * 100, color="#c05621", ls="--", lw=1.5,
        label=f"rules baseline, all tickets ({rules_acc:.0%})",
    )
    ax.axhline(op["target_auto_accuracy"] * 100, color="k", ls=":", lw=1)
    ax.plot(op["frac_auto"] * 100, op["auto_accuracy"] * 100, "o", color="#276749", ms=9)
    ax.annotate(
        f"operating point\n{op['frac_auto']:.0%} auto-routed at "
        f"{op['auto_accuracy']:.1%} (tau={op['tau']:.2f})",
        xy=(op["frac_auto"] * 100, op["auto_accuracy"] * 100),
        xytext=(op["frac_auto"] * 100 - 5, op["auto_accuracy"] * 100 - 6),
        ha="right", fontsize=9,
        arrowprops={"arrowstyle": "->", "lw": 1},
    )
    ax.set_xlabel("% of tickets auto-routed (rest go to a human)")
    ax.set_ylabel("Accuracy on auto-routed tickets (%)")
    ax.set_title("The automation curve: coverage vs. auto-route accuracy")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_cost_comparison(results: dict, op: dict, path: Path) -> None:
    policies = [
        ("All-human\n(perfect, slow)", 0.0, "#718096"),
        ("Rules router\n(all tickets)", results["rules_baseline"]["mean_delay_days"], "#c05621"),
        ("Logistic\n(all tickets)", results["logistic"]["mean_delay_days"], "#6b46c1"),
        ("XGBoost\n(all tickets)", results["xgboost"]["mean_delay_days"], "#2b6cb0"),
        (
            f"Confidence gate\n({op['frac_auto']:.0%} auto + human rest)",
            op["hybrid_delay_days"],
            "#276749",
        ),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    names = [p[0] for p in policies]
    vals = [p[1] for p in policies]
    ax.bar(names, vals, color=[p[2] for p in policies])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f"{v:.02f}", ha="center", fontsize=9)
    ax.set_ylabel("Mean misroute delay (days per ticket)")
    ax.set_title("What each routing policy costs in re-queue delay")
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
