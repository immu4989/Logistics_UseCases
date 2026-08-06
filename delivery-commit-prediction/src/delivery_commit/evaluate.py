"""Evaluation for an imbalanced operational classifier.

Metrics chosen for how the model is actually used:

- **PR-AUC** — the headline number. With ~10% positives, ROC-AUC flatters
  everyone; average precision reflects how clean the "intervene on these"
  queue really is.
- **ROC-AUC, Brier** — comparability and probability quality.
- **Lift by decile** — the operations conversation: "the riskiest 10% of
  shipments contains X% of all misses". This is what earns the model a seat
  in the morning ops call.
- **Recall / precision at a capacity threshold** — interventions (reroute,
  upgrade, proactive customer notice) have a daily capacity; we report
  precision and recall when flagging exactly the top k% the ops team can act on.
- **Calibration curve** — if you tell a customer "82% risk of missing the
  commit", it should miss about 82% of the time.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import conformal as conformal_mod


def decile_lift(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"y": y_true, "p": y_prob}).sort_values("p", ascending=False)
    df["decile"] = np.ceil((np.arange(len(df)) + 1) / len(df) * 10).astype(int)
    base_rate = df["y"].mean()
    out = (
        df.groupby("decile")
        .agg(shipments=("y", "size"), misses=("y", "sum"), miss_rate=("y", "mean"))
        .reset_index()
    )
    out["lift"] = out["miss_rate"] / base_rate
    out["cum_miss_capture"] = out["misses"].cumsum() / out["misses"].sum()
    return out


def summarize(y_true: np.ndarray, y_prob: np.ndarray, flag_frac: float = 0.10) -> dict:
    """Core metric dict for one model."""
    threshold = float(np.quantile(y_prob, 1 - flag_frac))
    y_flag = (y_prob >= threshold).astype(int)
    lift = decile_lift(y_true, y_prob)
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "base_miss_rate": float(np.mean(y_true)),
        "flag_frac": flag_frac,
        "flag_threshold": threshold,
        "precision_at_flag": float(precision_score(y_true, y_flag)),
        "recall_at_flag": float(recall_score(y_true, y_flag)),
        "top_decile_lift": float(lift.loc[lift["decile"] == 1, "lift"].iloc[0]),
        "top_2_decile_miss_capture": float(
            lift.loc[lift["decile"] == 2, "cum_miss_capture"].iloc[0]
        ),
    }


def evaluate_models(models, splits, out_dir: str | Path, flag_frac: float = 0.10) -> dict:
    """Score baseline + XGBoost on the held-out period; write metrics + plots."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_test, y_test = splits["X_test"], splits["y_test"]
    probs = {
        "logistic_baseline": models.baseline.predict_proba(X_test)[:, 1],
        "xgboost": models.xgb.predict_proba(X_test)[:, 1],
    }

    results = {name: summarize(y_test, p, flag_frac) for name, p in probs.items()}
    results["test_period_start"] = models.cutoff_date
    results["n_test"] = int(len(y_test))

    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    decile_lift(y_test, probs["xgboost"]).to_csv(out_dir / "lift_by_decile.csv", index=False)

    _plot_calibration(y_test, probs, out_dir / "calibration.png")
    _plot_lift(y_test, probs["xgboost"], out_dir / "lift_by_decile.png")
    return results


def conformal_report(models, splits, out_dir: str | Path) -> dict | None:
    """Evaluate the conformal layer on the held-out period.

    Writes crc_table.csv (target vs realized miss capture and the flag-rate
    cost at each alpha) plus one plot, and returns {"table", "brier"} for the
    CLI. Returns None for pre-conformal model artifacts (getattr, because old
    pickles predate the fields).

    Reminder when reading the table: the CRC guarantee is on the EXPECTED FNR
    under exchangeability of calibration and deployment shipments; a single
    test period's realized capture is one draw around the target, not the
    guarantee itself.
    """
    calibrator = getattr(models, "calibrator", None)
    thresholds = getattr(models, "crc_thresholds", None)
    if calibrator is None or not thresholds:
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_test, y_test = splits["X_test"], splits["y_test"]
    p_raw = models.xgb.predict_proba(X_test)[:, 1]
    p_cal = calibrator.predict_proba(X_test)[:, 1]

    # Thresholds live on the raw-score scale (see train.py: CRC on the raw
    # ranking, isotonic only for probability levels).
    table = conformal_mod.crc_report(thresholds, p_raw, y_test)
    table.to_csv(out_dir / "crc_table.csv", index=False)
    brier = {
        "raw": float(brier_score_loss(y_test, p_raw)),
        "calibrated": float(brier_score_loss(y_test, p_cal)),
    }
    _plot_crc(table, out_dir / "crc_guarantee.png")
    return {"table": table, "brier": brier}


def _plot_crc(table: pd.DataFrame, path: Path) -> None:
    """Realized miss capture vs the CRC target, with the flag-rate cost."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(table))
    ax.bar(x, table["realized_capture"], width=0.55, color="#2b6cb0", label="realized capture")
    ax.plot(
        x,
        table["target_capture"],
        ls="none",
        marker="_",
        ms=42,
        mew=2.5,
        color="#c05621",
        label="guaranteed expected capture (1 - alpha)",
    )
    for i, row in table.iterrows():
        ax.text(
            i,
            row["realized_capture"] + 0.012,
            f"{row['realized_capture']:.1%}",
            ha="center",
            fontsize=9,
        )
        ax.text(
            i,
            0.04,
            f"flags {row['flag_rate']:.0%}\nof shipments",
            ha="center",
            fontsize=8,
            color="white",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"alpha = {a:.2f}" for a in table["alpha"]])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Share of actual misses flagged")
    ax.set_title("CRC guarantee vs held-out period: capture and its flag-rate cost", pad=30)
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2, fontsize=8, frameon=False
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_calibration(y_test, probs: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, p in probs.items():
        frac_pos, mean_pred = calibration_curve(y_test, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    lim = max(max(p.max() for p in probs.values()), 0.5) * 1.05
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Predicted miss probability")
    ax.set_ylabel("Observed miss rate")
    ax.set_title("Calibration (held-out period)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_lift(y_test, y_prob, path: Path) -> None:
    lift = decile_lift(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(lift["decile"], lift["lift"], color="#2b6cb0")
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.set_xticks(lift["decile"])
    ax.set_xlabel("Risk decile (1 = highest predicted risk)")
    ax.set_ylabel("Lift vs. base miss rate")
    ax.set_title("XGBoost lift by risk decile (held-out period)")
    for _, row in lift.iterrows():
        ax.text(row["decile"], row["lift"] + 0.05, f"{row['lift']:.1f}x", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
