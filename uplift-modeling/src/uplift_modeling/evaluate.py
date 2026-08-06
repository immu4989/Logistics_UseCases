"""The uplift evaluation stack: Qini curves, calibration, policy value, autopsy.

Everything here runs on the HELD-OUT 30% and leans on the generator's exposed
``true_cate``, which makes two very different evaluations possible:

- OBSERVED Qini: the estimate you could compute on a real pilot. Rank by the
  method's score, walk down the ranking, and at each depth estimate misses
  prevented as depth * (control miss rate - treated miss rate) among the
  shipments ranked so far. It is unbiased under randomization but noisy,
  especially near the top of the ranking where the arms are still small.
- EXACT expected Qini: the cumulative sum of TRUE cate in score order — the
  quantity the observed curve is estimating, computable only because this is
  synthetic. The gap between the two curves on the same plot is a free lesson
  in how much variance a real uplift evaluation carries.

AUUC here is the area BETWEEN a method's exact Qini curve and the random
diagonal (untargeted treatment prevents misses at a constant rate, so its
curve is the straight line to the same endpoint — area above it is what
targeting itself contributes). Normalized so the oracle (ranking by true_cate)
scores 1.0; random lands at ~0 by construction.

Policy value prices the ranking: treat the top k%, $4 per treated shipment,
$35 per miss. A flat $35 keeps the arithmetic inspectable; the tiered
miss-cost model from intervention-optimization (base cost x tier multiplier
+ % of declared value) would slot in wherever ``MISS_COST_USD`` appears, and
turns this into value-weighted uplift targeting.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import synthetic

TREAT_COST_USD = 4.0
MISS_COST_USD = 35.0
POLICY_KS = [0.10, 0.20, 0.30]

CURVE_ORDER = ["oracle", "dr_learner", "t_learner", "s_learner", "risk_targeting", "random"]
CURVE_COLORS = {
    "oracle": "#2f2f2f",
    "dr_learner": "#2b6cb0",
    "t_learner": "#5aa9d6",
    "s_learner": "#8d99ae",
    "risk_targeting": "#c0392b",
    "random": "#c3cbd6",
}
SEGMENTS = ["routing_driven", "weather_driven", "overnight", "other"]


def qini_exact(scores: np.ndarray, true_cate: np.ndarray) -> np.ndarray:
    """Cumulative TRUE misses prevented when treating in descending score order."""
    order = np.argsort(-scores, kind="stable")
    return np.cumsum(true_cate[order])


def qini_observed(scores: np.ndarray, treated: np.ndarray, missed: np.ndarray) -> np.ndarray:
    """Classic observed-outcome Qini: estimated misses prevented at each depth.

    At depth n: n * (miss rate among controls in top-n - miss rate among
    treated in top-n). Valid because assignment was randomized; on targeted
    logs this difference confounds effect with selection and means nothing.
    """
    order = np.argsort(-scores, kind="stable")
    t, y = treated[order], missed[order]
    n = np.arange(1, len(t) + 1)
    nt = np.cumsum(t)
    nc = n - nt
    mt = np.cumsum(y * t)
    mc = np.cumsum(y * (1 - t))
    with np.errstate(divide="ignore", invalid="ignore"):
        prevented = n * (mc / np.maximum(nc, 1) - mt / np.maximum(nt, 1))
    prevented[(nt == 0) | (nc == 0)] = 0.0
    return prevented


def auuc(curve: np.ndarray) -> float:
    """Area between a Qini curve and the random diagonal, over fraction 0..1.

    Every curve ends at the same point (treat everyone -> total true effect),
    so subtracting the straight line to that endpoint isolates the value of
    the RANKING and puts random targeting at ~0.
    """
    x = np.arange(1, len(curve) + 1) / len(curve)
    diagonal = x * curve[-1]
    return float(np.trapezoid(curve - diagonal, x))


def policy_value(scores: np.ndarray, true_cate: np.ndarray, k: float) -> dict:
    """Exact net dollars of 'treat the top k% by this score', from true_cate."""
    n_treat = round(k * len(scores))
    top = np.argsort(-scores, kind="stable")[:n_treat]
    prevented = float(true_cate[top].sum())
    saved = MISS_COST_USD * prevented
    spend = TREAT_COST_USD * n_treat
    return {
        "treated": n_treat,
        "misses_prevented": round(prevented, 1),
        "saved_usd": round(saved, 2),
        "spend_usd": round(spend, 2),
        "net_usd": round(saved - spend, 2),
    }


def evaluate_all(
    test_df: pd.DataFrame,
    scores: pd.DataFrame,
    seed: int = 7,
    out_dir: str | Path = "artifacts/reports",
) -> dict:
    """Score every targeting policy on the held-out set; write plots + tables."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    true_cate = test_df["true_cate"].to_numpy()
    treated = test_df[synthetic.TREATMENT_COL].to_numpy()
    missed = test_df[synthetic.LABEL_COL].to_numpy()

    all_scores = scores.copy()
    all_scores["oracle"] = true_cate
    all_scores["random"] = np.random.default_rng(seed + 404).random(len(test_df))

    exact = {m: qini_exact(all_scores[m].to_numpy(), true_cate) for m in CURVE_ORDER}
    observed = {
        m: qini_observed(all_scores[m].to_numpy(), treated, missed) for m in CURVE_ORDER
    }
    oracle_auuc = auuc(exact["oracle"])
    auuc_table = pd.DataFrame(
        {
            "method": CURVE_ORDER,
            "auuc_exact": [round(auuc(exact[m]), 2) for m in CURVE_ORDER],
            "auuc_observed": [round(auuc(observed[m]), 2) for m in CURVE_ORDER],
            "auuc_vs_oracle": [round(auuc(exact[m]) / oracle_auuc, 3) for m in CURVE_ORDER],
        }
    )
    auuc_table.to_csv(out_dir / "auuc.csv", index=False)

    # --- policy value ---------------------------------------------------------
    policy_rows = []
    for m in ["dr_learner", "risk_targeting", "random", "oracle"]:
        for k in POLICY_KS:
            row = {"method": m, "k": k}
            row.update(policy_value(all_scores[m].to_numpy(), true_cate, k))
            policy_rows.append(row)
    policy_table = pd.DataFrame(policy_rows)
    policy_table.to_csv(out_dir / "policy_value.csv", index=False)

    # --- CATE calibration (dr_learner deciles) ---------------------------------
    dr = all_scores["dr_learner"].to_numpy()
    decile = pd.qcut(pd.Series(dr).rank(method="first"), 10, labels=False) + 1
    calib = (
        pd.DataFrame({"decile": decile, "predicted": dr, "true": true_cate})
        .groupby("decile")
        .mean()
        .reset_index()
    )
    calib.to_csv(out_dir / "cate_calibration.csv", index=False)

    # --- segment autopsy ---------------------------------------------------------
    autopsy = (
        pd.DataFrame(
            {
                "segment": test_df["segment_true"].to_numpy(),
                "true_cate": true_cate,
                "dr_predicted_cate": dr,
                "risk_score": all_scores["risk_targeting"].to_numpy(),
                "control_miss_prob": test_df["p0_true"].to_numpy(),
            }
        )
        .groupby("segment")
        .agg(
            n=("true_cate", "size"),
            mean_risk_score=("risk_score", "mean"),
            mean_control_miss_prob=("control_miss_prob", "mean"),
            mean_true_cate=("true_cate", "mean"),
            mean_dr_predicted_cate=("dr_predicted_cate", "mean"),
        )
        .reindex(SEGMENTS)
        .round(4)
        .reset_index()
    )
    autopsy.to_csv(out_dir / "segment_autopsy.csv", index=False)

    _plot_qini(exact, observed, out_dir / "qini_curves.png")
    _plot_calibration(calib, out_dir / "cate_calibration.png")
    _plot_policy_value(policy_table, out_dir / "policy_value.png")

    metrics = {
        "n_test": len(test_df),
        "seed": int(seed),
        "treat_cost_usd": TREAT_COST_USD,
        "miss_cost_usd": MISS_COST_USD,
        "auuc": auuc_table.to_dict(orient="records"),
        "policy_value": policy_table.to_dict(orient="records"),
        "segment_autopsy": autopsy.to_dict(orient="records"),
        "cate_calibration": calib.round(4).to_dict(orient="records"),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


# ---------------------------------------------------------------------------- plots


def _fractions(n: int) -> np.ndarray:
    return np.arange(1, n + 1) / n


def _plot_qini(exact: dict, observed: dict, path: Path) -> None:
    """Two panels: the exact expected curves, and what a real pilot would see."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, curves, title in [
        (axes[0], exact, "Exact expected Qini (from true CATE — synthetic luxury)"),
        (axes[1], observed, "Observed Qini (what a real pilot can compute)"),
    ]:
        for m in CURVE_ORDER:
            y = curves[m]
            x = _fractions(len(y))
            step = max(1, len(y) // 400)
            ax.plot(
                x[::step],
                y[::step],
                label=m if ax is axes[0] else None,
                color=CURVE_COLORS[m],
                ls="--" if m == "oracle" else "-",
                lw=1.8 if m in ("oracle", "dr_learner") else 1.2,
            )
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("Fraction of shipments treated (by each method's ranking)")
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("Cumulative misses prevented")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_calibration(calib: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(calib["decile"], calib["true"], color="#5aa9d6", label="mean TRUE cate")
    ax.plot(
        calib["decile"],
        calib["predicted"],
        color="#2f2f2f",
        marker="o",
        lw=1.5,
        label="mean predicted cate (dr_learner)",
    )
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(
        TREAT_COST_USD / MISS_COST_USD,
        color="#c0392b",
        ls="--",
        lw=1.2,
        label=f"break-even cate ({TREAT_COST_USD / MISS_COST_USD:.3f})",
    )
    ax.set_xticks(calib["decile"])
    ax.set_xlabel("Predicted-uplift decile (10 = highest predicted uplift)")
    ax.set_ylabel("Miss-probability reduction")
    ax.set_title("CATE calibration: does predicted uplift match true uplift?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_policy_value(policy_table: pd.DataFrame, path: Path) -> None:
    methods = ["dr_learner", "risk_targeting", "random", "oracle"]
    colors = ["#2b6cb0", "#c0392b", "#c3cbd6", "#2f2f2f"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.2
    xs = np.arange(len(POLICY_KS))
    for i, (m, c) in enumerate(zip(methods, colors)):
        vals = [
            policy_table.query("method == @m and k == @k")["net_usd"].iloc[0]
            for k in POLICY_KS
        ]
        ax.bar(xs + (i - 1.5) * width, vals, width, label=m, color=c)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"treat top {int(k * 100)}%" for k in POLICY_KS])
    ax.set_ylabel("Net $ (saved misses - treatment spend)")
    # NB: escape the dollar signs or matplotlib mathtext italicizes the title.
    ax.set_title(
        f"Policy value at \\${TREAT_COST_USD:.0f}/treatment, \\${MISS_COST_USD:.0f}/miss "
        "(exact, from true CATE)"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
