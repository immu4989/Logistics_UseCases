"""Evaluation built around how a workshop actually consumes a risk score.

PR-AUC and ROC-AUC are reported for comparability, but the numbers that decide
whether this model earns bay time are operational:

- **Precision / recall at workshop capacity, day by day.** The shop can pull
  about 3% of the fleet off route per day. So each test day we flag exactly
  the top 3% for that day and score precision/recall *within the day*, then
  average across days. Pooling all test days and cutting one global threshold
  would inflate the numbers: the pooled policy spends its whole budget on a
  few catastrophic days, which a real shop cannot do — its bays do not roll
  over to next week.
- **Median warning lead time.** Days between the first capacity-flag and the
  breakdown it warned of. A perfect flag one day ahead still means a tow
  truck; five days ahead means a scheduled bay.
- **The money table.** An unplanned breakdown costs ~$2,800 (tow, roadside
  labor, missed routes); a planned service slot ~$600. Each policy is charged
  $600 per service action (a flagged vehicle is only actioned if it wasn't
  already serviced in the past 14 days) and credited $2,800 per breakdown its
  action preceded. What matters is the delta versus the mileage rule at the
  SAME capacity — that is the number the fleet manager takes to finance.

Caveat encoded here on purpose: this is offline replay. It assumes servicing a
flagged vehicle does not change the logged future (in reality it would avert
the failure, which is the point). Every predictive-maintenance backtest shares
this assumption; state it rather than hide it.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import synthetic

COST_UNPLANNED = 2_800  # tow + roadside labor + missed routes
COST_PLANNED = 600      # a scheduled bay slot
CAPACITY_FRAC = 0.03    # the shop can service ~3% of the fleet per day
ACTION_COOLDOWN_DAYS = 14  # a vehicle just serviced is not pulled again

POLICIES = ["mileage_rule", "logistic", "xgboost"]


def score_policies(models, splits) -> pd.DataFrame:
    """Return the test panel with one score column per policy."""
    test = splits["test"][
        [synthetic.ID_COL, synthetic.DATE_COL, synthetic.LABEL_COL, synthetic.EVENT_COL,
         "miles_since_maint"]
    ].copy()
    test["score_mileage_rule"] = models.mileage_rule.score(splits["test"])
    test["score_logistic"] = models.baseline.predict_proba(splits["X_test"])[:, 1]
    test["score_xgboost"] = models.xgb.predict_proba(splits["X_test"])[:, 1]
    return test.reset_index(drop=True)


def daily_flags(test: pd.DataFrame, policy: str, capacity: float = CAPACITY_FRAC) -> pd.Series:
    """Flag the top `capacity` share of vehicles independently on EACH test day."""
    col = f"score_{policy}"
    rank = test.groupby(synthetic.DATE_COL)[col].rank(method="first", ascending=False)
    n = test.groupby(synthetic.DATE_COL)[col].transform("size")
    k = np.maximum(1, np.ceil(capacity * n))
    return rank <= k


def capacity_metrics(test: pd.DataFrame, flags: pd.Series) -> dict:
    """Day-by-day precision/recall at capacity, averaged across test days.

    Averaged, not pooled: the daily flag budget is a hard constraint, so each
    day is its own experiment. Recall is averaged over days that had at least
    one true positive-window vehicle (recall is undefined on an all-quiet day).
    """
    df = pd.DataFrame(
        {"date": test[synthetic.DATE_COL], "y": test[synthetic.LABEL_COL], "flag": flags}
    )
    by_day = df.groupby("date").apply(
        lambda g: pd.Series(
            {
                "precision": g.loc[g["flag"], "y"].mean() if g["flag"].any() else np.nan,
                "recall": g.loc[g["y"] == 1, "flag"].mean() if (g["y"] == 1).any() else np.nan,
            }
        ),
        include_groups=False,
    )
    return {
        "precision_at_capacity": float(by_day["precision"].mean()),
        "recall_at_capacity": float(by_day["recall"].mean()),
    }


def lead_times(test: pd.DataFrame, flags: pd.Series) -> pd.DataFrame:
    """For each test-period breakdown, days between its first warning flag and the failure.

    A flag at day t warns of failures in t+1..t+14 (the label horizon), so the
    lookback for a failure on day f is [f-14, f-1]. Failures with no flag in
    that window were missed by the policy.
    """
    flagged = test.loc[flags.to_numpy(), [synthetic.ID_COL, synthetic.DATE_COL]]
    flag_days = flagged.groupby(synthetic.ID_COL)[synthetic.DATE_COL].apply(list).to_dict()

    rows = []
    failures = test[test[synthetic.EVENT_COL] == 1]
    for _, row in failures.iterrows():
        veh, fday = row[synthetic.ID_COL], row[synthetic.DATE_COL]
        window = [
            d for d in flag_days.get(veh, [])
            if pd.Timedelta(days=1) <= fday - d <= pd.Timedelta(days=synthetic.HORIZON_DAYS)
        ]
        lead = (fday - min(window)).days if window else np.nan
        rows.append({synthetic.ID_COL: veh, "failure_date": fday, "lead_days": lead})
    return pd.DataFrame(rows)


def money(test: pd.DataFrame, flags: pd.Series) -> dict:
    """Replay the flag stream as service actions and price the outcome.

    A flag becomes an action only if the vehicle was not actioned in the past
    ACTION_COOLDOWN_DAYS (the bay just serviced it). Each action costs
    COST_PLANNED; an action whose 14-day window really contained a breakdown
    is credited COST_UNPLANNED averted.
    """
    flagged = (
        test.loc[flags.to_numpy(), [synthetic.ID_COL, synthetic.DATE_COL, synthetic.LABEL_COL]]
        .sort_values(synthetic.DATE_COL, kind="stable")
    )
    last_action: dict[str, pd.Timestamp] = {}
    n_actions = 0
    n_averted = 0
    for _, row in flagged.iterrows():
        veh, day = row[synthetic.ID_COL], row[synthetic.DATE_COL]
        prev = last_action.get(veh)
        if prev is not None and (day - prev).days <= ACTION_COOLDOWN_DAYS:
            continue
        last_action[veh] = day
        n_actions += 1
        n_averted += int(row[synthetic.LABEL_COL])
    net = COST_UNPLANNED * n_averted - COST_PLANNED * n_actions
    return {
        "service_actions": n_actions,
        "breakdowns_averted": n_averted,
        "net_value_usd": int(net),
    }


def evaluate_models(models, splits, out_dir: str | Path, capacity: float = CAPACITY_FRAC) -> dict:
    """Score every policy on the held-out period; write metrics, tables and plots."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test = score_policies(models, splits)
    y = test[synthetic.LABEL_COL].to_numpy()
    n_days = test[synthetic.DATE_COL].nunique()

    results: dict = {
        "base_rate": float(y.mean()),
        "n_test_days": int(n_days),
        "n_test_rows": int(len(test)),
        "n_test_failures": int(test[synthetic.EVENT_COL].sum()),
        "capacity_frac": capacity,
        "cost_unplanned_usd": COST_UNPLANNED,
        "cost_planned_usd": COST_PLANNED,
    }
    leads_by_policy = {}
    for policy in POLICIES:
        s = test[f"score_{policy}"].to_numpy()
        flags = daily_flags(test, policy, capacity)
        leads = lead_times(test, flags)
        leads_by_policy[policy] = leads
        caught = leads["lead_days"].notna()
        results[policy] = {
            "pr_auc": float(average_precision_score(y, s)),
            "roc_auc": float(roc_auc_score(y, s)),
            **capacity_metrics(test, flags),
            "median_lead_days": float(leads.loc[caught, "lead_days"].median())
            if caught.any()
            else None,
            "failures_caught_frac": float(caught.mean()) if len(leads) else None,
            **money(test, flags),
        }
    for policy in ["logistic", "xgboost"]:
        results[policy]["net_value_vs_mileage_rule_usd"] = (
            results[policy]["net_value_usd"] - results["mileage_rule"]["net_value_usd"]
        )

    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    _write_policy_table(results, out_dir / "policy_table.csv")
    _plot_capacity_bars(results, capacity, out_dir / "capacity_precision_recall.png")
    _plot_lead_hist(leads_by_policy["xgboost"], out_dir / "lead_time_hist.png")
    _plot_vehicle_trace(models, splits, test, out_dir / "vehicle_trace.png")
    return results


def _write_policy_table(results: dict, path: Path) -> None:
    rows = []
    for policy in POLICIES:
        m = results[policy]
        rows.append(
            {
                "policy": policy,
                "pr_auc": round(m["pr_auc"], 3),
                "roc_auc": round(m["roc_auc"], 3),
                "precision_at_capacity": round(m["precision_at_capacity"], 3),
                "recall_at_capacity": round(m["recall_at_capacity"], 3),
                "median_lead_days": m["median_lead_days"],
                "service_actions": m["service_actions"],
                "breakdowns_averted": m["breakdowns_averted"],
                "net_value_usd": m["net_value_usd"],
                "net_vs_mileage_rule_usd": m.get("net_value_vs_mileage_rule_usd", 0),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _plot_capacity_bars(results: dict, capacity: float, path: Path) -> None:
    labels = ["Mileage rule", "Logistic", "XGBoost"]
    prec = [results[p]["precision_at_capacity"] for p in POLICIES]
    rec = [results[p]["recall_at_capacity"] for p in POLICIES]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    b1 = ax.bar(x - 0.18, prec, width=0.36, label="Precision", color="#2b6cb0")
    b2 = ax.bar(x + 0.18, rec, width=0.36, label="Recall", color="#dd6b20")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008,
                    f"{b.get_height():.2f}", ha="center", fontsize=9)
    ax.axhline(results["base_rate"], color="k", ls="--", lw=1,
               label=f"base rate {results['base_rate']:.1%}")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Day-by-day average on held-out months")
    ax.set_title(f"Who to pull off route: precision & recall at {capacity:.0%}/day capacity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_lead_hist(leads: pd.DataFrame, path: Path) -> None:
    caught = leads["lead_days"].dropna()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(caught, bins=np.arange(0.5, synthetic.HORIZON_DAYS + 1.5), color="#2b6cb0",
            edgecolor="white")
    med = caught.median()
    ax.axvline(med, color="#dd6b20", lw=2, label=f"median {med:.0f} days")
    ax.set_xlabel("Warning lead time (days between first flag and breakdown)")
    ax.set_ylabel("Breakdowns")
    n_missed = int(leads["lead_days"].isna().sum())
    ax.set_title(f"How much warning the shop gets (XGBoost; {n_missed} breakdowns unflagged)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_vehicle_trace(models, splits, test: pd.DataFrame, path: Path) -> None:
    """The money chart: one vehicle's sensors and risk score walking into a breakdown."""
    flags = daily_flags(test, "xgboost")
    leads = lead_times(test, flags)
    good = leads.dropna(subset=["lead_days"]).sort_values("lead_days", ascending=False)
    if good.empty:
        return
    pick = good.iloc[min(1, len(good) - 1)]
    veh, fail_date = pick[synthetic.ID_COL], pick["failure_date"]

    panel = splits["test"]
    window = panel[
        (panel[synthetic.ID_COL] == veh)
        & (panel[synthetic.DATE_COL] >= fail_date - pd.Timedelta(days=56))
        & (panel[synthetic.DATE_COL] <= fail_date + pd.Timedelta(days=5))
    ].sort_values(synthetic.DATE_COL)
    risk = pd.Series(
        models.xgb.predict_proba(splits["X_test"].loc[window.index])[:, 1],
        index=window.index,
    )
    veh_flags = test.loc[flags.to_numpy() & (test[synthetic.ID_COL] == veh).to_numpy()]
    first_flag = veh_flags.loc[
        (fail_date - veh_flags[synthetic.DATE_COL]).between(
            pd.Timedelta(days=1), pd.Timedelta(days=synthetic.HORIZON_DAYS)
        ),
        synthetic.DATE_COL,
    ].min()

    sensors = [
        ("vibration_index", "Vibration index"),
        ("avg_engine_temp", "Avg engine temp (C)"),
        ("oil_pressure", "Oil pressure (psi)"),
        ("battery_voltage", "Battery voltage (V)"),
    ]
    fig, axes = plt.subplots(5, 1, figsize=(8.5, 9.5), sharex=True)
    for ax, (col, label) in zip(axes[:4], sensors):
        ax.plot(window[synthetic.DATE_COL], window[col], color="#2b6cb0", lw=1.2)
        ax.plot(window[synthetic.DATE_COL], window[f"{col}_mean_7d"], color="#1a365d", lw=1.8,
                label="7d mean")
        ax.set_ylabel(label, fontsize=8)
        ax.legend(loc="upper left", fontsize=7)
    axes[4].plot(window[synthetic.DATE_COL], risk, color="#c53030", lw=1.8)
    axes[4].set_ylabel("Predicted 14d\nbreakdown risk", fontsize=8)
    for ax in axes:
        ax.axvline(fail_date, color="#c53030", ls="-", lw=1.4)
        if pd.notna(first_flag):
            ax.axvline(first_flag, color="#dd6b20", ls="--", lw=1.4)
    if pd.notna(first_flag):
        axes[0].text(first_flag, axes[0].get_ylim()[1], " first flag", color="#dd6b20",
                     fontsize=8, va="top")
    axes[0].text(fail_date, axes[0].get_ylim()[1], " breakdown", color="#c53030",
                 fontsize=8, va="top")
    lead = (fail_date - first_flag).days if pd.notna(first_flag) else 0
    fig.suptitle(
        f"Vehicle {veh}: flagged {lead} days before it would have failed on the road",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=150)
    plt.close(fig)
