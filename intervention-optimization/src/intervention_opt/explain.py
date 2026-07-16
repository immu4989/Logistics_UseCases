"""Per-decision rationale: the audit trail an ops manager signs off on.

No SHAP here, on purpose. The upstream risk model owns "why is this shipment
risky" (use case 1 ships that explanation). What THIS layer must explain is
"why did we spend money here and not there" — and for an expected-value
policy the honest explanation is the arithmetic itself. Every decision is
four multiplications an operator can check on a napkin:

    expected savings = miss_cost x p_hat x effect - action_cost

Outputs:

- ``rationale.md``        — a handful of representative decisions with the
  full arithmetic and a one-line plain-English why: the highest-stakes
  upgrade, a cheap notify-only case, and the case audits always ask about —
  a scary score the policy deliberately did NOT spend on.
- ``decision_policy.csv`` — the complete EV matrix (every action) for those
  example shipments, so a reviewer can verify no better option existed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import interventions, policy


def _pick_examples(merged: pd.DataFrame) -> list[tuple[str, pd.Series, str]]:
    """Choose (label, row, why) examples covering the policy's decision modes."""
    examples: list[tuple[str, pd.Series, str]] = []

    upgrades = merged[merged["action"] == "upgrade_service"]
    if len(upgrades):
        row = upgrades.loc[upgrades["expected_savings"].idxmax()]
        examples.append(
            (
                "highest-stakes upgrade",
                row,
                f"{row['p_hat']:.0%} risk on a {row['customer_tier']} shipment with a "
                f"${row['miss_cost_usd']:,.0f} miss cost: the strongest lever pays for "
                f"itself many times over.",
            )
        )

    notifies = merged[merged["action"] == "notify_customer"]
    if len(notifies):
        row = notifies.loc[notifies["expected_savings"].idxmax()]
        examples.append(
            (
                "notify-only",
                row,
                "physical intervention is not worth it here, but fifty cents to make "
                "the likely miss a cheap, warned one is.",
            )
        )

    # The audit favorite: a high score the policy refused to upgrade because
    # the shipment is not worth the spend. Risk is not the objective.
    skipped = merged[
        (merged["p_hat"] >= 0.4)
        & (merged["ev_upgrade_service"] < 0)
        & (merged["action"].isin(["do_nothing", "notify_customer"]))
    ]
    if len(skipped):
        row = skipped.loc[skipped["p_hat"].idxmax()]
        examples.append(
            (
                "high risk, deliberately not upgraded",
                row,
                f"{row['p_hat']:.0%} risk looks scary, but the miss only costs "
                f"${row['miss_cost_usd']:,.0f}: a $15 upgrade would destroy money "
                f"(EV ${row['ev_upgrade_service']:,.2f}).",
            )
        )
    return examples


def write_rationale(
    df: pd.DataFrame,
    greedy_decisions: pd.DataFrame,
    out_dir: str | Path,
) -> pd.DataFrame:
    """Write rationale.md + decision_policy.csv; return the example table."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ev = policy.ev_matrix(df).add_prefix("ev_")
    merged = pd.concat(
        [
            df.reset_index(drop=True),
            greedy_decisions[["action", "expected_savings", "spent"]].reset_index(drop=True),
            ev.reset_index(drop=True),
        ],
        axis=1,
    )
    merged["miss_cost_usd"] = interventions.miss_cost(
        merged["customer_tier"], merged["declared_value_usd"]
    )

    examples = _pick_examples(merged)
    rows = []
    for label, row, why in examples:
        act = interventions.BY_NAME[row["action"]]
        rows.append(
            {
                "case": label,
                "shipment_id": row["shipment_id"],
                "p_hat": round(float(row["p_hat"]), 3),
                "customer_tier": row["customer_tier"],
                "declared_value_usd": round(float(row["declared_value_usd"]), 2),
                "miss_cost_usd": round(float(row["miss_cost_usd"]), 2),
                "action": act.name,
                "expected_cost_reduction_usd": round(
                    float(row["miss_cost_usd"] * row["p_hat"] * act.effect), 2
                ),
                "action_cost_usd": act.cost_usd,
                "expected_net_savings_usd": round(float(row[f"ev_{act.name}"]), 2),
                "why": why,
            }
        )
    table = pd.DataFrame(rows)

    lines = [
        "# Why the policy spent (or refused to spend) on these shipments",
        "",
        "Every number below is checkable by hand: "
        "`expected savings = miss_cost x p_hat x effect - action_cost`.",
        "",
        "| Case | Shipment | p_hat | Tier | Value | Miss cost | Action | "
        "Exp. cost reduction | Action cost | Exp. net savings | Why |",
        "|---|---|---:|---|---:|---:|---|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['case']} | {r['shipment_id']} | {r['p_hat']:.0%} | {r['customer_tier']} "
            f"| ${r['declared_value_usd']:,.0f} | ${r['miss_cost_usd']:,.0f} | {r['action']} "
            f"| ${r['expected_cost_reduction_usd']:,.2f} | ${r['action_cost_usd']:.2f} "
            f"| ${r['expected_net_savings_usd']:,.2f} | {r['why']} |"
        )
    lines += [
        "",
        "_The full expected-value matrix for these shipments (every candidate "
        "action, so a reviewer can verify no better option existed) is in "
        "`decision_policy.csv`._",
    ]
    (out_dir / "rationale.md").write_text("\n".join(lines))

    # Full EV matrix for the example shipments: one row per shipment x action.
    ex_ids = [r["shipment_id"] for r in rows]
    ex = merged[merged["shipment_id"].isin(ex_ids)]
    matrix_rows = []
    for _, row in ex.iterrows():
        for act in interventions.CATALOG:
            matrix_rows.append(
                {
                    "shipment_id": row["shipment_id"],
                    "p_hat": round(float(row["p_hat"]), 4),
                    "miss_cost_usd": round(float(row["miss_cost_usd"]), 2),
                    "action": act.name,
                    "action_cost_usd": act.cost_usd,
                    "effect": act.effect,
                    "expected_net_savings_usd": round(float(row[f"ev_{act.name}"]), 2),
                    "chosen": act.name == row["action"],
                }
            )
    pd.DataFrame(matrix_rows).to_csv(out_dir / "decision_policy.csv", index=False)
    return table
