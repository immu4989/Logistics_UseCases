"""Per-quote rationale: the audit trail a pricing manager signs off on.

No SHAP in this use case, on purpose. The question a pricing desk audits is
not "which feature moved the score" but "why did we quote THIS number" — and
for an expected-margin policy the honest answer is the sweep itself: at the
cost-plus price the model predicted this win probability and this margin, at
the chosen price it predicted that, and the chosen one is bigger. Every row
below is arithmetic an analyst can recompute by hand:

    expected margin = (price - cost) x P_hat(accept | price)

Outputs ``rationale.md`` with three representative quotes: an elastic spot
quote priced DOWN to win, an inelastic premium-express quote priced UP, and
a quote pinned at the guardrail cap — the case audits always ask about,
because it is where the optimizer wanted to go further and the desk's rules
said no.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import price, train


def _cases(merged: pd.DataFrame) -> list[tuple[str, pd.Series, str]]:
    """Choose (label, row, why) examples covering the policy's decision modes."""
    cases: list[tuple[str, pd.Series, str]] = []
    used: set[str] = set()

    # 1. Elastic spot freight priced down: win probability bought cheaply.
    spot = merged[
        (merged["customer_segment"] == "spot")
        & (merged["model_price"] < merged["cost_plus_price"] * 0.97)
    ]
    if len(spot):
        row = spot.loc[(spot["em_model"] - spot["em_cost_plus"]).idxmax()]
        used.add(row["quote_id"])
        cases.append(
            (
                "elastic spot, priced down to win",
                row,
                f"spot shippers walk over small premiums: cutting the quote "
                f"{1 - row['model_price'] / row['cost_plus_price']:.0%} lifts the win odds from "
                f"{row['p_cost_plus']:.0%} to {row['p_model']:.0%}, and the extra volume more "
                f"than pays for the thinner markup.",
            )
        )

    # 2. Premium express priced up: urgency, not price, drives this acceptance.
    prem = merged[
        (merged["customer_segment"] == "premium")
        & (merged["urgency"] == "express")
        & (merged["model_price"] > merged["cost_plus_price"] * 1.03)
        & (~merged["quote_id"].isin(used))
    ]
    if len(prem):
        row = prem.loc[(prem["em_model"] - prem["em_cost_plus"]).idxmax()]
        used.add(row["quote_id"])
        cases.append(
            (
                "inelastic premium express, priced up",
                row,
                f"an express premium shipper needs the freight moved, not shopped: "
                f"{row['model_price'] / row['cost_plus_price'] - 1:.0%} more price costs only "
                f"{row['p_cost_plus'] - row['p_model']:.0%} of win probability.",
            )
        )

    # 3. Pinned at the guardrail cap: the optimizer wanted more, the desk said no.
    capped = merged[
        (merged["model_multiplier"] >= price.GUARDRAIL_CAP - 0.01)
        & (~merged["quote_id"].isin(used))
    ]
    if len(capped):
        row = capped.loc[capped["em_model"].idxmax()]
        cases.append(
            (
                "pinned at the guardrail cap",
                row,
                f"the sweep still slopes upward at 1.6x cost, so the optimizer would quote "
                f"higher — but the model is least trustworthy at prices it never saw in "
                f"training, and the {price.GUARDRAIL_CAP:.1f}x cap is doing exactly its job.",
            )
        )
    return cases


def write_rationale(
    test_df: pd.DataFrame,
    prices: dict[str, np.ndarray],
    models: train.TrainedModels,
    out_dir: str | Path,
) -> pd.DataFrame:
    """Write rationale.md; return the example table."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cost = test_df["our_cost_usd"].to_numpy()
    cp, mp = prices["cost_plus"], prices["model_pricing"]
    p_cp = train.predict_accept(models, test_df, cp)
    p_mp = train.predict_accept(models, test_df, mp)

    merged = test_df.reset_index(drop=True).copy()
    merged["cost_plus_price"] = cp
    merged["model_price"] = mp
    merged["model_multiplier"] = mp / cost
    merged["p_cost_plus"] = p_cp
    merged["p_model"] = p_mp
    merged["em_cost_plus"] = (cp - cost) * p_cp
    merged["em_model"] = (mp - cost) * p_mp

    rows = []
    for label, row, why in _cases(merged):
        rows.append(
            {
                "case": label,
                "quote_id": row["quote_id"],
                "segment": row["customer_segment"],
                "urgency": row["urgency"],
                "cost_usd": round(float(row["our_cost_usd"]), 2),
                "cost_plus_price_usd": round(float(row["cost_plus_price"]), 2),
                "model_price_usd": round(float(row["model_price"]), 2),
                "win_prob_cost_plus": round(float(row["p_cost_plus"]), 3),
                "win_prob_model": round(float(row["p_model"]), 3),
                "exp_margin_cost_plus_usd": round(float(row["em_cost_plus"]), 2),
                "exp_margin_model_usd": round(float(row["em_model"]), 2),
                "why": why,
            }
        )
    table = pd.DataFrame(rows)

    lines = [
        "# Why the model quoted these prices",
        "",
        "Every number below is checkable by hand: "
        "`expected margin = (price - cost) x P_hat(accept | price)`. "
        "Win probabilities are the trained model's, i.e. what the desk sees at quote time.",
        "",
        "| Case | Quote | Segment | Urgency | Cost | Cost-plus price | Model price | "
        "Win prob (cost-plus) | Win prob (model) | Exp. margin (cost-plus) | "
        "Exp. margin (model) | Why |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['case']} | {r['quote_id']} | {r['segment']} | {r['urgency']} "
            f"| ${r['cost_usd']:,.0f} | ${r['cost_plus_price_usd']:,.0f} "
            f"| ${r['model_price_usd']:,.0f} | {r['win_prob_cost_plus']:.0%} "
            f"| {r['win_prob_model']:.0%} | ${r['exp_margin_cost_plus_usd']:,.0f} "
            f"| ${r['exp_margin_model_usd']:,.0f} | {r['why']} |"
        )
    lines += [
        "",
        "_The full policy comparison, segment uplift table and demand-curve "
        "validation chart are alongside this file in `artifacts/reports/`._",
    ]
    (out_dir / "rationale.md").write_text("\n".join(lines))
    return table
