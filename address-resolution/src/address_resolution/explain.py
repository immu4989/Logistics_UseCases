"""Per-decision rationale cards from the logistic coefficients.

This is why the scorer is a logistic regression and not a GBM: every match
probability decomposes exactly into one contribution per feature
(coefficient x standardized value), and that decomposition IS the screen a
review-queue operator sees next to the label. No sampling, no approximation,
no explanation drift after a refactor — the rationale is the model.

Three cards are written, one per decision archetype:

1. A confident auto-match that pushed through two or more corruptions —
   the case that earns the system its coverage.
2. A correct reject of an orphan label — the case that earns it trust.
3. A near-miss where a unit conflict blocked the match — the case a reviewer
   resolves in seconds *because* the card says which signal killed it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .train import MatchRun

_DISPLAY = {
    "name_trigram_jaccard": "Street-name character overlap",
    "token_set_overlap": "Typed tokens found in the record",
    "number_exact": "Street number matches exactly",
    "number_transposed": "Street number matches up to transposed digits",
    "unit_agree": "Unit agrees",
    "unit_conflict": "Unit conflicts (both present, different)",
    "unit_unresolvable": "Unit missing on label, building is multi-unit",
    "street_type_match": "Street type agrees",
    "zip_distance": "Zip distance (grid blocks)",
}


def contributions(resolver, x: np.ndarray) -> pd.DataFrame:
    """Exact per-feature log-odds contributions for one pair."""
    scaler = resolver.pipeline.named_steps["scale"]
    logreg = resolver.pipeline.named_steps["logreg"]
    z = (x - scaler.mean_) / scaler.scale_
    contrib = logreg.coef_[0] * z
    df = pd.DataFrame(
        {
            "feature": resolver.feature_names,
            "value": x,
            "contribution": contrib,
        }
    )
    return df.reindex(df["contribution"].abs().sort_values(ascending=False).index)


def _address_of(points: pd.DataFrame, idx: int) -> str:
    r = points.iloc[idx]
    unit = f" Apt {r['unit']}" if r["unit"] else ""
    return f"{r['street_number']} {r['street_name']} {r['street_type']}{unit}, {r['zip']}"


def _card(title: str, run: MatchRun, dec_row: pd.Series, verdict: str) -> list[str]:
    resolver = run.resolver
    pair = int(dec_row["best_pair"])
    x = run.X[pair]
    cand_idx = int(run.pi[pair])
    label_row = run.labels.loc[run.labels["label_id"] == dec_row["label_id"]].iloc[0]

    lines = [
        f"## {title}",
        "",
        f'Label: `"{label_row["address_text"]}"`  (zip {label_row["zip"]})',
        f"Recorded corruptions: `{label_row['corruptions'] or 'none'}`",
        f"Best candidate: {_address_of(run.points, cand_idx)}  "
        f"(`{run.points.iloc[cand_idx]['point_id']}`)",
        f"Match probability: **{dec_row['p_best']:.3f}** vs threshold "
        f"{resolver.threshold:.3f} -> **{verdict}**",
        "",
        "| Signal | Value | Pull on match log-odds |",
        "|---|---:|---:|",
    ]
    for _, row in contributions(resolver, x).iterrows():
        lines.append(
            f"| {_DISPLAY[row['feature']]} | {row['value']:.2f} | {row['contribution']:+.2f} |"
        )
    lines.append("")
    return lines


def write_rationale(run: MatchRun, out_dir: str | Path) -> Path:
    """Pick the three archetype decisions from the held-out labels and write
    their cards to rationale.md."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dec = run.decisions[~run.decisions["is_train"]]
    unit_conflict_col = run.resolver.feature_names.index("unit_conflict")
    number_exact_col = run.resolver.feature_names.index("number_exact")

    n_corr = dec["corruptions"].astype(str).apply(
        lambda c: 0 if c == "" or c.startswith("orphan_") else c.count("|") + 1
    )
    lines = [
        "# Why each decision was made",
        "",
        "Exact per-feature contributions from the logistic scorer (coefficient x",
        "standardized feature). Positive pulls toward a match, negative away.",
        "The threshold below each card is the trained operating point.",
        "",
    ]

    # 1. Confident auto-match through >= 2 corruptions.
    c1 = dec[dec["auto_correct"] & (n_corr >= 2)].sort_values("p_best", ascending=False)
    if not c1.empty:
        lines += _card(
            "Card 1 — auto-match that survived two corruptions",
            run, c1.iloc[0], "auto-match (correct)",
        )

    # 2. Correct reject-to-review of an orphan.
    c2 = dec[
        dec["is_orphan"] & ~dec["accepted"] & (dec["best_pair"] >= 0)
        & (dec["corruptions"] == "orphan_novel_street")
    ].sort_values("p_best", ascending=False)
    if c2.empty:
        c2 = dec[dec["is_orphan"] & ~dec["accepted"] & (dec["best_pair"] >= 0)]
    if not c2.empty:
        lines += _card(
            "Card 2 — orphan correctly routed to review "
            "(no delivery point exists for this label)",
            run, c2.iloc[0], "review queue (correct: true no-match)",
        )

    # 3. Near-miss blocked by a unit conflict.
    has_pair = dec["best_pair"] >= 0
    bp = dec.loc[has_pair, "best_pair"].to_numpy()
    conflict = run.X[bp, unit_conflict_col] > 0.5
    numeq = run.X[bp, number_exact_col] > 0.5
    c3 = dec.loc[has_pair].loc[conflict & numeq & ~dec.loc[has_pair, "accepted"]]
    if not c3.empty:
        lines += _card(
            "Card 3 — everything agrees except the unit, and that is enough to stop it",
            run, c3.sort_values("p_best", ascending=False).iloc[0],
            "review queue (unit conflict blocked the match)",
        )

    path = out_dir / "rationale.md"
    path.write_text("\n".join(lines))
    return path
