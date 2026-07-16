"""SHAP driver analysis, per queue: from "route it there" to "here is why".

Outputs:

- ``per_queue_drivers.png``  — for each resolution queue, the features that
  push tickets INTO that queue (mean positive SHAP toward the class)
- ``per_queue_drivers.csv``  — the same ranking as data, one-hot columns
  re-aggregated to their operational lever
- ``ticket_card_autoroute.md``  — a confidently auto-routed ticket: top class
  probabilities, SHAP contributions, and the plain-language line an ops
  screen would show
- ``ticket_card_escalation.md`` — a low-confidence ticket the gate sends to a
  human, same format

On the synthetic dataset the per-queue rankings should recover the
generator's documented mapping (``synthetic.TRUE_RULES``): the damage flag
tops damage_claims, the weather flag tops hold_and_monitor, customs presence
tops customs_docs, and the planted noise (csr_id, ticket-creation hour) stays
buried. Those checks live in the test suite, so a refactor that silently
breaks explanations fails CI.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from . import features, schema

# Plain-language fragments for the ticket cards, keyed by matrix column.
_PHRASES = {
    "damage_scan_flag": "a damage scan on record",
    "weather_event_at_location": "an active weather event at the last scan location",
    "address_validation_failed": "a failed address validation",
    "return_to_sender_flag": "a label already marked return-to-sender",
    "is_international": "an international shipment",
    "last_scan_location_type_customs": "last seen at customs",
    "last_scan_location_type_linehaul": "last seen on a linehaul leg",
    "last_scan_location_type_hub": "last seen at a hub",
    "last_scan_location_type_last_mile": "last seen out for delivery",
    "scan_gap_hours": "{v:.0f} hour{s} since the last scan",
    "delivery_attempt_count": "{v:.0f} delivery attempt{s} so far",
    "prior_exceptions": "{v:.0f} prior exception{s} on this shipment",
    "declared_value_usd": "a declared value of ${v:,.0f}",
}


def _group_feature(col: str) -> str:
    """Map a model-matrix column back to its operational lever."""
    for prefix in features.ONE_HOT_COLS:
        if col.startswith(prefix + "_"):
            return prefix
    if col.endswith("__was_missing"):
        return col.replace("__was_missing", "") + " (missingness)"
    return col


def _shap_per_class(models, X_sample: pd.DataFrame) -> np.ndarray:
    """SHAP values as an (n_rows, n_features, n_classes) array."""
    explainer = shap.TreeExplainer(models.xgb)
    values = explainer.shap_values(X_sample)
    if isinstance(values, list):  # older shap: one (n, f) array per class
        values = np.stack(values, axis=-1)
    return values


def explain(models, splits, out_dir: str | Path, max_background: int = 4000) -> pd.DataFrame:
    """Compute per-queue SHAP rankings on the held-out period; write reports.

    Returns a tidy DataFrame: (queue, driver, mean_push_shap, rank) with rank 1
    being that queue's strongest driver.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_test = splits["X_test"]
    if len(X_test) > max_background:
        X_sample = X_test.sample(max_background, random_state=0)
    else:
        X_sample = X_test

    shap_values = _shap_per_class(models, X_sample)  # (n, f, c)
    pred_idx = models.xgb.predict_proba(X_sample).argmax(axis=1)

    rows = []
    for j, queue in enumerate(models.classes):
        # For the tickets the model routes INTO this queue, which features
        # pushed them there? Two deliberate choices versus a plain global
        # |SHAP| ranking: condition on the queue's own tickets (the question
        # the team owning the queue actually asks), and keep only positive
        # contributions (scan-gap hours argue *against* hold_and_monitor far
        # more often than for it, and |SHAP| would reward that).
        in_queue = pred_idx == j
        sv = shap_values[in_queue, :, j] if in_queue.any() else shap_values[:, :, j]
        push = pd.Series(np.clip(sv, 0, None).mean(axis=0), index=X_sample.columns)
        grouped = push.groupby(push.index.map(_group_feature)).sum()
        grouped = grouped.sort_values(ascending=False)
        for rank, (driver, value) in enumerate(grouped.items(), start=1):
            rows.append(
                {"queue": queue, "driver": driver, "mean_push_shap": float(value), "rank": rank}
            )
    ranking = pd.DataFrame(rows)
    ranking.to_csv(out_dir / "per_queue_drivers.csv", index=False)

    _plot_per_queue_drivers(ranking, out_dir / "per_queue_drivers.png")
    _write_ticket_cards(models, splits, X_sample, shap_values, out_dir)
    return ranking


def _plot_per_queue_drivers(ranking: pd.DataFrame, path: Path) -> None:
    """One panel per queue: the top features pushing tickets into it."""
    queues = list(ranking["queue"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=False)
    for ax, queue in zip(axes.ravel(), queues):
        top = ranking[ranking["queue"] == queue].nsmallest(6, "rank").iloc[::-1]
        ax.barh(top["driver"], top["mean_push_shap"], color="#2b6cb0")
        ax.set_title(queue, fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=7)
    fig.suptitle("What routes a ticket into each queue (mean positive SHAP toward the class)", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _phrase(col: str, value: float) -> str:
    template = _PHRASES.get(col)
    if template is None:
        return f"{col} = {value:.3g}"
    return template.format(v=value, s="" if round(value) == 1 else "s")


def _write_card(
    title: str,
    verdict: str,
    ticket_id: str,
    probs: pd.Series,
    contribs: pd.Series,
    row: pd.Series,
    path: Path,
) -> None:
    top_probs = probs.sort_values(ascending=False).head(3)
    top_contribs = contribs.reindex(contribs.abs().sort_values(ascending=False).head(5).index)

    lines = [
        f"# {title}",
        "",
        f"Ticket `{ticket_id}`",
        "",
        "| Queue | Probability |",
        "|---|---:|",
    ]
    lines += [f"| {q} | {p:.1%} |" for q, p in top_probs.items()]
    lines += [
        "",
        f"**Routing decision: {verdict}**",
        "",
        f"Top drivers toward `{top_probs.index[0]}`:",
        "",
        "| Driver | Value | SHAP (log-odds) |",
        "|---|---:|---:|",
    ]
    for col, val in top_contribs.items():
        lines.append(f"| {col} | {row[col]:.3g} | {val:+.2f} |")

    # Only verbalize positive drivers that are actually present on the ticket
    # ("an active weather event" when the flag is 0 would be nonsense; the
    # SHAP value is still real, the absence just isn't a story to tell).
    positive = [c for c in top_contribs.index if top_contribs[c] > 0 and row[c] >= 0.5][:2]
    reasons = (
        " and ".join(_phrase(c, row[c]) for c in positive)
        if positive
        else "no single strong driver; mostly the absence of counter-signals"
    )
    lines += ["", f"_In plain language: {reasons} point this ticket at {top_probs.index[0]}._"]
    path.write_text("\n".join(lines))


def _write_ticket_cards(models, splits, X_sample, shap_values, out_dir: Path) -> None:
    """One confidently auto-routed ticket, one human escalation."""
    test_df = splits["test"]
    probs = models.xgb.predict_proba(X_sample)
    max_prob = probs.max(axis=1)
    pred_idx = probs.argmax(axis=1)
    y_true = test_df.loc[X_sample.index, schema.LABEL_COL].to_numpy()
    correct = np.array(models.classes)[pred_idx] == y_true

    # Auto-route card: the gate's best case — maximum confidence, and correct.
    auto_i = int(np.argmax(np.where(correct, max_prob, -1.0)))
    # Escalation card: the gate's reason to exist — the least confident ticket.
    esc_i = int(np.argmin(max_prob))

    for i, title, verdict, fname in [
        (
            auto_i,
            "Ticket card: auto-routed",
            "auto-route (confidence clears the gate)",
            "ticket_card_autoroute.md",
        ),
        (
            esc_i,
            "Ticket card: escalated to a human",
            "escalate to the human triage queue (confidence below the gate)",
            "ticket_card_escalation.md",
        ),
    ]:
        row = X_sample.iloc[i]
        ticket_id = str(test_df.loc[X_sample.index[i], schema.ID_COL])
        prob_series = pd.Series(probs[i], index=models.classes)
        contribs = pd.Series(shap_values[i, :, pred_idx[i]], index=X_sample.columns)
        _write_card(title, verdict, ticket_id, prob_series, contribs, row, out_dir / fname)
