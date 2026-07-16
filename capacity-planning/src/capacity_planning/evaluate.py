"""Counterfactual evaluation: every policy costed on the exact same weeks.

Because the generator exposes the true demand distribution of every
(lane, week), outcomes can be simulated EXACTLY rather than estimated:

- Draw one matrix of demand realizations for the test window (200 replications
  per lane-week by default) and cost EVERY policy — and both spot-price
  scenarios — against the SAME matrix (common random numbers). Bookings do not
  change demand, so reusing the draws makes the comparison paired: the
  difference between two policies on a replication is pure policy, never
  between-run luck. Cost each policy on fresh draws instead and the small
  margins that matter here (a 4-percentage-point fractile shift) can flip
  sign run to run for no real reason.
- Averaging over replications prices each policy's *expected* season, which is
  what a booking rule signs you up for; the single realized season the money
  chart shows is one draw from exactly this distribution.

Every metric is measured against the ``book_last_year`` habit (the savings the
project claims) and against the ``oracle`` (the floor no forecast can beat, so
the remaining gap prices what a better forecast is worth).

The peak breakout re-reports the same metrics on ISO weeks 46-52 only — the
weeks that hurt, where the habit re-books last year's ramp a year out of date.

The spot-price sensitivity reruns the tight-market scenario ($3,200 spot) on
the same draws: the habit's bookings cannot move, the newsvendor's fractile
rises from 0.462 to 0.632 and its bookings climb to meet the more expensive
shortfalls. The point is not the specific dollar figure; it is that the METHOD
adapts to a repriced market by re-running one division.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import decide, forecast, synthetic

N_REPS = 200

COLORS = {"committed": "#2b6cb0", "empty": "#8d99ae", "spot": "#e8a33d", "accent": "#c0392b"}


def build_bookings(models: forecast.TrainedModels, splits: dict) -> pd.DataFrame:
    """Assemble the test-period decision frame every policy books from.

    Merges the model quantiles, the habit's number and the ground-truth
    quantiles onto the test rows, then lets decide.py turn them into integer
    bookings.
    """
    test = splits["test"][
        [synthetic.LANE_COL, synthetic.WEEK_COL, synthetic.TARGET_COL, "naive_seasonal"]
    ].reset_index(drop=True)
    q = forecast.predict_quantiles(models, splits["X_test"]).reset_index(drop=True)
    frame = pd.concat([test, q], axis=1)
    for col, alpha in [
        ("true_q_base", decide.critical_fractile()),
        ("true_q_tight", decide.critical_fractile(decide.SPOT_COST_TIGHT_USD)),
    ]:
        frame[col] = synthetic.true_demand_quantile(
            frame[synthetic.LANE_COL], frame[synthetic.WEEK_COL], alpha
        )
    return decide.make_bookings(frame)


def _policy_metrics(comps: dict, n_weeks: int) -> dict:
    return {
        "total_cost_usd": round(float(np.mean(comps["total_cost"])), 0),
        "committed_trailers_per_week": round(comps["booked_trailers"] / n_weeks, 1),
        "spot_teq": round(float(np.mean(comps["spot_teq"])), 1),
        "empty_teq": round(float(np.mean(comps["empty_teq"])), 1),
        "service_level": round(float(np.mean(comps["service_level"])), 4),
        "committed_moved_cost_usd": round(float(np.mean(comps["committed_moved_cost"])), 0),
        "empty_cost_usd": round(float(np.mean(comps["empty_cost"])), 0),
        "spot_cost_usd": round(float(np.mean(comps["spot_cost"])), 0),
    }


def _comparison_table(
    bookings: pd.DataFrame,
    demand: np.ndarray,
    policies: dict[str, str],
    spot_cost: float,
    mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """Cost every policy on the shared demand matrix; derive savings and gaps."""
    idx = np.arange(len(bookings)) if mask is None else np.flatnonzero(mask)
    n_weeks = bookings[synthetic.WEEK_COL].iloc[idx].nunique()
    rows = []
    for name, col in policies.items():
        comps = decide.cost_components(
            bookings[col].to_numpy()[idx], demand[idx], spot_cost=spot_cost
        )
        rows.append({"policy": name, **_policy_metrics(comps, n_weeks)})
    table = pd.DataFrame(rows)
    habit = table.loc[table["policy"] == "book_last_year", "total_cost_usd"].iloc[0]
    oracle_row = table["policy"].str.startswith("oracle")
    oracle = table.loc[oracle_row, "total_cost_usd"].iloc[0]
    table["savings_vs_habit_usd"] = (habit - table["total_cost_usd"]).round(0)
    table["savings_vs_habit_pct"] = ((habit - table["total_cost_usd"]) / habit * 100).round(2)
    table["excess_vs_oracle_pct"] = ((table["total_cost_usd"] / oracle - 1) * 100).round(2)
    return table


def evaluate_all(
    bookings: pd.DataFrame,
    seed: int = 7,
    out_dir: str | Path = "artifacts/reports",
    n_reps: int = N_REPS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cost every policy under both spot scenarios; write metrics + plots.

    Returns (base-scenario comparison, tight-scenario sensitivity table).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ONE demand matrix, shared by every policy and both scenarios (CRN).
    demand = synthetic.simulate_demand(bookings, seed, n_reps)

    comparison = _comparison_table(
        bookings, demand, decide.POLICY_COLUMNS, decide.SPOT_COST_USD
    )

    woy = pd.DatetimeIndex(bookings[synthetic.WEEK_COL]).isocalendar().week.to_numpy()
    peak = _comparison_table(
        bookings, demand, decide.POLICY_COLUMNS, decide.SPOT_COST_USD,
        mask=np.isin(woy, synthetic.PEAK_WOY),
    )

    # Tight market: the habit and the stale newsvendor keep their bookings;
    # the retuned newsvendor books at the higher fractile the new price implies.
    tight_policies = {
        "book_last_year": "booked_last_year",
        "newsvendor_model_stale": "booked_newsvendor",
        "newsvendor_model_retuned": "booked_newsvendor_tight",
        "oracle_tight": "booked_oracle_tight",
    }
    sensitivity = _comparison_table(
        bookings, demand, tight_policies, decide.SPOT_COST_TIGHT_USD
    )

    # ---- persist ------------------------------------------------------------
    comparison.to_csv(out_dir / "policy_comparison.csv", index=False)
    num_cols = bookings.select_dtypes("number").columns
    bookings.assign(**{c: bookings[c].round(2) for c in num_cols}).to_csv(
        out_dir / "bookings.csv", index=False
    )
    metrics = {
        "n_lane_weeks": int(len(bookings)),
        "n_reps": int(n_reps),
        "seed": int(seed),
        "committed_cost_usd": decide.COMMITTED_COST_USD,
        "spot_cost_usd": decide.SPOT_COST_USD,
        "salvage_usd": decide.SALVAGE_USD,
        "critical_fractile_base": round(decide.critical_fractile(), 4),
        "spot_cost_tight_usd": decide.SPOT_COST_TIGHT_USD,
        "critical_fractile_tight": round(
            decide.critical_fractile(decide.SPOT_COST_TIGHT_USD), 4
        ),
        "policies": comparison.to_dict(orient="records"),
        "peak_weeks": peak.to_dict(orient="records"),
        "sensitivity_tight_spot": sensitivity.to_dict(orient="records"),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    _plot_policy_costs(comparison, out_dir / "policy_costs.png")
    _plot_lane_chart(bookings, out_dir / "lane_money_chart.png")
    _plot_booking_vs_spot(bookings, out_dir / "booking_vs_spot.png")
    return comparison, sensitivity


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_policy_costs(comparison: pd.DataFrame, path: Path) -> None:
    """Stacked cost bars per policy; the oracle drawn as the floor, not a bar."""
    plot_df = comparison[comparison["policy"] != "oracle"]
    oracle = comparison.loc[comparison["policy"] == "oracle", "total_cost_usd"].iloc[0]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = np.arange(len(plot_df))
    bottom = np.zeros(len(plot_df))
    for key, label, color in [
        ("committed_moved_cost_usd", "committed capacity that moved freight", COLORS["committed"]),
        ("empty_cost_usd", "empty trailers (net of salvage)", COLORS["empty"]),
        ("spot_cost_usd", "spot buys", COLORS["spot"]),
    ]:
        vals = plot_df[key].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, color=color, label=label)
        bottom += vals
    for i, total in enumerate(plot_df["total_cost_usd"]):
        ax.text(i, bottom[i], f"${total / 1e6:.2f}M", ha="center", va="bottom", fontsize=9)
    ax.axhline(oracle, color=COLORS["accent"], ls="--", lw=1.2,
               label=f"oracle (true distribution): ${oracle / 1e6:.2f}M")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["policy"])
    ax.set_ylabel("Total cost, 16 test weeks (USD)")
    ax.set_title("Same weeks, same demand draws: only the booking rule differs")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_lane_chart(bookings: pd.DataFrame, path: Path, lane: str = "LAX-PHX") -> None:
    """One growing lane through the peak: the habit books last year, the model
    rides the ramp."""
    df = bookings[bookings[synthetic.LANE_COL] == lane].sort_values(synthetic.WEEK_COL)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(df[synthetic.WEEK_COL], df[synthetic.TARGET_COL], marker="o", ms=4,
            color="#444444", lw=1.4, label="realized demand (trailer-equivalents)")
    ax.step(df[synthetic.WEEK_COL], df["booked_last_year"], where="mid",
            color=COLORS["empty"], lw=2.0, label="book_last_year (the habit)")
    ax.step(df[synthetic.WEEK_COL], df["booked_newsvendor"], where="mid",
            color=COLORS["committed"], lw=2.0, label="newsvendor_model (q* forecast)")
    woy = pd.DatetimeIndex(df[synthetic.WEEK_COL]).isocalendar().week.to_numpy()
    in_peak = np.isin(woy, synthetic.PEAK_WOY)
    if in_peak.any():
        weeks = df[synthetic.WEEK_COL].to_numpy()
        ax.axvspan(weeks[in_peak].min(), weeks[in_peak].max(), color=COLORS["spot"],
                   alpha=0.12, label="peak weeks (ISO 46-52)")
    ax.set_ylabel("Trailers per week")
    ax.set_title(f"Lane {lane} (growing lane), held-out test weeks")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_booking_vs_spot(bookings: pd.DataFrame, path: Path) -> None:
    """Optimal network booking as the spot price moves: the fractile is a dial."""
    level = synthetic.demand_level(
        pd.DatetimeIndex(sorted(bookings[synthetic.WEEK_COL].unique()))
    )
    keyed = level.set_index([synthetic.LANE_COL, synthetic.WEEK_COL])["level"]
    lv = keyed.loc[
        list(zip(bookings[synthetic.LANE_COL], pd.to_datetime(bookings[synthetic.WEEK_COL])))
    ].to_numpy()
    lanes = bookings[synthetic.LANE_COL].to_numpy()

    spots = np.arange(1_600, 4_001, 100)
    totals = []
    for s in spots:
        q = decide.critical_fractile(float(s))
        mult_q = {ln: np.quantile(synthetic._multiplier_sample(synthetic.lane_sigma(ln)), q)
                  for ln in synthetic.LANES}
        booked = np.ceil(lv * np.array([mult_q[ln] for ln in lanes]))
        totals.append(booked.sum() / bookings[synthetic.WEEK_COL].nunique())

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(spots, totals, color=COLORS["committed"], lw=2.0)
    for s, label in [(decide.SPOT_COST_USD, "base"), (decide.SPOT_COST_TIGHT_USD, "tight")]:
        q = decide.critical_fractile(s)
        ax.axvline(s, color=COLORS["accent"], ls="--", lw=1.2)
        ax.annotate(f"{label}: spot ${s:,.0f}\nq* = {q:.3f}", xy=(s, np.interp(s, spots, totals)),
                    xytext=(8, -34), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Spot price per trailer-equivalent (USD)")
    ax.set_ylabel("Optimal committed trailers per week (network)")
    ax.set_title("The booking is a cost ratio: reprice the spot market, the fractile moves")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
