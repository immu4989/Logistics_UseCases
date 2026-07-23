"""The operational loop, wired for real: risk scores in, funded decisions out.

This demo makes the repo's central claim literal. delivery-commit-prediction
trains a miss-risk model on its own synthetic history; intervention-optimization
allocates a $6,000 daily budget by expected value. Here the first actually
feeds the second, and the whole loop is graded on the scoring day's REALIZED
missed_commit outcomes — not on the model's opinion of itself.

The pipeline, end to end:

1. Train the commit-risk model on a training draw (seed 7, the standard
   delivery-commit path: messy extract -> audited cleaning -> time split ->
   XGBoost with early stopping). The scoring day never touches training.
2. Generate one operating day: 20,000 shipments, a DIFFERENT seed, messy,
   with ``return_true_risk=True`` so the generator's true miss probability
   rides along for oracle pricing (it never reaches the model — asserted in
   delivery-commit's tests).
3. Score the day -> ``p_hat``.
4. Map shipments into intervention-optimization's expected frame:
   ``declared_value_usd`` passes through; ``customer_tier`` is assigned by
   declared-value terciles (rationale below); miss cost comes from
   ``intervention_opt.interventions.miss_cost``.
5. Run the policy ladder (none / top-K risk / EV-greedy on ``p_hat``) plus
   EV-greedy on the TRUE risk — the perfect-upstream-model oracle.
6. Evaluate every policy on the day's realized outcomes, with intervention
   effects applied counterfactually via common random numbers: one uniform
   per shipment, drawn CONDITIONAL on the realized outcome (a shipment that
   actually missed gets u ~ U(0, p_true); one that didn't gets
   u ~ U(p_true, 1)). Under do-nothing the simulation then reproduces the
   realized day exactly, and ``intervention_opt.evaluate.simulate`` applies
   action effects with the same paired-draws semantics the intervention use
   case documents.

Tier-mapping rationale: the delivery-commit generator has no customer tier,
but its declared value is the natural proxy — in intervention-optimization's
own generator, tier is exactly what shifts the declared-value distribution
(contract freight is pallets, not envelopes). We assign tier by declared-value
rank at the SAME 70/22/8 mix that generator documents: bottom 70% standard,
next 22% premium, top 8% contract. Two reasons this beats an equal-thirds
split. First, comparability: the loop's dollars land on the same customer-mix
economics as intervention-optimization's own README. Second, and decisive: an
equal-thirds split brands a third of parcels "contract" (a x6 miss-cost
multiplier from a ~$48 value boundary), which floods the allocator's
max-EV-then-per-dollar heuristic with upgrade-best shipments it can't fund —
and in that regime MORE accurate risk scores allocate WORSE, so the oracle
stops being an upper bound and the model-quality gap loses its meaning. We
verified that inversion numerically before settling on this mapping. Declared
value carries ~zero risk signal in the upstream generator (a documented noise
feature), so either mapping preserves the risk-independent-of-value structure
that makes EV allocation earn its keep.

Deterministic end to end: fixed seeds, no clocks, no environment reads. Two
runs produce byte-identical tables.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from delivery_commit import cleaning, features, schema, synthetic
from delivery_commit import train as dc_train
from intervention_opt import evaluate as io_evaluate
from intervention_opt import interventions as io_interventions
from intervention_opt import policy as io_policy

HERE = Path(__file__).resolve().parent

TRAIN_SEED = 7          # the standard delivery-commit training draw
DAY_SEED = 31_337       # scoring day: a seed no training run has ever seen
EVAL_SEED = 7           # common-random-numbers stream for outcome draws
BUDGET_USD = 6_000.0
N_TRAIN = 60_000
N_DAY = 20_000

# Policies reported, in ladder order. Oracle = same EV-greedy allocator fed
# the generator's true risk instead of the model score.
POLICY_LABELS = {
    "none": "none",
    "top_k_risk": "top-K risk (p_hat)",
    "expected_value_greedy": "EV-greedy (p_hat)",
    "oracle_true_risk": "EV-greedy (p_true, oracle)",
}


def train_commit_model(n_train: int = N_TRAIN, config: dc_train.TrainConfig | None = None):
    """Standard delivery-commit training path on its own draw. Returns TrainedModels."""
    raw = synthetic.make_dataset(n=n_train, seed=TRAIN_SEED, messy=True)
    clean_df, _ = cleaning.clean(raw)
    models, _ = dc_train.train(clean_df, config)
    return models


def make_scoring_day(n_day: int = N_DAY, seed: int = DAY_SEED) -> pd.DataFrame:
    """One messy scoring batch, cleaned, with the generator's true risk attached.

    "Day" here means the decision problem — one batch of 20,000 shipments,
    one $6,000 budget — not one calendar date. The generator places seasonal
    surges by position inside its date window, so a literal single-date batch
    is all-peak by construction (any short window sits inside both surge
    windows) and lands in a ~24% miss-rate regime neither upstream use case
    is calibrated for. Drawing the batch across the full operating window
    instead gives the day the same 10-12% risk mix the commit model was
    trained on and intervention-optimization's economics are documented for.
    """
    raw = synthetic.make_dataset(
        n=n_day,
        seed=seed,
        messy=True,
        return_true_risk=True,
    )
    day, _ = cleaning.clean(raw)
    return day


def score_day(models, day: pd.DataFrame) -> np.ndarray:
    """Score the day with the trained XGBoost exactly as delivery-commit would."""
    X = features.to_matrix(features.engineer(day))
    X = X.reindex(columns=models.feature_columns, fill_value=0.0)
    return models.xgb.predict_proba(X)[:, 1]


TIER_QUANTILES = (0.70, 0.92)  # intervention-opt's documented 70/22/8 customer mix


def assign_tiers(declared_value: pd.Series) -> np.ndarray:
    """Declared-value rank -> customer tier at the downstream generator's 70/22/8 mix.

    See the module docstring for why this mix and not equal thirds: an
    equal-thirds split makes 33% of parcels "contract" (x6 miss cost above a
    ~$48 value boundary) and provably breaks the allocator's information
    monotonicity — the oracle stops bounding the model from above.
    """
    q1, q2 = declared_value.quantile(list(TIER_QUANTILES))
    return np.where(
        declared_value <= q1, "standard", np.where(declared_value <= q2, "premium", "contract")
    )


def build_expected_frame(day: pd.DataFrame, p_hat: np.ndarray) -> pd.DataFrame:
    """Map the scored day into intervention_opt's expected columns."""
    return pd.DataFrame(
        {
            "shipment_id": day[schema.ID_COL].to_numpy(),
            "customer_tier": assign_tiers(day["declared_value_usd"]),
            "declared_value_usd": day["declared_value_usd"].to_numpy(),
            "p_true": day["p_miss_true"].to_numpy(),
            "p_hat": p_hat,
        }
    )


def conditional_uniforms(day: pd.DataFrame, seed: int = EVAL_SEED) -> np.ndarray:
    """One uniform per shipment, conditioned on the REALIZED outcome.

    intervention_opt.evaluate.simulate realizes a miss under action a iff
    ``u < p_true * (1 - p_reduction)``. Drawing u | miss ~ U(0, p_true) and
    u | no-miss ~ U(p_true, 1) makes the do-nothing simulation reproduce the
    day's actual missed_commit column exactly, while keeping the paired,
    monotone counterfactual semantics for every intervention effect: an
    action with reduction r saves a realized miss with probability r, and
    any miss a weak action prevents, a stronger one also prevents. Same seed
    offset as intervention_opt so the draw stream never collides with
    generator or policy randomness.
    """
    v = np.random.default_rng(seed + 1_000_003).random(len(day))
    p = day["p_miss_true"].to_numpy()
    missed = day[schema.LABEL_COL].to_numpy()
    return np.where(missed == 1, v * p, p + v * (1 - p))


def expected_net_usd(frame: pd.DataFrame, decisions: pd.DataFrame) -> float:
    """Luck-free net savings of a decision frame: p_true-weighted, no draws.

    The realized single-day numbers include Bernoulli outcome luck (which
    specific shipments happened to miss). Differencing p_true-weighted
    expected miss costs removes that luck entirely, which is the honest way
    to quote "what better model accuracy is worth" from one day.
    """
    p_red = decisions["action"].map(
        {a.name: a.p_reduction for a in io_interventions.CATALOG}
    ).to_numpy()
    c_red = decisions["action"].map(
        {a.name: a.cost_reduction for a in io_interventions.CATALOG}
    ).to_numpy()
    cost = io_interventions.miss_cost(frame["customer_tier"], frame["declared_value_usd"])
    p = frame["p_true"].to_numpy()
    avoided = (p * cost).sum() - (p * (1 - p_red) * cost * (1 - c_red)).sum()
    return float(avoided - decisions["spent"].sum())


def run_policies(frame: pd.DataFrame, budget: float, u: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Run the ladder + oracle, evaluate all on the same realized day."""
    decisions = {
        "none": io_policy.policy_none(frame, budget, seed=EVAL_SEED),
        "top_k_risk": io_policy.policy_top_k_risk(frame, budget, seed=EVAL_SEED),
        "expected_value_greedy": io_policy.policy_ev_greedy(frame, budget, seed=EVAL_SEED),
        "oracle_true_risk": io_policy.policy_ev_greedy(
            frame, budget, seed=EVAL_SEED, prob_col="p_true"
        ),
    }
    baseline = io_evaluate.simulate(frame, decisions["none"], u)
    rows = []
    for name, dec in decisions.items():
        sim = io_evaluate.simulate(frame, dec, u)
        avoided = baseline["realized_miss_cost"] - sim["realized_miss_cost"]
        net = avoided - sim["spend"]
        rows.append(
            {
                "policy": POLICY_LABELS[name],
                "spend_usd": round(sim["spend"], 2),
                "misses_prevented": baseline["misses"] - sim["misses"],
                "miss_cost_avoided_usd": round(avoided, 2),
                "net_savings_usd": round(net, 2),
                "roi": round(net / sim["spend"], 3) if sim["spend"] > 0 else None,
            }
        )
    comparison = pd.DataFrame(rows)
    oracle_net = comparison.loc[
        comparison["policy"] == POLICY_LABELS["oracle_true_risk"], "net_savings_usd"
    ].iloc[0]
    comparison["pct_of_oracle"] = (comparison["net_savings_usd"] / oracle_net * 100).round(1)
    return comparison, decisions


def plot_comparison(comparison: pd.DataFrame, path: Path) -> None:
    """The money chart: net savings per policy, oracle drawn as the ceiling.

    House style (matches intervention-optimization's policy chart): recessive
    gray for the comparators, blue for the wired policy, oracle as a dashed
    reference line. Every bar carries a direct dollar label, so no value is
    color-alone.
    """
    idx = comparison.set_index("policy")
    bars = [POLICY_LABELS["none"], POLICY_LABELS["top_k_risk"], POLICY_LABELS["expected_value_greedy"]]
    oracle_net = idx.loc[POLICY_LABELS["oracle_true_risk"], "net_savings_usd"]
    greedy_net = idx.loc[POLICY_LABELS["expected_value_greedy"], "net_savings_usd"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    values = [idx.loc[b, "net_savings_usd"] for b in bars]
    gap = oracle_net - greedy_net
    ax.bar(["none", "top-K risk\n(p_hat)", "EV-greedy\n(p_hat)"], values,
           color=["#8d99ae", "#8d99ae", "#2b6cb0"], width=0.62)
    # NB: escape the dollar signs — a paired "$...$" in a legend label turns
    # on mathtext and typesets the sentence as italic algebra.
    ax.axhline(oracle_net, color="#c0392b", ls="--", lw=1.3,
               label=f"true-risk oracle \\${oracle_net:,.0f} — "
                     f"model leaves \\${gap:,.0f} on the day")
    ax.axhline(0, color="k", lw=0.8)
    for i, v in enumerate(values):
        ax.text(i, v + oracle_net * 0.015, f"${v:,.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(top=oracle_net * 1.18)
    ax.yaxis.set_major_formatter(lambda x, _: f"${x:,.0f}")
    ax.set_ylabel("Net savings (USD, one realized day)")
    ax.set_title("Use case 1's scores driving use case 3's budget, graded on realized misses")
    ax.legend(loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(
    n_day: int = N_DAY,
    n_train: int = N_TRAIN,
    budget: float = BUDGET_USD,
    train_config: dc_train.TrainConfig | None = None,
    day_seed: int = DAY_SEED,
    out_dir: str | Path = HERE / "artifacts",
    chart_path: str | Path | None = HERE / "docs" / "img" / "policy_comparison.png",
) -> dict:
    """Run the full loop. Returns {"comparison", "decisions", "summary"}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] training commit-risk model on its own draw (n={n_train:,}, seed={TRAIN_SEED})")
    models = train_commit_model(n_train, train_config)

    print(f"[2/5] generating the scoring day (n={n_day:,}, seed={day_seed}, unseen by training)")
    day = make_scoring_day(n_day, seed=day_seed)

    print("[3/5] scoring the day and mapping into the intervention frame")
    p_hat = score_day(models, day)
    frame = build_expected_frame(day, p_hat)

    print(f"[4/5] allocating ${budget:,.0f} across {len(frame):,} shipments, "
          "grading on realized outcomes")
    u = conditional_uniforms(day)
    comparison, decisions = run_policies(frame, budget, u)

    idx = comparison.set_index("policy")["net_savings_usd"]
    greedy = idx[POLICY_LABELS["expected_value_greedy"]]
    top_k = idx[POLICY_LABELS["top_k_risk"]]
    oracle = idx[POLICY_LABELS["oracle_true_risk"]]
    # Luck-free version of the model-quality gap: same decisions, p_true-
    # weighted expectations instead of one day's Bernoulli draws.
    exp_gap = expected_net_usd(frame, decisions["oracle_true_risk"]) - expected_net_usd(
        frame, decisions["expected_value_greedy"]
    )
    summary = {
        "n_shipments": int(len(frame)),
        "budget_usd": float(budget),
        "day_seed": int(day_seed),
        "realized_misses": int(day[schema.LABEL_COL].sum()),
        "realized_miss_rate": round(float(day[schema.LABEL_COL].mean()), 4),
        "net_savings_usd": {k: float(idx[v]) for k, v in POLICY_LABELS.items()},
        "loop_vs_top_k_usd": round(float(greedy - top_k), 2),
        "model_quality_gap_realized_usd_per_day": round(float(oracle - greedy), 2),
        "model_quality_gap_expected_usd_per_day": round(exp_gap, 2),
        "model_quality_gap_expected_usd_per_year": round(exp_gap * 365, 2),
        "pct_of_oracle_captured": round(float(greedy / oracle * 100), 1),
    }

    print("[5/5] writing artifacts")
    comparison.to_csv(out_dir / "policy_comparison.csv", index=False)
    (out_dir / "loop_summary.json").write_text(json.dumps(summary, indent=2))
    if chart_path is not None:
        plot_comparison(comparison, Path(chart_path))

    print("\n== Policy comparison (one realized day) " + "=" * 30)
    print(comparison.to_string(index=False))
    print("\n== Value of the loop " + "=" * 49)
    print(f"  EV-greedy on model scores nets      ${greedy:>12,.2f}  "
          f"(vs ${top_k:,.2f} for top-K, $0.00 for none)")
    print(f"  Wiring the loop beats top-K by      ${greedy - top_k:>12,.2f} / day")
    print(f"  True-risk oracle bound              ${oracle:>12,.2f}  "
          f"({summary['pct_of_oracle_captured']}% captured by the model)")
    print(f"  Model-quality gap, this day         ${oracle - greedy:>12,.2f} / day realized")
    print(f"  Model-quality gap, luck-free        ${exp_gap:>12,.2f} / day expected  "
          f"= ${summary['model_quality_gap_expected_usd_per_year']:,.0f} / year")
    return {"comparison": comparison, "decisions": decisions, "summary": summary}


if __name__ == "__main__":
    run()
