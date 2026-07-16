# 🚛 Logistics UseCases

**Open-source, end-to-end machine learning for the logistics and shipping community.**

[![CI](https://github.com/immu4989/Logistics_UseCases/actions/workflows/ci.yml/badge.svg)](https://github.com/immu4989/Logistics_UseCases/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-green)
![Use cases](https://img.shields.io/badge/use%20cases-4%20ready-brightgreen)
![Explainability](https://img.shields.io/badge/every%20model-explained%20%26%20tested-purple)

Each use case in this repo is a complete, self-contained project: dataset story, audited
cleaning, feature engineering, modeling, evaluation, and explainability that is grounded
by tests rather than eyeballed. The patterns come from production carrier ML, so shipping
teams can adapt working code instead of starting from a blank notebook.

Together the four projects cover the full operational loop: **predict** which shipments
are at risk, **quote** delivery promises you can keep, **act** on the riskiest shipments
within a real budget, and **watch** the network for drift before the monthly report sees
it.

## Use cases

| Use case | The question it answers | Status |
|---|---|---|
| [📦 delivery-commit-prediction](delivery-commit-prediction/) | Which shipments will miss their delivery commitment, and which operational drivers cause it? | ✅ Ready |
| [⏱️ eta-regression](eta-regression/) | When will each package *actually* arrive, and what transit time can you promise and keep 9 times out of 10? | ✅ Ready |
| [🎛️ intervention-optimization](intervention-optimization/) | Given miss-risk scores and a daily ops budget, which shipments get rerouted, upgraded, or flagged to the customer? | ✅ Ready |
| [📡 network-anomaly-detection](network-anomaly-detection/) | Which lanes are drifting toward trouble weeks before the monthly OTP report notices? | ✅ Ready |

Every use case runs end-to-end on synthetic data with one command, in about a minute,
with no proprietary data and no downloads:

```bash
cd <use-case-folder>
pip install -e .
<use-case-cli> all      # e.g. delivery-commit all, eta-regression all
```

## 📦 Delivery commit prediction

Ranks shipments by risk of missing the committed delivery date using only information
available at induction time, then explains every score. Includes an adapter for the
public Olist dataset (~100k real orders with real promised-vs-actual delivery dates).
On the held-out final month: **ROC-AUC 0.81**, the riskiest decile misses at **3.6x**
the base rate, flagging the top 10% catches **36% of all misses**, and the probabilities
are calibrated (30% means 30%).

The SHAP driver analysis reads like an operations briefing, and it's asserted in CI
against the synthetic generator's documented ground truth — planted noise features land
at the bottom, real drivers at the top:

![What drives missed delivery commitments](delivery-commit-prediction/docs/img/shap_summary.png)

| Riskiest 10% carries 3.6x the misses | Probabilities you can quote to a customer |
|---|---|
| ![Lift](delivery-commit-prediction/docs/img/lift_by_decile.png) | ![Calibration](delivery-commit-prediction/docs/img/calibration.png) |

Full write-up: [delivery-commit-prediction/README.md](delivery-commit-prediction/README.md)

## ⏱️ ETA regression

A tracking page wants one number; a customer promise needs three. XGBoost quantile
models predict each shipment's transit time as **P10 / P50 / P90** at induction time,
with noise that honestly widens on long congested lanes. Median ETA lands within
**0.44 days** of the truth; the P10–P90 interval covers **78.6%** against an 80% target,
holding across short, medium and long lanes.

The product punchline is the promise table: quote ceil(P50) and the promise breaks one
time in five; quote ceil(P90) and it holds for **97% of shipments** at a cost of just
**0.69 extra quoted days**:

| Median ETA vs actual | The promise curve |
|---|---|
| ![Predicted vs actual](eta-regression/docs/img/pred_vs_actual.png) | ![Promise curve](eta-regression/docs/img/promise_curve.png) |

Full write-up: [eta-regression/README.md](eta-regression/README.md)

## 🎛️ Intervention optimization

A risk score is not a decision. This use case turns miss probabilities into a
budget-constrained action plan over a priced catalog (notify the customer, reroute,
upgrade the service), allocating by expected savings per dollar and evaluating against
counterfactual ground truth with paired random draws.

At a $6,000 daily budget on 20,000 shipments, expected-value greedy nets **$32,577/day
at 5.4x ROI** and captures **91.5% of the oracle bound**. The classic ops rule — upgrade
the highest-risk shipments until the budget runs out — nets $2,435 (0.4x ROI), because
a 90% risk on a $5 shipment is worth less than a 40% risk on a contract pallet. The
oracle gap (~$3,000/day) literally prices what a better upstream model is worth.

| Policy shoot-out | Diminishing returns by budget |
|---|---|
| ![Policy comparison](intervention-optimization/docs/img/policy_comparison.png) | ![Savings vs budget](intervention-optimization/docs/img/savings_vs_budget.png) |

Full write-up: [intervention-optimization/README.md](intervention-optimization/README.md)

## 📡 Network anomaly detection

Your monthly OTP report is a rear-view mirror. This use case watches ~120 lanes daily
with empirical-Bayes shrinkage (so 10-shipment/day lanes don't cry wolf), removes
network-wide effects (a stormy Tuesday is not a lane anomaly), and runs a CUSUM that
accumulates small persistent shifts until they're undeniable.

Against injected ground-truth drifts: **100% of step drifts caught in 3.1 days on
average versus 21.9 days** for the monthly report, ramps caught mid-climb, and
**0.47 false alarms per clean-lane-year**. Every alarm ships as a plain-language
incident card with the estimated cost in extra misses per week.

![Step drift caught in days](network-anomaly-detection/docs/img/example_step_lane.png)

Full write-up: [network-anomaly-detection/README.md](network-anomaly-detection/README.md)

## Repository conventions

Every use case keeps two invariants:

1. **No leakage past decision time.** Features must be knowable at the moment the
   prediction would actually be consumed. For commit prediction and ETAs that means
   induction time — no scan events or dwell times accumulated during transit.
2. **No unexplained model.** Explainability output is grounded by tests, on synthetic
   data with known drivers wherever possible, so a refactor that breaks explanations
   fails CI. Where SHAP isn't the right tool (decision optimization, monitoring), the
   explanation is a per-decision rationale or incident card an ops manager can sign off.

Each folder carries its own `pyproject.toml`, tests, and README. The shared CI matrix in
[.github/workflows/ci.yml](.github/workflows/ci.yml) runs every use case on Python 3.10
and 3.12.

## Contributing

Suggestions and contributions are welcome. Open an issue describing the operational
question, the data you'd model it on (public or synthetic), and the decision it informs.
New use cases should follow the two invariants above and register themselves in the CI
matrix. Ideas on the roadmap: demand/volume forecasting for hub staffing, dynamic
promise-date quoting at checkout, and claims-cost modeling.

## License

Apache-2.0
