# 🚛 Logistics UseCases

**Open-source, end-to-end machine learning for the logistics and shipping community.**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-green)
![Explainability](https://img.shields.io/badge/every%20model-explained%20%26%20tested-purple)

Each use case in this repo is a complete, self-contained project: dataset story, audited
cleaning, feature engineering, training, evaluation, and explainability that is grounded
by tests rather than eyeballed. The patterns come from production carrier ML, so shipping
teams can adapt working code instead of starting from a blank notebook.

## Use cases

| Use case | The question it answers | Status |
|---|---|---|
| [📦 delivery-commit-prediction](delivery-commit-prediction/) | Which shipments will miss their delivery commitment, and which operational drivers cause it? | ✅ Ready |
| ⏱️ eta-regression | When will each package *actually* arrive, with quantile bounds for customer promises? | 🗺️ Planned |
| 🎛️ intervention-optimization | Given miss-risk scores and a daily ops budget, which shipments get rerouted, upgraded, or flagged to the customer? | 🗺️ Planned |
| 📡 network-anomaly-detection | Which lanes and hubs are drifting toward trouble before it shows up in monthly OTP reports? | 🗺️ Planned |

## 📦 Delivery commit prediction

Ranks shipments by risk of missing the committed delivery date using only information
available at induction time, then explains every score. Runs end-to-end on synthetic
data in about a minute, and includes an adapter for the public Olist dataset (~100k real
orders with real promised-vs-actual delivery dates).

```bash
cd delivery-commit-prediction
pip install -e .
delivery-commit all
```

On the held-out final month: **ROC-AUC 0.81**, the riskiest decile of shipments misses
at **3.6x** the base rate, flagging the top 10% catches **36% of all misses**, and the
predicted probabilities are calibrated (30% means 30%).

The SHAP driver analysis reads like an operations briefing — weather, lane distance,
peak surges, hub congestion and late pickups at the top, while planted noise features
(declared value, package weight) correctly land at the bottom. That ranking is asserted
in CI against the synthetic generator's documented ground truth:

![What drives missed delivery commitments](delivery-commit-prediction/docs/img/shap_summary.png)

| Riskiest 10% carries 3.6x the misses | Probabilities you can quote to a customer |
|---|---|
| ![Lift](delivery-commit-prediction/docs/img/lift_by_decile.png) | ![Calibration](delivery-commit-prediction/docs/img/calibration.png) |

Full write-up, per-shipment explanations, and the adaptation guide:
[delivery-commit-prediction/README.md](delivery-commit-prediction/README.md)

## Repository conventions

Every use case keeps two invariants:

1. **No leakage past decision time.** Features must be knowable at the moment the
   prediction would actually be consumed. For commit prediction that means induction
   time — no scan events or dwell times accumulated during transit.
2. **No unexplained model.** Explainability output is grounded by tests, on synthetic
   data with known drivers wherever possible, so a refactor that breaks explanations
   fails CI.

Each folder carries its own `pyproject.toml`, tests, and README. The shared CI matrix in
[.github/workflows/ci.yml](.github/workflows/ci.yml) runs every use case on Python 3.10
and 3.12.

## Contributing

Suggestions and contributions are welcome. Open an issue describing the operational
question, the data you'd model it on (public or synthetic), and the decision it informs.
New use cases should follow the two invariants above and register themselves in the CI
matrix.

## License

Apache-2.0
