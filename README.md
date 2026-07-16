# Logistics UseCases

Open-source, end-to-end machine learning use cases for the logistics and shipping
community. Each use case is a self-contained project — its own dataset story, cleaning,
feature engineering, training, evaluation, and explainability — built from patterns that
held up in production carrier ML, so teams can adapt them instead of starting from a
blank notebook.

## Use cases

| Use case | Question it answers | Status |
|---|---|---|
| [delivery-commit-prediction](delivery-commit-prediction/) | Which shipments will miss their delivery commitment, and which operational drivers cause it? | Ready |

### delivery-commit-prediction

Ranks shipments by risk of missing the committed delivery date, using only information
available at induction time. Ships with a messy synthetic generator whose ground-truth
drivers are documented (so the SHAP explanations are *tested*, not just plotted), an
adapter for the public Olist e-commerce dataset, calibration and decile-lift evaluation,
and a scoring CLI. Held-out results on the synthetic network: ROC-AUC 0.81, 3.6x lift in
the top risk decile, calibrated probabilities.

```
cd delivery-commit-prediction
pip install -e .
delivery-commit all
```

## Planned use cases

Candidates on the roadmap, in the same end-to-end format:

- **ETA regression** — predict the actual delivery timestamp, not just the miss flag,
  with quantile bounds for customer-facing promises.
- **Intervention optimization** — turn miss-risk scores into reroute / service-upgrade /
  proactive-notification decisions under a daily ops budget.
- **Network anomaly detection** — flag lanes or hubs whose miss rate is drifting before
  it shows up in monthly OTP reports.

Suggestions and contributions are welcome — open an issue describing the operational
question, the data you'd model it on (public or synthetic), and the decision it informs.

## Repository conventions

Every use case keeps two invariants:

1. **No leakage past decision time.** Features must be knowable at the moment the
   prediction would actually be consumed (for commit prediction, when the package enters
   the network).
2. **No unexplained model.** Explainability output is grounded by tests — on synthetic
   data with known drivers where possible — so a refactor that breaks explanations
   fails CI.

Each folder carries its own `pyproject.toml`, tests, and README; the shared CI matrix in
[.github/workflows/ci.yml](.github/workflows/ci.yml) runs every use case on Python 3.10
and 3.12.

## License

Apache-2.0
