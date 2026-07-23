# Model card: delivery-commit miss prediction

## What it does
Scores parcels at induction time with the probability of missing the committed
delivery date, so operations can intervene (reroute, upgrade, notify) before the
promise breaks. Outputs a calibrated probability; consumers rank by it and act on
the top slice their daily capacity allows.

## Training data
Synthetic shipment extracts from `delivery_commit.synthetic` (documented generative
process, no real customer data), or your own extract mapped to `schema.py`. The
reference run trains on ~48k shipments over ~5 months and holds out the final month.

## Evaluation (reference run, held-out final month)
PR-AUC 0.46, ROC-AUC 0.81, top-decile lift 3.6x, recall 36% at a 10% flag budget,
calibrated within a few points across the probability range (see
`artifacts/reports/`). On the real Olist dataset the logistic baseline holds
ROC-AUC 0.75 while the tree model collapses under a regime shift; read the README's
Olist section before trusting any model of this family across a policy change.

## Intended use
Ranking shipments for proactive intervention within a capacity budget. Probabilities
are calibrated on the training distribution; treat absolute values with suspicion
after network changes (peak onset, service redesign, carrier strikes).

## Not intended for
Customer-facing promises (use eta-regression's quantiles), performance evaluation of
drivers or facilities (features are correlational, not causal attributions), or
unmonitored autonomous action.

## Retraining and monitoring
Retrain at least weekly in production; the features are cheap and the regime risk is
real. Monitor input drift (the cleaning report's touch-counts) and outcome drift
(the network-anomaly-detection use case exists for exactly this). Alert if realized
miss rate in any predicted-probability bucket departs from the bucket's nominal rate.

## Explainability
Global and per-shipment SHAP reports ship with every training run and are asserted
against the synthetic generator's ground truth in CI. If the SHAP ranking shifts
materially between retrains, treat it as a data-pipeline incident, not a curiosity.
