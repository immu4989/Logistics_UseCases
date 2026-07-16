# Delivery commit prediction

Predict which parcels will miss their delivery commitment, and explain why.

An end-to-end, open reference pipeline for on-time-performance (OTP) modeling in parcel
logistics: raw shipment extract → cleaning → feature engineering → training → evaluation →
SHAP driver analysis. Built from patterns that held up in production carrier ML, released
here so any shipping company can adapt them without starting from a blank notebook.

```
pip install -e .
delivery-commit all
```

That one command generates 60k synthetic shipments (deliberately messy, like a real
extract), cleans them, trains a logistic baseline and an XGBoost model on a time-based
split, evaluates on the held-out period, and writes a SHAP report of what drives misses.
No proprietary data, no downloads, ~1 minute on a laptop.

## Why this exists

Every parcel network fights the same fire: a small fraction of shipments miss their
committed delivery date, and by the time you know which ones, it is too late to act.
A model that ranks tomorrow's shipments by miss risk turns that into an operations lever:
reroute, upgrade the service, or notify the customer before the promise breaks.

The modeling itself is not exotic. What is hard to find in the open is a complete,
honest treatment of the parts that decide whether the model survives contact with
operations: label leakage, time-based validation, calibration, and explanations an
ops manager will accept. That is what this repo tries to demonstrate.

## Results (synthetic dataset, held-out final month)

| Model | PR-AUC | ROC-AUC | Top-decile lift | Recall @ 10% flagged |
|---|---|---|---|---|
| Logistic baseline | 0.466 | 0.814 | 3.6x | 36.2% |
| XGBoost | 0.456 | 0.811 | 3.6x | 35.7% |

Base miss rate in the test period is ~14%. The riskiest decile of shipments carries
3.6x the base miss rate, and flagging the top 10% catches over a third of all misses,
which is the number that decides whether a proactive-intervention program pays for itself.

![Lift by risk decile](docs/img/lift_by_decile.png)

Predicted probabilities are calibrated out of the box, because we deliberately do not
reweight classes (see design notes below). When the model says 30%, it misses about
30% of the time:

![Calibration](docs/img/calibration.png)

## The SHAP report, checked against ground truth

The synthetic generator has a documented causal process
([synthetic.py](src/delivery_commit/synthetic.py)): hub congestion, destination weather,
late pickup, peak surges and lane distance drive misses, while declared value, package
weight and signature flags are planted noise. The point of generating data this way is
that the explainability layer can be *tested*, not just admired. The test suite asserts
that SHAP's driver ranking recovers the real drivers and does not promote the noise
features, so a refactor that silently breaks explanations fails CI.

![SHAP summary](docs/img/shap_summary.png)

The pipeline also writes a per-shipment explanation
(`artifacts/reports/example_shipment.md`) in the shape an ops team would see next to a
flagged package, plus `driver_ranking.csv` with one-hot columns re-aggregated back to
operational levers.

## Running on real data (Olist)

The repo includes an adapter for the public
[Olist Brazilian e-commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
(~100k real orders with both a promised and an actual delivery date, so the miss label
is real):

```
kaggle datasets download -d olistbr/brazilian-ecommerce -p data/olist --unzip
delivery-commit all --source olist --olist-dir data/olist
```

[olist.py](src/delivery_commit/olist.py) is intentionally the template to copy when you write
the adapter for your own company's data: it maps what Olist has onto the canonical
schema, fills what it lacks (hub telemetry, weather) with neutral constants, and the
rest of the pipeline runs unchanged.

## Adapting to your own shipment data

1. Produce a DataFrame with the canonical columns in
   [schema.py](src/delivery_commit/schema.py). Only three are truly required to be meaningful:
   a shipment id, a ship date, and the `missed_commit` label. Every feature column you
   can populate improves the model; anything you can't, fill with a neutral constant.
2. **Respect induction time.** Every feature must be knowable when the package enters
   your network. Scan counts, dwell times and exception codes accumulated *during*
   transit predict lateness brilliantly and uselessly; they are how OTP models leak.
3. Run `delivery-commit all` and read `artifacts/reports/`. Then score new shipments with
   `delivery-commit score --input tomorrow.csv`.

## Design notes

**Time-based split, never random.** Shipments from the same lane and day are heavily
correlated. A random split leaks future operating conditions into training and inflates
offline metrics that then evaporate in deployment. Training uses the first ~80% of ship
dates; everything after the cutoff is held out.

**No class reweighting at ~10% positives.** `scale_pos_weight` and friends buy nothing
for ranking metrics at this imbalance, and they wreck calibration: early versions of
this pipeline told ops "80% risk" for shipments that missed 30% of the time. If your
miss rate is far rarer (<1–2%), reweight for trainability and recalibrate on a held-out
slice before anyone consumes the probabilities.

**The baseline is load-bearing.** On this dataset the logistic baseline ties XGBoost,
because the synthetic generative process is nearly additive once the features are
engineered well. That is the honest general lesson: gradient boosting earns its keep on
the messy interactions of real operational data, and if it cannot beat your linear
baseline, ship the linear model. The baseline exists to force that conversation.

**Missingness is signal.** Cleaning imputes medians but keeps `__was_missing` flags.
A hub too overwhelmed to report congestion is probably congested; on the synthetic data
the missingness flag for weather shows up in the SHAP report exactly as designed.

**Cleaning is audited, not silent.** Every cleaning step logs how many rows it touched
(`cleaning_report.csv`). In production, alert when a step's touch-count jumps; upstream
schema drift is the number-one silent model killer.

## Repository layout

```
src/delivery_commit/
  schema.py      canonical shipment schema + validation
  synthetic.py   messy synthetic generator with documented ground-truth drivers
  cleaning.py    audited cleaning: duplicates, sentinels, bounds, imputation
  features.py    feature engineering, time-based split, model matrix
  train.py       logistic baseline + XGBoost with early stopping
  evaluate.py    PR-AUC, calibration, decile lift, capacity-threshold metrics
  explain.py     SHAP global + local reports, driver ranking vs ground truth
  olist.py       adapter: public Olist dataset -> canonical schema
  cli.py         delivery-commit generate | all | score
tests/           end-to-end tests incl. "SHAP recovers the true drivers"
```

## Contributing

Issues and PRs welcome, especially adapters for other public logistics datasets,
alternative model families, and intervention-cost analysis (turning risk scores into
reroute/upgrade decisions). Please keep the two invariants: no feature that isn't
knowable at induction time, and no explanation output without a test that grounds it.

## License

Apache-2.0
