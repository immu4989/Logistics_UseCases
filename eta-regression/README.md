# ⏱️ ETA Regression

**When will each package actually arrive — and what transit time can you promise a customer and keep 9 times out of 10?**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-green)
![Models](https://img.shields.io/badge/models-XGBoost%20quantile%20%2B%20linear-orange)
![Explainability](https://img.shields.io/badge/explainability-SHAP%2C%20tested-purple)
![Lint](https://img.shields.io/badge/lint-ruff-red)

A tracking page wants one number; a customer promise needs three. This project predicts
transit time **at induction time** (the moment a package enters the network) as a full
quantile set: P50 is the ETA you display, P10–P90 is the honest range, and ceil(P90) is
the transit time you can commit to. The distinction matters because transit-time
variance is not constant — a 3,000-mile ground lane through congested hubs is roughly
three times noisier than a short intra-region hop, so any single-number ETA is precise
exactly where it's safe and vague exactly where it's expensive.

One command runs the entire journey, no data downloads, ~2 minutes on a laptop:

```bash
pip install -e .
eta-regression all
```

```mermaid
flowchart LR
    A[("Raw extract<br/>60k shipments,<br/>deliberately messy")] --> B["Audited cleaning<br/>+ label sanity"]
    B --> C["Feature<br/>engineering"]
    C --> D["Time-based<br/>split"]
    D --> E["Linear baseline<br/>+ XGBoost point<br/>+ quantile models"]
    E --> F["Evaluation<br/>MAE · pinball · coverage<br/>· promise table"]
    E --> G["SHAP driver<br/>analysis (P50)"]
    F --> H[("artifacts/reports/<br/>metrics, plots,<br/>per-shipment ETAs")]
    G --> H
```

## 🎯 The headline numbers

Held-out final month, ~11.7k shipments, mean actual transit 3.69 days. Nothing here was
tuned on the test period.

| | Linear baseline | XGBoost point | XGBoost P50 |
|---|---:|---:|---:|
| MAE | 0.601 d | **0.439 d** | 0.441 d |
| RMSE | 0.776 d | **0.610 d** | 0.612 d |
| Median APE | 14.1% | **9.3%** | 9.3% |

The point ETA is the easy part: the gradient-boosted model lands within about
**10.5 hours** of the truth on average. The product, though, is the interval:

| P10–P90 interval | Value |
|---|---:|
| Empirical coverage (nominal 80%) | **78.6%** |
| Mean width | 1.38 days |
| Pinball loss q10 / q50 / q90 | 0.098 / 0.220 / 0.099 |

![Median ETA vs actual](docs/img/pred_vs_actual.png)

## 🤝 The promise table

This is the trade the revenue and ops teams argue over. Quote ceil(P50) and the site
shows fast delivery dates that break one time in five. Quote ceil(P90) and the promise
holds for **97% of shipments** at a cost of **0.69 extra quoted days** per shipment:

| Quoting policy | Promise kept | Mean promised days |
|---|---:|---:|
| ceil(P50) | 79.3% | 4.16 d |
| ceil(P90) | **97.0%** | 4.85 d |

The full curve, from real quantile models trained at seven alphas, shows the whole
frontier — the elbow between 0.7 and 0.9 is where an extra tenth of a quoted day buys
the most reliability:

![Promise curve](docs/img/promise_curve.png)

Note the ceiling effect: quoting ceil(P90) keeps 97% of promises, not 90%, because
rounding a 4.2-day P90 up to 5 whole days adds cushion. If 97% is more reliability than
you want to pay for, the curve says quoting the 0.7 or 0.8 quantile already clears a
90% kept-promise bar on this network.

## 📏 Intervals that stay honest where the noise lives

The generator's noise grows with distance and congestion by design. A model that only
matched the *average* spread would over-cover short lanes and under-cover long ones —
the failure mode that gets interval models quietly turned off after the first quarter.
Segmented coverage is the check: all three lane groups sit near the nominal 80%, and the
model pays for it with wider intervals on long lanes (1.76 d) than short ones (1.00 d),
which is exactly the honest behavior:

| Lane group | Mean distance | Coverage | Mean width |
|---|---:|---:|---:|
| Short | 364 mi | 78.6% | 1.00 d |
| Medium | 958 mi | 79.0% | 1.38 d |
| Long | 1,647 mi | 78.1% | 1.76 d |

![Coverage by lane length](docs/img/coverage_by_distance.png)

## 🔍 What actually drives transit time

SHAP on the P50 model, where every value reads directly in days. Ground service and
lane distance dominate (as any dispatcher would tell you), with congestion, weather and
peak surges layered on top:

![SHAP summary](docs/img/shap_summary.png)

Grouped back to operational levers (one-hot columns re-aggregated), the global ranking:

| Rank | Driver | Share of model explanation | |
|---:|---|---:|---|
| 1 | Service level | 36.8% | `██████████████████` |
| 2 | Lane distance | 19.6% | `██████████` |
| 3 | Total hub congestion | 8.2% | `████` |
| 4 | Destination weather severity | 8.1% | `████` |
| 5 | Peak-season surge | 6.6% | `███` |
| 6 | Rural destination | 5.2% | `███` |
| 7 | Weekend induction | 3.5% | `██` |
| 8 | Route stop density | 3.2% | `██` |
| 9 | Estimated linehaul legs | 2.8% | `█` |
| 10 | Origin hub congestion | 1.8% | `█` |

And the sanity check that makes this ranking trustworthy: the generator plants known
noise features, and the model correctly buries them.

| Planted noise feature | Share of model explanation |
|---|---:|
| Package weight | 0.29% |
| Package volume | 0.25% |
| Declared value | 0.20% |
| Signature required | 0.07% |

### Dependence plots: the shape of each driver

Distance contributes time smoothly and then steepens past ~2,000 miles, where extra
sort legs stack up; the ground-service dummy is a clean two-level split worth roughly a
day and a half against the express services:

| Lane distance | Ground service |
|---|---|
| ![Distance dependence](docs/img/shap_dependence_distance_miles.png) | ![Ground dependence](docs/img/shap_dependence_service_level_ground.png) |

### From global drivers to one quoted window

The pipeline also writes the explanation a customer-service screen would show next to a
quoted window (`artifacts/reports/example_shipment.md`). A real example from this run,
a 2,970-mile ground shipment to a rural destination during peak — quoted **P50 8.6 days,
P90 10.3 days, promise 11 days**:

| Driver | Value | Contribution to P50 ETA |
|---|---:|---:|
| Lane distance | 2,970 mi | +2.15 d |
| Ground service | yes | +0.92 d |
| Estimated linehaul legs | 4 | +0.35 d |
| Rural destination | yes | +0.30 d |
| Total hub congestion | 1.14 / 2.0 | +0.27 d |
| Route stop density | 7.51 | +0.19 d |
| Peak season | yes | +0.19 d |

Nobody needs a data science degree to believe this window, which is precisely what
makes it quotable.

### The trick that makes the SHAP report testable

The synthetic generator ([synthetic.py](src/eta_regression/synthetic.py)) has a
*documented causal process*: the per-service speeds, congestion and weather delay
coefficients, and — critically — the heteroscedastic noise formula are right there in
the code, alongside deliberately planted noise features. The test suite asserts that
SHAP recovers the real drivers, that the planted noise stays buried, and that P10–P90
coverage on held-out data lands in [0.70, 0.90]. If a refactor silently breaks the
intervals or the explanations, **CI fails**. Keep this harness when you adapt the
pipeline to your own data.

## 🚚 Run it on real data (Olist)

The repo includes an adapter for the public
[Olist Brazilian e-commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce):
~96k real delivered orders with purchase and customer-delivery timestamps, so the
transit-time label is real, not simulated.

```bash
kaggle datasets download -d olistbr/brazilian-ecommerce -p data/olist --unzip
eta-regression all --source olist --olist-dir data/olist
# artifacts land in ./artifacts-olist, never clobbering the synthetic run
```

[olist.py](src/eta_regression/olist.py) is intentionally the template to copy when you
write the adapter for your own company's data: it maps what Olist has onto the canonical
schema (seller→customer haversine distance from zip-prefix centroids, product
weight/dims, coarse state regions, the checkout promise window), fills what it lacks
(hub telemetry, weather) with neutral constants, and the rest of the pipeline runs
unchanged.

### Olist results: real transit times, and the strike that bends the intervals

Held out: the final ~3 months (19,150 orders after 2018-05-27), mean actual transit
8.68 days on a network whose overall median is ~10 days. Nothing was tuned on the test
period.

| | Linear baseline | XGBoost point | XGBoost P50 |
|---|---:|---:|---:|
| MAE | 4.64 d | 4.95 d | **3.82 d** |
| Median APE | 54.0% | 59.9% | **41.5%** |

Note the inversion versus the synthetic network: the squared-error tree model loses to
the linear baseline here, while the pinball-loss P50 beats both by a wide margin. The
reason is the same event that dominates the delivery-commit use case's Olist run —
Brazil's May 2018 truckers' strike sits right at the train/test boundary. Squared error
chases the strike-era slow tail it was trained on; a median objective largely shrugs
it off.

| P10–P90 interval (nominal 80%) | Raw | + conformal |
|---|---:|---:|
| Empirical coverage | 75.3% | **77.9%** |
| Mean width | 14.83 d | 15.16 d |

The raw quantile models run miscalibrated on real data — there is no coverage guarantee
in gradient boosting, which is why the pipeline now carries a split-conformal (CQR)
correction calibrated on the last slice of training time. It buys back about half the
gap (qhat = +0.16 d per side). The other half is not a calibration problem at all:

![Olist coverage by month](docs/img/olist_coverage_by_month.png)

The miss is one-sided — 21.7% of orders finish *below* P10 versus 3.0% above P90 —
because models trained through the strike keep over-predicting a network that is
recovering underneath them. Post-strike, actual transit falls month over month (11.8 →
9.2 → 8.9 → 7.7 days) and Olist simultaneously unwinds its inflated checkout promises
(mean promise window 36 → 15 days). That promise window is the model's second-biggest
driver (28% of the SHAP explanation, behind distance at 34%), so predictions deflate
with it and coverage climbs back from 70% in June to 81% by August. A constant conformal
offset, calibrated on strike-era months, cannot fix a drift — only a retrain cadence and
drift monitoring can. This is a finding to keep, not a number to tune away.

The promise table still earns its keep on real orders: quote ceil(P50) and 78.3% of
promises hold at 11.0 quoted days; quote ceil(P90) and **97.2%** hold at 21.1 days. Ten
extra quoted days is the honest price of a 97% promise on a continent-sized network
predicted from purchase-time features alone — and the curve shows the cheaper middle:
the 0.7 quantile lands essentially on a 90% kept-promise bar (89.8%) at 14.1 quoted
days, and the 0.8 quantile clears it (94.2%) at 16.5.

## 🏭 Adapting to your own shipment data

1. Produce a DataFrame with the canonical columns in
   [schema.py](src/eta_regression/schema.py). Only three are truly required to be
   meaningful: a shipment id, a ship date, and the `actual_transit_days` label. Every
   feature you can populate improves the model; anything you can't, fill with a neutral
   constant.
2. **Respect induction time.** Every feature must be knowable when the package enters
   your network. Scan counts and dwell times accumulated during transit predict arrival
   brilliantly and uselessly; they are how ETA models leak.
3. Run `eta-regression all`, read `artifacts/reports/`, then quote new shipments:

```bash
eta-regression score --input tomorrow.csv
# shipment_id, eta_p10, eta_p50, eta_p90, promised_days (= ceil of P90)
```

## 🧠 Design decisions that make or break ETA models

**Time-based split, never random.** Shipments from the same lane and day are heavily
correlated. A random split leaks future operating conditions into training and inflates
offline metrics that then evaporate in deployment. Training uses the first ~80% of ship
dates; everything after the cutoff is held out, and early stopping validates on the
last slice of *training* time only.

**Quantiles, not a point ETA.** Transit-time noise grows with distance and congestion,
so one number is simultaneously too cautious for the short-haul overnight and a lie for
the 3,000-mile ground lane. Quantile regression prices the uncertainty per shipment;
the promise table then turns that into a business decision instead of a modeling one.
The three product quantiles are monotone-rearranged at prediction time (sorted per
row), because independently trained quantile models can cross, and one crossed pair on
an ops screen discredits all three numbers.

**Coverage is segmented, not just averaged.** Aggregate 80% coverage can hide 90% on
short lanes and 65% on long ones — honest on average, wrong for every promise that
matters. The evaluation breaks coverage out by distance tercile and the test suite
checks the aggregate; when you adapt this, segment by whatever your quoting surface is
(lane, service, region).

**Label sanity is part of cleaning, and it is audited.** A classifier shrugs off a few
mislabeled rows; a regressor pulls its fit toward them. The raw extract contains
impossible zero/negative transit times (a delivery scan mis-keyed before induction),
and cleaning drops them — never clips, never imputes — and logs the count in
`cleaning_report.csv` alongside every other step, so upstream schema drift shows up as
a touch-count spike instead of a silent metric slide.

<details>
<summary>📁 Repository layout</summary>

```
src/eta_regression/
  schema.py      canonical shipment schema + validation
  synthetic.py   messy synthetic generator with documented transit process
                 and heteroscedastic noise (the reason quantiles exist)
  cleaning.py    audited cleaning: duplicates, sentinels, bounds, label sanity
  olist.py       adapter: public Olist dataset -> canonical schema (real transit labels)
  features.py    feature engineering, time-based split, model matrix
  train.py       linear baseline + XGBoost point + quantile models (0.1..0.95)
  evaluate.py    MAE/pinball/coverage, conformal (CQR) correction, segmented +
                 monthly coverage, promise table + curve
  explain.py     SHAP on the P50 model: global + local reports, driver ranking
  cli.py         eta-regression generate | all [--source olist] | score
tests/           end-to-end tests incl. coverage honesty and
                 "SHAP recovers the true drivers"
```

</details>

## 🤝 Contributing

Issues and PRs welcome, especially adapters for more public datasets with real transit
times, per-segment conformal calibration (one qhat per lane group instead of a global
constant), and promise-policy cost analysis (turning the promise curve into revenue
impact). Please keep the two
invariants: no feature that isn't knowable at induction time, and no explanation or
interval output without a test that grounds it.

## License

Apache-2.0
