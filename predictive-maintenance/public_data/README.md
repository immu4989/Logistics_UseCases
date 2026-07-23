# AI4I 2020 Predictive Maintenance Dataset

`ai4i2020.csv` — 10,000 machine records with a binary failure label, committed here
verbatim so the real-data tests and the `fleet-maint ai4i` command run without any
download step.

- **Source:** UCI Machine Learning Repository, dataset 601:
  <https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset>
  (downloaded from
  <https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip>)
- **Citation:** S. Matzka, "Explainable Artificial Intelligence for Predictive
  Maintenance Applications", 2020 Third International Conference on Artificial
  Intelligence for Industries (AI4I), 2020, pp. 69–74, doi:10.1109/AI4I49448.2020.00023.
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The dataset is
  redistributed here unmodified, with attribution, as the license permits. It is a
  *synthetic* dataset published by its author to reflect real predictive-maintenance
  data encountered in industry.

Columns: `UDI`, `Product ID`, `Type` (L/M/H quality variant), `Air temperature [K]`,
`Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`,
`Machine failure` (the label), and five failure-mode indicators `TWF`, `HDF`, `PWF`,
`OSF`, `RNF`.

**Warning:** the five failure-mode columns are components of the label, not features.
See [`src/fleet_maintenance/ai4i.py`](../src/fleet_maintenance/ai4i.py) and the
"Real data: AI4I 2020" section of the project README before using them.
