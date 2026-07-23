# Contributing

Thanks for your interest. This repo grows one way: complete, self-contained use cases
that a shipping team could actually adapt. Partial pipelines, notebooks without tests,
and models without explanations don't fit here, however clever the modeling.

## The two invariants

Every use case keeps these, and PRs that break them will be asked to fix it:

1. **No leakage past decision time.** Every feature must be knowable at the moment the
   prediction or decision would actually be consumed: induction time for shipment risk,
   order time for returns, booking time for capacity. If a feature is observed after
   that moment, it does not enter the model matrix, no matter how much AUC it buys.
2. **No unexplained model.** Explainability output is grounded by tests. On synthetic
   data that means the generator documents its true drivers (and plants noise features)
   and the test suite asserts the explanation layer recovers the former and buries the
   latter. Where SHAP is the wrong tool, the explanation is a per-decision rationale or
   incident card, still exercised by tests.

## Adding a new use case

Copy the structure of an existing one (`delivery-commit-prediction/` is the canonical
template). The checklist:

- [ ] Own folder with `pyproject.toml` (Apache-2.0, `requires-python >= 3.10`,
      dev extras `pytest` + `ruff`), `src/<package>/`, `tests/`, `README.md`, `docs/img/`.
- [ ] A synthetic generator with a **documented ground-truth process** (module
      docstring + exposed constants) and a `messy=True` mode injecting realistic data
      defects; an audited `CleaningReport`-style cleaning step that undoes them.
- [ ] Time-based train/test split wherever the data has a time axis. Never random.
- [ ] An honest status-quo baseline (a rule, a habit, a fixed policy — the thing ops
      does today), not a strawman.
- [ ] Evaluation in operational units someone budgets in: dollars, delay-days,
      wrong doors, understaffed days.
- [ ] A CLI (`<name> generate | all`) with staged progress output; artifacts land in
      `artifacts/` (gitignored).
- [ ] Tests green: `pytest -q`, `ruff check src tests`, and the CLI `all` run.
- [ ] README in the house voice with the **real numbers from your run**, a mermaid
      pipeline diagram, figures committed to `docs/img/`, and a design-decisions
      section that says why, not just what.
- [ ] Registered in the CI matrix (`.github/workflows/ci.yml`) and the root README
      catalog table.

Two environment gotchas that will bite you otherwise:

- **shap/xgboost pairing:** `shap>=0.52` (needed for xgboost 3.x models) only ships for
  Python >= 3.12. Copy the environment-marker dependency block from
  `delivery-commit-prediction/pyproject.toml` verbatim if your use case uses SHAP.
- **pandas 3.x strings:** the default string dtype is `str`, not `object`. Use
  `pd.api.types.is_string_dtype(...)`, never `dtype == object`.

## Adding a real-data adapter

Adapters that validate a use case on public data are among the most valuable
contributions. Rules:

- **Check the license before anything else.** Redistribution-permitted data (e.g.
  CC BY) may be committed with attribution; restricted data (e.g. Olist's CC BY-NC-SA)
  is never committed — ship a loader plus download instructions, and guard the tests
  with `pytest.mark.skipif(<data missing>)` so CI stays green.
- Map onto the use case's existing schema; fill columns the source lacks with neutral
  constants and say so.
- Report what the real data actually shows, including failures. The Olist section in
  `delivery-commit-prediction/README.md` documents a tree-model collapse under a
  regime shift rather than tuning it away; that honesty is the house style.

## Development setup

```bash
cd <use-case-folder>
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q && .venv/bin/ruff check src tests
```

## Questions and proposals

Open an issue describing the operational question, the data you'd model it on (public
or synthetic), and the decision the output informs. That framing, not the model
architecture, is what gets a proposal accepted quickly.
