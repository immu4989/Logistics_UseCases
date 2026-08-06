---
title: Shipping & Logistics ML Demo
emoji: 🚚
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.50.0
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
---

# Shipping & Logistics ML — interactive demo

A live front end for three of the twelve use cases in
**[Shipping and Logistics Use Cases](https://github.com/immu4989/Logistics_UseCases)**.
Build a shipment and watch two models score it, then spend an intervention budget and
see which allocation policy actually saves money.

- **Will it miss the promise?** — `delivery-commit-prediction`: an XGBoost risk score with
  a per-shipment SHAP breakdown of what drove it.
- **When will it arrive?** — `eta-regression`: XGBoost quantile models return a P10/P50/P90
  transit-time interval and a keepable promise date.
- **Spend the budget** — `intervention-optimization`: expected-value allocation over a day
  of 20,000 shipments, compared against flagging the riskiest and against a perfect-scores
  oracle.

Everything runs on documented synthetic generators trained in-process; there is no private
data behind this Space, and the numbers reproduce the repository's tests. Models are
trained small for a fast demo (cold start ~20s while they build, then cached); the repo's
headline numbers come from full runs.

## 🔌 Agents can call these models too (MCP)

The Space doubles as an **MCP server**: the three functions are exposed as tools with
typed schemas, so any MCP-capable agent (Claude, or your own) can score shipments,
pull ETA intervals, and run the budget allocator programmatically.

```
https://immu4989-logistics-usecases-demo.hf.space/gradio_api/mcp/
```

Tools: `score_commit`, `predict_eta`, `run_budget` — parameter names and descriptions
come straight from [logic.py](logic.py)'s signatures and docstrings. Point an MCP
client at the URL above (streamable HTTP transport) and the tools appear; no auth
needed for this public demo.

## Running locally

```bash
cd demo
python3.12 -m venv .venv
.venv/bin/pip install -e ../delivery-commit-prediction -e ../eta-regression \
                      -e ../intervention-optimization "gradio[mcp]>=5,<6"
.venv/bin/python app.py
```

## Deploying to Hugging Face Spaces

Push this `demo/` folder to a Gradio Space. `requirements.txt` installs the three
use-case packages directly from the GitHub repo, so the Space needs no vendored code.
Pin the git URLs to a tag or commit for reproducible builds.
