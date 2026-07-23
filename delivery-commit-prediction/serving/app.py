"""FastAPI scoring service for the delivery-commit miss model.

This is the copy-me deployment template for every use case in the repo: load the
trained artifact once, run the SAME cleaning and feature code used in training
(never a reimplementation — training/serving skew is how scoring services rot),
and return ranked probabilities.

Run locally:
    pip install -e ".[serve]"
    delivery-commit all                       # produces artifacts/models/
    uvicorn serving.app:app --port 8000

Score:
    curl -s localhost:8000/health
    curl -s -X POST localhost:8000/score -H "content-type: application/json" \
         -d @serving/example_request.json

The model directory is resolved from $DELIVERY_COMMIT_MODEL_DIR (default:
artifacts/models). The service refuses to start scoring until a model loads;
/health reports which artifact is live so your orchestrator can gate rollouts.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from delivery_commit import cleaning, features, schema
from delivery_commit import train as train_mod

app = FastAPI(title="delivery-commit scoring", version="1.0")

_state: dict = {"models": None, "model_dir": None}


def _model_dir() -> Path:
    return Path(os.environ.get("DELIVERY_COMMIT_MODEL_DIR", "artifacts/models"))


def _load() -> None:
    if _state["models"] is None:
        d = _model_dir()
        try:
            _state["models"] = train_mod.load(d)
            _state["model_dir"] = str(d)
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=503,
                detail=f"no model artifact at {d}; train first (delivery-commit all) "
                f"or set DELIVERY_COMMIT_MODEL_DIR",
            ) from e


class ScoreRequest(BaseModel):
    # Each shipment is a dict of canonical raw columns (see schema.py). The
    # service cleans them with the training-time cleaning code, so the same
    # sentinels/casing mess the warehouse system emits is handled here too.
    shipments: list[dict]


@app.get("/health")
def health() -> dict:
    loaded = _state["models"] is not None
    return {
        "status": "ok" if loaded else "no_model_loaded",
        "model_dir": _state["model_dir"],
        "trained_through": getattr(_state["models"], "cutoff_date", None) if loaded else None,
    }


@app.post("/score")
def score(req: ScoreRequest) -> dict:
    _load()
    models = _state["models"]

    if not req.shipments:
        raise HTTPException(status_code=400, detail="empty shipment list")
    df = pd.DataFrame(req.shipments)
    if schema.DATE_COL in df.columns:
        df[schema.DATE_COL] = pd.to_datetime(df[schema.DATE_COL], errors="coerce")
    try:
        schema.validate(df, require_label=False)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    clean_df, _ = cleaning.clean(df)
    X = features.to_matrix(features.engineer(clean_df))
    X = X.reindex(columns=models.feature_columns, fill_value=0.0)
    probs = models.xgb.predict_proba(X)[:, 1]

    out = (
        pd.DataFrame(
            {schema.ID_COL: clean_df[schema.ID_COL], "miss_probability": probs.round(6)}
        )
        .sort_values("miss_probability", ascending=False)
        .to_dict(orient="records")
    )
    return {"scored": len(out), "trained_through": models.cutoff_date, "results": out}
