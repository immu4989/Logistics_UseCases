# Serving the commit-miss model

The deployment template for this repo: one FastAPI service that loads the trained
artifact and scores raw shipment rows with the exact cleaning and feature code used in
training. Copy this folder's pattern into any other use case.

## Local

```bash
pip install -e ".[serve]"
delivery-commit all                     # writes artifacts/models/
uvicorn serving.app:app --port 8000
curl -s localhost:8000/health
curl -s -X POST localhost:8000/score \
     -H "content-type: application/json" \
     -d @serving/example_request.json
```

Responses return shipments ranked by miss probability, plus the training cutoff date
(`trained_through`) so callers can detect a stale model.

## Docker

```bash
docker build -f serving/Dockerfile -t delivery-commit-scoring .
docker run -p 8000:8000 \
  -v $(pwd)/artifacts/models:/app/artifacts/models \
  delivery-commit-scoring
```

The model is mounted, not baked in, so the image is rebuilt for code changes and the
artifact rotates on your retrain cadence independently.

## Batch scoring

Nightly files don't need HTTP. The CLI already is the batch path:

```bash
delivery-commit score --input tomorrow.csv --out scored.csv
```

## Design notes

Serving reuses `cleaning.clean` and `features.to_matrix` from the package rather than
reimplementing them: training/serving skew is the most common way scoring services rot,
and the way to prevent it is to make skew impossible, not to test for it. The service
refuses to score until an artifact loads and reports its training cutoff in `/health`;
wire that into your rollout gate. Retraining cadence and drift monitoring expectations
live in [MODEL_CARD.md](MODEL_CARD.md).
