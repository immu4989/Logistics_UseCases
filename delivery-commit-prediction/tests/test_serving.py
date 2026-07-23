"""Serving-layer tests. Skipped automatically when the serve extra isn't installed."""

import json
import sys
from pathlib import Path

import pytest

# serving/ is a deployment folder, not part of the installed package; make it
# importable when tests run from the repo checkout.
sys.path.insert(0, str(Path(__file__).parent.parent))

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from delivery_commit import cleaning, schema, synthetic  # noqa: E402
from delivery_commit import train as train_mod  # noqa: E402


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    """Train a small but real model artifact for the service to load."""
    raw = synthetic.make_dataset(n=6000, seed=21, messy=True)
    clean_df, _ = cleaning.clean(raw)
    models, _ = train_mod.train(clean_df, train_mod.TrainConfig(n_estimators=60, seed=21))
    d = tmp_path_factory.mktemp("models")
    train_mod.save(models, d)
    return d


@pytest.fixture()
def client(model_dir, monkeypatch):
    monkeypatch.setenv("DELIVERY_COMMIT_MODEL_DIR", str(model_dir))
    import serving.app as app_mod

    app_mod._state["models"] = None  # force reload against the fixture artifact
    app_mod._state["model_dir"] = None
    return TestClient(app_mod.app)


def _example_shipments(n=5):
    df = synthetic.make_dataset(n=n, seed=99, messy=False).drop(columns=[schema.LABEL_COL])
    df[schema.DATE_COL] = df[schema.DATE_COL].dt.strftime("%Y-%m-%d")
    return df.to_dict(orient="records")


def test_health_before_and_after_scoring(client):
    r = client.get("/health")
    assert r.status_code == 200

    r = client.post("/score", json={"shipments": _example_shipments()})
    assert r.status_code == 200

    r = client.get("/health")
    assert r.json()["status"] == "ok"
    assert r.json()["trained_through"]


def test_score_returns_ranked_probabilities(client):
    r = client.post("/score", json={"shipments": _example_shipments(8)})
    assert r.status_code == 200
    body = r.json()
    assert body["scored"] == 8
    probs = [row["miss_probability"] for row in body["results"]]
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert probs == sorted(probs, reverse=True)


def test_score_rejects_missing_columns(client):
    r = client.post("/score", json={"shipments": [{"shipment_id": "X", "ship_date": "2025-01-01"}]})
    assert r.status_code == 422


def test_score_rejects_empty(client):
    r = client.post("/score", json={"shipments": []})
    assert r.status_code == 400


def test_example_request_file_is_valid(client):
    payload = json.loads(
        (Path(__file__).parent.parent / "serving" / "example_request.json").read_text()
    )
    r = client.post("/score", json=payload)
    assert r.status_code == 200
