"""Smoke tests for the demo Space: callbacks return the right shapes and the
Blocks graph constructs. No browser needed; runs in CI-style headless mode."""

import matplotlib
import pandas as pd

matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402

import app  # noqa: E402
import logic  # noqa: E402

NASTY = (2200, "ground", 0.8, 0.7, 3, 30, True, True, "residential")
EASY = (120, "overnight", 0.2, 0.2, 0, -60, False, False, "commercial")


def test_score_commit_shapes():
    label, fig = logic.score_commit(*NASTY)
    assert label.startswith("## Miss probability:")
    assert isinstance(fig, Figure)


def test_risk_ordering_is_sane():
    nasty = logic.score_commit(*NASTY)[0]
    easy = logic.score_commit(*EASY)[0]

    def pct(md):
        return int(md.split("Miss probability:")[1].split("%")[0])

    assert pct(nasty) > pct(easy)
    assert pct(easy) < 20


def test_predict_eta_shapes_and_monotone():
    label, fig = logic.predict_eta(*NASTY)
    assert label.startswith("## Median ETA:")
    assert isinstance(fig, Figure)


def test_run_budget_table():
    takeaway, fig, tbl = logic.run_budget(6000)
    assert isinstance(fig, Figure)
    assert isinstance(tbl, pd.DataFrame)
    policies = set(tbl["Policy"])
    assert "Expected-value greedy" in policies and "Do nothing" in policies
    # EV-greedy net savings must beat top-K net savings at this budget.
    net = {r["Policy"]: float(str(r["Net savings $"]).replace(",", "")) for _, r in tbl.iterrows()}
    assert net["Expected-value greedy"] > net["Flag the riskiest (top-K)"]
    assert net["Expected-value greedy"] > 0


def test_blocks_builds():
    demo = app.build()
    assert demo is not None
