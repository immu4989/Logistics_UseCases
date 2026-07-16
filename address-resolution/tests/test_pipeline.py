"""End-to-end tests on a small synthetic city.

The load-bearing assertions are the operational ones: blocking recall (a true
candidate that falls out of every block is unrecoverable), the scorer beating
both status-quo baselines on the axis each one loses (false-match rate against
fuzzy top-1 at equal coverage, coverage against exact match at equal
precision), orphan detection at the default operating point, and threshold
monotonicity — the guarantee that lets an operator turn the dial toward
precision without reading the code.
"""

import numpy as np
import pandas as pd
import pytest

from address_resolution import evaluate, explain, resolve, synthetic
from address_resolution import train as train_mod

SEED = 11


@pytest.fixture(scope="session")
def city():
    points = synthetic.make_city(seed=SEED, streets_per_zip=6, buildings_per_street=10)
    labels = synthetic.make_labels(points, n=6000, seed=SEED)
    return points, labels


@pytest.fixture(scope="session")
def run(city):
    points, labels = city
    return train_mod.run(points, labels, train_mod.ResolverConfig(seed=SEED))


@pytest.fixture(scope="session")
def test_dec(run):
    return run.decisions[~run.decisions["is_train"]].reset_index(drop=True)


def test_generator_is_deterministic():
    pa = synthetic.make_city(seed=3, streets_per_zip=4, buildings_per_street=6)
    pb = synthetic.make_city(seed=3, streets_per_zip=4, buildings_per_street=6)
    pd.testing.assert_frame_equal(pa, pb)
    la = synthetic.make_labels(pa, n=800, seed=3)
    lb = synthetic.make_labels(pb, n=800, seed=3)
    pd.testing.assert_frame_equal(la, lb)


def test_corruption_ladder_is_recorded_correctly(city):
    """The corruptions column must describe what actually happened to the text.

    Checked property-by-property against the source delivery point, because
    the whole error taxonomy in evaluate.py hangs off this record.
    """
    points, labels = city
    pts = points.set_index("point_id")
    lnorm = resolve.normalize_labels(labels)
    matchable = labels[labels["true_point_id"] != ""]
    for i in matchable.index[:2000]:
        corr = set(labels.loc[i, "corruptions"].split("|")) - {""}
        src = pts.loc[labels.loc[i, "true_point_id"]]
        norm = lnorm.iloc[i]
        if "unit_dropped" in corr:
            assert norm["unit"] == "" and src["unit"] != ""
        if "digits_transposed" in corr:
            assert norm["number"] != str(src["street_number"])
            assert sorted(norm["number"]) == sorted(str(src["street_number"]))
        if "wrong_zip" in corr:
            assert labels.loc[i, "zip"] != src["zip"]
            assert synthetic.zip_distance(labels.loc[i, "zip"], src["zip"]) == 1
        if "typo_street_name" in corr:
            assert norm["name"] != src["street_name"].upper()
        # Everything the normalizer is supposed to absorb must normalize back
        # to the canonical record exactly.
        cosmetic = {"street_type_variant", "unit_format", "extra_tokens", "casing_whitespace"}
        if corr <= cosmetic:
            assert norm["number"] == str(src["street_number"])
            assert norm["name"] == src["street_name"].upper()
            assert norm["unit"] == src["unit"].upper()


def test_blocking_recall(run):
    assert run.blocking_recall >= 0.97


def test_scorer_beats_fuzzy_at_equal_coverage(run, test_dec):
    """At threshold 0 both systems accept whenever candidates exist — same
    coverage by construction — so any false-match gap is pure ranking skill."""
    test_mask = ~run.is_train
    fuzzy = resolve.fuzzy_top1_baseline(run.lnorm, run.pnorm, run.li, run.pi, test_mask)
    idx_of = {pid: i for i, pid in enumerate(run.points["point_id"])}
    true_idx = np.array(
        [idx_of.get(t, -1) for t in run.labels["true_point_id"].fillna("")], dtype=np.int64
    )
    fz = fuzzy[test_mask]
    fz_acc = fz >= 0
    fz_precision = float((fz[fz_acc] == true_idx[test_mask][fz_acc]).mean())

    m0 = evaluate.point_metrics(test_dec, threshold=0.0)
    assert abs(m0["coverage"] - fz_acc.mean()) < 1e-9
    assert (1 - m0["precision"]) < (1 - fz_precision), (
        f"scorer false-match rate {1 - m0['precision']:.4f} not below "
        f"fuzzy top-1 {1 - fz_precision:.4f} at equal coverage"
    )


def test_scorer_coverage_far_above_exact_match_at_high_precision(run, test_dec):
    test_mask = ~run.is_train
    exact = resolve.exact_match_baseline(run.lnorm, run.pnorm)[test_mask]
    ex_coverage = float((exact >= 0).mean())
    curve = evaluate.precision_coverage_curve(test_dec)
    cov995 = evaluate.coverage_at_precision(curve, 0.995)["coverage"]
    assert cov995 >= ex_coverage + 0.20, (
        f"coverage at 99.5% precision ({cov995:.1%}) should clear exact match "
        f"({ex_coverage:.1%}) by 20+ points"
    )
    # Exact match's virtue for context: it is (nearly) perfectly precise.
    idx_of = {pid: i for i, pid in enumerate(run.points["point_id"])}
    true_idx = np.array(
        [idx_of.get(t, -1) for t in run.labels["true_point_id"].fillna("")], dtype=np.int64
    )
    acc = exact >= 0
    assert float((exact[acc] == true_idx[test_mask][acc]).mean()) >= 0.999


def test_orphan_recall_at_default_threshold(run, test_dec):
    m = evaluate.point_metrics(test_dec, run.resolver.threshold)
    assert m["orphan_recall"] >= 0.70, (
        f"only {m['orphan_recall']:.1%} of true no-matches reached the review queue"
    )


def test_threshold_monotonicity(test_dec):
    """Raising the threshold must shrink the auto-match queue and must never
    lower its precision on this data — the property that makes the operating
    table safe to hand to a non-modeler."""
    grid = [0.3, 0.5, 0.7, 0.9, 0.97]
    metrics = [evaluate.point_metrics(test_dec, t) for t in grid]
    for lo, hi in zip(metrics, metrics[1:]):
        assert hi["coverage"] <= lo["coverage"]
        assert hi["precision"] >= lo["precision"]


def test_rationale_cards_written(run, tmp_path):
    path = explain.write_rationale(run, tmp_path)
    text = path.read_text()
    assert text.count("## Card") == 3
    assert "unit conflict blocked the match" in text
    assert "true no-match" in text
    for feature in run.resolver.feature_names:
        assert explain._DISPLAY[feature] in text


def test_resolver_roundtrip(run, tmp_path):
    train_mod.save(run.resolver, tmp_path)
    loaded = train_mod.load(tmp_path)
    assert loaded.threshold == run.resolver.threshold
    p = loaded.pipeline.predict_proba(run.X[:100])[:, 1]
    assert np.allclose(p, run.probs[:100])
