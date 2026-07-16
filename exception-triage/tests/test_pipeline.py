"""End-to-end tests on a mid-sized synthetic sample.

The tests that matter most are the ground-truth checks: because the
generator's routing process is documented (synthetic.TRUE_RULES), we can
assert that the per-queue SHAP analysis surfaces the real drivers of each
queue and buries the planted noise, and that the confidence gate actually
clears the automation bar it advertises. If a refactor silently breaks any
of that, CI fails.
"""

import numpy as np
import pandas as pd
import pytest

from exception_triage import cleaning, evaluate, explain, schema, synthetic
from exception_triage import train as train_mod
from exception_triage.train import rules_route

N_TEST_SAMPLE = 20_000


@pytest.fixture(scope="session")
def raw():
    return synthetic.make_dataset(n=N_TEST_SAMPLE, seed=11, messy=True)


@pytest.fixture(scope="session")
def clean_df(raw):
    df, _ = cleaning.clean(raw)
    return df


@pytest.fixture(scope="session")
def trained(clean_df):
    cfg = train_mod.TrainConfig(n_estimators=300, seed=11)
    return train_mod.train(clean_df, cfg)


@pytest.fixture(scope="session")
def shap_ranking(trained, tmp_path_factory):
    models, splits = trained
    out = tmp_path_factory.mktemp("shap_reports")
    ranking = explain.explain(models, splits, out, max_background=3000)
    return ranking, out


def test_generator_is_deterministic():
    a = synthetic.make_dataset(n=500, seed=3, messy=True)
    b = synthetic.make_dataset(n=500, seed=3, messy=True)
    pd.testing.assert_frame_equal(a, b)


def test_generator_schema_and_queue_mix():
    df = synthetic.make_dataset(n=8000, seed=1)
    schema.validate(df)
    mix = df[schema.LABEL_COL].value_counts(normalize=True)
    # Realistic imbalance, not a knife-edge: generous bands around the
    # calibration targets (addr 24 / reroute 20 / customs 8 / damage 7 /
    # hold 26 / callback 15).
    assert 0.16 < mix["address_correction"] < 0.32
    assert 0.12 < mix["reroute"] < 0.28
    assert 0.04 < mix["customs_docs"] < 0.13
    assert 0.03 < mix["damage_claims"] < 0.12
    assert 0.18 < mix["hold_and_monitor"] < 0.34
    assert 0.08 < mix["customer_callback"] < 0.22
    # customs_docs is impossible for domestic shipments.
    customs = df[df[schema.LABEL_COL] == "customs_docs"]
    assert (customs["is_international"] == 1).all()


def test_cleaning_dedupes_ticket_ids(raw, clean_df):
    assert not raw[schema.ID_COL].is_unique  # the mess is really there
    assert clean_df[schema.ID_COL].is_unique


def test_cleaning_fixes_negative_scan_gaps(raw, clean_df):
    assert (raw["scan_gap_hours"].dropna() < 0).any()
    assert (clean_df["scan_gap_hours"] >= 0).all()
    assert "scan_gap_hours__was_missing" in clean_df.columns


def test_cleaning_normalizes_label_casing(raw, clean_df):
    # The raw extract mixes "Address Correction", "REROUTE", " hold... "
    # variants; after cleaning only the six canonical queue names remain.
    assert raw[schema.LABEL_COL].nunique() > len(schema.QUEUES)
    assert set(clean_df[schema.LABEL_COL].unique()) == set(schema.QUEUES)


def test_cleaning_imputes_missing_flags(raw, clean_df):
    assert raw["weather_event_at_location"].isna().any()
    for col in ["weather_event_at_location", "address_validation_failed"]:
        assert clean_df[col].notna().all()
        assert set(clean_df[col].unique()) <= {0, 1}
        assert f"{col}__was_missing" in clean_df.columns


def test_time_split_no_leakage(trained):
    _, splits = trained
    assert splits["train"][schema.DATE_COL].max() <= splits["test"][schema.DATE_COL].min()


def test_xgboost_beats_logistic_and_rules(trained):
    models, splits = trained
    test_df, X_test = splits["test"], splits["X_test"]
    y_true = test_df[schema.LABEL_COL].to_numpy()
    classes = np.array(models.classes)

    m_rules = evaluate.summarize(y_true, rules_route(test_df))
    m_logit = evaluate.summarize(y_true, classes[models.baseline.predict(X_test)])
    m_xgb = evaluate.summarize(
        y_true, classes[models.xgb.predict_proba(X_test).argmax(axis=1)]
    )

    # The GBM must beat both comparators by a healthy margin, not a rounding
    # error — that's the bar for replacing a rules router anyone can read.
    assert m_xgb["macro_f1"] > m_rules["macro_f1"] + 0.05
    assert m_xgb["macro_f1"] > m_logit["macro_f1"] + 0.02
    assert m_xgb["accuracy"] > m_rules["accuracy"] + 0.05


def test_cost_weighted_delay_model_below_rules(trained):
    models, splits = trained
    test_df, X_test = splits["test"], splits["X_test"]
    y_true = test_df[schema.LABEL_COL].to_numpy()
    classes = np.array(models.classes)

    delay_rules = evaluate.mean_delay_days(y_true, rules_route(test_df))
    delay_xgb = evaluate.mean_delay_days(
        y_true, classes[models.xgb.predict_proba(X_test).argmax(axis=1)]
    )
    assert delay_xgb < delay_rules


def test_automation_operating_point(trained):
    models, splits = trained
    y_true = splits["test"][schema.LABEL_COL].to_numpy()
    probs = models.xgb.predict_proba(splits["X_test"])

    sweep = evaluate.automation_sweep(y_true, probs, models.classes)
    op = evaluate.pick_operating_point(sweep, target_accuracy=0.97)

    assert op["auto_accuracy"] >= 0.97
    # The product bar: at 97% auto-route accuracy, at least 40% of tickets
    # never wait for a human.
    assert op["frac_auto"] >= 0.40
    # And the hybrid policy's misroute cost stays below the rules router's.
    delay_rules = evaluate.mean_delay_days(y_true, rules_route(splits["test"]))
    assert op["hybrid_delay_days"] < delay_rules


def test_shap_recovers_true_queue_mapping(shap_ranking):
    ranking, _ = shap_ranking

    def top(queue: str, k: int = 1) -> list[str]:
        sub = ranking[ranking["queue"] == queue].nsmallest(k, "rank")
        return sub["driver"].tolist()

    # The documented mapping (synthetic.TRUE_RULES) must be recovered.
    assert top("damage_claims") == ["damage_scan_flag"]
    assert top("hold_and_monitor") == ["weather_event_at_location"]
    assert top("address_correction") == ["address_validation_failed"]
    assert top("customs_docs")[0] in {"last_scan_location_type", "is_international"}

    # Planted noise must stay buried in every queue. Rank alone is brittle in
    # queues with one dominant driver (everything below it is near zero and
    # ordinal position among near-zeros is arbitrary), so assert magnitude:
    # noise never reaches the podium, and never carries a tenth of the push
    # of the queue's real top driver.
    noise = {"csr_id", "ticket_created_hour_of_day"}
    for queue in schema.QUEUES:
        sub = ranking[ranking["queue"] == queue]
        assert not set(top(queue, 3)) & noise
        top_push = sub[sub["rank"] == 1]["mean_push_shap"].iloc[0]
        noise_push = sub[sub["driver"].isin(noise)]["mean_push_shap"].max()
        assert noise_push < 0.10 * top_push, (
            f"noise not buried for {queue}: {noise_push:.3f} vs top {top_push:.3f}"
        )


def test_ticket_cards_written(shap_ranking):
    _, out = shap_ranking
    auto = (out / "ticket_card_autoroute.md").read_text()
    esc = (out / "ticket_card_escalation.md").read_text()
    assert "auto-route" in auto
    assert "escalate" in esc
    for card in (auto, esc):
        assert "%" in card  # class probabilities are shown
        assert "SHAP" in card
        assert "In plain language" in card
