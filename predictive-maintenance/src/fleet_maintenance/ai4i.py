"""Adapter for the UCI AI4I 2020 Predictive Maintenance dataset (real-data validation).

10,000 machine records, binary ``Machine failure`` label at a ~3.4% base rate.
S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance
Applications", AI4I 2020. CC BY 4.0; the CSV is committed at
``public_data/ai4i2020.csv`` with attribution so this runs with no download step.

Two traps this adapter exists to encode, because most public AI4I notebooks fall
into the first one:

1. **The five failure-mode columns (TWF, HDF, PWF, OSF, RNF) are COMPONENTS of
   the label, not features.** ``Machine failure`` is set precisely when one of
   the mode indicators fires, so a model given them "predicts" failure with
   ~99% accuracy by reading the answer off the sheet. They are excluded here
   with ``FAILURE_MODE_COLS``, ``load()`` refuses to emit them, and a test
   asserts the model matrix never contains them.
2. **RNF is label noise by construction.** The generator gives every record a
   0.1% random-failure chance regardless of the process parameters, so ~19
   rows carry a failure signal no feature can explain — and in the published
   file the bookkeeping is itself inconsistent (18 of the 19 RNF rows have
   ``Machine failure = 0``, while 9 positives carry no mode flag at all).
   That caps achievable precision and is a reason to distrust any AI4I result
   claiming near-perfect scores.

**No timestamps exist in AI4I**, so the house rule — time-based splits only —
cannot apply: there is no time axis to split on. ``stratified_split`` is a
plain stratified random split, and that is a real concession. AI4I is a
components-bench dataset (independent records from a documented generator),
not an operations log; nothing here validates the temporal-leakage discipline
the synthetic fleet pipeline exists to demonstrate. It validates the rest:
modelling a rare-failure label on real-schema data, ops-style top-k metrics,
and whether SHAP recovers the documented failure physics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LABEL_COL = "machine_failure"

# COMPONENTS OF THE LABEL — NEVER FEATURES. Machine failure == OR(these).
# Including any of them is answer-key leakage, not modelling.
FAILURE_MODE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

# Identifiers: UDI is a row counter; Product ID is Type + a serial number, so
# it also smuggles no legitimate signal beyond the Type one-hot below.
ID_COLS = ["UDI", "Product ID"]

RAW_FEATURE_RENAMES = {
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
}

TYPE_DUMMIES = ["type_L", "type_M", "type_H"]
FEATURE_COLS = list(RAW_FEATURE_RENAMES.values()) + TYPE_DUMMIES

DEFAULT_CSV = Path(__file__).resolve().parents[2] / "public_data" / "ai4i2020.csv"


def load(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load AI4I into a minimal modelling frame: FEATURE_COLS + LABEL_COL only.

    The failure-mode columns are dropped here, at the door, so nothing
    downstream can pick them up by accident.
    """
    path = Path(csv_path) if csv_path is not None else DEFAULT_CSV
    raw = pd.read_csv(path, encoding="utf-8-sig")

    missing = [c for c in RAW_FEATURE_RENAMES if c not in raw.columns]
    if missing:
        raise ValueError(f"{path} does not look like ai4i2020.csv; missing columns: {missing}")

    df = raw.rename(columns=RAW_FEATURE_RENAMES)[list(RAW_FEATURE_RENAMES.values())].copy()
    for t in ("L", "M", "H"):
        df[f"type_{t}"] = (raw["Type"] == t).astype(float)
    df[LABEL_COL] = raw["Machine failure"].astype(int)

    leaked = set(df.columns) & set(FAILURE_MODE_COLS)
    assert not leaked, f"failure-mode columns leaked into the AI4I frame: {leaked}"
    return df


def to_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Model matrix + label vector. Refuses to emit any failure-mode column."""
    forbidden = set(FAILURE_MODE_COLS) | {LABEL_COL}
    X = df[[c for c in FEATURE_COLS if c in df.columns]].astype(float)
    assert not set(X.columns) & forbidden
    return X, df[LABEL_COL].to_numpy()


def stratified_split(
    df: pd.DataFrame, test_frac: float = 0.25, seed: int = 7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified random split. NOT a time split — AI4I has no timestamps.

    Records are independent draws from the published generator, so a random
    split is statistically sound here; on any operations log with a time axis
    it would be leakage, and features.time_split is the rule.
    """
    rng = np.random.default_rng(seed)
    test_idx = []
    for _, grp in df.groupby(LABEL_COL):
        n_test = int(round(test_frac * len(grp)))
        test_idx.extend(rng.choice(grp.index.to_numpy(), size=n_test, replace=False))
    test_mask = df.index.isin(test_idx)
    return df[~test_mask].copy(), df[test_mask].copy()


# ---------------------------------------------------------------------------
# Documented failure physics (Matzka 2020), used to audit what SHAP recovers.
# HDF: heat dissipation fails when (process - air) temp difference < 8.6 K
#      AND rotational speed < 1380 rpm.
# PWF: power = torque * angular speed outside [3500 W, 9000 W].
# OSF: overstrain when tool wear * torque exceeds 11/12/13 kminNm for L/M/H.
# TWF: tool wear in the 200-240 min replacement window.
# ---------------------------------------------------------------------------

def physics_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean masks for each documented failure-mode condition, from features only."""
    power_w = df["torque_nm"] * df["rotational_speed_rpm"] * 2 * np.pi / 60.0
    osf_limit = 11000.0 * df["type_L"] + 12000.0 * df["type_M"] + 13000.0 * df["type_H"]
    return pd.DataFrame(
        {
            "HDF_zone": (df["process_temp_k"] - df["air_temp_k"] < 8.6)
            & (df["rotational_speed_rpm"] < 1380),
            "PWF_zone": (power_w < 3500) | (power_w > 9000),
            "OSF_zone": df["tool_wear_min"] * df["torque_nm"] > osf_limit,
            "TWF_zone": df["tool_wear_min"].between(200, 240),
        },
        index=df.index,
    )


def physics_check(test_df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """Does the model score the documented failure zones as risky?

    For each zone: rows in it, observed failure rate in/out of it, and the
    model's mean predicted risk in/out. A recovered mode shows both ratios
    well above 1. This is the real-data analogue of the synthetic suite's
    "SHAP buries the planted noise" test: the ground truth is the published
    generator equations instead of our own.
    """
    zones = physics_conditions(test_df)
    y = test_df[LABEL_COL].to_numpy()
    rows = []
    for zone in zones.columns:
        m = zones[zone].to_numpy()
        if m.sum() == 0 or (~m).sum() == 0:
            continue
        rows.append(
            {
                "zone": zone,
                "n_rows": int(m.sum()),
                "failure_rate_in": float(y[m].mean()),
                "failure_rate_out": float(y[~m].mean()),
                "model_risk_in": float(scores[m].mean()),
                "model_risk_out": float(scores[~m].mean()),
                "model_risk_ratio": float(scores[m].mean() / max(scores[~m].mean(), 1e-9)),
            }
        )
    return pd.DataFrame(rows)


def precision_at_frac(y: np.ndarray, scores: np.ndarray, frac: float = 0.03) -> dict:
    """Precision/recall flagging the top ``frac`` of records — the analogue of the
    fleet pipeline's precision-at-workshop-capacity, minus the per-day budget
    (AI4I has no days to budget over)."""
    k = max(1, int(round(frac * len(y))))
    top = np.argsort(-scores)[:k]
    flagged_pos = int(y[top].sum())
    return {
        "flagged": int(k),
        "precision": float(flagged_pos / k),
        "recall": float(flagged_pos / max(y.sum(), 1)),
    }


# ---------------------------------------------------------------------------
# Training + full pipeline (clean -> train -> evaluate -> explain)
# ---------------------------------------------------------------------------

FLAG_FRAC = 0.03  # same 3% flag budget the fleet pipeline uses

GROUPS = {
    "air_temp_k": "air temperature",
    "process_temp_k": "process temperature",
    "rotational_speed_rpm": "rotational speed",
    "torque_nm": "torque",
    "tool_wear_min": "tool wear",
    "type_L": "machine type",
    "type_M": "machine type",
    "type_H": "machine type",
}


def train_models(X_train: pd.DataFrame, y_train: np.ndarray, seed: int = 7):
    """Logistic baseline + XGBoost, same recipe as train.py minus the time axis:
    early-stop against a stratified slice of TRAIN, then refit at that size."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    logit = Pipeline(
        [
            ("scale", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=3000, random_state=seed)),
        ]
    )
    logit.fit(X_train, y_train)

    X_es, X_val, y_es, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=seed
    )
    xgb_params = dict(
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=5,
        reg_lambda=2.0,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
    )
    probe = XGBClassifier(n_estimators=800, early_stopping_rounds=50, **xgb_params)
    probe.fit(X_es, y_es, eval_set=[(X_val, y_val)], verbose=False)
    n_trees = max(int(probe.best_iteration) + 1, 50)
    xgb = XGBClassifier(n_estimators=n_trees, **xgb_params)
    xgb.fit(X_train, y_train, verbose=False)
    return logit, xgb


def run(
    csv_path: str | Path | None = None,
    out_dir: str | Path = "artifacts-ai4i",
    seed: int = 7,
) -> dict:
    """Full AI4I pipeline; writes metrics, plots and rankings, returns the metrics dict."""
    import json

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load(csv_path)
    train_df, test_df = stratified_split(df, seed=seed)
    X_train, y_train = to_xy(train_df)
    X_test, y_test = to_xy(test_df)

    logit, xgb = train_models(X_train, y_train, seed=seed)

    results: dict = {
        "n_rows": int(len(df)),
        "n_test_rows": int(len(test_df)),
        "base_rate": float(df[LABEL_COL].mean()),
        "test_base_rate": float(y_test.mean()),
        "flag_frac": FLAG_FRAC,
        "split": "stratified random (AI4I has no timestamps; see module docstring)",
    }
    scores = {}
    for name, model in [("logistic", logit), ("xgboost", xgb)]:
        s = model.predict_proba(X_test)[:, 1]
        scores[name] = s
        results[name] = {
            "pr_auc": float(average_precision_score(y_test, s)),
            "roc_auc": float(roc_auc_score(y_test, s)),
            **{f"{k}_at_{FLAG_FRAC:.0%}_flagged": v
               for k, v in precision_at_frac(y_test, s, FLAG_FRAC).items()},
        }

    # --- PR curves -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, color in [("logistic", "#dd6b20"), ("xgboost", "#2b6cb0")]:
        prec, rec, _ = precision_recall_curve(y_test, scores[name])
        ax.plot(rec, prec, color=color,
                label=f"{name} (PR-AUC {results[name]['pr_auc']:.3f})")
    ax.axhline(y_test.mean(), color="k", ls="--", lw=1,
               label=f"base rate {y_test.mean():.1%}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("AI4I 2020 held-out precision-recall (failure-mode columns excluded)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curve_ai4i.png", dpi=150)
    plt.close(fig)

    # --- SHAP beeswarm + grouped ranking --------------------------------------
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_test)
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False, max_display=10)
    plt.title("AI4I 2020: what drives machine-failure risk", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary_ai4i.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=X_test.columns)
    ranking = (
        mean_abs.groupby(mean_abs.index.map(lambda c: GROUPS.get(c, c)))
        .sum()
        .sort_values(ascending=False)
        .rename("mean_abs_shap")
        .reset_index()
        .rename(columns={"index": "driver"})
    )
    ranking["share_of_explanation"] = ranking["mean_abs_shap"] / ranking["mean_abs_shap"].sum()
    ranking.to_csv(out_dir / "driver_ranking_ai4i.csv", index=False)
    results["shap_top_drivers"] = ranking["driver"].head(5).tolist()

    # --- physics audit ---------------------------------------------------------
    physics = physics_check(test_df, scores["xgboost"])
    physics.to_csv(out_dir / "physics_check_ai4i.csv", index=False)
    results["physics_check"] = physics.to_dict(orient="records")

    (out_dir / "metrics_ai4i.json").write_text(json.dumps(results, indent=2))
    return results
