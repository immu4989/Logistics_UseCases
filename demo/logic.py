"""Inference layer for the demo Space.

Keeps Gradio out of the modeling code: this module trains (once, cached to
disk) small models from the use-case packages and exposes three functions the
UI calls — commit-miss risk with a per-shipment SHAP breakdown, ETA quantiles
with a promise recommendation, and the budgeted-intervention economics.

Models are trained on the packages' own synthetic generators, so the Space
needs no data files and no network. Training is deliberately small (a cold
start is ~20s); results are cached under MODEL_CACHE so warm starts are
instant.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# delivery-commit-prediction
from delivery_commit import cleaning as dc_cleaning
from delivery_commit import features as dc_features
from delivery_commit import schema as dc_schema
from delivery_commit import synthetic as dc_synthetic
from delivery_commit import train as dc_train

# eta-regression
from eta_regression import cleaning as eta_cleaning
from eta_regression import features as eta_features
from eta_regression import synthetic as eta_synthetic
from eta_regression import train as eta_train

# intervention-optimization
from intervention_opt import evaluate as iv_evaluate
from intervention_opt import synthetic as iv_synthetic

# Pre-trained models shipped with the deployed Space (built by build_models.py)
# so visitors never wait for a cold-start train. Absent in the git checkout,
# where the app trains on demand and caches instead — see _commit_models.
BUNDLED = Path(__file__).parent / "models"
MODEL_CACHE = Path(__file__).parent / ".model_cache"

# Dark palette: the app is forced to dark mode (see app.py), so figures render
# transparent with light foreground and brightened accent colors that pop on a
# dark card.
INK = "#60a5fa"      # blue accent (oracle / interval)
BAD = "#f87171"      # raises risk
GOOD = "#4ade80"     # lowers risk / the winning policy
NEUTRAL = "#94a3b8"  # inert bars
FG = "#e5e7eb"       # text, ticks, labels
GRID = "#334155"     # spines, gridlines

_state: dict = {"commit": None, "eta": None, "shap": None, "iv_day": None}


def _style_dark(fig, axes) -> None:
    """Make a figure blend into the dark UI: transparent canvas, light text."""
    fig.patch.set_alpha(0.0)
    for ax in (axes if isinstance(axes, (list, tuple)) else [axes]):
        ax.set_facecolor("none")
        ax.title.set_color(FG)
        ax.xaxis.label.set_color(FG)
        ax.yaxis.label.set_color(FG)
        ax.tick_params(colors=FG)
        for spine in ax.spines.values():
            spine.set_color(GRID)


def _cache_dir():
    """A writable cache dir, or None. Some hosts (HF Spaces) mount the app dir
    read-only, in which case we skip caching and just retrain per cold start."""
    try:
        MODEL_CACHE.mkdir(exist_ok=True)
        probe = MODEL_CACHE / ".probe"
        probe.write_text("ok")
        probe.unlink()
        return MODEL_CACHE
    except OSError:
        return None


def _try_save(save_fn, models, cache, name) -> None:
    if cache is not None:
        try:
            save_fn(models, cache / name)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Model loading (train once, cache)
# ---------------------------------------------------------------------------
def _load_bundled(load_fn, name):
    """Load a pre-trained model shipped with the Space, or None if absent."""
    path = BUNDLED / name
    if path.exists():
        try:
            return load_fn(path)
        except Exception:  # noqa: BLE001 - version skew etc. -> fall back to training
            return None
    return None


def build_commit(save_to=None):
    """Train the commit-risk models (used at cold start and by build_models.py)."""
    raw = dc_synthetic.make_dataset(n=14000, seed=7, messy=True)
    clean, _ = dc_cleaning.clean(raw)
    models, _ = dc_train.train(clean, dc_train.TrainConfig(n_estimators=120, seed=7))
    if save_to is not None:
        dc_train.save(models, save_to)
    return models


def build_eta(save_to=None):
    """Train the ETA quantile models (used at cold start and by build_models.py)."""
    raw = eta_synthetic.make_dataset(n=14000, seed=7, messy=True)
    clean, _ = eta_cleaning.clean(raw)
    cfg = eta_train.TrainConfig(n_estimators=120, seed=7, quantiles=eta_train.PRODUCT_QUANTILES)
    models, _ = eta_train.train(clean, cfg)
    if save_to is not None:
        eta_train.save(models, save_to)
    return models


def _commit_models():
    if _state["commit"] is None:
        _state["commit"] = _load_bundled(dc_train.load, "commit")
    if _state["commit"] is None:
        cache = _cache_dir()
        try:
            _state["commit"] = dc_train.load(cache / "commit") if cache else None
            if _state["commit"] is None:
                raise FileNotFoundError
        except Exception:  # noqa: BLE001 - any load failure -> retrain from generator
            models = build_commit()
            _try_save(dc_train.save, models, cache, "commit")
            _state["commit"] = models
    return _state["commit"]


def _eta_models():
    if _state["eta"] is None:
        _state["eta"] = _load_bundled(eta_train.load, "eta")
    if _state["eta"] is None:
        cache = _cache_dir()
        try:
            _state["eta"] = eta_train.load(cache / "eta") if cache else None
            if _state["eta"] is None:
                raise FileNotFoundError
        except Exception:  # noqa: BLE001
            models = build_eta()
            _try_save(eta_train.save, models, cache, "eta")
            _state["eta"] = models
    return _state["eta"]


def _shap_explainer():
    if _state["shap"] is None:
        import shap

        _state["shap"] = shap.TreeExplainer(_commit_models().xgb)
    return _state["shap"]


def warmup() -> None:
    """Train/load everything up front so the first user click is fast."""
    _commit_models()
    _eta_models()
    _shap_explainer()


# ---------------------------------------------------------------------------
# Shared shipment row
# ---------------------------------------------------------------------------
def _raw_row(
    distance,
    service_level,
    origin_cong,
    dest_cong,
    weather,
    minutes_after_cutoff,
    peak,
    rural,
    dest_type,
) -> pd.DataFrame:
    """Build a one-row raw shipment covering every column both models need."""
    if service_level == "overnight":
        promised = 1
    elif service_level == "two_day":
        promised = 2
    else:
        promised = int(min(7, max(3, math.ceil(1 + distance / 600))))
    row = {
        dc_schema.ID_COL: "DEMO0001",
        dc_schema.DATE_COL: pd.Timestamp("2025-06-03"),
        "distance_miles": float(distance),
        "package_weight_lb": 6.0,
        "package_volume_cuft": 0.9,
        "declared_value_usd": 150.0,
        "minutes_after_cutoff": float(minutes_after_cutoff),
        "origin_hub_congestion": float(origin_cong),
        "dest_hub_congestion": float(dest_cong),
        "dest_weather_severity": int(weather),
        "route_stop_density": 4.5 if dest_type == "residential" else 2.5,
        "promised_transit_days": promised,
        "service_level": service_level,
        "origin_region": "midwest",
        "dest_region": "northeast" if distance > 400 else "midwest",
        "dest_type": dest_type,
        "day_of_week": 1,
        "is_peak_season": int(peak),
        "is_rural_dest": int(rural),
        "signature_required": 0,
    }
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Tab 1a: commit-miss risk + SHAP
# ---------------------------------------------------------------------------
# NB: these callbacks use named, typed parameters (not *args) on purpose — the
# demo also runs as an MCP server, and the generated tool schemas come from the
# signatures and docstrings below.
def score_commit(
    distance_miles: float,
    service_level: str,
    origin_hub_congestion: float,
    dest_hub_congestion: float,
    dest_weather_severity: int,
    minutes_after_cutoff: float,
    is_peak_season: bool,
    is_rural_dest: bool,
    dest_type: str,
):
    """Score one parcel's probability of missing its delivery commitment.

    Uses only induction-time information. Args: distance_miles (5-3000),
    service_level (overnight|two_day|ground), origin/dest hub congestion (0-1),
    dest_weather_severity (0 clear .. 3 severe), minutes_after_cutoff (pickup
    relative to facility cutoff; positive = late), is_peak_season,
    is_rural_dest, dest_type (residential|commercial). Returns a markdown
    verdict with the miss probability and a SHAP chart of the drivers.
    """
    models = _commit_models()
    raw = _raw_row(
        distance_miles, service_level, origin_hub_congestion, dest_hub_congestion,
        dest_weather_severity, minutes_after_cutoff, is_peak_season, is_rural_dest,
        dest_type,
    )
    clean, _ = dc_cleaning.clean(raw)
    X = dc_features.to_matrix(dc_features.engineer(clean))
    X = X.reindex(columns=models.feature_columns, fill_value=0.0)
    prob = float(models.xgb.predict_proba(X)[:, 1][0])

    contrib = pd.Series(_shap_explainer().shap_values(X)[0], index=X.columns)
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).head(8).index)[::-1]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = [BAD if v > 0 else GOOD for v in top.values]
    ax.barh(range(len(top)), top.values, color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([_pretty(c) for c in top.index], fontsize=9)
    ax.axvline(0, color=FG, lw=0.8, alpha=0.6)
    ax.set_xlabel("push toward missing  →   (log-odds contribution)   ←  push toward on-time")
    ax.set_title("Why this shipment scored the way it did")
    _style_dark(fig, ax)
    fig.tight_layout()

    verdict = "high risk" if prob >= 0.35 else "elevated" if prob >= 0.15 else "low risk"
    label = f"## Miss probability: {prob:.0%}  \n**{verdict}** — network base rate is ~11%."
    return label, fig


def _pretty(col: str) -> str:
    names = {
        "dest_weather_severity": "destination weather",
        "distance_miles": "lane distance",
        "total_hub_congestion": "total hub congestion",
        "origin_hub_congestion": "origin congestion",
        "dest_hub_congestion": "destination congestion",
        "is_peak_season": "peak season",
        "is_rural_dest": "rural destination",
        "late_pickup": "late pickup",
        "late_pickup_minutes": "minutes late at pickup",
        "minutes_after_cutoff": "pickup vs cutoff",
        "miles_per_promised_day": "miles per promised day",
        "route_stop_density": "route stop density",
    }
    if col in names:
        return names[col]
    if col.startswith("service_level_"):
        return col.replace("service_level_", "") + " service"
    if col.startswith("dest_type_"):
        return col.replace("dest_type_", "") + " destination"
    return col.replace("_", " ")


# ---------------------------------------------------------------------------
# Tab 1b: ETA quantiles + promise
# ---------------------------------------------------------------------------
def predict_eta(
    distance_miles: float,
    service_level: str,
    origin_hub_congestion: float,
    dest_hub_congestion: float,
    dest_weather_severity: int,
    minutes_after_cutoff: float,
    is_peak_season: bool,
    is_rural_dest: bool,
    dest_type: str,
):
    """Predict one parcel's transit time as a P10/P50/P90 quantile interval.

    Same induction-time inputs as score_commit. Returns a markdown summary with
    the median ETA, the honest P10-P90 range, and the promise (ceil of P90 —
    the transit days you can quote and keep ~9 times in 10), plus an interval
    chart.
    """
    models = _eta_models()
    raw = _raw_row(
        distance_miles, service_level, origin_hub_congestion, dest_hub_congestion,
        dest_weather_severity, minutes_after_cutoff, is_peak_season, is_rural_dest,
        dest_type,
    )
    clean, _ = eta_cleaning.clean(raw)
    X = eta_features.to_matrix(eta_features.engineer(clean))
    X = X.reindex(columns=models.feature_columns, fill_value=0.0)
    q = eta_train.predict_quantiles(models, X, alphas=(0.1, 0.5, 0.9)).iloc[0]
    p10, p50, p90 = float(q[0.1]), float(q[0.5]), float(q[0.9])
    promise = math.ceil(p90)

    fig, ax = plt.subplots(figsize=(7, 1.9))
    ax.hlines(0, p10, p90, color=INK, lw=6, alpha=0.5)
    ax.plot(p50, 0, "o", color=INK, ms=12)
    for x, lab in [(p10, f"P10\n{p10:.1f}d"), (p50, f"P50\n{p50:.1f}d"), (p90, f"P90\n{p90:.1f}d")]:
        ax.annotate(lab, (x, 0), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=9, color=FG)
    ax.axvline(promise, color=GOOD, ls="--", lw=1.5)
    ax.annotate(f"promise {promise}d", (promise, 0), textcoords="offset points",
                xytext=(0, -22), ha="center", color=GOOD, fontsize=9)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.set_xlabel("transit days")
    ax.set_title("Predicted transit-time interval")
    _style_dark(fig, ax)
    fig.tight_layout()

    label = (
        f"## Median ETA: {p50:.1f} days  \n"
        f"Honest range **{p10:.1f}–{p90:.1f} days**. Quote **{promise} days** "
        f"(ceil of P90) and you keep the promise ~9 times in 10."
    )
    return label, fig


# ---------------------------------------------------------------------------
# Tab 2: intervention economics
# ---------------------------------------------------------------------------
def _iv_day():
    if _state["iv_day"] is None:
        _state["iv_day"] = iv_synthetic.make_day(n=20000, seed=7)
    return _state["iv_day"]


_POLICY_LABELS = {
    "none": "Do nothing",
    "random": "Random spend",
    "top_k_risk": "Flag the riskiest (top-K)",
    "expected_value_greedy": "Expected-value greedy",
    "oracle": "Oracle (perfect scores)",
}


def run_budget(budget: float):
    """Allocate a daily intervention budget across 20,000 at-risk shipments.

    Compares five policies at the given budget (USD, 1000-20000): do nothing,
    random spend, flag-the-riskiest (top-K), expected-value greedy, and a
    perfect-scores oracle. Returns a markdown takeaway, a net-savings chart,
    and the full policy comparison table (spend, misses prevented, cost
    avoided, net savings, ROI, % of oracle).
    """
    import tempfile

    df = _iv_day()
    with tempfile.TemporaryDirectory() as tmp:
        comparison, _ = iv_evaluate.evaluate_all(df, budget=float(budget), out_dir=tmp)

    comp = comparison.set_index("policy")
    order = ["none", "random", "top_k_risk", "expected_value_greedy", "oracle"]
    comp = comp.reindex([p for p in order if p in comp.index])

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    net = comp["net_savings_usd"].to_numpy()
    colors = [
        GOOD if p == "expected_value_greedy" else NEUTRAL if p != "oracle" else INK
        for p in comp.index
    ]
    ax.bar(range(len(comp)), net, color=colors)
    ax.set_xticks(range(len(comp)))
    ax.set_xticklabels([_POLICY_LABELS[p] for p in comp.index], rotation=20, ha="right", fontsize=9)
    ax.axhline(0, color=FG, lw=0.8, alpha=0.6)
    ax.set_ylabel("net savings on the day ($)")
    ax.set_title(f"Every policy, same ${budget:,.0f} budget")
    for i, v in enumerate(net):
        va = "bottom" if v >= 0 else "top"
        ax.annotate(f"${v:,.0f}", (i, v), textcoords="offset points",
                    xytext=(0, 4 if v >= 0 else -6), ha="center", va=va, fontsize=8, color=FG)
    ax.margins(y=0.15)
    _style_dark(fig, ax)
    fig.tight_layout()

    tbl = comp.reset_index()
    tbl["policy"] = tbl["policy"].map(_POLICY_LABELS)
    tbl = tbl.rename(
        columns={
            "policy": "Policy",
            "spend_usd": "Spend $",
            "misses_prevented": "Misses prevented",
            "miss_cost_avoided_usd": "Cost avoided $",
            "net_savings_usd": "Net savings $",
            "roi": "ROI",
            "pct_of_oracle": "% of oracle",
        }
    )
    for c in ["Spend $", "Cost avoided $", "Net savings $"]:
        tbl[c] = tbl[c].map(lambda x: f"{x:,.0f}")

    greedy = comp.loc["expected_value_greedy"]
    topk = comp.loc["top_k_risk", "net_savings_usd"]
    takeaway = (
        f"### Expected-value greedy nets ${greedy['net_savings_usd']:,.0f} "
        f"({greedy['roi']:.1f}× ROI, {greedy['pct_of_oracle']:.0f}% of the oracle) — "
        f"**${greedy['net_savings_usd'] - topk:,.0f} more** than flagging the riskiest shipments, "
        f"because a 90% risk on a $5 parcel is worth less than a 40% risk on a contract pallet."
    )
    return takeaway, fig, tbl
