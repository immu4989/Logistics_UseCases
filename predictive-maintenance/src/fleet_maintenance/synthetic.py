"""Synthetic fleet telematics generator with a documented ground-truth wear process.

Why synthetic data? Fleet maintenance logs are proprietary, so this repo ships
with a generator whose *causal structure is known*. Each vehicle carries three
hidden wear states — brakes, engine, battery — that the model never sees. Wear
accumulates with mileage, load and age, jumps occasionally (potholes, part
defects), and is reset by maintenance. What the model *does* see is what a
telematics unit actually reports: temperatures, pressures, voltages, vibration
and fault codes, all of which are noisy functions of the hidden wear. That gap
between hidden state and noisy observation is the realism: a fleet model is
always inferring wear it cannot measure.

The ground-truth process (all constants in TRUE_PROCESS):

    wear increments per active day, scaled by a per-vehicle quality multiplier:
        brakes  += 2.2e-5 * miles * load_factor + 1.5e-3 * hard_braking_count
        engine  += 3.3e-5 * miles * load_factor * (1 + 0.05 * age_years)
        battery += 2.4e-3 * (1 + 0.12 * age_years) + cold-day stress
        shocks: with p=0.005/day one component jumps by U(0.08, 0.35)

    daily failure hazard per component (the key nonlinearity):
        h(wear) = 3.0e-5 * exp(6.0 * wear), capped at 0.5
        brakes: the exponent is scaled by (0.7 + 0.3 * driver aggression), so
        worn brakes under an aggressive driver fail much sooner than the same
        wear under a gentle one — observable through hard_braking x vibration
        battery: the exponent is scaled by (1 + 0.3 * cold), so a worn battery
        survives mild days and dies on the first cold snap — observable
        through battery_voltage x season
        Both are real interactions that a purely additive model cannot express.

    a failure causes 1-3 days of downtime and REPAIRS that component (wear -> ~0);
    scheduled service every ~15,000 miles resets all three components.

    observed sensors (noisy reads of the hidden state; deliberately not all
    linear, because real gauges are not — oil pressure holds and then sags,
    a battery holds voltage and then falls off a cliff):
        avg_engine_temp   = 88 + 0.35*(ambient-15) + 14*engine_wear + noise
        oil_pressure      = 43 - 13*engine_wear^1.7 + noise
        vibration_index   = 1 + 2.6*brake_wear^1.8 + 1.1*engine_wear + noise
        battery_voltage   = 12.7 - 0.55*battery_wear - 0.9*battery_wear^3
                            - cold dip + noise
        fault_code_count  ~ Poisson(0.02 + 0.35*total_wear^2)

cabin_temp_setting and radio_volume_avg carry zero true signal; they exist so
the explainability step has genuine negatives to rank below the real sensors.

The label is `failure_within_14d`: does any component fail in days t+1..t+14?
Fourteen extra days are simulated and discarded so no label is censored.

`make_fleet(..., messy=True)` injects the defects real telematics feeds have:
sensor-dropout days (and dropout probability *rises with wear* — a unit that
stops reporting often belongs to a vehicle in trouble, so missingness itself
is signal), duplicated (vehicle, day) rows, frozen-sensor stretches (a stuck
transmitter repeats the same value for days — every real feed does this), and
impossible negative mileage from odometer glitches.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ID_COL = "vehicle_id"
DATE_COL = "date"
LABEL_COL = "failure_within_14d"
HORIZON_DAYS = 14

# Bookkeeping columns: ground truth for evaluation, never model features.
EVENT_COL = "failure_event"
COMPONENT_COL = "failed_component"

COMPONENTS = ["brakes", "engine", "battery"]

# Continuous sensor channels a telematics unit streams. These are the columns
# that suffer dropout and frozen-transmitter stretches.
SENSOR_COLS = [
    "avg_engine_temp",
    "oil_pressure",
    "vibration_index",
    "battery_voltage",
    "cabin_temp_setting",
    "radio_volume_avg",
]

# Ground-truth constants, exposed so tests and the explain step can compare
# the model's SHAP ranking against reality.
TRUE_PROCESS = {
    "brake_wear_per_mile": 2.2e-5,
    "brake_wear_per_hard_brake": 1.5e-3,
    "engine_wear_per_mile": 3.3e-5,
    "engine_wear_age_gain": 0.05,       # per year of vehicle age
    "battery_wear_per_day": 2.4e-3,
    "battery_wear_age_gain": 0.12,
    "battery_cold_stress": 1.5e-3,      # extra wear on a freezing day
    "shock_prob": 0.005,
    "shock_range": (0.08, 0.35),
    "hazard_base": 3.0e-5,              # daily failure prob at zero wear
    "hazard_slope": 6.0,                # h(w) = base * exp(slope * w)
    "brake_hazard_aggression_mix": 0.3,  # slope *= (0.7 + 0.3 * driver aggression)
    "battery_hazard_cold_mix": 0.3,      # battery slope *= (1 + 0.3 * cold index)
    "service_interval_miles": 15_000,
    "temp_per_engine_wear": 14.0,       # deg C added to avg_engine_temp
    "oil_pressure_per_engine_wear": -13.0,  # applied to wear^1.7
    "oil_pressure_wear_exp": 1.7,
    "vibration_per_brake_wear": 2.6,        # applied to wear^1.8
    "vibration_wear_exp": 1.8,
    "vibration_per_engine_wear": 1.1,
    "voltage_per_battery_wear": -0.55,
    "voltage_cliff_per_wear_cubed": -0.9,   # batteries hold, then fall off a cliff
    "fault_rate_per_total_wear_sq": 0.35,
    "dropout_base_prob": 0.008,
    "dropout_per_total_wear": 0.075,    # missingness rises with hidden wear
}

# Sensors that carry real signal vs. channels planted as pure noise. The test
# suite asserts SHAP ranks the former above the latter.
SIGNAL_SENSORS = [
    "avg_engine_temp",
    "oil_pressure",
    "vibration_index",
    "battery_voltage",
    "fault_code_count",
]
NOISE_SENSORS = ["cabin_temp_setting", "radio_volume_avg"]


def make_fleet(
    n_vehicles: int = 600,
    n_days: int = 545,
    seed: int = 7,
    start_date: str = "2025-07-06",
    messy: bool = False,
) -> pd.DataFrame:
    """Simulate `n_vehicles` observed daily for `n_days`; return the vehicle-day panel."""
    rng = np.random.default_rng(seed)
    tp = TRUE_PROCESS
    sim_days = n_days + HORIZON_DAYS  # extra tail so no label is censored

    # --- per-vehicle characteristics ---------------------------------------
    age0 = rng.uniform(0.3, 9.0, n_vehicles)              # years at day 0
    base_miles = rng.gamma(9.0, 17.0, n_vehicles).clip(40, 320)
    load_factor = rng.uniform(0.45, 1.0, n_vehicles)      # typical payload share
    # Build quality / duty-cycle multiplier: the heterogeneity a mileage rule
    # cannot see. Some vehicles wear twice as fast per mile as others.
    wear_mult = rng.lognormal(0.0, 0.35, n_vehicles)
    brake_rate = rng.gamma(4.0, 0.25, n_vehicles)         # hard brakes per 100 mi
    svc_interval = tp["service_interval_miles"] * rng.uniform(0.9, 1.1, n_vehicles)

    # --- mutable state ------------------------------------------------------
    wear = {c: rng.uniform(0.0, 0.35, n_vehicles) for c in COMPONENTS}
    miles_since = rng.uniform(0, 8000, n_vehicles)
    days_since = (miles_since / base_miles).astype(int)
    downtime_left = np.zeros(n_vehicles, dtype=int)

    dates = pd.date_range(start_date, periods=sim_days, freq="D")
    doy = dates.dayofyear.to_numpy()
    weekend = dates.dayofweek.to_numpy() >= 5

    # --- per-day records (n_days x n_vehicles matrices) ---------------------
    rec = {
        k: np.zeros((sim_days, n_vehicles))
        for k in [
            "daily_miles", "engine_hours", "avg_engine_temp", "oil_pressure",
            "vibration_index", "battery_voltage", "hard_braking_count",
            "fault_code_count", "cabin_temp_setting", "radio_volume_avg",
            "vehicle_age_years", "days_since_maint", "miles_since_maint",
            "total_wear",
        ]
    }
    fail_event = np.zeros((sim_days, n_vehicles), dtype=int)
    fail_comp = np.full((sim_days, n_vehicles), "", dtype=object)

    for t in range(sim_days):
        active = downtime_left == 0
        ambient = 15 + 12 * np.sin(2 * np.pi * (doy[t] - 100) / 365) + rng.normal(0, 3, n_vehicles)
        age = age0 + t / 365.0

        miles = rng.gamma(8.0, base_miles / 8.0) * (0.55 if weekend[t] else 1.0)
        miles = np.where(active, miles.clip(0, 500), 0.0)
        hard_brakes = rng.poisson(brake_rate * miles / 100.0)

        # --- wear accumulation (active vehicles only) -----------------------
        cold = np.maximum(0.0, 2.0 - ambient) / 2.0  # 0 above 2C, 1 at freezing
        inc_b = wear_mult * (
            tp["brake_wear_per_mile"] * miles * load_factor
            + tp["brake_wear_per_hard_brake"] * hard_brakes
        )
        inc_e = wear_mult * (
            tp["engine_wear_per_mile"] * miles * load_factor
            * (1 + tp["engine_wear_age_gain"] * age)
        )
        inc_v = wear_mult * (
            tp["battery_wear_per_day"] * (1 + tp["battery_wear_age_gain"] * age)
            + tp["battery_cold_stress"] * cold
        )
        wear["brakes"] += np.where(active, inc_b, 0.0)
        wear["engine"] += np.where(active, inc_e, 0.0)
        wear["battery"] += np.where(active, inc_v, 0.0)

        # Stochastic shocks: pothole hits, part defects.
        shocked = active & (rng.random(n_vehicles) < tp["shock_prob"])
        if shocked.any():
            comp_idx = rng.integers(0, len(COMPONENTS), n_vehicles)
            jump = rng.uniform(*tp["shock_range"], n_vehicles)
            for ci, comp in enumerate(COMPONENTS):
                hit = shocked & (comp_idx == ci)
                wear[comp][hit] += jump[hit]

        total_wear = wear["brakes"] + wear["engine"] + wear["battery"]

        # --- observed sensors (read BEFORE any repair resets today) ---------
        rec["daily_miles"][t] = miles
        rec["engine_hours"][t] = np.where(
            active, miles / 42.0 + rng.normal(1.1, 0.3, n_vehicles).clip(0.2, 3), 0.0
        )
        rec["avg_engine_temp"][t] = (
            88 + 0.35 * (ambient - 15)
            + tp["temp_per_engine_wear"] * wear["engine"]
            + rng.normal(0, 2.0, n_vehicles)
        )
        rec["oil_pressure"][t] = (
            43
            + tp["oil_pressure_per_engine_wear"] * wear["engine"] ** tp["oil_pressure_wear_exp"]
            + rng.normal(0, 1.5, n_vehicles)
        )
        rec["vibration_index"][t] = (
            1.0
            + tp["vibration_per_brake_wear"] * wear["brakes"] ** tp["vibration_wear_exp"]
            + tp["vibration_per_engine_wear"] * wear["engine"]
            + 0.15 * load_factor
            + rng.normal(0, 0.28, n_vehicles)
        ).clip(0)
        rec["battery_voltage"][t] = (
            12.7 + tp["voltage_per_battery_wear"] * wear["battery"]
            + tp["voltage_cliff_per_wear_cubed"] * wear["battery"] ** 3
            - 0.25 * cold
            + rng.normal(0, 0.07, n_vehicles)
        )
        rec["hard_braking_count"][t] = hard_brakes
        rec["fault_code_count"][t] = rng.poisson(
            0.02 + tp["fault_rate_per_total_wear_sq"] * total_wear**2
        )
        # Planted noise: deliberately free of any vehicle-stable component. A
        # per-vehicle "preference" would let trees fingerprint individual
        # vehicles and inherit their fixed risk, which would make these
        # channels quietly informative and wreck the noise-control test.
        rec["cabin_temp_setting"][t] = rng.normal(21.5, 1.5, n_vehicles)
        rec["radio_volume_avg"][t] = rng.normal(16.0, 6.0, n_vehicles).clip(0, 40)
        rec["vehicle_age_years"][t] = age
        rec["days_since_maint"][t] = days_since
        rec["miles_since_maint"][t] = miles_since + miles
        rec["total_wear"][t] = total_wear

        # --- failures --------------------------------------------------------
        # Brakes carry a real interaction: the same wear level is far more
        # dangerous under an aggressive driver (brake_rate ~ 1.0 fleet-average).
        mix = tp["brake_hazard_aggression_mix"]
        cold_mix = tp["battery_hazard_cold_mix"]
        slope_by_comp = {
            "brakes": tp["hazard_slope"] * ((1 - mix) + mix * brake_rate),
            "engine": tp["hazard_slope"],
            # A worn battery survives mild days and dies on the first cold
            # snap: another interaction (voltage level x season) that additive
            # models cannot express.
            "battery": tp["hazard_slope"] * (1 + cold_mix * cold),
        }
        hazards = np.stack(
            [
                (tp["hazard_base"] * np.exp(slope_by_comp[c] * wear[c])).clip(0, 0.5)
                for c in COMPONENTS
            ]
        )  # (3, n_vehicles)
        draws = rng.random((len(COMPONENTS), n_vehicles))
        failed_any = active & (draws < hazards).any(axis=0)
        if failed_any.any():
            # If several components trip on the same day, blame the worst one.
            worst = np.argmax(np.where(draws < hazards, hazards, -1.0), axis=0)
            for vi in np.flatnonzero(failed_any):
                comp = COMPONENTS[worst[vi]]
                fail_event[t, vi] = 1
                fail_comp[t, vi] = comp
                wear[comp][vi] = rng.uniform(0.0, 0.05)  # emergency repair
            downtime_left[failed_any] = rng.integers(1, 4, failed_any.sum())

        # --- scheduled service (planned, no downtime charged) -----------------
        serviced = active & ~failed_any & (miles_since + miles >= svc_interval)
        if serviced.any():
            for comp in COMPONENTS:
                wear[comp][serviced] = rng.uniform(0.0, 0.08, serviced.sum())
            miles_since[serviced] = 0.0
            days_since[serviced] = 0

        keep = ~serviced
        miles_since = np.where(keep, miles_since + miles, miles_since)
        days_since = np.where(keep, days_since + 1, days_since)
        downtime_left = np.maximum(downtime_left - 1, 0)

    # --- label: any failure in the next 14 days ------------------------------
    fail_cum = np.vstack([np.zeros((1, n_vehicles), dtype=int), np.cumsum(fail_event, axis=0)])
    label = (fail_cum[HORIZON_DAYS + 1:] - fail_cum[1 : sim_days - HORIZON_DAYS + 1] > 0).astype(
        int
    )  # (n_days, n_vehicles): failures strictly in t+1..t+14

    # Downtime days: the unit is powered off, so sensor channels are absent.
    down = rec["daily_miles"] == 0.0
    for col in SENSOR_COLS:
        rec[col][down] = np.nan

    # --- flatten to a tidy panel (first n_days only; tail was label fuel) -----
    n_keep = n_days
    df = pd.DataFrame(
        {
            ID_COL: np.tile([f"VEH{seed:02d}{v:04d}" for v in range(n_vehicles)], n_keep),
            DATE_COL: np.repeat(dates[:n_keep], n_vehicles),
            **{k: rec[k][:n_keep].ravel().round(4) for k in rec if k != "total_wear"},
            EVENT_COL: fail_event[:n_keep].ravel(),
            COMPONENT_COL: fail_comp[:n_keep].ravel(),
            LABEL_COL: label.ravel(),
        }
    )

    if messy:
        df = _inject_mess(df, rec["total_wear"][:n_keep].ravel(), rng)

    return df.sort_values([ID_COL, DATE_COL], kind="stable").reset_index(drop=True)


def _inject_mess(df: pd.DataFrame, total_wear: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """Add the defects real telematics feeds have. cleaning.py must undo all of this."""
    df = df.copy()
    tp = TRUE_PROCESS

    # 1. Sensor dropout days: the unit fails to report. Crucially the dropout
    #    probability rises with hidden wear — a vehicle shaking itself apart is
    #    also shaking its transmitter loose — so missingness itself is signal
    #    and cleaning.py's __was_missing flags are load-bearing, not cosmetic.
    p_drop = tp["dropout_base_prob"] + tp["dropout_per_total_wear"] * total_wear.clip(0, 2.5)
    dropped = rng.random(len(df)) < p_drop
    df.loc[dropped, SENSOR_COLS] = np.nan

    # 2. Duplicated (vehicle, day) rows: the feed replays a day after an outage.
    dupes = df.sample(frac=0.008, random_state=int(rng.integers(1e6)))
    df = pd.concat([df, dupes], ignore_index=True)

    # 3. Frozen-sensor stretches: a stuck transmitter repeats its last value
    #    verbatim for days. Real feeds do this constantly and a naive pipeline
    #    happily models the frozen value as a calm, healthy vehicle.
    vehicles = df[ID_COL].unique()
    freeze_sensors = ["avg_engine_temp", "vibration_index", "oil_pressure", "battery_voltage"]
    for _ in range(max(4, len(vehicles) // 40)):
        veh = vehicles[rng.integers(len(vehicles))]
        col = freeze_sensors[rng.integers(len(freeze_sensors))]
        idx = df.index[df[ID_COL] == veh].to_numpy()
        if len(idx) < 30:
            continue
        start = rng.integers(0, len(idx) - 20)
        run = idx[start : start + int(rng.integers(8, 16))]
        frozen_val = df.loc[run[0], col]
        if np.isfinite(frozen_val):
            df.loc[run, col] = frozen_val

    # 4. Impossible negative mileage: odometer rollover glitches in the feed.
    idx = df.sample(frac=0.002, random_state=int(rng.integers(1e6))).index
    df.loc[idx, "daily_miles"] = -df.loc[idx, "daily_miles"].abs() - 1.0

    return df.sample(frac=1, random_state=int(rng.integers(1e6))).reset_index(drop=True)
