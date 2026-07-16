"""Incident cards: turn each alarm into the message you'd paste in the ops channel.

An alarm without a narrative gets ignored; a rate without a cost gets deferred.
Each card states what the lane is doing versus what it should be doing, and
converts the gap into excess misses per week at the lane's current volume,
because "this drift costs ~40 extra misses a week" starts a staffing
conversation that "z-score exceeded threshold" never will.

On synthetic data we also print the ground truth (days since the injected
onset, or a false-alarm confession on a clean lane). Keep that habit when you
adapt this: every retro on a real alarm should end by labeling it, so you
accumulate your own ground truth for tuning k and h.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .detect import DetectionResult

TRAILING_DAYS = 14
EPISODE_GAP_DAYS = 14  # alarms on the same lane closer than this are one incident


def _episodes(annotated: pd.DataFrame) -> pd.DataFrame:
    """Collapse re-alarms into incidents.

    A still-broken lane re-crosses h every few days after each reset. Ops does
    not want 15 pages about one broken lane; they want one incident that notes
    it is still firing. First alarm of each episode carries the card, and we
    count the re-alarms behind it.
    """
    rows = []
    for lane, grp in annotated.sort_values("day").groupby("lane"):
        grp = grp.reset_index(drop=True)
        episode_start = 0
        for i in range(1, len(grp) + 1):
            if i == len(grp) or grp.loc[i, "day"] - grp.loc[i - 1, "day"] > EPISODE_GAP_DAYS:
                first = grp.loc[episode_start].to_dict()
                first["re_alarms"] = i - episode_start - 1
                rows.append(first)
                episode_start = i
    return pd.DataFrame(rows).sort_values("day").reset_index(drop=True)


def _card(res: DetectionResult, alarm: dict) -> tuple[str, float]:
    """One incident card. Returns (markdown, excess misses/week) for ranking."""
    i = res.lanes.index(alarm["lane"])
    d = int(alarm["day"])
    # Window = the CUSUM's own accumulation run (walk back while the statistic
    # was nonzero), capped at TRAILING_DAYS. A trunk lane that alarms the day
    # a step lands gets a 1-day window over thousands of shipments; a thin
    # lane that took three weeks to accumulate gets those three weeks. A fixed
    # window would dilute a fresh break with pre-onset days and understate it.
    lo = d
    while lo - 1 >= res.config.baseline_days and res.cusum[i, lo - 1] > 0:
        lo -= 1
    lo = max(lo, d - TRAILING_DAYS + 1)
    vol = np.nansum(res.volume[i, lo : d + 1])
    misses = np.nansum(res.obs_rate[i, lo : d + 1] * res.volume[i, lo : d + 1])
    obs = misses / max(vol, 1.0)
    exp = float(res.expected.loc[alarm["lane"]])
    daily_vol = vol / max(np.isfinite(res.obs_rate[i, lo : d + 1]).sum(), 1)
    excess_wk = (obs - exp) * daily_vol * 7

    lines = [
        f"### {alarm['lane']}: alarm {alarm['date']:%Y-%m-%d}",
        "",
        f"- Miss rate over the {d - lo + 1} day(s) the statistic accumulated: **{obs:.1%}** "
        f"vs expected **{exp:.1%}** (CUSUM {alarm['cusum']:.1f} at h={res.config.h:g})",
        f"- At ~{daily_vol:,.0f} shipments/day, this drift costs **~{max(excess_wk, 0):,.0f} "
        "extra misses/week** if left alone",
    ]
    if alarm.get("re_alarms", 0):
        lines.append(
            f"- Still firing: re-alarmed {int(alarm['re_alarms'])} more time(s) since, "
            "not yet fixed"
        )
    status = alarm.get("status", "")
    if status == "detection":
        lines.append(
            f"- Ground truth: injected {alarm['anomaly_type']} drift, caught "
            f"**{int(alarm['days_since_onset'])} days** after onset"
        )
    elif status == "spike_window":
        lines.append(
            "- Ground truth: one-day spike (storm). Self-recovering; verify and close, "
            "no intervention expected"
        )
    elif status == "false_alarm":
        lines.append("- Ground truth: FALSE ALARM — this lane is clean. Counted against "
                     "the alarm budget.")
    return "\n".join(lines), float(excess_wk)


def write_cards(res: DetectionResult, annotated: pd.DataFrame, out_dir: str | Path) -> list[str]:
    """Write alarms.md (chronological); return cards sorted by weekly cost, worst first."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    episodes = _episodes(annotated)
    built = [_card(res, ep) for ep in episodes.to_dict("records")]

    header = (
        "# Lane alarms — incident cards\n\n"
        f"Detector: EB shrinkage + CUSUM (k={res.config.k:g}, h={res.config.h:g}, "
        f"baseline={res.config.baseline_days} days). "
        f"{len(built)} incidents ({len(annotated)} raw alarms; re-alarms within "
        f"{EPISODE_GAP_DAYS} days collapse into their incident).\n"
    )
    (out / "alarms.md").write_text(header + "\n\n".join(c for c, _ in built) + "\n")

    return [c for c, _ in sorted(built, key=lambda t: -t[1])]
