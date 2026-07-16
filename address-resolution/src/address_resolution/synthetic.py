"""Synthetic delivery-point database + shipping-label generator with a recorded
corruption ladder.

Two tables come out of here:

1. The **canonical delivery-point database** (~8,000 rows): the addresses the
   carrier believes exist. Street number, a street name drawn from a generated
   lexicon, street type, an optional unit for multi-unit buildings, one of 25
   zips laid out on a 5x5 grid, and lat/lon jittered around the zip centroid.
2. **Shipping labels** (~20,000 rows): what customers actually typed. Each
   label starts from a real delivery point and then walks a documented
   corruption ladder; the applied rungs are recorded on the label. That record
   is simultaneously the training ground truth and the error-taxonomy key the
   evaluation uses — you can ask "which corruption types survive all the way
   to a wrong door?" without any manual annotation.

Unlike the other use cases in this repo there is no ``messy=True`` switch:
the mess IS the dataset. The whole problem is the gap between the typed label
and the canonical record, so corruption is not an optional garnish here.

The corruption ladder (each label draws 0-3 rungs):

    typo_street_name     keyboard-adjacent substitution ("Marwick" -> "Marwock")
    street_type_variant  abbreviation flips ("St" <-> "Street" <-> "St.")
    unit_dropped         the apartment number simply isn't on the label
    unit_format          "Apt 4B" -> "#4B" / "Unit 4-B" / "Apt. 4B"
    digits_transposed    adjacent digits swapped in the street number
    wrong_zip            a grid-adjacent zip (customer moved, stale autofill)
    extra_tokens         "c/o Dana Harmon", business names, front or back
    casing_whitespace    ALL CAPS / lowercase / doubled spaces

Plus ~8% of labels generated from NOWHERE — new construction the database
hasn't heard of, a street name that doesn't exist, or a unit the building
doesn't have. These are the true no-matches: the resolver's job is to send
them to a human, not to force them onto the nearest plausible door.

One deliberate concession to measurability: corruptions are resampled when
they would land exactly on a *different* real address (e.g. a transposed
street number that happens to exist three doors down). Such labels are
unresolvable from the text alone — both addresses are real — and no matcher
can be graded on them. Real systems catch that class with proof-of-delivery
feedback loops, not with string features; see the README.

Inspired by Amazon's delivery-point resolution work, which mapped ~2.8 million
apartment addresses onto the right physical buildings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- city geometry
ZIP_BASE = 46200
GRID = 5  # 5x5 grid of zips
N_ZIPS = GRID * GRID


def zip_codes() -> list[int]:
    return [ZIP_BASE + i for i in range(N_ZIPS)]


def zip_coord(z: int) -> tuple[int, int]:
    i = z - ZIP_BASE
    return divmod(i, GRID)


def adjacent_zips(z: int) -> list[int]:
    """4-neighborhood on the zip grid. A production system would load this
    from zip-boundary geometry; the matcher only needs "which zips touch"."""
    r, c = zip_coord(z)
    out = []
    for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= rr < GRID and 0 <= cc < GRID:
            out.append(ZIP_BASE + rr * GRID + cc)
    return out


def zip_distance(a: int, b: int) -> int:
    ra, ca = zip_coord(a)
    rb, cb = zip_coord(b)
    return abs(ra - rb) + abs(ca - cb)


# --------------------------------------------------------------------------- lexicons
_PREFIXES = [
    "Ash", "Bel", "Cold", "Dun", "Elm", "Fair", "Glen", "Hol", "Iron", "Kes",
    "Lin", "Mar", "Nor", "Oak", "Pem", "Quar", "Row", "Stone", "Thorn", "Wex",
]
_SUFFIXES = ["brook", "bury", "crest", "dale", "field", "ford", "haven", "hurst", "mont", "wick"]

STREET_TYPES = ["St", "Ave", "Blvd", "Ln", "Dr"]
TYPE_VARIANTS = {
    "St": ["Street", "St."],
    "Ave": ["Avenue", "Ave."],
    "Blvd": ["Boulevard", "Blvd."],
    "Ln": ["Lane", "Ln."],
    "Dr": ["Drive", "Dr."],
}

_UNIT_CHOICES = [f"{f}{c}" for f in "1234" for c in "ABCDEF"]

_FIRST_NAMES = ["Dana", "Luis", "Priya", "Marcus", "Elena", "Kofi", "Ingrid", "Tomas"]
_LAST_NAMES = ["Harmon", "Okafor", "Reyes", "Lindqvist", "Patel", "Novak", "Byrne", "Sato"]
_BIZ_WORDS = ["Cardinal", "Beacon", "Summit", "Harbor", "Pioneer", "Keystone"]
_BIZ_KINDS = ["Logistics", "Freight", "Supply", "Systems"]

# QWERTY adjacency for keyboard-slip typos.
_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
_ADJ: dict[str, list[str]] = {}
for _r, _row in enumerate(_ROWS):
    for _i, _ch in enumerate(_row):
        _n = []
        if _i > 0:
            _n.append(_row[_i - 1])
        if _i < len(_row) - 1:
            _n.append(_row[_i + 1])
        for _rr in (_r - 1, _r + 1):
            if 0 <= _rr < len(_ROWS) and _i < len(_ROWS[_rr]):
                _n.append(_ROWS[_rr][_i])
        _ADJ[_ch] = _n

CORRUPTION_ORDER = [
    "typo_street_name",
    "street_type_variant",
    "unit_dropped",
    "unit_format",
    "digits_transposed",
    "wrong_zip",
    "extra_tokens",
    "casing_whitespace",
]
_CORRUPTION_WEIGHTS = [0.20, 0.13, 0.13, 0.10, 0.13, 0.09, 0.11, 0.11]
ORPHAN_MODES = ["orphan_novel_street", "orphan_unknown_number", "orphan_unknown_unit"]


# --------------------------------------------------------------------------- delivery points
def make_city(
    seed: int = 7,
    streets_per_zip: int = 10,
    buildings_per_street: int = 20,
) -> pd.DataFrame:
    """The canonical delivery-point database (~8,000 rows at the defaults).

    Street names repeat across zips on purpose — "Marwick Ave" existing in
    three zips is exactly the ambiguity a real resolver fights — and ~18% of
    buildings are multi-unit, which is where confident wrong doors live.
    """
    rng = np.random.default_rng(seed)
    combos = [(p, s) for p in _PREFIXES for s in _SUFFIXES]
    order = rng.permutation(len(combos))
    city_names = []
    for i in order[:60]:
        p, s = combos[i]
        # A slice of two-word names so the token features have real work to do.
        city_names.append(f"{p} {s.capitalize()}" if rng.random() < 0.15 else p + s)

    rows = []
    for z in zip_codes():
        r, c = zip_coord(z)
        lat0, lon0 = 39.60 + 0.030 * r, -86.40 + 0.030 * c
        streets = rng.choice(len(city_names), streets_per_zip, replace=False)
        for si in streets:
            name = city_names[si]
            stype = str(rng.choice(STREET_TYPES))
            numbers = rng.choice(np.arange(100, 9900), buildings_per_street, replace=False)
            for num in numbers:
                lat = lat0 + rng.uniform(-0.012, 0.012)
                lon = lon0 + rng.uniform(-0.012, 0.012)
                if rng.random() < 0.18:
                    k = int(rng.integers(2, 7))
                    units = rng.choice(len(_UNIT_CHOICES), k, replace=False)
                    for u in units:
                        rows.append((int(num), name, stype, _UNIT_CHOICES[u], z, lat, lon))
                else:
                    rows.append((int(num), name, stype, "", z, lat, lon))

    points = pd.DataFrame(
        rows, columns=["street_number", "street_name", "street_type", "unit", "zip", "lat", "lon"]
    )
    points.insert(0, "point_id", [f"DP{i:06d}" for i in range(len(points))])
    return points


# --------------------------------------------------------------------------- corruption ladder
def _apply_typo(state: dict, rng, city_names_lower: set[str]) -> bool:
    name = state["name"]
    for _ in range(6):
        pos = int(rng.integers(0, len(name)))
        ch = name[pos].lower()
        if ch not in _ADJ:
            continue
        new = str(rng.choice(_ADJ[ch]))
        cand = name[:pos] + (new.upper() if name[pos].isupper() else new) + name[pos + 1 :]
        # Never typo INTO another real street name: that label would point at a
        # different valid address and no text matcher could be graded on it.
        if cand.lower() not in city_names_lower:
            state["name"] = cand
            return True
    return False


def _apply_transpose(state: dict, rng, taken_numbers: set[int]) -> bool:
    num = state["number"]
    positions = [i for i in range(len(num) - 1) if num[i] != num[i + 1]]
    for i in rng.permutation(positions) if positions else []:
        cand = num[:i] + num[i + 1] + num[i] + num[i + 2 :]
        if int(cand) not in taken_numbers:
            state["number"] = cand
            return True
    return False


def _apply_wrong_zip(state: dict, rng, point_keys: set) -> bool:
    neighbors = adjacent_zips(state["zip"])
    for nz in rng.permutation(neighbors):
        # Don't relocate onto a zip where this exact street+number also exists.
        if (int(nz), state["name"].lower(), int(state["number"])) not in point_keys:
            state["zip"] = int(nz)
            return True
    return False


def _render(state: dict, rng) -> str:
    unit_txt = ""
    if state["unit"]:
        u = state["unit"]
        style = state["unit_style"]
        if style == "APT":
            unit_txt = f"Apt {u}"
        elif style == "HASH":
            unit_txt = f"#{u}"
        elif style == "UNITDASH":
            unit_txt = f"Unit {u[0]}-{u[1:]}" if len(u) > 1 else f"Unit {u}"
        else:
            unit_txt = f"Apt. {u}"
    parts = [state["prefix"], state["number"], state["name"], state["stype_txt"], unit_txt,
             state["suffix"]]
    text = " ".join(p for p in parts if p)
    case = state["case"]
    if case == "lower":
        text = text.lower()
    elif case == "upper":
        text = text.upper()
    elif case == "spaces":
        toks = text.split(" ")
        gap = int(rng.integers(1, len(toks)))
        text = " ".join(toks[:gap]) + "  " + " ".join(toks[gap:])
    return text


# --------------------------------------------------------------------------- labels
def make_labels(
    points: pd.DataFrame,
    n: int = 20_000,
    orphan_frac: float = 0.08,
    seed: int = 7,
) -> pd.DataFrame:
    """Shipping labels generated from the delivery points, plus true orphans.

    Columns: label_id, address_text, zip (its own field on a label, like the
    real thing), true_point_id ("" for orphans), corruptions ("|"-joined
    ladder record, "" when the customer typed it perfectly).
    """
    rng = np.random.default_rng(seed + 1)

    city_names_lower = {n_.lower() for n_ in points["street_name"].unique()}
    combos = {p + s for p in _PREFIXES for s in _SUFFIXES}
    novel_pool = sorted(nm for nm in combos if nm.lower() not in city_names_lower)

    # Lookup structures for collision-free corruption (see module docstring).
    numbers_by_street: dict[tuple[int, str], set[int]] = {}
    point_keys: set[tuple[int, str, int]] = set()
    units_by_building: dict[tuple[int, str, str, int], set[str]] = {}
    for row in points.itertuples(index=False):
        key = (int(row.zip), row.street_name.lower())
        numbers_by_street.setdefault(key, set()).add(int(row.street_number))
        point_keys.add((int(row.zip), row.street_name.lower(), int(row.street_number)))
        bkey = (int(row.zip), row.street_name.lower(), row.street_type, int(row.street_number))
        units_by_building.setdefault(bkey, set()).add(row.unit)

    def _numbers_nearby(z: int, name: str) -> set[int]:
        out: set[int] = set()
        for zz in [z, *adjacent_zips(z)]:
            out |= numbers_by_street.get((zz, name.lower()), set())
        return out

    n_orphan = int(round(n * orphan_frac))
    n_match = n - n_orphan
    src = rng.integers(0, len(points), n_match)
    pt = points.iloc[src].reset_index(drop=True)

    records: list[tuple] = []
    for i in range(n_match):
        row = pt.iloc[i]
        state = {
            "number": str(int(row["street_number"])),
            "name": row["street_name"],
            "stype_txt": row["street_type"],
            "unit": row["unit"],
            "unit_style": "APT",
            "zip": int(row["zip"]),
            "prefix": "",
            "suffix": "",
            "case": None,
        }
        k = int(rng.choice(4, p=[0.15, 0.40, 0.30, 0.15]))
        chosen = rng.choice(len(CORRUPTION_ORDER), size=k, replace=False, p=_CORRUPTION_WEIGHTS)
        applied = []
        for ci in sorted(chosen):
            c = CORRUPTION_ORDER[ci]
            ok = False
            if c == "typo_street_name":
                ok = _apply_typo(state, rng, city_names_lower)
            elif c == "street_type_variant":
                state["stype_txt"] = str(rng.choice(TYPE_VARIANTS[row["street_type"]]))
                ok = True
            elif c == "unit_dropped" and state["unit"]:
                state["unit"] = ""
                ok = True
            elif c == "unit_format" and state["unit"]:
                state["unit_style"] = str(rng.choice(["HASH", "UNITDASH", "APTDOT"]))
                ok = True
            elif c == "digits_transposed":
                ok = _apply_transpose(state, rng, _numbers_nearby(state["zip"], row["street_name"]))
            elif c == "wrong_zip":
                ok = _apply_wrong_zip(state, rng, point_keys)
            elif c == "extra_tokens":
                if rng.random() < 0.5:
                    state["suffix"] = (
                        f"c/o {rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}"
                    )
                else:
                    state["prefix"] = f"{rng.choice(_BIZ_WORDS)} {rng.choice(_BIZ_KINDS)} LLC"
                ok = True
            elif c == "casing_whitespace":
                state["case"] = str(rng.choice(["lower", "upper", "spaces"]))
                ok = True
            if ok:
                applied.append(c)
        records.append((_render(state, rng), state["zip"], row["point_id"], "|".join(applied)))

    # ---- true no-matches: addresses the database has never heard of ---------
    multiunit = points[points["unit"] != ""]
    buildings = multiunit.drop_duplicates(
        subset=["zip", "street_name", "street_type", "street_number"]
    ).reset_index(drop=True)
    streets = points.drop_duplicates(subset=["zip", "street_name", "street_type"]).reset_index(
        drop=True
    )
    modes = rng.choice(ORPHAN_MODES, n_orphan, p=[0.5, 0.3, 0.2])
    for mode in modes:
        if mode == "orphan_novel_street":
            name = novel_pool[int(rng.integers(0, len(novel_pool)))]
            stype = str(rng.choice(STREET_TYPES))
            z = int(rng.choice(zip_codes()))
            number = str(int(rng.integers(100, 9900)))
        elif mode == "orphan_unknown_number":
            srow = streets.iloc[int(rng.integers(0, len(streets)))]
            name, stype, z = srow["street_name"], srow["street_type"], int(srow["zip"])
            taken = _numbers_nearby(z, name)
            taken_keys = {"".join(sorted(str(t))) for t in taken}
            while True:
                cand = int(rng.integers(100, 9900))
                # Also dodge digit-multiset collisions: an unknown number that is
                # a transposition of a real one reads as a recoverable typo.
                if cand not in taken and "".join(sorted(str(cand))) not in taken_keys:
                    number = str(cand)
                    break
        else:  # orphan_unknown_unit — new construction inside a real building
            brow = buildings.iloc[int(rng.integers(0, len(buildings)))]
            name, stype, z = brow["street_name"], brow["street_type"], int(brow["zip"])
            number = str(int(brow["street_number"]))
        state = {
            "number": number, "name": name, "stype_txt": stype, "unit": "",
            "unit_style": "APT", "zip": z, "prefix": "", "suffix": "", "case": None,
        }
        if mode == "orphan_unknown_unit":
            bkey = (z, name.lower(), stype, int(number))
            have = units_by_building.get(bkey, set())
            free = [u for u in _UNIT_CHOICES if u not in have]
            state["unit"] = free[int(rng.integers(0, len(free)))]
        records.append((_render(state, rng), z, "", mode))

    labels = pd.DataFrame(records, columns=["address_text", "zip", "true_point_id", "corruptions"])
    labels = labels.sample(frac=1, random_state=seed + 2).reset_index(drop=True)
    labels.insert(0, "label_id", [f"LB{i:07d}" for i in range(len(labels))])
    return labels


def make_dataset(seed: int = 7, n_labels: int = 20_000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper: (delivery points, shipping labels), both seeded."""
    points = make_city(seed=seed)
    labels = make_labels(points, n=n_labels, seed=seed)
    return points, labels
