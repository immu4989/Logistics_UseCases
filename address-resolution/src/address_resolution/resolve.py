"""The staged matcher: normalize -> block -> score -> accept-or-review.

The stage order mirrors how production address resolvers are built, and each
stage exists for an operational reason:

**Normalization** eats the cheap variance first. Casing, whitespace,
"St" <-> "Street", "Apt 4B" <-> "#4B" — none of that deserves a model. A
surprising share of labels become exact matches after this stage alone, which
is why exact-match-after-normalization is baseline #1 below.

**Blocking** exists because you cannot afford to score everything: 20k labels
x 8k delivery points is 160M pair evaluations per batch, and a national-scale
database makes it 10^12. We fetch candidates by zip (plus adjacent zips,
because zips are the field customers get *almost* right) and three redundant
keys: street-name first character, a crude phonetic key, and the sorted-digit
multiset of the street number (invariant to transposition). The price of
blocking is a recall risk — a true match that falls out of every block can
never be recovered downstream, no matter how good the scorer is — so blocking
recall is measured and printed, not assumed. It stays high here because the
corruption ladder never garbles the number, name and zip simultaneously; real
mail sometimes does, and then you widen the blocks and pay the compute.

**Scoring** turns each (label, candidate) pair into a handful of features a
reviewer would recognize — name similarity, number agreement, unit agreement,
zip distance — and a logistic regression maps them to a match probability.

**Decision**: accept the best candidate if its probability clears the
threshold, otherwise route the label to the human-review queue. The reject
option is the product. Both baselines lack it in opposite directions: exact
match rejects everything imperfect, fuzzy top-1 rejects nothing and delivers
its mistakes.
"""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import pandas as pd

from . import synthetic

FEATURES = [
    "name_trigram_jaccard",
    "token_set_overlap",
    "number_exact",
    "number_transposed",
    "unit_agree",
    "unit_conflict",
    "unit_unresolvable",
    "street_type_match",
    "zip_distance",
]

_TYPE_CANON = {}
for _short, _variants in synthetic.TYPE_VARIANTS.items():
    _TYPE_CANON[_short.upper()] = _short.upper()
    for _v in _variants:
        _TYPE_CANON[_v.upper()] = _short.upper()

_UNIT_MARKERS = {"APT", "APT.", "UNIT", "UNIT."}
_VOWELS = set("AEIOU")


# --------------------------------------------------------------------------- normalization
def _clean_unit(tok: str) -> str:
    return "".join(ch for ch in tok if ch.isalnum()).upper()


def normalize_text(text: str) -> tuple[str, str, str, str]:
    """Parse one free-text address line into (number, name, type, unit).

    Anchors on the first all-digit token as the street number, which is how
    lightweight production parsers work too: business names and "c/o" riders
    contain no bare digit tokens, so they fall away on either side.
    """
    t = re.sub(r"\s+", " ", text.upper()).strip()
    tokens = t.split(" ")
    num_i = next((i for i, tok in enumerate(tokens) if tok.isdigit()), None)
    if num_i is None:
        return "", "", "", ""
    number = tokens[num_i]

    name_tokens: list[str] = []
    stype = ""
    i = num_i + 1
    while i < len(tokens):
        canon = _TYPE_CANON.get(tokens[i])
        if canon is not None:
            stype = canon
            i += 1
            break
        name_tokens.append(tokens[i])
        i += 1

    unit = ""
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("#") and len(tok) > 1:
            unit = _clean_unit(tok[1:])
            break
        if tok in _UNIT_MARKERS and i + 1 < len(tokens):
            unit = _clean_unit(tokens[i + 1])
            break
        i += 1  # anything else after the street type is a rider; ignore it
    return number, " ".join(name_tokens), stype, unit


def normalize_labels(labels: pd.DataFrame) -> pd.DataFrame:
    texts = labels["address_text"].tolist()
    parsed = [normalize_text(t) for t in texts]
    out = pd.DataFrame(parsed, columns=["number", "name", "stype", "unit"])
    out["zip"] = labels["zip"].astype(int).to_numpy()
    # The raw upper-cased line survives alongside the parse: one feature
    # compares what the customer *typed* against the record, so a parser
    # mistake degrades a score instead of silently corrupting a field.
    out["raw"] = [re.sub(r"\s+", " ", t.upper()).strip() for t in texts]
    return out


def normalize_points(points: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "number": points["street_number"].astype(int).astype(str),
            "name": points["street_name"].astype(str).str.upper(),
            "stype": points["street_type"].astype(str).str.upper().map(_TYPE_CANON),
            "unit": [_clean_unit(u) for u in points["unit"].astype(str).tolist()],
            "zip": points["zip"].astype(int).to_numpy(),
        }
    )
    return out


# --------------------------------------------------------------------------- similarity keys
def _trigrams(s: str) -> frozenset:
    p = f"##{s}##"
    return frozenset(p[i : i + 3] for i in range(len(p) - 2))


def _jaccard(a: frozenset | set, b: frozenset | set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def _phonetic(name: str) -> str:
    letters = [c for c in name.upper() if c.isalpha()]
    if not letters:
        return ""
    key = [letters[0]]
    for c in letters[1:]:
        if c not in _VOWELS and c != key[-1]:
            key.append(c)
    return "".join(key[:4])


def _digit_sort(number: str) -> str:
    return "".join(sorted(number))


def _is_adjacent_transposition(a: str, b: str) -> bool:
    """True when exactly one adjacent digit pair is swapped ("5742" vs "5472").

    Deliberately stricter than sharing a digit multiset: "5427" shares 5742's
    digits but no typist produces it with one slip, and treating it as a typo
    match once delivered a parcel to the wrong end of the street in an early
    version of this pipeline.
    """
    if len(a) != len(b) or a == b:
        return False
    diff = [i for i, (x, z) in enumerate(zip(a, b)) if x != z]
    return (
        len(diff) == 2
        and diff[1] == diff[0] + 1
        and a[diff[0]] == b[diff[1]]
        and a[diff[1]] == b[diff[0]]
    )


# --------------------------------------------------------------------------- blocking
class Blocker:
    """Candidate retrieval by (zip or adjacent zip) x three redundant keys.

    The keys are deliberately overlapping so that any single corruption leaves
    at least one intact: a street-name typo breaks the first-char/phonetic
    keys but not the number key; a transposed number breaks nothing because
    the digit-multiset key is transposition-invariant; a wrong-but-adjacent
    zip is covered by searching neighbors. Recall only dies when several of
    those fail at once.
    """

    def __init__(self, pnorm: pd.DataFrame):
        self._first: dict[tuple[int, str], list[int]] = defaultdict(list)
        self._phon: dict[tuple[int, str], list[int]] = defaultdict(list)
        self._dsort: dict[tuple[int, str], list[int]] = defaultdict(list)
        for idx, (z, name, num) in enumerate(
            zip(pnorm["zip"].tolist(), pnorm["name"].tolist(), pnorm["number"].tolist())
        ):
            self._first[(z, name[:1])].append(idx)
            self._phon[(z, _phonetic(name))].append(idx)
            self._dsort[(z, _digit_sort(num))].append(idx)

    def candidates(self, z: int, name: str, number: str) -> list[int]:
        f, ph, ds = name[:1], _phonetic(name), _digit_sort(number)
        out: set[int] = set()
        for zz in (z, *synthetic.adjacent_zips(z)):
            out.update(self._first.get((zz, f), ()))
            out.update(self._phon.get((zz, ph), ()))
            out.update(self._dsort.get((zz, ds), ()))
        return sorted(out)


def build_pairs(lnorm: pd.DataFrame, blocker: Blocker) -> tuple[np.ndarray, np.ndarray]:
    """All (label, candidate) pairs that survive blocking, label-major order."""
    li: list[int] = []
    pi: list[int] = []
    zs = lnorm["zip"].tolist()
    names = lnorm["name"].tolist()
    nums = lnorm["number"].tolist()
    for a in range(len(lnorm)):
        cands = blocker.candidates(zs[a], names[a], nums[a])
        li.extend([a] * len(cands))
        pi.extend(cands)
    return np.asarray(li, dtype=np.int64), np.asarray(pi, dtype=np.int64)


# --------------------------------------------------------------------------- features
def featurize(
    lnorm: pd.DataFrame, pnorm: pd.DataFrame, li: np.ndarray, pi: np.ndarray
) -> np.ndarray:
    """Per-pair features, each one a signal a human reviewer would also use.

    ``unit_unresolvable`` deserves a note: it fires when the label has no unit
    but the candidate is a unit inside a multi-unit building. Every unit in
    that building then shows *identical* features, so no scorer can pick the
    right door — the only correct output is "ask a human". The logistic
    regression learns exactly that (the positive is 1-of-k among identical
    rows), which is how unit-less apartment labels end up in the review queue
    instead of being confidently delivered to apartment 1A.
    """
    n = len(li)
    lnum = lnorm["number"].to_numpy(dtype="U8")
    pnum = pnorm["number"].to_numpy(dtype="U8")
    lsort = np.array([_digit_sort(s) for s in lnorm["number"].tolist()], dtype="U8")
    psort = np.array([_digit_sort(s) for s in pnorm["number"].tolist()], dtype="U8")
    lunit = lnorm["unit"].to_numpy(dtype="U8")
    punit = pnorm["unit"].to_numpy(dtype="U8")
    ltype = lnorm["stype"].to_numpy(dtype="U8")
    ptype = pnorm["stype"].to_numpy(dtype="U8")

    number_exact = (lnum[li] == pnum[pi]).astype(np.float32)
    # Multiset equality is the cheap vectorized prefilter (same key blocking
    # uses); the feature itself demands a genuine one-slip adjacent swap.
    number_trans = (lsort[li] == psort[pi]).astype(np.float32) * (1.0 - number_exact)
    for j in np.where(number_trans > 0)[0]:
        if not _is_adjacent_transposition(str(lnum[li[j]]), str(pnum[pi[j]])):
            number_trans[j] = 0.0

    lu, pu = lunit[li], punit[pi]
    l_has, p_has = lu != "", pu != ""
    unit_agree = (lu == pu).astype(np.float32)  # includes "both blank" — vacuous agreement
    unit_conflict = (l_has & p_has & (lu != pu)).astype(np.float32)
    unit_unresolvable = (~l_has & p_has).astype(np.float32)

    stype_match = (ltype[li] == ptype[pi]).astype(np.float32)

    lz = lnorm["zip"].to_numpy() - synthetic.ZIP_BASE
    pz = pnorm["zip"].to_numpy() - synthetic.ZIP_BASE
    lr, lc = lz // synthetic.GRID, lz % synthetic.GRID
    pr, pc = pz // synthetic.GRID, pz % synthetic.GRID
    zip_dist = np.minimum(
        np.abs(lr[li] - pr[pi]) + np.abs(lc[li] - pc[pi]), 3
    ).astype(np.float32)

    # Name similarity: memoized per label over the (few) distinct candidate
    # street names, because thousands of candidate rows share a handful of names.
    tri = np.empty(n, dtype=np.float32)
    tok = np.empty(n, dtype=np.float32)
    lnames = lnorm["name"].tolist()
    pnames = pnorm["name"].tolist()
    # Record token sets, one per delivery point: number, name words, type, unit.
    # token_set_overlap compares them against the tokens the customer typed —
    # deliberately the RAW line, not the parse, so it stays honest when the
    # parser misfires and it prices in rider junk the parser threw away.
    point_tokens = [
        frozenset(t for t in (num, *nm.split(), st, un) if t)
        for num, nm, st, un in zip(
            pnorm["number"], pnorm["name"], pnorm["stype"], pnorm["unit"]
        )
    ]
    label_tokens = [frozenset(t.replace("#", "") for t in raw.split()) for raw in lnorm["raw"]]
    pi_list = pi.tolist()
    starts = np.searchsorted(li, np.arange(len(lnorm) + 1))
    for a in range(len(lnorm)):
        s, e = int(starts[a]), int(starts[a + 1])
        if s == e:
            continue
        lt = _trigrams(lnames[a])
        ltk = label_tokens[a]
        memo: dict[str, float] = {}
        for j in range(s, e):
            c = pi_list[j]
            nm = pnames[c]
            v = memo.get(nm)
            if v is None:
                v = _jaccard(lt, _trigrams(nm))
                memo[nm] = v
            tri[j] = v
            tok[j] = _jaccard(ltk, point_tokens[c])

    X = np.column_stack(
        [tri, tok, number_exact, number_trans, unit_agree, unit_conflict,
         unit_unresolvable, stype_match, zip_dist]
    ).astype(np.float32)
    return X


# --------------------------------------------------------------------------- decision
def decide(
    labels: pd.DataFrame,
    points: pd.DataFrame,
    li: np.ndarray,
    pi: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """Pick the best candidate per label; accept it or route to review.

    Returns one row per label with everything evaluation and explanation need:
    the best pair's probability, whether the best candidate is the true point
    (``hit_best``), and the accept decision at the given threshold.
    """
    n = len(labels)
    best_pair = np.full(n, -1, dtype=np.int64)
    if len(li):
        order = np.lexsort((probs, li))
        lo = li[order]
        last = np.r_[lo[1:] != lo[:-1], True]
        winners = order[last]
        best_pair[li[winners]] = winners

    p_best = np.where(best_pair >= 0, probs[np.maximum(best_pair, 0)], -1.0)
    matched_idx = np.where(best_pair >= 0, pi[np.maximum(best_pair, 0)], -1)
    point_ids = points["point_id"].to_numpy(dtype="U12")
    matched_id = np.where(matched_idx >= 0, point_ids[np.maximum(matched_idx, 0)], "")

    true_id = labels["true_point_id"].fillna("").to_numpy(dtype="U12")
    accepted = (best_pair >= 0) & (p_best >= threshold)
    hit_best = (matched_id == true_id) & (matched_id != "")

    return pd.DataFrame(
        {
            "label_id": labels["label_id"].to_numpy(),
            "true_point_id": true_id,
            "matched_point_id": np.where(accepted, matched_id, ""),
            "best_point_id": matched_id,
            "best_pair": best_pair,
            "p_best": p_best,
            "n_candidates": np.bincount(li, minlength=n) if len(li) else np.zeros(n, dtype=int),
            "accepted": accepted,
            "hit_best": hit_best,
            "auto_correct": accepted & hit_best,
            "is_orphan": true_id == "",
            "corruptions": labels["corruptions"].fillna("").to_numpy(),
        }
    )


# --------------------------------------------------------------------------- baselines
def exact_match_baseline(lnorm: pd.DataFrame, pnorm: pd.DataFrame) -> np.ndarray:
    """Status quo #1: accept only a perfect match after normalization.

    Perfectly precise and pathologically shy — one keyboard slip and the label
    goes to a human. Its coverage is the floor any scorer must clear.
    """
    index: dict[tuple, int] = {}
    for idx, key in enumerate(
        zip(pnorm["number"], pnorm["name"], pnorm["stype"], pnorm["unit"], pnorm["zip"])
    ):
        index[key] = idx
    out = np.full(len(lnorm), -1, dtype=np.int64)
    for a, key in enumerate(
        zip(lnorm["number"], lnorm["name"], lnorm["stype"], lnorm["unit"], lnorm["zip"])
    ):
        out[a] = index.get(key, -1)
    return out


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def fuzzy_top1_baseline(
    lnorm: pd.DataFrame,
    pnorm: pd.DataFrame,
    li: np.ndarray,
    pi: np.ndarray,
    label_mask: np.ndarray,
) -> np.ndarray:
    """Status quo #2, the dangerous one: nearest edit distance, no reject.

    Whatever candidate sits closest in edit distance gets the parcel — always.
    An orphan label has no right answer, so its nearest neighbor is delivered
    to by construction. The distance is field-aligned Levenshtein (number +
    name + type + unit, memoized per field); ranking matches composed-string
    distance because the fields line up, and it keeps the demo fast.
    """
    out = np.full(len(lnorm), -1, dtype=np.int64)
    lnum = lnorm["number"].tolist()
    lname = lnorm["name"].tolist()
    ltype = lnorm["stype"].tolist()
    lunit = lnorm["unit"].tolist()
    pnum = pnorm["number"].tolist()
    pname = pnorm["name"].tolist()
    ptype = pnorm["stype"].tolist()
    punit = pnorm["unit"].tolist()
    pi_list = pi.tolist()
    starts = np.searchsorted(li, np.arange(len(lnorm) + 1))
    for a in np.where(label_mask)[0]:
        s, e = int(starts[a]), int(starts[a + 1])
        if s == e:
            continue
        num_memo: dict[str, int] = {}
        name_memo: dict[str, int] = {}
        best_d, best_c = 10**9, -1
        for j in range(s, e):
            c = pi_list[j]
            d = num_memo.get(pnum[c])
            if d is None:
                d = _levenshtein(lnum[a], pnum[c])
                num_memo[pnum[c]] = d
            dn = name_memo.get(pname[c])
            if dn is None:
                dn = _levenshtein(lname[a], pname[c])
                name_memo[pname[c]] = dn
            d += dn + (ltype[a] != ptype[c]) + _levenshtein(lunit[a], punit[c])
            if d < best_d:  # ties: first candidate wins, silently — the danger
                best_d, best_c = d, c
        out[a] = best_c
    return out
