"""Clinker planking read from the sections: the lands (the steps where a strake laps over the one below) found on
every station's section, side by side, and joined along the hull into land lines.

Method (the design's B1, after Fröhlich & Nielsen's description of a clinker hull, whose automatic fitting they
leave to future work): per station, per side (``port``: y > 0 in ship coordinates, ``starboard``: y < 0, split at the
keel = the section's lowest point within 5 cm of y = 0), the raw section (never the water-corrected one) as a
polyline from the rabbet outward (the top of a keel's side standing below the garboards; the keel itself on a hull
without one) with its arc length ``s``, ended where it turns back (round the sheer onto the inner
face, or where it bends back into sediment, a trench or the seabed: a rule on the direction of the smoothed
profile, never on heights); pieces that do not continue it (a loose timber)
are left apart and counted, a piece that goes on beyond a gap makes the side unreadable. A moving average over
``window`` metres of arc, applied twice, is the hull "without planks"; the signed offset ``d(s)`` of the raw profile
from it, along its outward normal, is a saw tooth on a clinker hull: walking from the sheer down to the keel ``d``
rises gently inside a strake and drops by the step at each land (the lower edge of a strake lies outside the strake
below). A land is where ``d`` changes by more than ``min_step`` within ``step_len`` of arc, the lands at least
``min_strake`` apart; its position is the centre of the step (the centroid of the rise of ``d``); its step is read
on ``d`` - the same as on the raw profile once the slope of the flats is taken out: the two flats' lines (three
samples each side, their median, the common slope removed) differenced at the land. Lands too near an end of the
profile to read both flats are left out and counted. Nothing is faired or filled: a side with too few vertices, a
gap or no keel gives no lands and is listed. Where the preserved edge meets flat sediment at less than about 45
degrees the profile may run on into the sediment: narrow --half-width or check the stations listed as uncertain.

Across the stations the most common count of lands is the hull's on that side; a station with another count
(stealer, drop strake, eroded sheer, damage) is ``uncertain`` and its lands are matched to the lines by arc from the
rabbet (each line's arc interpolated along x from the stations that agree), the extra land listed, a missing one left
out. The land lines run through the matched lands in ship coordinates, ordered by x.

Kept in ``chunk.meta["shipwreck"]["clinker"]``; files in ``<chunk data>/shipwreck/clinker/``. The chunk's transform,
model and products are never changed."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

from pyarchinit_photogrammetry.core.progress import as_progress

from . import lines, ship
from .texts import num, plural, t

DEFAULTS = {"min_step": 0.008, "step_len": 0.03, "min_strake": 0.08, "window": 0.12, "keel_half_width": 0.10}
"""The parameters of the lands, in metres: ``min_step`` the least step of a land; ``step_len`` the arc it may take;
``min_strake`` the least distance between two lands; ``window`` the arc of the moving average (the hull "without
planks"); ``keel_half_width`` how far from the keel's y a keel standing below the planking may reach: a run of
segments steeper than ``STEEP_DEG`` (80 degrees), rising at least ``RABBET_RISE`` (2 cm) and lying ALL within it is
the keel's side, and the profile starts at its top, the rabbet. A steep V of planking (the end stations) goes on
beyond it and is not taken for a keel."""
SIDES = ("port", "starboard")
SIDE_SIGN = {"port": 1.0, "starboard": -1.0}
MIN_VERTICES = 10
JOIN = 0.02  # pieces of a side's line farther apart than this: the model is missing there
KEEL_BAND = 0.05  # the keel: the hull's lowest point within 5 cm of y = 0
KEEL_FLAT = 0.002  # ... the middle of the bottom within 2 mm of that height (a flat keel, a keel off the centreline)
KEEL_REACH = 1.5  # the hull among the pieces near y = 0: the one that rises most within 1.5 m of it
TURN_DEG = 150.0  # the profile has turned round the sheer onto the inner face
STEEP_DEG = 80.0  # a keel's side: steeper than this, all of it within ``keel_half_width`` of the keel's y...
RABBET_RISE = 0.02  # ... and rising at least 2 cm: the planking starts at its top, the rabbet
BEND_DEG = 45.0  # ... or bends back this much below the steepest it reached, for ``min_strake``: sediment, a trench
SAMPLE = 0.002  # the profile resampled every 2 mm of arc
FLAT = 0.005  # the flats' samples beside a land, 5 mm apart
LAYERS = {"port": "FASCE_BABORDO", "starboard": "FASCE_TRIBORDO"}
TABLE = "fasce_linee"


def _side_name(side: str) -> str:
    return t("babordo", "port") if side == "port" else t("tribordo", "starboard")


def _reason_text(reason: str) -> str:
    return {"empty": t("nessuna linea", "no line"),
            "few": t(f"meno di {MIN_VERTICES} vertici", f"fewer than {MIN_VERTICES} vertices"),
            "gap": t("sezione interrotta: il modello manca", "section broken: the model is missing"),
            "keel": t("la sezione non arriva alla chiglia", "the section does not reach the keel")}.get(reason, reason)


# ----------------------------------------------------------------------------------- one profile
def _runs(p: np.ndarray, sign: float) -> list[np.ndarray]:
    """The parts of a polyline ``(N, 2)`` (y, z) on the side ``sign`` of y = 0, cut where it crosses y = 0 (the
    crossing point added, at y = 0)."""
    q = sign * p[:, 0]
    out, run = [], []
    for i in range(len(p)):
        if q[i] >= 0:
            if not run and i > 0 and q[i - 1] < 0:
                f = q[i - 1] / (q[i - 1] - q[i])
                run.append(np.array([0.0, p[i - 1, 1] + f * (p[i, 1] - p[i - 1, 1])]))
            run.append(p[i])
        elif run:
            if q[i - 1] > 0:
                f = q[i - 1] / (q[i - 1] - q[i])
                run.append(np.array([0.0, p[i - 1, 1] + f * (p[i, 1] - p[i - 1, 1])]))
            out.append(np.asarray(run))
            run = []
    if run:
        out.append(np.asarray(run))
    return [r for r in out if len(r) > 1]


def _crossings(p: np.ndarray) -> list[np.ndarray]:
    """The points where a polyline crosses y = 0, the lowest first."""
    out = []
    for i in np.flatnonzero(np.sign(p[:-1, 0]) * np.sign(p[1:, 0]) < 0):
        f = p[i, 0] / (p[i, 0] - p[i + 1, 0])
        out.append(p[i] + f * (p[i + 1] - p[i]))
    return sorted(out, key=lambda q: q[1])


def _keel(polylines: list[np.ndarray]) -> np.ndarray | None:
    """The keel of a section: on the hull's piece (of the pieces that cross or come within ``KEEL_BAND`` of y = 0,
    the one that rises most within ``KEEL_REACH`` of it: a trench floor under a raised hull or a loose timber does
    not), its lowest point near y = 0, moved to the middle of the bottom that lies within ``KEEL_FLAT`` of that
    height (a flat keel is read at its middle, a hull lying a few cm off the centreline from its own keel). None
    when no piece comes near y = 0."""
    best, score = None, -np.inf
    for p in polylines:
        near = np.abs(p[:, 0]) <= KEEL_BAND
        if not near.any() and not (np.sign(p[:-1, 0]) * np.sign(p[1:, 0]) < 0).any():
            continue
        low = float(p[near, 1].min()) if near.any() else float(_crossings(p)[0][1])
        rise = float(p[np.abs(p[:, 0]) <= KEEL_REACH, 1].max()) - low
        if rise > score:
            best, score = p, rise
    if best is None:
        return None
    p = best
    near = np.flatnonzero(np.abs(p[:, 0]) <= KEEL_BAND)
    if not len(near):  # the piece crosses y = 0 between two vertices: its lowest crossing
        return _crossings(p)[0]
    i0 = int(near[np.argmin(p[near, 1])])
    flat = p[:, 1] <= p[i0, 1] + KEEL_FLAT
    lo, hi = i0, i0
    while lo > 0 and flat[lo - 1]:
        lo -= 1
    while hi < len(p) - 1 and flat[hi + 1]:
        hi += 1
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(p[lo:hi + 1], axis=0).T))]
    mid = s[-1] / 2.0
    return np.array([np.interp(mid, s, p[lo:hi + 1, 0]), np.interp(mid, s, p[lo:hi + 1, 1])])


def _resample(points: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    su = np.linspace(0.0, float(s[-1]), int(np.ceil(float(s[-1]) / SAMPLE)) + 1)
    return su, np.column_stack([np.interp(su, s, points[:, 0]), np.interp(su, s, points[:, 1])])


def _end(points: np.ndarray, s: np.ndarray, sign: float, params: dict) -> tuple[float, str] | None:
    """Where the hull's profile ends before the line does, and why: the direction of the profile smoothed over
    ``window`` (0 degrees along a level bottom, 90 up a vertical side, a little more with tumblehome) ``turn``s past
    ``TURN_DEG`` round the sheer onto the inner face, or ``bend``s back more than ``BEND_DEG`` below the steepest it
    reached, for ``min_strake`` of arc, once the profile has climbed (its direction risen ``BEND_DEG`` above the
    lowest before): sediment at any height, a trench wall, the seabed. The end is the sharpest
    point of that turn or bend; nothing is in metres of height, so a level bottom stepping down by its lands, or a
    side with tumblehome, is read whole. None when the line ends first."""
    if s[-1] < 4 * SAMPLE:
        return None
    su, p = _resample(points, s)
    h = su[1] - su[0]
    n = int(round(params["window"] / h))
    if len(p) <= n + 2:
        return None
    p = _smooth(_smooth(p, n), n)
    d = np.gradient(p, h, axis=0)
    angle = np.degrees(np.unwrap(np.arctan2(d[:, 1], sign * d[:, 0])))
    rate = np.gradient(angle, h)
    events = []
    turned = np.flatnonzero(angle > TURN_DEG)
    if len(turned):
        i = int(turned[0])
        lo, hi = max(0, i - n), min(len(angle), i + n + 1)
        events.append((i, lo + int(np.argmax(rate[lo:hi])), "turn"))
    # a bend counts once the profile has climbed: its direction risen BEND_DEG above the lowest before (a hollow
    # garboard flattening out near the keel, before the bilge rises, is not sediment)
    climbed = np.flatnonzero(angle - np.minimum.accumulate(angle) >= BEND_DEG)
    back = np.maximum.accumulate(angle) - angle > BEND_DEG
    back[:climbed[0] if len(climbed) else len(back)] = False
    held = max(1, int(round(params["min_strake"] / h)))
    run = 0
    for i, b in enumerate(back):
        run = run + 1 if b else 0
        if run == held:
            start = i - held + 1
            top = int(np.argmax(angle[:start + 1]))
            events.append((start, top + int(np.argmin(rate[top:start + 1])), "bend"))
            break
    if not events:
        return None
    _i, j, reason = min(events)
    return float(su[j]), reason


def _rabbet(points: np.ndarray, keel: np.ndarray, sign: float, half_width: float) -> int:
    """The index where the planking leaves the keel: the top of the last run of segments steeper than ``STEEP_DEG``
    that lies all within ``half_width`` of the keel's y and rises at least ``RABBET_RISE`` (the keel's side); 0 when
    there is none (a hull without a keel standing below it: the profile starts at the keel). Near-vertical, because a
    V of planking at 65-75 degrees above a keel would otherwise continue the keel's run; all within, because a steep
    V (an end station) passes ``half_width`` after a few decimetres of rise and must not be read as keel."""
    d = np.diff(points, axis=0)
    u = sign * (points[:, 0] - keel[0])
    steep = np.degrees(np.arctan2(d[:, 1], sign * d[:, 0])) >= STEEP_DEG
    found, i = 0, 0
    while i < len(steep):
        if steep[i] and u[i] <= half_width:
            j = i
            while j < len(steep) and steep[j]:
                j += 1
            if points[j, 1] - points[i, 1] >= RABBET_RISE and u[i:j + 1].max() <= half_width:
                found = j
            i = j
        else:
            i += 1
    return found


def _profile(section: dict, side: str, params: dict | None = None):
    """``(points, s, info)`` of the side from the keel outward, or the reason why not: ``empty``, ``keel``, ``gap``,
    ``few``. ``info``: ``end`` "turn" or "bend" when the profile was ended there (:func:`_end`; a piece beyond that
    end is a stray, never a gap), else "line"; ``stray``: pieces left apart; ``keel``: the keel's y (the sides split
    there); ``rabbet``: (y, z) where the profile and its arcs start (:func:`_rabbet`)."""
    sign = SIDE_SIGN[side]
    polylines = [np.asarray(line, float).reshape(-1, 2) for line in section.get("lines") or []]
    polylines = [p for p in polylines if len(p) > 1]
    if not polylines:
        return "empty"
    keel = _keel(polylines)
    if keel is None:
        return "keel"
    pieces = []
    for p in polylines:
        for r in _runs(p - [keel[0], 0.0], sign):
            r = r + [keel[0], 0.0]
            pieces.append(r if np.hypot(*(r[0] - keel)) <= np.hypot(*(r[-1] - keel)) else r[::-1])
    if not pieces:
        return "empty"
    pieces.sort(key=lambda r: np.hypot(*(r[0] - keel)))
    if np.hypot(*(pieces[0][0] - keel)) > JOIN:
        return "keel"
    chain, rest = [pieces[0]], pieces[1:]
    while rest:  # the piece whose end is nearest to the chain's end, turned to continue it
        end = chain[-1][-1]
        gap, i = min((min(np.hypot(*(r[0] - end)), np.hypot(*(r[-1] - end))), i) for i, r in enumerate(rest))
        if gap > JOIN:
            break
        r = rest.pop(i)
        chain.append(r if np.hypot(*(r[0] - end)) <= np.hypot(*(r[-1] - end)) else r[::-1])
    points = np.vstack(chain)
    points = points[np.r_[True, np.hypot(*np.diff(points, axis=0).T) > 1e-9]]
    if len(points) < MIN_VERTICES:
        return "few"
    params = DEFAULTS if params is None else {**DEFAULTS, **params}
    info = {"end": "line", "stray": len(rest), "keel": round(float(keel[0]), 5)}
    j = _rabbet(points, keel, sign, params["keel_half_width"])
    if j:
        points = points[j:]
        if len(points) < MIN_VERTICES:
            return "few"
    info["rabbet"] = [round(float(v), 5) for v in points[0]]
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(points, axis=0).T))]
    ended = _end(points, s, sign, params)
    if ended is not None:
        cut, info["end"] = ended
        keep = s < cut
        points = np.vstack([points[keep], [np.interp(cut, s, points[:, 0]), np.interp(cut, s, points[:, 1])]])
        s = np.r_[s[keep], cut]
        if len(points) < MIN_VERTICES:
            return "few"
    else:  # the line ends by itself: a piece farther out and higher goes on beyond a gap, the model is missing there
        end = points[-1]
        for r in rest:
            if ((sign * (r[:, 0] - end[0]) > JOIN) & (r[:, 1] > end[1])).any():
                return "gap"
    return points, s, info


def profile(section: dict, side: str) -> tuple[np.ndarray, np.ndarray] | None:
    """The side's polyline of a station's section from the rabbet (where the planking leaves the keel; the keel
    itself, its lowest point near y = 0, when no keel stands below the planking) outward, ``(N, 2)``
    (y, z), and its arc length ``s`` ``(N,)``, ended where it turns round the sheer or bends back (sediment, a trench);
    None when the side has fewer than 10 vertices, does not reach the keel, or goes on beyond a gap of more than
    2 cm (the model is missing there). Pieces that do not continue it (a loose timber) are left apart."""
    out = _profile(section, side)
    return None if isinstance(out, str) else out[:2]


def _smooth(a: np.ndarray, n: int) -> np.ndarray:
    """Moving average of ``n`` samples along the rows of ``a``; the ends extended by their point reflection, so
    that a straight end stays straight."""
    n = max(3, n | 1)
    k = n // 2
    head = 2.0 * a[0] - a[1:k + 1][::-1]
    tail = 2.0 * a[-1] - a[-k - 1:-1][::-1]
    padded = np.vstack([head, a, tail])
    c = np.cumsum(np.vstack([np.zeros((1, a.shape[1])), padded]), axis=0)
    return (c[n:] - c[:-n]) / n


def lands_of_profile(points, s, params: dict | None = None, side: str | None = None) -> list[dict]:
    """The lands of one side's profile (from the keel outward, as :func:`profile` gives it): ``{"s", "y", "z",
    "step_mm", "drop"}`` ordered from the keel; ``s`` the arc of the land's centre, ``y, z`` its point on the raw
    profile, ``step_mm`` the step, ``drop`` the fall of the offset ``d`` across ``step_len`` seen from the sheer
    (metres). ``side``: "port" or "starboard" (which way is outward); None: the side the profile lies on."""
    return _lands(points, s, params, side)[0]


def _lands(points, s, params=None, side=None) -> tuple[list[dict], int]:
    """:func:`lands_of_profile` and the number of lands left out because they lie too near an end of the profile to
    read both flats."""
    p = {**DEFAULTS, **(params or {})}
    points, s = np.asarray(points, float), np.asarray(s, float)
    length = float(s[-1]) if len(s) else 0.0
    half = p["step_len"] / 2.0
    m = max(1, int(round(half / SAMPLE)))
    if len(points) < 2 or length < 2.0 * (half + 2 * FLAT) + 4 * SAMPLE:
        return [], 0
    su, raw = _resample(points, s)
    h = su[1] - su[0]
    n = int(round(p["window"] / h))
    if len(raw) <= n + 2:
        return [], 0
    smooth = _smooth(_smooth(raw, n), n)
    tangent = np.gradient(smooth, h, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1), 1e-12)[:, None]
    sign = SIDE_SIGN[side] if side is not None else (1.0 if (raw[:, 0] - raw[0, 0]).mean() >= 0 else -1.0)
    normal = sign * np.column_stack([tangent[:, 1], -tangent[:, 0]])  # outward: away from the centreline, down at the keel
    d = ((raw - smooth) * normal).sum(axis=1)
    rise = np.full(len(d), -np.inf)
    rise[m:-m] = d[2 * m:] - d[:-2 * m]
    peaks = []
    above = rise > p["min_step"]
    i = 0
    while i < len(d):  # the highest point of each run above the threshold
        if above[i]:
            j = i
            while j < len(d) and above[j]:
                j += 1
            peaks.append(i + int(np.argmax(rise[i:j])))
            i = j
        else:
            i += 1
    kept: list[int] = []
    for i in sorted(peaks, key=lambda i: -rise[i]):
        if all(abs(su[i] - su[k]) >= p["min_strake"] for k in kept):
            kept.append(i)
    slope = np.gradient(d, h)
    out, dropped = [], 0
    for i in sorted(kept):
        lo, hi = max(0, i - m), min(len(d), i + m + 1)
        w = np.clip(slope[lo:hi], 0.0, None)
        centre = float((su[lo:hi] * w).sum() / w.sum()) if w.sum() > 0 else float(su[i])
        offsets = half + FLAT * np.arange(3)
        if centre - offsets[-1] < 0.0 or centre + offsets[-1] > length:
            dropped += 1  # too near an end of the profile to read both flats
            continue
        below, upper = np.interp(centre - offsets, su, d), np.interp(centre + offsets, su, d)
        g = float(np.median(np.r_[np.diff(below[::-1]), np.diff(upper)] / FLAT))  # the flats' slope
        step = float(np.median(upper - g * offsets) - np.median(below + g * offsets))
        if step < p["min_step"]:
            continue
        out.append({"s": round(centre, 5), "y": round(float(np.interp(centre, su, raw[:, 0])), 5),
                    "z": round(float(np.interp(centre, su, raw[:, 1])), 5), "step_mm": round(step * 1000.0, 2),
                    "drop": round(float(rise[i]), 5)})
    return out, dropped


# ------------------------------------------------------------------------------------ the hull
def _state(chunk) -> dict:
    return ship.state(chunk).setdefault("clinker", {})


def _signature(data: dict | None) -> dict | None:
    """What the lands were read from: the stations' raw lines (a hash) and when they were cut. The same sections
    cut again give the same hash: the lands still hold."""
    if data is None:
        return None
    stations = [[s["name"], s["value"], s["lines"]] for s in data.get("sections", []) if s["kind"] == "station"]
    digest = hashlib.sha256(json.dumps(stations, separators=(",", ":")).encode()).hexdigest()
    return {"built": data.get("built"), "sections": digest}


def detect_lands(chunk, params: dict | None = None, progress=None) -> dict:
    """The lands of every station and side, the count per side (``mode``), the ``uncertain`` stations, the
    ``skipped`` sides with the reason, then the land lines (:func:`match_lines`); written to
    ``chunk.meta["shipwreck"]["clinker"]``."""
    progress = as_progress(progress)
    ship.need_frame(chunk)
    data = lines.load(chunk)
    sections = sorted((s for s in (data or {}).get("sections", []) if s["kind"] == "station"), key=lambda s: s["value"])
    if not sections:
        raise ship.NotReady(t("Taglia prima le sezioni alle stazioni.", "Cut the sections at the stations first."))
    if lines.is_stale(chunk, data):
        raise ship.NotReady(t("Le sezioni non corrispondono più al modello o al sistema della nave: tagliale di nuovo.",
                              "The sections no longer match the model or the ship frame: cut them again."))
    p = {**DEFAULTS, **(params or {})}
    found, stations, skipped = [], [], []
    for n, section in enumerate(sections):
        progress(0.05 + 0.85 * n / len(sections), t(f"Gradini: {section['name']}", f"Lands: {section['name']}"))
        for side in SIDES:
            prof = _profile(section, side, p)
            if isinstance(prof, str):
                skipped.append({"station": section["name"], "x": float(section["value"]), "side": side, "reason": prof})
                continue
            points, arc, info = prof
            here, dropped = _lands(points, arc, p, side)
            stations.append({"name": section["name"], "x": float(section["value"]), "side": side, "count": len(here),
                             "length": round(float(arc[-1]), 5), "end": info["end"], "stray": info["stray"],
                             "keel": info["keel"], "rabbet": info["rabbet"], "dropped_at_ends": dropped})
            found += [{"station": section["name"], "x": float(section["value"]), "side": side, **d, "k": None}
                      for d in here]
    mode, uncertain = {}, {}
    for side in SIDES:
        counts = [st["count"] for st in stations if st["side"] == side]
        values, freq = np.unique(counts, return_counts=True) if counts else (np.array([0]), np.array([1]))
        best = values[freq == freq.max()].max()  # a tie: the larger count, fewer lands left out
        mode[side] = int(best)
        uncertain[side] = [st["name"] for st in stations if st["side"] == side and st["count"] != mode[side]]
    state = _state(chunk)
    for key in ("lines", "unmatched"):
        state.pop(key, None)
    state.update({"params": p, "source": _signature(data), "lands": found, "stations": stations, "mode": mode,
                  "uncertain": uncertain, "skipped": skipped, "detected": datetime.now().isoformat(timespec="seconds")})
    progress(0.95, t("Linee delle fasce", "Land lines"))
    match_lines(chunk)
    progress(1.0, t("Gradini trovati", "Lands found"))
    return {"lands": found, "stations": stations, "mode": mode, "uncertain": uncertain, "skipped": skipped,
            "lines": state["lines"], "unmatched": state["unmatched"]}


def _need_lands(chunk) -> dict:
    state = ship.state(chunk).get("clinker") or {}
    if "lands" not in state:
        raise ship.NotReady(t("Rileva prima i gradini delle fasce.", "Find the lands first."))
    return state


def match_lines(chunk) -> list[dict]:
    """The land lines ``{"side", "k", "points": [[x, y, z], ...], "uncertain_stations", "unmatched"}`` (``k`` from the
    keel, points in ship coordinates by x; ``unmatched``: the stations where the line found no land), written to
    ``["lines"]``; every land gets its ``k`` (None for a land on no line, listed in ``["unmatched"]``)."""
    from scipy.optimize import linear_sum_assignment

    state = _need_lands(chunk)
    by_station: dict = {}
    for d in state["lands"]:
        d["k"] = None
        by_station.setdefault((d["side"], d["station"]), []).append(d)
    for found in by_station.values():
        found.sort(key=lambda d: d["s"])
    out = []
    for side in SIDES:
        m = int(state["mode"].get(side, 0))
        if m == 0:
            continue
        rows = sorted((st for st in state["stations"] if st["side"] == side), key=lambda st: st["x"])
        here = {st["name"]: by_station.get((side, st["name"]), []) for st in rows}
        certain = [st for st in rows if st["count"] == m]
        for st in certain:
            for k, d in enumerate(here[st["name"]], 1):
                d["k"] = k
        # the arc of each line from the keel along x, from the stations that agree: it follows the girth smoothly,
        # and a station whose profile is shorter (an eroded sheer) keeps its lands numbered from the keel
        xs = np.array([st["x"] for st in certain])
        arcs = np.array([[d["s"] for d in here[st["name"]]] for st in certain])  # (stations, m)
        missing = {k: [] for k in range(1, m + 1)}
        for st in rows:
            if st["count"] == m:
                continue
            expected = np.array([np.interp(st["x"], xs, arcs[:, k]) for k in range(m)])
            # a land is a line's when it lies nearer to it than half the way to the next line
            spacing = np.abs(np.diff(expected))
            gate = np.full(m, float(state["params"]["min_strake"]))
            if m > 1:
                gate = 0.5 * np.minimum(np.r_[np.inf, spacing], np.r_[spacing, np.inf])
            lands = here[st["name"]]
            taken = set()
            if lands:
                cost = np.abs(np.array([d["s"] for d in lands])[:, None] - expected[None])
                for i, k in zip(*linear_sum_assignment(cost)):
                    if cost[i, k] <= gate[k]:
                        lands[i]["k"] = int(k) + 1
                        taken.add(int(k) + 1)
            for k in range(1, m + 1):
                if k not in taken:
                    missing[k].append(st["name"])
        flagged = [st["name"] for st in rows if st["count"] != m]
        for k in range(1, m + 1):
            points = [[d["x"], d["y"], d["z"]] for st in rows for d in here[st["name"]] if d["k"] == k]
            out.append({"side": side, "k": k, "points": points, "uncertain_stations": flagged, "unmatched": missing[k]})
    state["lines"] = out
    state["unmatched"] = [{key: d[key] for key in ("station", "x", "side", "s", "y", "z", "step_mm")}
                          for d in state["lands"] if d["k"] is None]
    return out


def is_stale(chunk) -> bool:
    """The sections were cut again (other stations, another model, another frame) after the lands were found."""
    state = ship.state(chunk).get("clinker") or {}
    if "lands" not in state:
        return False
    data = lines.load(chunk)
    if data is None or lines.is_stale(chunk, data):
        return True
    old, new = state.get("source") or {}, _signature(data)
    return old.get("sections") != new["sections"]


def _why(st: dict) -> str:
    """An uncertain station with what may explain it: its count, lands too near an end, a profile that turned back."""
    notes = [plural(st["count"], ("gradino", "gradini"), ("land", "lands"))]
    if st.get("dropped_at_ends"):
        notes.append(t(f"{st['dropped_at_ends']} vicino a un'estremità del profilo, non misurabile",
                       f"{st['dropped_at_ends']} too near an end of the profile to measure"))
    if st.get("end") in ("turn", "bend"):
        notes.append(t("profilo chiuso dove gira o piega", "profile ended where it turns or bends back"))
    return f"{st['name']} ({'; '.join(notes)})"


def summary(chunk) -> str:
    """The pane's text: strakes per side, uncertain stations, extra lands, skipped sides, and whether the sections
    changed since."""
    state = ship.state(chunk).get("clinker") or {}
    if "lands" not in state:
        return t("Fasce non ancora rilevate: taglia le sezioni alle stazioni e rileva i gradini.",
                 "Strakes not found yet: cut the sections at the stations and find the lands.")
    parts = []
    if is_stale(chunk):
        parts.append(t("Le sezioni sono cambiate dopo il rilevamento: rileva di nuovo i gradini.",
                       "The sections changed after the lands were found: find the lands again."))
    step = num(state["params"]["min_step"] * 1000.0, 0)
    if not state["lands"]:
        parts.append(t(f"Nessun gradino di fasciame trovato sulle sezioni (soglia {step} mm): lo scafo appare liscio, "
                       "nessuna fascia ricostruita.",
                       f"No plank land found on the sections (threshold {step} mm): the hull looks smooth, no strake "
                       "made up."))
    else:
        for side in SIDES:
            m = int(state["mode"].get(side, 0))
            name = _side_name(side).capitalize()
            if m == 0:
                parts.append(t(f"{name}: nessun gradino di fasciame trovato.", f"{name}: no plank land found."))
                continue
            text = t(f"{name}: {plural(m + 1, ('fascia', 'fasce'), ('strake', 'strakes'))} "
                     f"({plural(m, ('linea di gradini', 'linee di gradini'), ('land line', 'land lines'))}).",
                     f"{name}: {plural(m + 1, ('fascia', 'fasce'), ('strake', 'strakes'))} "
                     f"({plural(m, ('linea di gradini', 'linee di gradini'), ('land line', 'land lines'))}).")
            flagged = state["uncertain"].get(side) or []
            if flagged:
                why = ", ".join(_why(st) for st in state["stations"] if st["side"] == side and st["name"] in flagged)
                text += " " + t(f"Stazioni incerte (un altro numero di gradini): {why}.",
                                f"Uncertain stations (another count of lands): {why}.")
            parts.append(text)
    extra = state.get("unmatched") or []
    if extra:
        listed = ", ".join(f"{d['station']} {_side_name(d['side'])}" for d in extra)
        parts.append(t(f"Gradini fuori dalle linee: {listed}.", f"Lands on no line: {listed}."))
    loose = [f"{st['name']} {_side_name(st['side'])}" for st in state.get("stations", []) if st.get("stray")]
    if loose:
        parts.append(t(f"Pezzi di sezione staccati dal profilo, lasciati da parte: {', '.join(loose)}.",
                       f"Pieces of section apart from the profile, left aside: {', '.join(loose)}."))
    if state.get("skipped"):
        listed = "; ".join(f"{d['station']} {_side_name(d['side'])} ({_reason_text(d['reason'])})" for d in state["skipped"])
        parts.append(t(f"Lati non letti: {listed}.", f"Sides not read: {listed}."))
    return " ".join(parts)


# ------------------------------------------------------------------------------------- the files
def write_lines(chunk, folder=None) -> dict:
    """``lands.dxf`` (a 3D polyline per land line in ship coordinates, a layer per side, the line's number beside
    its first point) and ``lands.gpkg`` (table ``fasce_linee``, LINESTRING Z in the chunk's coordinates) in
    ``folder`` (default ``<chunk data>/shipwreck/clinker``)."""
    import ezdxf
    import shapely

    from pyarchinit_photogrammetry.io import gpkg

    frame = ship.need_frame(chunk)
    state = _need_lands(chunk)
    found = [line for line in state.get("lines") or [] if len(line["points"]) > 1]
    if not found:
        raise ship.NotReady(t("Nessuna linea di fasce da scrivere.", "No land line to write."))
    folder = Path(folder) if folder is not None else ship.folder(chunk) / "clinker"
    folder.mkdir(parents=True, exist_ok=True)

    doc = ezdxf.new("R2010", setup=True)
    doc.units = ezdxf.units.M
    for side, layer in LAYERS.items():
        doc.layers.add(layer, color=1 if side == "port" else 3)
    msp = doc.modelspace()
    for line in found:
        layer = LAYERS[line["side"]]
        msp.add_polyline3d([tuple(p) for p in line["points"]], dxfattribs={"layer": layer})
        msp.add_text(str(line["k"]), height=0.05, dxfattribs={"layer": layer}).set_placement(tuple(line["points"][0]))
    dxf = folder / "lands.dxf"
    tmp = folder / "lands.dxf.tmp"
    doc.saveas(tmp)
    os.replace(tmp, dxf)

    path = folder / "lands.gpkg"
    srs = gpkg.srs_of(chunk.crs)
    if not gpkg.is_geopackage(path) and path.exists():
        gpkg.refuse_foreign(path)
        path.unlink()  # empty: nothing to lose
    features = [(shapely.LineString(frame.to_world(line["points"])),
                 {"side": line["side"], "k": line["k"], "uncertain": len(line["uncertain_stations"]),
                  "stations": len(line["points"])}) for line in found]
    con = gpkg.connect(path)
    try:
        con.execute("BEGIN IMMEDIATE")
        gpkg.create(con, srs)
        cols = gpkg.ensure_table(con, TABLE, "LINESTRING", {"side": "TEXT", "k": "INTEGER", "uncertain": "INTEGER",
                                                            "stations": "INTEGER"}, srs[0], recreate=True, z=True)
        gpkg.replace_rows(con, TABLE, cols, features, srs[0], z=True)
        con.execute("COMMIT")
    except BaseException:
        if con.in_transaction:
            con.execute("ROLLBACK")
        raise
    finally:
        con.close()
    return {"dxf": dxf, "gpkg": path}
