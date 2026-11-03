"""Clinker planking read from the sections: the lands (the steps where a strake laps over the one below) found on
every station's section, side by side, and joined along the hull into land lines.

Method (the design's B1, after Fröhlich & Nielsen's description of a clinker hull, whose automatic fitting they
leave to future work): per station, per side (``port``: y > 0 in ship coordinates, ``starboard``: y < 0, split at the
keel = the section's lowest point within 5 cm of y = 0), the raw section (never the water-corrected one) as a
polyline from the rabbet outward (the top of a keel's side standing below the garboards; the keel itself on a hull
without one) with its arc length ``s``, ended where it turns back (round the sheer onto the inner
face, or where it bends back into sediment, a trench or the seabed: a rule on the direction of the smoothed
profile, never on heights); pieces that do not continue it (a loose timber) are left apart and counted, a piece that
goes on beyond a gap makes the side unreadable, a side whose keel cannot be told from the planking (steep from the
start within the keel's half width with no sole; a steep run rising from the sole that leaves the half width with no
corner, or whose steepness comes back after it) is skipped. The keel, the sole, the rabbet, the direction rules and
the ARC are read on a working profile: the raw one resampled every ``SAMPLE`` of arc and averaged twice over
``SMOOTH`` (a triangle over 1 cm, below the lap, so the lands' steps survive); a scan's noise lengthens the raw arc
by its variance over the vertex spacing squared (1 % at half a millimetre on 5 mm) and breaks any rule read on
single segments. A moving average over ``window`` metres of arc, applied twice, is the hull "without planks"; the
signed offset ``d(s)`` of the raw profile from it, along its outward normal, is a saw tooth on a clinker hull:
walking from the sheer down to the keel ``d``
rises gently inside a strake and drops by the step at each land (the lower edge of a strake lies outside the strake
below). A land is where ``d`` changes by more than ``min_step`` within ``step_len`` of arc, the lands at least
``min_strake`` apart; its position is the centre of the step (the centroid of the rise of ``d``); its step is read
on ``d`` - the same as on the raw profile once the slope of the flats is taken out: the two flats' lines (five
samples each side, ``FLAT`` apart over 35 mm of arc from the land's centre, their median, the common slope removed)
differenced at the land; its arc ``s`` is the working
profile's arc at that point, from the rabbet. Lands too near an end of the
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

DEFAULTS = {"min_step": 0.008, "step_len": 0.03, "min_strake": 0.08, "window": 0.12, "keel_half_width": 0.10,
            "sole_width": 0.02}
"""The parameters of the lands, in metres: ``min_step`` the least step of a land; ``step_len`` the arc it may take;
``min_strake`` the least distance between two lands; ``window`` the arc of the moving average (the hull "without
planks"); ``keel_half_width`` how far from the keel's y a keel standing below the planking may reach; ``sole_width``
the least width of a SOLE over both sides (2 cm: a narrow worn keel that comes out ``keel_unclear`` at every station
is read by lowering it). The keel's side, whose top is the rabbet where the profile starts, is read on the working
profile (resampled every ``SAMPLE``, averaged twice over ``SMOOTH``: the rules below hold at half a millimetre of
noise on vertices 5 mm apart, mostly at 1 mm) in two ways. On a sole (the bottom through the keel point flatter than
``SOLE_DEG``, 30 degrees, and at least ``sole_width`` wide over both sides: the apex of a V has none, a heeled wreck's
bottom still has one) a run of segments steeper than ``SOLE_STEEP_DEG`` (60 degrees: a tapered keel, worn edges, the
low side of a wreck heeled by 12-15 degrees) that rises at least ``RABBET_RISE`` (2 cm) within the half width is the
keel's side; the rabbet is its sharpest drop of direction within the half width (at least ``CORNER_DEG``, 10 degrees,
read on chords of ``CORNER_ARC``, 2 cm: 90 to 70 degrees under a V of planking above a keel, 90 to the deadrise
otherwise), else its top when the run ends within the half width AND its steepness does not come back within
``min_strake`` after it (a land's ramp, after which a V of planking goes on: planking, not keel; a chord within
``RETURN_TOL_DEG`` of the threshold counts). The first such run,
the one rising from the sole, that leaves the half width with no corner, or is refused because its steepness comes
back, gives ``keel_unclear``: a keel's side collinear with the garboards (the rabbet invisible), a V of 80 degrees or
more above a keel (the corner under 10 degrees) are flagged sides, never measured from the keel's bottom with the
keel's top as a land (known limits). Without a sole only a run steeper than ``STEEP_DEG`` (80 degrees) that lies ALL
within the half width, rises ``RABBET_RISE`` and whose steepness does not come back after it is the keel's side (a
steep V of planking, the end stations, goes on beyond the half width or comes back after each land); a profile that
starts (within ``sole_width`` of arc) with a run between 60 and 80 degrees, within the half width and rising 2 cm,
whose steepness does not return within ``min_strake`` after it is ``keel_unclear``: that side is skipped and
listed, never measured from a wrong origin. A pointed keel (no sole) collinear with the planking is read as planking
from its bottom: nothing tells them apart."""
SIDES = ("port", "starboard")
SIDE_SIGN = {"port": 1.0, "starboard": -1.0}
MIN_VERTICES = 10
JOIN = 0.02  # pieces of a side's line farther apart than this: the model is missing there
KEEL_BAND = 0.05  # the keel: the hull's lowest point within 5 cm of y = 0
KEEL_FLAT = 0.002  # ... the middle of the bottom within 2 mm of that height (a flat keel, a keel off the centreline)
KEEL_REACH = 1.5  # the hull among the pieces near y = 0: the one that rises most within 1.5 m of it
SOLE_DEG = 30.0  # a sole: the bottom through the keel point flatter than this (and at least ``sole_width`` wide)
TURN_DEG = 150.0  # the profile has turned round the sheer onto the inner face
STEEP_DEG = 80.0  # no sole: a keel's side is steeper than this, all of it within ``keel_half_width`` of the keel's y...
RABBET_RISE = 0.02  # ... and rising at least 2 cm: the planking starts at its top, the rabbet
SOLE_STEEP_DEG = 60.0  # a keel's side on a sole: steeper than this (a tapered keel, a heeled wreck)...
CORNER_ARC = 0.02  # ... the rabbet where its direction drops most, read on chords of 2 cm either side of a vertex...
CORNER_DEG = 10.0  # ... by at least this much; else its top, when the run ends within the half width
RETURN_TOL_DEG = 1.0  # a chord this much under a run's threshold still brings its steepness back (a V of planking built
# or scanned right at the threshold reads 80 degrees on one chord and 79.9 on the next)
BEND_DEG = 45.0  # ... or bends back this much below the steepest it reached, for ``min_strake``: sediment, a trench
SAMPLE = 0.002  # the profile resampled every 2 mm of arc
SMOOTH = 0.006  # the working profile: the raw one averaged twice over this much arc (a triangle over 1 cm, below the
# lap: the lands' steps stay; a box of 1 cm cut 0.3 mm off each corner of a lap, 5 mm over eight lands)
FLAT = 0.005  # the flats' samples beside a land, 5 mm apart
LAYERS = {"port": "FASCE_BABORDO", "starboard": "FASCE_TRIBORDO"}
TABLE = "fasce_linee"
JUMP = 0.02  # a station whose rabbet or garboard lies farther than this from the curve through its neighbours
NOISE_FACTOR = 4.0  # a width's uncertainty: this many times the section's noise, plus half the step (see build_strakes)
MESH_STEP = 0.10  # the strakes' mesh: a row at every station and every 10 cm of x between them
SIDE_CODES = {"port": ("B", 0), "starboard": ("T", 1)}  # babordo / tribordo: the letter of the name, the PLY's side
CATALOGUE_NOTES = ("modello delle fasce (ipotesi)", "strake model (hypothesis)")  # the notes of the records it writes
HYPOTHESIS = ("modello delle fasce: ipotesi dalle sezioni, non una misura",
              "strake model: a hypothesis from the sections, not a measure")


def _side_name(side: str) -> str:
    return t("babordo", "port") if side == "port" else t("tribordo", "starboard")


def _reason_text(reason: str) -> str:
    return {"empty": t("nessuna linea", "no line"),
            "few": t(f"meno di {MIN_VERTICES} vertici", f"fewer than {MIN_VERTICES} vertices"),
            "gap": t("sezione interrotta: il modello manca", "section broken: the model is missing"),
            "keel": t("la sezione non arriva alla chiglia", "the section does not reach the keel"),
            "keel_unclear": t("chiglia o fasciame? il fianco della chiglia non si distingue dal fasciame (ripido "
                              "dall'inizio entro la mezza larghezza della chiglia senza suola, o senza spigolo in "
                              "cima)",
                              "keel or planking? the keel's side cannot be told from the planking (steep from the "
                              "start within the keel's half width with no sole, or with no corner at its top)"),
            }.get(reason, reason)


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


def _hull_piece(polylines: list[np.ndarray]) -> np.ndarray | None:
    """The hull among the pieces of a section: of the pieces that cross or come within ``KEEL_BAND`` of y = 0, the
    one that rises most within ``KEEL_REACH`` of it (a trench floor under a raised hull or a loose timber does not).
    None when no piece comes near y = 0."""
    best, score = None, -np.inf
    for p in polylines:
        near = np.abs(p[:, 0]) <= KEEL_BAND
        if not near.any() and not (np.sign(p[:-1, 0]) * np.sign(p[1:, 0]) < 0).any():
            continue
        low = float(p[near, 1].min()) if near.any() else float(_crossings(p)[0][1])
        rise = float(p[np.abs(p[:, 0]) <= KEEL_REACH, 1].max()) - low
        if rise > score:
            best, score = p, rise
    return best


def _keel(p: np.ndarray, params: dict) -> tuple[np.ndarray, float]:
    """The keel of the hull's working piece ``p`` (:func:`_working`) and the width of its sole. The keel: the
    piece's lowest point near y = 0, moved to the middle of the bottom that lies within ``KEEL_FLAT`` of that height
    (a flat keel is read at its middle, a hull lying a few cm off the centreline from its own keel); the heights are
    those of the piece smoothed twice over ``window`` (on a round bottom the band is 15 cm wide and one noisy sample
    2 mm up cut it short by centimetres), the point itself lies on ``p``. The sole: the bottom through the keel
    point flatter than ``SOLE_DEG``, measured over both sides (:func:`_sole`)."""
    near = np.flatnonzero(np.abs(p[:, 0]) <= KEEL_BAND)
    if not len(near):  # the piece crosses y = 0 between two vertices: its lowest crossing
        keel = _crossings(p)[0]
    else:
        n = int(round(params["window"] / SAMPLE))
        z = (_smooth(_smooth(p, n), n) if len(p) > n + 2 else p)[:, 1]
        i0 = int(near[np.argmin(z[near])])
        flat = z <= z[i0] + KEEL_FLAT
        lo, hi = i0, i0
        while lo > 0 and flat[lo - 1]:
            lo -= 1
        while hi < len(p) - 1 and flat[hi + 1]:
            hi += 1
        s = _arc(p[lo:hi + 1])
        mid = s[-1] / 2.0
        keel = np.array([np.interp(mid, s, p[lo:hi + 1, 0]), np.interp(mid, s, p[lo:hi + 1, 1])])
    return keel, _sole(p, keel)


def _sole(p: np.ndarray, keel: np.ndarray) -> float:
    """The width (across y) of the sole under the keel point: the segments flatter than ``SOLE_DEG`` on either side
    of the piece's vertex nearest to it, as far as they go on. 0 at the apex of a V; the whole bottom of a keel,
    heeled or not."""
    d = np.diff(p, axis=0)
    level = np.abs(d[:, 1]) <= np.tan(np.radians(SOLE_DEG)) * np.abs(d[:, 0])
    i = int(np.argmin(np.hypot(*(p - keel).T)))
    lo, hi = i, i  # the segments lo .. hi - 1
    while lo > 0 and level[lo - 1]:
        lo -= 1
    while hi < len(level) and level[hi]:
        hi += 1
    return float(np.abs(d[lo:hi, 0]).sum())


def _spans(mask: np.ndarray) -> list[tuple[int, int]]:
    """The runs of True in ``mask`` as ``(i, j)``: the items ``i`` to ``j - 1``."""
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def _arc(points: np.ndarray) -> np.ndarray:
    """The arc length along a polyline, 0 at its first vertex."""
    return np.r_[0.0, np.cumsum(np.hypot(*np.diff(points, axis=0).T))]


def _resample(points: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    su = np.linspace(0.0, float(s[-1]), int(np.ceil(float(s[-1]) / SAMPLE)) + 1)
    return su, np.column_stack([np.interp(su, s, points[:, 0]), np.interp(su, s, points[:, 1])])


def _working(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(raw, su, smooth)`` of a polyline: its vertices resampled every ``SAMPLE`` of its own arc ``su``, and the
    working profile, those samples averaged twice over ``SMOOTH`` of arc (:func:`_smooth`; a polyline shorter than
    the average is left as it is), sample by sample beside them."""
    su, raw = _resample(points, _arc(points))
    n = int(round(SMOOTH / SAMPLE))
    return raw, su, _smooth(_smooth(raw, n), n) if len(raw) > n else raw.copy()


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


def _rabbet(points: np.ndarray, keel: np.ndarray, sign: float, params: dict, sole: bool) -> int | str:
    """The index where the planking leaves the keel, the top of the keel's side, on the working profile
    ``points``; 0 when there is none (a hull without a keel standing below it: the profile starts at the keel);
    ``"keel_unclear"`` when the keel cannot be told from the planking. A run's steepness RETURNS when, within
    ``min_strake`` of arc after its top, the direction never falls under the run's threshold less ``RETURN_TOL_DEG``
    (the profile goes on as steep) or comes back over it within ``min_strake`` of where it first fell under (a
    land's ramp in a V of planking, after which the next strake runs at least that far); such a run is planking, not
    keel. With a ``sole`` under the keel point the keel's
    side is a run of segments steeper than ``SOLE_STEEP_DEG`` rising at least ``RABBET_RISE`` within
    ``keel_half_width`` of the keel's y: the rabbet is the sharpest drop of its direction (at least ``CORNER_DEG``,
    read on chords of ``CORNER_ARC`` either side of a vertex, over the whole profile) among its vertices within the
    half width and half a chord past its top (the smoothing rounds a corner: its middle lies a sample or two past
    the last steep segment) - the corner where the planking leaves it, 90 to 70 degrees under a V, 90 to the deadrise
    otherwise; a drop within a chord of the top of a run whose steepness returns is a land's ramp, never the rabbet -
    else its top when it ends within the half width and its steepness does not return; the last such run counts (a
    worn step in the side). The first run, the one rising from the sole,
    that leaves the half width with no corner (the keel's side collinear with the garboards: the rabbet invisible)
    or whose top is refused (a V of 80 degrees or more above the keel, its corner too small) gives ``keel_unclear``.
    Without a sole (the apex of a V) only a run steeper than ``STEEP_DEG`` that lies ALL within the half width and
    whose steepness does not return is the keel's side (a steep V of planking, an end station, goes on beyond it or
    comes back after each land); and a profile that starts (within ``sole_width`` of arc: a rounded apex) with a
    run between ``SOLE_STEEP_DEG`` and ``STEEP_DEG``, within the half width and rising ``RABBET_RISE``, is a V of
    planking when its steepness returns, else ``keel_unclear``."""
    half_width = params["keel_half_width"]
    q = np.column_stack([sign * (points[:, 0] - keel[0]), points[:, 1]])  # (out from the keel's y, height)
    u = q[:, 0]
    arc = _arc(points)
    # the direction of each segment, read on a chord of CORNER_ARC centred on it (shorter at the profile's ends):
    # the noise of a scan, even averaged, breaks a run read on single segments of 2 mm; rounded to a millionth of
    # a degree, so that a side built right at a threshold reads the same on every chord
    mid = 0.5 * (arc[:-1] + arc[1:])
    back = np.maximum(np.searchsorted(arc, mid - 0.5 * CORNER_ARC, side="right") - 1, 0)
    fore = np.minimum(np.searchsorted(arc, mid + 0.5 * CORNER_ARC, side="left"), len(arc) - 1)
    angle = np.round(np.degrees(np.arctan2(q[fore, 1] - q[back, 1], q[fore, 0] - q[back, 0])), 6)

    def returns(j: int, steep: float) -> bool:
        """Whether the steepness of a run ending at vertex ``j`` comes back: within ``min_strake`` after it the
        direction never falls under ``steep`` less ``RETURN_TOL_DEG`` (the run ended on a chord within the
        tolerance, or at the profile's end: the profile goes on as steep), or comes back over it within
        ``min_strake`` of where it first fell under (a land's ramp, after which the next strake runs at least
        that far)."""
        span, low = params["min_strake"], steep - RETURN_TOL_DEG
        later = j + np.flatnonzero(arc[j:-1] - arc[j] <= span)
        fallen = later[angle[later] < low]
        if not len(fallen):
            return True
        after = fallen[0] + np.flatnonzero(arc[fallen[0]:-1] - arc[fallen[0]] <= span)
        return bool((angle[after] >= low).any())

    found = 0
    if sole:
        first = True  # the next run rising within the half width is the one rising from the sole
        for i, j in _spans(angle >= SOLE_STEEP_DEG):
            within = _spans(u[i:j + 1] <= half_width)
            if not within:
                continue
            a, b = i + within[0][0], i + within[0][1] - 1  # the run's vertices within the half width
            if b <= a or q[b, 1] - q[a, 1] < RABBET_RISE:
                continue
            from_sole, first = first and a == i, False
            v = np.flatnonzero((arc >= arc[a]) & (arc <= arc[b] + 0.5 * CORNER_ARC))
            if b == j and returns(j, SOLE_STEEP_DEG):  # a drop at the top of this run is a land's ramp
                v = v[arc[b] - arc[v] >= CORNER_ARC]
            if len(v):
                back = np.maximum(np.searchsorted(arc, arc[v] - CORNER_ARC, side="right") - 1, 0)
                fore = np.minimum(np.searchsorted(arc, arc[v] + CORNER_ARC, side="left"), len(arc) - 1)
                before = np.degrees(np.arctan2(q[v, 1] - q[back, 1], q[v, 0] - q[back, 0]))
                after = np.degrees(np.arctan2(q[fore, 1] - q[v, 1], q[fore, 0] - q[v, 0]))
                drop = before - after
                k = int(np.argmax(drop))
                if drop[k] >= CORNER_DEG:
                    found = int(v[k])
                    continue
            if b == j and not returns(j, SOLE_STEEP_DEG):
                found = j
                continue
            if from_sole:  # no corner, and it leaves the half width or its steepness comes back: keel or planking?
                return "keel_unclear"
        return found
    for i, j in _spans(angle >= STEEP_DEG):
        if (u[i] <= half_width and q[j, 1] - q[i, 1] >= RABBET_RISE and u[i:j + 1].max() <= half_width
                and not returns(j, STEEP_DEG)):
            found = j
    if found:
        return found
    # steep from the start (a rounded apex: within sole_width of arc)
    runs = [(i, j) for i, j in _spans(angle >= SOLE_STEEP_DEG) if arc[i] <= params["sole_width"]]
    if runs:
        i, j = runs[0]
        if u[:j + 1].max() <= half_width and q[j, 1] - q[0, 1] >= RABBET_RISE and not returns(j, SOLE_STEEP_DEG):
            return "keel_unclear"
    return 0


def _profile(section: dict, side: str, params: dict | None = None):
    """``(points, s, info)`` of the side from the keel outward, or the reason why not: ``empty``, ``keel``, ``gap``,
    ``few``, ``keel_unclear`` (the keel cannot be told from the planking: :func:`_rabbet`). ``points, s``: the
    working profile (:func:`_working`) and its arc; ``info["raw"]``: the raw profile and its own arc, sample by
    sample beside it (the lands are read on the raw one, their arcs on the working one). ``info``: ``end`` "turn" or
    "bend" when the profile was ended there (:func:`_end`; a piece beyond that end is a stray, never a gap), else
    "line"; ``stray``: pieces left apart; ``keel``: the keel's y (the sides split there); ``sole``: the width of the
    bottom under it (:func:`_sole`); ``rabbet``: (y, z) where the profile and its arcs start (:func:`_rabbet`);
    ``noise``: the noise of the raw vertices of the profile (:func:`_noise`)."""
    sign = SIDE_SIGN[side]
    polylines = [np.asarray(line, float).reshape(-1, 2) for line in section.get("lines") or []]
    polylines = [p for p in polylines if len(p) > 1]
    if not polylines:
        return "empty"
    piece = _hull_piece(polylines)
    if piece is None:
        return "keel"
    params = DEFAULTS if params is None else {**DEFAULTS, **params}
    keel, sole = _keel(_working(piece)[2], params)
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
    vertices = np.vstack(chain)
    vertices = vertices[np.r_[True, np.hypot(*np.diff(vertices, axis=0).T) > 1e-9]]
    if len(vertices) < MIN_VERTICES:
        return "few"
    info = {"end": "line", "stray": len(rest), "keel": round(float(keel[0]), 5), "sole": round(sole, 4)}
    s_raw = _arc(vertices)  # the raw vertices' arcs: how many of them a cut profile keeps
    raw, su, points = _working(vertices)
    s = _arc(points)
    j = _rabbet(points, keel, sign, params, sole >= params["sole_width"])
    if isinstance(j, str):
        return j
    lo = 0.0
    if j:
        lo = float(su[j])
        raw, su, points, s = raw[j:], su[j:] - su[j], points[j:], s[j:] - s[j]
        if np.count_nonzero(s_raw >= lo - 1e-9) < MIN_VERTICES:
            return "few"
    info["rabbet"] = [round(float(v), 5) for v in points[0]]
    ended = _end(points, s, sign, params)
    if ended is not None:
        cut, info["end"] = ended
        f = min(float(np.interp(cut, s, np.arange(len(s)))), len(s) - 1.0)
        k = int(np.ceil(f))  # the samples before the cut, then the cut itself, on both profiles alike
        at = (lambda a: a[k - 1] + (f - (k - 1)) * (a[k] - a[k - 1])) if k else (lambda a: a[0])
        points, raw = np.vstack([points[:k], at(points)]), np.vstack([raw[:k], at(raw)])
        s, su = np.r_[s[:k], cut], np.r_[su[:k], at(su)]
        if np.count_nonzero((s_raw >= lo - 1e-9) & (s_raw <= lo + su[-1] + 1e-9)) < MIN_VERTICES:
            return "few"
    else:  # the line ends by itself: a piece farther out and higher goes on beyond a gap, the model is missing there
        end = points[-1]
        for r in rest:
            if ((sign * (r[:, 0] - end[0]) > JOIN) & (r[:, 1] > end[1])).any():
                return "gap"
    info["raw"] = (raw, su)
    info["noise"] = _noise(vertices[(s_raw >= lo - 1e-9) & (s_raw <= lo + su[-1] + 1e-9)])
    return points, s, info


def profile(section: dict, side: str) -> tuple[np.ndarray, np.ndarray] | None:
    """The side's working profile of a station's section (the raw line resampled every 2 mm of arc and averaged
    twice over 6 mm: the noise of a scan taken out, the lands' steps kept) from the rabbet (where the planking
    leaves the keel; the keel itself, its lowest point near y = 0, when no keel stands below the planking) outward,
    ``(N, 2)`` (y, z), and its arc length ``s`` ``(N,)``, ended where it turns round the sheer or bends back
    (sediment, a trench);
    None when the side has fewer than 10 vertices, does not reach the keel, goes on beyond a gap of more than
    2 cm (the model is missing there), or its keel cannot be told from the planking (steep from the start within the
    keel's half width with no sole; a steep run from the sole with no corner at its top). Pieces that do not continue
    it (a loose timber) are left apart."""
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
    "step_mm", "drop"}`` ordered from the keel; ``s`` the arc of the land's centre, ``y, z`` its point on the
    profile, ``step_mm`` the step, ``drop`` the fall of the offset ``d`` across ``step_len`` seen from the sheer
    (metres). ``side``: "port" or "starboard" (which way is outward); None: the side the profile lies on."""
    return _lands(points, s, params, side)[0]


def _lands(points, s, params=None, side=None, arc=None) -> tuple[list[dict], int]:
    """:func:`lands_of_profile` and the number of lands left out because they lie too near an end of the profile to
    read both flats. ``arc``: the working profile's arc, sample by sample beside ``s`` (:func:`_profile`): the lands
    are read on ``points`` (the raw profile: their points and steps) and their ``s`` given on ``arc``, which the
    noise of a scan does not lengthen."""
    p = {**DEFAULTS, **(params or {})}
    points, s = np.asarray(points, float), np.asarray(s, float)
    arc = s if arc is None else np.asarray(arc, float)
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
        offsets = half + FLAT * np.arange(5)
        if centre - offsets[-1] < 0.0 or centre + offsets[-1] > length:
            dropped += 1  # too near an end of the profile to read both flats
            continue
        below, upper = np.interp(centre - offsets, su, d), np.interp(centre + offsets, su, d)
        g = float(np.median(np.r_[np.diff(below[::-1]), np.diff(upper)] / FLAT))  # the flats' slope
        step = float(np.median(upper - g * offsets) - np.median(below + g * offsets))
        if step < p["min_step"]:
            continue
        out.append({"s": round(float(np.interp(centre, s, arc)), 5),
                    "y": round(float(np.interp(centre, su, raw[:, 0])), 5),
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
            raw, raw_arc = info.pop("raw")
            here, dropped = _lands(raw, raw_arc, p, side, arc=arc)
            stations.append({"name": section["name"], "x": float(section["value"]), "side": side, "count": len(here),
                             "length": round(float(arc[-1]), 5), "end": info["end"], "stray": info["stray"],
                             "keel": info["keel"], "sole": info["sole"], "rabbet": info["rabbet"],
                             "sheer": [round(float(v), 5) for v in points[-1]],
                             "noise_mm": round(info["noise"] * 1000.0, 3), "dropped_at_ends": dropped})
            found += [{"station": section["name"], "x": float(section["value"]), "side": side, **d, "k": None}
                      for d in here]
    mode, uncertain = {}, {}
    for side in SIDES:
        counts = [st["count"] for st in stations if st["side"] == side]
        values, freq = np.unique(counts, return_counts=True) if counts else (np.array([0]), np.array([1]))
        best = values[freq == freq.max()].max()  # a tie: the larger count, fewer lands left out
        mode[side] = int(best)
        uncertain[side] = [st["name"] for st in stations if st["side"] == side and st["count"] != mode[side]]
    jumps = {side: _jumps([st for st in stations if st["side"] == side], found) for side in SIDES}
    state = _state(chunk)
    for key in ("lines", "unmatched", "strakes", "strakes_built"):
        state.pop(key, None)
    state.update({"params": p, "source": _signature(data), "lands": found, "stations": stations, "mode": mode,
                  "uncertain": uncertain, "jumps": jumps, "skipped": skipped,
                  "detected": datetime.now().isoformat(timespec="seconds")})
    progress(0.95, t("Linee delle fasce", "Land lines"))
    match_lines(chunk)
    progress(1.0, t("Gradini trovati", "Lands found"))
    return {"lands": found, "stations": stations, "mode": mode, "uncertain": uncertain, "jumps": jumps,
            "skipped": skipped, "lines": state["lines"], "unmatched": state["unmatched"]}


def _noise(vertices: np.ndarray) -> float:
    """The noise of a side's section, metres: the robust spread (1.4826 MAD) of each raw vertex's offset from the
    local smooth line through its two neighbours (their chord), divided by ``sqrt(1 + w1^2 + w2^2)`` (the chord's
    own noise, ``w`` the neighbours' weights at the vertex; 1.5 for even spacing). Curvature adds the chord's sag
    (0.04 mm on 2 cm at a radius of 1.3 m) and the lands' corners a few vertices, which do not move a median. The
    offset from the working profile (averaged over 1 cm) is not it: on vertices 5 mm apart that average follows
    the noise and reads a fifth of it."""
    v = np.asarray(vertices, float)
    if len(v) < 3:
        return 0.0
    a, b, c = v[:-2], v[1:-1], v[2:]
    ac = c - a
    span = np.hypot(*ac.T)
    ok = span > 1e-9
    if not ok.any():
        return 0.0
    a, b, ac, span = a[ok], b[ok], ac[ok], span[ok]
    dist = (ac[:, 0] * (b[:, 1] - a[:, 1]) - ac[:, 1] * (b[:, 0] - a[:, 0])) / span
    w2 = np.clip(((b - a) * ac).sum(axis=1) / span**2, 0.0, 1.0)  # the weight of c at b's place along the chord
    e = dist / np.sqrt(1.0 + w2**2 + (1.0 - w2) ** 2)
    return float(1.4826 * np.median(np.abs(e - np.median(e))))


def _neighbours(x: np.ndarray, i: int, keep: list[int], each: int = 2) -> list[int]:
    """The ``2 * each`` items of ``keep`` nearest to ``i`` along x: ``each`` on each side, made up from the other
    side where one side has fewer (at the ends the nearest four)."""
    before = sorted((j for j in keep if x[j] < x[i]), key=lambda j: x[i] - x[j])
    after = sorted((j for j in keep if x[j] > x[i]), key=lambda j: x[j] - x[i])
    near = before[:each] + after[:each]
    extra = sorted(before[each:] + after[each:], key=lambda j: abs(x[j] - x[i]))
    return near + extra[:2 * each - len(near)]


def _residuals(x: np.ndarray, values: np.ndarray, keep: list[int]) -> dict[int, tuple[float, float]]:
    """For each item of ``keep``, ``(distance, limit)``: its distance from the least-squares curve along x through
    its neighbours among ``keep`` (:func:`_neighbours`; a parabola through four of them, so that the smooth curve
    of a keel's rocker, a rising forefoot or a hogged wreck is followed, a straight line through two or three;
    none with fewer than two), and the limit it is judged by: ``JUMP``, or three times the robust scatter (1.4826
    times the median distance) of the neighbours' own distances from that curve when that is larger (a station
    is not a jump where its neighbours scatter as much)."""
    out = {}
    for i in keep:
        near = _neighbours(x, i, keep)
        if len(near) < 2:
            continue
        u = x[near] - x[i]
        a = np.column_stack([u**2, u, np.ones(len(near))]) if len(near) >= 4 else np.column_stack([u, np.ones(len(near))])
        coef = np.linalg.lstsq(a, values[near], rcond=None)[0]
        scatter = 1.4826 * float(np.median(np.linalg.norm(values[near] - a @ coef, axis=1)))
        out[i] = (float(np.linalg.norm(values[i] - coef[-1])), max(JUMP, 3.0 * scatter))  # the curve's value at x[i]
    return out


def _outliers(x: np.ndarray, values: np.ndarray) -> dict[int, float]:
    """The items whose ``values`` (N, d) lie beyond their limit from the curve through their neighbours
    (:func:`_residuals`), one at a time: of those beyond it, the one whose removal leaves the others nearest to
    their curves, measured in limits (a jump at the third station also throws its neighbours off their curves,
    fitted across it: taking it out explains them, taking them out does not), then again without it; an item
    found is never in another's fit. ``{index: distance}``."""
    out: dict[int, float] = {}
    while True:
        keep = [i for i in range(len(x)) if i not in out]
        r = _residuals(x, values, keep)
        beyond = [i for i, (d, limit) in r.items() if d > limit]
        if not beyond:
            return out
        scores = {}
        for i in beyond:
            rest = _residuals(x, values, [j for j in keep if j != i]).values()
            scores[i] = (max((d / limit for d, limit in rest), default=0.0), -r[i][0] / r[i][1])
        best = min(beyond, key=scores.__getitem__)
        out[best] = r[best][0]


def _jumps(rows: list[dict], lands: list[dict]) -> list[str]:
    """The stations of one side whose rabbet (y, z) or garboard (the arc to the first land) jumps more than ``JUMP``
    (or the neighbours' own scatter) from the curve through the neighbouring stations (:func:`_outliers`): the count of lands is right but the
    origin may be shifted. Each such station's record gets ``jump`` (what jumped, mm); names by x."""
    rows = sorted(rows, key=lambda st: st["x"])
    found: dict[str, dict] = {}
    if rows:
        x = np.array([st["x"] for st in rows])
        for i, r in _outliers(x, np.array([st["rabbet"] for st in rows], float)).items():
            found.setdefault(rows[i]["name"], {})["rabbet_mm"] = round(r * 1000.0, 1)
        first = {}
        for d in lands:
            if d["side"] == rows[0]["side"] and (d["station"] not in first or d["s"] < first[d["station"]]):
                first[d["station"]] = d["s"]
        with_land = [st for st in rows if st["name"] in first]
        if with_land:
            x = np.array([st["x"] for st in with_land])
            g = np.array([[first[st["name"]]] for st in with_land])
            for i, r in _outliers(x, g).items():
                found.setdefault(with_land[i]["name"], {})["garboard_mm"] = round(r * 1000.0, 1)
    for st in rows:
        st.pop("jump", None)
        if st["name"] in found:
            st["jump"] = found[st["name"]]
    return [st["name"] for st in rows if st["name"] in found]


def _need_lands(chunk) -> dict:
    state = ship.state(chunk).get("clinker") or {}
    if "lands" not in state:
        raise ship.NotReady(t("Rileva prima i gradini delle fasce.", "Find the lands first."))
    return state


def match_lines(chunk) -> list[dict]:
    """The land lines ``{"side", "k", "points": [[x, y, z], ...], "uncertain_stations", "unmatched"}`` (``k`` from the
    keel, points in ship coordinates by x; ``uncertain_stations``: those with another count of lands or whose rabbet
    or garboard jumps (:func:`_jumps`), matched by arc like the others' lands; ``unmatched``: the stations where the
    line found no land), written to ``["lines"]``; every land gets its ``k`` (None for a land on no line, listed in ``["unmatched"]``)."""
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
        jumps = set((state.get("jumps") or {}).get(side) or [])
        jumped = set(jumps)
        certain = [st for st in rows if st["count"] == m and st["name"] not in jumped]
        if not certain:  # every station with the count jumps: their arcs are all there is
            certain, jumped = [st for st in rows if st["count"] == m], set()
        for st in certain:
            for k, d in enumerate(here[st["name"]], 1):
                d["k"] = k
        # the arc of each line from the keel along x, from the stations that agree: it follows the girth smoothly,
        # and a station whose profile is shorter (an eroded sheer) keeps its lands numbered from the keel
        xs = np.array([st["x"] for st in certain])
        arcs = np.array([[d["s"] for d in here[st["name"]]] for st in certain])  # (stations, m)
        missing = {k: [] for k in range(1, m + 1)}
        for st in rows:
            if st["count"] == m and st["name"] not in jumped:
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
        flagged = [st["name"] for st in rows if st["count"] != m or st["name"] in jumps]
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
            jumped = (state.get("jumps") or {}).get(side) or []
            if jumped:
                cm = num(JUMP * 100.0, 0)
                text += " " + t(f"Stazioni incerte (battura o torello che saltano di oltre {cm} cm rispetto alle "
                                f"stazioni vicine): {', '.join(jumped)}.",
                                f"Uncertain stations (the rabbet or the garboard jumping more than {cm} cm from the "
                                f"stations beside them): {', '.join(jumped)}.")
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


# ------------------------------------------------------------------------------------- the strakes
def _need_current(chunk) -> dict:
    """The lands, refused when the sections changed after them."""
    state = _need_lands(chunk)
    if is_stale(chunk):
        raise ship.NotReady(t("Le sezioni sono cambiate dopo il rilevamento: rileva di nuovo i gradini.",
                              "The sections changed after the lands were found: find the lands again."))
    return state


def _boundaries(state: dict, side: str) -> tuple[list[dict], list[dict]]:
    """The stations of a side (by x) and its ``mode + 2`` boundary lines, from the keel: the rabbet (the first
    vertex of each station's profile), the land lines 1 .. mode, the sheer (the last vertex). Each boundary maps a
    station's name to ``{"p": [x, y, z], "s": arc from the rabbet, "step_mm": the land's step or None}``."""
    rows = sorted((st for st in state["stations"] if st["side"] == side), key=lambda st: st["x"])
    m = int(state["mode"].get(side, 0))
    keel = {st["name"]: {"p": [st["x"], *st["rabbet"]], "s": 0.0, "step_mm": None} for st in rows}
    sheer = {st["name"]: {"p": [st["x"], *st["sheer"]], "s": st["length"], "step_mm": None} for st in rows}
    lands = [{} for _ in range(m)]
    for d in state["lands"]:
        if d["side"] == side and d["k"] is not None:
            lands[d["k"] - 1][d["station"]] = {"p": [d["x"], d["y"], d["z"]], "s": d["s"], "step_mm": d["step_mm"]}
    return rows, [keel, *lands, sheer]


def _polyline(boundary: dict) -> np.ndarray:
    return np.array(sorted(v["p"] for v in boundary.values()), float).reshape(-1, 3)


def _along(b: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """A boundary ``(N, 3)`` by x read at ``xs``: through its points, a cubic spline along x (not-a-knot; a
    parabola through three points, a straight line through two). A land line follows the hull's curve in plan:
    straight between stations 1 m apart it cuts in by more than a centimetre where the breadth changes."""
    if len(b) < 3:
        return np.column_stack([xs, np.interp(xs, b[:, 0], b[:, 1]), np.interp(xs, b[:, 0], b[:, 2])])
    from scipy.interpolate import CubicSpline

    spline = CubicSpline(b[:, 0], b[:, 1:], axis=0)
    out = spline(np.clip(xs, b[0, 0], b[-1, 0]))
    out[np.isin(xs, b[:, 0])] = b[np.searchsorted(b[:, 0], xs[np.isin(xs, b[:, 0])]), 1:]  # the points as they are
    return np.column_stack([xs, out])


def _rows_between(lower: np.ndarray, upper: np.ndarray, stations: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """The two boundaries of a strake resampled at the same x (:func:`_along`): every station and every
    ``MESH_STEP`` within the x both run over (``(0, 3)`` arrays when they share no stretch of x)."""
    if not len(lower) or not len(upper):
        return np.zeros((0, 3)), np.zeros((0, 3))
    x0, x1 = max(lower[0, 0], upper[0, 0]), min(lower[-1, 0], upper[-1, 0])
    if x1 - x0 < 1e-6:
        return np.zeros((0, 3)), np.zeros((0, 3))
    xs = np.unique(np.r_[[v for v in stations if x0 - 1e-9 <= v <= x1 + 1e-9], np.arange(x0, x1, MESH_STEP), x1])
    xs = xs[np.r_[True, np.diff(xs) > 1e-3]]
    return _along(lower, xs), _along(upper, xs)


def _outer_face(lower: np.ndarray, upper: np.ndarray, side: str, h_lower: float, h_upper: float):
    """The strake's outer face at its two edges. A land line runs through the middle of a land's step: the strake
    above it laps outside, its outer face ``h`` (half the step) farther out; the strake below lies inside, its outer
    face ``h`` farther in. So the lower edge moves out by ``h_lower`` and the upper one in by ``h_upper`` along the
    outward normal of the section plane (from the chord between the edges): both then lie on the strake's own
    outer face, which the scan shows (0 at the rabbet and at the sheer, which are on the scan)."""
    chord = upper[:, 1:] - lower[:, 1:]
    chord /= np.maximum(np.linalg.norm(chord, axis=1), 1e-12)[:, None]
    normal = SIDE_SIGN[side] * np.column_stack([chord[:, 1], -chord[:, 0]])  # outward: away from the centreline
    lo, hi = lower.copy(), upper.copy()
    lo[:, 1:] += h_lower * normal
    hi[:, 1:] -= h_upper * normal
    return lo, hi


def _median_step(boundary: dict) -> float | None:
    steps = [v["step_mm"] for v in boundary.values() if v["step_mm"] is not None]
    return float(np.median(steps)) if steps else None


def _strake_name(side: str, k: int) -> str:
    return f"{SIDE_CODES[side][0]}{k}"


def build_strakes(chunk, lap: float = 0.02, thickness: float | None = None, progress=None) -> dict:
    """The strake model, a hypothesis from the sections: per side, strake ``k`` (1 = the garboard) between the
    boundary ``k - 1`` (the rabbet for k = 1) and ``k`` (the sheer for the last) - :func:`_boundaries`.

    Measures per strake: ``widths`` per station, the arc between its two boundaries on that station's section
    (None where the station is uncertain on that side - another count of lands, a jump of the rabbet or the
    garboard - or skipped, or a boundary has no point there: never interpolated); ``uncertainty_mm`` per station,
    ``NOISE_FACTOR`` times the section's noise (:func:`_noise`) plus half the step of the land that bounds it (the
    upper one; the lower for the top strake): a land's centre is read on a ramp as wide as the lap and as high as
    the step, so half the step is how far the plank's edge may lie from it, and the noise moves the centre of a
    step by a few times its spread (on the test sections at 0.5 mm the lands lie within 1 mm of the truth; four
    times the noise covers them with a margin); ``length``, the 3D length of the strake's upper boundary (the
    land line, the sheer for the top strake) as the polyline through its station points, not the spline of the mesh; ``thickness_mm``, the median step of the land where the strake laps
    over the one below (line k - 1; for the garboard, line 1), or ``thickness``; ``lap_mm``, ``lap`` as given.

    Mesh: the two boundaries resampled at every station and every 10 cm of x where both run, moved onto the
    strake's outer face (:func:`_outer_face`, half the thickness at each land), joined as a ruled surface of
    triangles. ``<data>/shipwreck/clinker/strakes.obj`` in SHIP coordinates (``o fascia_B3``: port 3, ``T``
    starboard), ``strakes.ply`` in the chunk's coordinates (``frame.to_world``, double precision) with the vertex
    properties ``strake`` (int: k) and ``side`` (uchar: 0 port, 1 starboard). Kept in ``["strakes"]``. Refused
    when no land was found (a plain hull) or the sections changed."""
    progress = as_progress(progress)
    frame = ship.need_frame(chunk)
    state = _need_current(chunk)
    if not any(line["points"] for line in state.get("lines") or []):
        raise ship.NotReady(t("Nessun gradino di fasciame sulle sezioni: lo scafo appare liscio, nessuna fascia da "
                              "costruire.",
                              "No plank land on the sections: the hull looks smooth, no strake to build."))
    out, meshes = [], []
    for side in SIDES:
        if int(state["mode"].get(side, 0)) == 0:
            continue
        rows, bounds = _boundaries(state, side)
        unsure = set(state["uncertain"].get(side) or []) | set((state.get("jumps") or {}).get(side) or [])
        xs = [st["x"] for st in rows]
        for k in range(1, len(bounds)):
            progress(0.8 * len(out) / (2 * len(bounds)), t(f"Fascia {_strake_name(side, k)}",
                                                          f"Strake {_strake_name(side, k)}"))
            lower, upper = bounds[k - 1], bounds[k]
            widths, uncertainty = {}, {}
            for st in rows:
                a, b = lower.get(st["name"]), upper.get(st["name"])
                if st["name"] in unsure or a is None or b is None:
                    widths[st["name"]] = uncertainty[st["name"]] = None
                    continue
                step = b["step_mm"] if b["step_mm"] is not None else a["step_mm"]
                widths[st["name"]] = round(float(b["s"] - a["s"]), 5)
                uncertainty[st["name"]] = round(NOISE_FACTOR * st.get("noise_mm", 0.0) + 0.5 * (step or 0.0), 1)
            own = _median_step(bounds[k - 1] if k > 1 else bounds[1])
            thick = float(thickness * 1000.0) if thickness is not None else own
            top = _polyline(upper)
            length = float(np.linalg.norm(np.diff(top, axis=0), axis=1).sum()) if len(top) > 1 else 0.0
            out.append({"side": side, "k": k, "name": _strake_name(side, k), "widths": widths,
                        "uncertainty_mm": uncertainty, "length": round(length, 4),
                        "thickness_mm": None if thick is None else round(thick, 2), "lap_mm": round(lap * 1000.0, 2)})
            lo, hi = _rows_between(_polyline(lower), top, xs)
            if len(lo) < 2:
                continue
            if thickness is not None:
                h_lo = h_hi = thickness / 2.0
            else:
                h_lo = (_median_step(lower) or 0.0) / 2000.0 if k > 1 else 0.0
                h_hi = (_median_step(upper) or 0.0) / 2000.0 if k < len(bounds) - 1 else 0.0
            if k == 1:
                h_lo = 0.0
            if k == len(bounds) - 1:
                h_hi = 0.0
            lo, hi = _outer_face(lo, hi, side, h_lo, h_hi)
            meshes.append((side, k, lo, hi))
    folder = ship.folder(chunk) / "clinker"
    folder.mkdir(parents=True, exist_ok=True)
    progress(0.85, t("Modello delle fasce", "Strake model"))
    obj, ply = _write_strakes(frame, meshes, folder)
    state["strakes"] = out
    state["strakes_built"] = datetime.now().isoformat(timespec="seconds")
    progress(1.0, t("Fasce costruite", "Strakes built"))
    return {"strakes": out, "obj": obj, "ply": ply}


def _faces(lo: np.ndarray, hi: np.ndarray, side: str, start: int) -> np.ndarray:
    """The triangles of a ruled strip between the rows ``lo`` and ``hi`` (ship coordinates; the lower row's
    vertices numbered first from ``start``, then the upper), wound so that their normals point outward, away from
    the centreline (the section's outward normal of the chord from the lower to the upper edge,
    :func:`_outer_face`): the same on both sides, whichever way x runs."""
    n = len(lo)
    i = np.arange(n - 1)
    a, b = start + i, start + n + i
    f = np.vstack([np.column_stack([a, a + 1, b]), np.column_stack([a + 1, b + 1, b])])
    v = np.vstack([lo, hi])
    normal = np.cross(v[f[:, 1] - start] - v[f[:, 0] - start], v[f[:, 2] - start] - v[f[:, 0] - start])
    chord = (hi - lo)[np.r_[i, i]]
    outward = SIDE_SIGN[side] * np.column_stack([np.zeros(len(chord)), chord[:, 2], -chord[:, 1]])
    return f if (normal * outward).sum() >= 0 else f[:, [0, 2, 1]]


def _write_strakes(frame: ship.Frame, meshes: list, folder: Path) -> tuple[Path, Path]:
    from pyarchinit_photogrammetry.io.mesh_ply import MeshData, write_mesh

    hypothesis = t(*HYPOTHESIS)
    frame_text = (f"frame: ship coordinates (x along the keel, y to port, z up); start {list(frame.start)}, "
                  f"end {list(frame.end)}, bow {frame.bow}, z0 {frame.z0}, x0 {frame.x0}")
    text = [f"# {hypothesis}", f"# {frame_text}", "# o fascia_B<k>: port (babordo), o fascia_T<k>: starboard (tribordo)"]
    xyz, faces, strake, sides, start = [], [], [], [], 0
    for side, k, lo, hi in meshes:
        v = np.vstack([lo, hi])
        f = _faces(lo, hi, side, start)
        text.append(f"o fascia_{_strake_name(side, k)}")
        text += [f"v {a:.6f} {b:.6f} {c:.6f}" for a, b, c in v]
        text += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in f]
        xyz.append(v)
        faces.append(f)
        strake.append(np.full(len(v), k, np.int32))
        sides.append(np.full(len(v), SIDE_CODES[side][1], np.uint8))
        start += len(v)
    obj = folder / "strakes.obj"
    tmp = folder / "strakes.obj.tmp"
    tmp.write_text("\n".join(text) + "\n", encoding="utf-8")
    os.replace(tmp, obj)
    world = frame.to_world(np.vstack(xyz)) if xyz else np.zeros((0, 3))
    vertices = np.empty(len(world), [("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("strake", "<i4"), ("side", "u1")])
    vertices["x"], vertices["y"], vertices["z"] = world.T
    vertices["strake"] = np.concatenate(strake) if strake else []
    vertices["side"] = np.concatenate(sides) if sides else []
    comments = ["strake model: a hypothesis from the sections, not a measure (modello delle fasce: ipotesi dalle "
                "sezioni, non una misura)", "coordinates of the chunk (world), double precision",
                "strake = k from the keel (1 = garboard); side = 0 port (B), 1 starboard (T)"]
    triangles = np.vstack(faces) if faces else np.zeros((0, 3), np.int64)
    if frame.sign < 0:  # ship to world is a mirror there: the same winding would point inward
        triangles = triangles[:, [0, 2, 1]]
    ply = write_mesh(folder / "strakes.ply", MeshData(vertices, triangles, comments=comments))
    return obj, Path(ply)


def _need_strakes(chunk) -> dict:
    state = _need_current(chunk)
    if not state.get("strakes"):
        raise ship.NotReady(t("Costruisci prima il modello delle fasce.", "Build the strake model first."))
    return state


def strake_rows(chunk) -> list[dict]:
    """One row per strake and side, port first, from the keel: ``strake`` (B3), ``side``, ``k``, ``length`` (m),
    ``thickness_mm``, ``lap_mm``, ``widths`` and ``uncertainty_mm`` per station name, in the stations' order
    (None where nothing was measured)."""
    state = _need_strakes(chunk)
    order = [st["name"] for st in sorted(state["stations"], key=lambda st: st["x"])]
    order = list(dict.fromkeys(order))
    rows = []
    for d in sorted(state["strakes"], key=lambda d: (SIDES.index(d["side"]), d["k"])):
        rows.append({"strake": d["name"], "side": d["side"], "k": d["k"], "length": d["length"],
                     "thickness_mm": d["thickness_mm"], "lap_mm": d["lap_mm"],
                     "widths": {n: d["widths"].get(n) for n in order},
                     "uncertainty_mm": {n: d["uncertainty_mm"].get(n) for n in order}})
    return rows


def write_tables(chunk, folder=None) -> dict:
    """``strakes.csv`` and ``strakes.xlsx`` in ``folder`` (default ``<chunk data>/shipwreck/clinker``): a row per
    strake and side (:func:`strake_rows`), for each station its width (m) and its uncertainty (± mm); the CSV's
    first line and the sheet's title say it is a hypothesis from the sections."""
    import csv

    from pyarchinit_photogrammetry.io.tables import Sheet, write_xlsx

    rows = strake_rows(chunk)
    folder = Path(folder) if folder is not None else ship.folder(chunk) / "clinker"
    folder.mkdir(parents=True, exist_ok=True)
    stations = list(rows[0]["widths"]) if rows else []
    headers = [t("fascia", "strake"), t("lato", "side"), "k", t("lunghezza (m)", "length (m)"),
               t("spessore (mm)", "thickness (mm)"), t("sovrapposizione (mm)", "lap (mm)")]
    for name in stations:
        headers += [f"{name} (m)", f"{name} ± mm"]
    table = []
    for r in rows:
        line = [r["strake"], _side_name(r["side"]), r["k"], r["length"], r["thickness_mm"], r["lap_mm"]]
        for name in stations:
            line += [r["widths"][name], r["uncertainty_mm"][name]]
        table.append(line)
    title = t("Fasce (ipotesi dalle sezioni)", "Strakes (hypothesis from sections)")
    path_csv = folder / "strakes.csv"
    tmp = folder / "strakes.csv.tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        f.write(f"# {t(*HYPOTHESIS)}\n")
        w = csv.writer(f)
        w.writerow(headers)
        for line in table:
            w.writerow(["" if v is None else v for v in line])
    os.replace(tmp, path_csv)
    decimals = [None, None, 0, 3, 1, 1] + [3, 1] * len(stations)
    path_xlsx = write_xlsx(folder / "strakes.xlsx", {title: Sheet(headers, table, decimals)},
                           title=t(*HYPOTHESIS))
    return {"csv": path_csv, "xlsx": Path(path_xlsx)}


def to_catalogue(chunk) -> int:
    """One ``plank`` record of the catalogue per strake: its outline in plan (its two boundaries, the land lines
    k - 1 and k, in the chunk's x, y, closed), named ``fascia B3``, no station, noted as a hypothesis. Records of
    the model are replaced, never doubled: every record an earlier call wrote goes first (by the ids kept in
    ``["catalogue"]`` and by the hypothesis note, in either language), so the strakes of an earlier model with more
    lands leave no orphans. The strakes are those of ``["strakes"]``, their outlines from the same boundaries.
    Returns how many."""
    from . import timbers

    frame = ship.need_frame(chunk)
    state = _need_strakes(chunk)
    ours = set(state.get("catalogue") or [])
    timbers.remove(chunk, [r["id"] for r in timbers.timbers_of(chunk)
                           if r["id"] in ours or (r["kind"] == "plank" and r.get("notes") in CATALOGUE_NOTES)])
    written = []
    bounds = {side: _boundaries(state, side) for side in SIDES if any(d["side"] == side for d in state["strakes"])}
    for d in sorted(state["strakes"], key=lambda d: (SIDES.index(d["side"]), d["k"])):
        rows, lines_ = bounds[d["side"]]
        if d["k"] >= len(lines_):
            continue
        lo, hi = _rows_between(_polyline(lines_[d["k"] - 1]), _polyline(lines_[d["k"]]), [st["x"] for st in rows])
        if len(lo) < 2:
            continue
        outline = frame.to_world(np.vstack([lo, hi[::-1]]))[:, :2]
        record = timbers.add(chunk, "plank", d["side"], f"fascia {d['name']}", timbers.geometry("polygon", outline),
                             station="", notes=t(*CATALOGUE_NOTES))
        written.append(record["id"])
    state["catalogue"] = written
    return len(written)
