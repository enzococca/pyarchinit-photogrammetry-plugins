"""Proposals read on the relief of the hull from above (:mod:`relief`): the seams of the planking, the frames, the
longitudinal timbers (keelson, stringers) and the strakes between the seams. They are proposals: the user accepts them
into the catalogue (:mod:`tracing`), corrects them by hand or rejects them; a new reading never touches the
catalogue's records and never proposes again what was accepted or rejected.

Everything is worked on a grid in the ship plan (rows along ship y, increasing; columns along ship x), the relief's
height raster turned so (its v is ``sign`` * y).

- Seams: a land of lapped planking is a step of 1-2.5 cm across the hull. The transverse gradient ``g = dz/dy`` minus
  its running median over ``trend`` on either side (the hull's own slope taken out) has a ridge there; ``|g|`` smoothed over
  ``sigma_cells`` across gives its crest in each column (sub-cell, on a parabola), the step of a crest is ``g``
  integrated across its ridge. Two crests of a side closer than ``trend`` cannot be told apart (the ends of a hull,
  where the strakes seen from above are a few centimetres wide): neither is kept. Crests are followed along x, each
  line along its own course, with hysteresis (a line starts where the step reaches
  ``min_step`` and goes on while it keeps ``low_step``); pieces closer than ``gap`` along x and collinear within 2 cm
  are merged; lines from ``min_seam`` long are kept. Crests on a longitudinal timber's side (its step is the timber's,
  not a land's) are left out.
- Frames and longitudinal timbers: DIRECTIONAL top-hats. The height minus its grey opening by a line of ``opening``
  along x keeps what is narrower than that along x: the frames (a keelson, long along x, survives the opening and
  leaves no residue); the same along y keeps the longitudinal timbers. An isotropic opening would merge the keelson
  with the frames it crosses. Before an opening the cells off the model are filled with the nearest height along the
  line of the opening (a minimum there would make the model's rim a ridge as high as the hull is deep); after it they
  are off again. The residue is thresholded at ``top_hat``; components from ``min_area``; a frame has its long axis
  within ``frame_deg`` of y and an elongation (long / short side of the minimum rotated rectangle) from
  ``elongation``; the two pieces of a frame cut by the keelson are joined when their x differ by less than the room;
  a longitudinal within ``keel_band`` of y 0 is the keelson, else a stringer; a frame shorter than ``min_frame`` or a
  longitudinal shorter than ``min_longitudinal`` (a loose timber) is ``other``, as anything else is (the residues of
  one loose timber in both openings are one item).
- Strakes: per side, each seam's land is its rank from the keel (:func:`lands`: the pieces of one land, fore and aft of
  a hole, share it; a land lost over part of the hull keeps its rank; a piece between two consecutive lands in a band
  of a strake's width - the edge of a loose plank, noise - is a stray, proposed as ``comento B?`` / ``T?`` with a low
  score and renaming nothing; a duplicate is left out); per x, a band between consecutive lands where both are there; the first from the keel line offset by
  ``clinker.DEFAULTS["keel_half_width"]``, the last to the model's edge. Names ``fascia B1..`` (port) and ``fascia
  T1..`` (starboard) from the keel, ``comento B1..`` for the lands; a strake broken by a hole keeps its name. The
  bands run under the frames (the outline is not cut by them).

The plugin writes ``chunk.meta["shipwreck"]["proposals"]`` only.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from . import clinker, relief, ship, timbers
from .texts import num, plural, t

DEFAULTS = {"sigma_cells": 1.5, "trend": 0.05, "min_step": 0.008, "low_step": 0.004, "min_seam": 0.5, "gap": 0.3,
            "opening": 0.3, "top_hat": 0.02, "elongation": 3.0, "frame_deg": 30.0, "keel_band": 0.15, "min_area": 0.002,
            "max_step": 0.04, "along_cells": 3.0, "min_frame": 0.5, "min_longitudinal": 1.0}
"""The parameters of the reading, metres unless said.

- ``sigma_cells``: the smoothing of ``|dz/dy|`` across the hull, in cells; ``along_cells``: the smoothing of ``dz/dy``
  along x before the crests, in cells (the lands are long along x: it keeps a seam whole under the noise of a scan).
- ``trend``: the width across, on either side of a cell, of the running median of ``dz/dy`` taken out of it (the
  hull's own slope).
- ``min_step``: the step a seam must reach somewhere; ``low_step``: the step it keeps along x; ``max_step``: the
  largest step of a land (the side of a timber, a hole's edge is higher).
- ``min_seam``: the least length of a seam; ``gap``: the longest gap along x across which two pieces of a seam are
  joined.
- ``opening``: the length of the lines of the directional openings (longer than a frame's room, shorter than the
  planking between two frames); ``top_hat``: the least height of a timber above the opened surface.
- ``elongation``: the least ratio of the long to the short side of a timber; ``frame_deg``: the largest angle of a
  frame's long axis from y (and of a longitudinal's from x); ``min_frame``, ``min_longitudinal``: the least length of
  a frame (or of a piece of one) and of a longitudinal timber (a loose timber lying on the hull is shorter: ``other``).
- ``keel_band``: how far from y 0 a longitudinal is the keelson; ``min_area``: the least area of a timber, square
  metres.

Limits measured on the synthetic hull at 5 mm cells (lands of 1.5 cm): frames and keelson unaffected by any noise
tried; the seams stay whole under 2 mm of white noise on the heights; with 2 mm of noise correlated over a few cells
they break into pieces, which the strakes still take as pieces of their lands (:func:`lands`: same names, widths
within 2 %); with 4 mm of white noise stray pieces beyond the last land can add a strake (B6) and should be rejected.
A remnant of a land whose neighbours run on as consecutive lands with a band of a strake's width over the same x is
taken for a stray (``comento B?``): accept it by hand and the strake beside it splits in two there."""

KINDS = ("seam", "frame", "keelson", "stringer", "plank", "other")
COLLINEAR = 0.02  # metres: two pieces of a seam continue each other within this across
LINK_CELLS = 1.5  # a crest continues a line from the column before within this many cells across (at least 1 cm)...
PREDICT = 10  # ... of the course of its last crests
SIDE_MASK_CELLS = 1  # crests this many cells from a longitudinal timber are its side, not a land
SIMPLIFY = 0.002  # metres: the tolerance of the lines kept
SMOOTH_ALONG = 5  # cells: the running mean along x of a seam's y
OVERLAP = 0.5  # a new item more than this inside an accepted record or a rejected geometry is not proposed
LINE_BUFFER = 0.02  # metres around a line when it is compared with others


# ----------------------------------------------------------------------------- the grid in the ship plan
class _Grid:
    """The height raster in the ship plan: ``z`` rows along y (increasing), columns along x; ``x``, ``y`` the cells'
    centres; ``cell`` metres."""

    def __init__(self, z, transform, sign: int):
        if abs(transform.b) > 1e-12 or abs(transform.d) > 1e-12 or not np.isclose(abs(transform.a), abs(transform.e)):
            raise ValueError("the relief's grid must be square and not rotated")
        self.cell = float(abs(transform.a))
        rows, cols = z.shape
        self.x = transform.c + transform.a * (np.arange(cols) + 0.5)
        v = transform.f + transform.e * (np.arange(rows) + 0.5)
        y = sign * v
        order = np.argsort(y)
        self.y = y[order]
        self.z = np.asarray(z, float)[order]
        self.ok = np.isfinite(self.z)

    def affine(self):
        """The affine of the grid in the ship plan (x, y): rows along increasing y."""
        from rasterio.transform import Affine

        return Affine(self.cell, 0.0, self.x[0] - self.cell / 2.0, 0.0, self.cell, self.y[0] - self.cell / 2.0)

    def cells(self, metres: float, odd: bool = True) -> int:
        n = max(1, int(round(metres / self.cell)))
        return n + 1 if odd and n % 2 == 0 else n

    def edges(self) -> dict:
        """The model's outer edge per side: ``{side: (x, y)}``, the outermost cell's outer side in each column."""
        out = {}
        cols = np.flatnonzero(self.ok.any(axis=0))
        if not len(cols):
            return out
        ok = self.ok[:, cols]
        top = self.y[len(self.y) - 1 - np.argmax(ok[::-1], axis=0)] + self.cell / 2.0
        bottom = self.y[np.argmax(ok, axis=0)] - self.cell / 2.0
        out["port"] = (self.x[cols], np.maximum(top, 0.0))
        out["starboard"] = (self.x[cols], np.minimum(bottom, 0.0))
        return out


def _params(params) -> dict:
    out = dict(DEFAULTS)
    if params:
        unknown = set(params) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown parameters: {', '.join(sorted(unknown))}")
        out.update({k: float(v) for k, v in params.items()})
    return out


def _side_of(y: float) -> str:
    if abs(y) <= timbers.SIDE_TOLERANCE:
        return "centre"
    return "port" if y > 0 else "starboard"


# ----------------------------------------------------------------------------- directional top-hats
def _fill_along(z: np.ndarray, axis: int) -> np.ndarray:
    """``z`` with the NaN cells given the nearest finite value along ``axis`` (the rows or columns without any: the
    lowest height of the raster)."""
    a = np.moveaxis(z, axis, -1)
    ok = np.isfinite(a)
    n = a.shape[-1]
    idx = np.arange(n)
    before = np.maximum.accumulate(np.where(ok, idx, -1), axis=-1)
    after = np.minimum.accumulate(np.where(ok, idx, n)[..., ::-1], axis=-1)[..., ::-1]
    use_after = (before < 0) | ((after < n) & (after - idx < idx - before))
    src = np.where(use_after, after, before)
    low = np.nanmin(z) if np.isfinite(z).any() else 0.0
    filled = np.take_along_axis(a, np.clip(src, 0, n - 1), axis=-1)
    filled = np.where((src < 0) | (src >= n), low, filled)
    return np.moveaxis(filled, -1, axis)


def _top_hat(grid: _Grid, length: float, axis: int) -> np.ndarray:
    """The height above its grey opening by a line of ``length`` along ``axis`` (1: x, 0: y); NaN off the model."""
    from scipy import ndimage

    n = grid.cells(length)
    size = (1, n) if axis == 1 else (n, 1)
    filled = _fill_along(grid.z, axis)
    # the raster's border carried on for a whole line first: the hull rising to its rim one cell from the border would
    # else be opened as a ridge (a reflection mirrors it, the border mode of each pass is too short a plateau)
    pad = [(0, 0), (0, 0)]
    pad[axis] = (n, n)
    opened = ndimage.grey_opening(np.pad(filled, pad, mode="edge"), size=size)
    residue = filled - (opened[:, n:-n] if axis == 1 else opened[n:-n])
    residue[~grid.ok] = np.nan
    return residue


def _components(grid: _Grid, mask: np.ndarray, min_area: float):
    """The connected components of ``mask`` (8 neighbours) from ``min_area``, as shapely polygons in the ship plan with
    their labels and the label image."""
    import shapely
    from rasterio import features
    from scipy import ndimage

    labels, count = ndimage.label(mask, structure=np.ones((3, 3), bool))
    if not count:
        return [], labels
    areas = np.bincount(labels.ravel(), minlength=count + 1) * grid.cell**2
    keep = areas >= min_area
    keep[0] = False
    labels = np.where(keep[labels], labels, 0).astype(np.int32)
    parts: dict[int, list] = {}
    for geom, value in features.shapes(labels, mask=labels > 0, transform=grid.affine(), connectivity=8):
        parts.setdefault(int(value), []).append(shapely.geometry.shape(geom))
    out = []
    for value, polys in sorted(parts.items()):
        g = shapely.union_all(polys)
        if g.geom_type == "MultiPolygon":
            g = max(g.geoms, key=lambda p: p.area)
        out.append((value, g))
    return out, labels


def _rectangle(poly) -> tuple[float, float, float]:
    """Long side, short side and the long side's angle from x (degrees, 0-90) of the minimum rotated rectangle."""
    import shapely

    with np.errstate(divide="ignore", invalid="ignore"):  # GEOS on a polygon of cells with sides along the axes
        c = np.asarray(shapely.minimum_rotated_rectangle(poly).exterior.coords)
    e1, e2 = c[1] - c[0], c[2] - c[1]
    a, b = float(np.hypot(*e1)), float(np.hypot(*e2))
    long_edge = e1 if a >= b else e2
    angle = float(np.degrees(np.arctan2(abs(long_edge[1]), abs(long_edge[0]))))
    return max(a, b), min(a, b), angle


def _ring(poly) -> list:
    return [[float(x), float(y)] for x, y in np.asarray(poly.exterior.coords)[:-1]]


def _join_frames(pieces: list[dict], room_gap: float, cell: float) -> list[dict]:
    """The pieces of a frame cut by the keelson, one on each side with their x closer than the room and their inner
    ends within ``room_gap`` of each other across y 0, joined: their union and the band between their inner ends."""
    import shapely

    pieces = sorted(pieces, key=lambda p: p["x"])
    used = [False] * len(pieces)
    out = []
    for i, a in enumerate(pieces):
        if used[i]:
            continue
        partner = None
        for j in range(i + 1, len(pieces)):
            b = pieces[j]
            if used[j] or b["x"] - a["x"] >= min(a["room"], b["room"]):
                if b["x"] - a["x"] >= max(a["room"], b["room"]):
                    break
                continue
            ya, yb = a["poly"].bounds, b["poly"].bounds
            lower, upper = (a, b) if ya[1] < yb[1] else (b, a)
            gap = upper["poly"].bounds[1] - lower["poly"].bounds[3]
            if lower["poly"].bounds[3] < 0 < upper["poly"].bounds[1] and gap <= room_gap:
                partner = j
                break
        if partner is None:
            out.append(a)
            continue
        b = pieces[partner]
        used[partner] = True
        reach = max(abs(a["poly"].bounds[3]) if a["poly"].bounds[3] < 0 else abs(a["poly"].bounds[1]),
                    abs(b["poly"].bounds[3]) if b["poly"].bounds[3] < 0 else abs(b["poly"].bounds[1])) + 2.0 * cell
        band = shapely.box(-1e9, -reach, 1e9, reach)
        inner = shapely.union_all([a["poly"].intersection(band), b["poly"].intersection(band)])
        poly = shapely.union_all([a["poly"], b["poly"], inner.convex_hull])
        if poly.geom_type != "Polygon":
            poly = shapely.union_all([a["poly"], b["poly"]]).convex_hull
        _long, short_side, _angle = _rectangle(poly)
        out.append({**a, "poly": poly, "room": short_side, "x": float(poly.centroid.x), "joined": True,
                    "score": max(a["score"], b["score"])})
    return out


def _union_others(others: list[dict]) -> list[dict]:
    """The ``other`` components that touch each other as one (a loose timber leaves a residue in both openings)."""
    import shapely

    groups: list[list[dict]] = []
    for o in others:
        hit = [g for g in groups if any(o["poly"].intersects(q["poly"]) for q in g)]
        for g in hit:
            groups.remove(g)
        groups.append(sum(hit, []) + [o])
    out = []
    for g in groups:
        if len(g) == 1:
            out.append(g[0])
            continue
        poly = shapely.union_all([q["poly"] for q in g])
        if poly.geom_type != "Polygon":
            poly = poly.convex_hull
        _long, short, _angle = _rectangle(poly)
        out.append({"poly": poly, "room": short, "x": float(poly.centroid.x), "height": max(q["height"] for q in g),
                    "score": max(q["score"] for q in g), "length": _long, "kind": "other"})
    return out


def ridges(z, transform, sign: int, params=None) -> list[dict]:
    """The frames and longitudinal timbers on the height raster ``z`` (relief (u, v) with ``transform``; v = ``sign``
    * y): dicts with ``kind`` (frame, keelson, stringer, other), ``side``, ``ship_xy`` (the polygon's ring in the ship
    plan), ``room`` (the short side of its minimum rotated rectangle), ``x`` (its centroid's), ``height`` (the median
    residue, metres) and ``score``."""
    p = _params(params)
    grid = _Grid(z, transform, sign)
    if not grid.ok.any():
        return []
    across = _top_hat(grid, p["opening"], axis=1)  # narrow along x: frames
    along = _top_hat(grid, p["opening"], axis=0)  # narrow along y: longitudinals
    out, pieces, others = [], [], []
    for residue, direction in ((across, "x"), (along, "y")):
        mask = np.nan_to_num(residue, nan=0.0) > p["top_hat"]
        comps, labels = _components(grid, mask, p["min_area"])
        for value, poly in comps:
            long_side, short_side, angle = _rectangle(poly)
            elongated = short_side > 0 and long_side / short_side >= p["elongation"]
            height = float(np.median(residue[labels == value]))
            score = float(min(1.0, height / (3.0 * p["top_hat"])))
            item = {"poly": poly, "room": short_side, "x": float(poly.centroid.x), "height": height, "score": score,
                    "length": long_side}
            if direction == "x" and elongated and angle >= 90.0 - p["frame_deg"] and long_side >= p["min_frame"]:
                pieces.append({**item, "kind": "frame"})
            elif direction == "y" and elongated and angle <= p["frame_deg"] and long_side >= p["min_longitudinal"]:
                kind = "keelson" if abs(poly.centroid.y) <= p["keel_band"] else "stringer"
                out.append({**item, "kind": kind})
            else:
                others.append({**item, "kind": "other"})
    out = _join_frames(pieces, 2.0 * p["keel_band"], grid.cell) + out + _union_others(others)
    result = []
    for r in out:
        side = "centre" if r.get("joined") or r["kind"] == "keelson" else _side_of(float(r["poly"].centroid.y))
        result.append({"kind": r["kind"], "side": side, "ship_xy": _ring(r["poly"]), "room": float(r["room"]),
                       "x": float(r["x"]), "height": float(r["height"]), "score": float(r["score"])})
    return sorted(result, key=lambda r: (KINDS.index(r["kind"]), r["x"]))


# ----------------------------------------------------------------------------- seams
def _nan_smooth(a: np.ndarray, sigma: float, axis: int) -> np.ndarray:
    from scipy import ndimage

    ok = np.isfinite(a)
    num_ = ndimage.gaussian_filter1d(np.where(ok, a, 0.0), sigma, axis=axis, mode="constant")
    den = ndimage.gaussian_filter1d(ok.astype(float), sigma, axis=axis, mode="constant")
    out = num_ / np.maximum(den, 1e-12)
    out[~ok] = np.nan
    return out


def _trend(g: np.ndarray, size: int) -> np.ndarray:
    """The hull's own slope across: the running median of ``g`` over ``size`` cells along y, ``trend`` on either side
    of the cell (a median, not a mean, and over both sides: the central differences spread a land's step over four
    cells, which would pull a mean - or a median - of the five cells of ``trend`` alone, and the step integrated on
    the rest would come out a third of the true one); the cells off the model take the nearest value first."""
    from scipy import ndimage

    out = ndimage.median_filter(_fill_along(g, 0), size=(max(3, size), 1), mode="nearest")
    out[~np.isfinite(g)] = np.nan
    return out


def _crests(grid: _Grid, p: dict, exclude: np.ndarray):
    """The crests of the transverse step in each column: arrays of column, sub-cell y, step (metres)."""
    g = np.gradient(grid.z, grid.cell, axis=0)
    g[~np.isfinite(g) | exclude] = np.nan  # a timber's side would fill the trend's window and the smoothing
    if p["along_cells"] > 0:  # the lands run along x: the noise of a scan does not
        g = _nan_smooth(g, p["along_cells"], axis=1)
    r = g - _trend(g, 2 * grid.cells(p["trend"], odd=False) + 1)
    s = _nan_smooth(np.abs(r), p["sigma_cells"], axis=0)
    s0 = np.nan_to_num(s, nan=-1.0)
    up = np.r_[np.full((1, s0.shape[1]), -1.0), s0[:-1]]
    down = np.r_[s0[1:], np.full((1, s0.shape[1]), -1.0)]
    peak = (s0 > 0) & (s0 >= up) & (s0 > down) & (up >= 0) & (down >= 0)
    # the step of a crest: r integrated over the run of cells around it where r keeps the crest's sign
    sgn = np.sign(np.nan_to_num(r, nan=0.0)).astype(np.int8)
    change = np.ones_like(sgn, bool)
    change[1:] = sgn[1:] != sgn[:-1]
    runs = np.cumsum(change.ravel(order="F")).reshape(sgn.shape, order="F")
    sums = np.bincount(runs.ravel(), weights=np.nan_to_num(r, nan=0.0).ravel())
    step = np.abs(sums[runs]) * grid.cell
    rows, cols = np.nonzero(peak)
    a, b, c = up[rows, cols], s0[rows, cols], down[rows, cols]
    denom = a - 2.0 * b + c
    shift = np.where(np.abs(denom) > 1e-12, 0.5 * (a - c) / np.where(np.abs(denom) > 1e-12, denom, 1.0), 0.0)
    y = grid.y[rows] + np.clip(shift, -0.5, 0.5) * grid.cell
    st = step[rows, cols]
    keep = (st >= p["low_step"]) & (st <= p["max_step"])
    cols, y, st = cols[keep], y[keep], st[keep]
    if len(cols) < 2:
        return cols, y, st
    # two crests of one side closer than ``trend`` share the window of the trend and cannot be told apart (the ends of a
    # hull, where the strakes are a few centimetres wide seen from above): neither is a land
    order = np.lexsort((y, np.sign(y), cols))
    cols, y, st = cols[order], y[order], st[order]
    near = (np.diff(cols) == 0) & (np.sign(y[1:]) == np.sign(y[:-1])) & (np.diff(y) < p["trend"])
    crowded = np.r_[near, False] | np.r_[False, near]
    return cols[~crowded], y[~crowded], st[~crowded]


def _follow(grid: _Grid, cols, ys, steps, p: dict) -> list[dict]:
    """Crests linked along x into pieces: a crest continues the line of its side whose last crest is in the column
    before (or the one before that) and whose course - a straight line through its last :data:`PREDICT` crests - passes
    within :data:`LINK_CELLS` cells (at least 1 cm) of it, the closest pairs first (where the lands come close together
    at the ends of the hull, a line goes on along its own course, not to the nearest crest); pieces whose step never
    reaches ``min_step`` are dropped (the hysteresis: ``low_step`` keeps a line, ``min_step`` starts it)."""
    tol = max(LINK_CELLS * grid.cell, 0.01)
    order = np.lexsort((ys, cols))
    cols, ys, steps = cols[order], ys[order], steps[order]
    active: list[dict] = []
    done: list[dict] = []
    bounds = np.searchsorted(cols, np.arange(len(grid.x) + 1))
    for j in range(len(grid.x)):
        lo, hi = bounds[j], bounds[j + 1]
        done += [tr for tr in active if j - tr["cols"][-1] > 2]
        active = [tr for tr in active if j - tr["cols"][-1] <= 2]
        pairs = []
        for i, tr in enumerate(active):
            c, y = np.asarray(tr["cols"][-PREDICT:], float), np.asarray(tr["ys"][-PREDICT:])
            ahead = y[-1] if len(c) < 3 else float(np.polyval(np.polyfit(c, y, 1), j))
            for k in range(lo, hi):
                d = abs(ahead - ys[k])
                if d <= tol and (y[-1] > 0) == (ys[k] > 0):
                    pairs.append((d, i, k))
        used_tracks, used_crests = set(), set()
        for _d, i, k in sorted(pairs):
            if i in used_tracks or k in used_crests:
                continue
            used_tracks.add(i)
            used_crests.add(k)
            active[i]["cols"].append(j)
            active[i]["ys"].append(ys[k])
            active[i]["steps"].append(steps[k])
        for k in range(lo, hi):
            if k not in used_crests:
                active.append({"cols": [j], "ys": [ys[k]], "steps": [steps[k]]})
    done += active
    out = []
    for tr in done:
        st = np.asarray(tr["steps"])
        if st.max() < p["min_step"] or len(tr["cols"]) < 2:
            continue
        out.append({"x": grid.x[np.asarray(tr["cols"])], "y": np.asarray(tr["ys"]), "step": st})
    return out


def _tail(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """End x, end y and the slope of the last 10 cm of a piece (0 with fewer than three crests there)."""
    tail = x >= x[-1] - 0.1
    slope = float(np.polyfit(x[tail], y[tail], 1)[0]) if tail.sum() >= 3 else 0.0
    return float(x[-1]), float(y[-1]), slope


def _merge(pieces: list[dict], p: dict) -> list[dict]:
    """Pieces of one seam merged, in one pass along x: a piece continues the line of its side that ended less than
    ``gap`` before it starts, when that line carried on (the slope of its last 10 cm) meets it within
    :data:`COLLINEAR` (the closest such line)."""
    chains: list[dict] = []
    for q in sorted(pieces, key=lambda q: q["x"][0]):
        port = bool(q["y"].mean() > 0)
        best, miss_best = None, COLLINEAR
        for c in chains:
            if c["port"] != port:
                continue
            x_end, y_end, slope = c["tail"]
            dx = q["x"][0] - x_end
            if not 0 < dx < p["gap"]:
                continue
            miss = abs(q["y"][0] - (y_end + slope * dx))
            if miss <= miss_best:
                best, miss_best = c, miss
        if best is None:
            chains.append({"port": port, "parts": [q], "tail": _tail(q["x"], q["y"])})
        else:
            best["parts"].append(q)
            best["tail"] = _tail(q["x"], q["y"]) if len(q["x"]) >= 3 else (float(q["x"][-1]), float(q["y"][-1]),
                                                                           best["tail"][2])
    return [{k: np.concatenate([q[k] for q in c["parts"]]) for k in ("x", "y", "step")} for c in chains]


def _line(x: np.ndarray, y: np.ndarray) -> list:
    import shapely

    if len(y) >= SMOOTH_ALONG:
        kernel = np.ones(SMOOTH_ALONG) / SMOOTH_ALONG
        pad = SMOOTH_ALONG // 2
        y = np.convolve(np.pad(y, pad, mode="edge"), kernel, mode="valid")
    line = shapely.LineString(np.column_stack([x, y])).simplify(SIMPLIFY)
    return [[float(a), float(b)] for a, b in line.coords]


def seams(z, transform, sign: int, params=None, exclude=None) -> list[dict]:
    """The seams of the planking on the height raster ``z`` (relief (u, v) with ``transform``; v = ``sign`` * y): dicts
    with ``side``, ``ship_xy`` (the line along x in the ship plan), ``step_mm`` (the median step), ``length`` and
    ``score`` (the share of the line whose step reaches ``min_step``). ``exclude``: a boolean grid in the ship plan
    (rows along increasing y, as :class:`_Grid`) of cells that are not lands (a longitudinal timber's sides); by
    default the cells within :data:`SIDE_MASK_CELLS` of a longitudinal top-hat residue over ``top_hat``."""
    from scipy import ndimage

    p = _params(params)
    grid = _Grid(z, transform, sign)
    if not grid.ok.any():
        return []
    if exclude is None:
        along = np.nan_to_num(_top_hat(grid, p["opening"], axis=0), nan=0.0) > p["top_hat"]
        exclude = ndimage.binary_dilation(along, structure=np.ones((2 * SIDE_MASK_CELLS + 1, 1), bool))
    cols, ys, steps = _crests(grid, p, exclude)
    out = []
    for piece in _merge(_follow(grid, cols, ys, steps, p), p):
        length = float(np.hypot(np.diff(piece["x"]), np.diff(piece["y"])).sum())
        if length < p["min_seam"]:
            continue
        out.append({"side": "port" if piece["y"].mean() > 0 else "starboard", "ship_xy": _line(piece["x"], piece["y"]),
                    "step_mm": float(np.median(piece["step"]) * 1000.0), "length": length,
                    "score": float((piece["step"] >= p["min_step"]).mean())})
    return sorted(out, key=lambda s: (s["side"], float(np.mean(np.abs(np.asarray(s["ship_xy"])[:, 1])))))


# ----------------------------------------------------------------------------- strakes
def _y_on(line: np.ndarray, xs: np.ndarray) -> np.ndarray:
    order = np.argsort(line[:, 0])
    return np.interp(xs, line[order, 0], line[order, 1])


LAND_SAMPLE = 0.02  # metres along x between the points where the ranks of the seams are read
CONFLICT = 0.1  # metres: two pieces of one land overlapping along x more than this are one too many
BAND_STEP = 0.01  # metres along x between the points of a strake's outline (simplified after)


def _longest_path(nodes: list[int], edges: dict) -> dict:
    """1 + the longest chain of "nearer the keel" edges ``{(i, j): weight}`` (i nearer than j) ending at each node; a
    cycle (pieces that cross each other) is broken at its weakest edge."""
    edges = dict(edges)
    while True:
        incoming = {n: 0 for n in nodes}
        after: dict[int, list[int]] = {n: [] for n in nodes}
        for i, j in edges:
            incoming[j] += 1
            after[i].append(j)
        rank = {n: 1 for n in nodes}
        ready = [n for n in nodes if incoming[n] == 0]
        done = 0
        while ready:
            i = ready.pop()
            done += 1
            for j in after[i]:
                rank[j] = max(rank[j], rank[i] + 1)
                incoming[j] -= 1
                if incoming[j] == 0:
                    ready.append(j)
        if done == len(nodes):
            return rank
        stuck = {n for n in nodes if incoming[n] > 0}
        weakest = min((e for e in edges if e[0] in stuck and e[1] in stuck), key=lambda e: edges[e])
        del edges[weakest]


STRAY_SCORE = 0.1  # the score of a stray seam piece proposed as ``comento ?``
WIDE_BAND = 1.5  # a band this many times as wide as the side's others there has lost a land: a piece in it is one


def _wide_band(i: int, lo: int, hi: int, placed: list[int], rank: dict, ys: dict) -> bool:
    """Whether the band between the lands ``lo`` and ``hi`` is at least :data:`WIDE_BAND` times as wide as the other
    bands between consecutive lands of the side, where the piece ``i`` is (medians over its samples); True when there
    is no other band to compare with."""
    where = np.isfinite(ys[i])

    def land(r):
        y = np.full(len(where), np.nan)
        for j in placed:
            if rank[j] == r:
                y = np.where(np.isfinite(y), y, ys[j])
        return np.where(where, y, np.nan)

    present = sorted({rank[j] for j in placed})
    lines = {r: land(r) for r in present}
    band = lines[hi] - lines[lo]
    others = [lines[b] - lines[a] for a, b in zip(present, present[1:]) if b == a + 1 and (a, b) != (lo, hi)]
    ref = np.concatenate([o[np.isfinite(o)] for o in others]) if others else np.array([])
    band = band[np.isfinite(band)]
    if not len(ref) or not len(band):
        return True
    return float(np.median(band)) >= WIDE_BAND * float(np.median(ref))


def lands(seam_list: list[dict], params=None) -> list[int | None]:
    """The land of each seam: its rank from the keel on its side, 1 the innermost; 0 for a stray (proposed as
    ``comento ?``), None for a duplicate (left out).

    The pieces of a side are compared two by two where they overlap along x (sampled every :data:`LAND_SAMPLE`): one
    within :data:`COLLINEAR` of a longer one over more than :data:`CONFLICT` is a duplicate; otherwise the one nearer
    the keel there lies inside the other. The pieces are then placed one by one, the longest first, among those
    already placed, ranked by the longest chain of placed pieces inside them (a land lost over part of the hull - under
    sediment, a broken plank - keeps the rank of its pieces, and so do the lands beyond it). A piece is placed when it
    lies inside the innermost land, beyond the outermost, between two lands that are not consecutive, or between two
    consecutive lands whose band is at least :data:`WIDE_BAND` times as wide there as the other bands of the side (a
    land lost over the rest of the hull, its remnant here). Otherwise - between two consecutive lands, in a band of a
    strake's width - it is a stray: the edge of a loose plank lying along the hull, a stray of noise, but also the
    remnant of a land whose neighbours run on as consecutive lands elsewhere with a band of normal width; it is
    proposed as ``comento ?`` with a low score and never counts in the ranks, so it renames nothing. Pieces of one rank
    are one land (fore and aft of a hole, broken by noise). A stray in a free place (a piece of noise at an end of the
    hull where the innermost land is not seen) takes that place's rank."""
    _params(params)  # checked: the reading has no parameter of its own
    out: list[int | None] = [None] * len(seam_list)  # None: a duplicate
    for side in ("port", "starboard"):
        live = [i for i, sm in enumerate(seam_list) if sm["side"] == side]
        if not live:
            continue
        lines = {i: np.asarray(seam_list[i]["ship_xy"], float) for i in live}
        span = {i: (float(lines[i][:, 0].min()), float(lines[i][:, 0].max())) for i in live}
        x0, x1 = min(v[0] for v in span.values()), max(v[1] for v in span.values())
        xs = np.linspace(x0, x1, max(2, int(round((x1 - x0) / LAND_SAMPLE)) + 1))
        ys = {}
        for i in live:
            inside = (xs >= span[i][0]) & (xs <= span[i][1])
            ys[i] = np.where(inside, np.abs(_y_on(lines[i], xs)), np.nan)
        order = sorted(live, key=lambda i: (span[i][0], span[i][1]))
        relation = {}  # (i, j) -> median of |y_j| - |y_i| on their overlap
        for k, i in enumerate(order):
            for j in order[k + 1:]:
                if span[j][0] > span[i][1]:
                    break
                both = np.isfinite(ys[i]) & np.isfinite(ys[j])
                if both.sum() >= 3:
                    d = ys[j][both] - ys[i][both]
                    relation[(i, j)] = (float(np.median(d)), float(np.median(np.abs(d))),
                                        float(xs[both][-1] - xs[both][0]))
        length = {i: span[i][1] - span[i][0] for i in live}
        kept: list[int] = []
        for i in sorted(live, key=lambda i: -length[i]):
            duplicate = any(r[1] < COLLINEAR and r[2] > CONFLICT
                            for j in kept for r in [relation.get((i, j)) or relation.get((j, i))] if r)
            if not duplicate:
                kept.append(i)

        def nearer(i, j):  # +1 when i lies nearer the keel than j where they overlap, -1 farther, 0 not comparable
            if (i, j) in relation:
                return 1 if relation[(i, j)][0] > 0 else -1
            if (j, i) in relation:
                return -1 if relation[(j, i)][0] > 0 else 1
            return 0

        rank: dict[int, int] = {}
        placed: list[int] = []
        edges: dict = {}
        for i in sorted(kept, key=lambda i: -length[i]):
            inner = [rank[j] for j in placed if nearer(j, i) > 0]
            outer = [rank[j] for j in placed if nearer(j, i) < 0]
            lo, hi = max(inner, default=0), min(outer, default=None)
            if hi is not None and lo >= 1 and hi <= lo + 1 and not _wide_band(i, lo, hi, placed, rank, ys):
                out[i] = 0
                continue
            for j in placed:
                r = relation.get((i, j)) or relation.get((j, i))
                if r is not None:
                    edges[(i, j) if nearer(i, j) > 0 else (j, i)] = abs(r[0])
            placed.append(i)
            rank = _longest_path(placed, edges)
        for i, r in rank.items():
            out[i] = r
    return out


def _land_lines(seam_list, ranks, side, xs, gap):
    """``{rank: y at xs}`` of one side's lands (NaN where no piece of the land is), gaps shorter than ``gap`` bridged."""
    out = {}
    for i, (sm, r) in enumerate(zip(seam_list, ranks)):
        if not r or sm["side"] != side:  # a duplicate or a stray
            continue
        line = np.asarray(sm["ship_xy"], float)
        y = out.setdefault(r, np.full(len(xs), np.nan))
        inside = (xs >= line[:, 0].min()) & (xs <= line[:, 0].max())
        y[inside] = _y_on(line, xs[inside])
    for y in out.values():
        ok = np.flatnonzero(np.isfinite(y))
        for a, b in zip(ok[:-1], ok[1:]):
            if b - a > 1 and xs[b] - xs[a] < gap:
                y[a + 1:b] = np.interp(xs[a + 1:b], [xs[a], xs[b]], [y[a], y[b]])
    return out


def _runs(valid: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[0, valid.astype(np.int8), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _clean_ring(ring: np.ndarray):
    """A band's outline simplified, the largest part when it folds on itself; None when nothing is left."""
    import shapely

    poly = shapely.Polygon(ring)
    if not poly.is_valid:
        parts = [g for g in getattr(shapely.make_valid(poly), "geoms", [shapely.make_valid(poly)])
                 if g.geom_type == "Polygon"]
        if not parts:
            return None
        poly = max(parts, key=lambda g: g.area)
    poly = poly.simplify(SIMPLIFY)
    return _ring(poly) if poly.geom_type == "Polygon" and poly.area > 0 else None


def strakes(seam_list: list[dict], sign: int, params=None, edges: dict | None = None) -> list[dict]:
    """The strakes between the lands (:func:`lands`): per side and per x, the band between land k - 1 and land k
    where both are there (``fascia B``k / ``T``k from the keel, whatever pieces the lands are made of: a strake broken by
    a hole keeps its name fore and aft of it, two items of one name), the first from the keel line offset by
    ``clinker.DEFAULTS["keel_half_width"]`` (``edge`` "keel line"), the last from the outermost land to the model's
    edge when ``edges`` gives it (``{side: (x, y)}``, :meth:`_Grid.edges`; ``edge`` "model"). Gaps of a land shorter
    than ``gap`` are bridged. ``sign`` is the frame's; the seams are already in the ship plan. Dicts with ``side``,
    ``name``, ``ship_xy`` (the ring), ``seams`` (the indexes in ``seam_list`` of the pieces bounding it) and ``edge``."""
    p = _params(params)
    half = clinker.DEFAULTS["keel_half_width"]
    word = t("fascia", "strake")
    ranks = lands(seam_list, p)
    out = []
    for side, s_side in (("port", 1.0), ("starboard", -1.0)):
        mine = [i for i, (sm, r) in enumerate(zip(seam_list, ranks)) if sm["side"] == side and r]
        if not mine:
            continue
        x0 = min(float(np.min(np.asarray(seam_list[i]["ship_xy"])[:, 0])) for i in mine)
        x1 = max(float(np.max(np.asarray(seam_list[i]["ship_xy"])[:, 0])) for i in mine)
        xs = np.linspace(x0, x1, max(2, int(round((x1 - x0) / BAND_STEP)) + 1))
        land = _land_lines(seam_list, ranks, side, xs, p["gap"])
        top = max(land)
        land[0] = np.full(len(xs), s_side * half)
        if edges and side in edges:
            ex, ey = edges[side]
            land[top + 1] = np.where((xs >= ex.min()) & (xs <= ex.max()), np.interp(xs, ex, ey), np.nan)
        for k in range(1, top + 2):
            inner, outer = land.get(k - 1), land.get(k)
            if inner is None or outer is None:
                continue
            for a, b in _runs(np.isfinite(inner) & np.isfinite(outer)):
                if b - a < 2:
                    continue
                ring = _clean_ring(np.vstack([np.column_stack([xs[a:b], inner[a:b]]),
                                              np.column_stack([xs[a:b][::-1], outer[a:b][::-1]])]))
                if ring is None:
                    continue
                lo, hi = xs[a], xs[b - 1]
                bounding = [i for i in mine if ranks[i] in (k - 1, k) and
                            np.asarray(seam_list[i]["ship_xy"])[:, 0].min() <= hi and
                            np.asarray(seam_list[i]["ship_xy"])[:, 0].max() >= lo]
                out.append({"side": side, "name": f"{word} {clinker._strake_name(side, k)}", "ship_xy": ring,
                            "seams": bounding, "edge": "keel line" if k == 1 else "model" if k == top + 1 else "seam"})
    return out


# ----------------------------------------------------------------------------- building the proposals
def _geometry(xy, line: bool):
    import shapely

    xy = np.asarray(xy, float)
    if line:
        return shapely.LineString(xy)
    return shapely.Polygon(xy) if len(xy) >= 3 else shapely.LineString(xy)


def _taken(chunk, frame: ship.Frame) -> list:
    """The geometries no proposal may cover again: the accepted records (``ship_xy``, else their geometry in the
    chunk's coordinates through the frame) and the rejected ones."""
    import shapely

    out = []
    for r in timbers.timbers_of(chunk):
        xy = r.get("ship_xy")
        geom = r.get("geometry") or {}
        if xy is None and geom.get("xy"):
            w = np.asarray(geom["xy"], float)
            xy = frame.to_ship(np.column_stack([w, np.zeros(len(w))]))[:, :2]
        if xy is None or geom.get("type") == "point" or len(xy) < 2:
            continue
        out.append(_geometry(xy, geom.get("type") == "line" or len(xy) < 3))
    for text in (ship.state(chunk).get("proposals") or {}).get("rejected", []):
        try:
            out.append(shapely.from_wkt(text))
        except Exception:  # noqa: BLE001 - a broken entry blocks nothing
            continue
    return out


def _covered(item: dict, taken: list) -> bool:
    """Whether more than :data:`OVERLAP` of the item (its area, or its length for a seam) lies inside an accepted
    record or a rejected geometry (lines widened by :data:`LINE_BUFFER`)."""
    is_line = item["kind"] == "seam"
    g = _geometry(item["ship_xy"], is_line)
    size = g.length if is_line else g.area
    if size <= 0:
        return False
    for other in taken:
        zone = other.buffer(LINE_BUFFER) if (is_line or other.geom_type in ("LineString", "MultiLineString")) else other
        if not zone.is_valid:
            zone = zone.buffer(0)
        part = g.intersection(zone)
        if (part.length if is_line else part.area) > OVERLAP * size:
            return True
    return False


def frame_space(frames: list[dict]) -> float:
    """The median space of the frames, read on each side by itself (``x`` of dicts with ``side``; a frame on the
    centre counts on both): mixed, the two pieces of one frame a little apart would make the space their distance."""
    diffs: list[float] = []
    for side in ("port", "starboard"):
        xs = sorted(f["x"] for f in frames if f["side"] in (side, "centre"))
        diffs += np.diff(xs).tolist()
    return float(np.median(diffs)) if diffs else 0.0


def _name_frames(frames: list[dict], bow_x: float) -> None:
    """``ordinata N`` from the bow; frames on different sides within half the space (:func:`frame_space`) share N."""
    if not frames:
        return
    order = sorted(frames, key=lambda f: abs(f["x"] - bow_x))
    space = frame_space(frames)
    word = t("ordinata", "frame")
    groups: list[list[dict]] = []
    for f in order:
        group = next((g for g in groups if abs(g[0]["x"] - f["x"]) < space / 2.0 and
                      all(o["side"] != f["side"] and "centre" not in (o["side"], f["side"]) for o in g)), None)
        if group is None:
            groups.append([f])
        else:
            group.append(f)
    for n, group in enumerate(groups, 1):
        for f in group:
            f["name"] = f"{word} {n}"


def build(chunk, params=None, progress=None) -> dict:
    """Read the proposals on the relief of the hull: ``state["proposals"] = {"items": [...], "rejected": [WKT],
    "params", "built", "next"}``, each item ``{"id": "P1", "kind", "side", "name", "ship_xy", "score", "step_mm",
    "room"}``. An item covering an accepted record or a rejected geometry (:func:`_covered`) is not proposed; the
    catalogue's records and the rejected list are never changed; the ids go on from the previous reading. Returns
    the proposals."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    p = _params(params)
    frame = ship.need_frame(chunk)
    if relief.entry(chunk) is None:
        raise ship.NotReady(t("Costruisci prima il rilievo dello scafo.", "Build the relief of the hull first."))
    if relief.is_stale(chunk):
        raise ship.NotReady(t("Il rilievo dello scafo è di un altro sistema della nave o di un altro modello: "
                              "costruiscilo di nuovo.",
                              "The relief of the hull is of another ship frame or another model: build it again."))
    progress(0.0, t("Proposte dal rilievo dello scafo", "Proposals from the relief of the hull"))
    z, transform = relief.load(chunk, "height")
    grid = _Grid(z, transform, frame.sign)
    progress(0.1, t("Ordinate e legni longitudinali", "Frames and longitudinal timbers"))
    found = ridges(z, transform, frame.sign, p)
    progress(0.5, t("Gradini del fasciame", "Lands of the planking"))
    seam_list = seams(z, transform, frame.sign, p)
    progress(0.8, t("Fasce", "Strakes"))
    bands = strakes(seam_list, frame.sign, p, edges=grid.edges())
    words = {"keelson": t("paramezzale", "keelson"), "stringer": t("serretta", "stringer"),
             "other": t("altro", "other"), "seam": t("comento", "seam")}
    items: list[dict] = []
    for s, rank in zip(seam_list, lands(seam_list, p)):
        if rank is None:  # a duplicate of a land already there
            continue
        code = clinker._strake_name(s["side"], rank) if rank else f"{clinker.SIDE_CODES[s['side']][0]}?"
        score = s["score"] if rank else min(s["score"], STRAY_SCORE)  # a stray: proposed, unnamed, doubtful
        items.append({"kind": "seam", "side": s["side"], "name": f"{words['seam']} {code}", "ship_xy": s["ship_xy"],
                      "score": score, "step_mm": s["step_mm"], "room": None})
    frames = [dict(r) for r in found if r["kind"] == "frame"]
    _name_frames(frames, frame.bow_x)
    counters: dict[str, int] = {}
    keelsons = sum(1 for r in found if r["kind"] == "keelson")
    for r in frames + [r for r in found if r["kind"] != "frame"]:
        if r["kind"] == "frame":
            name = r["name"]
        else:
            counters[r["kind"]] = counters.get(r["kind"], 0) + 1
            numbered = r["kind"] != "keelson" or keelsons > 1  # one keelson is the keelson; pieces of it are numbered
            name = f"{words[r['kind']]} {counters[r['kind']]}" if numbered else words[r["kind"]]
        items.append({"kind": r["kind"], "side": r["side"], "name": name, "ship_xy": r["ship_xy"], "score": r["score"],
                      "step_mm": None, "room": r["room"]})
    for b in bands:
        steps = [seam_list[i]["step_mm"] for i in b["seams"]]
        scores = [seam_list[i]["score"] for i in b["seams"]]
        items.append({"kind": "plank", "side": b["side"], "name": b["name"], "ship_xy": b["ship_xy"],
                      "score": float(min(scores)) if scores else 0.0,
                      "step_mm": float(np.median(steps)) if steps else None, "room": None})
    taken = _taken(chunk, frame)
    st = ship.state(chunk)
    previous = st.get("proposals") or {}
    n = int(previous.get("next", 1))
    kept = []
    for item in items:
        if taken and _covered(item, taken):
            continue
        kept.append({"id": f"P{n}", **item})
        n += 1
    st["proposals"] = {"items": kept, "rejected": list(previous.get("rejected", [])), "params": p,
                       "built": datetime.now(timezone.utc).isoformat(timespec="seconds"), "next": n}
    progress(1.0, summary(chunk))
    return st["proposals"]


def summary(chunk) -> str:
    """A line on the proposals: how many of each kind; "nessun gradino di fasciame trovato" without seams."""
    prop = (chunk.meta.get(ship.KEY) or {}).get("proposals")
    if not prop:
        return t("Nessuna proposta: leggi prima il rilievo dello scafo.", "No proposals: read the relief of the hull first.")
    count = {k: sum(1 for i in prop["items"] if i["kind"] == k) for k in KINDS}
    named = {}  # a frame in two pieces (one name on both sides) is one frame
    for i in prop["items"]:
        if i["kind"] == "frame":
            named.setdefault(i["name"], []).append(float(np.mean(np.asarray(i["ship_xy"])[:, 0])))
    count["frame"] = len(named)
    parts = []
    if count["seam"]:
        parts.append(plural(count["seam"], ("comento", "comenti"), ("seam", "seams")))
    else:
        parts.append(t("nessun gradino di fasciame trovato", "no land of the planking found"))
    for kind, it, en in (("plank", ("fascia", "fasce"), ("strake", "strakes")),
                         ("frame", ("ordinata", "ordinate"), ("frame", "frames")),
                         ("keelson", ("paramezzale", "paramezzali"), ("keelson", "keelsons")),
                         ("stringer", ("serretta", "serrette"), ("stringer", "stringers")),
                         ("other", ("altro elemento", "altri elementi"), ("other item", "other items"))):
        if count[kind]:
            parts.append(plural(count[kind], it, en))
    text = t("Proposte: ", "Proposals: ") + ", ".join(parts)
    xs = sorted(float(np.mean(v)) for v in named.values())
    if len(xs) > 1:
        space = num(float(np.median(np.diff(xs))), 2)
        text += t(f" (passo mediano delle ordinate {space} m)", f" (median frame space {space} m)")
    return text + "."
