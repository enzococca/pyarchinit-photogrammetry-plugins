"""The faired hull (``ricostruito: carenatura``): a surface lofted through the stations' sections where neither side of
the wreck exists, a stated hypothesis like the rest of the reconstruction.

- At each station (:func:`lines.sections_of`, in order of x) the half-section of each side is the model's cut on
  that side (the RAW lines), less the points lying on a timber over the planking (the catalogue's frames, keelson,
  stringers, ceiling, any other solid, buffered by :data:`COVER` - so it is the hull's own surface), ordered from the
  keel to the sheer (the highest point), with the keel point in front: y = 0 at the height of the keel line
  (:func:`solids._keel_line`: the keelson's bottom, else the lowest of the relief across it; the lowest point of the
  section where that has nothing). Resampled at ``samples`` points evenly along its arc, keel to sheer. A side with
  fewer than :data:`MIN_POINTS` points takes the other side's reflected; a station with neither is left out.
- Per side and per sample a natural cubic spline along x (y and z), evaluated every ``step`` from the first to the
  last station; the two sides joined at the keel (the keel point is shared).
- Ends: a stem or stern given (:func:`reconstruct.set_ends`; a traced line: its x on the keel line) past the last
  station on its side carries the last section on to it, the breadth tapered to zero by a quarter ellipse
  (y x sqrt(1 - ((x - x_last) / (x_end - x_last))^2), z kept); one inside the stations' range stops the loft there.
  Without one the loft stops at the last station and that end is listed in ``open_ends`` ("stem", "stern").
- Faces kept only where the wreck shows nothing: the relief's cell under the face's centre has no value (the empty
  cells opened by :data:`reconstruct.OPENING` cells, as the mirror reads them) and no mirrored timber lies there.

An open surface: ``volume`` 0, status ``reconstructed:fairing``, name ``carenatura``, kind ``hull``, written to
``<chunk data>/shipwreck/reconstruction/carenatura.ply``; ``state["reconstruction"]["loft"]`` says what it was made
from (``file``, ``open_ends``, ``stations``, ``faces``, ``area`` in plan, ``length`` along x)."""
from __future__ import annotations

import numpy as np

from . import lines, relief, ship, solids, tracing
from .ship import NotReady
from .texts import t

STATUS = "reconstructed:fairing"
NAME = "carenatura"
KIND = "hull"
MIN_POINTS = 5  # a half-section with fewer points is missing: the other side's taken
COVER = 0.01  # m: the timbers over the planking buffered by this when their points are left out of a section
KEEL_SIDING = 0.10  # m: the band across the keel line where the keel's height is read without a keelson


def _cover(chunk, frame):
    """The catalogue's timbers lying over the planking (what is not a strake or a line), buffered by :data:`COVER`."""
    import shapely

    polys = [solids._polygon(tracing.ship_xy_of(chunk, r, frame)) for r in ship.state(chunk).get("timbers", [])
             if r["kind"] in solids.SOLIDS and r["geometry"]["type"] == "polygon"]
    return shapely.union_all(polys).buffer(COVER) if polys else None


def _half(polylines, sign: float, cover, x: float) -> np.ndarray | None:
    """The points (|y|, z) of one side of a station's cut, keel to sheer, without those on a covering timber."""
    import shapely

    runs = []
    for p in polylines:
        p = np.asarray(p, float).reshape(-1, 2)
        keep = sign * p[:, 0] > 0
        if cover is not None and keep.any():
            keep &= ~shapely.contains_xy(cover, np.full(len(p), x), p[:, 0])
        idx = np.flatnonzero(keep)
        if not len(idx):
            continue
        for seg in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            q = np.column_stack([np.abs(p[seg, 0]), p[seg, 1]])
            runs.append(q[::-1] if q[0, 0] > q[-1, 0] else q)
    if not runs:
        return None
    runs.sort(key=lambda q: float(q[:, 0].min()))
    pts = np.vstack(runs)
    pts = pts[: int(np.argmax(pts[:, 1])) + 1]  # up to the sheer
    return pts if len(pts) >= MIN_POINTS else None


def _resample(pts: np.ndarray, keel_z: float, samples: int) -> np.ndarray:
    """``samples`` points (|y|, z) evenly along the arc of the keel point (0, ``keel_z``) and ``pts``."""
    pts = np.vstack([[0.0, keel_z], pts[pts[:, 0] > 1e-9]])
    step = np.hypot(*np.diff(pts, axis=0).T)
    pts = pts[np.r_[True, step > 1e-12]]
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    at = np.linspace(0.0, s[-1], samples)
    return np.column_stack([np.interp(at, s, pts[:, 0]), np.interp(at, s, pts[:, 1])])


def _keel_heights(chunk, xs: np.ndarray, halves) -> np.ndarray:
    """The keel line's height at the stations: :func:`solids._keel_line`, else the lowest point of the section."""
    lowest = np.array([min(float(h[:, 1].min()) for h in pair if h is not None) for pair in halves])
    keel = (ship.state(chunk).get("keel") or {}).get("siding") or KEEL_SIDING
    try:
        z = solids._keel_line(chunk, xs, float(keel))
    except NotReady:
        return lowest
    return np.where(np.isfinite(z), z, lowest)


def _end_x(e: dict) -> float:
    if "x" in e:
        return float(e["x"])
    line = np.asarray(e["line"], float)
    order = np.argsort(line[:, 1])
    return float(np.interp(0.0, line[order, 1], line[order, 0]))


def _ends(chunk, frame, lo: float, hi: float):
    """``{-1: (name, x) | None, +1: ...}``: the end given at the smaller and at the larger x, and the names of the two
    ends (the one given, else the bow's end of the keel line is the stem)."""
    from . import reconstruct

    given = {}
    for key, e in (reconstruct.state(chunk).get("ends") or {}).items():
        x = _end_x(e)
        side = 1 if x > 0.5 * (lo + hi) else -1
        if side in given:
            raise NotReady(t("Prua e poppa stanno dalla stessa parte del relitto.",
                             "Stem and stern lie on the same side of the wreck."))
        given[side] = (key, x)
    names = {}
    for side in (-1, 1):
        if side in given:
            names[side] = given[side][0]
        elif -side in given:
            names[side] = "stern" if given[-side][0] == "stem" else "stem"
        else:
            bow_low = abs(frame.bow_x - lo) <= abs(frame.bow_x - hi)
            names[side] = ("stem" if bow_low else "stern") if side < 0 else ("stern" if bow_low else "stem")
    return given, names


def _lost(chunk, frame):
    """A test of plan points: the relief's cell under them has no value (opened as the mirror does) or lies off it."""
    from scipy import ndimage

    from . import reconstruct

    z, a = relief.load(chunk, "height")
    k = reconstruct.OPENING + 1
    lost = ndimage.binary_opening(~np.isfinite(z), structure=np.ones((k, k), bool), border_value=1)
    inv = ~a
    sign = frame.sign

    def test(x, y):
        u, v = np.asarray(x, float), sign * np.asarray(y, float)
        col = np.floor(inv.a * u + inv.b * v + inv.c).astype(np.int64)
        row = np.floor(inv.d * u + inv.e * v + inv.f).astype(np.int64)
        out = np.ones(len(u), bool)
        ok = (row >= 0) & (row < lost.shape[0]) & (col >= 0) & (col < lost.shape[1])
        out[ok] = lost[row[ok], col[ok]]
        return out

    return test


def _cross2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


def _taper(x_last: float, x_end: float, step: float):
    """The columns past the last station to an end, and the factor of the breadth at each (a quarter ellipse)."""
    side = 1.0 if x_end > x_last else -1.0
    n = max(1, int(np.ceil(abs(x_end - x_last) / step)))
    xs = x_last + side * np.minimum(np.arange(1, n + 1) * step, abs(x_end - x_last))
    xs[-1] = x_end
    r = (xs - x_last) / (x_end - x_last)
    return xs, np.sqrt(np.clip(1.0 - r**2, 0.0, 1.0))


def loft(chunk, step: float = 0.10, samples: int = 41) -> solids.Solid | None:
    """The faired hull where neither side of the wreck exists (see the module); None (and no file) when fewer than
    two stations have a half-section, or nothing is left where the wreck shows nothing."""
    import shapely
    from scipy.interpolate import CubicSpline

    from . import reconstruct

    frame = ship.need_frame(chunk)
    if relief.entry(chunk) is None:
        raise NotReady(t("Costruisci prima il rilievo dello scafo.", "Build the relief of the hull first."))
    tracing.need_fresh_relief(chunk)
    if step <= 0 or samples < 3:
        raise ValueError("step > 0 and samples >= 3")
    path = reconstruct.folder(chunk) / reconstruct.LOFT_FILE
    st = reconstruct.state(chunk)
    cover = _cover(chunk, frame)
    sections = sorted(lines.sections_of(chunk, "station"), key=lambda s: float(s["value"]))
    xs, halves = [], []
    for s in sections:
        x = float(s["value"])
        if xs and abs(x - xs[-1]) < 1e-9:
            continue
        pair = (_half(s.get("lines") or [], 1.0, cover, x), _half(s.get("lines") or [], -1.0, cover, x))
        if pair[0] is None and pair[1] is None:
            continue
        xs.append(x)
        halves.append(pair)

    def nothing(reason):
        path.unlink(missing_ok=True)
        st["loft"] = {"file": None, "open_ends": [], "stations": xs, "faces": 0, "area": 0.0, "length": 0.0,
                      "reason": reason}
        return None

    if len(xs) < 2:
        return nothing(t("meno di due stazioni con una mezza sezione", "fewer than two stations with a half-section"))
    xs_a = np.asarray(xs)
    keel = _keel_heights(chunk, xs_a, halves)
    port, star = [], []
    for (hp, hs), zk in zip(halves, keel):
        p = _resample(hp if hp is not None else hs, zk, samples)
        q = _resample(hs if hs is not None else hp, zk, samples)
        port.append(p)
        star.append(q)
    data = np.stack([np.stack(port), np.stack(star)], axis=1)  # (stations, side, samples, (|y|, z))
    spline = CubicSpline(xs_a, data, bc_type="natural", axis=0)
    cols = np.unique(np.r_[np.arange(xs_a[0], xs_a[-1], step), xs_a[-1]])
    values = spline(cols)
    values[..., 0] = np.maximum(values[..., 0], 0.0)
    given, names = _ends(chunk, frame, float(xs_a[0]), float(xs_a[-1]))
    open_ends = []
    for side in (-1, 1):
        last = float(cols[0] if side < 0 else cols[-1])
        if side not in given:
            open_ends.append(names[side])
            continue
        x_end = given[side][1]
        if side * (x_end - last) <= 1e-9:  # an end inside the stations' range: the loft stops there
            keep = cols >= x_end if side < 0 else cols <= x_end
            cols, values = cols[keep], values[keep]
            continue
        tx, factor = _taper(last, x_end, step)
        edge = values[0 if side < 0 else -1]
        more = np.repeat(edge[None], len(tx), axis=0)
        more[..., 0] *= factor[:, None, None]
        if side < 0:
            cols, values = np.r_[tx[::-1], cols], np.concatenate([more[::-1], values])
        else:
            cols, values = np.r_[cols, tx], np.concatenate([values, more])
    if len(cols) < 2:
        return nothing(t("le estremità date tolgono tutte le stazioni", "the ends given leave no station"))
    # one row of vertices per column: starboard from the sheer to the keel, then port from the keel to the sheer
    y = np.concatenate([-values[:, 1, :0:-1, 0], values[:, 0, :, 0]], axis=1)
    z = np.concatenate([values[:, 1, :0:-1, 1], values[:, 0, :, 1]], axis=1)
    m, w = y.shape
    X = np.repeat(cols[:, None], w, axis=1)
    vertices = np.stack([X, y, z], axis=-1).reshape(-1, 3)
    i, j = np.meshgrid(np.arange(m - 1), np.arange(w - 1), indexing="ij")
    a = (i * w + j).ravel()
    b, c, d = a + w, a + w + 1, a + 1
    faces = np.vstack([np.column_stack([a, b, c]), np.column_stack([a, c, d])])
    centre = vertices[faces].mean(axis=1)
    keep = _lost(chunk, frame)(centre[:, 0], centre[:, 1])
    mirrored = [solids._polygon(r["ship_xy"]) for r in reconstruct.records_of(chunk)
                if r.get("status") == reconstruct.STATUS]
    if mirrored and keep.any():
        over = shapely.union_all(mirrored)
        keep &= ~shapely.contains_xy(over, centre[:, 0], centre[:, 1])
    faces = faces[keep]
    tri = vertices[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    faces = faces[area > 1e-12]
    if not len(faces):
        return nothing(t("lo scafo è tutto nel rilievo o nella simmetria", "the hull is all in the relief or the mirror"))
    used = np.zeros(len(vertices), bool)
    used[faces.ravel()] = True
    remap = np.cumsum(used) - 1
    vertices, faces = vertices[used], remap[faces].astype(np.int64)
    solid = solids.Solid(vertices, faces, np.zeros(len(faces), bool), STATUS, NAME, KIND, 0.0, None, None)
    solids.write_ply(path, solid, frame)
    plan = vertices[faces][..., :2]
    st["loft"] = {"file": str(path.relative_to(chunk.data_dir)), "open_ends": open_ends,
                  "stations": [float(v) for v in xs_a],
                  "faces": int(len(faces)),
                  "area": float(np.abs(_cross2(plan[:, 1] - plan[:, 0], plan[:, 2] - plan[:, 0])).sum() / 2.0),
                  "length": float(np.ptp(vertices[:, 0]))}
    return solid
