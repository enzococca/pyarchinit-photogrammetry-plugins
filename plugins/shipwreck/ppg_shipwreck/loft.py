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
  cells opened by :data:`reconstruct.OPENING` cells, as the mirror reads them) and no mirrored timber lies there:
  the test is on each triangle's centroid, so the loft keeps to the lost parts to within one face.
- A station whose keel no traced keelson covers, and whose section's lowest point lies more than :data:`KEEL_STEP`
  below the relief across the keel, has an untraced timber standing on the keel: the keel is read from the
  section's lowest point, the points standing over it at the keel are left out, and ``warnings`` names the
  stations (trace the timber). Unless the ridge is the typed keel itself (:func:`_is_keel`: a keel typed, the
  ridge seen over at least :data:`KEEL_RUN` of the stations, no wider than its siding and two cells, the frames
  running over it): the loft follows the planking all the same and ``notes`` says the keel is its own solid.
- Refused (:class:`NotReady`) without station sections, or with sections cut on another frame or model.
- ``records``: the measured records to read (default the catalogue; the merged records of several chunks,
  :func:`chunks.merged`); the relief is the reconstruction's (:func:`reconstruct.height_path`); ``sections``: the
  station sections (default the chunk's; those of all the chunks of the wreck, :func:`chunks.station_sections`).

An open surface: ``volume`` 0, status ``reconstructed:fairing``, name ``carenatura``, kind ``hull``, written to
``<chunk data>/shipwreck/reconstruction/carenatura.ply``; ``state["reconstruction"]["loft"]`` says what it was made
from (``file``, ``open_ends``, ``stations``, ``faces``, ``area`` in plan, ``length`` along x, ``warnings``)."""
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
KEEL_STEP = 0.03  # m: a section this much lower than the relief across the keel: a timber on the keel, not traced
RIDGE_SEEN = 0.01  # m: the relief across the keel this much over the section's lowest point: a ridge on the keel line
KEEL_RUN = 0.8  # share of the stations a ridge runs over to be the keel's run
OVER_FRAME = 0.01  # m: a ridge this much over a frame's top beside it stands over the frames (a keelson)


def _catalogue(chunk, records) -> list[dict]:
    return ship.state(chunk).get("timbers", []) if records is None else records


def _cover(chunk, frame, records=None):
    """The catalogue's timbers lying over the planking (what is not a strake or a line), buffered by :data:`COVER`."""
    import shapely

    polys = [solids._polygon(tracing.ship_xy_of(chunk, r, frame)) for r in _catalogue(chunk, records)
             if r["kind"] in solids.SOLIDS and r["geometry"]["type"] == "polygon"]
    return shapely.union_all(polys).buffer(COVER) if polys else None


def _half(polylines, sign: float, cover, x: float) -> np.ndarray | None:
    """The points (|y|, z) of one side of a station's cut, keel to sheer, without those on a covering timber."""
    import shapely

    runs = []
    for p in polylines:
        p = np.asarray(p, float).reshape(-1, 2)
        keep = (sign * p[:, 0] > 0) & np.isfinite(p).all(axis=1)
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


def _keelson_cover(chunk, frame, records=None):
    """The keelsons traced (the catalogue's and the mirrored), buffered by :data:`COVER`; None without any."""
    import shapely

    from . import reconstruct

    polys = [solids._polygon(tracing.ship_xy_of(chunk, r, frame)) for r in _catalogue(chunk, records)
             if r["kind"] == "keelson" and r["geometry"]["type"] == "polygon"]
    polys += [solids._polygon(r["ship_xy"]) for r in reconstruct.records_of(chunk)
              if r["kind"] == "keelson" and r.get("ship_xy")]
    return shapely.union_all(polys).buffer(COVER) if polys else None


def _keel_heights(chunk, frame, xs: np.ndarray, halves, records=None) -> tuple[np.ndarray, np.ndarray]:
    """The keel line's height at the stations and which stations show an untraced timber over the keel.

    Under a traced keelson: :func:`solids._keel_line` (its bottom). Elsewhere the lower of :func:`solids._keel_line`
    (the lowest relief across the keel) and the section's lowest point; a section's lowest point more than
    :data:`KEEL_STEP` below the relief across the keel there is a timber standing on the keel that is not traced."""
    import shapely

    from . import reconstruct

    lowest = np.array([min(float(h[:, 1].min()) for h in pair if h is not None) for pair in halves])
    siding = (ship.state(chunk).get("keel") or {}).get("siding") or KEEL_SIDING
    keelsons = None if records is None else [r for r in records if r["kind"] == "keelson"
                                             and r["geometry"]["type"] == "polygon"]
    try:
        line = solids._keel_line(chunk, xs, float(siding), path=reconstruct.height_path(chunk), keelsons=keelsons,
                                 records=records)
    except NotReady:
        line = np.full(len(xs), np.nan)
    cover = _keelson_cover(chunk, frame, records)
    covered = (shapely.contains_xy(cover, xs, np.zeros(len(xs))) if cover is not None
               else np.zeros(len(xs), bool))
    z = np.where(covered & np.isfinite(line), line, np.fmin(line, lowest))
    untraced = ~covered & np.isfinite(line) & (lowest < line - KEEL_STEP)
    return z, untraced


def _band_low(chunk, xs: np.ndarray, siding: float) -> np.ndarray:
    """The lowest relief across the keel line, ``siding`` wide, at ``xs`` (NaN where the relief has none)."""
    import warnings as _w

    from . import reconstruct

    grid = solids._Grid(chunk, (float(xs.min()), -siding / 2.0, float(xs.max()), siding / 2.0), margin=1,
                        path=reconstruct.height_path(chunk))
    band = np.abs(grid.y) <= max(siding / 2.0, grid.cell)
    with _w.catch_warnings():
        _w.simplefilter("ignore", RuntimeWarning)
        low = np.nanmin(np.where(band[:, None], grid.z, np.nan), axis=0)
    ok = np.isfinite(low)
    out = np.full(len(xs), np.nan)
    if ok.any():
        near = np.abs(xs[:, None] - grid.x[ok][None]).min(axis=1) <= grid.cell
        out[near] = np.interp(xs[near], grid.x[ok], low[ok])
    return out


def _ridge_width(pair, keel_z: float) -> float:
    """The width across of what stands more than :data:`KEEL_STEP` over the keel at the start of the half-sections
    (0 when nothing does: a timber traced and left out)."""
    half = 0.0
    for pts in pair:
        if pts is None:
            continue
        high = pts[:, 1] > keel_z + KEEL_STEP
        first = int(np.argmin(high)) if not high.all() else len(pts)
        if first:
            half = max(half, float(pts[:first, 0].max()))
    return 2.0 * half


def _over_frames(chunk, frame, half: float, records=None) -> bool:
    """Whether the ridge on the keel line stands over the frames' tops (a keelson over the frames) rather than under
    them (the keel, the frames running over it): at most of the traced frames crossing the keel line."""
    import shapely

    from . import reconstruct

    z, a = reconstruct.height_load(chunk)
    cell = float(abs(a.a))
    band = shapely.box(-1e4, -half - 3 * cell, 1e4, half + 3 * cell)
    votes = []
    for r in _catalogue(chunk, records):
        if r["kind"] != "frame" or r["geometry"]["type"] != "polygon" or r.get("loose"):
            continue
        part = solids._polygon(tracing.ship_xy_of(chunk, r, frame)).intersection(band)
        if part.is_empty or part.area <= 0:
            continue
        ys = np.arange(-half - 3 * cell, half + 3 * cell + 1e-9, cell / 2.0)
        row, col, ok = relief.cells_at(z.shape, a, frame.sign, np.full(len(ys), part.centroid.x), ys)
        vals = np.full(len(ys), np.nan)
        vals[ok] = z[row[ok], col[ok]]
        inner = vals[np.abs(ys) <= half]
        outer = vals[(np.abs(ys) > half + cell) & (np.abs(ys) <= half + 3 * cell)]
        inner, outer = inner[np.isfinite(inner)], outer[np.isfinite(outer)]
        if len(inner) and len(outer):
            votes.append(float(inner.max()) - float(np.median(outer)) > OVER_FRAME)
    return bool(votes) and sum(votes) > len(votes) / 2.0


def _is_keel(chunk, frame, xs, halves, keel, untraced, records=None) -> bool:
    """Whether the ridge the untraced stations show is the typed keel itself: a keel typed (:func:`solids.set_keel`),
    the ridge seen over at least :data:`KEEL_RUN` of the stations, no wider than the siding and two cells, and
    running under the frames, not over them."""
    typed = ship.state(chunk).get("keel") or {}
    siding = typed.get("siding")
    if not siding:
        return False
    lowest = np.array([min(float(h[:, 1].min()) for h in pair if h is not None) for pair in halves])
    try:
        seen = _band_low(chunk, xs, float(siding)) - lowest > RIDGE_SEEN
    except NotReady:
        return False
    if seen.mean() < KEEL_RUN:
        return False
    from . import reconstruct

    cell = float(abs(reconstruct.height_load(chunk)[1].a))
    widths = [_ridge_width(pair, zk) for pair, zk, u in zip(halves, keel, untraced) if u]
    if any(w > float(siding) + 2 * cell for w in widths):
        return False
    return not _over_frames(chunk, frame, float(siding) / 2.0, records)


def _off_the_keel(pts: np.ndarray | None, keel_z: float) -> np.ndarray | None:
    """A half-section without its leading points standing more than :data:`KEEL_STEP` over the keel (an untraced
    timber on it); as it was when too few points would be left."""
    if pts is None:
        return None
    high = pts[:, 1] > keel_z + KEEL_STEP
    first = int(np.argmin(high)) if not high.all() else len(pts)
    rest = pts[first:]
    return rest if len(rest) >= MIN_POINTS else pts


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
    from . import reconstruct

    z, a = reconstruct.height_load(chunk)
    lost = reconstruct.opened_lost(np.isfinite(z))

    def test(x, y):
        row, col, ok = relief.cells_at(lost.shape, a, frame.sign, x, y)
        out = np.ones(len(row), bool)
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


def loft(chunk, step: float = 0.10, samples: int = 41, records=None, sections=None) -> solids.Solid | None:
    """The faired hull where neither side of the wreck exists (see the module); None (and no file) when fewer than
    two stations have a half-section, or nothing is left where the wreck shows nothing. ``sections``: the station
    sections to loft through (default the chunk's own; :func:`chunks.station_sections` for several chunks)."""
    import shapely
    from scipy.interpolate import CubicSpline

    from . import reconstruct

    frame = ship.need_frame(chunk)
    if relief.entry(chunk) is None:
        raise NotReady(t("Costruisci prima il rilievo dello scafo.", "Build the relief of the hull first."))
    tracing.need_fresh_relief(chunk)
    if step <= 0 or samples < 3:
        raise ValueError("step > 0 and samples >= 3")
    data = lines.load(chunk)
    given = sections is not None
    if not given:
        sections = [s for s in (data or {}).get("sections", []) if s["kind"] == "station"]
    if not sections:
        raise NotReady(t("Taglia prima le sezioni alle stazioni.", "Cut the sections at the stations first."))
    if not given and lines.is_stale(chunk, data):
        raise NotReady(t("Le sezioni non sono aggiornate (altro sistema della nave o altro modello): tagliale di "
                         "nuovo.",
                         "The sections are out of date (another ship frame or another model): cut them again."))
    path = reconstruct.folder(chunk) / reconstruct.LOFT_FILE
    st = reconstruct.state(chunk)
    cover = _cover(chunk, frame, records)
    sections = sorted(sections, key=lambda s: float(s["value"]))
    xs, halves, names = [], [], []
    for s in sections:
        x = float(s["value"])
        if xs and abs(x - xs[-1]) < 1e-9:
            continue
        pair = (_half(s.get("lines") or [], 1.0, cover, x), _half(s.get("lines") or [], -1.0, cover, x))
        if pair[0] is None and pair[1] is None:
            continue
        xs.append(x)
        halves.append(pair)
        names.append(s.get("name") or f"x {x:.2f}")
    warnings: list[str] = []
    notes: list[str] = []

    def nothing(reason):
        path.unlink(missing_ok=True)
        st["loft"] = {"file": None, "open_ends": [], "stations": xs, "faces": 0, "area": 0.0, "length": 0.0,
                      "reason": reason, "warnings": warnings, "notes": notes}
        return None

    if len(xs) < 2:
        return nothing(t("meno di due stazioni con una mezza sezione", "fewer than two stations with a half-section"))
    xs_a = np.asarray(xs)
    keel, untraced = _keel_heights(chunk, frame, xs_a, halves, records)
    if untraced.any() and _is_keel(chunk, frame, xs_a, halves, keel, untraced, records):
        notes.append(t("La carenatura segue il fasciame; la chiglia è il suo solido.",
                       "The faired hull follows the planking; the keel is its own solid."))
    elif untraced.any():
        listed = ", ".join(n for n, u in zip(names, untraced) if u)
        warnings.append(t(f"Un legno sopra la chiglia non tracciato alle stazioni {listed}: lì la chiglia è letta dal "
                          "punto più basso della sezione e i punti sopra di essa sono lasciati fuori; traccialo (il "
                          "paramezzale) perché la carenatura non lo segua.",
                          f"A timber over the keel not traced at the stations {listed}: there the keel is read from "
                          "the lowest point of the section and the points over it are left out; trace it (the "
                          "keelson) so that the faired hull does not follow it."))
    if untraced.any():
        halves = [(_off_the_keel(hp, zk), _off_the_keel(hs, zk)) if u else (hp, hs)
                  for (hp, hs), zk, u in zip(halves, keel, untraced)]
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
    # one vertex per place: the two sides meet at the tip of a closed end
    vertices, inverse = np.unique(vertices, axis=0, return_inverse=True)
    faces = inverse.reshape(-1)[faces]
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])]
    solid = solids.Solid(vertices, faces, np.zeros(len(faces), bool), STATUS, NAME, KIND, 0.0, None, None)
    solids.write_ply(path, solid, frame)
    plan = vertices[faces][..., :2]
    st["loft"] = {"file": str(path.relative_to(chunk.data_dir)), "open_ends": open_ends,
                  "stations": [float(v) for v in xs_a],
                  "faces": int(len(faces)),
                  "area": float(np.abs(_cross2(plan[:, 1] - plan[:, 0], plan[:, 2] - plan[:, 0])).sum() / 2.0),
                  "length": float(np.ptp(vertices[:, 0])), "warnings": warnings,
                  "notes": notes}
    return solid
