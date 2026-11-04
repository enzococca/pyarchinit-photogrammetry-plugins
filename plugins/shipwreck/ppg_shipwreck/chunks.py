"""Several chunks of the same wreck feeding one reconstruction.

Other campaigns or other parts of a wreck are often other chunks of the project. They feed the reconstruction of the
MAIN chunk (the one built) when they are COMPATIBLE with it (:func:`compatible`):

- aligned to it (the program's *Flusso di lavoro > Allinea chunk*: ``meta["chunk_alignment"]["reference_key"]`` is
  the main's key, or the main's is theirs, or both point to the same reference) AND with the same ship frame (the
  frame dict: ends, bow, datum, x0 within 1e-6, the same system; :func:`use_frame_of` copies it), or
- with a ship frame taken from markers of the same names (``state["frame_markers"]``, :func:`ship.set_frame`) with
  the same bow, x0 and datum rule, both chunks in metres: the markers are the same physical points, so the ship
  frames coincide by construction even where the chunks were never aligned.

Either way the ship plan of every member is the same, so their records and reliefs are merged there:

- :func:`members`: the main and the compatible chunks with a current relief (the others refused, with the reason),
  each with its relief's ``cell`` and ``coverage`` (the share of the union of all the members' footprints it covers);
- :func:`merged`: the measured, not loose records of all the members grouped by kind, side and name (``fascia B3``
  of one campaign is ``fascia B3`` of the other: the frames must carry the field's numbers, see
  :func:`tracing.rename_frames`), the records of a group that meet (within :data:`TOUCH`) one record: a polygon is the
  union in the ship plan where, over the part the better-measured member surveyed (the share of its polygon on its own
  relief, a tie within :data:`SHARE_TIE` to the finer cell), that member's outline is the one kept; a line (a seam)
  is joined end to end along x, the better member's points where both have some; a point is kept as it is. ``id``:
  the main's record's, else ``<chunk key>:<id>``; ``sources``: ``[{"chunk", "id", "coverage", "cell"}]``, the better
  first. The main's records that are not measured or are loose are passed on as they are;
- :func:`prepare`: the merged height raster on the ship plan at the finest member's cell, each cell from the best
  member covering it (``<main data>/shipwreck/reconstruction/members/height.tif``, ``members.json`` beside it),
  which the reconstruction reads through :func:`reconstruct.height_path` while it is current (:func:`is_current`).

Only the main chunk's folder is written; the members (their state, records, files) are read only."""
from __future__ import annotations

import copy
import functools
import json
import math
from pathlib import Path

import numpy as np

from . import relief, ship, timbers, tracing
from .ship import NotReady
from .texts import t

FOLDER = "members"  # inside the reconstruction's folder
RASTER = "height.tif"
INDEX = "members.json"
SHARE_TIE = 0.02  # shares of a polygon on the relief this close are a tie: the finer cell wins
TOUCH = 0.02  # m: records of one name and kind closer than this, in two chunks, are one timber
JOIN = 0.03  # m: a worse-measured outline goes this far into the better member's survey, so that the two meet
FOOTPRINT_SIDE = 1000  # the footprint of a relief is read at most this many cells a side
SAMPLES = 20000  # at most this many points of a record read on a relief for its share
EPS = 1e-6


def _st(chunk) -> dict:
    """The plugin's part of a chunk's meta, never created (the members are read only)."""
    return chunk.meta.get(ship.KEY) or {}


# ----------------------------------------------------------------------------- compatibility
def _reference(chunk):
    return (chunk.meta.get("chunk_alignment") or {}).get("reference_key")


def _aligned(main, other) -> bool:
    ro, rm = _reference(other), _reference(main)
    return ro == main.key or rm == other.key or (ro is not None and ro == rm)


def _close(a, b) -> bool:
    return np.allclose(np.asarray(a, float), np.asarray(b, float), rtol=0.0, atol=EPS)


def _same_frame(a: dict, b: dict) -> bool:
    return (a.get("bow", "start") == b.get("bow", "start") and a.get("crs") == b.get("crs")
            and _close(a["start"], b["start"]) and _close(a["end"], b["end"])
            and _close([a.get("z0"), a.get("x0", 0.0)], [b.get("z0"), b.get("x0", 0.0)]))


def _default_datum(d: dict) -> bool:
    return d.get("z0") is None or abs(float(d["z0"]) - float(d["start"][2])) < EPS


def _same_markers(main, other, a: dict, b: dict, ma, mb) -> bool:
    """Frames from markers of the same names: the same physical points, the same bow, x0 and datum rule, metres."""
    if not ma or not mb or list(ma) != list(mb):
        return False
    if a.get("bow", "start") != b.get("bow", "start") or abs(float(a.get("x0", 0.0)) - float(b.get("x0", 0.0))) > EPS:
        return False
    if not (_default_datum(a) and _default_datum(b)) and abs(float(a["z0"]) - float(b["z0"])) > EPS:
        return False
    return ship.is_scaled(main) and ship.is_scaled(other)


def _not_aligned(main, other) -> str:
    return t(f"Allinea prima il chunk «{other.label}» a «{main.label}» (Flusso di lavoro > Allinea chunk) o definisci il "
             "sistema della nave con gli stessi marker",
             f"Align chunk «{other.label}» to «{main.label}» first (Workflow > Align chunks) or define the ship's "
             "system with the same markers")


def _compatible(main, other, frame: dict | None, markers) -> tuple[bool, str]:
    mine = _st(main).get("frame")
    if mine is None:
        return False, t(f"Imposta prima il sistema della nave del chunk «{main.label}».",
                        f"Set the ship frame of chunk «{main.label}» first.")
    if frame is not None:
        if _aligned(main, other) and _same_frame(mine, frame):
            return True, ""
        if _same_markers(main, other, mine, frame, _st(main).get("frame_markers"), markers):
            return True, ""
    if _aligned(main, other):
        return False, t(f"Il chunk «{other.label}» è allineato a «{main.label}» ma non ha il suo sistema della nave: "
                        f"usa il sistema della nave di «{main.label}».",
                        f"Chunk «{other.label}» is aligned to «{main.label}» but has not its ship frame: use the ship "
                        f"frame of «{main.label}».")
    return False, _not_aligned(main, other)


def compatible(main, other) -> tuple[bool, str]:
    """Whether ``other`` can feed the reconstruction of ``main`` (see the module): ``(True, "")``, else ``(False,
    the reason for the user)``."""
    st = _st(other)
    return _compatible(main, other, st.get("frame"), st.get("frame_markers"))


def _has_markers(chunk, names) -> bool:
    return all(any(str(n) in (m.label, str(m.key)) and m.position is not None for m in chunk.markers) for n in names)


def use_frame_of(target, source) -> dict:
    """The ship frame of ``source`` for ``target``: copied when the two are aligned (the same world), else taken from
    ``target``'s own markers of the names ``source``'s frame came from (the same bow, x0 and datum rule); the frame's
    markers and the stations go with it, and the sections ``source`` has are cut again on ``target``'s model.
    :class:`NotReady` with :func:`compatible`'s reason when neither works (nothing changed). Returns the frame dict."""
    src = _st(source).get("frame")
    if src is None:
        raise NotReady(t(f"Imposta prima il sistema della nave del chunk «{source.label}».",
                         f"Set the ship frame of chunk «{source.label}» first."))
    markers = _st(source).get("frame_markers")
    mine = markers if markers and _has_markers(target, markers) else None
    if _aligned(source, target):
        frame = ship.Frame.from_dict(src)
        candidate = {**frame.to_dict(), "crs": src.get("crs", source.crs)}
    elif mine:
        z0 = None if _default_datum(src) else float(src["z0"])
        frame = ship.frame_from_markers(target, mine[0], mine[1], src.get("bow", "start"), z0, src.get("x0", 0.0))
        candidate = {**frame.to_dict(), "crs": target.crs}
    else:
        raise NotReady(_not_aligned(source, target))
    ok, reason = _compatible(source, target, candidate, mine)
    if ok and candidate.get("crs") != target.crs:
        ok, reason = False, _not_aligned(source, target)
    if not ok:
        raise NotReady(reason)
    st = ship.state(target)
    saved = {k: copy.deepcopy(st[k]) for k in ("frame", "frame_markers", "stations") if k in st}
    try:
        ship.set_frame(target, frame, markers=mine)
        stations = ship.stations_of(source)
        if stations:
            ship.set_stations(target, stations)
        _cut_sections(target, source)
    except BaseException:
        for k in ("frame", "frame_markers", "stations"):
            st.pop(k, None)
        st.update(saved)
        raise
    return st["frame"]


def _cut_sections(target, source) -> None:
    """The sections ``source`` has (stations, longitudinals, horizontals) cut again on ``target``'s model."""
    from . import lines

    data = lines.load(source)
    path = target.product_path("model")
    if not data or path is None or not Path(path).exists():
        return
    sections = data.get("sections") or []
    longitudinals = [float(s["value"]) for s in sections if s["kind"] == "longitudinal"]
    horizontals = [float(s["value"]) for s in sections if s["kind"] == "horizontal"]
    if ship.stations_of(target) or longitudinals or horizontals:
        lines.build(target, longitudinals, horizontals)


# ----------------------------------------------------------------------------- the members
def _cell(chunk) -> float:
    import rasterio

    with rasterio.open(relief.path_of(chunk, "height")) as src:
        return float(abs(src.transform.a))


def _footprint(chunk):
    """The relief's cells with a value as a polygon of the ship plan (read at most :data:`FOOTPRINT_SIDE` a side)."""
    import rasterio
    import shapely
    from rasterio import features
    from rasterio.enums import Resampling

    sign = ship.need_frame(chunk).sign
    with rasterio.open(relief.path_of(chunk, "height")) as src:
        step = max(1, int(math.ceil(max(src.width, src.height) / FOOTPRINT_SIDE)))
        shape = (max(1, int(math.ceil(src.height / step))), max(1, int(math.ceil(src.width / step))))
        z = src.read(1, out_shape=shape, resampling=Resampling.nearest).astype(np.float64)
        if src.nodata is not None and np.isfinite(src.nodata):
            z[z == src.nodata] = np.nan
        a = src.transform @ src.transform.scale(src.width / shape[1], src.height / shape[0])
    mask = np.isfinite(z)
    if not mask.any():
        return shapely.Polygon()
    shapes = [shapely.geometry.shape(g) for g, v in features.shapes(mask.astype("uint8"), mask=mask, transform=a)
              if v == 1]
    return shapely.transform(shapely.union_all(shapes), lambda p: relief.ship_of_uv(p, sign))


def members(doc, main) -> tuple[list[dict], list[dict]]:
    """The chunks feeding the reconstruction of ``main``: ``[{"chunk", "cell", "coverage"}]``, the main first, then the
    compatible chunks of ``doc`` with a current relief, in the project's order; and the others refused,
    ``[{"chunk", "reason"}]`` (not compatible, no relief, a relief of another frame or model, disabled)."""
    import shapely

    ship.need_frame(main)
    if relief.entry(main) is None:
        raise NotReady(t("Costruisci prima il rilievo dello scafo.", "Build the relief of the hull first."))
    found, refused = [main], []
    for c in doc.chunks:
        if c is main or c.key == main.key:
            continue
        ok, reason = compatible(main, c)
        if not ok:
            refused.append({"chunk": c, "reason": reason})
        elif not getattr(c, "enabled", True):
            refused.append({"chunk": c, "reason": t(f"Il chunk «{c.label}» è disattivato.",
                                                    f"Chunk «{c.label}» is disabled.")})
        elif relief.entry(c) is None:
            refused.append({"chunk": c, "reason": t(f"Il chunk «{c.label}» non ha il rilievo dello scafo: costruiscilo.",
                                                    f"Chunk «{c.label}» has no relief of the hull: build it.")})
        elif relief.is_stale(c):
            refused.append({"chunk": c, "reason": t(f"Il rilievo del chunk «{c.label}» è di un altro sistema della nave "
                                                    "o di un altro modello: costruiscilo di nuovo.",
                                                    f"The relief of chunk «{c.label}» is of another ship frame or "
                                                    "another model: build it again.")})
        else:
            found.append(c)
    out = [{"chunk": c, "cell": _cell(c), "coverage": 1.0} for c in found]
    if len(out) > 1:
        prints = [_footprint(c) for c in found]
        total = shapely.union_all(prints).area
        for m, fp in zip(out, prints):
            m["coverage"] = float(fp.area / total) if total > 0 else 0.0
    return out, refused


def _better(a: dict, b: dict, share_a: float, share_b: float, order) -> int:
    """-1 when ``a`` is the better measured: the higher share (beyond :data:`SHARE_TIE`), else the finer cell, else
    the earlier member."""
    if abs(share_a - share_b) > SHARE_TIE:
        return -1 if share_a > share_b else 1
    if abs(a["cell"] - b["cell"]) > 1e-12:
        return -1 if a["cell"] < b["cell"] else 1
    return -1 if order[a["chunk"].key] <= order[b["chunk"].key] else 1


def ranked(found: list[dict]) -> list[dict]:
    """The members best first: the coverage (a tie within :data:`SHARE_TIE` to the finer cell)."""
    order = {m["chunk"].key: i for i, m in enumerate(found)}
    return sorted(found, key=functools.cmp_to_key(lambda a, b: _better(a, b, a["coverage"], b["coverage"], order)))


# ----------------------------------------------------------------------------- the merged records
class _Survey:
    """A member's relief, read once: the share of a record on it, its footprint."""

    def __init__(self, member: dict):
        self.member = member
        self.chunk = member["chunk"]
        self.cell = member["cell"]
        self.sign = ship.need_frame(self.chunk).sign
        self._z = self._fp = None

    @property
    def z(self):
        if self._z is None:
            self._z = relief.load(self.chunk, "height")
        return self._z

    def footprint(self):
        if self._fp is None:
            self._fp = _footprint(self.chunk)
        return self._fp

    def share(self, gtype: str, xy: np.ndarray) -> float:
        """The share of a record's points (a polygon's cells, a line's run, a point) on the relief's cells with a value."""
        import shapely

        z, a = self.z
        if gtype == "polygon":
            poly = _polygon(xy)
            if poly.is_empty or poly.area <= 0:
                return 0.0
            step = max(self.cell, math.sqrt(poly.area / SAMPLES))
            x0, y0, x1, y1 = poly.bounds
            gx, gy = np.meshgrid(np.arange(x0 + step / 2.0, x1, step), np.arange(y0 + step / 2.0, y1, step))
            inside = shapely.contains_xy(poly, gx, gy)
            px, py = gx[inside], gy[inside]
            if not len(px):
                px, py = np.array([poly.representative_point().x]), np.array([poly.representative_point().y])
        else:
            pts = _densify(xy, self.cell) if gtype == "line" and len(xy) > 1 else xy
            px, py = pts[:, 0], pts[:, 1]
        row, col, ok = relief.cells_at(z.shape, a, self.sign, px, py)
        value = np.zeros(len(px), bool)
        value[ok] = np.isfinite(z[row[ok], col[ok]])
        return float(value.mean()) if len(value) else 0.0


def _polygon(xy):
    import shapely

    poly = shapely.Polygon(np.asarray(xy, float).reshape(-1, 2))
    return poly if poly.is_valid else poly.buffer(0)


def _densify(xy: np.ndarray, step: float) -> np.ndarray:
    xy = np.asarray(xy, float).reshape(-1, 2)
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    if s[-1] <= 0:
        return xy[:1]
    at = np.linspace(0.0, s[-1], min(SAMPLES, max(2, int(math.ceil(s[-1] / step)) + 1)))
    return np.column_stack([np.interp(at, s, xy[:, 0]), np.interp(at, s, xy[:, 1])])


def _shape(gtype: str, xy):
    import shapely

    if gtype == "polygon":
        return _polygon(xy)
    if gtype == "line" and len(xy) > 1:
        return shapely.LineString(xy)
    return shapely.Point(xy[0])


def _clusters(entries: list[dict]) -> list[list[dict]]:
    """The entries of one group joined where records of two chunks meet (within :data:`TOUCH`); a point is alone."""
    n = len(entries)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            a, b = entries[i], entries[j]
            if a["survey"].chunk.key == b["survey"].chunk.key or "point" in (a["type"], b["type"]):
                continue
            if a["shape"].distance(b["shape"]) <= TOUCH:
                parent[find(i)] = find(j)
    groups: dict[int, list[dict]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(entries[i])
    return list(groups.values())


def _one_polygon(geom, cell: float, warnings: list, name: str):
    """A single polygon: the pieces closed over a small gap, else the largest (said in ``warnings``)."""
    import shapely

    if geom.geom_type == "Polygon":
        return geom
    for c in (cell, 2.0 * cell, 0.01, 0.02, 0.05):
        closed = geom.buffer(c, join_style="mitre").buffer(-c, join_style="mitre")
        if closed.geom_type == "Polygon" and not closed.is_empty:
            return closed
    pieces = [g for g in getattr(geom, "geoms", [geom]) if g.geom_type == "Polygon" and not g.is_empty]
    warnings.append(t(f"{name}: le parti dei chunk non si toccano; tenuta la più grande.",
                      f"{name}: the parts of the chunks do not meet; the largest is kept."))
    return shapely.geometry.polygon.orient(max(pieces, key=lambda g: g.area), 1.0)


def _merge_polygons(ranked_entries: list[dict], warnings: list, name: str) -> np.ndarray:
    import shapely

    from . import reconstruct

    result = ranked_entries[0]["shape"]
    surveyed = ranked_entries[0]["survey"].footprint().buffer(-JOIN)
    for e in ranked_entries[1:]:
        part = e["shape"].difference(surveyed) if not surveyed.is_empty else e["shape"]
        result = shapely.union_all([result, part])
        surveyed = shapely.union_all([surveyed, e["survey"].footprint().buffer(-JOIN)])
    cell = max(e["survey"].cell for e in ranked_entries)
    poly = _one_polygon(result, cell, warnings, name)
    return reconstruct._ring(poly)


def _merge_lines(ranked_entries: list[dict]) -> np.ndarray:
    """The lines joined end to end along x: the better's points over its x range, the others' beyond."""
    taken: list[tuple[float, float]] = []
    pts = []
    for e in ranked_entries:
        xy = e["xy"]
        free = np.ones(len(xy), bool)
        for a, b in taken:
            free &= ~((xy[:, 0] >= a - EPS) & (xy[:, 0] <= b + EPS))
        pts.append(xy[free])
        taken.append((float(xy[:, 0].min()), float(xy[:, 0].max())))
    out = np.vstack(pts)
    return out[np.argsort(out[:, 0], kind="stable")]


def merged(doc, main, found: list[dict] | None = None) -> list[dict]:
    """The records of all the members merged in the ship plan (see the module); ``found``: :func:`members`' first list
    (read again when None). Nothing of any chunk is written."""
    from . import reconstruct

    found = members(doc, main)[0] if found is None else found
    frame = ship.need_frame(main)
    key = relief.frame_key(frame)
    order = {m["chunk"].key: i for i, m in enumerate(found)}
    groups: dict[tuple, list[dict]] = {}
    passed = []
    for m in found:
        survey = _Survey(m)
        ch = m["chunk"]
        fr = ship.need_frame(ch)
        for r in copy.deepcopy(timbers.timbers_of(ch)):
            if not timbers.is_measured(r) or r.get("loose"):
                if ch is main:
                    passed.append(r)
                continue
            gtype = r["geometry"]["type"]
            xy = tracing.ship_xy_of(ch, r, fr)
            if gtype == "polygon" and _polygon(xy).is_empty:
                continue
            groups.setdefault((r["kind"], r["side"], reconstruct._base_name(r["name"])), []).append(
                {"record": r, "survey": survey, "type": gtype, "xy": xy, "shape": _shape(gtype, xy),
                 "share": survey.share(gtype, xy)})
    out, warnings = [], []
    for entries in groups.values():
        for cluster in _clusters(entries):
            best = sorted(cluster, key=functools.cmp_to_key(
                lambda a, b: _better(a["survey"].member, b["survey"].member, a["share"], b["share"], order)))
            own = [e for e in cluster if e["survey"].chunk is main]
            base = (own or best)[0]
            rec = copy.deepcopy(base["record"])
            single = len(cluster) == 1
            gtype = base["type"]
            if single:
                xy = base["xy"]
            elif gtype == "polygon":
                xy = _merge_polygons(best, warnings, rec.get("name") or rec["id"])
            elif gtype == "line":
                xy = _merge_lines(best)
            else:
                xy = base["xy"]
            if not (single and own):
                for k in ("solid", "volume", "visible_xy"):
                    rec.pop(k, None)
            if not own:
                rec["id"] = f"{base['survey'].chunk.key}:{base['record']['id']}"
            rec.update({"ship_xy": np.asarray(xy, float).tolist(), "frame_key": key, "chunk": main.key,
                        "geometry": timbers.geometry(gtype, frame.to_world(np.column_stack([xy, np.zeros(len(xy))]))[:, :2]),
                        "sources": [{"chunk": e["survey"].chunk.key, "id": e["record"]["id"],
                                     "coverage": round(float(e["share"]), 4), "cell": e["survey"].cell} for e in best]})
            out.append(rec)
    return out + passed


# ----------------------------------------------------------------------------- the merged relief
def relief_key(chunk) -> dict | None:
    """What a member's relief is now: its elevation, the frame it was built in, its file (None without one)."""
    e = relief.entry(chunk)
    if e is None:
        return None
    path = relief.path_of(chunk, "height")
    if not path.exists():
        return None
    st = path.stat()
    return {"elevation": e.get("id"), "frame": relief._frame_key(_st(chunk)["relief"].get("frame")),
            "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def is_current(main, entry: dict) -> bool:
    """Whether the merged raster of ``entry`` (``state["reconstruction"]["members"]``) still holds: every member in
    the project, compatible, its relief the one recorded."""
    doc = getattr(main, "document", None)
    if doc is None:
        return False
    by_key = {c.key: c for c in doc.chunks}
    for m in entry.get("chunks") or []:
        ch = by_key.get(m["chunk"])
        if ch is None or relief_key(ch) != m.get("relief"):
            return False
        if ch is not main and not compatible(main, ch)[0]:
            return False
    return True


def _grid(found: list[dict]):
    """The merged grid on the ship plan (u, v): the union of the members' rasters at the finest cell, on whole cells."""
    import rasterio
    from rasterio.transform import Affine

    bounds = []
    for m in found:
        with rasterio.open(relief.path_of(m["chunk"], "height")) as src:
            a = src.transform
            if abs(a.b) > 1e-12 or abs(a.d) > 1e-12 or a.e >= 0:
                raise NotReady(t("Il rilievo di un chunk non è sulla griglia della nave.",
                                 "The relief of a chunk is not on the ship grid."))
            bounds.append((a.c, a.f + a.e * src.height, a.c + a.a * src.width, a.f))
    b = np.asarray(bounds)
    cell = min(m["cell"] for m in found)
    u0 = math.floor(b[:, 0].min() / cell + 1e-6) * cell
    v1 = math.ceil(b[:, 3].max() / cell - 1e-6) * cell
    w = int(math.ceil((b[:, 2].max() - u0) / cell - 1e-6))
    h = int(math.ceil((v1 - b[:, 1].min()) / cell - 1e-6))
    if h * w > relief.MAX_CELLS:
        raise NotReady(t(f"Il rilievo dei chunk insieme fa {h} x {w} celle, troppe: costruisci i rilievi con una cella "
                         "più grande.",
                         f"The relief of the chunks together makes {h} x {w} cells, too many: build the reliefs with a "
                         "larger cell."))
    return Affine(cell, 0.0, u0, 0.0, -cell, v1), h, w, cell


def _resample(z: np.ndarray, a, us: np.ndarray, vs: np.ndarray):
    """The raster ``z`` (affine ``a``) at the cell centres ``us`` x ``vs``: the value where the cell under the centre
    has one (as the nearest cell), bilinear over the neighbours that have one."""
    from scipy.ndimage import map_coordinates

    cf = (us - a.c) / a.a
    rf = (vs - a.f) / a.e
    ci, ri = np.floor(cf).astype(np.int64), np.floor(rf).astype(np.int64)
    okc, okr = (ci >= 0) & (ci < z.shape[1]), (ri >= 0) & (ri < z.shape[0])
    R, C = np.meshgrid(rf - 0.5, cf - 0.5, indexing="ij")
    finite = np.isfinite(z)
    num = map_coordinates(np.where(finite, z, 0.0), [R, C], order=1, mode="nearest")
    den = map_coordinates(finite.astype(np.float64), [R, C], order=1, mode="nearest")
    valid = np.zeros(R.shape, bool)
    sub = np.ix_(okr, okc)
    valid[sub] = finite[np.ix_(ri[okr], ci[okc])]
    out = np.full(R.shape, np.nan)
    good = valid & (den > 1e-9)
    out[good] = num[good] / den[good]
    return out


def prepare(doc, main, progress=None, found: list[dict] | None = None) -> dict:
    """The merged height raster of the members (see the module) and ``members.json``, in the main chunk's
    reconstruction folder (written aside, then put in place). Returns ``{"raster", "chunks"}``: the raster's path
    relative to the main's data folder and the members ``[{"chunk", "label", "cell", "coverage", "relief"}]`` (main
    first) - the entry :func:`is_current` checks."""
    import rasterio

    from pyarchinit_photogrammetry.core.progress import as_progress

    from . import reconstruct

    progress = as_progress(progress)
    found = members(doc, main)[0] if found is None else found
    frame = ship.need_frame(main)
    affine, h, w, cell = _grid(found)
    out = np.full((h, w), np.nan, np.float32)
    us = affine.c + (np.arange(w) + 0.5) * cell
    block = max(1, min(h, 4_000_000 // max(1, w)))
    best = ranked(found)
    for k, m in enumerate(best):
        progress(k / len(best), t(f"Rilievo dei chunk: {m['chunk'].label}", f"Relief of the chunks: {m['chunk'].label}"))
        z, a = relief.load(m["chunk"], "height")
        z = z.astype(np.float64)
        for r0 in range(0, h, block):
            r1 = min(h, r0 + block)
            vs = affine.f - (np.arange(r0, r1) + 0.5) * cell
            part = out[r0:r1]
            empty = ~np.isfinite(part)
            if not empty.any():
                continue
            values = _resample(z, a, us, vs)
            part[empty] = values[empty].astype(np.float32)
        del z
    folder = reconstruct.folder(main) / FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    path, tmp = folder / RASTER, folder / ("height.tmp.tif")
    profile = {"driver": "GTiff", "height": h, "width": w, "count": 1, "dtype": "float32", "transform": affine,
               "nodata": float("nan"), "compress": "deflate", "BIGTIFF": "IF_SAFER"}
    if w >= 256 and h >= 256:
        profile.update(tiled=True, blockxsize=256, blockysize=256)
    entries = [{"chunk": m["chunk"].key, "label": m["chunk"].label, "cell": m["cell"], "coverage": m["coverage"],
                "relief": relief_key(m["chunk"])} for m in found]
    try:
        with rasterio.open(tmp, "w", **profile) as dst:
            dst.update_tags(**relief._tags(frame, "height"),
                            ppg_members=json.dumps([e["chunk"] for e in entries]))
            dst.write(out, 1)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    index = folder / INDEX
    staged = folder / (INDEX + ".tmp")
    staged.write_text(json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")
    staged.replace(index)
    progress(1.0, t(f"Rilievo dei chunk: {len(found)} chunk, {h} x {w} celle",
                    f"Relief of the chunks: {len(found)} chunks, {h} x {w} cells"))
    return {"raster": str(path.relative_to(main.data_dir)), "chunks": entries}
