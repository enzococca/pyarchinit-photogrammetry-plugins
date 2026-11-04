"""The reconstruction of the missing parts of the hull: a stated hypothesis, kept apart from the measured catalogue.

This module holds the MIRROR of the preserved side (``ricostruito: simmetria``): each measured, not loose, port or
starboard polygon of the catalogue reflected across the keel plane (ship y -> -y) where the other side has nothing:

- the reflected polygon is cut to the cells of the relief (:mod:`relief`) where the model is NOT (the wreck shows
  nothing there; one-cell slivers along the model's edge taken off); each piece that is left is kept when fewer than
  :data:`COVERED` of its cells have the model and it overlaps the measured records of the same kind of the other side
  (or the centre) by no more than :data:`OVERLAP` of its area. A timber whose reflection lies on the model, a loose
  timber, a centre timber are never mirrored: no hypothesis where the wreck shows something;
- the piece is a record of ``state["reconstruction"]["timbers"]`` (never of the catalogue): ``id`` R1, R2, ...,
  the other side's ``name`` (``fascia B3`` <-> ``fascia T3``; the frames keep ``ordinata N``), ``kind``, ``side``,
  ``status`` ``reconstructed:mirror``, ``source`` (the original record's id), ``ship_xy``, ``geometry`` (the chunk's
  coordinates through the ship frame), ``frame_key``, ``chunk``, ``clipped`` (only a part of the original),
  ``solid`` (``shipwreck/reconstruction/<id>.ply``) and ``volume``;
- the solid is the original's reflected (y -> -y, the winding reversed so the faces stay turned outward): the
  original's 3D file when the piece is the whole timber and the solids are current, else the original's solid built
  now on the part that is mirrored (:mod:`solids`), then reflected.

Where both sides are measured the two are compared, not averaged (the wreck is copied as it lies):
:func:`disagreement` lists per station and seam the port and starboard half-breadths and their difference
(``state["reconstruction"]["table"]["mirror"]``).

Only ``chunk.meta["shipwreck"]["reconstruction"]`` and ``<chunk data>/shipwreck/reconstruction/`` are written; the
measured records, the chunk's transform, model and products never."""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import lines, relief, ship, solids, timbers, tracing
from .ship import NotReady
from .texts import t

OVERLAP = 0.20  # share of a mirrored piece's area over measured records of the other side: more, not mirrored
COVERED = 0.20  # share of a mirrored piece's cells with the model under them: more, not mirrored
MIN_CELLS = 25  # a piece smaller than this many cells of the relief is not mirrored (a sliver at the model's edge)
SIDES = ("port", "starboard")
STATUS = "reconstructed:mirror"
FOLDER = "reconstruction"
_CODE = re.compile(r"^([BT])(\d+|\?)$")


def other_side(side: str) -> str:
    return {"port": "starboard", "starboard": "port"}.get(side, side)


def other_name(name: str) -> str:
    """The name of a timber on the other side: the side's letter of its last word swapped (``fascia B3`` <->
    ``fascia T3``, ``comento T2`` -> ``comento B2``); any other name (``ordinata 7``) as it is."""
    words = str(name).split(" ")
    m = _CODE.match(words[-1]) if words else None
    if not m:
        return str(name)
    words[-1] = ("T" if m.group(1) == "B" else "B") + m.group(2)
    return " ".join(words)


def state(chunk) -> dict:
    """``chunk.meta["shipwreck"]["reconstruction"]`` (created when first asked for)."""
    return ship.state(chunk).setdefault("reconstruction", {})


def records_of(chunk) -> list[dict]:
    return list(ship.state(chunk).get("reconstruction", {}).get("timbers", []))


def folder(chunk) -> Path:
    d = ship.folder(chunk) / FOLDER
    d.mkdir(parents=True, exist_ok=True)
    return d


# ----------------------------------------------------------------------------- the relief
class _Relief:
    """The relief's height raster, read once: the cells of a polygon of the ship plan, the parts where the model is
    not."""

    def __init__(self, chunk, frame: ship.Frame):
        tracing.need_fresh_relief(chunk)
        self.z, self.affine = relief.load(chunk, "height")
        self.cell = float(abs(self.affine.a))
        self.sign = frame.sign

    def _window(self, poly_uv):
        from rasterio.windows import Window

        h, w = self.z.shape
        u0, v0, u1, v1 = poly_uv.bounds
        inv = ~self.affine
        cols, rows = zip(*(inv * (u, v) for u in (u0, u1) for v in (v0, v1)))
        c0, r0 = max(0, int(math.floor(min(cols))) - 1), max(0, int(math.floor(min(rows))) - 1)
        c1, r1 = min(w, int(math.ceil(max(cols))) + 1), min(h, int(math.ceil(max(rows))) + 1)
        return Window(c0, r0, max(0, c1 - c0), max(0, r1 - r0))

    def read(self, poly):
        """For a polygon of the ship plan: its cells (centres inside) and whether each has the model, as two
        boolean arrays on a window of the raster, with the window's affine."""
        import shapely
        from rasterio import features, windows

        uv = shapely.transform(poly, lambda p: relief.uv_of_ship(p, self.sign))
        win = self._window(uv)
        if win.width == 0 or win.height == 0:
            return np.zeros((0, 0), bool), np.zeros((0, 0), bool), None
        a = windows.transform(win, self.affine)
        z = self.z[win.row_off:win.row_off + win.height, win.col_off:win.col_off + win.width]
        inside = features.rasterize([(uv, 1)], out_shape=z.shape, transform=a, fill=0, dtype="uint8").astype(bool)
        return inside, np.isfinite(z), a

    def lost(self, poly):
        """The part of ``poly`` (ship plan) over cells without the model, less what is narrower than two cells."""
        import shapely
        from rasterio import features

        inside, model, a = self.read(poly)
        if a is None or not inside.any():
            return shapely.Polygon()
        empty = (~model).astype("uint8")
        shapes = [shapely.geometry.shape(g) for g, v in features.shapes(empty, mask=empty.astype(bool), transform=a)
                  if v == 1]
        if not shapes:
            return shapely.Polygon()
        lost_uv = shapely.union_all(shapes)
        lost = shapely.transform(lost_uv, lambda p: relief.ship_of_uv(p, self.sign))
        piece = poly.intersection(lost)
        return piece.buffer(-self.cell, join_style="mitre").buffer(self.cell, join_style="mitre").intersection(poly)

    def covered(self, poly) -> tuple[float, int]:
        """The share of ``poly``'s cells that have the model, and how many cells it has."""
        inside, model, _a = self.read(poly)
        n = int(inside.sum())
        return (float((inside & model).sum()) / n if n else 1.0), n


# ----------------------------------------------------------------------------- the mirror
def _polygon(xy):
    import shapely

    poly = shapely.Polygon(np.asarray(xy, float).reshape(-1, 2))
    return poly if poly.is_valid else poly.buffer(0)


def _world(frame: ship.Frame, xy) -> np.ndarray:
    xy = np.asarray(xy, float).reshape(-1, 2)
    return frame.to_world(np.column_stack([xy, np.zeros(len(xy))]))[:, :2]


def _volume(v: np.ndarray, f: np.ndarray) -> float:
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return abs(float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0))


def _reflect(xy) -> np.ndarray:
    return np.asarray(xy, float).reshape(-1, 2) * [1.0, -1.0]


def _candidates(records: list[dict]) -> list[dict]:
    return [r for r in records if r["side"] in SIDES and r["geometry"]["type"] == "polygon"
            and timbers.is_measured(r) and not r.get("loose")]


def _others(chunk, records, frame, kind: str, side: str) -> list:
    """The measured polygons of ``kind`` on ``side`` or the centre: what a mirrored piece must not lie on."""
    return [_polygon(tracing.ship_xy_of(chunk, r, frame)) for r in records
            if r["kind"] == kind and r["side"] in (side, "centre") and r["geometry"]["type"] == "polygon"
            and timbers.is_measured(r)]


def _pieces(geom) -> list:
    return [g for g in getattr(geom, "geoms", [geom]) if g.geom_type == "Polygon" and not g.is_empty]


def _ring(poly, cell: float) -> list[list[float]]:
    """The outer ring of a piece (simplified to half a cell: the cells' staircase), without the closing point."""
    import shapely

    poly = shapely.geometry.polygon.orient(poly.simplify(cell / 2.0, preserve_topology=True), 1.0)
    return [[float(x), float(y)] for x, y in np.asarray(poly.exterior.coords)[:-1]]


def _read_solid(path: Path):
    from pyarchinit_photogrammetry.io.mesh_ply import PlyMesh

    mesh = PlyMesh(path)
    v = np.array(mesh.xyz(), float)
    f = np.concatenate(list(mesh.face_chunks())) if mesh.n_faces else np.zeros((0, 3), np.int64)
    hidden = (np.asarray(mesh._faces["hidden"]).astype(bool) if "hidden" in mesh.face_dtype.names
              else np.zeros(len(f), bool))
    del mesh
    return v, np.asarray(f, np.int64), hidden


def _original_solid(chunk, source: dict, part_xy, whole: bool, frame) -> solids.Solid:
    """The original's solid over the part that is mirrored (``part_xy``, on the original's side): its 3D file when
    the part is the whole timber and the solids are current, else built now on the part."""
    if whole and source.get("solid") and not solids.is_stale(chunk):
        path = chunk.data_dir / source["solid"]
        if path.exists():
            v, f, hidden = _read_solid(path)
            return solids.Solid(v, f, hidden, timbers.status_of(source), source["name"], source["kind"],
                                _volume(v, f),
                                source.get("thickness_mm"), source["id"])
    part = {**source, "ship_xy": np.asarray(part_xy, float).tolist(), "frame_key": relief.frame_key(frame),
            "geometry": timbers.geometry("polygon", _world(frame, part_xy))}
    if source["kind"] in solids.SHELLS:
        mm, _how = solids.thickness_of(chunk, source)
        return solids.strake_shell(chunk, part, mm)
    return solids.timber_solid(chunk, part)


def _mirrored_solid(solid: solids.Solid, rid: str, name: str) -> solids.Solid:
    v = np.asarray(solid.vertices, float) * [1.0, -1.0, 1.0]
    f = np.asarray(solid.faces, np.int64)[:, ::-1]  # a reflection turns the faces inward: the winding reversed
    return solids.Solid(v, f, np.asarray(solid.hidden, bool), STATUS, name, solid.kind, abs(solid.volume),
                        solid.thickness_mm, rid)


def _next_id(records: list[dict]) -> int:
    return 1 + max((int(r["id"][1:]) for r in records if str(r.get("id", ""))[1:].isdigit()), default=0)


def _clear(chunk) -> None:
    """The earlier mirror out of the reconstruction, its files too."""
    st = state(chunk)
    kept, gone = [], []
    for r in st.get("timbers", []):
        (gone if r.get("status") == STATUS else kept).append(r)
    st["timbers"] = kept
    d = ship.folder(chunk) / FOLDER
    for r in gone:
        (d / f"{r['id']}.ply").unlink(missing_ok=True)
        (d / f"{r['id']}.ply.tmp").unlink(missing_ok=True)


def _clean(chunk, d: Path) -> None:
    """The files of the folder no reconstruction record holds (an earlier mirror undone, a removed record)."""
    keep = {f"{r['id']}.ply" for r in records_of(chunk)}
    for old in list(d.glob("*.ply")) + list(d.glob("*.ply.tmp")):
        if old.name not in keep:
            old.unlink(missing_ok=True)


def mirror(chunk, progress=None) -> list[dict]:
    """The preserved side mirrored where the other side is lost (see the module): the earlier mirror replaced, the
    new records added to ``state["reconstruction"]["timbers"]``, their solids written. Returns the new records; the
    timbers refused (no thickness, a solid that cannot be built) go to ``state["reconstruction"]["mirror"]``."""
    import shapely

    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    frame = ship.need_frame(chunk)
    if relief.entry(chunk) is None:
        raise NotReady(t("Costruisci prima il rilievo dello scafo.", "Build the relief of the hull first."))
    grid = _Relief(chunk, frame)
    records = timbers.timbers_of(chunk)
    candidates = _candidates(records)
    _clear(chunk)
    st = state(chunk)
    store = st.setdefault("timbers", [])
    n = _next_id(store)
    out, refused = [], []
    d = folder(chunk)
    key = relief.frame_key(frame)
    for k, source in enumerate(candidates):
        progress(k / max(1, len(candidates)), t(f"Simmetria: {source['name'] or source['id']}",
                                                f"Mirror: {source['name'] or source['id']}"))
        original = _polygon(tracing.ship_xy_of(chunk, source, frame))
        reflected = _polygon(_reflect(np.asarray(original.exterior.coords)[:-1]))
        if reflected.is_empty or reflected.area <= 0:
            continue
        others = _others(chunk, records, frame, source["kind"], other_side(source["side"]))
        over = shapely.union_all(others) if others else None
        for piece in _pieces(grid.lost(reflected)):
            share, cells = grid.covered(piece)
            if cells < MIN_CELLS or share >= COVERED:
                continue
            if over is not None and piece.intersection(over).area > OVERLAP * piece.area:
                continue
            xy = np.asarray(_ring(piece, grid.cell))
            if len(xy) < 3:
                continue
            whole = piece.symmetric_difference(reflected).area <= 0.01 * reflected.area
            if whole:
                xy = _reflect(np.asarray(original.exterior.coords)[:-1])
            rid = f"R{n}"
            name = other_name(source["name"])
            try:
                solid = _mirrored_solid(_original_solid(chunk, source, _reflect(xy), whole, frame), rid, name)
                path = solids.write_ply(d / f"{rid}.ply", solid, frame)
            except Exception as exc:  # noqa: BLE001 - one timber refused, never the whole mirror
                reason = str(exc) if isinstance(exc, NotReady) else f"{type(exc).__name__}: {exc}"
                refused.append({"source": source["id"], "name": source["name"], "reason": reason})
                continue
            n += 1
            record = {"id": rid, "kind": source["kind"], "side": other_side(source["side"]), "name": name,
                      "station": source.get("station", ""), "status": STATUS, "source": source["id"],
                      "ship_xy": xy.tolist(), "geometry": timbers.geometry("polygon", _world(frame, xy)),
                      "frame_key": key, "chunk": chunk.key, "clipped": not whole,
                      "solid": str(path.relative_to(chunk.data_dir)), "volume": float(solid.volume)}
            if solid.thickness_mm is not None:
                record["thickness_mm"] = float(solid.thickness_mm)
            store.append(record)
            out.append(record)
    _clean(chunk, d)
    st["mirror"] = {"built": datetime.now(timezone.utc).isoformat(timespec="seconds"), "count": len(out),
                    "refused": refused, "frame_key": key}
    progress(1.0, t(f"Simmetria: {len(out)} legni ricostruiti, {len(refused)} rifiutati",
                    f"Mirror: {len(out)} timbers reconstructed, {len(refused)} refused"))
    return out


# ----------------------------------------------------------------------------- the disagreement
def _rank(name: str) -> int | None:
    words = str(name).split(" ")
    m = _CODE.match(words[-1]) if words else None
    return int(m.group(2)) if m and m.group(2).isdigit() else None


def _at(xy: np.ndarray, x: float) -> float | None:
    """|y| of a line of the ship plan at ``x`` (None outside its x range)."""
    xs, ys = xy[:, 0], xy[:, 1]
    best = None
    for i in range(len(xy) - 1):
        a, b = xs[i], xs[i + 1]
        if min(a, b) <= x <= max(a, b):
            y = ys[i] if a == b else ys[i] + (x - a) * (ys[i + 1] - ys[i]) / (b - a)
            best = abs(float(y)) if best is None else min(best, abs(float(y)))
    return best


def _inner_edge(poly, x: float) -> float | None:
    """The |y| of a strake's edge nearest the keel at ``x``."""
    import shapely

    x0, _y0, x1, _y1 = poly.bounds
    if not x0 < x < x1:
        return None
    cut = poly.intersection(shapely.LineString([(x, -1e6), (x, 1e6)]))
    if cut.is_empty:
        return None
    return float(np.abs(shapely.get_coordinates(cut)[:, 1]).min())


def _stations(chunk) -> list[dict]:
    sections = [{"name": s.get("name") or "", "x": float(s["value"])} for s in lines.sections_of(chunk, "station")]
    return sections or [{"name": s["name"], "x": float(s["x"])} for s in ship.stations_of(chunk)]


def _seams(chunk, frame, side: str, x: float) -> dict[int, float]:
    """The |y| of the measured seams of ``side`` at ``x`` by rank: the seam records, and where a rank has none
    there the inner edge of the strake above it (strake k + 1 starts on seam k)."""
    out = {}
    for r in timbers.timbers_of(chunk):
        if r["side"] != side or r.get("loose") or not timbers.is_measured(r):
            continue
        rank = _rank(r["name"])
        if rank is None:
            continue
        if r["kind"] == "seam" and r["geometry"]["type"] == "line":
            y = _at(tracing.ship_xy_of(chunk, r, frame), x)
            if y is not None:
                out[rank] = y
    for r in timbers.timbers_of(chunk):
        if (r["side"] != side or r.get("loose") or not timbers.is_measured(r) or r["kind"] != "plank"
                or r["geometry"]["type"] != "polygon"):
            continue
        rank = _rank(r["name"])
        if rank is None or rank < 2 or (rank - 1) in out:
            continue
        y = _inner_edge(_polygon(tracing.ship_xy_of(chunk, r, frame)), x)
        if y is not None:
            out[rank - 1] = y
    return out


def disagreement(chunk) -> list[dict]:
    """Per station (the sections' stations, else the chunk's) and seam rank where both sides have a measured seam
    (or strake edge): ``station``, ``x``, ``rank``, ``port_y`` and ``starboard_y`` (the half-breadths, |y|),
    ``difference_mm`` (port - starboard). Kept in ``state["reconstruction"]["table"]["mirror"]``."""
    frame = ship.need_frame(chunk)
    stations = _stations(chunk)
    if not stations:
        raise NotReady(t("Nessuna stazione: dai prima le stazioni.", "No stations: set the stations first."))
    rows = []
    for s in stations:
        port, starboard = _seams(chunk, frame, "port", s["x"]), _seams(chunk, frame, "starboard", s["x"])
        for rank in sorted(set(port) & set(starboard)):
            p, q = port[rank], starboard[rank]
            if not (math.isfinite(p) and math.isfinite(q)):
                continue
            rows.append({"station": s["name"], "x": s["x"], "rank": rank, "port_y": p, "starboard_y": q,
                         "difference_mm": (p - q) * 1000.0})
    state(chunk).setdefault("table", {})["mirror"] = rows
    return rows
