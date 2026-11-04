"""The reconstruction of the missing parts of the hull: a stated hypothesis, kept apart from the measured catalogue.

This module holds the MIRROR of the preserved side (``ricostruito: simmetria``): each measured, not loose, port or
starboard polygon of the catalogue reflected across the keel plane (ship y -> -y) where the other side has nothing:

- the reflected polygon is cut to the cells of the relief (:mod:`relief`) where the model is NOT (the wreck shows
  nothing there): the mask of those cells opened by :data:`OPENING` cells (the empty cells along the model's own
  edge are no loss), the polygon kept whole when all its cells are lost (``clipped`` False), else cut to its lost
  cells. Each piece is kept when it is at least :data:`MIN_AREA`, at least :data:`MIN_WIDTH` wide (or
  :data:`NARROW` of the timber's own width, for a narrow timber), fewer than :data:`COVERED` of its cells have the
  model, it overlaps the measured records of the same kind of the other side (or the centre) by no more than
  :data:`OVERLAP` of its area, and, when it is NARROW (narrower than both :data:`DEFORMATION` and :data:`NARROW` of
  the timber's width), it does not lie ALONGSIDE (within :data:`TOUCH`) the other side's measured timber of its
  name: that is the same timber read wider on one side - a deformed wreck - and its difference goes in the table of
  :func:`disagreement`, not into a timber. A wide piece alongside (the upper part of an eroded strake) is a lost
  part and is mirrored; a piece meeting it end to end, where the other side is lost beyond it, is the timber going
  on. A timber whose reflection lies on the model, a loose timber, a centre timber are never
  mirrored: no hypothesis where the wreck shows something;
- the piece is a record of ``state["reconstruction"]["timbers"]`` (never of the catalogue): ``id`` R1, R2, ...,
  the other side's ``name`` (``fascia B3`` <-> ``fascia T3``; the frames keep ``ordinata N``; the pieces of one
  timber ``fascia B3 (1)``, ``(2)``), ``kind``, ``side``,
  ``status`` ``reconstructed:mirror``, ``source`` (the original record's id), ``ship_xy``, ``geometry`` (the chunk's
  coordinates through the ship frame, counter-clockwise), ``frame_key``, ``chunk``, ``clipped`` (only a part of the original),
  ``solid`` (``shipwreck/reconstruction/<id>.ply``) and ``volume``;
- the solid is the original's reflected (y -> -y, the winding reversed so the faces stay turned outward): the
  original's 3D file when the piece is the whole timber and the solids are current, else the original's solid built
  now on the part that is mirrored (:mod:`solids`), then reflected.

Where both sides are measured the two are compared, not averaged (the wreck is copied as it lies):
:func:`disagreement` lists per station and seam the port and starboard half-breadths and their difference
(``state["reconstruction"]["table"]["mirror"]``).

The EXTENSION (``ricostruito: prolungamento``, :func:`extend`) continues each strake between two seams along the
hull with a fair spline through its seams, inside the surveyed length (:func:`surveyed_range`) and beyond it only to
the stem and stern the user gives (:func:`set_ends`); the FRAMES BY SPACING (``ricostruito: passo delle ordinate``,
:func:`frames_by_spacing`) add the frames missing between two frames at the measured room and space. Both are records
of the same list, each kind replaced as a whole when built again, without solids (the extension's edges keep their z).

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
MIN_AREA = 0.0025  # m2: a smaller piece is not mirrored
MIN_WIDTH = 0.05  # m: a piece narrower (2 area / perimeter) is not mirrored, unless the timber itself is narrow
NARROW = 0.5  # ... then a piece of at least this share of the timber's own width is
TOUCH = 0.02  # m: a NARROW piece lying this close beside the other side's measured timber of its name ...
DEFORMATION = 0.10  # m: ... narrower than this (and than NARROW of the timber's width) is a disagreement
OPENING = 2  # cells: lost areas narrower than this along the model's edge are not lost (the edge's own cells)
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
        cols, rows = zip(*(inv @ (u, v) for u in (u0, u1) for v in (v0, v1)))
        pad = OPENING + 2
        c0, r0 = max(0, int(math.floor(min(cols))) - pad), max(0, int(math.floor(min(rows))) - pad)
        c1, r1 = min(w, int(math.ceil(max(cols))) + pad), min(h, int(math.ceil(max(rows))) + pad)
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
        """The part of ``poly`` (ship plan) over cells without the model and whether that is the whole of it. The
        MASK of the lost cells is opened by :data:`OPENING` cells first (a strip of empty cells along the model's
        edge is not a lost part); a polygon all of whose cells are lost comes back whole, untrimmed. Otherwise the
        piece is the polygon over its lost cells and the ring of lost cells its outline crosses (never a strip of
        the polygon lying over the empty cells beyond the model's edge, where none of its cell centres is)."""
        import shapely
        from rasterio import features
        from scipy import ndimage

        inside, model, a = self.read(poly)
        if a is None or not inside.any():
            return shapely.Polygon(), False
        # a square of OPENING + 1 cells: strips of empty cells up to OPENING wide go
        lost = ndimage.binary_opening(~model, structure=np.ones((OPENING + 1, OPENING + 1), bool), border_value=1)
        if lost[inside].all():
            return poly, True
        # the polygon's lost cells and the ring of lost cells its outline crosses (a cell whose centre is outside):
        # never a strip of the polygon over the empty cells beyond the model's edge, where no cell centre of it lies
        cells = inside & lost
        cells = ndimage.binary_dilation(cells, structure=np.ones((3, 3), bool)) & lost
        empty = cells.astype("uint8")
        shapes = [shapely.geometry.shape(g) for g, v in features.shapes(empty, mask=cells, transform=a) if v == 1]
        if not shapes:
            return shapely.Polygon(), False
        lost_ship = shapely.transform(shapely.union_all(shapes), lambda p: relief.ship_of_uv(p, self.sign))
        return poly.intersection(lost_ship), False

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


def _others(chunk, records, frame, kind: str, side: str, name: str | None = None) -> list:
    """The measured polygons of ``kind`` on ``side`` or the centre (``name``: on ``side`` only, of that name):
    what a mirrored piece must not lie on, or beside."""
    return [_polygon(tracing.ship_xy_of(chunk, r, frame)) for r in records
            if r["kind"] == kind and r["geometry"]["type"] == "polygon" and timbers.is_measured(r)
            and (r["side"] in (side, "centre") if name is None else (r["side"] == side and r["name"] == name))]


def _pieces(geom) -> list:
    return [g for g in getattr(geom, "geoms", [geom]) if g.geom_type == "Polygon" and not g.is_empty]


def _ring(poly, cell: float | None = None) -> np.ndarray:
    """The outer ring of a piece counter-clockwise, without the closing point; ``cell``: simplified to half a cell
    (the cells' staircase at a cut)."""
    import shapely

    if cell is not None:
        poly = poly.simplify(cell / 2.0, preserve_topology=True)
    poly = shapely.geometry.polygon.orient(poly, 1.0)
    return np.asarray(poly.exterior.coords, float)[:-1, :2]


def _width(poly) -> float:
    """The mean width of a strip: twice its area over its perimeter."""
    return 2.0 * poly.area / poly.length if poly.length > 0 else 0.0


def _beside(piece, record_poly) -> bool:
    """Whether ``piece`` lies alongside ``record_poly`` (within :data:`TOUCH` over a run along the piece's long
    axis longer than across it): the same timber read wider on one side - a disagreement, not a lost part. A piece
    that only meets it end to end (the timber going on where the other side is lost) does not."""
    import shapely

    contact = piece.boundary.intersection(record_poly.buffer(TOUCH))
    if contact.is_empty:
        return False
    with np.errstate(invalid="ignore"):  # shapely's envelope on a piece of a few cells
        box = shapely.get_coordinates(shapely.minimum_rotated_rectangle(piece))[:4]
    sides = np.diff(np.vstack([box, box[:1]]), axis=0)[:2] if len(box) == 4 else np.zeros((0, 2))
    if not len(sides) or not np.isfinite(sides).all() or np.hypot(*sides.T).max() <= 0:
        x0, y0, x1, y1 = piece.bounds
        sides = np.array([[x1 - x0, 0.0], [0.0, y1 - y0]])
    along = sides[np.argmax(np.hypot(*sides.T))]
    along = along / max(np.hypot(*along), 1e-12)
    across = np.array([-along[1], along[0]])
    pts = shapely.get_coordinates(contact)
    return float(np.ptp(pts @ along)) > max(float(np.ptp(pts @ across)), TOUCH)


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


def _swap(chunk, new: list[dict], tmp: Path) -> None:
    """The earlier mirror replaced by ``new`` (records and their files, written in ``tmp``); the files no
    reconstruction record holds removed (an earlier mirror undone, a removed record)."""
    st = state(chunk)
    d = folder(chunk)
    kept = [r for r in st.get("timbers", []) if r.get("status") != STATUS]
    for r in st.get("timbers", []):
        if r.get("status") == STATUS:
            (d / f"{r['id']}.ply").unlink(missing_ok=True)
    for r in new:
        (tmp / f"{r['id']}.ply").replace(d / f"{r['id']}.ply")
    st["timbers"] = kept + new
    keep = {f"{r['id']}.ply" for r in st["timbers"]}
    for old in list(d.glob("*.ply")) + list(d.glob("*.ply.tmp")):
        if old.name not in keep:
            old.unlink(missing_ok=True)


def _kept(grid: "_Relief", piece, source_width: float, over, beside) -> bool:
    """Whether a lost piece is mirrored: big and wide enough, off the model, not on the other side's records of
    its kind, not alongside the other side's timber of its name."""
    share, _cells = grid.covered(piece)
    if piece.area < MIN_AREA or share >= COVERED:
        return False
    if _width(piece) < min(MIN_WIDTH, NARROW * source_width):
        return False
    if over is not None and piece.intersection(over).area > OVERLAP * piece.area:
        return False
    if _width(piece) >= min(NARROW * source_width, DEFORMATION):
        return True  # as wide as a lost part: mirrored even alongside the measured part (an eroded sheer)
    return not any(_beside(piece, b) for b in beside)


def mirror(chunk, progress=None) -> list[dict]:
    """The preserved side mirrored where the other side is lost (see the module): the earlier mirror replaced, the
    new records added to ``state["reconstruction"]["timbers"]``, their solids written (into a folder of their own
    first: whatever stops the mirror halfway leaves the earlier one as it was). Returns the new records; the timbers
    refused (no thickness, a solid that cannot be built) go to ``state["reconstruction"]["mirror"]``."""
    import shutil

    import shapely

    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    frame = ship.need_frame(chunk)
    if relief.entry(chunk) is None:
        raise NotReady(t("Costruisci prima il rilievo dello scafo.", "Build the relief of the hull first."))
    grid = _Relief(chunk, frame)
    records = timbers.timbers_of(chunk)
    candidates = _candidates(records)
    st = state(chunk)
    n = _next_id([r for r in st.get("timbers", []) if r.get("status") != STATUS])
    out, refused = [], []
    key = relief.frame_key(frame)
    tmp = ship.folder(chunk) / (FOLDER + ".new")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        for k, source in enumerate(candidates):
            progress(k / max(1, len(candidates)), t(f"Simmetria: {source['name'] or source['id']}",
                                                    f"Mirror: {source['name'] or source['id']}"))
            original = _polygon(tracing.ship_xy_of(chunk, source, frame))
            if original.is_empty or original.area <= 0:
                continue
            reflected = _polygon(_reflect(_ring(original)))
            side, name = other_side(source["side"]), other_name(source["name"])
            others = _others(chunk, records, frame, source["kind"], side)
            over = shapely.union_all(others) if others else None
            beside = _others(chunk, records, frame, source["kind"], side, name)
            lost, whole = grid.lost(reflected)
            pieces = [p for p in _pieces(lost) if _kept(grid, p, _width(original), over, beside)]
            for j, piece in enumerate(pieces):
                xy = _ring(piece) if whole else _ring(piece, grid.cell)
                if len(xy) < 3:
                    continue
                rid = f"R{n}"
                label = name if len(pieces) == 1 else f"{name} ({j + 1})"
                try:
                    solid = _mirrored_solid(_original_solid(chunk, source, _reflect(xy), whole, frame), rid, label)
                    path = solids.write_ply(tmp / f"{rid}.ply", solid, frame)
                except Exception as exc:  # noqa: BLE001 - one timber refused, never the whole mirror
                    reason = str(exc) if isinstance(exc, NotReady) else f"{type(exc).__name__}: {exc}"
                    refused.append({"source": source["id"], "name": source["name"], "reason": reason})
                    continue
                n += 1
                record = {"id": rid, "kind": source["kind"], "side": side, "name": label,
                          "station": source.get("station", ""), "status": STATUS, "source": source["id"],
                          "ship_xy": xy.tolist(), "geometry": timbers.geometry("polygon", _world(frame, xy)),
                          "frame_key": key, "chunk": chunk.key, "clipped": not whole,
                          "solid": str((folder(chunk) / path.name).relative_to(chunk.data_dir)),
                          "volume": float(solid.volume)}
                if solid.thickness_mm is not None:
                    record["thickness_mm"] = float(solid.thickness_mm)
                out.append(record)
        _swap(chunk, out, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
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


# ----------------------------------------------------------------------------- the ends and the surveyed length
EXTENSION = "reconstructed:extension"
SPACING = "reconstructed:spacing"
STEP = 0.05  # m: the bins along x of a seam's points, and of a mirrored strake's edges
GRID = 0.02  # m: the step along x of the extended edges
BLEND = 0.5  # m: past the last traced point the derivative goes over into the trend over this length
MIN_GAP = 0.05  # m: a shorter gap of a strake is not filled
GAP_FACTOR = 1.5  # a space between two frames longer than this many median spaces has frames missing
_STRAKE = re.compile(r"([BT])(\d+)(?:\s*\(\d+\))?\s*$")
_FRAME_NUMBER = re.compile(r"^(?:ordinata|frame)\s+(\d+)\s*$")
_FAR = 1.0e4


def _end(value) -> dict | None:
    """An end as stored: ``{"x": ship x}`` (a number, or a dict with ``x``) or ``{"line": [[x, y], ...]}`` (a dict
    with ``line``, or the points themselves: a line traced in the ship plan)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return {"x": float(value)}
    if isinstance(value, dict) and "x" in value:
        return {"x": float(value["x"])}
    pts = value.get("line") if isinstance(value, dict) else value
    xy = np.asarray(pts, float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2 or not np.isfinite(xy).all():
        raise ValueError(t("Un'estremità è una x della nave o una linea di almeno due punti (x, y).",
                           "An end is a ship x or a line of at least two points (x, y)."))
    return {"line": xy.tolist()}


def set_ends(chunk, stem=None, stern=None) -> dict:
    """The stem and the stern of the hull, as far as the strakes may be extended beyond the surveyed length: a ship
    x typed (a number) or a line traced in the ship plan (``{"line": [[x, y], ...]}``); None: no such end. Both
    are set by each call (``state["ends"]``); returns them."""
    ends = {key: e for key, e in (("stem", _end(stem)), ("stern", _end(stern))) if e is not None}
    st = state(chunk)
    if ends:
        st["ends"] = ends
    else:
        st.pop("ends", None)
    return dict(ends)


def surveyed_range(chunk) -> tuple[float, float]:
    """The ship x range of the relief's cells with a value, both sides taken together (the surveyed wreck)."""
    ship.need_frame(chunk)
    if relief.entry(chunk) is None:
        raise NotReady(t("Costruisci prima il rilievo dello scafo.", "Build the relief of the hull first."))
    z, affine = relief.load(chunk, "height")
    cols = np.flatnonzero(np.isfinite(z).any(axis=0))
    if not len(cols):
        raise NotReady(t("Il rilievo dello scafo è vuoto.", "The relief of the hull is empty."))
    u = affine.c + affine.a * (np.array([cols[0], cols[-1]], float) + 0.5)
    return float(u.min()), float(u.max())


def _limits(chunk, surveyed: tuple[float, float]):
    """How far the strakes may go: the surveyed range, each end given setting its own side (the one beyond the
    middle of the surveyed range: the stem typed inside the range shortens it); the regions past a traced end."""
    lo, hi = surveyed
    mid = 0.5 * (lo + hi)
    cuts, sides = [], set()
    for key, e in (state(chunk).get("ends") or {}).items():
        if "x" in e:
            x, line = float(e["x"]), None
        else:
            line = np.asarray(e["line"], float)
            x = float(line[:, 0].mean())
        side = 1 if x > mid else -1
        if side in sides:
            raise NotReady(t("Prua e poppa stanno dalla stessa parte del relitto.",
                             "Stem and stern lie on the same side of the wreck."))
        sides.add(side)
        if line is None:
            lo, hi = (lo, x) if side > 0 else (x, hi)
            continue
        if side > 0:
            hi = float(line[:, 0].max())
        else:
            lo = float(line[:, 0].min())
        cuts.append(_beyond(line, side))
    return lo, hi, cuts


def _beyond(line: np.ndarray, side: int):
    """The region of the plan past a traced end (``side`` +1: larger x): the line continued across at its ends."""
    import shapely

    pts = line[np.argsort(line[:, 1])]
    far = side * _FAR
    ring = [(pts[0, 0], -_FAR)] + [tuple(p) for p in pts] + [(pts[-1, 0], _FAR), (far, _FAR), (far, -_FAR)]
    poly = shapely.Polygon(ring)
    return poly if poly.is_valid else poly.buffer(0)


# ----------------------------------------------------------------------------- the curves along x
class _Curve:
    """A value along x fitted on its points: a smoothing spline (GCV), a polynomial for a few points; its derivative
    and, outside the points, the line of its end (the derivative held)."""

    def __init__(self, x: np.ndarray, v: np.ndarray):
        from scipy.interpolate import make_smoothing_spline

        self.lo, self.hi = float(x[0]), float(x[-1])
        self.spline = None
        if len(x) >= 5:
            try:
                self.spline = make_smoothing_spline(x, v)
                self.slope = self.spline.derivative()
                self.bend = self.slope.derivative()
            except Exception:  # noqa: BLE001 - GCV on awkward points: a polynomial instead
                self.spline = None
        if self.spline is None:
            self.poly = np.polyfit(x, v, min(2, len(x) - 1)) if len(x) > 1 else np.array([float(v[0])])

    def __call__(self, x):
        x = np.asarray(x, float)
        if self.spline is not None:
            return self.spline(x)
        return np.polyval(self.poly, x)

    def d(self, x):
        x = np.asarray(x, float)
        if self.spline is not None:
            return self.slope(x)
        return np.polyval(np.polyder(self.poly), x) if len(self.poly) > 1 else np.zeros_like(x)

    def trend(self, x):
        """The derivative as a trend: the spline's inside its points, outside them its end's derivative going on
        with the end's bending for up to 1 m, then held."""
        x = np.asarray(x, float)
        if self.spline is None:
            return self.d(x)
        e = np.clip(x, self.lo, self.hi)
        reach = np.clip(x - e, -1.0, 1.0)
        return self.slope(e) + self.bend(e) * reach


def _curve(x, v):
    x, v = np.asarray(x, float), np.asarray(v, float)
    ok = np.isfinite(x) & np.isfinite(v)
    if ok.sum() < 2:
        return None
    x, v = x[ok], v[ok]
    order = np.argsort(x)
    x, v = x[order], v[order]
    keep = np.r_[True, np.diff(x) > 1e-9]
    return _Curve(x[keep], v[keep]) if keep.sum() >= 2 else None


# ----------------------------------------------------------------------------- the seams' points
def _strake_rank(name: str) -> tuple[str, int] | None:
    m = _STRAKE.search(str(name))
    return (m.group(1), int(m.group(2))) if m else None


def _base_name(name: str) -> str:
    return re.sub(r"\s*\(\d+\)\s*$", "", str(name))


def _densify(xy: np.ndarray, step: float = 0.01) -> np.ndarray:
    seg = np.hypot(*np.diff(xy, axis=0).T)
    s = np.r_[0.0, np.cumsum(seg)]
    if s[-1] <= 0:
        return xy[:1]
    at = np.linspace(0.0, s[-1], max(2, int(np.ceil(s[-1] / step)) + 1))
    return np.column_stack([np.interp(at, s, xy[:, 0]), np.interp(at, s, xy[:, 1])])


def _edge_points(poly, outer: bool) -> np.ndarray:
    """Points of a strake's edge across x (``outer``: the edge farthest from the keel), every :data:`STEP`."""
    import shapely

    x0, _y0, x1, _y1 = poly.bounds
    out = []
    for x in np.arange(x0 + 0.02, x1 - 0.02 + 1e-9, STEP):
        cut = poly.intersection(shapely.LineString([(x, -1e3), (x, 1e3)]))
        if cut.is_empty:
            continue
        y = shapely.get_coordinates(cut)[:, 1]
        a = np.abs(y)
        out.append([x, float(y[np.argmax(a)] if outer else y[np.argmin(a)])])
    return np.asarray(out, float).reshape(-1, 2)


class _Heights:
    """Ship z read on the relief at plan points (bilinear, NaN off the model or on a timber over the planking)."""

    def __init__(self, chunk, frame: ship.Frame, records: list[dict]):
        import shapely

        self.z, self.affine = relief.load(chunk, "height")
        self.sign = frame.sign
        cutters = [_polygon(tracing.ship_xy_of(chunk, r, frame)) for r in records
                   if r["kind"] in tracing.CUTTERS and r["geometry"]["type"] == "polygon" and not r.get("loose")]
        self.cutters = shapely.union_all(cutters).buffer(0.01) if cutters else None

    def at(self, xy: np.ndarray) -> np.ndarray:
        import shapely
        from scipy.ndimage import map_coordinates

        xy = np.asarray(xy, float).reshape(-1, 2)
        inv = ~self.affine
        u, v = xy[:, 0], self.sign * xy[:, 1]
        col, row = inv.a * u + inv.b * v + inv.c, inv.d * u + inv.e * v + inv.f
        z = map_coordinates(self.z, [row - 0.5, col - 0.5], order=1, mode="constant",
                            cval=np.nan)
        if self.cutters is not None and len(xy):
            z[shapely.contains_xy(self.cutters, xy[:, 0], xy[:, 1])] = np.nan
        return z


def _binned(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """Means of |y| and medians of z per :data:`STEP` of x."""
    if not len(x):
        return np.zeros(0), np.zeros(0), np.zeros(0)
    idx = np.floor(x / STEP).astype(int)
    keys, inv = np.unique(idx, return_inverse=True)
    n = np.bincount(inv)
    xb = np.bincount(inv, x) / n
    yb = np.bincount(inv, np.abs(y)) / n
    zb = np.full(len(keys), np.nan)
    for i in range(len(keys)):
        v = z[inv == i]
        v = v[np.isfinite(v)]
        if len(v):
            zb[i] = float(np.median(v))
    return xb, yb, zb


class _Seam:
    """The points of the seam ``rank`` of a side (|y|, ship z by x): its measured seam records, where there are none
    the edges of the strakes beside it (a mirrored strake's z read on the other side, where the wreck is)."""

    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z
        self.fy = _curve(x, y)
        self.fz = _curve(x, z)
        self.lo, self.hi = (float(x.min()), float(x.max())) if len(x) else (None, None)


def _seam(chunk, frame, heights: _Heights, records, recon, side: str, rank: int) -> _Seam | None:
    code = "B" if side == "port" else "T"
    seams = [r for r in records if r["kind"] == "seam" and r["side"] == side and r["geometry"]["type"] == "line"
             and not r.get("loose") and timbers.is_measured(r) and _strake_rank(r["name"]) == (code, rank)]
    xs, ys, zs = [], [], []
    for r in seams:
        xy = _densify(tracing.ship_xy_of(chunk, r, frame))
        xs.append(xy[:, 0])
        ys.append(xy[:, 1])
        zs.append(heights.at(xy))
    seam_bins = set(np.floor(np.concatenate(xs) / STEP).astype(int).tolist()) if xs else set()
    planks = [(r, _strake_rank(r["name"])) for r in records + recon
              if r["kind"] == "plank" and r["side"] == side and r["geometry"]["type"] == "polygon"
              and not r.get("loose") and (r.get("status") == STATUS or timbers.is_measured(r))]
    above = any(rk == (code, rank + 1) for _r, rk in planks)
    for r, rk in planks:
        if rk not in ((code, rank), (code, rank + 1)) or (rk == (code, rank) and not above):
            continue
        mirrored = r.get("status") == STATUS
        if not mirrored and seams:
            continue  # a measured strake's edge only where its seam was not traced
        xy = _edge_points(_polygon(tracing.ship_xy_of(chunk, r, frame)), outer=rk == (code, rank))
        if not len(xy):
            continue
        keep = ~np.isin(np.floor(xy[:, 0] / STEP).astype(int), list(seam_bins))
        xy = xy[keep]
        xs.append(xy[:, 0])
        ys.append(xy[:, 1])
        zs.append(heights.at(xy * [1.0, -1.0] if mirrored else xy))
    if not xs:
        return None
    x, y, z = _binned(np.concatenate(xs), np.concatenate(ys), np.concatenate(zs))
    return _Seam(x, y, z) if len(x) >= 2 else None


# ----------------------------------------------------------------------------- the trends
class _Trend:
    """Where a seam goes past its points: the 0.2.0 land line of its rank on its side, else the stations' sheer
    (|y| in proportion to the seam's at its end, z as the sheer's), else nothing (the seam's own end slope)."""

    def __init__(self, kind: str, fy=None, fz=None):
        self.kind, self.fy, self.fz = kind, fy, fz

    def dy(self, x, anchor_x: float, anchor_y: float):
        if self.fy is None:
            return None
        d = self.fy.trend(x)
        if self.kind == "sheer":
            ref = float(self.fy(np.clip(anchor_x, self.fy.lo, self.fy.hi)))
            d = d * (anchor_y / ref if ref > 1e-6 else 1.0)
        return d

    def dz(self, x):
        return None if self.fz is None else self.fz.trend(x)


def _land_lines(chunk) -> list[dict]:
    from . import clinker

    found = (ship.state(chunk).get("clinker") or {}).get("lines") or []
    if found and clinker.is_stale(chunk):
        return []
    return found


def _sheers(chunk) -> dict[str, tuple]:
    """Per side the sheer at the stations' sections (the highest point of the side's lines): (x, |y|, z)."""
    out: dict[str, list] = {"port": [], "starboard": []}
    for s in lines.sections_of(chunk, "station"):
        pts = [np.asarray(p, float).reshape(-1, 2) for p in s.get("lines") or []]
        if not pts:
            continue
        p = np.vstack(pts)
        for side, sign in (("port", 1.0), ("starboard", -1.0)):
            mine = p[sign * p[:, 0] > 0]
            if len(mine):
                top = mine[np.argmax(mine[:, 1])]
                out[side].append((float(s["value"]), abs(float(top[0])), float(top[1])))
    return {k: np.asarray(sorted(v), float).reshape(-1, 3) for k, v in out.items()}


def _trend(chunk, side: str, rank: int, sheers: dict) -> _Trend:
    for line in _land_lines(chunk):
        if line.get("side") == side and int(line.get("k") or 0) == rank and len(line.get("points") or []) >= 2:
            p = np.asarray(line["points"], float)
            fy = _curve(p[:, 0], np.abs(p[:, 1]))
            if fy is not None:
                return _Trend("land line", fy, _curve(p[:, 0], p[:, 2]))
    p = sheers.get(side)
    if p is not None and len(p) >= 2:
        fy = _curve(p[:, 0], p[:, 1])
        if fy is not None:
            return _Trend("sheer", fy, _curve(p[:, 0], p[:, 2]))
    return _Trend("spline")


# ----------------------------------------------------------------------------- the extended edges
def _grid(lo: float, hi: float, extra=()) -> np.ndarray:
    xs = np.r_[np.arange(lo, hi, GRID), hi, np.asarray(list(extra), float)]
    return np.unique(np.clip(xs, lo, hi))


def _go_on(curve: _Curve, end: float, xs: np.ndarray, trend) -> np.ndarray:
    """The value of ``curve`` continued from its point ``end`` over ``xs`` (all on one side of it, ordered away from
    it): the derivative at ``end`` going linearly over :data:`BLEND` into ``trend(x)`` (None: held)."""
    out = np.empty(len(xs))
    v, x0 = float(curve(end)), end
    d_end = float(curve.d(end))
    for i, x in enumerate(xs):
        m = 0.5 * (x0 + x)
        w = min(abs(m - end) / BLEND, 1.0)
        d = d_end if trend is None else (1.0 - w) * d_end + w * float(trend(m))
        v += d * (x - x0)
        out[i], x0 = v, x
    return out


def _edge(seam: _Seam, trend: _Trend, xs: np.ndarray) -> np.ndarray:
    """The seam over ``xs``: (x, |y|, z), the spline inside its points, continued past them (:func:`_go_on`)."""
    y, z = np.full(len(xs), np.nan), np.full(len(xs), np.nan)
    inside = (xs >= seam.lo) & (xs <= seam.hi)
    y[inside] = seam.fy(xs[inside])
    if seam.fz is not None:
        z[inside] = seam.fz(xs[inside])
    for part, end in ((xs < seam.lo, seam.lo), (xs > seam.hi, seam.hi)):
        if not part.any():
            continue
        order = np.argsort(np.abs(xs[part] - end))
        sub = xs[part][order]
        y_end = float(seam.fy(end))
        yv = _go_on(seam.fy, end, sub, (lambda m, e=end, ye=y_end: trend.dy(m, e, ye)) if trend.fy is not None else None)
        idx = np.flatnonzero(part)[order]
        y[idx] = np.maximum(yv, 0.0)
        if seam.fz is not None:
            z[idx] = _go_on(seam.fz, end, sub, trend.dz if trend.fz is not None else None)
    return np.column_stack([xs, y, z])


def _intervals(polys) -> list[tuple[float, float]]:
    spans = sorted((p.bounds[0], p.bounds[2]) for p in polys if not p.is_empty)
    out: list[list[float]] = []
    for a, b in spans:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _gaps(covered, lo: float, hi: float) -> list[tuple[float, float]]:
    out, at = [], lo
    for a, b in covered:
        if a > at:
            out.append((at, min(a, hi)))
        at = max(at, b)
    if at < hi:
        out.append((at, hi))
    return [(a, b) for a, b in out if b - a >= MIN_GAP]


def _band(lower: np.ndarray, upper: np.ndarray, sign: float):
    """The polygon of a band between two edges (x, |y|, z) on the same x, on the side ``sign``."""
    ylo, yhi = lower[:, 1], np.maximum(upper[:, 1], lower[:, 1])
    ring = np.vstack([np.column_stack([lower[:, 0], sign * ylo]), np.column_stack([upper[::-1, 0], sign * yhi[::-1]])])
    return _polygon(ring)


def _replace(chunk, status: str, new: list[dict]) -> None:
    st = state(chunk)
    st["timbers"] = [r for r in st.get("timbers", []) if r.get("status") != status] + new


def _ids(chunk, status: str):
    """Ids for the new records of ``status``: an earlier record's for the same source and position, else the next
    free one (never one another record held)."""
    old = {(r.get("source"), tuple(r.get("position") or ())): r["id"] for r in state(chunk).get("timbers", [])
           if r.get("status") == status}
    n = [_next_id(state(chunk).get("timbers", []))]

    def take(source, at) -> str:
        at = tuple(at)
        if (source, at) in old:
            return old.pop((source, at))
        rid = f"R{n[0]}"
        n[0] += 1
        return rid

    return take


def extend(chunk, progress=None) -> list[dict]:
    """Each strake between two seams (measured, or mirrored: its edges are its seams) continued along the hull: its
    seams (|y|, and z from the relief) fitted in x by a smoothing spline, over the strake's gaps inside
    :func:`surveyed_range` and, beyond it, only up to the ends :func:`set_ends` gave; past the last point of a seam
    its derivative goes over linearly in :data:`BLEND` into the trend of the 0.2.0 land line of its rank, else of
    the stations' sheer. The new parts only, as band polygons (``reconstructed:extension``, named as the strake,
    ``source`` the strake's record nearest, ``edges`` lower/upper (x, y, z), ``seams``, ``trend``, ``beyond``: past
    the surveyed range); the earlier extension replaced. No solid yet."""
    import shapely

    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    frame = ship.need_frame(chunk)
    surveyed = surveyed_range(chunk)
    tracing.need_fresh_relief(chunk)
    lo, hi, cuts = _limits(chunk, surveyed)
    records = timbers.timbers_of(chunk)
    recon = [r for r in records_of(chunk) if r.get("status") == STATUS]
    heights = _Heights(chunk, frame, [r for r in records if timbers.is_measured(r)])
    sheers = _sheers(chunk)
    beyond = shapely.union_all(cuts) if cuts else None
    strakes: dict[tuple[str, int], list[dict]] = {}
    for r in records + recon:
        rk = _strake_rank(r["name"])
        if (r["kind"] == "plank" and r["side"] in SIDES and r["geometry"]["type"] == "polygon" and rk
                and not r.get("loose") and (r.get("status") == STATUS or timbers.is_measured(r))
                and rk[0] == ("B" if r["side"] == "port" else "T")):
            strakes.setdefault((r["side"], rk[1]), []).append(r)
    take = _ids(chunk, EXTENSION)
    key = relief.frame_key(frame)
    out, skipped, cache = [], [], {}
    for i, ((side, rank), members) in enumerate(sorted(strakes.items())):
        name = _base_name(members[0]["name"])
        progress(i / max(1, len(strakes)), t(f"Prolungamento: {name}", f"Extension: {name}"))
        seams = []
        for r in (rank - 1, rank):
            if (side, r) not in cache:
                cache[(side, r)] = (_seam(chunk, frame, heights, records, recon, side, r) if r >= 1 else None,
                                    _trend(chunk, side, r, sheers))
            seams.append(cache[(side, r)])
        if any(s is None or s.fy is None for s, _tr in seams):
            skipped.append({"name": name, "side": side,
                            "reason": t("manca un comento su uno dei due lati", "a seam is missing on one edge")})
            continue
        polys = {r["id"]: _polygon(tracing.ship_xy_of(chunk, r, frame)) for r in members}
        gaps = _gaps(_intervals(polys.values()), lo, hi)
        sign = 1.0 if side == "port" else -1.0
        for a, b in gaps:
            xs = _grid(a, b)
            (s0, tr0), (s1, tr1) = seams
            lower, upper = _edge(s0, tr0, xs), _edge(s1, tr1, xs)
            band = _band(lower, upper, sign)
            if beyond is not None:
                band = band.difference(beyond)
            for piece in _pieces(band):
                if piece.area < MIN_AREA:
                    continue
                xy = _ring(piece)
                px0, _py0, px1, _py1 = piece.bounds
                source = min(polys, key=lambda rid: max(0.0, polys[rid].bounds[0] - px1, px0 - polys[rid].bounds[2]))
                at = (round(px0, 2), round(px1, 2))
                keep = (xs >= px0 - 1e-9) & (xs <= px1 + 1e-9)

                def signed(e, keep=keep, sign=sign):
                    e = e[keep].copy()
                    e[:, 1] *= sign
                    return [[float(v) if np.isfinite(v) else None for v in row] for row in e]

                out.append({"id": take(source, at), "kind": "plank", "side": side, "name": name, "station": "",
                            "status": EXTENSION, "source": source, "sources": sorted(polys), "ship_xy": xy.tolist(),
                            "geometry": timbers.geometry("polygon", _world(frame, xy)), "frame_key": key,
                            "chunk": chunk.key, "edges": {"lower": signed(lower), "upper": signed(upper)},
                            "seams": [rank - 1, rank], "trend": seams[0][1].kind if seams[0][1].kind == seams[1][1].kind
                            else f"{seams[0][1].kind}/{seams[1][1].kind}",
                            "beyond": bool(px0 < surveyed[0] - 1e-6 or px1 > surveyed[1] + 1e-6), "position": list(at)})
    _replace(chunk, EXTENSION, out)
    state(chunk)["extension"] = {"built": datetime.now(timezone.utc).isoformat(timespec="seconds"), "count": len(out),
                                 "skipped": skipped, "range": [lo, hi], "surveyed": list(surveyed), "frame_key": key}
    progress(1.0, t(f"Prolungamento: {len(out)} parti di fasce", f"Extension: {len(out)} parts of strakes"))
    return out


# ----------------------------------------------------------------------------- frames at the measured spacing
def _room(poly) -> float:
    """A frame's room: the short side of the rectangle around it."""
    import shapely

    with np.errstate(invalid="ignore", divide="ignore"):  # shapely's envelope on an axis-aligned rectangle
        box = shapely.get_coordinates(shapely.minimum_rotated_rectangle(poly))[:4]
    if len(box) < 4:
        x0, _y0, x1, _y1 = poly.bounds
        return float(x1 - x0)
    sides = np.hypot(*np.diff(np.vstack([box, box[:1]]), axis=0)[:2].T)
    return float(sides.min())


def _span(poly, side: str) -> tuple[float, float] | None:
    """A frame's outline across on ``side``: (|y| nearest the keel, |y| farthest), None when it has nothing there."""
    import shapely

    half = shapely.box(-_FAR, 0.0, _FAR, _FAR) if side == "port" else shapely.box(-_FAR, -_FAR, _FAR, 0.0)
    part = poly.intersection(half)
    if part.is_empty or part.area <= 0:
        return None
    y = np.abs(shapely.get_coordinates(part)[:, 1])
    return float(y.min()), float(y.max())


def _frame_number(name: str) -> int | None:
    m = _FRAME_NUMBER.match(str(name).strip())
    return int(m.group(1)) if m else None


def frames_by_spacing(chunk) -> list[dict]:
    """Frames added where they are missing (``reconstructed:spacing``): per side, the measured (not loose) and the
    mirrored frames in order of x (a centre frame counts for both sides; :func:`timbers.frame_spacing`); a space
    between two of them longer than :data:`GAP_FACTOR` median spaces gets ``round(space / median) - 1`` frames,
    evenly (the median space when the space is a multiple of it), each of the median room, its span across
    interpolated from the two neighbours. A frame added on both sides at one place between centre frames is one
    centre frame. Named ``ordinata N`` after the highest number (the existing numbers kept), ``notes`` saying where;
    ``source`` the frame before it, ``sources`` both. The earlier ones replaced; the medians in
    ``state["reconstruction"]["spacing"]``."""
    import shapely

    frame = ship.need_frame(chunk)
    records = timbers.timbers_of(chunk)
    recon = [r for r in records_of(chunk) if r.get("status") == STATUS]
    frames = [r for r in records + recon if r["kind"] == "frame" and r["geometry"]["type"] == "polygon"
              and not r.get("loose") and (r.get("status") == STATUS or timbers.is_measured(r))]
    polys = {r["id"]: _polygon(tracing.ship_xy_of(chunk, r, frame)) for r in frames}
    by_id = {r["id"]: r for r in frames}
    rows = [{"id": r["id"], "name": r["name"], "kind": "frame", "side": r["side"], "x": float(polys[r["id"]].centroid.x)}
            for r in frames if not polys[r["id"]].is_empty]
    spacing = timbers.frame_spacing(rows)
    found: list[dict] = []
    medians = {}
    for side, d in spacing.items():
        median = d["median"]
        if median is None or median <= 0 or len(d["frames"]) < 3:
            continue
        mine = [polys[f["id"]] for f in d["frames"]]
        room = float(np.median([_room(p) for p in mine]))
        medians[side] = {"space": median, "room": room}
        for left, right, space in zip(d["frames"][:-1], d["frames"][1:], d["spaces"]):
            if space <= GAP_FACTOR * median:
                continue
            n = max(1, int(round(space / median)) - 1)
            a, b = _span(polys[left["id"]], side), _span(polys[right["id"]], side)
            if a is None or b is None:
                continue
            for j in range(1, n + 1):
                f = j / (n + 1)
                found.append({"side": side, "x": left["x"] + f * space, "room": room, "left": left, "right": right,
                              "span": (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]))})
    # one centre frame where both sides miss it between the same centre frames
    merged, used = [], set()
    for i, p in enumerate(found):
        if i in used:
            continue
        twin = next((j for j, q in enumerate(found) if j not in used and j != i and q["side"] != p["side"]
                     and abs(q["x"] - p["x"]) < 0.5 * p["room"] and {q["left"]["id"], q["right"]["id"]}
                     == {p["left"]["id"], p["right"]["id"]}), None)
        both = (twin is not None and by_id[p["left"]["id"]]["side"] == "centre"
                and by_id[p["right"]["id"]]["side"] == "centre")
        if both:
            used.update((i, twin))
            q = found[twin]
            port, starboard = (p, q) if p["side"] == "port" else (q, p)
            merged.append({**p, "side": "centre", "x": 0.5 * (p["x"] + q["x"]),
                           "across": (-starboard["span"][1], port["span"][1])})
        else:
            used.add(i)
            sign = 1.0 if p["side"] == "port" else -1.0
            ys = sorted((sign * p["span"][0], sign * p["span"][1]))
            merged.append({**p, "across": tuple(ys)})
    merged.sort(key=lambda p: p["x"])
    others = records + [r for r in records_of(chunk) if r.get("status") != SPACING]
    number = 1 + max((n for n in (_frame_number(r.get("name", "")) for r in others) if n is not None), default=0)
    take = _ids(chunk, SPACING)
    key = relief.frame_key(frame)
    word = t("ordinata", "frame")
    out = []
    for p in merged:
        x, half = p["x"], 0.5 * p["room"]
        y0, y1 = p["across"]
        xy = _ring(shapely.box(x - half, y0, x + half, y1))
        left, right = p["left"], p["right"]
        at = (round(x, 2), round(y0, 2), round(y1, 2))
        out.append({"id": take(left["id"], at), "kind": "frame", "side": p["side"], "name": f"{word} {number}",
                    "station": "", "status": SPACING, "source": left["id"], "sources": [left["id"], right["id"]],
                    "ship_xy": xy.tolist(), "geometry": timbers.geometry("polygon", _world(frame, xy)),
                    "frame_key": key, "chunk": chunk.key, "room": p["room"], "position": list(at),
                    "notes": t(f"tra {left['name'] or left['id']} e {right['name'] or right['id']}, a x {x:.2f} m",
                               f"between {left['name'] or left['id']} and {right['name'] or right['id']}, "
                               f"at x {x:.2f} m")})
        number += 1
    _replace(chunk, SPACING, out)
    spaces = [m["space"] for m in medians.values()]
    state(chunk)["spacing"] = {"built": datetime.now(timezone.utc).isoformat(timespec="seconds"), "count": len(out),
                               "sides": medians, "median": float(np.median(spaces)) if spaces else None,
                               "frame_key": key}
    return out
