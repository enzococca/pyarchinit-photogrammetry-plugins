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
(``state["reconstruction"]["disagreement"]``).

The EXTENSION (``ricostruito: prolungamento``, :func:`extend`) continues each strake between two seams along the
hull with a fair spline through its seams, inside the surveyed length (:func:`surveyed_range`) and beyond it only to
the stem and stern the user gives (:func:`set_ends`); the FRAMES BY SPACING (``ricostruito: passo delle ordinate``,
:func:`frames_by_spacing`) add the frames missing between two frames at the measured room and space. Both are records
of the same list, each kind replaced as a whole when built again, without solids (the extension's edges keep their z,
None where nothing gives one). The order: the mirror first (the extension leaves its ground to it, and both refuse
a mirror older than the catalogue), then the extension and the frames.

The FAIRED HULL (``ricostruito: carenatura``, :func:`loft`, module :mod:`loft`) is lofted through the stations'
sections where neither side exists; :func:`build` (module :mod:`rebuild`) runs mirror, extension, frames by spacing
and loft in one go, builds the solids of the extended strakes and of the spaced frames on the hull's surface
(:func:`solids.surface_for`: the relief, else the mirrored solids, else the faired hull) and replaces the
reconstruction as a whole; :func:`table` and :func:`write_table` say what is measured and what is reconstructed.

SEVERAL CHUNKS of the same wreck (:mod:`chunks`: other campaigns or parts, aligned to the main chunk or sharing the
ship frame through markers of the same names) feed one reconstruction: :func:`build` merges their records
(:func:`chunks.merged`, passed as ``records`` to every step) and their reliefs (:func:`chunks.prepare`, the raster
every reader of the reconstruction reaches through :func:`height_path` / :func:`height_load`); only the main chunk's
state and folder are written.

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
LOFT_FILE = "carenatura.ply"  # the faired hull (:func:`loft`), kept beside the records' solids
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


def _records(chunk, records) -> list[dict]:
    return timbers.timbers_of(chunk) if records is None else list(records)


# ----------------------------------------------------------------------------- the height raster of the reconstruction
def height_path(chunk) -> Path:
    """The height raster the reconstruction reads: the merged one of several chunks (:func:`chunks.prepare`) when
    ``state["reconstruction"]["members"]`` names more than the main chunk and is current (every member still in the
    project, its relief the one recorded: elevation, frame, file), else the chunk's own relief."""
    from . import chunks

    members = (ship.state(chunk).get("reconstruction") or {}).get("members") or {}
    if members.get("raster") and len(members.get("chunks") or []) > 1:
        path = chunk.data_dir / members["raster"]
        if path.exists() and chunks.is_current(chunk, members):
            return path
    return relief.path_of(chunk, "height")


def height_load(chunk):
    """:func:`relief.load` of :func:`height_path`: the heights (NaN off the model) and the affine in (u, v)."""
    import rasterio

    with rasterio.open(height_path(chunk)) as src:
        arr = src.read(1)
        if src.nodata is not None and np.isfinite(src.nodata):
            arr = np.where(arr == src.nodata, np.nan, arr).astype(arr.dtype)
        return arr, src.transform


# ----------------------------------------------------------------------------- the relief
def opened_lost(model: np.ndarray) -> np.ndarray:
    """The cells without the model, opened by a square of :data:`OPENING` + 1 cells: strips of empty cells up to
    :data:`OPENING` wide (the model's own edge) are not lost; past the raster's edges all is lost."""
    from scipy import ndimage

    k = OPENING + 1
    return ndimage.binary_opening(~np.asarray(model, bool), structure=np.ones((k, k), bool), border_value=1)


class _Relief:
    """The relief's height raster, read once: the cells of a polygon of the ship plan, the parts where the model is
    not."""

    def __init__(self, chunk, frame: ship.Frame):
        tracing.need_fresh_relief(chunk)
        self.z, self.affine = height_load(chunk)
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
        lost = opened_lost(model)
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
        return solids.strake_shell(chunk, part, mm, path=height_path(chunk))
    return solids.timber_solid(chunk, part, path=height_path(chunk))


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
    remove_orphans(d, {f"{r['id']}.ply" for r in st["timbers"]} | {LOFT_FILE})


def remove_orphans(d: Path, keep: set) -> None:
    """The 3D files (and left-over temporary ones) of the reconstruction's folder not named in ``keep``."""
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


def mirror(chunk, progress=None, records=None) -> list[dict]:
    """The preserved side mirrored where the other side is lost (see the module): the earlier mirror replaced, the
    new records added to ``state["reconstruction"]["timbers"]``, their solids written (into a folder of their own
    first: whatever stops the mirror halfway leaves the earlier one as it was), each keeping the id of the earlier
    record from the same source at the same place (``position``: its plan bounds to the cm). Returns the new
    records; the timbers refused (no thickness, a solid that cannot be built) go to
    ``state["reconstruction"]["mirror"]``, with the fingerprint of the catalogue it read (``catalogue``). ``records``:
    the measured records to read (default the chunk's catalogue; :func:`chunks.merged` for several chunks)."""
    import shutil

    import shapely

    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    frame = ship.need_frame(chunk)
    if relief.entry(chunk) is None:
        raise NotReady(t("Costruisci prima il rilievo dello scafo.", "Build the relief of the hull first."))
    grid = _Relief(chunk, frame)
    records = _records(chunk, records)
    candidates = _candidates(records)
    st = state(chunk)
    take = _ids(chunk, STATUS)
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
                at = [round(float(v), 2) for v in (xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max())]
                rid = take(source["id"], at)
                label = name if len(pieces) == 1 else f"{name} ({j + 1})"
                try:
                    solid = _mirrored_solid(_original_solid(chunk, source, _reflect(xy), whole, frame), rid, label)
                    path = solids.write_ply(tmp / f"{rid}.ply", solid, frame)
                except Exception as exc:  # noqa: BLE001 - one timber refused, never the whole mirror
                    reason = str(exc) if isinstance(exc, NotReady) else f"{type(exc).__name__}: {exc}"
                    refused.append({"source": source["id"], "name": source["name"], "reason": reason})
                    continue
                record = {"id": rid, "kind": source["kind"], "side": side, "name": label,
                          "station": source.get("station", ""), "status": STATUS, "source": source["id"],
                          "ship_xy": xy.tolist(), "geometry": timbers.geometry("polygon", _world(frame, xy)),
                          "frame_key": key, "chunk": chunk.key, "clipped": not whole,
                          "solid": str((folder(chunk) / path.name).relative_to(chunk.data_dir)),
                          "volume": float(solid.volume), "position": at}
                if solid.thickness_mm is not None:
                    record["thickness_mm"] = float(solid.thickness_mm)
                out.append(record)
        _swap(chunk, out, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    st["mirror"] = {"built": datetime.now(timezone.utc).isoformat(timespec="seconds"), "count": len(out),
                    "refused": refused, "frame_key": key, "catalogue": _catalogue_key(chunk, records)}
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


def _seams(chunk, frame, side: str, x: float, records=None) -> dict[int, float]:
    """The |y| of the measured seams of ``side`` at ``x`` by rank: the seam records, and where a rank has none
    there the inner edge of the strake above it (strake k + 1 starts on seam k)."""
    out = {}
    records = _records(chunk, records)
    for r in records:
        if r["side"] != side or r.get("loose") or not timbers.is_measured(r):
            continue
        rank = _rank(r["name"])
        if rank is None:
            continue
        if r["kind"] == "seam" and r["geometry"]["type"] == "line":
            y = _at(tracing.ship_xy_of(chunk, r, frame), x)
            if y is not None:
                out[rank] = y
    for r in records:
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


def disagreement(chunk, records=None) -> list[dict]:
    """Per station (the sections' stations, else the chunk's) and seam rank where both sides have a measured seam
    (or strake edge): ``station``, ``x``, ``rank``, ``port_y`` and ``starboard_y`` (the half-breadths, |y|),
    ``difference_mm`` (port - starboard). Kept in ``state["reconstruction"]["disagreement"]`` (the table of
    :func:`table` reads them afresh)."""
    rows = disagreement_rows(chunk, records)
    state(chunk)["disagreement"] = rows
    return rows


def disagreement_rows(chunk, records=None) -> list[dict]:
    """The rows of :func:`disagreement`, kept nowhere."""
    frame = ship.need_frame(chunk)
    stations = _stations(chunk)
    if not stations:
        raise NotReady(t("Nessuna stazione: dai prima le stazioni.", "No stations: set the stations first."))
    rows = []
    for s in stations:
        port, starboard = (_seams(chunk, frame, "port", s["x"], records),
                           _seams(chunk, frame, "starboard", s["x"], records))
        for rank in sorted(set(port) & set(starboard)):
            p, q = port[rank], starboard[rank]
            if not (math.isfinite(p) and math.isfinite(q)):
                continue
            rows.append({"station": s["name"], "x": s["x"], "rank": rank, "port_y": p, "starboard_y": q,
                         "difference_mm": (p - q) * 1000.0})
    return rows


# ----------------------------------------------------------------------------- the ends and the surveyed length
EXTENSION = "reconstructed:extension"
SPACING = "reconstructed:spacing"
STEP = 0.05  # m: the bins along x of a seam's points, and of a mirrored strake's edges
GRID = 0.02  # m: the step along x of the extended edges
BLEND = 0.5  # m: past the last traced point the derivative goes over into the trend over this length
MIN_GAP = 0.05  # m: a shorter gap of a strake is not filled
END_FIT = 0.4  # m: the last stretch of a seam's points whose robust line gives its slope at the end
BORDER = 3  # cells: the relief's cells this close to the model's edge are not read for heights
BEND_FIT = 1.5  # m: the last stretch of a trend's points whose parabola gives its bending at the end
BEND_REACH = 1.0  # m: how far past its points a trend's derivative goes on bending (then held)
OPEN_X = 0.25  # m: heights read along a seam opened over this length (a timber's top narrower than it goes)
NO_TREND_REACH = 1.0  # m: how far a seam without a trend (no land line, no sections) goes past its points
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
    """The ship x range of the relief's cells with a value, both sides taken together (the surveyed wreck): the
    reconstruction's relief (:func:`height_load`: of all the chunks merged, when there are several)."""
    ship.need_frame(chunk)
    if relief.entry(chunk) is None:
        raise NotReady(t("Costruisci prima il rilievo dello scafo.", "Build the relief of the hull first."))
    z, affine = height_load(chunk)
    cols = np.flatnonzero(np.isfinite(z).any(axis=0))
    if not len(cols):
        raise NotReady(t("Il rilievo dello scafo è vuoto.", "The relief of the hull is empty."))
    u = affine.c + affine.a * (np.array([cols[0], cols[-1]], float) + 0.5)
    return float(u.min()), float(u.max())


def _limits(chunk, surveyed: tuple[float, float]):
    """How far the strakes may go: the surveyed range, each end given setting its own side (the side of the middle
    of the surveyed range it lies on: a stem typed inside the range shortens it); the regions past a traced end;
    the warnings (which end each one was taken for, an end inside the surveyed length)."""
    lo, hi = surveyed
    mid = 0.5 * (lo + hi)
    cuts, sides, warnings = [], set(), []
    names = {"stem": t("prua", "stem"), "stern": t("poppa", "stern")}
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
        where = t("x maggiore", "larger x") if side > 0 else t("x minore", "smaller x")
        warnings.append(t(f"La {names[key]} (x {x:.2f} m) è presa come l'estremità a {where}.",
                          f"The {names[key]} (x {x:.2f} m) is taken for the end at {where}."))
        if lo + GRID < x < hi - GRID:
            warnings.append(t(f"La {names[key]} (x {x:.2f} m) sta dentro la lunghezza rilevata: le fasce si "
                              "fermano lì.",
                              f"The {names[key]} (x {x:.2f} m) lies inside the surveyed length: the strakes stop "
                              "there."))
        if line is None:
            lo, hi = (lo, x) if side > 0 else (x, hi)
            continue
        if side > 0:
            hi = float(line[:, 0].max())
        else:
            lo = float(line[:, 0].min())
        cuts.append(_beyond(line, side))
    return lo, hi, cuts, warnings


def _beyond(line: np.ndarray, side: int):
    """The region of the plan past a traced end (``side`` +1: larger x): the line continued across at its ends."""
    import shapely

    pts = line[np.argsort(line[:, 1])]
    far = side * _FAR
    ring = [(pts[0, 0], -_FAR)] + [tuple(p) for p in pts] + [(pts[-1, 0], _FAR), (far, _FAR), (far, -_FAR)]
    poly = shapely.Polygon(ring)
    return poly if poly.is_valid else poly.buffer(0)


# ----------------------------------------------------------------------------- the curves along x
def _huber_slope(x: np.ndarray, v: np.ndarray) -> float:
    """The slope of a robust line through the points: Huber's weights (k = 1.345 sigma, sigma from the MAD),
    iterated, so that one point off the others (a border cell, a timber's top) does not tilt it."""
    a = np.column_stack([x - x.mean(), np.ones(len(x))])
    w = np.ones(len(x))
    coef = np.zeros(2)
    for _ in range(20):
        sw = np.sqrt(w)
        coef = np.linalg.lstsq(a * sw[:, None], v * sw, rcond=None)[0]
        r = v - a @ coef
        sigma = 1.4826 * float(np.median(np.abs(r - np.median(r))))
        k = 1.345 * max(sigma, 1e-6)
        w = np.minimum(1.0, k / np.maximum(np.abs(r), 1e-12))
    return float(coef[0])


class _Curve:
    """A value along x fitted on its points: a smoothing spline (GCV), a polynomial of degree <= 2 for a few points.
    :meth:`d` holds the end's derivative outside the points (a polynomial's derivative is never let run);
    :meth:`trend` lets it go on with the end's bending (a parabola through the last :data:`BEND_FIT` of points: the
    spline itself is natural, without bending at its ends) for :data:`BEND_REACH` at most, then holds it."""

    def __init__(self, x: np.ndarray, v: np.ndarray):
        from scipy.interpolate import make_smoothing_spline

        self.x, self.v = x, v
        self.lo, self.hi = float(x[0]), float(x[-1])
        self.spline = None
        if len(x) >= 5:
            try:
                self.spline = make_smoothing_spline(x, v)
                self.slope = self.spline.derivative()
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
        x = np.clip(np.asarray(x, float), self.lo, self.hi)
        if self.spline is not None:
            return self.slope(x)
        return np.polyval(np.polyder(self.poly), x) if len(self.poly) > 1 else np.zeros_like(x)

    def _bend(self, end: float) -> float:
        near = np.abs(self.x - end) <= BEND_FIT
        if near.sum() < 3 or np.ptp(self.x[near]) <= 0:
            return 0.0
        return float(2.0 * np.polyfit(self.x[near], self.v[near], 2)[0])

    def trend(self, x):
        x = np.asarray(x, float)
        e = np.clip(x, self.lo, self.hi)
        out = np.asarray(self.d(e), float)
        for end in (self.lo, self.hi):
            past = (x < self.lo) if end == self.lo else (x > self.hi)
            if np.any(past):
                reach = np.clip(x - end, -BEND_REACH, BEND_REACH)
                out = np.where(past, out + self._bend(end) * reach, out)
        return out

    def end_slope(self, end: float) -> float:
        """The slope at an end of the points: a robust line (:func:`_huber_slope`) over the points within
        :data:`END_FIT` of it; the curve's derivative with fewer than three."""
        near = np.abs(self.x - end) <= END_FIT
        if near.sum() < 3:
            return float(self.d(end))
        return _huber_slope(self.x[near], self.v[near])


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
    """Ship z read on the relief at plan points (bilinear): NaN off the model, within :data:`BORDER` cells of its
    edge (an open border of a model reads cells half on the void, centimetres off), or on a timber lying over the
    planking (the measured frames, keelson, stringers). Also the relief's cells, for the mirror's ground."""

    def __init__(self, chunk, frame: ship.Frame, records: list[dict]):
        import shapely
        from scipy import ndimage

        self.z, self.affine = height_load(chunk)
        self.cell = float(abs(self.affine.a))
        self.sign = frame.sign
        valid = ndimage.binary_erosion(np.isfinite(self.z), iterations=BORDER, border_value=0)
        self.read = np.where(valid, self.z, np.nan)
        cutters = [_polygon(tracing.ship_xy_of(chunk, r, frame)) for r in records
                   if r["kind"] in tracing.CUTTERS and r["geometry"]["type"] == "polygon" and not r.get("loose")]
        self.cutters = shapely.union_all(cutters).buffer(0.01) if cutters else None
        self._ground: dict[str, object] = {}

    def at(self, xy: np.ndarray) -> np.ndarray:
        import shapely
        from scipy.ndimage import map_coordinates

        xy = np.asarray(xy, float).reshape(-1, 2)
        if not len(xy):
            return np.zeros(0)
        inv = ~self.affine
        u, v = xy[:, 0], self.sign * xy[:, 1]
        col, row = inv.a * u + inv.b * v + inv.c, inv.d * u + inv.e * v + inv.f
        z = map_coordinates(self.read, [row - 0.5, col - 0.5], order=1, mode="constant", cval=np.nan)
        if self.cutters is not None:
            z[shapely.contains_xy(self.cutters, xy[:, 0], xy[:, 1])] = np.nan
        return z

    def ground(self, side: str):
        """The mirror's ground on ``side``: its cells without the model whose reflection across the keel line has
        it (where the other side is preserved and this one lost), opened by :data:`OPENING` cells as the mirror
        does, in the ship plan, grown by one cell. None when there is none."""
        if side in self._ground:
            return self._ground[side]
        import shapely
        from rasterio import features
        from scipy import ndimage

        a = self.affine
        present = np.isfinite(self.z)
        rows, cols = np.indices(present.shape)
        u = a.c + a.a * (cols + 0.5) + a.b * (rows + 0.5)
        v = a.f + a.d * (cols + 0.5) + a.e * (rows + 0.5)
        inv = ~a
        rc = np.floor(inv.a * u + inv.b * (-v) + inv.c).astype(int)
        rr = np.floor(inv.d * u + inv.e * (-v) + inv.f).astype(int)
        inside = (rr >= 0) & (rr < present.shape[0]) & (rc >= 0) & (rc < present.shape[1])
        mirrored = np.zeros_like(present)
        mirrored[inside] = present[rr[inside], rc[inside]]
        y = self.sign * v
        mine = y > 0 if side == "port" else y < 0
        lost = ndimage.binary_opening(~present & mirrored & mine, structure=np.ones((OPENING + 1, OPENING + 1), bool))
        poly = None
        if lost.any():
            shapes = [shapely.geometry.shape(g) for g, val in
                      features.shapes(lost.astype("uint8"), mask=lost, transform=a) if val == 1]
            uv = shapely.union_all(shapes)
            poly = shapely.transform(uv, lambda p: relief.ship_of_uv(p, self.sign)).buffer(self.cell)
        self._ground[side] = poly
        return poly


def _opened(x: np.ndarray, z: np.ndarray, width: float = OPEN_X) -> np.ndarray:
    """Heights along x opened (a running minimum, then a running maximum, each over ``width``): a raise narrower
    than ``width`` over the planking (a frame's top not in the catalogue, a stray block) goes, the planking's run
    stays (off by c w^2 / 8 where it bends with curvature c along x). NaN stays NaN, and is no part of a window."""
    z = np.asarray(z, float)
    out = np.full(len(z), np.nan)
    ok = np.isfinite(z) & np.isfinite(x)
    if ok.sum() < 3:
        return np.where(ok, z, np.nan)
    order = np.argsort(x[ok])
    xs, zs = x[ok][order], z[ok][order]
    lo = np.searchsorted(xs, xs - width / 2.0, "left")
    hi = np.searchsorted(xs, xs + width / 2.0, "right")
    low = np.array([zs[a:b].min() for a, b in zip(lo, hi)])
    opened = np.array([low[a:b].max() for a, b in zip(lo, hi)])
    idx = np.flatnonzero(ok)[order]
    out[idx] = opened
    return out


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
    """The points of the seam ``rank`` of a side (|y|, ship z by x): its measured seam records; when it has none,
    the edges of the strakes beside it (a mirrored strake's z read on the other side, where the wreck is)."""

    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z
        self.fy = _curve(x, y)
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
        zs.append(_opened(xy[:, 0], heights.at(xy)))
    if not seams:  # the strakes' edges (measured or mirrored) only when the seam was never traced
        planks = [(r, _strake_rank(r["name"])) for r in records + recon
                  if r["kind"] == "plank" and r["side"] == side and r["geometry"]["type"] == "polygon"
                  and not r.get("loose") and (r.get("status") == STATUS or timbers.is_measured(r))]
        above = any(rk == (code, rank + 1) for _r, rk in planks)
        for r, rk in planks:
            if rk not in ((code, rank), (code, rank + 1)) or (rk == (code, rank) and not above):
                continue
            xy = _edge_points(_polygon(tracing.ship_xy_of(chunk, r, frame)), outer=rk == (code, rank))
            if not len(xy):
                continue
            xs.append(xy[:, 0])
            ys.append(xy[:, 1])
            zs.append(_opened(xy[:, 0], heights.at(xy * [1.0, -1.0] if r.get("status") == STATUS else xy)))
    if not xs:
        return None
    x, y, z = _binned(np.concatenate(xs), np.concatenate(ys), np.concatenate(zs))
    return _Seam(x, y, z) if len(x) >= 2 else None


# ----------------------------------------------------------------------------- the trends
class _Trend:
    """Where a seam goes past its points: the 0.2.0 land line of its rank on its side, else the stations' sheer
    (|y| in proportion to the seam's at its end, z as the sheer's), else ``none`` (the seam's own end slope, for
    :data:`NO_TREND_REACH` at most)."""

    def __init__(self, kind: str, fy=None, fz=None):
        self.kind, self.fy, self.fz = kind, fy, fz

    def dy(self, x, anchor_x: float, anchor_y: float):
        d = self.fy.trend(x)
        if self.kind == "sheer":
            ref = float(self.fy(np.clip(anchor_x, self.fy.lo, self.fy.hi)))
            d = d * (anchor_y / ref if ref > 1e-6 else 1.0)
        return d

    def dz(self, x):
        return self.fz.trend(x)


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
    return _Trend("none")


# ----------------------------------------------------------------------------- the extended edges
def _grid(lo: float, hi: float) -> np.ndarray:
    return np.unique(np.clip(np.r_[np.arange(lo, hi, GRID), hi], lo, hi))


def _go_on(value: float, slope: float, end: float, xs: np.ndarray, trend) -> np.ndarray:
    """A value continued from ``value`` at ``end`` over ``xs`` (all on one side of it, ordered away from it): the
    slope at ``end`` going linearly over :data:`BLEND` into ``trend(x)`` (None: held)."""
    out = np.empty(len(xs))
    v, x0 = float(value), end
    for i, x in enumerate(xs):
        m = 0.5 * (x0 + x)
        w = min(abs(m - end) / BLEND, 1.0)
        d = slope if trend is None else (1.0 - w) * slope + w * float(trend(m))
        v += d * (x - x0)
        out[i], x0 = v, x
    return out


def _continued(curve: _Curve, xs: np.ndarray, trend) -> np.ndarray:
    """``curve`` over ``xs``: inside its points the curve, past them :func:`_go_on` from its end with its robust end
    slope (:meth:`_Curve.end_slope`) into ``trend(x, end, value)``; with no trend (None) the slope goes on with the
    curve's own bending at that end (:meth:`_Curve.trend`'s parabola) for :data:`BEND_REACH`, then is held."""
    out = np.full(len(xs), np.nan)
    inside = (xs >= curve.lo) & (xs <= curve.hi)
    out[inside] = curve(xs[inside])
    for part, end in ((xs < curve.lo, curve.lo), (xs > curve.hi, curve.hi)):
        if not part.any():
            continue
        order = np.argsort(np.abs(xs[part] - end))
        value, slope = float(curve(end)), curve.end_slope(end)
        bend = curve._bend(end) if trend is None else 0.0

        def tr(m, e=end, v=value, s0=slope, b=bend):
            if trend is None:
                return s0 + b * float(np.clip(m - e, -BEND_REACH, BEND_REACH))
            return trend(m, e, v)

        out[np.flatnonzero(part)[order]] = _go_on(value, slope, end, xs[part][order], tr)
    return out


def _edge(seam: _Seam, trend: _Trend, xs: np.ndarray, heights: _Heights, sign: float) -> np.ndarray:
    """The seam over ``xs``: (x, |y|, z).

    - |y|: the seam's spline inside its points, past them continued (:func:`_continued`) into the trend's |y|;
    - z: read on the relief at the edge wherever the model is (outside the timbers over the planking and the
      model's border); elsewhere a spline through those heights and the seam's own, past them continued into the
      trend's z (held without one); NaN when nothing gives a height. The stations' sections are cuts of the same
      model as the relief: where it has no height they have none either."""
    y = np.maximum(_continued(seam.fy, xs, trend.dy if trend.fy is not None else None), 0.0)
    read = _opened(xs, heights.at(np.column_stack([xs, sign * y])))
    zx = np.r_[seam.x, xs]
    zv = np.r_[seam.z, read]
    ok = np.isfinite(zv)
    z = np.full(len(xs), np.nan)
    if ok.sum() >= 2:
        bx, _by, bz = _binned(zx[ok], np.zeros(int(ok.sum())), zv[ok])
        fz = _curve(bx, bz)
        if fz is not None:
            dz = (lambda m, _e, _v: trend.dz(m)) if trend.fz is not None else None
            z = _continued(fz, xs, dz)
    z = np.where(np.isfinite(read), read, z)
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


def _edge_of_piece(e: np.ndarray, piece, sign: float, beyond) -> list:
    """An edge (x, |y|, z) kept over a piece's x and, with a traced end, cut where it crosses the end's line (the
    crossing added): ``[[x, y, z], ...]`` in the ship plan, z None where there is no height."""
    import shapely

    x0, _y0, x1, _y1 = piece.bounds
    keep = (e[:, 0] >= x0 - 1e-9) & (e[:, 0] <= x1 + 1e-9)
    e = e[keep]
    xy = np.column_stack([e[:, 0], sign * e[:, 1]])
    if beyond is not None and len(xy) >= 2:
        parts = [g for g in getattr(shapely.LineString(xy).difference(beyond), "geoms",
                                    [shapely.LineString(xy).difference(beyond)])
                 if g.geom_type == "LineString" and not g.is_empty]
        if parts:
            xy = np.asarray(max(parts, key=lambda g: g.length).coords, float)[:, :2]
        else:
            xy = xy[:0]
    ok = np.isfinite(e[:, 2])
    z = (np.interp(xy[:, 0], e[ok, 0], e[ok, 2]) if ok.sum() >= 2 else np.full(len(xy), np.nan))
    if ok.sum() >= 2:
        z[(xy[:, 0] < e[ok, 0].min() - GRID) | (xy[:, 0] > e[ok, 0].max() + GRID)] = np.nan
    return [[float(x), float(y), float(v) if np.isfinite(v) else None] for (x, y), v in zip(xy, z)]


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


def _catalogue_key(chunk, records=None) -> str:
    """A fingerprint of the catalogue's records as the mirror reads them (ids, kinds, sides, names, loose, status,
    geometry): another one later means the mirror is older than the records."""
    import hashlib
    import json

    rows = sorted([r["id"], r["kind"], r["side"], r["name"], bool(r.get("loose")), timbers.status_of(r),
                   np.round(np.asarray(r["geometry"]["xy"], float), 4).tolist()] for r in _records(chunk, records))
    return hashlib.sha1(json.dumps(rows).encode("utf-8")).hexdigest()


def _current_mirror(chunk, frame: ship.Frame, records=None) -> str | None:
    """The mirror's build stamp (None: never built); :class:`NotReady` when it is older than the catalogue or was
    made in another ship frame (build it again: what follows reads it)."""
    m = state(chunk).get("mirror")
    if m is None:
        return None
    if m.get("catalogue") != _catalogue_key(chunk, records) or m.get("frame_key") != relief.frame_key(frame):
        raise NotReady(t("La simmetria è più vecchia del catalogo dei legni: rifai prima la simmetria.",
                         "The mirror is older than the catalogue of timbers: build the mirror again first."))
    return m.get("built")


def extend(chunk, progress=None, records=None) -> list[dict]:
    """Each strake between two seams (measured, or mirrored: its edges are its seams) continued along the hull.

    - Its seams' |y| fitted in x by a smoothing spline (GCV) over the strake's gaps (x not covered by its measured
      records) inside :func:`surveyed_range` and, beyond it, only up to the ends :func:`set_ends` gave.
    - Past the last point of a seam the slope of a robust line through its last :data:`END_FIT` goes over linearly
      in :data:`BLEND` into the trend of the 0.2.0 land line of its rank, else of the stations' sheer; with neither
      (``trend`` "none") a seam goes on :data:`NO_TREND_REACH` at most and the piece is ``uncertain``; a piece
      reaching more than :data:`BEND_REACH` past both its seam's points and its trend's (where the slope is only
      held) is ``uncertain`` too.
    - z read on the relief along the edge wherever the model is (:func:`_edge`); past the model continued as |y|.
    - Never over the mirror's ground (:meth:`_Heights.ground`: this side lost, the other preserved): that is the
      mirror's to fill, whichever is built first; :class:`NotReady` while the mirror is older than the catalogue.

    The new parts only, as band polygons (``reconstructed:extension``, named as the strake, ``source`` the
    strake's record nearest, ``sources``, ``edges`` lower/upper ``[[x, y, z], ...]`` in the ship plan (z None where
    nothing gives a height), ``seams``, ``trend``, ``uncertain``, ``beyond``: past the surveyed range, ``position``);
    the earlier extension replaced (ids kept for the same source and position). No solid yet. The order to build:
    mirror, then extension and frames (``state["extension"]["mirror_built"]`` says which mirror it read).
    ``records``: as :func:`mirror`'s."""
    import shapely

    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    frame = ship.need_frame(chunk)
    records = _records(chunk, records)
    mirror_built = _current_mirror(chunk, frame, records)
    surveyed = surveyed_range(chunk)
    tracing.need_fresh_relief(chunk)
    lo, hi, cuts, warnings = _limits(chunk, surveyed)
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
        measured = [polys[r["id"]] for r in members if r.get("status") != STATUS]
        mirrored = [polys[r["id"]] for r in members if r.get("status") == STATUS]
        # where a seam has no trend it goes on NO_TREND_REACH at most
        a_lim, b_lim = lo, hi
        for s, tr in seams:
            if tr.kind == "none":
                a_lim, b_lim = max(a_lim, s.lo - NO_TREND_REACH), min(b_lim, s.hi + NO_TREND_REACH)
        sign = 1.0 if side == "port" else -1.0
        away = [g for g in (beyond, heights.ground(side)) if g is not None]
        away += [p.buffer(heights.cell) for p in mirrored]
        away = shapely.union_all(away) if away else None
        (s0, tr0), (s1, tr1) = seams
        for a, b in _gaps(_intervals(measured), a_lim, b_lim):
            xs = _grid(a, b)
            lower, upper = _edge(s0, tr0, xs, heights, sign), _edge(s1, tr1, xs, heights, sign)
            band = _band(lower, upper, sign)
            if away is not None:
                band = band.difference(away)
            for piece in _pieces(band):
                if piece.area < MIN_AREA:
                    continue
                xy = _ring(piece)
                px0, _py0, px1, _py1 = piece.bounds
                source = min(polys, key=lambda rid: max(0.0, polys[rid].bounds[0] - px1, px0 - polys[rid].bounds[2]))
                at = (round(px0, 2), round(px1, 2))
                guessed = any(tr.kind == "none" and (px0 < s.lo - GRID or px1 > s.hi + GRID) for s, tr in seams)
                # past both the seam's points and its trend's points by more than the trend's bending: held only
                held = any(tr.fy is not None and (px0 < min(s.lo, tr.fy.lo) - BEND_REACH - GRID
                                                  or px1 > max(s.hi, tr.fy.hi) + BEND_REACH + GRID)
                           for s, tr in seams)
                kinds = sorted({tr0.kind, tr1.kind})
                out.append({"id": take(source, at), "kind": "plank", "side": side, "name": name, "station": "",
                            "status": EXTENSION, "source": source, "sources": sorted(polys), "ship_xy": xy.tolist(),
                            "geometry": timbers.geometry("polygon", _world(frame, xy)), "frame_key": key,
                            "chunk": chunk.key,
                            "edges": {"lower": _edge_of_piece(lower, piece, sign, beyond),
                                      "upper": _edge_of_piece(upper, piece, sign, beyond)},
                            "seams": [rank - 1, rank], "trend": "/".join(kinds), "uncertain": bool(guessed or held),
                            "beyond": bool(px0 < surveyed[0] - 1e-6 or px1 > surveyed[1] + 1e-6),
                            "position": list(at)})
                if guessed:
                    warnings.append(t(f"{name}: nessuna linea dei gradini né sezioni, prolungata al più di "
                                      f"{NO_TREND_REACH:g} m (incerta)",
                                      f"{name}: no land line and no sections, extended {NO_TREND_REACH:g} m at most "
                                      "(uncertain)"))
                if held:
                    warnings.append(t(f"{name}: oltre {BEND_REACH:g} m dopo gli ultimi punti del comento e della sua "
                                      "tendenza la pendenza è solo tenuta (incerta)",
                                      f"{name}: more than {BEND_REACH:g} m past the last points of the seam and of its "
                                      "trend the slope is only held (uncertain)"))
    _replace(chunk, EXTENSION, out)
    state(chunk)["extension"] = {"built": datetime.now(timezone.utc).isoformat(timespec="seconds"), "count": len(out),
                                 "skipped": skipped, "warnings": list(dict.fromkeys(warnings)), "range": [lo, hi],
                                 "surveyed": list(surveyed), "frame_key": key, "mirror_built": mirror_built}
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


def frames_by_spacing(chunk, records=None) -> list[dict]:
    """Frames added where they are missing (``reconstructed:spacing``): per side, the measured (not loose) and the
    mirrored frames in order of x (a centre frame counts for both sides; :func:`timbers.frame_spacing`); a space
    between two of them longer than :data:`GAP_FACTOR` median spaces gets ``round(space / median) - 1`` frames,
    evenly (the median space when the space is a multiple of it), each of the median room, its span across
    interpolated from the two neighbours. A frame added on both sides at one place between centre frames is one
    centre frame. Named ``ordinata N`` after the highest number (the existing numbers kept), ``notes`` saying where;
    ``source`` the frame before it, ``sources`` both. The earlier ones replaced (ids kept for the same source and
    position); the medians and the mirror read (``mirror_built``) in ``state["reconstruction"]["spacing"]``;
    :class:`NotReady` while the mirror is older than the catalogue. ``records``: as :func:`mirror`'s."""
    import shapely

    frame = ship.need_frame(chunk)
    records = _records(chunk, records)
    mirror_built = _current_mirror(chunk, frame, records)
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
                               "frame_key": key, "mirror_built": mirror_built}
    return out


# ----------------------------------------------------------------------------- the faired hull, the build, the table
from .loft import loft  # noqa: E402, F401 - re-exported: reconstruct.loft
from .rebuild import build, table, write_table  # noqa: E402, F401 - re-exported
