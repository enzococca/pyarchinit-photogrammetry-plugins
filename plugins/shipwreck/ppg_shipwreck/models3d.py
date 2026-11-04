"""The 3D files of the traced timbers (the solids of :mod:`solids`): OBJ, GLB, IFC and the tables.

Every timber is one object: an OBJ object, a GLB node, an IFC element, named after the record (its name, else its id;
a name shared by two records takes the id). A reconstructed part is never presented as measured: a timber whose
status is not ``measured`` (:data:`timbers.STATUS`; the typed keel is ``reconstructed:given``) has the suffix
:data:`RIC` on its object's and its material's name, a lighter colour, half opacity, and its status in the IFC's
``Pset_PyArchInit_Timber`` and in the tables.

Two systems:

- ``"ship"``: (x, sign * y, z) of the ship frame (:class:`ship.Frame`): x along the keel line, y on the left of who
  walks along x (to starboard when x runs aft, to port when it runs forward), z up from the datum. The ship frame itself is a mirror image of a
  right-handed one when the bow is at the start of the keel line (``sign`` -1); turning y over there makes the files
  right-handed for both bows (a mirror would turn every solid inside out in a BIM or 3D program).
- ``"world"``: the chunk's coordinates (:meth:`ship.Frame.to_world`): double precision in the OBJ; the GLB holds
  float32, so a round origin is taken off and written in the scene's ``extras``; the IFC writes them relative to a
  round origin given in its ``IfcMapConversion``.

The faces of the solids are turned outward in the ship frame: where the change of system is a mirror they are turned
over, so a solid has a positive volume in every file. The IFC is IFC4 in metres: strakes and the ceiling are
``IfcPlate``, the other timbers ``IfcMember``, both USERDEFINED with ``ObjectType`` = the kind's English title, each
an ``IfcTriangulatedFaceSet``; ``IfcMapConversion`` when the chunk has a projected CRS (in the ship system its
eastings, northings and rotation are the ship frame's); GlobalIds stable across exports and systems
(:func:`stable_global_id` of ``(chunk key, "shipwreck", record id)``, in the namespace of the plugin suite).

The reconstruction of :mod:`reconstruct` (0.4.0) goes in with ``reconstruction=True`` (asked for: the default is False): after the
catalogue's timbers (and the typed keel) every reconstruction record with a solid (its name and :data:`RIC`, its own
status: ``ricostruito: simmetria`` ...), then the faired hull ``carenatura_ric`` (kind ``hull``, an open surface, an
``IfcMember`` with ``ObjectType`` "carenatura (ipotesi)", a face set not ``Closed``, no volume in its Pset), all
translucent like every hypothesis; GlobalIds stable from
the record's id ("carenatura" for the faired hull); refused (:class:`NotReady`) when it was built in another ship
frame, its solids lying elsewhere. ``reconstruction=False``, the default everywhere, writes what 0.3.0 wrote.

Only the files asked for are written (and the OBJ's ``.mtl`` beside it): the records, the chunk's transform, model and
products are never changed.
"""
from __future__ import annotations

import contextlib
import csv
import json
import locale
import re
import uuid
from pathlib import Path

import numpy as np

from . import relief, ship, solids, timbers
from .ship import NotReady
from .texts import t

KIND_COLOURS = {"keel": "#8b3a1a", "keelson": "#c0392b", "frame": "#d68910", "plank": "#2e86c1", "ceiling": "#17a589",
                "stringer": "#7d3c98", "other": "#7f8c8d"}
HULL_COLOUR = "#5d6d7e"  # the faired hull (kind "hull": no timber of the catalogue)
RIC = "_ric"  # the suffix of the objects and materials of a reconstructed part
SYSTEMS = ("ship", "world")
PLATES = ("plank", "ceiling")  # IfcPlate; the others IfcMember
HYPOTHESIS_ALPHA = 0.5
TABLE_COLUMNS = timbers.COLUMNS + ("volume", "thickness_mm", "hidden_share", "solid")
TABLE_NAME = "timbers_3d"
PSET = "Pset_PyArchInit_Timber"

# The plugin suite's namespace of stable GlobalIds, the same in masonry and conservation (their ``ifc.py``): an
# element exported again replaces the old one in a BIM model instead of adding a second one.
SUITE_NAMESPACE = uuid.UUID("6f1c2b0e-5a1d-4c8e-9b7a-3d2e1f0c9b8a")


def stable_global_id(*parts) -> str:
    """The IFC GlobalId of the parts joined by ``/`` (a UUID 5 in the suite's namespace, compressed)."""
    import ifcopenshell.guid

    return ifcopenshell.guid.compress(uuid.uuid5(SUITE_NAMESPACE, "/".join(str(p) for p in parts)).hex)


@contextlib.contextmanager
def c_numbers():
    """Numbers written with a decimal point while IfcOpenShell works: Qt sets the system's ``LC_NUMERIC`` (a decimal
    comma in Italian) and IfcOpenShell then reads "1.5" as 1 (as masonry's ``c_numbers``)."""
    try:
        old = locale.setlocale(locale.LC_NUMERIC)
    except locale.Error:  # pragma: no cover
        old = None
    try:
        locale.setlocale(locale.LC_NUMERIC, "C")
        yield
    finally:
        if old is not None:
            try:
                locale.setlocale(locale.LC_NUMERIC, old)
            except locale.Error:  # pragma: no cover
                pass


# ----------------------------------------------------------------------------- colours and systems
def _rgb(colour: str) -> tuple[float, float, float]:
    c = colour.lstrip("#")
    return tuple(int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def lighter(colour: str) -> str:
    """``colour`` (``#rrggbb``) halfway to white."""
    c = colour.lstrip("#")
    return "#" + "".join(f"{int(round((int(c[i:i + 2], 16) + 255) / 2.0)):02x}" for i in (0, 2, 4))


def colour_of(kind: str, status: str) -> str:
    base = HULL_COLOUR if kind == "hull" else KIND_COLOURS.get(kind, KIND_COLOURS["other"])
    return base if status == "measured" else lighter(base)


def material_of(kind: str, status: str) -> str:
    kind = kind if kind in KIND_COLOURS or kind == "hull" else "other"
    return kind if status == "measured" else kind + RIC


def _system(system: str) -> str:
    if system not in SYSTEMS:
        raise ValueError(f"system: one of {', '.join(SYSTEMS)}, not {system!r}")
    return system


def coords(chunk, xyz_ship, system: str) -> np.ndarray:
    """Ship-frame ``(N, 3)`` points in ``system``: "ship" = (x, sign * y, z), right-handed for both bows; "world" =
    the chunk's coordinates (:meth:`ship.Frame.to_world`)."""
    _system(system)
    frame = ship.need_frame(chunk)
    xyz = np.asarray(xyz_ship, float).reshape(-1, 3)
    if system == "world":
        return frame.to_world(xyz)
    return np.column_stack([xyz[:, 0], frame.sign * xyz[:, 1], xyz[:, 2]])


def _turned(chunk, faces: np.ndarray) -> np.ndarray:
    """The faces of a solid turned outward in either system: both are a mirror of the ship frame when its sign is -1
    (in "ship" y turns over; the world is right-handed and the ship frame is its mirror)."""
    return faces[:, ::-1].copy() if ship.need_frame(chunk).sign < 0 else faces


# ----------------------------------------------------------------------------- the timbers to write
def _safe(text: str) -> str:
    """A name usable in every format: no spaces nor punctuation that readers split on."""
    return re.sub(r"[^\w.-]+", "_", str(text), flags=re.UNICODE).strip("_")


def _check_solids(chunk) -> dict:
    frame = ship.need_frame(chunk)
    done = ship.state(chunk).get("solids")
    if not done:
        raise NotReady(t("Costruisci prima i legni in 3D.", "Build the timbers in 3D first."))
    if done.get("frame_key") != relief.frame_key(frame):
        raise NotReady(t("I legni in 3D sono di un altro sistema della nave: costruiscili di nuovo.",
                         "The timbers in 3D belong to another ship frame: build them again."))
    return done


def _read(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from pyarchinit_photogrammetry.io.mesh_ply import PlyMesh

    if not path.exists():
        raise NotReady(t(f"Manca il file {path.name} dei legni in 3D: costruiscili di nuovo.",
                         f"The file {path.name} of the timbers in 3D is missing: build them again."))
    mesh = PlyMesh(path)
    v = mesh.xyz()
    f = np.concatenate(list(mesh.face_chunks())) if mesh.n_faces else np.zeros((0, 3), np.int64)
    hidden = (np.asarray(mesh._faces["hidden"]).astype(bool) if "hidden" in mesh.face_dtype.names
              else np.zeros(len(f), bool))
    del mesh
    return v, f, hidden


def _hidden_share(v: np.ndarray, f: np.ndarray, hidden: np.ndarray) -> float:
    """The share of the surface of a solid interpolated under other timbers (by area)."""
    if not len(f):
        return 0.0
    area = 0.5 * np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)
    total = float(area.sum())
    return float(area[hidden].sum() / total) if total > 0 else 0.0


def _sources(record: dict) -> str:
    sources = record.get("sources")
    if sources:
        return ", ".join(str(s) if not isinstance(s, dict) else json.dumps(s, sort_keys=True) for s in sources)
    return str(record.get("source") or "")


def _names(entries: list[dict], taken: set[str] | None = None) -> set[str]:
    """Unique object names: the record's name (made safe), with its id where two share it (or where the name is in
    ``taken``, the names already given); :data:`RIC` for a reconstructed part. Returns the names taken."""
    bases = [_safe(e["label"]) or _safe(e["id"]) for e in entries]
    given = set(taken or ())
    taken = set(given)
    for e, base in zip(entries, bases):
        ric = "" if e["status"] == "measured" else RIC
        if bases.count(base) > 1 or base + ric in given:
            base = f"{base}_{_safe(e['id'])}"
        name, n = base + ("" if e["status"] == "measured" else RIC), 1
        while name in taken:  # a disambiguated name equal to another record's own name
            n += 1
            name = f"{base}_{n}" + ("" if e["status"] == "measured" else RIC)
        taken.add(name)
        e["name"] = name
    return taken


def items(chunk, system: str = "ship", reconstruction: bool = False) -> list[dict]:
    """The timbers built in 3D, in ``system``: ``[{"id", "name" (the object's), "label" (the record's name), "kind",
    "side", "status", "material", "colour", "vertices", "faces" (outward), "hidden", "volume", "thickness_mm",
    "hidden_share", "record" (None for the typed keel and the faired hull), "ship_vertices"}]``; the typed keel after
    the catalogue, id "keel"; with ``reconstruction``, then the reconstruction (:func:`_reconstruction_items`).
    :class:`NotReady` when the solids were not built, were built in another ship frame or a file is missing."""
    _system(system)
    _check_solids(chunk)
    out = []
    for r in timbers.timbers_of(chunk):
        if r["geometry"]["type"] != "polygon" or not r.get("solid"):
            continue
        v, f, hidden = _read(chunk.data_dir / r["solid"])
        status = timbers.status_of(r)
        out.append({"id": r["id"], "label": r.get("name") or r["id"], "kind": r["kind"], "side": r.get("side", ""),
                    "status": status, "ship_vertices": v, "faces": f, "hidden": hidden,
                    "volume": float(r.get("volume") or solids._signed_volume(v, f)),
                    "thickness_mm": r.get("thickness_mm") if r["kind"] in solids.SHELLS else None, "record": r})
    keel = ship.state(chunk).get("keel") or {}
    if keel.get("solid"):
        v, f, hidden = _read(chunk.data_dir / keel["solid"])
        out.append({"id": "keel", "label": t("chiglia", "keel"), "kind": "keel", "side": "centre",
                    "status": keel.get("status", "reconstructed:given"), "ship_vertices": v, "faces": f,
                    "hidden": hidden, "volume": float(keel.get("volume") or solids._signed_volume(v, f)),
                    "thickness_mm": None, "record": None})
    taken = _names(out)
    if reconstruction:
        more = _reconstruction_items(chunk)
        _names(more, taken)
        out += more
    for e in out:
        e["vertices"] = coords(chunk, e["ship_vertices"], system)
        e["faces"] = _turned(chunk, e["faces"])
        e["material"] = material_of(e["kind"], e["status"])
        e["colour"] = colour_of(e["kind"], e["status"])
        e["hidden_share"] = _hidden_share(e["ship_vertices"], e["faces"], e["hidden"])
    return out


def check_reconstruction(chunk) -> None:
    """:class:`NotReady` when the stored reconstruction was built in another ship frame than the chunk's now: its solids
    lie in that frame and would stand elsewhere in the 3D files (other changes only make it out of date:
    :func:`reconstruct.stale`)."""
    st = (chunk.meta.get(ship.KEY) or {}).get("reconstruction") or {}
    if (st.get("timbers") or (st.get("loft") or {}).get("file")) and \
            st.get("frame_key") != relief.frame_key(ship.need_frame(chunk)):
        raise NotReady(t("La ricostruzione è di un altro sistema della nave: ricostruisci (passo 9) prima di esportarla "
                         "in 3D.",
                         "The reconstruction belongs to another ship frame: reconstruct (step 9) before exporting it "
                         "in 3D."))


def _reconstruction_items(chunk) -> list[dict]:
    """The reconstruction's records with a solid (their status, ``sources``), then the faired hull (id
    "carenatura", kind "hull", volume 0: an open surface), in the ship frame, unnamed yet. :class:`NotReady` when the
    reconstruction was built in another ship frame than the chunk's now (a bow swapped, the keel traced again): its
    solids lie in that frame, and placed with this one they would stand elsewhere."""
    from . import loft as loft_mod

    st = ship.state(chunk).get("reconstruction") or {}
    if not (st.get("timbers") or (st.get("loft") or {}).get("file")):
        return []
    check_reconstruction(chunk)
    out = []
    for r in st.get("timbers", []):
        if not r.get("solid") or not (chunk.data_dir / r["solid"]).exists():
            continue
        v, f, hidden = _read(chunk.data_dir / r["solid"])
        out.append({"id": r["id"], "label": r.get("name") or r["id"], "kind": r["kind"], "side": r.get("side", ""),
                    "status": r.get("status") or timbers.status_of(r), "ship_vertices": v, "faces": f,
                    "hidden": hidden, "volume": float(r.get("volume") or solids._signed_volume(v, f)),
                    "thickness_mm": r.get("thickness_mm") if r["kind"] in solids.SHELLS else None, "record": r})
    loft = st.get("loft") or {}
    if loft.get("file") and (chunk.data_dir / loft["file"]).exists():
        v, f, hidden = _read(chunk.data_dir / loft["file"])
        out.append({"id": "carenatura", "label": loft_mod.NAME, "kind": loft_mod.KIND, "side": "centre",
                    "status": loft_mod.STATUS, "ship_vertices": v, "faces": f, "hidden": hidden, "volume": 0.0,
                    "thickness_mm": None, "record": None,
                    "sources": t("sezioni alle stazioni", "the sections at the stations")})
    return out


def _not_built(chunk) -> list[str]:
    return [r["id"] for r in timbers.timbers_of(chunk) if r["geometry"]["type"] == "polygon" and not r.get("solid")]


def _header(chunk, system: str, origin=None) -> dict:
    frame = ship.need_frame(chunk)
    axes = ("ship: x along the keel line, y = sign * the ship frame's y (to starboard when x runs aft, to port "
            "when it runs forward: right-handed for both bows), z up from the datum" if system == "ship" else
            "world: the chunk's coordinates" + (" minus origin" if origin is not None else ""))
    out = {"system": system, "axes": axes, "crs": chunk.crs, "ship_frame": frame.to_dict(), "sign": frame.sign,
           "hypotheses": f"objects and materials ending in {RIC}: reconstructed, not measured"}
    if origin is not None:
        out["origin"] = [float(v) for v in origin]
    return out


# ----------------------------------------------------------------------------- OBJ
def _rows_text(fmt: str, values: np.ndarray, block: int = 100_000) -> str:
    """The rows of ``values`` formatted with ``fmt`` (one row, its newline included), without the final newline:
    formatted a block at a time in one operation (a line at a time is several times slower on a million faces)."""
    values = np.asarray(values)
    flat = values.astype(np.int64) if "%d" in fmt else values.astype(float)
    parts = [(fmt * len(rows)) % tuple(rows.ravel().tolist()) for rows in
             (flat[i:i + block] for i in range(0, len(flat), block))]
    return "".join(parts).rstrip("\n")


def write_obj(chunk, path, system: str = "ship", reconstruction: bool = False) -> Path:
    """An OBJ of the timbers in ``system``: ``o <name>`` per timber, ``usemtl <kind>`` (``<kind>_ric`` for a
    hypothesis), the ``.mtl`` beside it (``Kd`` per kind, ``d 0.5`` and a lighter colour for the hypotheses),
    comments with the system and the ship frame; ``reconstruction``: the reconstruction too (:func:`items`). Returns
    the OBJ's path."""
    entries = items(chunk, system, reconstruction)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mtl = path.with_suffix(".mtl")
    head = _header(chunk, system)
    lines = ["# ppg_shipwreck timbers in 3D", f"# system {system}", f"# axes {head['axes']}",
             f"# crs {head['crs']}", f"# ship_frame {json.dumps(head['ship_frame'])}", f"# {head['hypotheses']}"]
    missing = _not_built(chunk)
    if missing:
        lines.append(f"# not built: {', '.join(missing)}")
    lines.append(f"mtllib {mtl.name}")
    offset = 1
    used = {}
    for e in entries:
        used[e["material"]] = (e["colour"], e["status"] != "measured")
        lines += [f"o {e['name']}", f"# id {e['id']} kind {e['kind']} status {e['status']}", f"usemtl {e['material']}"]
        lines.append(_rows_text("v %.6f %.6f %.6f\n", e["vertices"]))
        lines.append(_rows_text("f %d %d %d\n", e["faces"] + offset))
        offset += len(e["vertices"])
    materials = ["# ppg_shipwreck timbers: a material per kind, the hypotheses (_ric) lighter and half transparent"]
    for name, (colour, hypothesis) in sorted(used.items()):
        r, g, b = _rgb(colour)
        alpha = HYPOTHESIS_ALPHA if hypothesis else 1.0
        materials += [f"newmtl {name}", f"Kd {r:.4f} {g:.4f} {b:.4f}", "Ka 0.0000 0.0000 0.0000",
                      "Ks 0.0000 0.0000 0.0000", f"d {alpha:.1f}", f"Tr {1.0 - alpha:.1f}", "illum 1", ""]
    for target, text in ((mtl, materials), (path, lines)):
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text("\n".join(text) + "\n", encoding="utf-8")
        tmp.replace(target)
    return path


# ----------------------------------------------------------------------------- GLB
def write_glb(chunk, path, system: str = "ship", reconstruction: bool = False) -> Path:
    """A GLB of the timbers in ``system`` (Y up, as glTF wants: (x, z, -y)): a node per timber named as in the OBJ, a
    PBR colour per kind, the hypotheses lighter and translucent (``alphaMode`` BLEND, alpha 0.5). In "world" a round
    origin is taken off (float32) and given in the scene's ``extras`` with the system, the CRS and the ship frame;
    ``reconstruction``: the reconstruction too (:func:`items`)."""
    import trimesh
    from trimesh.visual.material import PBRMaterial

    entries = items(chunk, system, reconstruction)
    origin = None
    if system == "world" and entries:
        allv = np.vstack([e["vertices"] for e in entries])
        origin = np.round((allv.min(axis=0) + allv.max(axis=0)) / 2.0)
    scene = trimesh.Scene()
    for e in entries:
        v = e["vertices"] - (origin if origin is not None else 0.0)
        yup = np.column_stack([v[:, 0], v[:, 2], -v[:, 1]])
        hypothesis = e["status"] != "measured"
        r, g, b = _rgb(e["colour"])
        material = PBRMaterial(name=e["material"], baseColorFactor=[r, g, b, HYPOTHESIS_ALPHA if hypothesis else 1.0],
                               metallicFactor=0.0, roughnessFactor=1.0, alphaMode="BLEND" if hypothesis else "OPAQUE",
                               doubleSided=False)
        mesh = trimesh.Trimesh(yup, e["faces"], process=False,
                               visual=trimesh.visual.TextureVisuals(material=material))
        scene.add_geometry(mesh, node_name=e["name"], geom_name=e["name"])
    scene.metadata["extras"] = _header(chunk, system, origin)
    scene.metadata["extras"]["axes_gltf"] = "glTF Y up: (x, z, -y) of the system"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(scene.export(file_type="glb"))
    tmp.replace(path)
    return path


# ----------------------------------------------------------------------------- measures, tables
def _rows(chunk, entries: list[dict]) -> list[dict]:
    """The catalogue's rows (:func:`timbers.measure`) with ``volume``, ``thickness_mm``, ``hidden_share`` and
    ``solid``, and a row for the typed keel (its length, siding as width, its z range as heights)."""
    by_id = {e["id"]: e for e in entries}
    records = {r["id"]: r for r in timbers.timbers_of(chunk)}
    rows = timbers.measure(chunk)
    for row in rows:
        e = by_id.get(row["id"])
        rec = e["record"] if e else None
        row["volume"] = e["volume"] if e else None
        row["thickness_mm"] = (rec or records.get(row["id"]) or {}).get("thickness_mm")
        row["hidden_share"] = e["hidden_share"] if e else None
        row["solid"] = rec.get("solid") if rec else ""
    keel = by_id.get("keel")
    if keel is not None:
        k = ship.state(chunk).get("keel") or {}
        v = keel["ship_vertices"]
        rows.append({c: None for c in TABLE_COLUMNS} | {
            "id": "keel", "kind": "keel", "side": "centre", "name": keel["label"], "station": "",
            "status": keel["status"], "chunk": chunk.key, "loose": 0, "x": float(v[:, 0].mean()), "y": 0.0,
            "length": float(k.get("x1", v[:, 0].max()) - k.get("x0", v[:, 0].min())), "width": k.get("siding"),
            "z_min": float(v[:, 2].min()), "z_max": float(v[:, 2].max()),
            "notes": t(f"dimensioni date: larghezza {k.get('siding')} m, altezza {k.get('moulding')} m",
                       f"given dimensions: siding {k.get('siding')} m, moulding {k.get('moulding')} m"),
            "warning": "", "volume": keel["volume"], "hidden_share": keel["hidden_share"], "solid": k.get("solid")})
    return rows


def _cell(column: str, value):
    """A table's cell: as :func:`timbers.export_csv` writes it, the volume and the hidden share to 6 decimals (a
    strake holds about a hundredth of a cubic metre)."""
    if column in ("volume", "hidden_share") and isinstance(value, float):
        return round(value, 6)
    return timbers._cell(value)


def write_tables(chunk, folder) -> dict:
    """The catalogue with volume, thickness and status in ``folder``: ``timbers_3d.csv`` (codes, as
    :func:`timbers.export_csv`) and ``timbers_3d.xlsx`` (titles, as :func:`timbers.export_xlsx`); the typed keel a
    row of its own. Returns ``{"csv", "xlsx", "rows"}``."""
    from openpyxl import Workbook
    from openpyxl.styles import Font

    entries = items(chunk, "ship", reconstruction=False)
    rows = _rows(chunk, entries)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    csv_path, xlsx_path = folder / f"{TABLE_NAME}.csv", folder / f"{TABLE_NAME}.xlsx"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(TABLE_COLUMNS)
        for row in rows:
            w.writerow([_cell(c, row.get(c)) for c in TABLE_COLUMNS])
    book = Workbook()
    sheet = book.active
    sheet.title = t("Legni in 3D", "Timbers in 3D")
    sheet.append(list(TABLE_COLUMNS))
    for row in rows:
        row = timbers._titled(row)
        sheet.append([_cell(c, row.get(c)) for c in TABLE_COLUMNS])
    for cell in sheet[1]:
        cell.font = Font(bold=True)
    book.save(xlsx_path)
    return {"csv": csv_path, "xlsx": xlsx_path, "rows": len(rows)}


# ----------------------------------------------------------------------------- IFC
def _closed(faces) -> bool:
    """Whether a mesh is closed: every edge shared by exactly two faces (the faired hull is an open surface)."""
    f = np.asarray(faces, np.int64)
    if not len(f):
        return False
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _unique, counts = np.unique(edges, axis=0, return_counts=True)
    return bool((counts == 2).all())


def _pset(chunk, e: dict, row: dict | None) -> dict:
    row = row or {}
    rec = e["record"] or {}
    v = e["ship_vertices"]
    z_min, z_max = row.get("z_min"), row.get("z_max")
    if z_min is None or z_max is None:  # no model under it (or the typed keel): the solid's own heights
        z_min, z_max = float(v[:, 2].min()), float(v[:, 2].max())
    props = {"Id": e["id"], "Name": e["label"], "Kind": timbers.kind_title(e["kind"]),
             "Side": timbers.side_title(e["side"]) if e["side"] else "", "Chunk": str(rec.get("chunk") or chunk.key),
             "Status": timbers.status_title(e["status"]), "StatusCode": e["status"],
             "Length": row.get("length"), "Width": row.get("width"), "ZMin": float(z_min), "ZMax": float(z_max),
             "Volume": float(e["volume"]),
             "Thickness": float(e["thickness_mm"]) / 1000.0 if e["thickness_mm"] else None,
             "HiddenShare": float(e["hidden_share"]), "Sources": _sources(rec) if rec else e.get("sources") or t(
                 "dimensioni date", "given dimensions")}
    if e["kind"] == "hull":  # an open surface: no volume, no thickness
        props.pop("Volume")
        props.pop("Thickness")
    return {k: v for k, v in props.items() if v is not None}


MEASURES = {"Length": "IfcLengthMeasure", "Width": "IfcLengthMeasure", "ZMin": "IfcLengthMeasure",
            "ZMax": "IfcLengthMeasure", "Thickness": "IfcLengthMeasure", "Volume": "IfcVolumeMeasure",
            "HiddenShare": "IfcNormalisedRatioMeasure"}
"""The Pset's numbers with their IFC measure (metres, cubic metres, a share 0-1), so BIM tools show the unit."""


def _typed(model, props: dict) -> dict:
    """``props`` with the numbers of :data:`MEASURES` as IFC measures (the others as they are)."""
    return {k: (model.create_entity(MEASURES[k], float(v)) if k in MEASURES else v) for k, v in props.items()}


def write_ifc(chunk, path, system: str = "ship", project: str | None = None, site: str | None = None,
              reconstruction: bool = False) -> dict:
    """An IFC4 of the timbers in ``system`` (see the module), written with a decimal point whatever the numeric locale;
    ``reconstruction``: the reconstruction too (:func:`items`).
    Returns ``{"path", "elements", "map_conversion" (bool), "note" (why there is none), "not_built"}``."""
    try:
        import ifcopenshell  # noqa: F401
    except ImportError:
        raise NotReady(t("Per l'IFC serve IfcOpenShell (installa il programma con l'opzione «bim»).",
                         "The IFC needs IfcOpenShell (install the program with the «bim» option).")) from None
    with c_numbers():
        return _write_ifc(chunk, path, system, project, site, reconstruction)


def _write_ifc(chunk, path, system, project, site, reconstruction=False) -> dict:
    import ifcopenshell
    import ifcopenshell.api.aggregate
    import ifcopenshell.api.context
    import ifcopenshell.api.georeference
    import ifcopenshell.api.geometry
    import ifcopenshell.api.pset
    import ifcopenshell.api.root
    import ifcopenshell.api.spatial
    import ifcopenshell.api.style
    import ifcopenshell.api.unit

    from pyarchinit_photogrammetry.core import crs as crs_mod

    entries = items(chunk, system, reconstruction)
    rows = {r["id"]: r for r in _rows(chunk, entries)}
    frame = ship.need_frame(chunk)
    georeferenced = not (crs_mod.is_local(chunk.crs) or crs_mod.is_geographic(chunk.crs))
    if system == "world" and not georeferenced:  # no map conversion to keep an origin in: the coordinates as they are
        origin = np.zeros(3)
        placement = None
    elif system == "world":
        allv = np.vstack([e["vertices"] for e in entries]) if entries else np.zeros((1, 3))
        origin = np.floor((allv.min(axis=0) + allv.max(axis=0)) / 2.0)
        placement = {"Eastings": float(origin[0]), "Northings": float(origin[1]), "OrthogonalHeight": float(origin[2]),
                     "XAxisAbscissa": 1.0, "XAxisOrdinate": 0.0, "Scale": 1.0}
    else:
        origin = np.zeros(3)
        at = frame.to_world(np.zeros((1, 3)))[0]  # the ship origin in the chunk's coordinates
        dx, dy = frame.direction
        placement = {"Eastings": float(at[0]), "Northings": float(at[1]), "OrthogonalHeight": float(at[2]),
                     "XAxisAbscissa": float(dx), "XAxisOrdinate": float(dy), "Scale": 1.0}
    doc = getattr(chunk, "document", None)
    model = ifcopenshell.file(schema="IFC4")
    proj = ifcopenshell.api.root.create_entity(model, ifc_class="IfcProject", name=project or "Rilievo")
    ifcopenshell.api.unit.assign_unit(model, length={"is_metric": True, "raw": "METERS"},
                                      area={"is_metric": True, "raw": "METERS"},
                                      volume={"is_metric": True, "raw": "METERS"})
    ctx = ifcopenshell.api.context.add_context(model, context_type="Model")
    body = ifcopenshell.api.context.add_context(model, context_type="Model", context_identifier="Body",
                                                target_view="MODEL_VIEW", parent=ctx)
    mapped, note = False, ""
    where = (t("le coordinate del chunk, senza origine sottratta", "the chunk's coordinates, no origin taken off")
             if system == "world" else
             t("il sistema della nave (x, segno * y, z)", "the ship frame (x, sign * y, z)"))
    if crs_mod.is_geographic(chunk.crs):
        note = t(f"Il chunk è in coordinate geografiche ({chunk.crs}): nessuna IfcMapConversion (serve un sistema "
                 f"proiettato); coordinate: {where}.",
                 f"The chunk is in geographic coordinates ({chunk.crs}): no IfcMapConversion (it needs a projected "
                 f"system); coordinates: {where}.")
    elif not georeferenced:
        note = t(f"Il chunk è in un sistema locale: nessuna IfcMapConversion; coordinate: {where}.",
                 f"The chunk is in a local system: no IfcMapConversion; coordinates: {where}.")
    else:
        ifcopenshell.api.georeference.add_georeferencing(model)
        epsg, vertical = crs_mod.epsg_code(chunk.crs), crs_mod.vertical_part(chunk.crs)
        ifcopenshell.api.georeference.edit_georeferencing(
            model, projected_crs={"Name": f"EPSG:{epsg}" if epsg else str(crs_mod.horizontal(chunk.crs)),
                                  **({"VerticalDatum": vertical} if vertical else {})},
            coordinate_operation=placement)
        mapped = True
    site_name = site or ((doc.meta.get("sito") if doc is not None else None) or "Sito")
    site_e = ifcopenshell.api.root.create_entity(model, ifc_class="IfcSite", name=site_name)
    ifcopenshell.api.aggregate.assign_object(model, products=[site_e], relating_object=proj)
    styles: dict = {}
    for e in entries:
        ifc_class = "IfcPlate" if e["kind"] in PLATES else "IfcMember"
        el = ifcopenshell.api.root.create_entity(model, ifc_class=ifc_class, name=e["name"],
                                                 predefined_type="USERDEFINED")
        el.GlobalId = stable_global_id(chunk.key, "shipwreck", e["id"])
        el.ObjectType = (t("carenatura (ipotesi)", "fairing (hypothesis)") if e["kind"] == "hull" else
                         timbers.KIND_TITLES.get(e["kind"], (e["kind"], e["kind"]))[1])
        el.Description = timbers.status_title(e["status"])
        ifcopenshell.api.spatial.assign_container(model, products=[el], relating_structure=site_e)
        el.ObjectPlacement = model.createIfcLocalPlacement(None, model.createIfcAxis2Placement3D(
            model.createIfcCartesianPoint((0.0, 0.0, 0.0)), None, None))
        v = e["vertices"] - origin
        pts = model.createIfcCartesianPointList3D(np.asarray(v, float).tolist())
        faces = model.createIfcTriangulatedFaceSet(pts, None, _closed(e["faces"]),
                                                   (np.asarray(e["faces"], np.int64) + 1).tolist(), None)
        rep = model.createIfcShapeRepresentation(body, "Body", "Tessellation", [faces])
        ifcopenshell.api.geometry.assign_representation(model, product=el, representation=rep)
        style = styles.get(e["material"])
        if style is None:
            r, g, b = _rgb(e["colour"])
            style = ifcopenshell.api.style.add_style(model, name=e["material"])
            ifcopenshell.api.style.add_surface_style(
                model, style=style, ifc_class="IfcSurfaceStyleShading",
                attributes={"SurfaceColour": {"Name": None, "Red": r, "Green": g, "Blue": b},
                            "Transparency": 0.0 if e["status"] == "measured" else 1.0 - HYPOTHESIS_ALPHA})
            styles[e["material"]] = style
        ifcopenshell.api.style.assign_representation_styles(model, shape_representation=rep, styles=[style])
        pset = ifcopenshell.api.pset.add_pset(model, product=el, name=PSET)
        ifcopenshell.api.pset.edit_pset(model, pset=pset, properties=_typed(model, _pset(chunk, e, rows.get(e["id"]))))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)  # IfcOpenShell picks the format by the suffix
    model.write(str(tmp))
    tmp.replace(path)
    return {"path": path, "elements": len(entries), "map_conversion": mapped, "note": note,
            "not_built": _not_built(chunk), "system": system}
