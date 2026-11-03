"""The file link with the Clinker Boat Tool of Fred Fröhlich and Sigurd Solhaug Nielsen, a Houdini Digital Asset (HDA,
MIT; Zenodo 10.5281/zenodo.17488549) described in "From Photogrammetry to Construction Knowledge: 3D Documentation and
Parametric Reconstruction of Historic Clinker Boats" (CC BY 4.0; Zenodo 10.5281/zenodo.23422398). Nothing of the HDA is
used or copied: the plugin writes files the HDA's own steps take in, and reads back the planks it exports.

Export: ``<folder>/clinker_boat_tool/`` with the hull in the ship frame turned to Y up (Houdini's up axis, the keel
along X as the HDA's second step asks), the raw sections at the stations as polylines, the stations and the frame as
JSON, the land lines and the strake model when they exist, and a README with the HDA's six steps and the citations.

Y up: ``X = x, Y = z, Z = -y * sign`` (:data:`Y_UP`): ``sign`` is the frame's (:attr:`ship.Frame.sign`), so that world
to Y up is a rotation, never a mirror, whichever end the bow is at; the same hull gives the same file with either bow.

Import: an OBJ whose objects (``o``) - else groups (``g``) - are planks, read back from Y up to the ship frame and to
the chunk's coordinates, one ``plank`` record of the catalogue each, and their deviation from the scan."""
from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from pyarchinit_photogrammetry.core.progress import as_progress

from . import clinker, lines, ship, timbers
from .texts import t

Y_UP = "X = x, Y = z, Z = -y * sign"
FOLDER = "clinker_boat_tool"
SIMPLIFY = 0.01  # metres: the plank's outline in plan
MISSING = 0.5  # metres: a plank's vertex farther than this from the model has no model there
NOTE = ("importato dal Clinker Boat Tool: {}", "imported from the Clinker Boat Tool: {}")
PAPER_DOI = "10.5281/zenodo.23422398"
HDA_DOI = "10.5281/zenodo.17488549"
COLUMNS = ("id", "name", "side", "vertices", "x", "y", "median_mm", "p90_mm", "missing")


def to_y_up(xyz, sign: int) -> np.ndarray:
    """Ship ``(N, 3)`` to Houdini's Y up: ``X = x, Y = z, Z = -y * sign``."""
    s = np.asarray(xyz, float).reshape(-1, 3)
    return np.column_stack([s[:, 0], s[:, 2], -s[:, 1] * sign])


def from_y_up(xyz, sign: int) -> np.ndarray:
    """Y up ``(N, 3)`` back to ship coordinates: the exact inverse of :func:`to_y_up`."""
    u = np.asarray(xyz, float).reshape(-1, 3)
    return np.column_stack([u[:, 0], -u[:, 2] * sign, u[:, 1]])


def _name(text) -> str:
    """A name for an OBJ object: no spaces."""
    return re.sub(r"\s+", "_", str(text).strip()) or "_"


def _write_text(path: Path, parts) -> Path:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for part in parts:
            if isinstance(part, str):
                f.write(part + "\n")
            else:
                fmt, array = part
                if len(array):
                    np.savetxt(f, array, fmt=fmt)
    os.replace(tmp, path)
    return path


def _frame_comments(frame: ship.Frame, chunk) -> list[str]:
    return [f"# ship frame turned to Y up: {Y_UP} (sign {frame.sign}); metres",
            f"# ship frame: x along the keel, y to port, z up; start {list(frame.start)}, end {list(frame.end)}, "
            f"bow {frame.bow}, z0 {frame.z0}, x0 {frame.x0}, heading {frame.heading:.6f}, crs {chunk.crs}",
            "# see frame.json beside this file"]


def _frame_json(frame: ship.Frame, chunk) -> dict:
    return {"origin_world": list(frame.start), "end_world": list(frame.end), "heading": frame.heading,
            "z0": frame.z0, "x0": frame.x0, "bow": frame.bow, "sign": frame.sign, "length": frame.length,
            "slope": frame.slope, "y_up": Y_UP, "units": "m", "crs": chunk.crs,
            "ship_frame": "x along the keel from the start point, y to port, z up above z0",
            "to_world": "ship (x, y, z) = (X, -Z * sign, Y); world = start + (x - x0) * u + y * v, z + z0, with u the "
                        "direction of the keel in plan and v = sign * (-u_y, u_x)"}


def _readme(files: list[str]) -> str:
    return "\n".join([
        "Clinker Boat Tool - files from pyarchinit-photogrammetry (plugin Relitto / Shipwreck)",
        "",
        f"Coordinates: the ship frame turned to Y up, {Y_UP}; metres; the keel along X (frame.json).",
        "Files: " + ", ".join(files) + ".",
        "",
        "EN - The six steps of the HDA with these files:",
        "1. Import the scan: hull.obj (sections.obj, lands.obj and strakes.obj as further references).",
        "2. Calibrate and align: already done - real metres, the keel along X, Y up; nothing to move.",
        "3. Define sections: the stations' x are in stations.json; sections.obj shows the cuts of the scan there.",
        "4. Generate the plank layout: widths, thickness and lap per strake can be read from lands.obj and from the "
        "strake table (strakes.csv and strakes.xlsx in this folder), when present.",
        "5. Bend and align the planks on the scan.",
        "6. Presets and export: export the planks as OBJ, one object or group per plank, under a name of your own "
        "(planks.obj; not strakes.obj or lands.obj, which the plugin sets aside or replaces), and read them back in the "
        "plugin (step 7 of the Shipwreck pane, Planks from Houdini...): each plank becomes a timber record with its "
        "deviation from the scan.",
        "",
        "IT - I sei passi dell'HDA con questi file:",
        "1. Importa la scansione: hull.obj (sections.obj, lands.obj e strakes.obj come riferimenti in più).",
        "2. Calibra e allinea: già fatto - metri veri, chiglia lungo X, Y in alto; nulla da spostare.",
        "3. Definisci le sezioni: le x delle stazioni sono in stations.json; sections.obj mostra lì i tagli della "
        "scansione.",
        "4. Genera la disposizione delle fasce: larghezze, spessore e sovrapposizione per fascia si leggono da "
        "lands.obj e dalla tabella delle fasce (strakes.csv e strakes.xlsx in questa cartella), se presenti.",
        "5. Piega e allinea le fasce sulla scansione.",
        "6. Preset ed esportazione: esporta le fasce in OBJ, un oggetto o gruppo per fascia, con un nome tuo "
        "(tavole.obj; non strakes.obj o lands.obj, che il plugin mette da parte o sostituisce), e rileggile nel plugin "
        "(passo 7 del pannello Relitto, Fasce da Houdini...): ogni fascia diventa una scheda di legno con il suo "
        "scostamento dalla scansione.",
        "",
        "lands.obj and strakes.obj: a hypothesis from the sections, not a measure (ipotesi dalle sezioni, non una "
        "misura).",
        "",
        "Cite / Citare:",
        "Fröhlich, F., Nielsen, S. S. From Photogrammetry to Construction Knowledge: 3D Documentation and Parametric "
        f"Reconstruction of Historic Clinker Boats. CC BY 4.0. https://doi.org/{PAPER_DOI}",
        f"Fröhlich, F., Nielsen, S. S. Clinker Boat Tool (Houdini Digital Asset). MIT License. https://doi.org/{HDA_DOI}",
        "",
    ])


def export(chunk, folder, progress=None) -> dict:
    """The folder ``<folder>/clinker_boat_tool/`` for the Clinker Boat Tool, everything in the ship frame turned to Y
    up (:data:`Y_UP`):

    * ``hull.obj``, the chunk's model;
    * ``sections.obj``, an object per station, its raw polylines as ``l``;
    * ``stations.json`` and ``frame.json``;
    * ``lands.obj`` when the land lines exist and are current;
    * ``strakes.obj`` and its table ``strakes.csv``, ``strakes.xlsx`` when the strake model does;
    * ``README.txt``, the HDA's steps and the citations.

    The files of an earlier export that no longer hold are removed; the folder is kept in
    ``meta["shipwreck"]["houdini"]["export"]`` (a new detection of the lands sets its strake files aside).
    Returns the paths by name (``hull``, ``sections``, ``stations``, ``frame``, ``lands``, ``strakes``,
    ``strake_table``, ``strake_sheet``, ``readme``). Refused without the sections at the stations, or with sections
    cut on another frame."""
    progress = as_progress(progress)
    ship.check_metric(chunk)
    frame = ship.need_frame(chunk)
    data = lines.load(chunk)
    stations = [s for s in (data or {}).get("sections", []) if s["kind"] == "station"]
    if not stations:
        raise ship.NotReady(t("Taglia prima le sezioni alle stazioni.", "Cut the sections at the stations first."))
    if lines.is_stale(chunk, data):
        raise ship.NotReady(t("Le sezioni non sono aggiornate (altro sistema della nave o altro modello): tagliale di "
                              "nuovo.",
                              "The sections are out of date (another ship frame or another model): cut them again."))
    out_dir = Path(folder) / FOLDER
    out_dir.mkdir(parents=True, exist_ok=True)
    sign = frame.sign
    head = _frame_comments(frame, chunk)
    out = {}

    progress(0.05, t("Scafo", "Hull"))
    mesh = lines.ship_mesh(chunk, frame)  # world to Y up is a rotation: the model's winding stays as it is
    out["hull"] = _write_text(out_dir / "hull.obj", [
        "# hull: the 3D model of the chunk", *head, "o hull", ("v %.6f %.6f %.6f", to_y_up(mesh.vertices, sign)),
        ("f %d %d %d", np.asarray(mesh.faces, np.int64) + 1)])

    progress(0.6, t("Sezioni", "Sections"))
    parts = ["# sections at the stations, raw (as cut from the model, no water correction)", *head]
    count = 0
    for s in stations:
        parts.append(f"o {_name(s['name'])}")
        for p in s["lines"]:
            p = np.asarray(p, float)
            if not len(p):
                continue
            parts.append(("v %.6f %.6f %.6f", to_y_up(np.column_stack([np.full(len(p), s["value"]), p]), sign)))
            parts.append("l " + " ".join(str(i) for i in range(count + 1, count + len(p) + 1)))
            count += len(p)
    out["sections"] = _write_text(out_dir / "sections.obj", parts)

    unsure = _unsure(chunk)
    rows = [{"name": s["name"], "x": s["value"], "X": s["value"], "polylines": len(s["lines"]),
             "vertices": sum(len(p) for p in s["lines"]), "uncertain": s["name"] in unsure} for s in stations]
    out["stations"] = _write_json(out_dir / "stations.json", {"y_up": Y_UP, "units": "m", "stations": rows})
    out["frame"] = _write_json(out_dir / "frame.json", _frame_json(frame, chunk))

    progress(0.8, t("Linee delle fasce", "Land lines"))
    state = ship.state(chunk).get("clinker") or {}
    current = "lands" in state and not clinker.is_stale(chunk)
    found = [ln for ln in state.get("lines") or [] if len(ln["points"]) > 1] if current else []
    if found:
        parts = [f"# land lines: {t(*clinker.HYPOTHESIS)}", *head,
                 "# o linea_B<k>: port (babordo), o linea_T<k>: starboard (tribordo); k from the keel"]
        count = 0
        for line in found:
            p = np.asarray(line["points"], float)
            parts += [f"o linea_{clinker.SIDE_CODES[line['side']][0]}{line['k']}", ("v %.6f %.6f %.6f", to_y_up(p, sign)),
                      "l " + " ".join(str(i) for i in range(count + 1, count + len(p) + 1))]
            count += len(p)
        out["lands"] = _write_text(out_dir / "lands.obj", parts)
    source = ship.folder(chunk) / "clinker" / "strakes.obj"
    if current and state.get("strakes") and source.exists():
        out["strakes"] = _strakes(source, out_dir / "strakes.obj", sign, head)
        tables = clinker.write_tables(chunk, out_dir)  # widths, thickness and lap per strake, for the HDA's step 4
        out["strake_table"], out["strake_sheet"] = tables["csv"], tables["xlsx"]
    for name in ("lands", "strakes"):  # an earlier export's files no longer true
        if name not in out:
            (out_dir / f"{name}.obj").unlink(missing_ok=True)
    if "strakes" not in out:
        for name in ("strakes.csv", "strakes.xlsx"):
            (out_dir / name).unlink(missing_ok=True)

    names = [p.name for p in out.values()] + ["README.txt"]
    readme = out_dir / "README.txt"
    tmp = readme.with_name("README.txt.tmp")
    tmp.write_text(_readme(names), encoding="utf-8")
    os.replace(tmp, readme)
    out["readme"] = readme
    _store(chunk)["export"] = str(out_dir)
    progress(1.0, t("File per il Clinker Boat Tool pronti", "Files for the Clinker Boat Tool ready"))
    return out


def _unsure(chunk) -> set:
    state = ship.state(chunk).get("clinker") or {}
    names = set()
    for group in (state.get("uncertain") or {}, state.get("jumps") or {}):
        for value in group.values():
            names |= set(value or [])
    return names


def _store(chunk) -> dict:
    """``meta["shipwreck"]["houdini"]``; the rows of a single file kept by the development version at 489fcf4 (a
    list under ``planks``, the file under ``file``) become the entry of that file."""
    store = ship.state(chunk).setdefault("houdini", {})
    if isinstance(store.get("planks"), list):
        store["planks"] = {store.pop("file", "?"): store["planks"]}
    return store


def imports(chunk) -> dict:
    """The rows of every OBJ imported, by file name (:func:`import_planks`); read only, the chunk is not touched."""
    store = (chunk.meta.get(ship.KEY) or {}).get("houdini") or {}
    planks = store.get("planks") or {}
    return {store.get("file", "?"): planks} if isinstance(planks, list) else dict(planks)


def _write_json(path: Path, data: dict) -> Path:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


def _strakes(source: Path, path: Path, sign: int, head: list[str]) -> Path:
    """The plugin's strakes.obj (ship coordinates) turned to Y up; where that is a mirror of the ship frame
    (``sign`` -1) the faces are wound the other way, so they still point outward."""
    parts = [*head]
    for ln in source.read_text(encoding="utf-8").splitlines():
        if ln.startswith("v "):
            x, y, z = (float(a) for a in ln.split()[1:4])
            X, Y, Z = to_y_up([[x, y, z]], sign)[0]
            parts.append(f"v {X:.6f} {Y:.6f} {Z:.6f}")
        elif ln.startswith("f ") and sign < 0:
            a, b, c = ln.split()[1:4]
            parts.append(f"f {a} {c} {b}")
        elif ln.startswith("# frame"):
            continue  # the ship frame's line: ours above says how it was turned
        else:
            parts.append(ln)
    return _write_text(path, parts)


# ------------------------------------------------------------------------------------- the import
def read_obj(path) -> dict[str, np.ndarray]:
    """The planks of an OBJ as name -> vertices ``(N, 3)``: its objects (``o``) or its groups (``g``), whichever level
    has more names (one ``o`` wrapping a group per plank gives the groups). A ``g`` line with several names puts what
    follows in each of them (the OBJ convention, Houdini writes it), and a group holding every element of two or
    more other groups, or the same elements as a group before it (a catch-all such as "all": :func:`_catch_all`), is
    left out. The vertices of a plank are those its faces, lines and points use, or, when it has none, those written
    under it. Refused when the file has neither objects nor
    groups, or a line cannot be read (its number is given)."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ship.NotReady(t(f"Non riesco a leggere {path}: {exc}", f"Cannot read {path}: {exc}")) from None
    vertices: list[list[float]] = []
    current: dict[str, list[str]] = {"o": [], "g": []}
    declared: dict[str, dict] = {"o": {}, "g": {}}
    used: dict[str, dict] = {"o": {}, "g": {}}
    elements: dict[str, dict] = {"o": {}, "g": {}}
    total = 0

    def bad(number, ln):
        return ship.NotReady(t(f"Riga {number} del file illeggibile: {ln}", f"Line {number} of the file is unreadable: {ln}"))

    for number, raw in enumerate(text.splitlines(), 1):
        parts = raw.strip().split(None, 1)
        if not parts or parts[0].startswith("#"):
            continue
        key, rest = parts[0], parts[1] if len(parts) > 1 else ""
        if key in ("o", "g"):
            names = [rest.strip()] if key == "o" and rest.strip() else rest.split()
            current[key] = names
            for name in names:
                declared[key].setdefault(name, [])
                used[key].setdefault(name, [])
                elements[key].setdefault(name, set())
        elif key == "v":
            try:
                xyz = [float(v) for v in rest.split()[:3]]
            except ValueError:
                raise bad(number, raw.strip()) from None
            if len(xyz) < 3:
                raise bad(number, raw.strip())
            vertices.append(xyz)
            for kind in ("o", "g"):
                for name in current[kind]:
                    declared[kind][name].append(len(vertices) - 1)
        elif key in ("f", "l", "p"):
            index = []
            try:
                for token in rest.split():
                    i = int(token.split("/")[0])
                    index.append(i - 1 if i > 0 else len(vertices) + i)
            except ValueError:
                raise bad(number, raw.strip()) from None
            if not index:
                raise bad(number, raw.strip())
            total += 1
            for kind in ("o", "g"):
                for name in current[kind]:
                    used[kind][name].extend(index)
                    elements[kind][name].add(total)
    groups = list(declared["g"])
    if len(groups) > 1 and total:
        kept = [g for g in groups if not _catch_all(g, groups, elements["g"])]
        groups = kept or groups
    if not declared["o"] and not groups:
        raise ship.NotReady(t("Il file non ha fasce da leggere: nessun oggetto o gruppo nel file (righe o / g). "
                              "Esporta dal Clinker Boat Tool un oggetto o un gruppo per fascia.",
                              "The file has no planks to read: no object or group in the file (o / g lines). Export "
                              "one object or group per plank from the Clinker Boat Tool."))
    kind, names = ("g", groups) if len(groups) > len(declared["o"]) else ("o", list(declared["o"]))
    v = np.asarray(vertices, float).reshape(-1, 3)
    out = {}
    for name in names:
        index = used[kind][name] or declared[kind][name]
        index = np.unique(np.asarray(index, np.int64))
        if len(index) and (index.min() < 0 or index.max() >= len(v)):
            raise ship.NotReady(t(f"{name}: indice di vertice fuori dal file.", f"{name}: a vertex index beyond the file."))
        if len(index):
            out[name] = v[index]
    if not out:
        raise ship.NotReady(t("Gli oggetti del file non hanno vertici.", "The objects of the file have no vertices."))
    return out


def _catch_all(group: str, groups: list[str], elements: dict[str, set]) -> bool:
    """A group that is no plank of its own: one holding every element of at least two other groups ("all",
    "planks"; with the file's loose elements or not), or holding exactly the elements of a group named before it (a
    single plank and its catch-all: the first name is the plank's). A group inside one other group (``B1_end`` in
    ``B1``) is a plank, and so is the outer one."""
    mine = elements[group]
    if not mine:
        return False
    if any(elements[g] == mine for g in groups[:groups.index(group)]):
        return True
    return sum(1 for g in groups if g != group and elements[g] and elements[g] <= mine) >= 2


def _side(name: str, y: float, side_from: str) -> str:
    if side_from == "name":
        found = re.search(r"(?:^|[_\s])(B|T|port|starboard|babordo|tribordo)\d*(?:$|[_\s])", name, re.IGNORECASE)
        if found:
            word = found.group(1).lower()
            return "port" if word in ("b", "port", "babordo") else "starboard"
    if y > timbers.SIDE_TOLERANCE:
        return "port"
    if y < -timbers.SIDE_TOLERANCE:
        return "starboard"
    return "centre"


def _outline(world_xy: np.ndarray) -> dict:
    """The plank's outline in plan: the convex hull of its vertices, simplified at 1 cm."""
    import shapely

    centre = world_xy.mean(axis=0)  # relative: coordinates of millions of metres lose nothing
    hull = shapely.MultiPoint(world_xy - centre).convex_hull.simplify(SIMPLIFY)
    if hull.geom_type == "Polygon" and len(hull.exterior.coords) >= 4:
        return timbers.geometry("polygon", np.asarray(hull.exterior.coords)[:-1] + centre)
    if hull.geom_type == "LineString":
        return timbers.geometry("line", np.asarray(hull.coords) + centre)
    return timbers.geometry("point", centre[None])


class Distance:
    """The unsigned distance from points to a mesh, to its closest point. ``mesh``: a trimesh in the ship frame,
    whose coordinates are small enough for single precision. ``backend``: ``"open3d"`` (Open3D's raycasting scene,
    when the program has its ``mesh`` extra) or ``"trimesh"`` (``trimesh.proximity.closest_point`` on the R-tree of
    the triangles, both core dependencies of the program: slower, the same distances)."""

    def __init__(self, mesh):
        try:
            import open3d as o3d
        except ImportError:
            self.backend, self.mesh = "trimesh", mesh
            return
        self.backend, self.o3d = "open3d", o3d
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices, np.float32)),
                                 o3d.core.Tensor(np.asarray(mesh.faces, np.uint32)))

    def to(self, xyz) -> np.ndarray:
        points = np.asarray(xyz, float).reshape(-1, 3)
        if self.backend == "trimesh":
            import trimesh

            if not len(points):
                return np.zeros(0)
            return np.asarray(trimesh.proximity.closest_point(self.mesh, points)[1], float)
        return self.scene.compute_distance(self.o3d.core.Tensor(points.astype(np.float32))).numpy().astype(float)


def _notes(name: str) -> set:
    return {NOTE[0].format(name), NOTE[1].format(name)}


def import_planks(chunk, path, side_from: str = "y", deviation: bool = True, progress=None) -> dict:
    """The planks of an OBJ exported by the Clinker Boat Tool (:func:`read_obj`: an object, else a group, per plank;
    Y up as :data:`Y_UP`) as ``plank`` records of the catalogue: the name the object's, the side from the mean y of
    its vertices in the ship frame (``side_from="name"``: from B/T, port/starboard in the name, the y where the name
    says nothing), the geometry its outline in plan (the convex hull of its vertices, simplified at 1 cm), the note
    ``importato dal Clinker Boat Tool: <file name>``. The records of an earlier import of the same file name are
    replaced, never the user's.

    ``deviation``: per plank the median and the 90th percentile, in millimetres, of the distance of its vertices
    from the scan (the closest point of the chunk's model, :class:`Distance`, in any direction: right on steep
    planking too); ``missing`` the vertices farther than ``MISSING`` (no model there), left out of the two. Rows
    (``COLUMNS``) in ``chunk.meta["shipwreck"]["houdini"]["planks"][<file name>]`` (one entry per file imported;
    ``["imports"][<file name>]`` says when) and in ``<data>/shipwreck/clinker/<file stem>_scostamenti.csv`` and
    ``.xlsx``. The distance is computed with Open3D when the program has it, else with trimesh (:class:`Distance`);
    which one ran is ``backend`` (``None`` without a distance) in the result and in ``["imports"][<file name>]``.
    Returns ``planks`` (how many), ``replaced``, ``rows``, ``csv``, ``xlsx``, ``backend``."""
    progress = as_progress(progress)
    if side_from not in ("y", "name"):
        raise ValueError("side_from: 'y' or 'name'")
    ship.check_metric(chunk)
    frame = ship.need_frame(chunk)
    path = Path(path)
    progress(0.05, t("Lettura del file", "Reading the file"))
    objects = read_obj(path)
    surface = None
    if deviation:
        model = chunk.product_path("model")
        if model is not None and model.exists():
            progress(0.2, t("Lettura del modello", "Reading the model"))
            surface = Distance(lines.ship_mesh(chunk, frame))
    notes = _notes(path.name)
    old = [r["id"] for r in timbers.timbers_of(chunk) if r["kind"] == "plank" and r.get("notes") in notes]
    timbers.remove(chunk, old)
    note = t(*NOTE).format(path.name)
    rows = []
    for i, (name, up) in enumerate(objects.items()):
        progress(0.3 + 0.6 * i / len(objects), t(f"Fascia {name}", f"Plank {name}"))
        xyz = from_y_up(up, frame.sign)
        world = frame.to_world(xyz)
        y = float(xyz[:, 1].mean())
        record = timbers.add(chunk, "plank", _side(name, y, side_from), name, _outline(world[:, :2]), notes=note)
        row = {"id": record["id"], "name": name, "side": record["side"], "vertices": len(xyz),
               "x": float(xyz[:, 0].mean()), "y": y, "median_mm": None, "p90_mm": None, "missing": None}
        if deviation:
            d = surface.to(xyz) if surface is not None else np.full(len(xyz), np.inf)
            ok = d <= MISSING
            row["missing"] = int((~ok).sum())
            if ok.any():
                d = d[ok] * 1000.0
                row["median_mm"], row["p90_mm"] = round(float(np.median(d)), 2), round(float(np.percentile(d, 90)), 2)
        rows.append(row)
    store = _store(chunk)
    store.setdefault("planks", {})[path.name] = rows
    backend = surface.backend if surface is not None else None
    store.setdefault("imports", {})[path.name] = {"imported": datetime.now().isoformat(timespec="seconds"),
                                                  "y_up": Y_UP, "planks": len(rows), "deviation": bool(deviation),
                                                  "backend": backend}
    progress(0.92, t("Tabella degli scostamenti", "Table of the deviations"))
    files = _write_rows(chunk, rows, path.stem)
    progress(1.0, t("Fasce importate", "Planks imported"))
    return {"planks": len(rows), "replaced": len(old), "rows": rows, "backend": backend, **files}


def _write_rows(chunk, rows: list[dict], stem: str) -> dict:
    from pyarchinit_photogrammetry.io.tables import Sheet, write_xlsx

    folder = ship.folder(chunk) / "clinker"
    folder.mkdir(parents=True, exist_ok=True)
    headers = ["id", t("nome", "name"), t("lato", "side"), t("vertici", "vertices"), "x", "y",
               t("distanza dal modello, mediana (mm)", "distance from the model, median (mm)"),
               t("distanza dal modello, p90 (mm)", "distance from the model, p90 (mm)"),
               t(f"vertici senza modello entro {MISSING:g} m", f"vertices with no model within {MISSING:g} m")]
    table = [[r["id"], r["name"], timbers.side_title(r["side"]), r["vertices"], round(r["x"], 4), round(r["y"], 4),
              r["median_mm"], r["p90_mm"], r["missing"]] for r in rows]
    base = f"{_name(stem)}_scostamenti"
    path_csv = folder / f"{base}.csv"
    tmp = folder / f"{base}.csv.tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        f.write("# " + t("fasce dal Clinker Boat Tool: distanza dei vertici dal modello (punto più vicino)",
                         "planks from the Clinker Boat Tool: distance of the vertices from the model (closest point)") + "\n")
        w = csv.writer(f)
        w.writerow(headers)
        for line in table:
            w.writerow(["" if v is None else v for v in line])
    os.replace(tmp, path_csv)
    title = t("Scostamenti", "Deviations")
    path_xlsx = write_xlsx(folder / f"{base}.xlsx", {title: Sheet(headers, table, [None, None, None, 0, 3, 3, 1, 1, 0])},
                           title=t("Fasce dal Clinker Boat Tool", "Planks from the Clinker Boat Tool"))
    return {"csv": path_csv, "xlsx": Path(path_xlsx)}
