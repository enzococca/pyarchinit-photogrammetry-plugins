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
        "strake table of the plugin (strakes.csv), when present.",
        "5. Bend and align the planks on the scan.",
        "6. Presets and export: export the planks as OBJ, one object or group per plank, and read them back in the "
        "plugin (Import planks from the Clinker Boat Tool): each plank becomes a timber record with its deviation from "
        "the scan.",
        "",
        "IT - I sei passi dell'HDA con questi file:",
        "1. Importa la scansione: hull.obj (sections.obj, lands.obj e strakes.obj come riferimenti in più).",
        "2. Calibra e allinea: già fatto - metri veri, chiglia lungo X, Y in alto; nulla da spostare.",
        "3. Definisci le sezioni: le x delle stazioni sono in stations.json; sections.obj mostra lì i tagli della "
        "scansione.",
        "4. Genera la disposizione delle fasce: larghezze, spessore e sovrapposizione per fascia si leggono da "
        "lands.obj e dalla tabella delle fasce del plugin (strakes.csv), se presenti.",
        "5. Piega e allinea le fasce sulla scansione.",
        "6. Preset ed esportazione: esporta le fasce in OBJ, un oggetto o gruppo per fascia, e rileggile nel plugin "
        "(Importa le fasce dal Clinker Boat Tool): ogni fascia diventa una scheda di legno con il suo scostamento "
        "dalla scansione.",
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
    """The folder ``<folder>/clinker_boat_tool/`` for the Clinker Boat Tool: ``hull.obj``, ``sections.obj`` (an object
    per station, its raw polylines as ``l``), ``stations.json``, ``frame.json``, ``lands.obj`` and ``strakes.obj`` when
    the land lines and the strake model exist and are current, ``README.txt``. Everything in the ship frame turned to Y
    up (:data:`Y_UP`). Returns the paths by name (``hull``, ``sections``, ``stations``, ``frame``, ``lands``,
    ``strakes``, ``readme``). Refused without the sections at the stations, or with sections cut on another frame."""
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
    for name in ("lands", "strakes"):  # an earlier export's files no longer true
        if name not in out:
            (out_dir / f"{name}.obj").unlink(missing_ok=True)

    names = [p.name for p in out.values()] + ["README.txt"]
    readme = out_dir / "README.txt"
    tmp = readme.with_name("README.txt.tmp")
    tmp.write_text(_readme(names), encoding="utf-8")
    os.replace(tmp, readme)
    out["readme"] = readme
    progress(1.0, t("File per il Clinker Boat Tool pronti", "Files for the Clinker Boat Tool ready"))
    return out


def _unsure(chunk) -> set:
    state = ship.state(chunk).get("clinker") or {}
    names = set()
    for group in (state.get("uncertain") or {}, state.get("jumps") or {}):
        for value in group.values():
            names |= set(value or [])
    return names


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
    """The objects of an OBJ (``o``), else its groups (``g``), as name -> vertices ``(N, 3)``: the vertices its faces,
    lines and points use, or, when it has none, those written under it. Refused when the file has neither."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ship.NotReady(t(f"Non riesco a leggere {path}: {exc}", f"Cannot read {path}: {exc}")) from None
    vertices: list[list[float]] = []
    current = {"o": None, "g": None}
    declared = {"o": {}, "g": {}}
    used = {"o": {}, "g": {}}
    for raw in text.splitlines():
        ln = raw.strip()
        if not ln or ln.startswith("#"):
            continue
        key, _, rest = ln.partition(" ")
        if key in ("o", "g"):
            current[key] = rest.strip() or None
            if current[key] is not None:
                declared[key].setdefault(current[key], [])
                used[key].setdefault(current[key], [])
        elif key == "v":
            values = rest.split()
            try:
                vertices.append([float(v) for v in values[:3]])
            except ValueError:
                raise ship.NotReady(t(f"Vertice illeggibile nel file: {ln}", f"An unreadable vertex in the file: {ln}")) \
                    from None
            for kind in ("o", "g"):
                if current[kind] is not None:
                    declared[kind][current[kind]].append(len(vertices) - 1)
        elif key in ("f", "l", "p"):
            index = []
            for token in rest.split():
                i = int(token.split("/")[0])
                index.append(i - 1 if i > 0 else len(vertices) + i)
            for kind in ("o", "g"):
                if current[kind] is not None:
                    used[kind][current[kind]].extend(index)
    kind = "o" if declared["o"] else "g" if declared["g"] else None
    if kind is None:
        raise ship.NotReady(t("Il file non ha fasce da leggere: nessun oggetto o gruppo nel file (righe o / g). "
                              "Esporta dal Clinker Boat Tool un oggetto o un gruppo per fascia.",
                              "The file has no planks to read: no object or group in the file (o / g lines). Export "
                              "one object or group per plank from the Clinker Boat Tool."))
    v = np.asarray(vertices, float).reshape(-1, 3)
    out = {}
    for name in declared[kind]:
        index = used[kind][name] or declared[kind][name]
        index = np.unique(np.asarray(index, np.int64))
        if len(index) and (index.min() < 0 or index.max() >= len(v)):
            raise ship.NotReady(t(f"{name}: indice di vertice fuori dal file.", f"{name}: a vertex index beyond the file."))
        if len(index):
            out[name] = v[index]
    if not out:
        raise ship.NotReady(t("Gli oggetti del file non hanno vertici.", "The objects of the file have no vertices."))
    return out


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


def _notes(name: str) -> set:
    return {NOTE[0].format(name), NOTE[1].format(name)}


def import_planks(chunk, path, side_from: str = "y", deviation: bool = True, progress=None) -> dict:
    """The planks of an OBJ exported by the Clinker Boat Tool (:func:`read_obj`: an object, else a group, per plank;
    Y up as :data:`Y_UP`) as ``plank`` records of the catalogue: the name the object's, the side from the mean y of
    its vertices in the ship frame (``side_from="name"``: from B/T, port/starboard in the name, the y where the name
    says nothing), the geometry its outline in plan (the convex hull of its vertices, simplified at 1 cm), the note
    ``importato dal Clinker Boat Tool: <file name>``. The records of an earlier import of the same file name are
    replaced, never the user's.

    ``deviation``: per plank the median and the 90th percentile, in millimetres, of ``|z_plank - z_model|`` at its
    vertices (the model's height over each vertex, :class:`ship.Surface`); ``missing`` the vertices with no model
    under them. Rows (``COLUMNS``) in ``chunk.meta["shipwreck"]["houdini"]["planks"]`` and in
    ``<data>/shipwreck/clinker/<file stem>_scostamenti.csv`` and ``.xlsx``. Returns ``planks`` (how many),
    ``replaced``, ``rows``, ``csv``, ``xlsx``."""
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
            surface = ship.Surface(lines.ship_mesh(chunk, frame).vertices)
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
            z = surface.at(xyz[:, :2]) if surface is not None else np.full(len(xyz), np.nan)
            ok = np.isfinite(z)
            row["missing"] = int((~ok).sum())
            if ok.any():
                d = np.abs(xyz[ok, 2] - z[ok]) * 1000.0
                row["median_mm"], row["p90_mm"] = round(float(np.median(d)), 2), round(float(np.percentile(d, 90)), 2)
        rows.append(row)
    ship.state(chunk)["houdini"] = {"file": path.name, "imported": datetime.now().isoformat(timespec="seconds"),
                                    "y_up": Y_UP, "planks": rows}
    progress(0.92, t("Tabella degli scostamenti", "Table of the deviations"))
    files = _write_rows(chunk, rows, path.stem)
    progress(1.0, t("Fasce importate", "Planks imported"))
    return {"planks": len(rows), "replaced": len(old), "rows": rows, **files}


def _write_rows(chunk, rows: list[dict], stem: str) -> dict:
    from pyarchinit_photogrammetry.io.tables import Sheet, write_xlsx

    folder = ship.folder(chunk) / "clinker"
    folder.mkdir(parents=True, exist_ok=True)
    headers = ["id", t("nome", "name"), t("lato", "side"), t("vertici", "vertices"), "x", "y",
               t("scostamento mediano (mm)", "median deviation (mm)"), t("scostamento p90 (mm)", "p90 deviation (mm)"),
               t("vertici senza modello", "vertices with no model")]
    table = [[r["id"], r["name"], timbers.side_title(r["side"]), r["vertices"], round(r["x"], 4), round(r["y"], 4),
              r["median_mm"], r["p90_mm"], r["missing"]] for r in rows]
    base = f"{_name(stem)}_scostamenti"
    path_csv = folder / f"{base}.csv"
    tmp = folder / f"{base}.csv.tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        f.write("# " + t("fasce dal Clinker Boat Tool: scostamento verticale dal modello", "planks from the Clinker "
                         "Boat Tool: vertical deviation from the model") + "\n")
        w = csv.writer(f)
        w.writerow(headers)
        for line in table:
            w.writerow(["" if v is None else v for v in line])
    os.replace(tmp, path_csv)
    title = t("Scostamenti", "Deviations")
    path_xlsx = write_xlsx(folder / f"{base}.xlsx", {title: Sheet(headers, table, [None, None, None, 0, 3, 3, 1, 1, 0])},
                           title=t("Fasce dal Clinker Boat Tool", "Planks from the Clinker Boat Tool"))
    return {"csv": path_csv, "xlsx": Path(path_xlsx)}
