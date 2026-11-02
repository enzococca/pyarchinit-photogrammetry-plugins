"""Stone by stone: the measures of the stones drawn on a face, the courses they sit in, the mortar share, and a line on
the masonry type that the measures suggest.

The stones are the program's, kept with the elevation (``chunk.meta["drawings"]["frames"][<face>]["stones"][<face>]``,
polygons in the face's ``(u, v)`` metres): found by its stone tool (SAM at tiles) on the face's elevation, or drawn.

* each stone: its centre, width and height (its extent along ``u`` and ``v``), area, the best-fit ellipse (the ellipse
  with the same second moments: major and minor axis, elongation = major / minor, orientation of the major axis from
  the horizontal of the face, degrees), rectangularity (area / area of the smallest enclosing rectangle);
* the mortar share: 1 - the stones' area / the wall's area (the face's bounds, or a polygon drawn on the face);
* the courses: the stones' centres sorted by ``v`` and split where two neighbours are further apart than half the
  median stone height; each course with its mean height (``v`` of its centres), its number of stones, its thickness
  (the median height of its stones) and the slope of the line through its centres (horizontal within 2 degrees); a
  group is a course when it has 3 stones or more whose centres lie within a quarter of its thickness (rms) of that
  line - stones scattered at every height are not a course;
* the masonry type, as the measures suggest it - never a verdict - with the rules said with it (:data:`RULES`).

A face whose elevation is coarser than a quarter of its smallest stone is refused, with the numbers: the measures
would be those of the pixels."""
from __future__ import annotations

import csv
import json
import time

import numpy as np

from . import walls
from .texts import metres, num, t
from .walls import NotReady

COURSE_MIN_STONES = 3
COURSE_SCATTER = 0.25  # the centres of a course within a quarter of its thickness of its line
HORIZONTAL_DEG = 2.0
COURSED_SHARE = 0.70
REGULAR_SPREAD = 0.20
SQUARED_ELONGATION = 2.0
SQUARED_RECTANGULARITY = 0.8
RESOLUTION_SHARE = 0.25
RULES = (("A filari: almeno il 70 % delle pietre sta in filari (3 pietre o più) inclinati meno di 2°.",
          "Coursed: at least 70 % of the stones sit in courses (3 stones or more) sloping less than 2°."),
         ("Regolare: l'altezza dei filari varia poco (deviazione mediana assoluta / mediana < 20 %).",
          "Regular: the height of the courses varies little (median absolute deviation / median < 20 %)."),
         ("A conci squadrati: allungamento mediano delle pietre < 2 e rettangolarità mediana > 0,8.",
          "Squared: median elongation of the stones < 2 and median rectangularity > 0.8."))


def face_stones(chunk, eid: str) -> list:
    """The stones kept with the elevation ``eid``, as shapely polygons in its ``(u, v)``."""
    from shapely import wkt

    from pyarchinit_photogrammetry.engine.elevation import frame_drawings

    out = []
    for text in (frame_drawings(chunk, eid).get("stones") or {}).get(eid) or []:
        try:
            g = wkt.loads(text)
        except Exception:  # noqa: BLE001 - a broken outline is left out
            continue
        for part in getattr(g, "geoms", [g]):
            if part.geom_type == "Polygon" and part.area > 0:
                out.append(part)
    return out


def ellipse(poly) -> tuple[float, float, float]:
    """``(major, minor, orientation)`` of the ellipse with the second moments of the polygon's area: axis lengths in
    metres (for a rectangle ``w`` x ``h`` they are ``w`` and ``h``), orientation of the major axis in degrees from
    the horizontal (-90 to 90)."""
    xy = np.asarray(poly.exterior.coords)[:-1]
    c = np.asarray(poly.centroid.coords[0])
    x, y = (xy - c).T
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    area = cross.sum() / 2.0
    if abs(area) < 1e-15:
        return 0.0, 0.0, 0.0
    ixx = (cross * (y * y + y * y1 + y1 * y1)).sum() / 12.0  # integral of y^2
    iyy = (cross * (x * x + x * x1 + x1 * x1)).sum() / 12.0  # integral of x^2
    ixy = (cross * (x * y1 + 2 * x * y + 2 * x1 * y1 + x1 * y)).sum() / 24.0
    cov = np.array([[iyy, ixy], [ixy, ixx]]) / area
    w, vecs = np.linalg.eigh(cov)
    w = np.clip(w, 0, None)
    major, minor = np.sqrt(12.0 * w[1]), np.sqrt(12.0 * w[0])  # the rectangle's sides
    ang = float(np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1])))
    ang = (ang + 90.0) % 180.0 - 90.0
    return float(major), float(minor), ang


def measure(poly, n: int) -> dict:
    u0, v0, u1, v1 = poly.bounds
    major, minor, ang = ellipse(poly)
    rect = poly.minimum_rotated_rectangle.area
    c = poly.centroid
    return {"n": n, "u": float(c.x), "v": float(c.y), "width": float(u1 - u0), "height": float(v1 - v0),
            "area": float(poly.area), "major": major, "minor": minor,
            "elongation": float(major / minor) if minor > 0 else None, "orientation": ang,
            "rectangularity": float(poly.area / rect) if rect > 0 else None, "course": None}


def find_courses(rows: list[dict]) -> list[dict]:
    """The courses of the stones (``rows`` of :func:`measure`, their ``course`` set): centres sorted by ``v``, a new
    course where the gap to the previous centre is more than half the median stone height."""
    if not rows:
        return []
    gap = 0.5 * float(np.median([r["height"] for r in rows]))
    order = sorted(range(len(rows)), key=lambda i: rows[i]["v"])
    groups, current = [], [order[0]]
    for a, b in zip(order[:-1], order[1:]):
        if rows[b]["v"] - rows[a]["v"] > gap:
            groups.append(current)
            current = []
        current.append(b)
    groups.append(current)
    courses = []
    for k, idx in enumerate(groups):
        u = np.array([rows[i]["u"] for i in idx])
        v = np.array([rows[i]["v"] for i in idx])
        slope, scatter = None, float(v.std())
        if len(idx) >= 2 and np.ptp(u) > 1e-6:
            b, a = np.polyfit(u, v, 1)
            slope = float(np.degrees(np.arctan(b)))
            scatter = float(np.sqrt(np.mean((v - (a + b * u)) ** 2)))
        thickness = float(np.median([rows[i]["height"] for i in idx]))
        for i in idx:
            rows[i]["course"] = k + 1
        courses.append({"n": k + 1, "height": float(v.mean()), "stones": len(idx), "thickness": thickness,
                        "u_from": float(u.min()), "u_to": float(u.max()), "u_mean": float(u.mean()),
                        "slope_deg": slope, "scatter": scatter,
                        "horizontal": slope is not None and abs(slope) < HORIZONTAL_DEG,
                        "counts": len(idx) >= COURSE_MIN_STONES and scatter <= COURSE_SCATTER * thickness})
    return courses


def typology(rows: list[dict], courses: list[dict]) -> dict:
    """The masonry type the measures suggest (:data:`RULES`): ``coursed``, ``regular`` (None without courses),
    ``squared``, the numbers behind each, and the line said to the user."""
    n = len(rows)
    in_courses = sum(c["stones"] for c in courses if c["counts"] and c["horizontal"])
    share = in_courses / n if n else 0.0
    counted = [c["thickness"] for c in courses if c["counts"]]
    spread = None
    if len(counted) >= 2:
        med = float(np.median(counted))
        spread = float(np.median(np.abs(np.array(counted) - med)) / med) if med > 0 else None
    elong = [r["elongation"] for r in rows if r["elongation"] is not None]
    rect = [r["rectangularity"] for r in rows if r["rectangularity"] is not None]
    med_elong = float(np.median(elong)) if elong else None
    med_rect = float(np.median(rect)) if rect else None
    coursed = share >= COURSED_SHARE
    regular = None if spread is None else spread < REGULAR_SPREAD
    squared = bool(med_elong is not None and med_rect is not None and med_elong < SQUARED_ELONGATION
                   and med_rect > SQUARED_RECTANGULARITY)
    words = [t("a filari", "coursed") if coursed else t("senza filari evidenti", "uncoursed")]
    if regular is None:
        words.append(t("regolarità dei filari non valutabile", "regularity of the courses not assessable"))
    else:
        words.append(t("regolare", "regular") if regular else t("irregolare", "irregular"))
    words.append(t("a conci squadrati", "of squared stones") if squared else t("di pietre non squadrate",
                                                                               "of unsquared stones"))
    why = [t(f"{num(100 * share, 0)} % delle pietre in filari orizzontali",
             f"{num(100 * share, 0)} % of the stones in horizontal courses")]
    if spread is not None:
        why.append(t(f"variazione dell'altezza dei filari {num(100 * spread, 0)} %",
                     f"spread of the courses' height {num(100 * spread, 0)} %"))
    if med_elong is not None:
        why.append(t(f"allungamento mediano {num(med_elong, 2)}", f"median elongation {num(med_elong, 2)}"))
    if med_rect is not None:
        why.append(t(f"rettangolarità mediana {num(med_rect, 2)}", f"median rectangularity {num(med_rect, 2)}"))
    line = t(f"Le misure suggeriscono una muratura {', '.join(words)} ({'; '.join(why)}).",
             f"The measures suggest a masonry {', '.join(words)} ({'; '.join(why)}).")
    return {"coursed": coursed, "regular": regular, "squared": squared, "share_in_courses": share,
            "course_spread": spread, "median_elongation": med_elong, "median_rectangularity": med_rect, "line": line}


def region_of(entry: dict, region=None):
    """The wall's area on the face: ``region`` (a polygon in ``(u, v)``, shapely or vertices) or the face's bounds."""
    from shapely.geometry import Polygon, box

    if region is None:
        return box(*entry["bounds"])
    if hasattr(region, "geom_type"):
        return region.buffer(0)
    return Polygon(np.asarray(region, np.float64)[:, :2]).buffer(0)


def measure_face(chunk, wid: str, eid: str | None = None, region=None, progress=None) -> dict:
    """The measures of the stones of a face, its courses, the mortar share and the type suggested; the tables in the
    face's folder (``pietre.csv``, ``filari.csv``, ``pietre.xlsx``, ``pietre_misure.gpkg``, ``pietre.json``); the
    summary kept with the wall. ``region``: the wall's area on the face (default: the region kept from a previous
    run, else the face's bounds)."""
    from shapely import wkt as swkt

    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    w, entry = walls.need_face(chunk, wid, eid)
    stones = face_stones(chunk, entry["id"])
    if not stones:
        raise NotReady(t("La faccia non ha pietre: usa lo strumento Pietre del programma sul prospetto della faccia "
                         "(Trova le pietre), poi Misura le pietre.", "The face has no stones: use the program's "
                         "Stones tool on the face's elevation (Find the stones), then Measure the stones."))
    res = float(entry.get("resolution") or 0.0)
    rows = [measure(p, i + 1) for i, p in enumerate(stones)]
    smallest = min(min(r["width"], r["height"]) for r in rows)
    if res > RESOLUTION_SHARE * smallest:
        raise NotReady(t(f"Il prospetto della faccia ha {num(res * 1000, 1)} mm per pixel, più di un quarto della "
                         f"pietra più piccola ({num(smallest * 1000, 0)} mm): le misure sarebbero quelle dei pixel. "
                         f"Rifai la faccia a {num(RESOLUTION_SHARE * smallest * 1000, 1)} mm per pixel o meno.",
                         f"The face's elevation has {num(res * 1000, 1)} mm per pixel, more than a quarter of the "
                         f"smallest stone ({num(smallest * 1000, 0)} mm): the measures would be the pixels'. Make "
                         f"the face again at {num(RESOLUTION_SHARE * smallest * 1000, 1)} mm per pixel or finer."))
    previous = walls.results(chunk, w["id"], entry["id"]).get("stones") or {}
    if region is None and previous.get("region"):
        area_poly = swkt.loads(previous["region"])
        region_kind = "drawn"
    else:
        area_poly = region_of(entry, region)
        region_kind = "bounds" if region is None else "drawn"
    progress(0.3, t("Filari", "Courses"))
    courses = find_courses(rows)
    stones_area = float(sum(p.intersection(area_poly).area for p in stones))
    mortar = 1.0 - stones_area / area_poly.area if area_poly.area > 0 else None
    typ = typology(rows, courses)
    folder = walls.face_folder(chunk, w["id"], entry["id"])
    folder.mkdir(parents=True, exist_ok=True)
    progress(0.6, t("Tabelle", "Tables"))
    write_csv(folder / "pietre.csv", rows, STONE_COLUMNS)
    write_csv(folder / "filari.csv", courses, COURSE_COLUMNS)
    summary = {"stones": len(rows), "courses": len([c for c in courses if c["counts"]]), "mortar": mortar,
               "wall_area": float(area_poly.area), "stones_area": stones_area, "region": area_poly.wkt,
               "region_kind": region_kind, "resolution": res, "smallest": smallest,
               "median_width": float(np.median([r["width"] for r in rows])),
               "median_height": float(np.median([r["height"] for r in rows])),
               "typology": typ, "made": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_xlsx(folder / "pietre.xlsx", rows, courses, summary, entry)
    write_gpkg(folder / "pietre_misure.gpkg", stones, rows)
    write_world_gpkg(chunk, entry, folder / "pietre_misure_mondo.gpkg", stones, rows)
    (folder / "pietre.json").write_text(json.dumps({"stones": rows, "courses": courses, "summary": summary},
                                                   indent=1, ensure_ascii=False))
    summary["files"] = {k: str((folder / f).relative_to(chunk.data_dir)) for k, f in (
        ("csv", "pietre.csv"), ("courses_csv", "filari.csv"), ("xlsx", "pietre.xlsx"), ("gpkg", "pietre_misure.gpkg"),
        ("json", "pietre.json"))}
    summary["course_rows"] = courses
    walls.set_result(chunk, w["id"], entry["id"], "stones", summary)
    progress(1.0, t("Pietre misurate", "Stones measured"))
    return {**summary, "rows": rows}


def set_region(chunk, wid: str, eid: str, region) -> None:
    """The wall's area on the face (a polygon in ``(u, v)``) kept for the mortar share."""
    w, entry = walls.need_face(chunk, wid, eid)
    stones = dict(walls.results(chunk, w["id"], entry["id"]).get("stones") or {})
    stones["region"] = region_of(entry, region).wkt
    stones["region_kind"] = "drawn"
    walls.set_result(chunk, w["id"], entry["id"], "stones", stones)


STONE_COLUMNS = (("n", "n"), ("course", "filare"), ("u", "u_m"), ("v", "v_m"), ("width", "larghezza_m"),
                 ("height", "altezza_m"), ("area", "area_m2"), ("major", "asse_maggiore_m"),
                 ("minor", "asse_minore_m"), ("elongation", "allungamento"), ("orientation", "orientamento_gradi"),
                 ("rectangularity", "rettangolarita"))
COURSE_COLUMNS = (("n", "filare"), ("height", "quota_media_m"), ("stones", "pietre"), ("thickness", "spessore_m"),
                  ("u_from", "u_da_m"), ("u_to", "u_a_m"), ("slope_deg", "pendenza_gradi"),
                  ("horizontal", "orizzontale"))


def _cell(value):
    if isinstance(value, float):
        return round(value, 5)
    return value


def write_csv(path, rows: list[dict], columns) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        out = csv.writer(f)
        out.writerow([name for _key, name in columns])
        for r in rows:
            out.writerow([_cell(r.get(key)) for key, _name in columns])


def write_xlsx(path, rows, courses, summary, entry) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Font

    wb = Workbook()
    ws = wb.active
    ws.title = t("Pietre", "Stones")
    for sheet, data, columns in ((ws, rows, STONE_COLUMNS), (wb.create_sheet(t("Filari", "Courses")), courses,
                                                             COURSE_COLUMNS)):
        sheet.append([name for _k, name in columns])
        for c in sheet[1]:
            c.font = Font(bold=True)
        for r in data:
            sheet.append([_cell(r.get(k)) for k, _n in columns])
    s = wb.create_sheet(t("Sintesi", "Summary"))
    typ = summary["typology"]
    for label, value in ((t("Faccia", "Face"), entry.get("name")), (t("Pietre", "Stones"), summary["stones"]),
                         (t("Filari", "Courses"), summary["courses"]),
                         (t("Quota di malta", "Mortar share"), _cell(summary["mortar"])),
                         (t("Area del muro (m²)", "Wall area (m²)"), _cell(summary["wall_area"])),
                         (t("Area delle pietre (m²)", "Stones' area (m²)"), _cell(summary["stones_area"])),
                         (t("Risoluzione (m/px)", "Resolution (m/px)"), _cell(summary["resolution"])),
                         (t("Tipo suggerito", "Type suggested"), typ["line"])):
        s.append([label, value])
    s.append([])
    for it, en in RULES:
        s.append([t("Regola", "Rule"), t(it, en)])
    wb.save(path)


def write_gpkg(path, stones, rows) -> None:
    """``pietre_misure`` in the face's ``(u, v)`` (no CRS): the outline of each stone with its measures."""
    from pyarchinit_photogrammetry.io import gpkg

    if path.exists():
        gpkg.refuse_foreign(path)
        path.unlink()
    columns = {name: ("INTEGER" if key in ("n", "course") else "TEXT" if key == "horizontal" else "REAL")
               for key, name in STONE_COLUMNS}
    gpkg.write_table(path, "pietre_misure", "POLYGON", columns,
                     [(p, {name: _cell(r.get(key)) for key, name in STONE_COLUMNS}) for p, r in zip(stones, rows)],
                     None)


def write_world_gpkg(chunk, entry: dict, path, stones, rows) -> None:
    """``pietre_misure`` in 3D on the plane of the face, in the chunk's CRS (QGIS's project of the plan, in 3D)."""
    from pyarchinit_photogrammetry.io import gpkg

    if path.exists():
        gpkg.refuse_foreign(path)
        path.unlink()
    con = gpkg.connect(path)
    try:
        con.execute("BEGIN IMMEDIATE")
        gpkg.create(con, gpkg.srs_of(entry.get("crs") or chunk.crs))
        con.execute("COMMIT")
    finally:
        con.close()
    add_to_export(chunk, entry, {"gpkg_world": path}, rows, stones)


def add_to_export(chunk, entry: dict, paths: dict, rows: list[dict], stones: list) -> None:
    """The stones' measures as the table ``pietre_misure`` beside the ``pietre`` table of the GeoPackages the program's
    ``export_drawings`` wrote: in ``(u, v)`` in the unfolded one, in 3D on the plane of the face in the world one."""
    import shapely

    from pyarchinit_photogrammetry.engine.elevation import frame_from_dict
    from pyarchinit_photogrammetry.io import gpkg

    frame = frame_from_dict(entry["frame"])
    columns = {name: ("INTEGER" if key in ("n", "course") else "REAL") for key, name in STONE_COLUMNS}
    values = [{name: _cell(r.get(key)) for key, name in STONE_COLUMNS} for r in rows]
    for key, z in (("gpkg", False), ("gpkg_world", True)):
        path = paths.get(key)
        if path is None or not path.exists():
            continue
        if z:
            srs = gpkg.srs_of(entry.get("crs") or chunk.crs)[0]
            geoms = [shapely.transform(shapely.force_3d(p), lambda c: frame.point(c[:, 0], c[:, 1],
                                                                                   np.zeros(len(c))), include_z=True)
                     for p in stones]
        else:
            srs, geoms = -1, stones
        con = gpkg.connect(path)
        try:
            con.execute("BEGIN IMMEDIATE")
            cols = gpkg.ensure_table(con, "pietre_misure", "POLYGON", columns, srs, recreate=True, z=z)
            gpkg.replace_rows(con, "pietre_misure", cols, list(zip(geoms, values)), srs, z=z)
            con.execute("COMMIT")
        except BaseException:
            if con.in_transaction:
                con.execute("ROLLBACK")
            raise
        finally:
            con.close()


def export_face(chunk, wid: str, eid: str | None, folder, stem: str | None = None) -> dict:
    """The program's export of the face's drawings (``export_drawings``: DXF, SVG, PNG, the two GeoPackages) with the
    stones' measures added to the GeoPackages and the tables beside them. Returns the paths."""
    import shutil
    from pathlib import Path

    from pyarchinit_photogrammetry.engine.elevation import export_drawings

    w, entry = walls.need_face(chunk, wid, eid)
    paths = export_drawings(chunk, entry, folder, stem=stem)
    stones = face_stones(chunk, entry["id"])
    if stones:
        rows = [measure(p, i + 1) for i, p in enumerate(stones)]
        find_courses(rows)
        add_to_export(chunk, entry, paths, rows, stones)
    face = walls.face_folder(chunk, w["id"], entry["id"])
    stem = stem or Path(paths["dxf"]).name.replace("_disegni.dxf", "")
    for name in ("pietre.csv", "filari.csv", "pietre.xlsx", "sezioni.dxf", "sezioni.svg"):
        if (face / name).exists():
            target = Path(folder) / f"{stem}_{name}"
            shutil.copy2(face / name, target)
            paths[name.replace(".", "_")] = target
    return paths


def stones_text(s: dict) -> str:
    """The stones of a face in words, for the pane and the log."""
    if not s or not isinstance(s.get("typology"), dict):  # nothing measured yet (perhaps only the area drawn)
        return ""
    lines = [t(f"{s['stones']} pietre (larghezza mediana {metres(s['median_width'])}, altezza mediana "
               f"{metres(s['median_height'])}), {s['courses']} filari; malta {num(100 * (s['mortar'] or 0), 1)} % "
               f"dell'area del muro ({'disegnata' if s.get('region_kind') == 'drawn' else 'limiti della faccia'}).",
               f"{s['stones']} stones (median width {metres(s['median_width'])}, median height "
               f"{metres(s['median_height'])}), {s['courses']} courses; mortar {num(100 * (s['mortar'] or 0), 1)} % "
               f"of the wall's area ({'drawn' if s.get('region_kind') == 'drawn' else 'bounds of the face'}).")]
    for c in s.get("course_rows") or []:
        if not c["counts"]:
            continue
        slope = "-" if c["slope_deg"] is None else f"{num(c['slope_deg'], 1)}°"
        lines.append(t(f"filare {c['n']}: quota {num(c['height'], 3)}, {c['stones']} pietre, spessore "
                       f"{metres(c['thickness'])}, pendenza {slope}",
                       f"course {c['n']}: height {num(c['height'], 3)}, {c['stones']} stones, thickness "
                       f"{metres(c['thickness'])}, slope {slope}"))
    lines.append(s["typology"]["line"])
    return "\n".join(lines)
