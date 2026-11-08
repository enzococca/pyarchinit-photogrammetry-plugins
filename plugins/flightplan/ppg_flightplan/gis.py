"""The plan for the GIS: KML and GeoJSON of the routes.

Heights, and how the files declare them:

- **KML**: absolute heights, ``altitudeMode`` ``absolute``: the height of the takeoff ground plus the height
  above the takeoff, so that Google Earth and QGIS draw the route where it flies. The takeoff height and its
  height system (the public tiles, above sea level, or the project DEM's) are written in the document
  (``ppg:quotaDecollo``, ``ppg:riferimentoQuote`` and its description); without a known takeoff height it is 0
  and the description says the heights are above the takeoff. The description of every route keeps the
  height above the takeoff ("quota 110 m"). The drone never reads this file.
- **GeoJSON**: z = height above the takeoff, property ``riferimento_quote`` = ``"decollo"``.
- **GeoPackage** (:func:`write_gpkg`, in the chunk's system): ``quota_decollo_m`` = height above the takeoff the drone
  file of the leg uses, Z = the ground of that takeoff + it (in the height system of the DEM used), every table with
  a ``riferimento_quote`` column saying which; Z = the height above the takeoff when the takeoff's ground is not known.
- **Area KML** (:func:`area_kml`, SPERIMENTALE): only the outline of a grid, for the DJI apps that make their own
  strips; the values to set again are in its description.

Files name the DEM by its file name only, never by the folder it is in (a path of the user's disks).

Ported from the author's Photo3D planner (which wrote heights above the takeoff labelled ``relativeToGround``);
the absolute heights and the declarations are the plugin's.
"""
from __future__ import annotations

import json
from pathlib import PurePosixPath, PureWindowsPath
from typing import Optional

from .camera import Camera
from .export_dji import (DATA_HEIGHT_REFERENCE, DATA_STOPS, DATA_TAKEOFF_HEIGHT, DATA_TERRAIN, NS_KML,
                         _xml, route_description)
from .geometry import LatLon, distance
from .route import Route, shot_points
from .specials import num
from .texts import t

HEIGHTS_ABOVE_TAKEOFF = "decollo"


def file_name(path) -> str:
    """The name of a file without its folder, whatever the system that wrote the path (``C:\\...`` too)."""
    if not path:
        return ""
    text = str(path)
    return PureWindowsPath(text).name if "\\" in text else PurePosixPath(text).name


def height_reference(ground_takeoff: Optional[float], source: Optional[dict] = None) -> str:
    """In words, what the absolute heights of the KML are measured from."""
    if ground_takeoff is None:
        return t("quota del decollo sconosciuta: le quote del file sono sul punto di decollo (decollo = 0 m)",
                 "takeoff height unknown: the heights of the file are above the takeoff point (takeoff = 0 m)")
    source = source or {}
    if source.get("sorgente") == "dem":
        system = t("nel sistema di quote del DEM del progetto", "in the height system of the project DEM")
        if source.get("file"):
            system += f" ({file_name(source['file'])})"
    elif source.get("sorgente") == "tegole":
        system = t("sul livello del mare (Terrain Tiles)", "above sea level (Terrain Tiles)")
    else:
        system = t("in un sistema di quote non dichiarato", "in an undeclared height system")
    return t("quote assolute = quota del suolo al decollo ({0} m, {1}) + quota sul decollo",
             "absolute heights = ground height at the takeoff ({0} m, {1}) + height above the takeoff"
             ).format(num(ground_takeoff), system)


def kml(routes: list[Route], title: str, takeoffs: Optional[list[tuple[str, LatLon]]] = None,
        terrain: Optional[dict] = None, ground_takeoff: Optional[float] = None,
        height_source: Optional[dict] = None) -> str:
    """The routes of the plan, and the takeoff points if there are: in the field one sees where to stand.

    Heights absolute (``ground_takeoff`` + height above the takeoff; 0 + it when the takeoff height is not
    known), declared in the document."""
    base = float(ground_takeoff) if ground_takeoff is not None else 0.0
    reference = height_reference(ground_takeoff, height_source)
    out = ['<?xml version="1.0" encoding="UTF-8"?>\n',
           f'<kml xmlns="{NS_KML}"><Document><name>{_xml(title)}</name>\n',
           f"<description>{_xml(reference)}</description>\n",
           f'<ExtendedData><Data name="{DATA_TAKEOFF_HEIGHT}"><value>{num(base)}</value></Data>'
           f'<Data name="{DATA_HEIGHT_REFERENCE}"><value>{_xml(reference)}</value></Data>']
    if terrain:
        out.append(f'<Data name="{DATA_TERRAIN}"><value>'
                   f'{_xml(json.dumps(terrain, separators=(",", ":")))}</value></Data>')
    out.append("</ExtendedData>\n")
    for name, (lat, lon) in takeoffs or []:
        out.append(f"<Placemark><name>{_xml(name)}</name><description>decollo</description>"
                   f"<Point><coordinates>{lon:.7f},{lat:.7f},0</coordinates></Point></Placemark>\n")
    for p in routes:
        coords = " ".join(f"{lon:.7f},{lat:.7f},{num(round(base + p.height_at(i), 3))}"
                          for i, (lat, lon) in enumerate(p.points))
        stops = [s_.to_json(i) for i, s_ in enumerate(p.stops or []) if s_]
        data = (f'<ExtendedData><Data name="{DATA_STOPS}"><value>'
                f'{_xml(json.dumps(stops, separators=(",", ":")))}</value></Data></ExtendedData>\n' if stops else "")
        out.append(f"<Placemark><name>{_xml(p.name)}</name><description>{_xml(route_description(p))}</description>\n"
                   + data +
                   "<LineString><altitudeMode>absolute</altitudeMode><tessellate>1</tessellate>"
                   f"<coordinates>{coords} </coordinates></LineString></Placemark>\n")
        if p.poi:
            out.append(f"<Placemark><name>{_xml(p.name)} - centro</name><Point><coordinates>"
                       f"{p.poi[1]:.7f},{p.poi[0]:.7f},0</coordinates></Point></Placemark>\n")
    out.append("</Document></kml>\n")
    return "".join(out)


def geojson(routes: list[Route], cam: Camera, high: bool, areas: Optional[list[list[LatLon]]] = None) -> str:
    """The routes as LineStrings with z = height above the takeoff (``riferimento_quote`` = ``"decollo"``), the
    point of aim and the areas."""
    features = []
    for p in routes:
        features.append({"type": "Feature",
                         "properties": {"blocco": p.name, "quota_m": p.height, "camera_gradi": p.pitch,
                                        "gsd_cm": round(cam.gsd_cm(p.height, high), 2), "scatti": p.shots(),
                                        "minuti": round(p.minutes(), 1), "velocita_ms": round(p.effective_speed, 2),
                                        "intervallo_s": p.interval, "passo_m": round(p.effective_step, 1),
                                        "modo": p.mode, "riferimento_quote": HEIGHTS_ABOVE_TAKEOFF,
                                        **({"segue_terreno": True, "quote_m": p.heights} if p.heights else {}),
                                        **({"mira": [round(p.poi[0], 7), round(p.poi[1], 7)]} if p.poi else {})},
                         "geometry": {"type": "LineString",
                                      "coordinates": [[round(lon, 7), round(lat, 7), p.height_at(i)]
                                                      for i, (lat, lon) in enumerate(p.points)]}})
        if p.poi:
            features.append({"type": "Feature", "properties": {"blocco": p.name, "ruolo": "centro"},
                             "geometry": {"type": "Point", "coordinates": [round(p.poi[1], 7), round(p.poi[0], 7)]}})
    for k, area in enumerate(areas or []):
        ring_ = [[round(lon, 7), round(lat, 7)] for lat, lon in area]
        if ring_ and ring_[0] != ring_[-1]:
            ring_.append(ring_[0])
        features.append({"type": "Feature", "properties": {"ruolo": "area", "indice": k},
                         "geometry": {"type": "Polygon", "coordinates": [ring_]}})
    return json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False, indent=1)


def plan_kml(piano: dict, title: Optional[str] = None, ground_takeoff: Optional[float] = None,
             height_source: Optional[dict] = None) -> str:
    """The KML of the whole plan: the active routes, the takeoffs, the terrain grid, and the heights absolute
    from the takeoff height - the one given (for a plan without terrain following, which has no
    ``quotaDecollo``: e.g. the project DEM at the takeoff, with ``height_source`` saying which system), else the
    one the plan measured (``quotaDecollo`` in the system of ``fonteQuote``)."""
    from .design import require_current_heights
    from .plan import plan_routes, plan_takeoffs

    require_current_heights(piano)
    if ground_takeoff is None:
        ground = piano.get("quotaDecollo")
        ground_takeoff = None if ground is None else float(ground)
        height_source = piano.get("fonteQuote") if height_source is None else height_source
    return kml(plan_routes(piano), title or piano.get("nome", "piano"), plan_takeoffs(piano),
               terrain=piano.get("terreno"), ground_takeoff=ground_takeoff, height_source=height_source)


# ------------------------------------------------------------------ the areas for the DJI apps (SPERIMENTALE)

def _anticlockwise(area: list[LatLon]) -> list[LatLon]:
    """The outer boundary of a KML polygon goes anticlockwise (lon, lat on the plane)."""
    double = sum(a[1] * b[0] - b[1] * a[0] for a, b in zip(area, area[1:] + area[:1]))
    return area if double >= 0 else list(reversed(area))


def area_kml(b: dict, piano: dict) -> str:
    """SPERIMENTALE: the area of a grid block as a polygon, for the DJI apps that make the strips of a mapping
    mission by themselves (DJI Pilot with the Mavic 2 Enterprise Advanced, GS Pro with the Phantom 4
    Multispectral). Those apps read only the outline: height, overlaps and camera must be set again by hand, and
    for this they are in the description. Not tried with the two apps."""
    from .plan import plan_camera

    cam = plan_camera(piano)
    high = bool(b.get("alta", piano.get("alta", False)))
    area = [tuple(p) for p in b.get("area", [])]
    if len(area) < 3:
        raise ValueError(t("il blocco «{0}» non ha un'area", "the block “{0}” has no area").format(
            b.get("nome") or "griglia"))
    if distance(area[0], area[-1]) < 0.01:
        area = area[:-1]
    area = _anticlockwise(area)
    height = float(b.get("quota", 50))
    values = t("Quota {0} m sul decollo · sovrapposizione frontale {1}% · laterale {2}% · camera {3}° · "
               "direzione delle strisciate {4}° · GSD circa {5:.1f} cm/px ({6}). Questi valori vanno reimpostati "
               "nell'app DJI. SPERIMENTALE: area esportata da pyarchinit-photogrammetry, non ancora provata con "
               "DJI Pilot e GS Pro.",
               "Height {0} m above the takeoff · front overlap {1}% · side {2}% · camera {3}° · "
               "direction of the strips {4}° · GSD about {5:.1f} cm/px ({6}). These values must be set again "
               "in the DJI app. SPERIMENTALE (experimental): area exported by pyarchinit-photogrammetry, not yet "
               "tried with DJI Pilot and GS Pro.").format(
        num(height), num(float(b.get("avanti", 80))), num(float(b.get("lato", 65))),
        num(float(b.get("inclinazione", -90))), num(float(b.get("orientamento", 0))), cam.gsd_cm(height, high),
        cam.name)
    ring_ = " ".join(f"{lon:.7f},{lat:.7f},0" for lat, lon in area + area[:1])
    name = b.get("nome") or "griglia"
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<kml xmlns="{NS_KML}"><Document><name>{_xml(name)}</name>\n'
            f"<Placemark><name>{_xml(name)}</name><description>{_xml(values)}</description>\n"
            "<Polygon><outerBoundaryIs><LinearRing><coordinates>"
            f"{ring_}</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>\n"
            "</Document></kml>\n")


def areas_kml(piano: dict) -> list[tuple[str, str]]:
    """One KML per active grid block: (file name, text). One file per area, because the DJI apps import one
    per mission."""
    from .plan import active_blocks, slug

    files, seen = [], set()
    for b in active_blocks(piano):
        if b.get("tipo", "griglia") != "griglia":
            continue
        base = slug(b.get("nome") or "griglia")
        name, k = base, 2
        while name in seen:
            name, k = f"{base}-{k}", k + 1
        seen.add(name)
        files.append((f"{name}-area.kml", area_kml(b, piano)))
    if not files:
        raise ValueError(t("nel piano non c'è un blocco a griglia acceso: l'area si esporta solo da quelli",
                           "the plan has no active grid block: the area is exported only from those"))
    return files


# ------------------------------------------------------------------ GeoPackage in the chunk's system

GPKG_TABLES = ("ppg_area", "ppg_strisciate", "ppg_scatti", "ppg_decolli", "ppg_target")


def _leg_ground(piano: dict, leg, general: Optional[float]) -> Optional[float]:
    """The ground height where the drone of ``leg`` takes off, in the system of ``general``: the plan's takeoff
    (``general``), or the leg's own takeoff measured by the terrain (``suoloDecolli``); None when not known."""
    from .plan import plan_takeoff

    own = leg.takeoff
    if own is None or own == plan_takeoff(piano):
        return general
    entry = (piano.get("suoloDecolli") or {}).get(leg.key)
    if entry and entry.get("suolo") is not None and distance(tuple(entry["punto"]), own) <= 2.0:
        if general is None or piano.get("quotaDecollo") is None:
            return None
        # the measured ground is in the plan's system: moved to the system of ``general`` with the takeoff's
        return float(entry["suolo"]) + (general - float(piano["quotaDecollo"]))
    return None


def write_gpkg(piano: dict, path, crs: Optional[str], ground_takeoff: Optional[float] = None,
               dem_name: str = "") -> None:
    """The plan into the GeoPackage ``path``, in ``crs`` (the chunk's; its horizontal part for a compound system):

    - ``ppg_area`` (POLYGON): the areas of the active grids;
    - ``ppg_strisciate`` (LINESTRING Z): one line per leg, with blocco, tratta, quota_m, camera_gradi, gsd_cm,
      scatti, minuti, velocita_ms and the name of the leg's files;
    - ``ppg_scatti`` (POINT Z): every shot, with blocco, tratta, n, quota_decollo_m (above the takeoff the leg
      flies from), gsd_cm and agl_m (distance from the ground, when the plan's terrain grid knows it);
    - ``ppg_decolli`` (POINT): the takeoffs; ``ppg_target`` (POINT): the GNSS targets, with ``stimato`` = 1 for a
      position estimated by the survey (never a GCP).

    Z = the ground at the takeoff + the height above it, in the height system of the DEM used (``ground_takeoff``
    with ``dem_name`` from the caller, for a plan without terrain following; else the plan's ``quotaDecollo`` in the
    system of ``fonteQuote``); without a known takeoff ground Z is the height above the takeoff. Every table says
    which in ``riferimento_quote``. A file that is not a GeoPackage is never touched (FileExistsError); the tables
    of the plan are rewritten, the other tables of the file kept. ValueError for a chunk without a system."""
    import shapely
    from pyarchinit_photogrammetry.io import gpkg
    from pyproj import Transformer

    from .design import require_current_heights
    from .plan import active, plan_camera, plan_legs, plan_takeoffs
    from .specials import ground_of_grid

    require_current_heights(piano)

    srs = gpkg.srs_of(crs)
    if srs[0] <= 0:
        raise ValueError(t("serve un sistema di riferimento: un chunk locale non ha coordinate per il GeoPackage",
                           "a reference system is needed: a local chunk has no coordinates for the GeoPackage"))
    to_xy = Transformer.from_crs("EPSG:4326", f"EPSG:{srs[0]}", always_xy=True)

    def xy(lat: float, lon: float) -> tuple[float, float]:
        return to_xy.transform(lon, lat)

    if ground_takeoff is not None:
        general = float(ground_takeoff)
        source = {"sorgente": "dem", "file": file_name(dem_name)} if dem_name else piano.get("fonteQuote")
    else:
        q = piano.get("quotaDecollo")
        general = None if q is None else float(q)
        source = piano.get("fonteQuote")
    plan_reference = height_reference(general, source)
    cam = plan_camera(piano)
    battery_minutes = float(piano.get("minutiBatteria", 18))
    grid_ground = ground_of_grid(piano.get("terreno")) if piano.get("quotaDecollo") is not None else None
    blocks = piano.get("blocchi", [])

    areas = []
    for i, b in enumerate(blocks):
        if not active(b) or b.get("tipo", "griglia") != "griglia" or len(b.get("area") or []) < 3:
            continue
        ring_ = [xy(float(lat), float(lon)) for lat, lon in b["area"]]
        areas.append((shapely.Polygon(ring_), {"blocco": i, "nome": b.get("nome", ""),
                                               "quota_m": float(b.get("quota", 50)),
                                               "riferimento_quote": t("quota del blocco sul decollo",
                                                                      "height of the block above the takeoff")}))
    strips, shots = [], []
    for leg in plan_legs(piano):
        b = blocks[leg.block_index]
        high = bool(b.get("alta", piano.get("alta", False)))
        p = leg.route
        base = _leg_ground(piano, leg, general)
        plan_base = _leg_ground(piano, leg, None if piano.get("quotaDecollo") is None
                                else float(piano["quotaDecollo"]))
        if base is None and general is not None:
            reference = t("quote sul punto di decollo della tratta (suolo del decollo sconosciuto)",
                          "heights above the takeoff point of the leg (ground of the takeoff unknown)")
        else:
            reference = plan_reference if base == general else height_reference(base, source)
        z0 = base if base is not None else 0.0
        if len(p.points) >= 2:
            line = shapely.LineString([(*xy(lat, lon), z0 + p.height_at(k))
                                       for k, (lat, lon) in enumerate(p.points)])
            strips.append((line, {"blocco": leg.block_index, "tratta": leg.number, "quota_m": p.height,
                                  "camera_gradi": p.pitch, "gsd_cm": round(cam.gsd_cm(p.height, high), 2),
                                  "scatti": p.shots(), "minuti": round(p.minutes(), 1),
                                  "batterie": round(p.minutes() / max(battery_minutes, 1), 2),
                                  "velocita_ms": round(p.effective_speed, 2), "file": leg.file_name,
                                  "riferimento_quote": reference}))
        for n, ((lat, lon), h) in enumerate(shot_points(p), 1):
            agl = None
            if grid_ground is not None and plan_base is not None:
                g = grid_ground((lat, lon))
                agl = None if g is None else round(h - (g - plan_base), 2)
            gsd = cam.gsd_cm(agl, high) if agl is not None and agl > 0 else cam.gsd_cm(p.height, high)
            shots.append((shapely.Point(*xy(lat, lon), z0 + h),
                          {"blocco": leg.block_index, "tratta": leg.number, "n": n,
                           "quota_decollo_m": round(h, 3), "gsd_cm": round(gsd, 2), "agl_m": agl,
                           "riferimento_quote": reference}))
    takeoffs = []
    for name, (lat, lon) in plan_takeoffs(piano):
        takeoffs.append((shapely.Point(*xy(lat, lon)), {"nome": name, "riferimento_quote": plan_reference}))
    targets = []
    target = piano.get("target") or {}
    for pt in target.get("punti") or []:
        targets.append((shapely.Point(*xy(float(pt["lat"]), float(pt["lon"]))),
                        {"nome": str(pt.get("nome", "")), "codice": pt.get("codice"), "quota": pt.get("quota"),
                         "stimato": 1 if pt.get("stimato") else 0,
                         "riferimento_quote": t("quota del file dei target ({0})",
                                                "height of the targets' file ({0})").format(target.get("crs", ""))}))

    reference_col = {"riferimento_quote": "TEXT"}
    layout = [
        ("ppg_area", "POLYGON", False, {"blocco": "INTEGER", "nome": "TEXT", "quota_m": "REAL", **reference_col},
         areas),
        ("ppg_strisciate", "LINESTRING", True,
         {"blocco": "INTEGER", "tratta": "INTEGER", "quota_m": "REAL", "camera_gradi": "REAL", "gsd_cm": "REAL",
          "scatti": "INTEGER", "minuti": "REAL", "batterie": "REAL", "velocita_ms": "REAL", "file": "TEXT",
          **reference_col}, strips),
        ("ppg_scatti", "POINT", True,
         {"blocco": "INTEGER", "tratta": "INTEGER", "n": "INTEGER", "quota_decollo_m": "REAL", "gsd_cm": "REAL",
          "agl_m": "REAL", **reference_col}, shots),
        ("ppg_decolli", "POINT", False, {"nome": "TEXT", **reference_col}, takeoffs),
        ("ppg_target", "POINT", False, {"nome": "TEXT", "codice": "INTEGER", "quota": "REAL", "stimato": "INTEGER",
                                        **reference_col}, targets),
    ]
    from pathlib import Path

    path = Path(path)
    if path.exists() and not gpkg.is_geopackage(path):
        gpkg.refuse_foreign(path)
        path.unlink()                          # an empty file (left by a save dialog): nothing to lose
    con = gpkg.connect(path)
    try:
        con.execute("BEGIN IMMEDIATE")
        gpkg.create(con, srs)
        for table, gtype, z, columns, features in layout:
            cols = gpkg.ensure_table(con, table, gtype, columns, srs[0], recreate=True, z=z)
            gpkg.replace_rows(con, table, cols, features, srs[0], z=z)
        con.execute("COMMIT")
    except BaseException:
        if con.in_transaction:
            con.execute("ROLLBACK")
        raise
    finally:
        con.close()
