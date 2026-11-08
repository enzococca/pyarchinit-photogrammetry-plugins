"""The plan for the GIS: KML and GeoJSON of the routes.

Heights, and how the files declare them:

- **KML**: absolute heights, ``altitudeMode`` ``absolute``: the height of the takeoff ground plus the height
  above the takeoff, so that Google Earth and QGIS draw the route where it flies. The takeoff height and its
  height system (the public tiles, above sea level, or the project DEM's) are written in the document
  (``ppg:quotaDecollo``, ``ppg:riferimentoQuote`` and its description); without a known takeoff height it is 0
  and the description says the heights are above the takeoff. The description of every route keeps the
  height above the takeoff ("quota 110 m"). The drone never reads this file.
- **GeoJSON**: z = height above the takeoff, property ``riferimento_quote`` = ``"decollo"``.

Ported from the author's Photo3D planner (which wrote heights above the takeoff labelled ``relativeToGround``);
the absolute heights and the declarations are the plugin's.
"""
from __future__ import annotations

import json
from typing import Optional

from .camera import Camera
from .export_dji import (DATA_HEIGHT_REFERENCE, DATA_STOPS, DATA_TAKEOFF_HEIGHT, DATA_TERRAIN, NS_KML,
                         _xml, route_description)
from .geometry import LatLon
from .route import Route
from .specials import num
from .texts import t

HEIGHTS_ABOVE_TAKEOFF = "decollo"


def height_reference(ground_takeoff: Optional[float], source: Optional[dict] = None) -> str:
    """In words, what the absolute heights of the KML are measured from."""
    if ground_takeoff is None:
        return t("quota del decollo sconosciuta: le quote del file sono sul punto di decollo (decollo = 0 m)",
                 "takeoff height unknown: the heights of the file are above the takeoff point (takeoff = 0 m)")
    source = source or {}
    if source.get("sorgente") == "dem":
        system = t("nel sistema di quote del DEM del progetto", "in the height system of the project DEM")
        if source.get("file"):
            system += f" ({source['file']})"
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


def plan_kml(piano: dict, title: Optional[str] = None) -> str:
    """The KML of the whole plan: the active routes, the takeoffs, the terrain grid, and the heights absolute
    from the takeoff height the plan measured (``quotaDecollo`` in the system of ``fonteQuote``)."""
    from .plan import plan_routes, plan_takeoffs

    ground = piano.get("quotaDecollo")
    return kml(plan_routes(piano), title or piano.get("nome", "piano"), plan_takeoffs(piano),
               terrain=piano.get("terreno"), ground_takeoff=None if ground is None else float(ground),
               height_source=piano.get("fonteQuote"))
