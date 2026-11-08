"""From the result wanted to the flight plan.

Choose the goal (an orthophoto at 2 cm, a terrain model at 5 cm, a full 3D model...) and the height,
overlaps, blocks and, if the terrain is uneven, a height for every point that keeps the distance from the
ground constant are worked out here. It is all repeatable arithmetic.

The ground comes from the project's DEM (:class:`~ppg_flightplan.dem.RasterDem`) where it reaches, else from
the public tiles (:mod:`~ppg_flightplan.tiles`) moved by the offset measured between the two.

Ported from the author's Photo3D planner (``progetta``): algorithms, constants and rounding unchanged. New in
the plugin: a margin around the area, the crossed double grid, obliques forced on or off, the GSD of every
shot on the terrain (:func:`shot_report`) and the warnings of the design (:func:`design_warnings`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import plan as pl
from . import route as rt
from . import tiles
from .camera import Camera, camera as list_camera
from .geometry import LatLon, METRES_PER_DEGREE, centroid, distance, metres_per_degree_lon
from .texts import t

# The highest height allowed in Europe without authorisation, and the one below which nothing is designed.
MAX_HEIGHT = 120.0
MIN_HEIGHT = 10.0
# The 120 m limit is FROM THE GROUND, not from the takeoff. Following a terrain that rises, the height above
# the takeoff can and must go beyond: cutting it there would mean flying closer to the ground than wanted,
# and on a steep slope even below the ground. The ceiling here is the technical one of the DJI drones.
CEILING_ABOVE_TAKEOFF = 500.0
# Out of the box a DJI does not climb above 120 m over the takeoff: it must be said, and how much would be
# left from the ground if that limit is not raised.
FACTORY_LIMIT = 120.0

GOALS = [
    {"id": "orto", "nome": "Ortofoto", "descrizione": "Una foto piana e georiferita: la risoluzione è il GSD.",
     "nomeEn": "Orthophoto", "descrizioneEn": "A flat, georeferenced photo: the resolution is the GSD.",
     "unita": "cm/px", "valore": 2.0, "minimo": 0.2, "massimo": 10.0, "obliqui": False},
    {"id": "dem", "nome": "Modello del terreno (DEM)",
     "descrizione": "La maglia del terreno: serve un GSD più fine della cella.",
     "nomeEn": "Terrain model (DEM)", "descrizioneEn": "The terrain mesh: the GSD must be finer than the cell.",
     "unita": "cm", "valore": 5.0, "minimo": 1.0, "massimo": 50.0, "obliqui": False},
    {"id": "modello3d", "nome": "Modello 3D completo",
     "descrizione": "Nadirale più due obliqui incrociati: volumi e pareti.",
     "nomeEn": "Full 3D model", "descrizioneEn": "Nadir plus two crossed obliques: volumes and walls.",
     "unita": "cm", "valore": 2.0, "minimo": 0.3, "massimo": 10.0, "obliqui": True},
    {"id": "ispezione", "nome": "Ispezione ravvicinata",
     "descrizione": "Dettaglio fine su una struttura: quote basse, molte foto.",
     "nomeEn": "Close inspection", "descrizioneEn": "Fine detail on a structure: low altitude, many shots.",
     "unita": "cm/px", "valore": 0.5, "minimo": 0.1, "massimo": 3.0, "obliqui": True},
    {"id": "ricognizione", "nome": "Ricognizione veloce",
     "descrizione": "Vedere tutto in poco tempo, con poca batteria.",
     "nomeEn": "Quick survey", "descrizioneEn": "See everything in little time, on little battery.",
     "unita": "cm/px", "valore": 5.0, "minimo": 1.0, "massimo": 20.0, "obliqui": False},
    {"id": "gsd", "nome": "GSD a mano", "descrizione": "Decidi tu i centimetri per pixel.",
     "nomeEn": "GSD by hand", "descrizioneEn": "You choose the centimetres per pixel.",
     "unita": "cm/px", "valore": 2.0, "minimo": 0.1, "massimo": 20.0, "obliqui": False},
]

# Thresholds of the design warnings (new in the plugin).
GSD_TOLERANCE = 1.25           # a shot whose GSD is coarser than this times the block's nominal one
MIN_EFFECTIVE_SIDE = 0.6       # side overlap under which a strip is said to lose it
MIN_SHOTS_PER_STRIP = 3        # fewer than this in a strip: it adds turns and little coverage


def goal(id_: str) -> dict:
    for o in GOALS:
        if o["id"] == id_:
            return o
    raise ValueError(t("obiettivo sconosciuto: {0}", "unknown goal: {0}").format(id_))


def goal_name(o: dict) -> str:
    return t(o["nome"], o["nomeEn"])


def gsd_for_goal(id_: str, value: float) -> float:
    """The centimetres per pixel needed for that result."""
    if value <= 0:
        raise ValueError(t("il valore dell'obiettivo dev'essere positivo", "the value of the goal must be positive"))
    if id_ == "dem":
        return value / 2.5        # a clean DEM cell wants two or three pixels per side
    if id_ == "modello3d":
        return value              # the resolution of the model follows the GSD
    return value                  # orthophoto, inspection, survey, GSD by hand


def height_for_gsd(cam: Camera, gsd_cm: float, high: bool) -> float:
    """The height that gives that GSD with this camera: the inverse of Camera.gsd_cm()."""
    return gsd_cm * cam.focal_mm * cam.pixels_for(high) / (cam.width_mm * 100.0)


def _centre(area: list[LatLon]) -> LatLon:
    return (sum(p[0] for p in area) / len(area), sum(p[1] for p in area) / len(area))


def long_edge_heading(area: list[LatLon]) -> int:
    """The direction of the strips: along the longest side, fewer turns and more regular coverage."""
    best, longest = 0, -1.0
    for a, b in zip(area, area[1:] + area[:1]):
        d = distance(a, b)
        if d > longest:
            longest = d
            dlat = (b[0] - a[0]) * 111_320.0
            dlon = (b[1] - a[1]) * 111_320.0 * math.cos(math.radians(a[0]))
            best = int(round(math.degrees(math.atan2(dlon, dlat)))) % 180
    return best


def terrain_heights(points: list[LatLon], agl: float, ground_takeoff: float,
                    grounds: list[Optional[float]]) -> list[float]:
    """The height of every point so that the drone stays ``agl`` metres from the ground.

    The heights of the files are relative to the takeoff point: to the distance from the ground wanted is
    added how much the terrain rises (or falls) with respect to the takeoff. The limits are kept anyway.
    """
    out = []
    for ground in grounds:
        if ground is None:
            out.append(agl)
            continue
        h = agl + (ground - ground_takeoff)
        out.append(round(min(CEILING_ABOVE_TAKEOFF, max(MIN_HEIGHT, h)), 1))
    return out


def buffered_area(area: list[LatLon], margin_m: float) -> list[LatLon]:
    """The area grown by ``margin_m`` metres on every side, measured in a local azimuthal equidistant system
    centred on the area (pyproj), so the margin is true metres; corners kept square (mitred)."""
    if margin_m <= 0:
        return [tuple(p) for p in area]
    from pyproj import Transformer
    from shapely.geometry import Polygon

    c = _centre(area)
    local = f"+proj=aeqd +lat_0={c[0]} +lon_0={c[1]} +datum=WGS84 +units=m +no_defs"
    fwd = Transformer.from_crs("EPSG:4326", local, always_xy=True)
    back = Transformer.from_crs(local, "EPSG:4326", always_xy=True)
    xs, ys = fwd.transform([p[1] for p in area], [p[0] for p in area])
    poly = Polygon(list(zip(xs, ys)))
    if not poly.is_valid:
        poly = poly.buffer(0)
    grown = poly.buffer(float(margin_m), join_style="mitre", mitre_limit=2.0)
    if grown.geom_type != "Polygon":
        grown = max(grown.geoms, key=lambda g: g.area)
    ring = list(grown.exterior.coords)[:-1]
    lon, lat = back.transform([p[0] for p in ring], [p[1] for p in ring])
    return [(float(a), float(b)) for a, b in zip(lat, lon)]


def _custom(custom_camera) -> Optional[dict]:
    if custom_camera is None:
        return None
    return custom_camera.to_json() if isinstance(custom_camera, Camera) else dict(custom_camera)


def design(area: list[LatLon], goal_id: str, value: float, camera: str = "mini4pro", high: bool = True,
           battery_minutes: float = 18.0, follow_terrain: bool = False, mode: str = "per_punto",
           takeoff: Optional[LatLon] = None, name: str = "Piano progettato", dem=None, margin_m: float = 0.0,
           double_grid: bool = False, obliques: Optional[bool] = None, custom_camera=None) -> dict:
    """The complete plan for that goal, with the reason of every choice.

    ``margin_m`` grows the area before the grid; ``double_grid`` adds a second nadir grid crossed at 90
    degrees; ``obliques`` None follows the goal, True/False forces the two crossed oblique passes on or off;
    ``custom_camera`` (a :class:`Camera` or its JSON) is stored in the plan as ``cameraPersonale`` and used.
    """
    if len(area) < 3:
        raise ValueError(t("l'area ha bisogno di almeno tre vertici", "the area needs at least three vertices"))
    o = goal(goal_id)
    custom = _custom(custom_camera)
    if custom is not None:
        camera = str(custom.get("id"))
    cam = list_camera(camera, custom)
    gsd = gsd_for_goal(goal_id, value)
    height = height_for_gsd(cam, gsd, high)
    reasons: list[str] = []
    if height > MAX_HEIGHT:
        reasons.append(t("Con {0:.2f} cm/px servirebbero {1:.0f} m: oltre i {2:.0f} m consentiti, "
                         "quota portata al massimo (GSD reale {3:.2f} cm/px).",
                         "{0:.2f} cm/px would need {1:.0f} m: above the {2:.0f} m allowed, "
                         "height brought to the maximum (real GSD {3:.2f} cm/px).")
                       .format(gsd, height, MAX_HEIGHT, cam.gsd_cm(MAX_HEIGHT, high)))
        height = MAX_HEIGHT
    if height < MIN_HEIGHT:
        reasons.append(t("Quota alzata a {0:.0f} m per sicurezza (GSD reale {1:.2f} cm/px).",
                         "Height raised to {0:.0f} m for safety (real GSD {1:.2f} cm/px).")
                       .format(MIN_HEIGHT, cam.gsd_cm(MIN_HEIGHT, high)))
        height = MIN_HEIGHT
    height = round(height, 1)
    real_gsd = cam.gsd_cm(height, high)
    if margin_m and margin_m > 0:
        area = buffered_area(area, margin_m)
        reasons.append(t("Area allargata di {0:g} m su ogni lato, perché i bordi dell'area disegnata abbiano "
                         "la stessa copertura del centro.",
                         "Area grown by {0:g} m on every side, so the edges of the drawn area get the same "
                         "coverage as the centre.").format(margin_m))
    heading = long_edge_heading(area)
    reasons.insert(0, t("{0} a {1:g} {2}: serve un GSD di {3:.2f} cm/px, cioè {4:g} m con {5} ({6:.2f} cm/px reali).",
                        "{0} at {1:g} {2}: it needs a GSD of {3:.2f} cm/px, that is {4:g} m with {5} "
                        "({6:.2f} cm/px real).")
                   .format(goal_name(o), value, o["unita"], gsd, height, cam.name, real_gsd))

    front, side = (80, 70) if not o["obliqui"] else (85, 75)
    blocks = [{
        "tipo": "griglia", "nome": "A nadirale", "area": [list(p) for p in area], "quota": height,
        "inclinazione": -90, "avanti": front, "lato": side, "orientamento": heading,
        "velocita": 5.0, "modo": mode, "alta": high, "attivo": True,
    }]
    reasons.append(t("Strisciate a {0}°, lungo il lato più lungo dell'area, "
                     "sovrapposizione {1}% avanti e {2}% laterale.",
                     "Strips at {0}°, along the longest side of the area, overlap {1}% forward and {2}% sideways.")
                   .format(heading, front, side))
    if double_grid:
        crossed = (heading + 90) % 180
        blocks.append(dict(blocks[0], nome=f"A2 nadirale {crossed}°", orientamento=crossed,
                           area=[list(p) for p in area]))
        reasons.append(t("Seconda griglia nadirale incrociata a {0}°: le strisciate nei due versi legano meglio "
                         "il blocco e vedono le pareti da due lati.",
                         "Second nadir grid crossed at {0}°: strips in both directions tie the block better "
                         "and see the walls from two sides.").format(crossed))
    with_obliques = o["obliqui"] if obliques is None else bool(obliques)
    if with_obliques:
        for block_name, direction in (("B obliquo", heading), ("C obliquo", (heading + 90) % 180)):
            blocks.append({
                "tipo": "griglia", "nome": f"{block_name} {direction}°", "area": [list(p) for p in area],
                "quota": height, "inclinazione": -45, "avanti": 85, "lato": 75, "orientamento": direction,
                "velocita": 4.5, "modo": mode, "alta": False, "attivo": True,
            })
        reasons.append(t("Due passate oblique incrociate a 45°, che vedono le pareti verticali: senza di quelle "
                         "il modello 3D resta piatto sui fianchi.",
                         "Two crossed oblique passes at 45°, which see the vertical walls: without them the 3D "
                         "model stays flat on its sides."))
        if obliques is True and not o["obliqui"]:
            reasons.append(t("Passate oblique aggiunte su richiesta: l'obiettivo da solo non le chiede.",
                             "Oblique passes added on request: the goal alone does not ask for them."))
    elif obliques is False and o["obliqui"]:
        reasons.append(t("Passate oblique tolte su richiesta: senza di quelle le pareti verticali si vedono poco.",
                         "Oblique passes removed on request: without them the vertical walls are seen poorly."))

    piano = {"nome": name, "camera": camera, "alta": high, "minutiBatteria": battery_minutes,
             "blocchi": blocks, "obiettivo": {"id": goal_id, "valore": value, "gsd": round(real_gsd, 2)},
             "versione": 1}
    if custom is not None:
        piano["cameraPersonale"] = custom
    if margin_m and margin_m > 0:
        piano["margine"] = float(margin_m)
    if takeoff:
        piano["decollo"] = list(takeoff)

    if follow_terrain:
        for b in piano["blocchi"]:
            b["seguiTerreno"] = True
        try:
            outcome = apply_terrain(piano, dem=dem)
            if any(w["tipo"] == "scarto_non_misurato" for w in outcome["avvisi"]):
                reasons.append(t("Il DEM del progetto non si sovrappone mai alle quote libere sotto il volo: lo scarto "
                                 "fra le due sorgenti non si misura, e quote di riferimenti diversi non si mescolano. "
                                 "Si usano solo le quote libere.",
                                 "The project DEM never overlaps the free heights under the flight: the offset "
                                 "between the two sources cannot be measured, and heights of different references "
                                 "are not mixed. Only the free heights are used."))
            reasons.append(t("Il volo segue il terreno: ogni punto ha la sua quota rispetto al decollo "
                             "({0:g} m sul mare), così la distanza dal suolo — e quindi il GSD — resta la stessa "
                             "anche dove il terreno sale. Quote da {1:g} a {2:g} m.",
                             "The flight follows the terrain: every point has its height above the takeoff "
                             "({0:g} m above sea level), so the distance from the ground - and so the GSD - stays "
                             "the same where the terrain rises too. Heights from {1:g} to {2:g} m.")
                           .format(outcome["quotaDecollo"], outcome["minima"], outcome["massima"]))
        except (tiles.TerrainError, OSError) as e:
            reasons.append(t("Quote del terreno non disponibili ({0}): il volo resta a quota costante. "
                             "Si può riprovare quando c'è rete.",
                             "Terrain heights not available ({0}): the flight stays at a constant height. "
                             "It can be tried again with the network.").format(e))
    counts = pl.compute(piano)
    tot = counts["totale"]
    reasons.append(t("Vengono {0} scatti in {1:.0f} minuti, {2:.1f} batterie e {3:.1f} km di volo.",
                     "It makes {0} shots in {1:.0f} minutes, {2:.1f} batteries and {3:.1f} km of flight.")
                   .format(tot["scatti"], tot["minuti"], tot["batterie"], tot["km"]))
    if tot["batterie"] > 6:
        reasons.append(t("Sono tante batterie: conviene restringere l'area, accettare un GSD più grosso "
                         "o dividere il lavoro in più giornate.",
                         "That is many batteries: better to narrow the area, accept a coarser GSD or split "
                         "the work over several days."))
    return {"piano": piano, "motivi": reasons, "conti": counts, "gsd": round(real_gsd, 2), "quota": height}


# The grid of heights the plan carries: fine enough to read the ground under any takeoff point, small
# enough to stay inside the file.
PASSO_GRIGLIA_M = 20.0
MARGINE_GRIGLIA_M = 150.0
NODI_MASSIMI = 6000


def terrain_grid(piano: dict, dem, z: int, offset: Optional[float], ground_reference: float,
                 reference: LatLon) -> Optional[dict]:
    """The heights of the ground around the site, to put INSIDE the plan.

    The heights of a flight following the terrain are counted from where the drone really takes off. If the
    pilot takes off elsewhere than planned, all the heights must move by how much higher or lower the ground
    is there. With this grid the plan carries the ground with it, and that correction can be made without
    network.
    """
    points: list[LatLon] = [reference]
    for b in piano.get("blocchi", []):
        if pl.active(b):
            p, _ = pl.route_of_block({k: v for k, v in b.items() if k != "quote"}, piano)
            points += p.points
            for sp in b.get("speciali") or []:
                if sp.get("punto"):
                    points.append((float(sp["punto"][0]), float(sp["punto"][1])))
    flight = list(points)
    for d in (piano.get("decolli") or {}).values():
        q = (float(d[0]), float(d[1]))
        # a leg takeoff kilometres from the flight would widen the mesh until it is useless
        if min(distance(q, v) for v in flight) <= pl.FAR_TAKEOFF_M:
            points.append(q)
    if len(points) < 2:
        return None
    lat0 = sum(p[0] for p in points) / len(points)
    m_lat, m_lon = 111_320.0, 111_320.0 * math.cos(math.radians(lat0))
    south = min(p[0] for p in points) - MARGINE_GRIGLIA_M / m_lat
    north = max(p[0] for p in points) + MARGINE_GRIGLIA_M / m_lat
    west = min(p[1] for p in points) - MARGINE_GRIGLIA_M / m_lon
    east = max(p[1] for p in points) + MARGINE_GRIGLIA_M / m_lon
    step = PASSO_GRIGLIA_M
    while True:
        rows = int((north - south) * m_lat / step) + 2
        cols = int((east - west) * m_lon / step) + 2
        if rows * cols <= NODI_MASSIMI:
            break
        step *= 1.25                        # large areas: a wider mesh, the file stays light
    step_lat, step_lon = step / m_lat, step / m_lon
    nodes = [(south + r * step_lat, west + c * step_lon) for r in range(rows) for c in range(cols)]
    heights = ground_heights(nodes, dem, z, offset)
    if sum(1 for q in heights if q is not None) < len(heights) * 0.5:
        return None
    return {"sud": round(south, 7), "ovest": round(west, 7), "passoLat": step_lat, "passoLon": step_lon,
            "righe": rows, "colonne": cols, "passoM": round(step, 1),
            "quote": [None if q is None else round(q, 1) for q in heights],
            "suoloRiferimento": round(ground_reference, 1),
            "riferimento": [round(reference[0], 7), round(reference[1], 7)]}


def height_warnings(heights: list[float], grounds: list[Optional[float]], ground_takeoff: float) -> list[dict]:
    """What must be said before flying a plan that follows the terrain.

    A stock DJI does not climb above 120 m over the takeoff. If the plan asks for more and that limit is not
    raised, the drone stops there - and on a slope that means passing closer to the ground, or not passing at
    all. Here it is said how high the plan climbs and how much would be left from the ground.
    """
    if not heights:
        return []
    highest = max(heights)
    if highest <= FACTORY_LIMIT:
        return []
    distances = [min(q, FACTORY_LIMIT) - (s - ground_takeoff)
                 for q, s in zip(heights, grounds) if s is not None]
    lowest = min(distances) if distances else None
    warning = {"tipo": "oltre_limite_di_serie", "quotaMassima": round(highest, 1),
               "limite": FACTORY_LIMIT, "dalSuoloSeNonAlzi": None if lowest is None else round(lowest, 1)}
    warning["pericolo"] = lowest is not None and lowest < 20.0
    # every height at the technical ceiling: it is not a terrain, it is a wrong reference
    if min(heights) >= CEILING_ABOVE_TAKEOFF - 0.5:
        warning["alTetto"] = True
        warning["pericolo"] = True
    return [warning]


def source_offset(points: list[LatLon], dem, z: int, n: Optional[int] = 12) -> Optional[float]:
    """How much the project DEM and the public tiles tell different heights, where they overlap.

    It is nobody's fault: the tiles give heights above sea level (geoid), the DEM of a survey may take them
    from the GNSS of the photos (ellipsoid), and between the two there can be tens of metres. Measuring the
    offset where both are seen, the points falling outside the DEM can still be used without a step in the
    middle of the flight.

    ``n`` points are sampled, as the planner does; ``n=None`` uses them all (the plan measures it so, see
    :func:`_overlap`).
    """
    return _overlap(points, dem, z, n)[0]


def _overlap(points: list[LatLon], dem, z: int, n: Optional[int] = None) -> tuple[Optional[float], int, int]:
    """The offset between the project DEM and the tiles (the median where both answer), and at how many of
    the points each of them answers."""
    step = 1 if n is None else max(1, len(points) // max(n, 1))
    sampled = points[::step]
    offsets = []
    known_dem = known_tiles = 0
    for p, mine in zip(sampled, _dem_heights(sampled, dem)):
        free = tiles.height(p[0], p[1], z)
        known_dem += mine is not None
        known_tiles += free is not None
        if mine is not None and free is not None:
            offsets.append(mine - free)
    if not offsets:
        return None, known_dem, known_tiles
    offsets.sort()
    return offsets[len(offsets) // 2], known_dem, known_tiles   # the median: a wrong pixel does not move it


def _dem_heights(points: list[LatLon], dem) -> list[Optional[float]]:
    many = getattr(dem, "heights", None)
    if callable(many):
        return list(many(points))
    return [dem.height(p[0], p[1]) for p in points]


def ground_heights(points: list[LatLon], dem, z: int, offset: Optional[float]) -> list[Optional[float]]:
    """The heights of the terrain under the points: from the project DEM where it reaches, from the public
    tiles elsewhere (moved by the offset between the two)."""
    if dem is None:
        return tiles.point_heights(points, z)
    out: list[Optional[float]] = []
    for p, mine in zip(points, _dem_heights(points, dem)):
        if mine is not None:
            out.append(mine)
            continue
        free = tiles.height(p[0], p[1], z)
        out.append(None if free is None else free + (offset or 0.0))
    return out


class HeightsNotMixed(tiles.TerrainError):
    """The project DEM gives the takeoff but never meets the free heights, which give the rest: the offset
    between them cannot be measured and two height references are never mixed, so there is no terrain."""

    def __init__(self, dem: str):
        self.dem = dem
        super().__init__(t("quota del decollo solo dal DEM del progetto ({0}), che non si sovrappone mai alle quote "
                           "libere: lo scarto non si misura e quote di riferimenti diversi non si mescolano",
                           "takeoff height only from the project DEM ({0}), which never overlaps the free heights: "
                           "the offset cannot be measured and heights of different references are not mixed")
                         .format(Path(dem).name))


@dataclass
class _Ground:
    """The terrain reference of a plan: the point the heights refer to, the ground there, the offset between
    the project DEM and the tiles, the DEM actually used (None: the tiles alone) and whether a DEM was left
    out because the two sources could not be tied together."""
    reference: LatLon
    ground_takeoff: float
    offset: Optional[float]
    dem: object
    dem_left_out: Optional[str] = None


def _reference_and_ground(piano: dict, z: int, dem) -> _Ground:
    """The point the heights refer to (the takeoff, or the first waypoint if there is none), the height of the
    ground there and the offset between the project DEM and the public tiles.

    Two sources of heights are mixed only when the offset between them is measured: the tiles are above sea
    level, a survey DEM is often above the ellipsoid, tens of metres apart. The offset is measured on every
    point of the flight, the takeoff and the leg takeoffs (the planner sampled twelve route points, so a DEM
    around the takeoff alone gave the takeoff from the DEM and the route from the tiles, uncorrected: the
    whole flight that gap lower). If the DEM answers where the tiles answer too but never at the same point,
    the DEM is left out and the tiles are used for everything (``dem_left_out``); a DEM that answers nowhere
    on the flight is simply not used. Without the tiles (no network) the DEM is used alone, as the planner did.
    """
    blocks = piano.get("blocchi", [])
    reference = pl.plan_takeoff(piano)
    if reference is not None and pl.takeoff_far(piano) is not None:
        # A takeoff kilometres from the flight was left there by another plan: counting the heights from that
        # point would send them all to the ceiling (or to the floor). It is treated as missing, and the caller
        # tells the user (``decolloLontanoM`` in the counts).
        reference = None
    if reference is None:
        for b in blocks:
            if pl.active(b):
                p, _ = pl.route_of_block(b, piano)
                if p.points:
                    reference = p.points[0]
                    break
    if reference is None:
        raise ValueError(t("il piano non ha punti su cui misurare il terreno",
                           "the plan has no points to measure the terrain on"))
    # The takeoff is measured with the same source as the waypoints, or the heights would start shifted.
    offset = None
    ground_takeoff = None
    left_out = None
    if dem is not None:
        probe = [reference]
        for b in blocks:
            if pl.active(b):
                p, _ = pl.route_of_block(b, piano)
                probe += p.points
        probe += [(float(d[0]), float(d[1])) for d in (piano.get("decolli") or {}).values()]
        offset, known_dem, known_tiles = _overlap(probe, dem, z)
        if offset is None and known_tiles:
            if known_dem:
                info = dem.to_json() if callable(getattr(dem, "to_json", None)) else {}
                left_out = str(info.get("file") or "DEM")
            dem = None
    if dem is not None:
        ground_takeoff = dem.height(reference[0], reference[1])
        if ground_takeoff is None:
            free = tiles.height(reference[0], reference[1], z)
            ground_takeoff = None if free is None else free + (offset or 0.0)
    if ground_takeoff is None:
        ground_takeoff = tiles.height(reference[0], reference[1], z)
    if ground_takeoff is None and left_out:
        raise HeightsNotMixed(left_out)
    if ground_takeoff is None:
        raise tiles.TerrainError(t("quota del decollo non disponibile", "height of the takeoff not available"))
    return _Ground(reference, ground_takeoff, offset, dem, left_out)


def _height_source(dem, offset: Optional[float]) -> dict:
    """Which height system ``quotaDecollo`` and the terrain grid are in: the project DEM's (the tiles moved by
    the offset where it ends) or the public tiles' (above sea level)."""
    if dem is None:
        return {"sorgente": "tegole", "fonte": tiles.ATTRIBUTION}
    info = dem.to_json() if callable(getattr(dem, "to_json", None)) else {}
    return {"sorgente": "dem", "file": info.get("file"), "epsg": info.get("epsg"),
            "scartoM": None if offset is None else round(offset, 1)}


def _takeoffs_and_grid(piano: dict, dem, z: int, offset: Optional[float], ground_takeoff: float,
                       reference: LatLon) -> None:
    """What the plan carries of the terrain: the ground under every takeoff, the grid around the site and the
    height system they are in (``fonteQuote``: the KML's absolute heights and the GeoPackage need it)."""
    piano["quotaDecollo"] = round(ground_takeoff, 1)
    piano["fonteQuote"] = _height_source(dem, offset)
    declared = pl.plan_takeoff(piano)
    far = pl.takeoff_far(piano) is not None
    # with a far takeoff (ignored) the declared one is remembered anyway: otherwise "was the takeoff moved?"
    # would always answer yes, and the terrain would be redone forever
    piano["puntoDecollo"] = list(declared) if far and declared else [reference[0], reference[1]]
    if far:
        piano["riferimentoQuote"] = [reference[0], reference[1]]
    else:
        piano.pop("riferimentoQuote", None)
    # The heights of the file refer to the point the drone really takes off from. With per-leg takeoffs that
    # point changes: between the top of a mound and the plain there may be twenty metres, and the same
    # heights would give twenty metres of error on the distance from the ground. Here the ground under every
    # takeoff is measured; the correction is applied when the legs are made.
    ground_takeoffs = {}
    for key, d in (piano.get("decolli") or {}).items():
        point = (float(d[0]), float(d[1]))
        q = ground_heights([point], dem, z, offset)[0]
        if q is not None:
            ground_takeoffs[key] = {"punto": [point[0], point[1]], "suolo": round(q, 1)}
    piano["suoloDecolli"] = ground_takeoffs
    # the terrain around the site travels with the plan: it brings the heights back to the real takeoff,
    # and tells the special points where the ground is
    g = terrain_grid(piano, dem, z, offset, ground_takeoff, reference)
    if g is not None:
        piano["terreno"] = g


def _left_out_warning(g: _Ground) -> dict:
    return {"tipo": "scarto_non_misurato", "dem": g.dem_left_out}


def measure_ground(piano: dict, z: Optional[int] = None, dem=None) -> dict:
    """Measures the terrain around the site WITHOUT making any block follow it: it serves the special points
    that descend, to count the metres from the ground and not to touch it."""
    z = z if z is not None else tiles.zoom_for_detail(piano.get("dettaglioTerreno"))
    g = _reference_and_ground(piano, z, dem)
    _takeoffs_and_grid(piano, g.dem, z, g.offset, g.ground_takeoff, g.reference)
    out = {"quotaDecollo": round(g.ground_takeoff, 1), "griglia": piano.get("terreno") is not None}
    if g.dem_left_out:
        out["avvisi"] = [_left_out_warning(g)]
    return out


def apply_terrain(piano: dict, z: Optional[int] = None, only: Optional[int] = None, dem=None) -> dict:
    """Gives every waypoint its height, to stay at the same distance from the ground.

    The takeoff point is the reference, because that is what the drone refers its heights to.

    It works **block by block**: each has its own "follow the terrain" tick, because in a plan one may want it
    on a grid climbing the mound and not on a ring turning at a fixed height around it. With ``only`` a single
    block is redone (the one just ticked); without it, all those asking for it - and if none asks, all the
    active ones, because then the whole plan wants it.
    """
    # the detail chosen for this plan: how finely the heights are read
    z = z if z is not None else tiles.zoom_for_detail(piano.get("dettaglioTerreno"))
    blocks = piano.get("blocchi", [])
    g = _reference_and_ground(piano, z, dem)
    reference, ground_takeoff, offset, dem = g.reference, g.ground_takeoff, g.offset, g.dem
    asked = [i for i, b in enumerate(blocks) if pl.active(b) and b.get("seguiTerreno")]
    if only is not None:
        to_do = [only] if 0 <= only < len(blocks) else []
    elif asked:
        to_do = asked
    else:
        to_do = [i for i, b in enumerate(blocks) if pl.active(b)]
    every: list[float] = []
    grounds_seen: list[Optional[float]] = []
    points = 0
    for i in to_do:
        b = blocks[i]
        p, _ = pl.route_of_block(b, piano)
        if not p.points:
            continue
        ground = ground_heights(p.points, dem, z, offset)
        heights = terrain_heights(p.points, float(b.get("quota", p.height)), ground_takeoff, ground)
        b["quote"] = heights
        b["seguiTerreno"] = True
        every += heights
        grounds_seen += ground
        points += len(p.points)
    if not every:
        raise ValueError(t("nessun blocco da far seguire al terreno", "no block to make follow the terrain"))
    piano["seguiTerreno"] = any(b.get("seguiTerreno") for b in blocks)
    _takeoffs_and_grid(piano, dem, z, offset, ground_takeoff, reference)
    warnings = height_warnings(every, grounds_seen, ground_takeoff)
    if g.dem_left_out:
        warnings.insert(0, _left_out_warning(g))
    outcome = {"avvisi": warnings, "punti": points, "blocchi": len(to_do), "quotaDecollo": round(ground_takeoff, 1),
               "minima": round(min(every), 1), "massima": round(max(every), 1), "zoom": z,
               "magliaM": round(tiles.metres_per_pixel(z, reference[0]), 1)}
    if dem is not None:
        outcome["dem"] = dem.to_json()
        outcome["scartoM"] = None if offset is None else round(offset, 1)
    return outcome


def blocks_with_old_heights(piano: dict) -> list[int]:
    """The blocks that say they follow the terrain but whose heights no longer match their points.

    It happens when the block changes after the computation (shooting mode, height, overlaps): the points
    become others and the heights, still the old ones, would be dropped in silence - the file would fly flat
    while the pane says "follows the terrain".
    """
    old = []
    for i, b in enumerate(piano.get("blocchi", [])):
        if not b.get("seguiTerreno"):
            continue
        p, _ = pl.route_of_block({k: v for k, v in b.items() if k != "quote"}, piano)
        q = b.get("quote")
        if not q or len(q) != len(p.points):
            old.append(i)
    return old


def takeoffs_to_measure(piano: dict) -> bool:
    """A per-leg takeoff added or moved after the computation: its ground must be measured again. It holds
    also for a plan saved before the terrain travelled with the plan: it follows the terrain but carries no
    grid, and the heights could not be brought back to the real takeoff."""
    if not any(b.get("seguiTerreno") and b.get("quote") for b in piano.get("blocchi", [])) \
            and not pl.specials_need_ground(piano):
        return False
    if not piano.get("terreno") or not piano.get("puntoDecollo"):
        return True
    # heights counted from a takeoff left in another region, or the other way round: the takeoff is back in
    # its place and the heights still refer to the first waypoint
    if (pl.takeoff_far(piano) is not None) != bool(piano.get("riferimentoQuote")):
        return True
    if any(_outside_grid(piano["terreno"], sp["punto"]) for b in piano.get("blocchi", [])
           for sp in b.get("speciali") or [] if sp.get("punto")):
        return True                     # a special point put outside the known terrain
    known = piano.get("suoloDecolli") or {}
    for key, d in (piano.get("decolli") or {}).items():
        entry = known.get(key)
        if not entry or distance(tuple(entry["punto"]), (float(d[0]), float(d[1]))) > 2.0:
            return True
    general = pl.plan_takeoff(piano)
    old = piano.get("puntoDecollo")
    if general and old and distance(tuple(old), general) > 2.0:
        return True
    return False


def _outside_grid(g: dict, point) -> bool:
    lat, lon = float(point[0]), float(point[1])
    return not (g["sud"] <= lat <= g["sud"] + g["passoLat"] * (g["righe"] - 1)
                and g["ovest"] <= lon <= g["ovest"] + g["passoLon"] * (g["colonne"] - 1))


def refresh_terrain(piano: dict, dem=None) -> dict:
    """Redoes the heights of the blocks that have old ones: called before saving and exporting, so a file that
    says "follows the terrain" really follows it."""
    old = blocks_with_old_heights(piano)
    redone, failed = [], []
    if not old and takeoffs_to_measure(piano):
        # the reference changed: every block following the terrain is redone
        old = [i for i, b in enumerate(piano.get("blocchi", [])) if b.get("seguiTerreno")]
        if not old:
            # no block follows the terrain, but the special points want to know it
            try:
                measure_ground(piano, dem=dem)
                return {"rifatti": [], "falliti": [], "suolo": True}
            except (tiles.TerrainError, OSError, ValueError) as e:
                return {"rifatti": [], "falliti": [{"blocco": None, "motivo": str(e)}]}
    for i in old:
        try:
            apply_terrain(piano, only=i, dem=dem)
            redone.append(i)
        except (tiles.TerrainError, OSError, ValueError) as e:
            failed.append({"blocco": i, "motivo": str(e)})
    return {"rifatti": redone, "falliti": failed}


def remove_terrain(piano: dict, only: Optional[int] = None) -> dict:
    """Back to the flight at a constant height, on the whole plan or on one block."""
    blocks = piano.get("blocchi", [])
    count = 0
    for i, b in enumerate(blocks):
        if only is not None and i != only:
            continue
        if b.pop("quote", None):
            count += 1
        b.pop("seguiTerreno", None)
    if not any(b.get("seguiTerreno") for b in blocks) and not pl.specials_need_ground(piano):
        for key in ("seguiTerreno", "quotaDecollo", "puntoDecollo", "suoloDecolli", "terreno", "fonteQuote"):
            piano.pop(key, None)
    return {"blocchi": count}


# ------------------------------------------------------------------ new: GSD per shot and design warnings

def _block_high(b: dict, piano: dict) -> bool:
    return bool(b.get("alta", piano.get("alta", False)))


def shot_report(piano: dict, dem=None, z: Optional[int] = None) -> list[dict]:
    """Every shot of every active block on the terrain: where it is, its height above the takeoff, the ground
    under it, its distance from the ground (``agl``) and the GSD that distance gives.

    The ground comes from the same sources as the terrain following (project DEM, public tiles moved by the
    offset); ``ground``, ``agl`` and ``gsd_cm`` are None where the ground is unknown, all of them when even the
    takeoff's ground is unknown. The GSD is the nadir one (``cam.gsd_cm(agl)``), also for oblique blocks.
    """
    z = z if z is not None else tiles.zoom_for_detail(piano.get("dettaglioTerreno"))
    cam = pl.plan_camera(piano)
    left_out = None
    try:
        g = _reference_and_ground(piano, z, dem)
        ground_takeoff, offset, dem, left_out = g.ground_takeoff, g.offset, g.dem, g.dem_left_out
    except (tiles.TerrainError, OSError, ValueError) as e:
        ground_takeoff, offset = None, None
        left_out = getattr(e, "dem", None)
    out = []
    for i, b in enumerate(piano.get("blocchi", [])):
        if not pl.active(b):
            continue
        route, _ = pl.flown_route(b, piano)
        high = _block_high(b, piano)
        shots = rt.shot_points(route)
        latlons = [p for p, _ in shots]
        grounds = ground_heights(latlons, dem, z, offset) if (ground_takeoff is not None and latlons) \
            else [None] * len(latlons)
        rows = []
        for (p, h), g in zip(shots, grounds):
            agl = None if g is None or ground_takeoff is None else h - (g - ground_takeoff)
            gsd = cam.gsd_cm(agl, high) if agl is not None and agl > 0 else None
            rows.append({"latlon": p, "height_rel": h, "ground": g, "agl": agl, "gsd_cm": gsd})
        out.append({"block": i, "name": b.get("nome") or b.get("tipo", "griglia"),
                    "kind": b.get("tipo", "griglia"), "pitch": float(b.get("inclinazione", -90)),
                    "height": float(b.get("quota", route.height)), "high": high,
                    "nominal_gsd_cm": cam.gsd_cm(float(b.get("quota", route.height)), high),
                    "ground_takeoff": ground_takeoff, "dem_left_out": left_out, "shots": rows})
    return out


def _strips(b: dict, piano: dict, shots: list[dict]) -> list[list[dict]]:
    """The shots of a grid block grouped by strip, in the order of the strips.

    The strips are those of :func:`~ppg_flightplan.route.grid`, found with its own arithmetic: the same centre,
    the same rotated frame (``x cos a + y sin a`` across the strips) and the same positions
    (``min_x + width (s + 0.5) / strips``). Every shot goes to the nearest strip; a shot farther than a quarter
    of the pitch from every strip is on a turn (shooting by interval) and belongs to none.
    """
    cam = pl.plan_camera(piano)
    height = float(b.get("quota", 50))
    side = float(b.get("lato", 65)) / 100
    area = [tuple(p) for p in b.get("area", [])]
    if len(area) < 3 or not shots:
        return []
    c = centroid(area)
    k_lon = metres_per_degree_lon(c[0])
    ang = math.radians(float(b.get("orientamento", 0)))
    ca, sa = math.cos(ang), math.sin(ang)

    def across(p: LatLon) -> float:
        return (p[1] - c[1]) * k_lon * ca + (p[0] - c[0]) * METRES_PER_DEGREE * sa

    xs = [across(p) for p in area]
    min_x, width = min(xs), max(xs) - min(xs)
    strip_step = cam.footprint(height) * (1 - side)
    count = max(1, round(width / max(strip_step, 1.0)))
    pitch = width / count
    groups: list[list[dict]] = [[] for _ in range(count)]
    for s in shots:
        f = (across(s["latlon"]) - min_x) / pitch - 0.5 if pitch > 0 else 0.0
        k = min(count - 1, max(0, int(round(f))))
        if pitch <= 0 or abs(f - k) <= 0.25:
            groups[k].append(s)
    return [g for g in groups if g]


def design_warnings(piano: dict, dem=None, z: Optional[int] = None) -> list[dict]:
    """What should be said about the plan before flying it, measured on the terrain.

    - ``gsd_oltre``: shots whose GSD is coarser than 1.25 times the block's nominal one (a constant height
      over a terrain falling below the takeoff), with how many and the worst;
    - ``sovrapposizione_calata``: a nadir grid strip whose side overlap, at its lowest distance from the
      ground, falls under 60 % (``1 - strip spacing / footprint``), with the worst value;
    - ``troppo_vicino_al_suolo``: shots less than 10 m from the ground (or under it), with how many and the
      lowest distance - a constant height over a rising terrain;
    - ``strisciata_corta``: a grid strip with fewer than three shots;
    - ``oltre_limite_di_serie``: the planner's warning when the heights go above the stock 120 m.

    Without any ground (no DEM, no tiles) only ``strisciata_corta`` can be told.
    """
    cam = pl.plan_camera(piano)
    report = shot_report(piano, dem=dem, z=z)
    blocks = piano.get("blocchi", [])
    out: list[dict] = []
    every_h: list[float] = []
    every_g: list[Optional[float]] = []
    ground_takeoff = None
    for entry in report:
        b = blocks[entry["block"]]
        shots = entry["shots"]
        if entry["ground_takeoff"] is not None:
            ground_takeoff = entry["ground_takeoff"]
        every_h += [s["height_rel"] for s in shots]
        every_g += [s["ground"] for s in shots]
        nominal = entry["nominal_gsd_cm"]
        coarse = [s["gsd_cm"] for s in shots if s["gsd_cm"] is not None and s["gsd_cm"] > GSD_TOLERANCE * nominal]
        if coarse:
            out.append({"tipo": "gsd_oltre", "blocco": entry["block"], "nome": entry["name"],
                        "scatti": len(coarse), "nominale": round(nominal, 2), "massimo": round(max(coarse), 2)})
        low = [s["agl"] for s in shots if s["agl"] is not None and s["agl"] < MIN_HEIGHT]
        if low:
            out.append({"tipo": "troppo_vicino_al_suolo", "blocco": entry["block"], "nome": entry["name"],
                        "scatti": len(low), "minimo": round(min(low), 1)})
        if entry["kind"] != "griglia":
            continue
        strips = _strips(b, piano, shots)
        short = [len(s) for s in strips if len(s) < MIN_SHOTS_PER_STRIP]
        if short:
            out.append({"tipo": "strisciata_corta", "blocco": entry["block"], "nome": entry["name"],
                        "strisciate": len(short), "scattiMinimi": min(short)})
        if entry["pitch"] > -80:           # obliques: their footprint across is not the nadir one
            continue
        side = float(b.get("lato", 65)) / 100
        spacing = cam.footprint(entry["height"]) * (1 - side)
        worst = None
        for s in strips:
            agls = [x["agl"] for x in s if x["agl"] is not None]
            if not agls:
                continue
            lowest = min(agls)
            eff = 1.0 - spacing / cam.footprint(lowest) if lowest > 0 else -1.0
            worst = eff if worst is None else min(worst, eff)
        if worst is not None and worst < MIN_EFFECTIVE_SIDE:
            out.append({"tipo": "sovrapposizione_calata", "blocco": entry["block"], "nome": entry["name"],
                        "nominale": side, "peggiore": round(worst, 3)})
    if ground_takeoff is not None:
        out += height_warnings(every_h, every_g, ground_takeoff)
    left_out = next((e["dem_left_out"] for e in report if e.get("dem_left_out")), None)
    if left_out:
        out.insert(0, {"tipo": "scarto_non_misurato", "dem": left_out})
    return out
