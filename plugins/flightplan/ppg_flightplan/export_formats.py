"""The flight plans for the drones that do not speak DJI's KMZ.

Four formats, one per family, all written leg by leg as the KMZs:

- **QGroundControl ``.plan``** (JSON): ArduPilot and PX4, the open survey drones and the home-built ones; it is
  loaded in QGroundControl or Mission Planner and from there onto the drone.
- **Mission Planner ``.waypoints``** (text "QGC WPL 110"): the same mission as the ``.plan`` in the form Mission
  Planner (ArduPilot) writes and reads: a home row, then the commands of the ``.plan`` in the same order.
- **Parrot ``.mavlink``** (text "QGC WPL 120" in Parrot's dialect): Anafi, Anafi USA and Anafi Ai; FreeFlight 7
  imports it from the share sheet, GroundSDK loads it directly.
- **Litchi ``.csv``**: the DJI drones without waypoints in DJI Fly (Mini 2 and SE, Air 2 and 2S, Mavic 2,
  Phantom 4), flown from Litchi on iOS and Android; the CSV is imported in the Mission Hub.

Heights are above the takeoff point, as in the KMZ, declared as each format wants it (QGroundControl and Mission
Planner frame 3 = ``MAV_FRAME_GLOBAL_RELATIVE_ALT`` and ``AltitudeMode`` 1, Parrot frame 3, Litchi
``altitudemode`` 0), and they are **one per point**: a plan that follows the terrain keeps following it in these
files, every waypoint with its own height above the takeoff of its leg (the planner wrote the block's single
height). These files must be tried in the real app before flying. The interval shooting becomes a distance
shooting (metres between the photos): the systems all do it, and it does not depend on the speed.

Ported from the author's Photo3D planner; the per-point heights and the Mission Planner file are the plugin's.
"""
from __future__ import annotations

import io
import json
import math
import zipfile
from typing import Optional

from .geometry import LatLon
from .route import Route
from .texts import t

FORMATS = {"plan": ("plan", "application/json", "QGroundControl"),
           "waypoints": ("waypoints", "text/plain", "Mission Planner"),
           "mavlink": ("mavlink", "text/plain", "Parrot FreeFlight"),
           "litchi": ("csv", "text/csv", "Litchi")}


def course_degrees(a: LatLon, b: LatLon) -> float:
    """The direction from a to b, in degrees from north clockwise (on the sphere)."""
    lat1, lat2 = math.radians(a[0]), math.radians(b[0])
    dlon = math.radians(b[1] - a[1])
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def heading_at(p: Route, i: int) -> float:
    """Where the drone looks at point i: towards the point of aim, or along the route."""
    if p.poi:
        return course_degrees(p.points[i], p.poi)
    if i + 1 < len(p.points):
        return course_degrees(p.points[i], p.points[i + 1])
    if i > 0:
        return course_degrees(p.points[i - 1], p.points[i])
    return 0.0


def _step(p: Route) -> float:
    return round(max(p.effective_step if p.mode == "intervallo" else p.photo_step, 1.0), 2)


def _height(p: Route, i: int) -> float:
    """The height of waypoint i above the takeoff: its own when the flight follows the terrain."""
    return round(p.height_at(i), 1)


# ------------------------------------------------------------------ QGroundControl

def _qgc_items(p: Route, takeoff: Optional[LatLon]) -> list[dict]:
    """The items of the QGroundControl mission: takeoff, speed, camera pitched, distance shooting (or one per
    point), the waypoints each at its own height, and the return. Heights relative to the takeoff (frame 3)."""
    v = round(p.effective_speed, 2)
    home = takeoff or p.points[0]
    items: list[dict] = []

    def item(command: int, params: list, lat: Optional[float] = None, lon: Optional[float] = None,
             alt: Optional[float] = None) -> None:
        d = {"autoContinue": True, "command": command, "doJumpId": len(items) + 1,
             "frame": 3 if lat is not None else 2,
             "params": params + ([lat, lon, alt] if lat is not None else [0, 0, 0]), "type": "SimpleItem"}
        if lat is not None:
            d.update({"AMSLAltAboveTerrain": None, "Altitude": alt, "AltitudeMode": 1})
        items.append(d)

    # the climb goes straight to the height of the first waypoint (the block's height on a flat flight)
    item(22, [0, 0, 0, None], home[0], home[1], _height(p, 0))          # MAV_CMD_NAV_TAKEOFF
    item(178, [1, v, -1, 0])                                            # MAV_CMD_DO_CHANGE_SPEED, ground speed
    item(205, [round(p.pitch, 1), 0, 0, 0])                             # MAV_CMD_DO_MOUNT_CONTROL, mode 2 below
    items[-1]["params"] = [round(p.pitch, 1), 0, 0, 0, 0, 0, 2]         # MAV_MOUNT_MODE_MAVLINK_TARGETING
    if p.poi:
        item(195, [0, 0, 0, 0], p.poi[0], p.poi[1], 0)                  # MAV_CMD_DO_SET_ROI_LOCATION
        for k in ("AMSLAltAboveTerrain", "Altitude", "AltitudeMode"):
            items[-1].pop(k, None)
    for i, (lat, lon) in enumerate(p.points):
        item(16, [0, 0, 0, None], lat, lon, _height(p, i))              # MAV_CMD_NAV_WAYPOINT
        if i == 0 and p.mode == "intervallo":
            item(206, [_step(p), 0, 1, 0])                              # MAV_CMD_DO_SET_CAM_TRIGG_DIST: starts now
        if p.mode == "per_punto":
            item(2000, [0, 0, 1, 0])                                    # MAV_CMD_IMAGE_START_CAPTURE: one photo
    if p.mode == "intervallo":
        item(206, [0, 0, 0, 0])                                         # stop the distance shooting
    item(20, [0, 0, 0, 0])                                              # MAV_CMD_NAV_RETURN_TO_LAUNCH
    return items


def qgc_plan(p: Route, takeoff: Optional[LatLon] = None, name: str = "") -> str:
    """The QGroundControl plan of a route. Heights relative to the takeoff (frame 3, ``AltitudeMode`` 1,
    ``globalPlanAltitudeMode`` 1), one per waypoint."""
    v = round(p.effective_speed, 2)
    home = takeoff or p.points[0]
    plan = {
        "fileType": "Plan",
        "geoFence": {"circles": [], "polygons": [], "version": 2},
        "groundStation": "QGroundControl",
        "mission": {"cruiseSpeed": v, "firmwareType": 3, "globalPlanAltitudeMode": 1, "hoverSpeed": v,
                    "items": _qgc_items(p, takeoff), "plannedHomePosition": [home[0], home[1], 0], "vehicleType": 2,
                    "version": 2},
        "rallyPoints": {"points": [], "version": 2},
        "version": 1,
    }
    if name:
        plan["mission"]["name"] = name
    return json.dumps(plan, indent=2)


# ------------------------------------------------------------------ Mission Planner

def _wpl_number(x) -> str:
    return "%.6f" % float(x or 0)


def mission_planner_waypoints(p: Route, takeoff: Optional[LatLon] = None) -> str:
    """The mission in Mission Planner's text format ("QGC WPL 110"): row 0 the home at the takeoff (frame 0,
    altitude 0), then the commands of :func:`qgc_plan` in the same order and with the same parameters, every row in
    frame 3 (heights relative to the takeoff, one per waypoint), autocontinue 1. The point of aim is written as
    ``MAV_CMD_DO_SET_ROI`` (201) with ``MAV_ROI_LOCATION``, as ArduPilot reads it (and as Parrot's dialect)."""
    home = takeoff or p.points[0]
    rows = ["QGC WPL 110\n", "0\t1\t0\t16\t0\t0\t0\t0\t%.8f\t%.8f\t0\t1\n" % (home[0], home[1])]
    for k, it in enumerate(_qgc_items(p, takeoff), 1):
        command, params = it["command"], list(it["params"])
        if command == 195:
            command, params = 201, [3, 0, 0, 0] + params[4:]
        p1, p2, p3, p4, lat, lon, alt = params
        rows.append("%d\t0\t3\t%d\t%s\t%s\t%s\t%s\t%.8f\t%.8f\t%s\t1\n"
                    % (k, command, _wpl_number(p1), _wpl_number(p2), _wpl_number(p3), _wpl_number(p4),
                       float(lat or 0), float(lon or 0), _wpl_number(alt)))
    return "".join(rows)


# ------------------------------------------------------------------ Parrot

def _row(index: int, command: int, p1=0.0, p2=0.0, p3=0.0, p4=0.0, lat=0.0, lon=0.0, alt=0.0) -> str:
    # index, current waypoint (0), frame 3 (global, relative height), command, 4 parameters,
    # lat, lon, height, autocontinue 1: the row GroundSDK writes
    return "%d\t0\t3\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t1\n" % (index, command, p1, p2, p3, p4, lat, lon, alt)


def parrot_mavlink(p: Route) -> str:
    """The flight plan of Parrot (Anafi, Anafi USA, Anafi Ai), in GroundSDK's dialect: takeoff, speed, camera
    pitched, distance photos (GPS-lapse) or one per point, waypoints each at its own height with the nose along the
    route or towards the point of aim, return."""
    v = round(p.effective_speed, 2)
    rows = ["QGC WPL 120\n"]
    n = 0

    def add(*args) -> None:
        nonlocal n
        rows.append(_row(n, *args))
        n += 1

    add(22)                                                # TakeOff
    add(178, 1, v)                                         # ChangeSpeed: ground speed
    add(205, round(p.pitch, 1), 0, 0, 0, 0, 0, 2)          # MountControl: camera pitch
    if p.poi:
        add(201, 3, 0, 0, 0, p.poi[0], p.poi[1], 0)        # SetRoi: point of aim on the ground
    if p.mode == "intervallo":
        add(50001, 1, _step(p))                            # SetStillCaptureMode: GPS-lapse, metres between photos
        add(2000, 0, 0, 12)                                # StartPhotoCapture: mode interval, no end, photo levelled
    for i, (lat, lon) in enumerate(p.points):
        add(16, 0, 1, 0, round(heading_at(p, i), 1), lat, lon, _height(p, i))   # NavigateToWaypoint
        if p.mode == "per_punto":
            add(2000, 0, 1, 12)                            # one photo
    if p.mode == "intervallo":
        add(2001)                                          # StopPhotoCapture
    add(20)                                                # ReturnToLaunch
    return "".join(rows)


# ------------------------------------------------------------------ Litchi

LITCHI_HEADER = (["latitude", "longitude", "altitude(m)", "heading(deg)", "curvesize(m)", "rotationdir",
                  "gimbalmode", "gimbalpitchangle"]
                 + [f"action{k}{i}" for i in range(1, 16) for k in ("type", "param")]
                 + ["altitudemode", "speed(m/s)", "poi_latitude", "poi_longitude", "poi_altitude(m)",
                    "poi_altitudemode", "photo_timeinterval", "photo_distinterval"])


def litchi_csv(p: Route) -> str:
    """The mission for Litchi's Mission Hub: one waypoint per row at its own height, straight lines (curve 0),
    camera interpolated to the pitch asked or pointed at the point of aim, a photo every so many metres or a "take
    photo" action at each point. Heights above the takeoff (altitudemode 0)."""
    v = round(p.effective_speed, 1)
    rows = [",".join(LITCHI_HEADER)]
    for i, (lat, lon) in enumerate(p.points):
        actions = ["-1", "0"] * 15
        if p.mode == "per_punto":
            actions[0], actions[1] = "1", "0"                # TAKE_PHOTO
        gimbal = "1" if p.poi else "2"                       # FOCUS_POI or INTERPOLATE
        poi = [f"{p.poi[0]:.7f}", f"{p.poi[1]:.7f}", "0", "0"] if p.poi else ["0", "0", "0", "0"]
        dist = f"{_step(p):.1f}" if p.mode == "intervallo" else "-1"
        rows.append(",".join([f"{lat:.7f}", f"{lon:.7f}", f"{p.height_at(i):.1f}", f"{heading_at(p, i):.1f}", "0",
                              "0", gimbal, f"{p.pitch:.0f}"] + actions + ["0", f"{v}"] + poi + ["-1", dist]))
    return "\n".join(rows) + "\n"


# ------------------------------------------------------------------ per leg, and in packages

def leg_file(p: Route, fmt: str, takeoff: Optional[LatLon] = None, name: str = "") -> bytes:
    """The file of one leg in the format asked."""
    if fmt == "plan":
        text = qgc_plan(p, takeoff, name)
    elif fmt == "waypoints":
        text = mission_planner_waypoints(p, takeoff)
    elif fmt == "mavlink":
        text = parrot_mavlink(p)
    elif fmt == "litchi":
        text = litchi_csv(p)
    else:
        raise ValueError(f"formato sconosciuto: {fmt}")
    return text.encode("utf-8")


def files_per_leg(piano: dict, fmt: str) -> list[tuple[str, bytes]]:
    """One file per leg of every active block, in the order of flight, in the format asked."""
    from .design import require_current_heights
    from .plan import plan_legs

    if fmt not in FORMATS:
        raise ValueError(f"formato sconosciuto: {fmt}")
    require_current_heights(piano)      # never the old heights, nor one height, of a block that changed
    extension = FORMATS[fmt][0]
    return [(f"{leg.file_name}.{extension}", leg_file(leg.route, fmt, leg.takeoff, leg.file_name))
            for leg in plan_legs(piano, specials=False)]     # the actions of the special points do not fit here


_HEIGHTS_NOTE = ("Quote: una per waypoint, sopra il punto di decollo della tratta (se il piano segue il terreno,\n"
                 "anche il file lo segue). Prova sempre il file nell'app prima di volare.\n"
                 "Heights: one per waypoint, above the takeoff point of the leg (if the plan follows the terrain,\n"
                 "so does the file). Always try the file in the app before flying.\n")

README_TEXT = {
    "plan": (
        'Piani per QGroundControl (ArduPilot e PX4): apri il .plan in QGroundControl o Mission Planner, '
        'controlla quota e\n'
        'camera, caricalo sul drone. Lo scatto a distanza usa MAV_CMD_DO_SET_CAM_TRIGG_DIST: la camera deve essere\n'
        "collegata all'autopilota.\n"
        'QGroundControl plans (ArduPilot and PX4): open the .plan in QGroundControl or Mission Planner, check\n'
        'altitude and camera, upload to the drone. Distance triggering uses MAV_CMD_DO_SET_CAM_TRIGG_DIST: the camera\n'
        'must be wired to the autopilot.\n'
        + _HEIGHTS_NOTE),
    "waypoints": (
        'Missioni per Mission Planner (ArduPilot): apri il .waypoints in Mission Planner (Plan > Load WP File),\n'
        'controlla quota e camera, scrivilo sul drone (Write WPs). La riga 0 è la posizione di casa al decollo.\n'
        "Lo scatto a distanza usa MAV_CMD_DO_SET_CAM_TRIGG_DIST: la camera deve essere collegata all'autopilota.\n"
        'Mission Planner missions (ArduPilot): open the .waypoints in Mission Planner (Plan > Load WP File),\n'
        'check altitude and camera, write it to the drone (Write WPs). Row 0 is the home position at the takeoff.\n'
        'Distance triggering uses MAV_CMD_DO_SET_CAM_TRIGG_DIST: the camera must be wired to the autopilot.\n'
        "Il file è per ArduPilot: con PX4 usa il .plan (qui l'imbardata «lasciala com'è» è scritta 0, cioè nord).\n"
        'The file is for ArduPilot: with PX4 use the .plan (here the yaw "leave it as it is" is written 0, north).\n'
        + _HEIGHTS_NOTE),
    "mavlink": (
        'Piani per Parrot Anafi / Anafi Ai: condividi il .mavlink con FreeFlight 7 (o caricalo con '
        'GroundSDK). Le foto\n'
        'sono in GPS-lapse, una ogni tot metri.\n'
        'Parrot Anafi / Anafi Ai plans: share the .mavlink with FreeFlight 7 (or upload it with GroundSDK).\n'
        'Photos are GPS-lapse, one every N metres.\n'
        + _HEIGHTS_NOTE),
    "litchi": (
        "Missioni per Litchi: importa il .csv nel Mission Hub (flylitchi.com/hub), salva, e la trovi "
        "nell'app. Vale per i\n"
        'DJI senza waypoint in DJI Fly (Mini 2, Air 2S, Mavic 2, Phantom 4...).\n'
        'Litchi missions: import the .csv in the Mission Hub (flylitchi.com/hub), save, and it shows up in the app.\n'
        'For the DJI models without waypoints in DJI Fly (Mini 2, Air 2S, Mavic 2, Phantom 4...).\n'
        + _HEIGHTS_NOTE),
}


def zip_format(piano: dict, fmt: str) -> bytes:
    """All the files of the plan in one format, and the README of that format."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files_per_leg(piano, fmt):
            z.writestr(name, data)
        z.writestr("LEGGIMI.txt", README_TEXT[fmt])
    return buffer.getvalue()


def terrain_note(piano: dict) -> Optional[str]:
    """What to tell before exporting a plan that follows the terrain to these formats (None for a flat plan): the
    heights go point by point above the takeoff, and a file a drone app has never read must be tried there first."""
    from .design import blocks_with_old_heights
    from .plan import active

    stale = set(blocks_with_old_heights(piano))
    if not any(b.get("quote") and active(b) and i not in stale for i, b in enumerate(piano.get("blocchi", []))):
        return None
    return t("quote punto per punto sopra il decollo, seguendo il terreno: prova il file nell'app prima di volare",
             "heights point by point above the takeoff, following the terrain: try the file in the app before flying")
