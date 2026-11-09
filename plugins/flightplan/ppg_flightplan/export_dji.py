"""The DJI files of a flight: the KMZ (WPML) as DJI Fly writes it for the Mini 4 Pro, one per leg, the archive
of a whole plan, and the readers that bring a route back from a KMZ, a KML or a GeoJSON.

Heights are relative to the takeoff point, and the files say so: ``heightMode`` / ``executeHeightMode``
``relativeToStartPoint`` with ``coordinateMode`` ``WGS84``, one ``executeHeight`` per waypoint (the per-point
heights of a flight following the terrain go into the file as they are: that is how the GSD stays constant
uphill). The KMZ keeps the elements and values the planner writes, tried on the Mini 4 Pro.

Ported from the author's Photo3D planner: elements, values, rounding and the order of operations unchanged.
What the plugin changed: the author written in the KMZ (:data:`AUTHOR`), the name of the data the KML carries
(``ppg:terreno``, ``ppg:soste``; the readers still accept the planner's ``photo3d:...``) and, in the KML
written by :mod:`ppg_flightplan.gis`, absolute heights declared as such (the takeoff height is in the file,
the reader takes it off again).
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET

from .camera import Camera
from .geometry import LatLon, centroid, distance, simplify
from .route import Route, Stop, number_of
from .specials import num

NS_KML = "http://www.opengis.net/kml/2.2"
NS_WPML = "http://www.dji.com/wpmz/1.0.2"
AUTHOR = "pyarchinit-photogrammetry"

# The names under which the plan's terrain grid and the special points of a route travel in the KML: Google
# Earth and QGIS ignore them. The planner's names are still read.
DATA_TERRAIN = "ppg:terreno"
DATA_STOPS = "ppg:soste"
PLANNER_DATA_TERRAIN = "photo3d:terreno"
PLANNER_DATA_STOPS = "photo3d:soste"
# The KML of the plugin writes absolute heights (takeoff height + height above the takeoff): these say which
# takeoff height was added and in which height system, so that the reader takes it off again.
DATA_TAKEOFF_HEIGHT = "ppg:quotaDecollo"
DATA_HEIGHT_REFERENCE = "ppg:riferimentoQuote"


def _xml(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


# ------------------------------------------------------------------ KMZ (DJI WPML)

def _mission_config(cam: Camera, transfer_speed: float = 10.0) -> str:
    lines = ["    <wpml:missionConfig>",
             "      <wpml:flyToWaylineMode>safely</wpml:flyToWaylineMode>",
             "      <wpml:finishAction>goHome</wpml:finishAction>",
             "      <wpml:exitOnRCLost>executeLostAction</wpml:exitOnRCLost>",
             "      <wpml:executeRCLostAction>goBack</wpml:executeRCLostAction>"]
    if cam.enterprise:
        lines.append("      <wpml:takeOffSecurityHeight>20</wpml:takeOffSecurityHeight>")
    lines += [f"      <wpml:globalTransitionalSpeed>{num(transfer_speed)}</wpml:globalTransitionalSpeed>",
              "      <wpml:droneInfo>",
              f"        <wpml:droneEnumValue>{cam.drone_enum}</wpml:droneEnumValue>",
              f"        <wpml:droneSubEnumValue>{cam.drone_sub_enum}</wpml:droneSubEnumValue>",
              "      </wpml:droneInfo>"]
    if cam.enterprise and cam.payload is not None:
        lines += ["      <wpml:payloadInfo>",
                  f"        <wpml:payloadEnumValue>{cam.payload}</wpml:payloadEnumValue>",
                  "        <wpml:payloadSubEnumValue>0</wpml:payloadSubEnumValue>",
                  "        <wpml:payloadPositionIndex>0</wpml:payloadPositionIndex>",
                  "      </wpml:payloadInfo>"]
    lines.append("    </wpml:missionConfig>")
    return "\n".join(lines) + "\n"


def _turn(p: Route, cam: Camera) -> tuple[bool, str, int]:
    """At the turns the drone stops (interval shots on straight strips) or passes curving. DJI Fly always
    writes useStraightLine 0 and draws crooked whoever writes 1; the Enterprise are asked for the straight line."""
    stops = p.mode == "intervallo" and p.poi is None
    mode = "toPointAndStopWithDiscontinuityCurvature" if stops else "toPointAndPassWithContinuityCurvature"
    return stops, mode, 1 if (stops and cam.enterprise) else 0


def _nose(p: Route) -> tuple[str, str]:
    """Where the nose points: along the route, or towards the point of aim."""
    poi = f"{p.poi[0]:.8f},{p.poi[1]:.8f},0.000000" if p.poi else "0.000000,0.000000,0.000000"
    return ("towardPOI" if p.poi else "followWayline"), poi


def _header() -> str:
    return f'<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="{NS_KML}" xmlns:wpml="{NS_WPML}">\n  <Document>\n'


def template_kml(route: Route, cam: Camera, now_ms: int, author: str = AUTHOR) -> str:
    """The template of the mission: configuration, the folder of the "waypoint" model and the same points as
    waylines.wpml. DJI Fly flies from waylines.wpml but rebuilds the mission from the template when it saves it
    again on the remote controller or syncs it with the cloud: with a template without points the mission saved
    again came out scrambled. The Enterprise also want the global parameters the WPML specification puts in
    the template."""
    p = route
    _, turn_mode, straight = _turn(p, cam)
    heading, poi = _nose(p)
    globals_ = ""
    if cam.enterprise:
        globals_ = (f"      <wpml:globalWaypointTurnMode>{turn_mode}</wpml:globalWaypointTurnMode>\n"
                    f"      <wpml:globalUseStraightLine>{straight}</wpml:globalUseStraightLine>\n"
                    "      <wpml:globalWaypointHeadingParam>\n"
                    f"        <wpml:waypointHeadingMode>{heading}</wpml:waypointHeadingMode>\n"
                    "        <wpml:waypointHeadingAngle>0</wpml:waypointHeadingAngle>\n"
                    f"        <wpml:waypointPoiPoint>{poi}</wpml:waypointPoiPoint>\n"
                    "        <wpml:waypointHeadingPoiIndex>0</wpml:waypointHeadingPoiIndex>\n"
                    "      </wpml:globalWaypointHeadingParam>\n")
    return (_header()
            + f"    <wpml:author>{_xml(author)}</wpml:author>\n"
            f"    <wpml:createTime>{now_ms}</wpml:createTime>\n"
            f"    <wpml:updateTime>{now_ms}</wpml:updateTime>\n"
            + _mission_config(cam)
            + "    <Folder>\n"
            "      <wpml:templateType>waypoint</wpml:templateType>\n"
            "      <wpml:templateId>0</wpml:templateId>\n"
            "      <wpml:waylineCoordinateSysParam>\n"
            "        <wpml:coordinateMode>WGS84</wpml:coordinateMode>\n"
            "        <wpml:heightMode>relativeToStartPoint</wpml:heightMode>\n"
            "        <wpml:positioningType>GPS</wpml:positioningType>\n"
            "      </wpml:waylineCoordinateSysParam>\n"
            f"      <wpml:autoFlightSpeed>{num(p.effective_speed)}</wpml:autoFlightSpeed>\n"
            f"      <wpml:globalHeight>{num(p.height)}</wpml:globalHeight>\n"
            "      <wpml:caliFlightEnable>0</wpml:caliFlightEnable>\n"
            "      <wpml:gimbalPitchMode>usePointSetting</wpml:gimbalPitchMode>\n"
            + globals_ + _placemarks(p, cam, template=True)
            + "    </Folder>\n  </Document>\n</kml>\n")


def _group(id_: int, index: int, mode: str, action: str) -> str:
    return ("        <wpml:actionGroup>\n"
            f"          <wpml:actionGroupId>{id_}</wpml:actionGroupId>\n"
            f"          <wpml:actionGroupStartIndex>{index}</wpml:actionGroupStartIndex>\n"
            f"          <wpml:actionGroupEndIndex>{index}</wpml:actionGroupEndIndex>\n"
            f"          <wpml:actionGroupMode>{mode}</wpml:actionGroupMode>\n"
            "          <wpml:actionTrigger>\n"
            "            <wpml:actionTriggerType>reachPoint</wpml:actionTriggerType>\n"
            "          </wpml:actionTrigger>\n" + action + "        </wpml:actionGroup>\n")


def _action(id_: int, function: str, parameters: str) -> str:
    return ("            <wpml:action>\n"
            f"              <wpml:actionId>{id_}</wpml:actionId>\n"
            f"              <wpml:actionActuatorFunc>{function}</wpml:actionActuatorFunc>\n"
            "              <wpml:actionActuatorFuncParam>\n" + parameters +
            "              </wpml:actionActuatorFuncParam>\n"
            "            </wpml:action>\n")


def _camera_action(id_: int, degrees: float) -> str:
    return _action(id_, "gimbalRotate",
                   "                <wpml:gimbalHeadingYawBase>aircraft</wpml:gimbalHeadingYawBase>\n"
                   "                <wpml:gimbalRotateMode>absoluteAngle</wpml:gimbalRotateMode>\n"
                   "                <wpml:gimbalPitchRotateEnable>1</wpml:gimbalPitchRotateEnable>\n"
                   f"                <wpml:gimbalPitchRotateAngle>{num(degrees)}</wpml:gimbalPitchRotateAngle>\n"
                   "                <wpml:gimbalRollRotateEnable>0</wpml:gimbalRollRotateEnable>\n"
                   "                <wpml:gimbalRollRotateAngle>0</wpml:gimbalRollRotateAngle>\n"
                   "                <wpml:gimbalYawRotateEnable>0</wpml:gimbalYawRotateEnable>\n"
                   "                <wpml:gimbalYawRotateAngle>0</wpml:gimbalYawRotateAngle>\n"
                   "                <wpml:gimbalRotateTimeEnable>0</wpml:gimbalRotateTimeEnable>\n"
                   "                <wpml:gimbalRotateTime>0</wpml:gimbalRotateTime>\n"
                   "                <wpml:payloadPositionIndex>0</wpml:payloadPositionIndex>\n")


def _stop_action(id_: int, action: str, extra: str) -> str:
    """An action of a special point in the WPML form; an empty string if it is not known."""
    payload = "                <wpml:payloadPositionIndex>0</wpml:payloadPositionIndex>\n"
    if action == "foto":
        return _action(id_, "takePhoto", payload + extra)
    if action == "video_inizio":
        return _action(id_, "startRecord", payload + extra)
    if action == "video_fine":
        return _action(id_, "stopRecord", payload)
    if action.startswith("camera:"):
        return _camera_action(id_, number_of(action, -90.0))
    if action.startswith("pausa:"):
        return _action(id_, "hover",
                       f"                <wpml:hoverTime>{num(max(1.0, number_of(action, 2.0)))}</wpml:hoverTime>\n")
    return ""


def _actions_to_write(stop: Stop, p: Route) -> list[str]:
    """The actions of a stop as they go into the file, which depends on how the photos are taken.

    Shooting **by point** the photo is an action of the file, as on every other waypoint. Shooting **by time**
    it is not: a "takePhoto" with the camera on interval would start a burst, and to avoid it the camera would
    have to change mode at the right instant, which the phone does not know with that precision. So the photo
    is taken by the timer itself: the drone stops on the point a generous interval, and at least one shot falls
    there. It also holds with the screen remote controller, where the pilot starts the timer.
    """
    interval = p.interval
    if not interval or "foto" not in stop.actions:
        return list(stop.actions)
    wait = max([interval + 1.0] + [number_of(a, 0.0) for a in stop.actions if a.startswith("pausa:")])
    others = [a for a in stop.actions if a != "foto" and not a.startswith("pausa:")]
    return others + [f"pausa:{num(wait)}"]


def _placemarks(p: Route, cam: Camera, template: bool = False) -> str:
    """The Placemarks of the waypoints, the same in the two files of the KMZ; in the template the Enterprise
    also want the height in the form of the specification (height, ellipsoidHeight)."""
    v = p.effective_speed
    group_mode = "sequence" if cam.enterprise else "parallel"
    _, turn_mode, straight = _turn(p, cam)
    heading, poi = _nose(p)
    out: list[str] = []
    group = action = 0
    suffix = re.sub(r"[^A-Za-z0-9]", "", p.name)[:8] or "PPG"
    extra = ("" if not cam.enterprise else
             f"                <wpml:fileSuffix>{_xml(suffix)}</wpml:fileSuffix>\n"
             "                <wpml:useGlobalPayloadLensIndex>1</wpml:useGlobalPayloadLensIndex>\n")
    in_video = p.in_video
    for i, (lat, lon) in enumerate(p.points):
        stop = p.stop(i)
        # a special point may want the nose fixed (the photos of a panorama), to stop or not, and to go slowly:
        # all the rest of the route stays as it was
        v_point = stop.speed if stop and stop.speed else v
        if stop and stop.direction is not None:
            heading_point, angle, angle_on = "smoothTransition", num(round(stop.direction, 1)), 1
        else:
            heading_point, angle, angle_on = heading, "0", 0
        if stop and stop.stops is not None:
            turn_point = ("toPointAndStopWithDiscontinuityCurvature" if stop.stops
                          else "toPointAndPassWithContinuityCurvature")
        else:
            turn_point = turn_mode
        camera_point = stop.pitch if stop and stop.pitch is not None else p.pitch
        out.append("      <Placemark>\n        <Point>\n          <coordinates>\n"
                   f"            {lon:.8f},{lat:.8f}\n          </coordinates>\n        </Point>\n"
                   f"        <wpml:index>{i}</wpml:index>\n")
        if template and cam.enterprise:
            # the height of THIS point (relative to the takeoff, as the template's heightMode says): Pilot 2 and
            # FlightHub 2 rebuild the route from the template, and one height per block there would fly a plan
            # following the terrain at one height. Only the Enterprise template has it: the DJI Fly KMZ of the
            # Mini 4 Pro, the one proven in the field, stays as it was.
            out.append(f"        <wpml:ellipsoidHeight>{num(p.height_at(i))}</wpml:ellipsoidHeight>\n"
                       f"        <wpml:height>{num(p.height_at(i))}</wpml:height>\n"
                       "        <wpml:useGlobalHeight>0</wpml:useGlobalHeight>\n"
                       "        <wpml:useGlobalSpeed>0</wpml:useGlobalSpeed>\n"
                       "        <wpml:useGlobalHeadingParam>0</wpml:useGlobalHeadingParam>\n"
                       "        <wpml:useGlobalTurnParam>0</wpml:useGlobalTurnParam>\n")
        out.append(f"        <wpml:executeHeight>{num(p.height_at(i))}</wpml:executeHeight>\n"
                   f"        <wpml:waypointSpeed>{num(v_point)}</wpml:waypointSpeed>\n"
                   "        <wpml:waypointHeadingParam>\n"
                   f"          <wpml:waypointHeadingMode>{heading_point}</wpml:waypointHeadingMode>\n"
                   f"          <wpml:waypointHeadingAngle>{angle}</wpml:waypointHeadingAngle>\n"
                   f"          <wpml:waypointPoiPoint>{poi}</wpml:waypointPoiPoint>\n"
                   f"          <wpml:waypointHeadingAngleEnable>{angle_on}</wpml:waypointHeadingAngleEnable>\n"
                   "          <wpml:waypointHeadingPathMode>followBadArc</wpml:waypointHeadingPathMode>\n"
                   "          <wpml:waypointHeadingPoiIndex>0</wpml:waypointHeadingPoiIndex>\n"
                   "        </wpml:waypointHeadingParam>\n"
                   "        <wpml:waypointTurnParam>\n"
                   f"          <wpml:waypointTurnMode>{turn_point}</wpml:waypointTurnMode>\n"
                   "          <wpml:waypointTurnDampingDist>0</wpml:waypointTurnDampingDist>\n"
                   "        </wpml:waypointTurnParam>\n"
                   f"        <wpml:useStraightLine>{straight}</wpml:useStraightLine>\n")
        if i == 0:
            group += 1
            action += 1
            out.append(_group(group, i, group_mode, _camera_action(action, p.pitch)))
        if p.mode == "per_punto" and p.shoots_by_itself(i, in_video):
            group += 1
            action += 1
            out.append(_group(group, i, group_mode, _stop_action(action, "foto", extra)))
        if stop and stop.actions:
            # the actions of the special point are in one group only, in the order they are written
            written = []
            for a in _actions_to_write(stop, p):
                text = _stop_action(action + 1, a, extra)
                if text:
                    action += 1
                    written.append(text)
            if written:
                group += 1
                out.append(_group(group, i, group_mode, "".join(written)))
        out.append("        <wpml:waypointGimbalHeadingParam>\n"
                   f"          <wpml:waypointGimbalPitchAngle>{num(camera_point)}</wpml:waypointGimbalPitchAngle>\n"
                   "          <wpml:waypointGimbalYawAngle>0</wpml:waypointGimbalYawAngle>\n"
                   "        </wpml:waypointGimbalHeadingParam>\n      </Placemark>\n")
    return "".join(out)


def waylines_wpml(route: Route, cam: Camera) -> str:
    """The points to fly, as DJI Fly writes them."""
    p = route
    return (_header() + _mission_config(cam)
            + "    <Folder>\n      <wpml:templateId>0</wpml:templateId>\n"
            "      <wpml:executeHeightMode>relativeToStartPoint</wpml:executeHeightMode>\n"
            "      <wpml:waylineId>0</wpml:waylineId>\n"
            f"      <wpml:distance>{num(p.length)}</wpml:distance>\n"
            f"      <wpml:duration>{num(p.minutes() * 60)}</wpml:duration>\n"
            f"      <wpml:autoFlightSpeed>{num(p.effective_speed)}</wpml:autoFlightSpeed>\n"
            + _placemarks(p, cam) + "    </Folder>\n  </Document>\n</kml>\n")


def kmz(route: Route, cam: Camera, now_ms: Optional[int] = None) -> bytes:
    """The DJI KMZ: template.kml with configuration and points, waylines.wpml with the points to fly, as DJI
    Fly writes them."""
    now_ms = now_ms if now_ms is not None else int(datetime.now(timezone.utc).timestamp() * 1000)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("wpmz/template.kml", template_kml(route, cam, now_ms))
        z.writestr("wpmz/waylines.wpml", waylines_wpml(route, cam))
    return buffer.getvalue()


def kmz_per_leg(piano: dict) -> list[tuple[str, bytes]]:
    """A KMZ for every leg of every active block, in the order of flight: file name and bytes."""
    from .design import require_current_heights, require_safe_heights
    from .plan import plan_camera, plan_legs

    require_current_heights(piano)      # a block changed after its heights: never fly the old ones, nor one height
    require_safe_heights(piano)         # nor heights cut at the ceiling, nor a leg takeoff kilometres away
    cam = plan_camera(piano)
    return [(f"{leg.file_name}.kmz", kmz(leg.route, cam)) for leg in plan_legs(piano)]


def zip_kmz(piano: dict, ground_takeoff: Optional[float] = None, height_source: Optional[dict] = None) -> bytes:
    """An archive with a KMZ for every leg of every block, plus the KML of the whole plan (its takeoff height as
    :func:`ppg_flightplan.gis.plan_kml` takes it)."""
    from .gis import plan_kml
    from .plan import slug

    base = slug(piano.get("nome", "piano"))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for file_name, data in kmz_per_leg(piano):
            z.writestr(file_name, data)
        z.writestr(f"{base}.kml", plan_kml(piano, ground_takeoff=ground_takeoff, height_source=height_source))
    return buffer.getvalue()


# ------------------------------------------------------------------ the description line of the KML

# The KML written by the plugin says it in words when the flight follows the terrain: it is what tells "all
# heights equal because the ground is flat" from "a single height because the flight is flat".
TERRAIN_SIGN = "segue il terreno"


def route_description(p: Route) -> str:
    """The line the readers read back: height (above the takeoff), camera, shots, minutes, speed, interval. The
    words are the planner's and stay Italian whatever the interface: :func:`read_kml` looks for them."""
    tail = f", scatto ogni {num(p.interval)} s" if p.interval else ", uno scatto per punto"
    if p.follows_terrain:
        tail += f", {TERRAIN_SIGN} da {num(min(p.heights))} a {num(max(p.heights))} m"
    count = len({s_.special for s_ in p.stops or [] if s_})
    if count:
        tail += f", {count} punti speciali"
    if p.poi:
        tail += f", mira {p.poi[0]:.7f},{p.poi[1]:.7f}"
    return (f"quota {num(p.height)} m, camera {num(p.pitch)} gradi, {p.shots()} scatti, "
            f"{p.minutes():.0f} min, velocita {num(p.effective_speed)} m/s{tail}")


# ------------------------------------------------------------------ reading from files

@dataclass
class Imported:
    routes: list[Route]
    areas: list[list[LatLon]]


def _number_after(text: str, word: str, before: bool = False) -> Optional[float]:
    r = (re.compile(rf"(-?\d+(?:[.,]\d+)?)\s*{word}", re.I) if before
         else re.compile(rf"{word}\w*\s*:?\s*(-?\d+(?:[.,]\d+)?)", re.I))
    m = r.search(text)
    return float(m.group(1).replace(",", ".")) if m else None


def _coordinates(text: str) -> list[tuple[LatLon, Optional[float]]]:
    read = []
    for triple in text.split():
        c = triple.split(",")
        try:
            lon, lat = float(c[0]), float(c[1])
        except (ValueError, IndexError):
            continue
        height = None
        if len(c) > 2:
            try:
                height = float(c[2])
            except ValueError:
                height = None
        read.append(((lat, lon), height))
    return read


def _without_closure(ring_: list[LatLon]) -> list[LatLon]:
    return ring_[:-1] if len(ring_) > 3 and distance(ring_[0], ring_[-1]) < 0.5 else ring_


def _aim(text: str) -> Optional[LatLon]:
    """"mira 45.0000000,10.0000000" in the description: where the camera looks."""
    m = re.search(r"mira\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)", text, re.I)
    return (float(m.group(1)), float(m.group(2))) if m else None


def _heights_per_point(heights: list[Optional[float]], points: list[LatLon],
                       declared: bool = False) -> Optional[list[float]]:
    """The heights read point by point: they hold if they are all there, and if the file says the flight
    follows the terrain or if they change enough not to be a single height.

    Without the "declared", a flight following the terrain over a plain came back as a flight at a fixed
    height - true in the numbers, but on the screen it said the wrong thing.
    """
    if len(heights) != len(points) or any(q is None or q <= 0.5 for q in heights):
        return None
    values = [float(q) for q in heights]
    return values if declared or max(values) - min(values) > 2.0 else None


def _build(name: str, points: list[LatLon], height: float, pitch: Optional[float], speed: Optional[float],
           shots: Optional[int], cam: Camera, front: float, high: bool, aim: Optional[LatLon] = None,
           heights: Optional[list[float]] = None, stops: Optional[list[Optional[Stop]]] = None,
           flat_heights: bool = False) -> Route:
    # a closed track is a ring, even drawn by hand with few vertices: it looks at the centre
    closed = len(points) >= 4 and distance(points[0], points[-1]) < 3.0 and not stops
    ring_ = points[:-1] if closed else points
    # with a height per point the route is not simplified: removing a point would remove its height (and with
    # the special points it would remove the waypoints that make them)
    kept = points if heights or stops else simplify(points, 1.0)
    return Route(name=name, points=kept, height=height, heights=heights, stops=stops, flat_heights=flat_heights,
                 pitch=pitch if pitch is not None else -90.0,
                 speed=speed if speed is not None else 5.0,
                 photo_step=cam.footprint_short(height) * (1 - front), mode="intervallo",
                 poi=aim or (centroid(ring_) if closed else None),
                 min_interval=cam.min_interval_for(high), declared_shots=shots)


def _local(el: ET.Element) -> str:
    return el.tag.rsplit("}", 1)[-1]


def _children(el: ET.Element, name: str) -> list[ET.Element]:
    return [e for e in el.iter() if _local(e) == name]


def _text(el: ET.Element, name: str) -> Optional[str]:
    for e in el.iter():
        if _local(e) == name and e.text:
            return e.text.strip()
    return None


def _stops_read(pm: ET.Element, count: int) -> Optional[list[Optional[Stop]]]:
    """The special points the route carries in the KML; nothing if they do not match the points."""
    for data in _children(pm, "Data"):
        if data.get("name") not in (DATA_STOPS, PLANNER_DATA_STOPS):
            continue
        try:
            entries = json.loads(_text(data, "value") or "[]")
        except ValueError:
            return None
        stops: list[Optional[Stop]] = [None] * count
        for v in entries if isinstance(entries, list) else []:
            i = v.get("i") if isinstance(v, dict) else None
            if not isinstance(i, int) or not 0 <= i < count:
                return None                     # the file is of another route: better nothing than at random
            stops[i] = Stop.from_json(v)
        return stops if any(stops) else None
    return None


def _takeoff_height_written(root: ET.Element) -> float:
    """The takeoff height the plugin added to the heights of its KML (absolute heights); 0 for any other KML."""
    for data in _children(root, "Data"):
        if data.get("name") == DATA_TAKEOFF_HEIGHT:
            try:
                return float(_text(data, "value") or "0")
            except ValueError:
                return 0.0
    return 0.0


def read_kml(name: str, text: str, cam: Camera, front: float = 0.8, high: bool = False,
             height_if_missing: float = 50.0) -> Imported:
    root = ET.fromstring(text)
    added = _takeoff_height_written(root)
    routes, areas = [], []
    for k, pm in enumerate(_children(root, "Placemark")):
        entry_name = _text(pm, "name") or f"{name} {k + 1}"
        description = _text(pm, "description") or ""
        for line in _children(pm, "LineString"):
            coords = _text(line, "coordinates")
            read = _coordinates(coords) if coords else []
            if len(read) < 2:
                continue
            if added:
                # the plugin's absolute heights: the takeoff height comes off, back to heights above the takeoff
                read = [(p, None if q is None else round(q - added, 3)) for p, q in read]
            points_read = [p for p, _ in read]
            declared = TERRAIN_SIGN in description.lower()
            per_point = _heights_per_point([q for _, q in read], points_read, declared)
            stops = _stops_read(pm, len(points_read))
            heights = sorted(q for _, q in read if q is not None and q > 0.5)
            height = _number_after(description, "quota") or (heights[len(heights) // 2] if heights
                                                             else height_if_missing)
            shots = _number_after(description, "scatti", before=True)
            routes.append(_build(entry_name, points_read, height,
                                 _number_after(description, "camera"), _number_after(description, "velocit"),
                                 int(round(shots)) if shots else None, cam, front, high,
                                 aim=_aim(description), heights=per_point, stops=stops,
                                 flat_heights=per_point is not None and stops is not None and not declared))
        for polygon in _children(pm, "Polygon"):
            outer = (_children(polygon, "outerBoundaryIs") or [polygon])[0]
            coords = _text(outer, "coordinates")
            points = [p for p, _ in _coordinates(coords)] if coords else []
            if len(points) >= 3:
                areas.append(_without_closure(points))
    return Imported(routes, areas)


def _most_common(values: list):
    """The value of the route: the one almost every waypoint has."""
    known = [v for v in values if v is not None]
    return max(set(known), key=known.count) if known else None


def _stops_from_wpml(placemarks: list[dict], points: list[LatLon], heights: list[float],
                     with_shot: bool) -> tuple[Optional[list[Optional[Stop]]], bool]:
    """Rebuilds the special points from a DJI KMZ, and says whether the per-point heights are "flat".

    A KMZ written with special points, read again, lost the actions but kept the low waypoints: nobody knew
    any more that one does not start or resume from there, and the heights looked like a flight following the
    terrain. The file has all that is needed: the actions, the fixed nose, the turn, the speed and the camera
    of every waypoint. What cannot be known (the name of the point, whether the height was "from the ground")
    stays empty.
    """
    n = len(points)
    if n < 2 or len(placemarks) != n:
        return None, False

    # The values "of the route" are read on any waypoint: those without actions of their own and without a
    # fixed nose. Counting them on all of them goes wrong as soon as there is a video ring, which alone puts
    # thirteen.
    def ordinary(m: dict) -> bool:
        return m["direction"] is None and not m["actions"]
    normal = [m for m in placemarks[1:] if ordinary(m)] or placemarks
    route_speed = _most_common([m["speed"] for m in normal])
    # the camera of the route is the one set at the first waypoint, which the file always writes
    first = next((v for f, v in placemarks[0]["actions"] if f == "gimbalRotate" and v is not None), None)
    route_camera = first if first is not None else _most_common([m["camera"] for m in normal])
    stops: list[Optional[Stop]] = [None] * n

    def stop_of(m: dict) -> Stop:
        """The stop of a waypoint, from what the file says of it."""
        return Stop(actions=m.get("read", ()), direction=m["direction"], survey=bool(m.get("surveyPhoto")),
                    stops=None if m["turn"] is None else m["turn"] == "toPointAndStopWithDiscontinuityCurvature",
                    speed=m["speed"] if m["speed"] != route_speed else None,
                    pitch=m["camera"] if m["camera"] is not None and route_camera is not None
                    and abs(m["camera"] - route_camera) > 0.5 else None)

    for i, m in enumerate(placemarks):
        actions = []
        for function, value in m["actions"]:
            if function == "gimbalRotate":
                if i == 0 and not actions:
                    continue                          # the camera set at the first point: not a special point
                actions.append(f"camera:{num(value if value is not None else -90.0)}")
            elif function == "hover":
                # shooting by time the photo of a special point is written as a stop only
                actions += ["pausa:2", "foto"] if not with_shot else [f"pausa:{num(value or 2.0)}"]
            elif function == "takePhoto":
                actions.append("foto")
            elif function == "startRecord":
                actions.append("video_inizio")
            elif function == "stopRecord":
                actions.append("video_fine")
        if len(actions) > 1:
            actions = list(dict.fromkeys(actions))       # "pause, photo" and then "photo": once only
        m["read"] = tuple(actions)
        if actions or m["direction"] is not None:
            stops[i] = stop_of(m)
    # The low waypoints: lower than the last waypoint at cruise height, and either already special or two
    # steps from it (a descent). Two ordinary waypoints far apart at different heights are a flight following
    # the terrain, not a descent.
    high_i = 0
    for i in range(1, n):
        down = heights[i] <= heights[high_i] - 2.0
        if down and (stops[i] is not None or distance(points[i], points[high_i]) <= 3.0):
            stops[i] = replace(stops[i] or stop_of(placemarks[i]), low=True)
        else:
            high_i = i
    if not any(stops):
        return None, False
    # every group of special waypoints starts from the ordinary waypoint that precedes it closely: it is the one
    # at cruise height from which a restart can begin
    i = 0
    while i < n:
        if stops[i] is None:
            i += 1
            continue
        head = i
        if i > 0 and stops[i - 1] is None and distance(points[i - 1], points[i]) <= 3.0:
            head = i - 1
            stops[head] = Stop(stops=True, survey=True)
        stops[head] = replace(stops[head], head=True)
        while i < n and stops[i] is not None:
            i += 1
    of_survey = [heights[k] for k in range(n) if stops[k] is None]
    flat = bool(of_survey) and max(of_survey) - min(of_survey) <= 2.0
    return stops, flat


def read_wpml(name: str, text: str, cam: Camera, front: float = 0.8, high: bool = False) -> Imported:
    root = ET.fromstring(text)
    points, heights = [], []
    placemarks: list[dict] = []
    speed = _text(root, "autoFlightSpeed")
    poi = pitch = None
    with_shot = False
    for pm in _children(root, "Placemark"):
        coords = _text(pm, "coordinates")
        read = _coordinates(coords) if coords else []
        if not read:
            continue
        points.append(read[0][0])
        q = _text(pm, "executeHeight")
        if q:
            heights.append(float(q))
        if speed is None:
            speed = _text(pm, "waypointSpeed")
        if poi is None and _text(pm, "waypointHeadingMode") == "towardPOI":
            c = (_text(pm, "waypointPoiPoint") or "").split(",")
            try:
                lat, lon = float(c[0]), float(c[1])
                if abs(lat) > 1e-6 or abs(lon) > 1e-6:
                    poi = (lat, lon)
            except (ValueError, IndexError):
                pass
        if pitch is None:
            t_ = _text(pm, "gimbalPitchRotateAngle") or _text(pm, "waypointGimbalPitchAngle")
            pitch = float(t_) if t_ else None
        if any(e.text and e.text.strip() == "takePhoto" for e in _children(pm, "actionActuatorFunc")):
            with_shot = True

        def number(label: str, inside=pm) -> Optional[float]:
            t_ = _text(inside, label)
            try:
                return float(t_) if t_ else None
            except ValueError:
                return None
        actions, survey_photo = [], False
        for group in _children(pm, "actionGroup"):
            of_group = []
            for action in _children(group, "action"):
                function = _text(action, "actionActuatorFunc") or ""
                of_group.append((function, number("gimbalPitchRotateAngle", action) if function == "gimbalRotate"
                                 else number("hoverTime", action) if function == "hover" else None))
            if [f_ for f_, _ in of_group] == ["takePhoto"]:
                survey_photo = True          # a group with the photo only: the survey's, shooting by point
            else:
                actions += of_group
        placemarks.append({"actions": actions, "surveyPhoto": survey_photo, "turn": _text(pm, "waypointTurnMode"),
                           "speed": number("waypointSpeed"), "camera": number("waypointGimbalPitchAngle"),
                           "direction": number("waypointHeadingAngle")
                           if _text(pm, "waypointHeadingAngleEnable") == "1" else None})
    if len(points) < 2:
        return Imported([], [])
    stops, flat = (_stops_from_wpml(placemarks, points, heights, with_shot) if len(heights) == len(points)
                   else (None, False))
    per_point = _heights_per_point(heights, points)
    if stops:
        # the height of the flight is that of the ordinary waypoints, not the median the descents pull down
        ordinary = sorted(q for q, s_ in zip(heights, stops) if s_ is None) or sorted(heights)
        height = ordinary[len(ordinary) // 2]
        # the header says the route's speed
        speed = str(speed or _most_common([m["speed"] for m in placemarks]) or 5.0)
        return Imported([Route(name=name, points=points, height=height, heights=per_point, stops=stops,
                               flat_heights=flat and per_point is not None,
                               pitch=pitch if pitch is not None else -90.0,
                               speed=float(speed), photo_step=cam.footprint_short(height) * (1 - front),
                               mode="per_punto" if with_shot else "intervallo", poi=poi,
                               min_interval=cam.min_interval_for(high))], [])
    height = sorted(heights)[len(heights) // 2] if heights else 50.0
    return Imported([Route(name=name, points=points, height=height, heights=per_point,
                           pitch=pitch if pitch is not None else -90.0,
                           speed=float(speed) if speed else 5.0,
                           photo_step=cam.footprint_short(height) * (1 - front),
                           mode="per_punto" if with_shot else "intervallo", poi=poi,
                           min_interval=cam.min_interval_for(high))], [])


def read_geojson(name: str, text: str, cam: Camera, front: float = 0.8, high: bool = False,
                 height_if_missing: float = 50.0) -> Imported:
    root = json.loads(text)
    kind = root.get("type")
    if kind == "FeatureCollection":
        geometries = [(f.get("geometry") or {}, f.get("properties") or {}) for f in root.get("features", [])]
    elif kind == "Feature":
        geometries = [(root.get("geometry") or {}, root.get("properties") or {})]
    else:
        geometries = [(root, {})]
    routes, areas = [], []

    def number(prop: dict, *keys: str) -> Optional[float]:
        for c in keys:
            v = prop.get(c)
            if isinstance(v, (int, float)):
                return float(v)
        return None

    def points_of(coords: list) -> list[tuple[LatLon, Optional[float]]]:
        return [((c[1], c[0]), (float(c[2]) if len(c) > 2 else None)) for c in coords if len(c) >= 2]

    for k, (g, prop) in enumerate(geometries):
        entry_name = next((prop[c] for c in ("blocco", "name", "nome", "Name") if prop.get(c)), f"{name} {k + 1}")
        height_prop = number(prop, "quota_m", "quota", "altitude", "height")
        pitch = number(prop, "camera_gradi", "gimbal", "inclinazione", "pitch")
        speed = number(prop, "velocita_ms", "velocita", "speed")
        shots = number(prop, "scatti", "photos")
        aim_prop = prop.get("mira")
        aim = (float(aim_prop[0]), float(aim_prop[1])) if isinstance(aim_prop, list) and len(aim_prop) == 2 else None

        def line(coords: list, suffix: str = "") -> None:
            read = points_of(coords)
            if len(read) < 2:
                return
            heights = sorted(q for _, q in read if q is not None and q > 0.5)
            height = height_prop or (heights[len(heights) // 2] if heights else height_if_missing)
            routes.append(_build(str(entry_name) + suffix, [p for p, _ in read], height, pitch, speed,
                                 int(round(shots)) if shots else None, cam, front, high, aim=aim))

        t_ = g.get("type")
        if t_ == "LineString":
            line(g.get("coordinates", []))
        elif t_ == "MultiLineString":
            parts = g.get("coordinates", [])
            for i, part in enumerate(parts):
                line(part, f" {i + 1}" if len(parts) > 1 else "")
        elif t_ == "Polygon":
            rings = g.get("coordinates", [])
            if rings:
                points = [p for p, _ in points_of(rings[0])]
                if len(points) >= 3:
                    areas.append(_without_closure(points))
        elif t_ == "MultiPolygon":
            for polygon in g.get("coordinates", []):
                if polygon:
                    points = [p for p, _ in points_of(polygon[0])]
                    if len(points) >= 3:
                        areas.append(_without_closure(points))
    return Imported(routes, areas)


def read_file(file_name: str, data: bytes, cam: Camera, front: float = 0.8, high: bool = False,
              height_if_missing: float = 50.0) -> Imported:
    """Recognises the file from its content: a zip (DJI KMZ or zipped KML), JSON (GeoJSON), else KML."""
    name = Path(file_name).stem or "piano"
    if data[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = z.namelist()
            wpml = next((n for n in names if n.endswith("waylines.wpml")), None)
            if wpml:
                return read_wpml(name, z.read(wpml).decode("utf-8"), cam, front, high)
            kml_ = next((n for n in names if n.endswith(".kml")), None)
            if not kml_:
                return Imported([], [])
            return read_kml(name, z.read(kml_).decode("utf-8"), cam, front, high, height_if_missing)
    text = data.decode("utf-8", errors="replace")
    first = text.lstrip()[:1]
    if first in ("{", "["):
        return read_geojson(name, text, cam, front, high, height_if_missing)
    return read_kml(name, text, cam, front, high, height_if_missing)
