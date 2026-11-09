"""The route of a block: the serpentine grid and the ring, the waypoints with what the drone does on them,
the time, the batteries and the legs that fit one battery.

Heights are relative to the takeoff point (as the drone counts them). The shooting mode is stored with the
planner's codes: ``"intervallo"`` (a photo every so many seconds) or ``"per_punto"`` (a photo on every waypoint).

Ported from the author's Photo3D planner: algorithms, constants and rounding unchanged; the JSON keys of
:meth:`Stop.to_json` and :meth:`Route.to_json` are the planner's, so a plan of the planner reads as it is.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional

from .camera import Camera
from .geometry import (METRES_PER_DEGREE, LatLon, centroid, distance, interpolate, length, metres_per_degree_lon,
                       project_on_segment, simplify)

DJI_INTERVALS = (2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 60.0)
SECONDS_PER_TURN = 4.0
MAX_POINTS_PER_LEG = 100
VERTICAL_SPEED = 3.0          # m/s, to estimate the time of a descent and of the climb back
SPECIAL_POINTS_MARGIN = 16    # waypoints left free in a leg, because a special point is never split


# ------------------------------------------------------------------ grid and ring

def _spans(x: float, edge: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """The spans in which the vertical line at x lies inside the polygon (concave too)."""
    crossings = []
    j = len(edge) - 1
    for i in range(len(edge)):
        xi, yi = edge[i]
        xj, yj = edge[j]
        if (xi > x) != (xj > x):
            crossings.append(yi + (x - xi) * (yj - yi) / (xj - xi))
        j = i
    crossings.sort()
    return [(crossings[k], crossings[k + 1]) for k in range(0, len(crossings) - 1, 2)
            if crossings[k + 1] > crossings[k]]


def grid(area: list[LatLon], height: float, front: float, side: float, cam: Camera,
         heading_deg: float = 0.0) -> list[LatLon]:
    """The shots of the serpentine grid inside ``area``, half a step from the edges. The direction of the
    strips comes from rotating the frame of reference, not the points: so a narrow rotated area too gets its
    pass. ``front`` and ``side`` are the overlaps as fractions (0.8, 0.7)."""
    if len(area) < 3:
        return []
    centre = centroid(area)
    k_lon = metres_per_degree_lon(centre[0])
    ang = math.radians(heading_deg)
    ca, sa = math.cos(ang), math.sin(ang)

    def local(p: LatLon) -> tuple[float, float]:
        return ((p[1] - centre[1]) * k_lon, (p[0] - centre[0]) * METRES_PER_DEGREE)

    def to_grid(x: float, y: float) -> tuple[float, float]:
        return (x * ca + y * sa, -x * sa + y * ca)

    def to_world(x: float, y: float) -> tuple[float, float]:
        return (x * ca - y * sa, x * sa + y * ca)

    edge = [to_grid(*local(p)) for p in area]
    min_x, max_x = min(b[0] for b in edge), max(b[0] for b in edge)
    width = max_x - min_x
    strip_step = cam.footprint(height) * (1 - side)
    photo_step = cam.footprint_short(height) * (1 - front)
    strips = max(1, round(width / max(strip_step, 1.0)))
    points: list[LatLon] = []
    for s in range(strips):
        x = min_x + width * (s + 0.5) / strips
        shots = []
        for start, end in _spans(x, edge):
            span = end - start
            count = max(1, round(span / max(photo_step, 0.5)))
            shots.extend((x, start + span * (k + 0.5) / count) for k in range(count))
        if s % 2:
            shots.reverse()
        for px, py in shots:
            xm, ym = to_world(px, py)
            points.append((centre[0] + ym / METRES_PER_DEGREE, centre[1] + xm / k_lon))
    return points


def ring(centre: LatLon, radius_m: float, vertices: int = 36) -> list[LatLon]:
    """A round about the centre, closed on the first point: from the north, clockwise."""
    k_lon = metres_per_degree_lon(centre[0])
    points = []
    for k in range(vertices):
        a = 2 * math.pi * k / vertices
        points.append((centre[0] + radius_m * math.cos(a) / METRES_PER_DEGREE,
                       centre[1] + radius_m * math.sin(a) / k_lon))
    points.append(points[0])
    return points


# ------------------------------------------------------------------ the route

@dataclass(frozen=True)
class Stop:
    """What the drone does on a waypoint that is not just any.

    What is left of a special point (a detail photo, a panorama, a video) once it entered the route. Only
    the bricks DJI Fly writes by itself are used - a waypoint, its height, where the nose points, an
    action - because on the Mini 4 Pro those are known to work: a panorama is a ring of waypoints, not a
    sequence of actions on one point.
    """
    actions: tuple[str, ...] = ()        # "foto", "video_inizio", "video_fine", "camera:-45", "pausa:3"
    direction: Optional[float] = None    # where the nose points, degrees from north clockwise; None = along the route
    stops: Optional[bool] = None         # True stops on the point, False passes curving, None as the rest of the route
    speed: Optional[float] = None        # m/s from here to the next point; None = the route's
    pitch: Optional[float] = None        # the camera on this point; None = the route's
    low: bool = False                    # below the cruise height: never started or resumed from here
    from_ground: bool = False            # height from the ground: follows the real takeoff also in a flat flight
    head: bool = False                   # the first waypoint of the special point: a restart may begin here
    survey: bool = False                 # also a point of the survey: shooting by point it takes its photo
    special: str = ""                    # the name of the special point it belongs to
    kind: str = ""

    def to_json(self, i: int) -> dict:
        d: dict = {"i": i, "azioni": list(self.actions)}
        for key, value in (("direzione", self.direction), ("ferma", self.stops), ("velocita", self.speed),
                           ("inclinazione", self.pitch)):
            if value is not None:
                d[key] = value
        for key, value in (("bassa", self.low), ("dalSuolo", self.from_ground), ("capo", self.head),
                           ("rilievo", self.survey)):
            if value:
                d[key] = True
        if self.special:
            d["speciale"] = self.special
        if self.kind:
            d["tipo"] = self.kind
        return d

    @staticmethod
    def from_json(d: dict) -> "Stop":
        def number(key: str) -> Optional[float]:
            v = d.get(key)
            return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
        stops = d.get("ferma")
        return Stop(actions=tuple(str(a) for a in d.get("azioni") or ()), direction=number("direzione"),
                    stops=stops if isinstance(stops, bool) else None, speed=number("velocita"),
                    pitch=number("inclinazione"), low=bool(d.get("bassa")),
                    from_ground=bool(d.get("dalSuolo")), head=bool(d.get("capo")), survey=bool(d.get("rilievo")),
                    special=str(d.get("speciale") or ""), kind=str(d.get("tipo") or ""))

    @property
    def seconds(self) -> float:
        """How long the drone stays here, roughly: for the estimate of minutes and batteries."""
        t = 0.0
        for a in self.actions:
            if a.startswith("pausa:"):
                t += number_of(a, 2.0)
            elif a == "foto":
                t += 3.0
            else:
                t += 2.0
        return t


def number_of(action: str, if_missing: float) -> float:
    """"camera:-45" -> -45."""
    try:
        return float(action.split(":", 1)[1])
    except (IndexError, ValueError):
        return if_missing


@dataclass
class Route:
    name: str
    points: list[LatLon]
    height: float
    pitch: float = -90.0
    speed: float = 5.0
    photo_step: float = 20.0
    mode: str = "intervallo"          # or "per_punto"
    poi: Optional[LatLon] = None
    min_interval: float = 2.0
    declared_shots: Optional[int] = None
    # The flight that follows the terrain: one height per point instead of the single ``height``. The KMZ
    # writes them point by point (wpml:executeHeight), so the GSD stays constant uphill too.
    heights: Optional[list[float]] = None
    # The special points that entered the route: one entry per waypoint, None where it is just any point.
    stops: Optional[list[Optional[Stop]]] = None
    # The per-point heights are there only for the descents of the special points: the flight stays at a
    # fixed height, and is neither "following the terrain" nor moved with the takeoff point.
    flat_heights: bool = False

    @property
    def follows_terrain(self) -> bool:
        return self.heights is not None and not self.flat_heights

    def stop(self, i: int) -> Optional[Stop]:
        s = self.stops
        return s[i] if s is not None and 0 <= i < len(s) else None

    def _stops_from(self, start: int, end: int) -> Optional[list[Optional[Stop]]]:
        return self.stops[start:end] if self.stops else None

    def safe_start(self, i: int) -> int:
        """Where a restart can really begin: never from the middle of a special point. Restarting from the
        low point of a descent would mean reaching it horizontally at that height, across the site; the
        first waypoint of the special point is taken instead, at cruise height."""
        while i > 0 and (s := self.stop(i)) is not None and not s.head:
            i -= 1
        return i

    @property
    def in_video(self) -> list[bool]:
        """The waypoints between "start video" and "stop video", both included: no photos there."""
        inside, out = False, []
        for i in range(len(self.points)):
            actions = s.actions if (s := self.stop(i)) else ()
            if "video_inizio" in actions:
                inside = True
            out.append(inside)
            if "video_fine" in actions:
                inside = False
        return out

    def shoots_by_itself(self, i: int, in_video: Optional[list[bool]] = None) -> bool:
        """Shooting by point, does this waypoint take the survey's photo? The waypoints added by the special
        points do not (they have their own actions), nor those inside a video."""
        s = self.stop(i)
        if s is not None and not s.survey:
            return False
        return not (in_video if in_video is not None else self.in_video)[i]

    @property
    def special_photos(self) -> int:
        return sum(s.actions.count("foto") for s in (self.stops or []) if s)

    @property
    def stop_seconds(self) -> float:
        """The time the special points add: the actions, and the descents with their climbs back."""
        if not self.stops:
            return 0.0
        t = sum(s.seconds for s in self.stops if s)
        if self.interval:
            # shooting by time, every photo means stopping a generous interval, not two seconds
            t += self.special_photos * max(0.0, self.interval + 1.0 - 5.0)
        for i in range(1, len(self.points)):
            if self.stop(i) or self.stop(i - 1):
                t += abs(self.height_at(i) - self.height_at(i - 1)) / VERTICAL_SPEED
        return t

    def height_at(self, i: int) -> float:
        """The height of the waypoint: its own if the flight follows the terrain, else the single one."""
        q = self.heights
        return q[i] if q is not None and 0 <= i < len(q) else self.height

    def _heights_from(self, start: int, end: int) -> Optional[list[float]]:
        """The heights of the piece between two indices: cut as the points, or none if the flight is flat."""
        return self.heights[start:end] if self.heights else None

    @property
    def max_height(self) -> float:
        """The highest height really flown: following the terrain it is not the nominal one."""
        return max(self.heights) if self.heights else self.height

    @property
    def length(self) -> float:
        return length(self.points)

    @property
    def turns(self) -> int:
        return max(0, len(simplify(self.points, 1.0)) - 2)

    @property
    def interval(self) -> Optional[float]:
        """The shortest of the DJI intervals that, at the speed asked, does not shrink the step below the one
        asked; never below the camera's minimum."""
        if self.mode != "intervallo":
            return None
        ideal = max(self.photo_step / max(self.speed, 0.1), self.min_interval)
        return next((t for t in DJI_INTERVALS if t >= ideal - 1e-9), DJI_INTERVALS[-1])

    @property
    def effective_speed(self) -> float:
        if self.mode == "intervallo":
            return min(self.speed, self.photo_step / self.interval)
        return min(self.speed, self.photo_step / max(self.min_interval, 0.1))

    @property
    def effective_step(self) -> float:
        return self.effective_speed * self.interval if self.mode == "intervallo" else self.photo_step

    def shots(self) -> int:
        if self.mode == "per_punto":
            if not self.stops:
                return len(self.points)
            video = self.in_video
            return sum(1 for i in range(len(self.points)) if self.shoots_by_itself(i, video)) + self.special_photos
        return round(self.length / max(self.effective_step, 0.5)) + 1 + self.special_photos

    def from_resume(self, reached: int) -> "Route":
        """What is left to fly after the last waypoint reached: the flight restarts from there, so the strip
        interrupted is flown again whole. At least two points, so that it is still a route."""
        if len(self.points) < 2:
            return self
        i = self.safe_start(min(max(reached, 0), len(self.points) - 2))
        return replace(self, points=self.points[i:], heights=self._heights_from(i, len(self.points)),
                       stops=self._stops_from(i, len(self.points)), declared_shots=None)

    def minutes(self) -> float:
        return (self.length / max(self.effective_speed, 0.1) + self.turns * SECONDS_PER_TURN
                + self.stop_seconds) / 60.0

    def batteries(self, battery_minutes: float) -> float:
        return self.minutes() / max(battery_minutes, 1.0)

    def warnings(self, battery_minutes: float) -> list[str]:
        a = []
        if self.follows_terrain:
            # following the terrain the distance from the ground stays the one asked: the problem is not the
            # law, it is that a stock DJI does not climb more than 120 m above the takeoff
            if self.max_height > 120:
                a.append("terreno_oltre_120")
            if self.height > 120:
                a.append("quota_oltre_120")
        elif self.height > 120:
            a.append("quota_oltre_120")
        if self.effective_speed < self.speed - 0.05:
            a.append("velocita_ridotta")
        if len(self.legs(battery_minutes)) > 1:
            a.append("piu_tratte")
        if len(self.points) < 2:
            a.append("punti_pochi")
        return a

    def legs(self, battery_minutes: float, max_points: int = MAX_POINTS_PER_LEG) -> list["Leg"]:
        """Pieces that fit one battery and stay under the ceiling of waypoints. Shooting by interval the new
        leg restarts from the last point of the old one; by point, from the point after."""
        n = len(self.points)
        if n < 3:
            return [Leg(1, 1, 0, max(0, n - 1), self)]
        limit = max(battery_minutes, 1.0) * 60.0
        v = max(self.effective_speed, 0.1)
        if self.stops:
            # a special point is never split between two batteries: the cut waits for it to end, and to stay
            # under the ceiling of waypoints all the same some are left free
            max_points = max(10, max_points - SPECIAL_POINTS_MARGIN)
        turn = [1 <= i < n - 1 and project_on_segment(self.points[i], self.points[i - 1], self.points[i + 1])[1] > 1.0
                for i in range(n)]
        spans: list[tuple[int, int]] = []
        start, time, i = 0, 0.0, 1
        while i < n:
            cost = distance(self.points[i - 1], self.points[i]) / v + (SECONDS_PER_TURN if turn[i] else 0.0)
            if (s := self.stop(i)) is not None:
                cost += s.seconds + abs(self.height_at(i) - self.height_at(i - 1)) / VERTICAL_SPEED
            too_many = i - start + 1 > max_points
            inside_a_special = s is not None and not s.head
            if (time + cost > limit or too_many) and i - 1 > start and not inside_a_special:
                spans.append((start, i - 1))
                time = 0.0
                if self.mode == "intervallo":
                    start = i - 1
                else:
                    start = i
                    i += 1
                continue
            time += cost
            i += 1
        spans.append((start, n - 1))
        total = len(spans)
        return [Leg(k + 1, total, a, b,
                    replace(self, name=self.name if total == 1 else f"{self.name} {k + 1}/{total}",
                            points=self.points[a:b + 1], heights=self._heights_from(a, b + 1),
                            stops=self._stops_from(a, b + 1), declared_shots=None))
                for k, (a, b) in enumerate(spans)]

    def to_json(self, battery_minutes: float, cam: Camera, high: bool, kind: str = "percorso",
                perimeter: Optional[list[LatLon]] = None) -> dict:
        legs = self.legs(battery_minutes)
        return {
            "nome": self.name, "tipo": kind,
            "punti": [list(p) for p in self.points],
            "perimetro": [list(p) for p in (perimeter or [])],
            "poi": list(self.poi) if self.poi else None,
            "quota": self.height, "inclinazione": self.pitch,
            "velocita": self.speed, "velocitaEffettiva": round(self.effective_speed, 2),
            "modo": self.mode, "intervallo": self.interval, "passo": round(self.effective_step, 1),
            "scatti": self.shots(), "scattiDichiarati": self.declared_shots,
            "lunghezzaM": round(self.length), "minuti": round(self.minutes(), 1),
            "batterie": round(self.batteries(battery_minutes), 1),
            "gsdCm": round(cam.gsd_cm(self.height, high), 2),
            "seguiTerreno": self.follows_terrain,
            "soste": [s.to_json(i) for i, s in enumerate(self.stops or []) if s],
            "quote": [round(q, 1) for q in self.heights] if self.heights else None,
            "quotaMin": round(min(self.heights), 1) if self.heights else self.height,
            "quotaMax": round(self.max_height, 1),
            "waypoint": len(self.points),
            "tratte": [{"numero": leg.number, "totale": leg.total, "da": leg.start, "a": leg.end,
                        "punti": len(leg.route.points), "minuti": round(leg.route.minutes(), 1)} for leg in legs],
            "avvisi": self.warnings(battery_minutes),
        }


@dataclass
class Leg:
    number: int
    total: int
    start: int
    end: int
    route: Route


def shot_points(route: Route) -> list[tuple[LatLon, float]]:
    """Where the photos are taken, each with its height above the takeoff, in the order of the flight.

    By point (``per_punto``): the waypoints that take the survey's photo by themselves. By interval: one shot
    every ``effective_step`` metres along the whole track from its first point (the timer of the camera does
    not restart at the turns), the height interpolated linearly between the waypoints, as the drone flies;
    their number is :meth:`Route.shots` within one. In both modes, the photos of the special points are
    added on their waypoint (one per ``"foto"`` action).
    """
    pts = route.points
    out: list[tuple[LatLon, float]] = []
    if route.mode == "per_punto":
        video = route.in_video
        for i, p in enumerate(pts):
            if route.shoots_by_itself(i, video):
                out.append((p, route.height_at(i)))
            if (s := route.stop(i)) is not None:
                out.extend((p, route.height_at(i)) for _ in range(s.actions.count("foto")))
        return out
    if not pts:
        return out
    if len(pts) == 1:
        out.append((pts[0], route.height_at(0)))
    step = max(route.effective_step, 0.5)
    travelled = 0.0          # metres from the start to the beginning of the current segment
    next_shot = 0.0          # metres from the start to the next photo
    for i in range(len(pts)):
        if i > 0:
            a, b = pts[i - 1], pts[i]
            seg = distance(a, b)
            ha, hb = route.height_at(i - 1), route.height_at(i)
            while seg > 0 and next_shot <= travelled + seg + 1e-9:
                t = min(1.0, (next_shot - travelled) / seg)
                out.append((interpolate(a, b, t), ha + (hb - ha) * t))
                next_shot += step
            travelled += seg
        if (s := route.stop(i)) is not None:
            out.extend((pts[i], route.height_at(i)) for _ in range(s.actions.count("foto")))
    return out
