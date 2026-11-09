"""The whole plan: a JSON dict of blocks (grids, rings, routes given point by point) with the camera, the
battery, the takeoff and the terrain; its numbers block by block and in total, and its legs in the order of
flight, each with its key, its file name and its takeoff.

The dict keeps the planner's schema and codes (``nome``, ``camera``, ``alta``, ``minutiBatteria``,
``blocchi[{tipo, nome, area, quota, ...}]``, ``decollo``, ``decolli``, ``quotaDecollo``, ``suoloDecolli``,
``terreno``...), so a plan of the planner loads as it is. The plugin adds ``cameraPersonale`` (a camera
not in the list, :meth:`~ppg_flightplan.camera.Camera.to_json`).

Ported from the author's Photo3D planner: algorithms, constants and rounding unchanged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Optional

from .camera import Camera, camera
from .geometry import LatLon, centroid, densify, distance, simplify
from .route import Route, grid, ring
from . import specials as _specials

# Farther than this from the flight a takeoff point is not a takeoff: it was left there by another plan.
FAR_TAKEOFF_M = 2000.0

# The limits within which the heights of a flight following the terrain stay.
MIN_FLIGHT_HEIGHT = 10.0
MAX_FLIGHT_HEIGHT = 500.0      # the technical ceiling: the 120 m of the law are from the ground, not the takeoff


def plan_camera(piano: dict) -> Camera:
    """The camera of the plan: its own (``cameraPersonale``) when chosen, else the one of the list."""
    return camera(piano.get("camera", "mini4pro"), piano.get("cameraPersonale"))


def route_of_block(b: dict, piano: dict) -> tuple[Route, list[LatLon]]:
    """From a block of the plan (grid, ring or given route) to the route to fly and its perimeter."""
    cam = plan_camera(piano)
    high = bool(b.get("alta", piano.get("alta", False)))
    kind = b.get("tipo", "griglia")
    height = float(b.get("quota", 50))
    front = float(b.get("avanti", 80)) / 100
    side = float(b.get("lato", 65)) / 100
    mode = "per_punto" if b.get("modo") == "per_punto" else "intervallo"
    step = cam.footprint_short(height) * (1 - front)
    common = dict(height=height, pitch=float(b.get("inclinazione", -90)), speed=float(b.get("velocita", 5)),
                  photo_step=step, mode=mode, min_interval=cam.min_interval_for(high))
    name = b.get("nome") or kind

    def with_heights(p: Route) -> Route:
        """The per-point heights hold only if they are as many as the points: if the block changed after
        they were computed the single height comes back, instead of flying wrong heights."""
        q = b.get("quote")
        if not q or len(q) != len(p.points):
            return p
        return replace(p, heights=[float(x) for x in q])

    if kind == "griglia":
        area = [tuple(p) for p in b.get("area", [])]
        shots = grid(area, height, front, side, cam, float(b.get("orientamento", 0)))
        points = shots if mode == "per_punto" else simplify(shots, 1.0)
        return with_heights(Route(name, points, declared_shots=len(shots), **common)), area
    if kind == "anello":
        centre = tuple(b.get("centro") or (0.0, 0.0))
        radius = float(b.get("raggio", 100))
        vertices = ring(centre, radius, int(b.get("vertici", 36)))
        points = densify(vertices, step) if mode == "per_punto" else vertices
        return with_heights(Route(name, points, poi=centre, **common)), []
    # a route given point by point (imported or drawn by hand): a road if open, a ring round the outline of
    # the site if closed; the camera looks at the aim, or the centre, or ahead
    points = [tuple(p) for p in b.get("punti", [])]
    if b.get("chiuso") and len(points) >= 3 and distance(points[0], points[-1]) > 0.5:
        points = points + [points[0]]
    poi = tuple(b["poi"]) if b.get("poi") else None
    if poi is None and b.get("versoCentro") and len(points) >= 3:
        poi = centroid(points[:-1] if len(points) > 3 and distance(points[0], points[-1]) < 0.5 else points)
    if mode == "per_punto":
        points = densify(points, step)
    return with_heights(Route(name, points, poi=poi, **common)), []


def _terrain_for_specials(piano: dict):
    g = piano.get("terreno")
    ground = _specials.ground_of_grid(g)
    return (ground, float(g["suoloRiferimento"])) if ground and g.get("suoloRiferimento") is not None else (None, None)


def flown_route(b: dict, piano: dict) -> tuple[Route, list[LatLon]]:
    """The route the drone really flies: the block's, with its special points inside."""
    p, perimeter = route_of_block(b, piano)
    if not b.get("speciali"):
        return p, perimeter
    return _specials.with_specials(p, b["speciali"], *_terrain_for_specials(piano))[0], perimeter


def special_outcomes(b: dict, piano: dict) -> list[dict]:
    """What came of each special point: at which waypoint, at which height, with which warnings."""
    if not b.get("speciali"):
        return []
    p, _ = route_of_block(b, piano)
    return _specials.with_specials(p, b["speciali"], *_terrain_for_specials(piano))[1]


def specials_need_ground(piano: dict) -> bool:
    """Is there a descent to make and the plan carries no terrain? Then it must be measured: for the "metres
    from the ground" it is essential, for the "metres above the takeoff" it is the check that nothing is hit."""
    return any((sp.get("discesa") or {}).get("modo") in ("decollo", "suolo")
               for b in piano.get("blocchi", []) if active(b) for sp in b.get("speciali") or [])


def active(b: dict) -> bool:
    """A block can be switched off without removing it: it stays in the plan but does not fly, count or export."""
    return b.get("attivo", True) is not False


def compute(piano: dict) -> dict:
    """All the numbers of the plan, block by block and in total (only the active blocks)."""
    cam = plan_camera(piano)
    high = bool(piano.get("alta", False))
    battery_minutes = float(piano.get("minutiBatteria", 18))
    blocks = []
    for b in piano.get("blocchi", []):
        p, perimeter = flown_route(b, piano)
        entry = p.to_json(battery_minutes, cam, bool(b.get("alta", high)), b.get("tipo", "griglia"), perimeter)
        entry["attivo"] = active(b)
        entry["speciali"] = special_outcomes(b, piano)
        # the survey's waypoints, without those added by the special points
        survey = route_of_block(b, piano)[0].points if b.get("speciali") else p.points
        entry["puntiRilievo"] = [list(q) for q in survey] if b.get("speciali") else None
        entry["waypointRilievo"] = len(survey)        # the terrain heights are as many as THESE
        blocks.append(entry)
    on = [x for x in blocks if x["attivo"]]
    return {
        "camera": cam.to_json(),
        "blocchi": blocks,
        "decolloLontanoM": (lambda d: None if d is None else round(d))(takeoff_far(piano)),
        "totale": {"scatti": sum(x["scatti"] for x in on),
                   "minuti": round(sum(x["minuti"] for x in on), 1),
                   "batterie": round(sum(x["batterie"] for x in on), 1),
                   "km": round(sum(x["lunghezzaM"] for x in on) / 1000, 1),
                   "tratte": sum(len(x["tratte"]) for x in on),
                   "blocchi": len(on)},
    }


def active_blocks(piano: dict) -> list[dict]:
    return [b for b in piano.get("blocchi", []) if active(b)]


def active_areas(piano: dict) -> list[list[LatLon]]:
    return [[tuple(p) for p in b.get("area", [])] for b in active_blocks(piano) if b.get("tipo") == "griglia"]


def plan_routes(piano: dict, only_active: bool = True) -> list[Route]:
    """The routes to fly: as a rule only the active blocks, which are those that go into the files."""
    blocks = active_blocks(piano) if only_active else piano.get("blocchi", [])
    return [flown_route(b, piano)[0] for b in blocks]


def slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()
    return s[:40] or "piano"


@dataclass(frozen=True)
class PlanLeg:
    """A leg to fly, with its place in the plan: the block it comes from, the key by which a takeoff is
    given to it, the name of the files made from it and its takeoff (its own, or the general one)."""
    block_index: int
    key: str
    file_name: str
    number: int
    total: int
    route: Route
    takeoff: Optional[LatLon] = None
    # waypoints cut at the floor from the general takeoff whose ground is unknown: their height from the leg's own
    # takeoff cannot be told (:func:`_relative_to_its_takeoff`), and the export refuses the leg
    floor_unknown: int = 0

    def to_json(self) -> dict:
        return {"chiave": self.key, "blocco": self.block_index, "nome": self.file_name, "numero": self.number,
                "totale": self.total, "punti": len(self.route.points), "scatti": self.route.shots(),
                "minuti": round(self.route.minutes(), 1), "decollo": list(self.takeoff) if self.takeoff else None}


def plan_takeoff(piano: dict) -> Optional[LatLon]:
    d = piano.get("decollo")
    return (float(d[0]), float(d[1])) if d else None


def takeoff_distance(piano: dict) -> Optional[float]:
    """How far the takeoff is from the nearest point of the flight (the active blocks); None if either is missing."""
    takeoff = plan_takeoff(piano)
    if takeoff is None:
        return None
    points = [q for b in active_blocks(piano) for q in route_of_block(b, piano)[0].points]
    return min(distance(takeoff, q) for q in points) if points else None


def takeoff_far(piano: dict) -> Optional[float]:
    """The metres between the takeoff and the flight, if they are too many to be true; else None.

    Drawing a new area the takeoff may stay where it was, maybe in another region and off the screen. A
    flight following the terrain counts its heights from there: with a takeoff 136 km away and 500 m lower,
    every waypoint ended at the 500 m ceiling (a field case of the planner).
    """
    d = takeoff_distance(piano)
    return d if d is not None and d > FAR_TAKEOFF_M else None


def _relative_to_its_takeoff(p: Route, piano: dict, key: str, takeoff: Optional[LatLon]) -> tuple[Route, int]:
    """The heights of a leg that takes off from a point of its own, brought back to THAT point, and how many of its
    waypoints could not be (see below).

    The per-point heights are computed from the general takeoff; the drone measures them from where it
    really takes off. If the leg starts twenty metres higher (the top of the mound) its heights go down by
    twenty, or it would fly twenty metres higher than wanted - and the other way round, lower.

    A height cut at the floor (10 m above the general takeoff, where the ground falls more than the height above
    it) is not the height above the ground any more: moved by the drop it would fly too high (a leg taking off at
    the foot of a 200 m slope: 210 m instead of 60). Taking off lower, such a point is computed again from the
    ground the plan carries (its terrain grid): the height above the ground plus the ground there, counted from
    this takeoff. Where that ground is unknown the point is counted (the second value) and the export refuses
    the leg.
    """
    if not p.heights or takeoff is None:
        return p, 0
    general = piano.get("quotaDecollo")
    entry = (piano.get("suoloDecolli") or {}).get(key)
    if general is None or not entry or distance(tuple(entry["punto"]), takeoff) > 2.0:
        return p, 0                    # ground of the takeoff unknown or moved: better not to touch anything
    if p.points and min(distance(takeoff, q) for q in p.points) > FAR_TAKEOFF_M:
        return p, 0                    # a takeoff kilometres from the leg is not a takeoff: nothing is moved
    shift = float(general) - float(entry["suolo"])
    new = []
    unknown = 0
    ground = _specials.ground_of_grid(piano.get("terreno"))
    for i, q in enumerate(p.heights):
        s_ = p.stop(i)
        if p.flat_heights and s_ and s_.low and not s_.from_ground:
            # descent "above the takeoff" in a flat flight: the height stays, but from THIS takeoff it must
            # not end less than five metres from the ground - taking off lower that point goes down
            z = ground(p.points[i]) if ground else None
            floor = z - float(entry["suolo"]) + _specials.MIN_DISTANCE_FROM_GROUND if z is not None else None
            new.append(q if floor is None else round(max(q, floor), 1))
            continue
        if p.flat_heights and not (s_ and s_.from_ground):
            new.append(q)                # a flight at a fixed height: it holds from any takeoff
            continue
        minimum = _specials.MIN_SPECIAL_HEIGHT if s_ and s_.low else MIN_FLIGHT_HEIGHT
        if shift > 0.05 and not (s_ and s_.low) and q <= MIN_FLIGHT_HEIGHT + 0.05:
            # cut at the floor from the general takeoff: from this lower takeoff, from the ground itself
            z = ground(p.points[i]) if ground else None
            if z is None:
                unknown += 1
            else:
                new.append(round(min(MAX_FLIGHT_HEIGHT, max(minimum, p.height + z - float(entry["suolo"]))), 1))
                continue
        new.append(round(min(MAX_FLIGHT_HEIGHT, max(minimum, q + shift)), 1))
    return replace(p, heights=new), unknown


def plan_legs(piano: dict, specials: bool = True) -> list[PlanLeg]:
    """The legs of every active block, in the order of flight. The key is "index of the block-number of the
    leg"; the takeoff is the one chosen for the leg in ``decolli``, else the plan's.

    With ``specials=False`` only the survey: for the formats that cannot carry the actions of the
    special points (Litchi, QGroundControl, Parrot), where they would stay as meaningless extra waypoints."""
    base = slug(piano.get("nome", "piano"))
    minutes = float(piano.get("minutiBatteria", 18))
    general = plan_takeoff(piano)
    takeoffs = piano.get("decolli") or {}
    on = [(i, b) for i, b in enumerate(piano.get("blocchi", [])) if active(b)]
    out: list[PlanLeg] = []
    for k, (i, b) in enumerate(on):
        p, _ = flown_route(b, piano) if specials else route_of_block(b, piano)
        legs = p.legs(minutes)
        for leg in legs:
            name = f"{base}-{k + 1}-{slug(p.name)}" if len(on) > 1 else base
            file_name = name if len(legs) == 1 else f"{name}-tratta{leg.number}"
            key = f"{i}-{leg.number}"
            d = takeoffs.get(key)
            takeoff = (float(d[0]), float(d[1])) if d else general
            route, unknown = _relative_to_its_takeoff(leg.route, piano, key, takeoff if d else None)
            out.append(PlanLeg(i, key, file_name, leg.number, leg.total, route, takeoff, unknown))
    return out


def plan_takeoffs(piano: dict) -> list[tuple[str, LatLon]]:
    """The takeoff points to put on the maps and in the KML: the general one and those chosen for single legs."""
    out: list[tuple[str, LatLon]] = []
    general = plan_takeoff(piano)
    if general:
        out.append(("Decollo", general))
    for leg in plan_legs(piano):
        if leg.takeoff and leg.takeoff != general:
            out.append((f"Decollo {leg.file_name}", leg.takeoff))
    return out


def far_leg_takeoffs(piano: dict) -> list[tuple[str, str, float]]:
    """The legs whose own takeoff (``decolli``) is farther than :data:`FAR_TAKEOFF_M` from every point of the leg:
    (key, file name, metres). Such a takeoff is not taken as one for the heights (:func:`_relative_to_its_takeoff`),
    so the export refuses it (:class:`ppg_flightplan.design.FarLegTakeoff`) instead of counting the heights from
    another takeoff than the one set."""
    own = piano.get("decolli") or {}
    if not own:
        return []
    out = []
    for leg in plan_legs(piano):
        d = own.get(leg.key)
        if not d or not leg.route.points:
            continue
        point = (float(d[0]), float(d[1]))
        metres = min(distance(point, q) for q in leg.route.points)
        if metres > FAR_TAKEOFF_M:
            out.append((leg.key, leg.file_name, round(metres)))
    return out


# A leg whose first or last waypoint moved more than this, or whose waypoints are not as many, changed.
LEG_MOVED_M = 2.0


def moved_leg_takeoffs(old: dict, new: dict) -> list[str]:
    """The keys of the per-leg takeoffs of ``new`` (kept from ``old`` by key after a new design) whose leg is not the
    one it was: the keys are positional (block - number of the leg), so a kept takeoff may now belong to a leg in
    another place. The caller asks to check them."""
    own = new.get("decolli") or {}
    if not own:
        return []
    before = {leg.key: leg.route.points for leg in plan_legs(old)}
    out = []
    for leg in plan_legs(new):
        if leg.key not in own:
            continue
        was, now = before.get(leg.key), leg.route.points
        if not was or len(was) != len(now) or not now or distance(was[0], now[0]) > LEG_MOVED_M \
                or distance(was[-1], now[-1]) > LEG_MOVED_M:
            out.append(leg.key)
    return out


def site_centre(piano: dict) -> Optional[LatLon]:
    """Where the plan is: the centre of the waypoints of the active blocks, else of the areas of all its blocks (all
    switched off), else of the area drawn; None for an empty plan. Reading targets, the order of two columns of
    degrees is chosen by the side nearer to it."""
    points = [q for b in active_blocks(piano) for q in route_of_block(b, piano)[0].points]
    if not points:
        points = [(float(p[0]), float(p[1])) for b in piano.get("blocchi", []) for p in b.get("area") or []]
        points += [(float(c[0]), float(c[1])) for b in piano.get("blocchi", []) if b.get("centro")
                   for c in [b["centro"]]]
    if not points:
        points = [(float(p[0]), float(p[1])) for p in piano.get("areaDisegnata") or []]
    return centroid(points) if points else None
