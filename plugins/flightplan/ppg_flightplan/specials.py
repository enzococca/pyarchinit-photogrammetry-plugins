"""Special points of a block (detail photos, panoramas, video): what they add to the route.

A special point is a dict of the plan (``blocchi[].speciali``): ``tipo`` (one of :data:`SPECIAL_KINDS`),
``punto`` (lat, lon), ``aggancio`` (hooked on the nearest waypoint, the default, or free: visited at the end of
the block), ``discesa`` (``{"modo": "decollo" | "suolo", "metri": ...}``), ``inclinazione``, ``scaletta``,
``scatti`` (panorama), ``attivo``. :func:`with_specials` puts them into the route and says what came of each.

Ported from the author's Photo3D planner: algorithms, constants, rounding and the order of operations
unchanged; the warning codes are the planner's (see ``texts.WARNINGS``).
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Optional

from .geometry import LatLon, bearing, distance, interpolate, offset
from .route import Route, Stop

SPECIAL_KINDS = ("foto", "panoramica", "giro_video", "video_inizio", "video_fine")
MIN_DISTANCE_FROM_GROUND = 5.0   # never lower than this above the ground by itself
MIN_SPECIAL_HEIGHT = 5.0         # and never below five metres above the takeoff
STEP_BETWEEN_WAYPOINTS = 1.0     # DJI refuses two waypoints less than half a metre apart: one is kept
TRANSFER_STEP = 25.0             # going to a free point over rough terrain, a waypoint every so often
VIDEO_RING_RADIUS = 5.0
VIDEO_RING_POINTS = 12
VIDEO_RING_SPEED = 1.0
LADDER_RATIO = 0.6               # descending on a target, every step is at six tenths of the previous one
PANORAMA_PITCH = -10.0
VIDEO_RING_PITCH = -15.0


def num(x: float) -> str:
    """A number as the files and the actions write it: no trailing zeros ("60", "-45.5")."""
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.6f}".rstrip("0").rstrip(".")


def wpml_degrees(degrees: float) -> float:
    """The direction as the WPML wants it: from -180 to 180."""
    g = degrees % 360.0
    return round(g - 360.0 if g > 180.0 else g, 1)


def ground_of_grid(g: Optional[dict]):
    """From a terrain grid (the one the plan carries) to the function giving the height of the ground at a
    point: interpolated between the four nodes around it, None outside or where it is missing."""
    if not g or not g.get("quote"):
        return None
    south, west, step_lat, step_lon = g["sud"], g["ovest"], g["passoLat"], g["passoLon"]
    rows, cols, heights = int(g["righe"]), int(g["colonne"]), g["quote"]
    if rows < 2 or cols < 2 or len(heights) != rows * cols:
        return None

    def ground(p: LatLon) -> Optional[float]:
        fr, fc = (p[0] - south) / step_lat, (p[1] - west) / step_lon
        if fr < 0 or fc < 0 or fr > rows - 1 or fc > cols - 1:
            return None
        r0, c0 = min(int(fr), rows - 2), min(int(fc), cols - 2)
        tr, tc = fr - r0, fc - c0
        nodes = [heights[r0 * cols + c0], heights[r0 * cols + c0 + 1],
                 heights[(r0 + 1) * cols + c0], heights[(r0 + 1) * cols + c0 + 1]]
        known = [q for q in nodes if q is not None]
        if not known:
            return None
        if len(known) < 4:
            return sum(known) / len(known)
        return (nodes[0] * (1 - tr) * (1 - tc) + nodes[1] * (1 - tr) * tc
                + nodes[2] * tr * (1 - tc) + nodes[3] * tr * tc)
    return ground




def _clamp(x: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, x))


def with_specials(p: Route, specials: list[dict], ground=None,
                  ground_takeoff: Optional[float] = None) -> tuple[Route, list[dict]]:
    """Puts the special points into the route, and says what came of each.

    A **hooked** point is made on the nearest waypoint of the route, when the drone passes there. A **free**
    point is visited at the end of the block, nearest first: the survey is flown whole and without detours,
    then the drone goes round the free points and comes home.

    The **descent** is vertical, never diagonal: the drone arrives above the point at the cruise height,
    descends, does what it must and climbs back before leaving. Descending while flying towards the point
    would mean crossing the site low. The low height is given in two ways: metres above the takeoff (the
    pilot decides) or metres from the ground (the plan's terrain is needed). In both cases, if the terrain is
    known, the drone never goes below five metres from the ground.
    """
    from .plan import MAX_FLIGHT_HEIGHT, MIN_FLIGHT_HEIGHT   # plan imports this module

    valid = [sp for sp in specials or [] if sp.get("tipo") in SPECIAL_KINDS and sp.get("punto")
             and sp.get("attivo", True) is not False]
    if not valid or not p.points:
        return p, []
    relative = None
    if ground is not None and ground_takeoff is not None:
        def relative(q: LatLon) -> Optional[float]:            # the ground there, counted from the takeoff
            z = ground(q)
            return None if z is None else z - ground_takeoff
    pause = "pausa:2"                   # the time to stop; shooting by time, whoever writes the file lengthens it
    outcomes: list[dict] = []

    def cruise(q: LatLon) -> float:
        if not p.follows_terrain:
            return p.height
        r = relative(q) if relative else None
        if r is None:
            return p.max_height                    # terrain unknown there: high, which does no harm
        return round(_clamp(p.height + r, MIN_FLIGHT_HEIGHT, MAX_FLIGHT_HEIGHT), 1)

    def low_height(sp: dict, q: LatLon, high: float, outcome: dict) -> tuple[Optional[float], bool]:
        d = sp.get("discesa") or {}
        mode = d.get("modo")
        if mode not in ("decollo", "suolo"):
            return None, False
        try:
            metres = float(d.get("metri"))
        except (TypeError, ValueError):
            return None, False
        r = relative(q) if relative else None
        if mode == "suolo":
            if r is None:
                outcome["avvisi"].append("senza_terreno")    # nobody knows where the ground is: no descent
                return None, False
            metres = max(metres, MIN_DISTANCE_FROM_GROUND)
            low = r + metres
        else:
            low = metres
            if r is None:
                outcome["avvisi"].append("suolo_sconosciuto")    # it descends, but what is below is the pilot's
            elif low - r < MIN_DISTANCE_FROM_GROUND:
                low = r + MIN_DISTANCE_FROM_GROUND
                outcome["avvisi"].append("alzato")
        if low < MIN_SPECIAL_HEIGHT:
            low = MIN_SPECIAL_HEIGHT
            outcome["avvisi"].append("sotto_il_decollo")
        low = round(low, 1)
        if low > high - 1.0:
            outcome["avvisi"].append("gia_bassa")
            return None, False
        outcome["quotaBassa"] = low
        outcome["dalSuolo"] = None if r is None else round(low - r, 1)
        return low, mode == "suolo"

    def waypoints_of_the_special(sp: dict, where: LatLon, degrees: float, high: float, hooked: bool,
                                 outcome: dict) -> list[tuple[LatLon, float, Stop]]:
        """The waypoints of a special point: ``where`` is the first (on the route, if hooked), the others move
        away one metre at a time in the direction ``degrees``."""
        kind, name = sp["tipo"], str(sp.get("nome") or sp["tipo"])
        common = dict(special=name, kind=kind)
        if kind in ("video_inizio", "video_fine"):
            return [(where, high, Stop(actions=(kind,), head=True, survey=hooked, **common))]
        low, from_ground = low_height(sp, offset(where, degrees, STEP_BETWEEN_WAYPOINTS), high, outcome)
        action_height = low if low is not None else high
        down = dict(low=low is not None, from_ground=from_ground)
        if kind == "foto":
            wanted = sp.get("inclinazione")
            cam = float(wanted) if wanted is not None and abs(float(wanted) - p.pitch) > 0.5 else None
            if low is None and cam is None:
                if p.mode == "per_punto" and hooked:
                    outcome["avvisi"].append("nessun_effetto")     # that waypoint takes the photo already
                    return [(where, high, Stop(head=True, survey=True, **common))]
                return [(where, high, Stop(actions=("foto",), stops=True, head=True, survey=hooked, **common))]
            # The ladder: descending, a photo at every step (60 -> 36 -> 22 -> 13 -> 10 m). It serves the targets
            # on the ground: a single photo from ten metres, five times finer than the survey, hardly ties to
            # the others, and a target seen in a single photo has no position. So every photo resembles the
            # previous one, and the target is seen several times. The steps are one metre apart in plan,
            # alternating, the last one on the point.
            steps: list[tuple[LatLon, float, Stop]] = []
            if low is not None and sp.get("scaletta"):
                # the steps are counted FROM THE GROUND, if it is known: the distance from the ground makes the
                # detail of the photo, and on a target at the top of a hill counting them from the takeoff
                # would make the first jump too long
                ground_here = (relative(offset(where, degrees, STEP_BETWEEN_WAYPOINTS)) if relative else None) or 0.0
                levels, z = [], high - ground_here
                while z * LADDER_RATIO > (low - ground_here) + 2.0:
                    z *= LADDER_RATIO
                    levels.append(round(z + ground_here, 1))
                for j, z in enumerate(levels):
                    paces = 2 if (len(levels) - j) % 2 else 1          # ...+2, +1, +2, and the low point at +1
                    steps.append((offset(where, degrees, paces * STEP_BETWEEN_WAYPOINTS), z,
                                  Stop(actions=(pause, "foto"), stops=True, pitch=cam, **down, **common)))
                outcome["gradini"] = levels
            return [
                (where, high, Stop(actions=(f"camera:{num(cam)}",) if cam is not None else (), stops=True,
                                   head=True, survey=hooked, pitch=cam, **common)),
                *steps,
                (offset(where, degrees, STEP_BETWEEN_WAYPOINTS), action_height,
                 Stop(actions=(pause, "foto"), stops=True, pitch=cam, **down, **common)),
                (offset(where, degrees, 2 * STEP_BETWEEN_WAYPOINTS), high,
                 Stop(actions=(f"camera:{num(p.pitch)}",) if cam is not None else (), stops=True, **common)),
            ]
        # panorama by photos and ring by video: a ring of waypoints with the nose outwards
        video = kind == "giro_video"
        wanted = sp.get("inclinazione")
        cam = float(wanted) if wanted is not None else (VIDEO_RING_PITCH if video else PANORAMA_PITCH)
        if video:
            n, radius = VIDEO_RING_POINTS, VIDEO_RING_RADIUS
        else:
            n = int(_clamp(float(sp.get("scatti") or 8), 4, 16))
            radius = max(1.5, 1.2 * STEP_BETWEEN_WAYPOINTS / (2 * math.sin(math.pi / n)))
        out = [(where, high, Stop(actions=(f"camera:{num(cam)}",), stops=True, head=True, survey=hooked,
                                  pitch=cam, **common))]
        first = where
        if low is not None:
            first = offset(where, degrees, STEP_BETWEEN_WAYPOINTS)
            out.append((first, action_height, Stop(stops=True, pitch=cam, **down, **common)))
        centre = offset(first, degrees, STEP_BETWEEN_WAYPOINTS + radius)
        for k in range(n + 1 if video else n):
            angle = degrees + 180.0 + 360.0 * (k % n) / n
            if video:
                actions = ("video_inizio",) if k == 0 else ("video_fine",) if k == n else ()
                stop = Stop(actions=actions, direction=wpml_degrees(angle), stops=k in (0, n),
                            speed=VIDEO_RING_SPEED, pitch=cam, **down, **common)
            else:
                stop = Stop(actions=(pause, "foto"), direction=wpml_degrees(angle),
                            stops=True, speed=VIDEO_RING_SPEED, pitch=cam, **down, **common)
            out.append((offset(centre, angle, radius), action_height, stop))
        # climbing back above the point it descended from, and the camera goes back as it was
        out.append((first if low is not None else offset(where, degrees + 90.0, STEP_BETWEEN_WAYPOINTS), high,
                    Stop(actions=(f"camera:{num(p.pitch)}",), stops=True, **common)))
        return out

    # ---- who is hooked on which waypoint, and who is free
    hooked: dict[int, list[dict]] = {}
    free: list[dict] = []
    for sp in valid:
        point = (float(sp["punto"][0]), float(sp["punto"][1]))
        if sp.get("aggancio", True) or sp["tipo"] in ("video_inizio", "video_fine"):
            k = min(range(len(p.points)), key=lambda i: distance(p.points[i], point))
            hooked.setdefault(k, []).append(sp)
        else:
            free.append(sp)

    points: list[LatLon] = []
    heights: list[float] = []
    stops: list[Optional[Stop]] = []

    def add(entries: list[tuple[LatLon, float, Optional[Stop]]]) -> None:
        for q, z, s_ in entries:
            points.append(q)
            heights.append(round(z, 1))
            stops.append(s_)

    for i, q in enumerate(p.points):
        here = hooked.get(i)
        if not here:
            add([(q, p.height_at(i), None)])
            continue
        degrees = (bearing(q, p.points[i + 1]) if i + 1 < len(p.points)
                   else bearing(p.points[i - 1], q) if i > 0 else 0.0)
        for n_, sp in enumerate(here):
            # several special points on the same waypoint: the first is the waypoint, the others a little ahead
            where = q if n_ == 0 else offset(q, degrees, 4.0 * n_ + VIDEO_RING_RADIUS * 2 * n_)
            outcome = {"id": sp.get("id"), "nome": sp.get("nome"), "tipo": sp["tipo"], "waypoint": i,
                       "quota": p.height_at(i), "quotaBassa": None, "dalSuolo": None, "avvisi": []}
            add(waypoints_of_the_special(sp, where, degrees, p.height_at(i), n_ == 0, outcome))
            outcomes.append(outcome)

    # ---- the free points, at the end, nearest first
    to_do = list(free)
    while to_do:
        last = points[-1]
        sp = min(to_do, key=lambda x: distance(last, (float(x["punto"][0]), float(x["punto"][1]))))
        to_do.remove(sp)
        goal = (float(sp["punto"][0]), float(sp["punto"][1]))
        degrees = bearing(last, goal) if distance(last, goal) > 0.5 else 0.0
        high = cruise(goal)
        # over rough terrain the transfer follows the ground too, or between two low points the drone would
        # cut straight over (or into) what is in between
        if p.follows_terrain and relative:
            stretch = distance(last, goal)
            paces = int(stretch // TRANSFER_STEP)
            for k in range(1, paces):
                middle = interpolate(last, goal, k / paces)
                add([(middle, cruise(middle), None)])
        outcome = {"id": sp.get("id"), "nome": sp.get("nome"), "tipo": sp["tipo"], "waypoint": None,
                   "quota": high, "quotaBassa": None, "dalSuolo": None, "avvisi": []}
        # the right point is the one of the action: the drone stops a metre before, so the descent falls on it
        where = offset(goal, degrees + 180.0, STEP_BETWEEN_WAYPOINTS) if sp["tipo"] == "foto" else goal
        add(waypoints_of_the_special(sp, where, degrees, high, False, outcome))
        outcomes.append(outcome)

    needs_heights = p.heights is not None or any(abs(z - p.height) > 0.05 for z in heights)
    return replace(p, points=points, heights=heights if needs_heights else None, stops=stops,
                   flat_heights=needs_heights and p.heights is None, declared_shots=None), outcomes
