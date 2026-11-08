"""Geometry of the planner on WGS84 latitude and longitude, in local metres.

A point is ``(lat, lon)`` in degrees. Distances use the planner's flat-earth metres: 111 320 m per degree of
latitude, ``cos(lat)`` of that per degree of longitude. Over a survey (a few km) the error is a few parts per
thousand and always on the side of more overlap than nominal (the degree is 0.3 % shorter than 111 320 m at 36
degrees): the coverage the plugin shows is computed with pyproj and is the truth the user sees.

Ported from the author's Photo3D planner: algorithms, constants and rounding unchanged.
"""
from __future__ import annotations

import math

METRES_PER_DEGREE = 111_320.0

LatLon = tuple[float, float]  # (lat, lon)


def metres_per_degree_lon(lat: float) -> float:
    return METRES_PER_DEGREE * math.cos(math.radians(lat))


def distance(a: LatLon, b: LatLon) -> float:
    dx = (b[1] - a[1]) * metres_per_degree_lon((a[0] + b[0]) / 2)
    dy = (b[0] - a[0]) * METRES_PER_DEGREE
    return math.hypot(dx, dy)


def length(points: list[LatLon]) -> float:
    return sum(distance(a, b) for a, b in zip(points, points[1:]))


def centroid(points: list[LatLon]) -> LatLon:
    return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))


def interpolate(a: LatLon, b: LatLon, t: float) -> LatLon:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def project_on_segment(p: LatLon, a: LatLon, b: LatLon) -> tuple[float, float]:
    """Where ``p`` falls along the segment a->b: the fraction ``t`` and the distance in metres from the segment."""
    k = metres_per_degree_lon(a[0])
    bx, by = (b[1] - a[1]) * k, (b[0] - a[0]) * METRES_PER_DEGREE
    px, py = (p[1] - a[1]) * k, (p[0] - a[0]) * METRES_PER_DEGREE
    l2 = bx * bx + by * by
    if l2 < 1e-9:
        return 0.0, math.hypot(px, py)
    t = (px * bx + py * by) / l2
    tc = min(1.0, max(0.0, t))
    return t, math.hypot(px - bx * tc, py - by * tc)


def simplify(points: list[LatLon], tolerance_m: float) -> list[LatLon]:
    """Douglas-Peucker: only the points that deviate more than ``tolerance_m`` stay. On the serpentine grid
    the turns stay; on a ring the vertices that keep it on the circle."""
    if len(points) < 3:
        return list(points)
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        start, end = stack.pop()
        worst, dmax = -1, 0.0
        for i in range(start + 1, end):
            d = project_on_segment(points[i], points[start], points[end])[1]
            if d > dmax:
                dmax, worst = d, i
        if worst >= 0 and dmax > tolerance_m:
            keep[worst] = True
            stack.append((start, worst))
            stack.append((worst, end))
    return [p for p, k in zip(points, keep) if k]


def densify(points: list[LatLon], step_m: float) -> list[LatLon]:
    """Puts a point back every ``step_m`` along the track, turns included."""
    if len(points) < 2 or step_m <= 0.5:
        return list(points)
    out = [points[0]]
    for i in range(1, len(points)):
        a, b = points[i - 1], points[i]
        seg = distance(a, b)
        travelled = step_m
        while travelled < seg - 1e-9:
            out.append(interpolate(a, b, travelled / seg))
            travelled += step_m
        if distance(out[-1], b) > step_m * 0.3 or i == len(points) - 1:
            out.append(b)
    return out


def bearing(a: LatLon, b: LatLon) -> float:
    """The direction from ``a`` to ``b``, in degrees from north clockwise."""
    north = (b[0] - a[0]) * METRES_PER_DEGREE
    east = (b[1] - a[1]) * metres_per_degree_lon(a[0])
    return math.degrees(math.atan2(east, north)) % 360.0


def offset(p: LatLon, degrees: float, metres: float) -> LatLon:
    """The point ``metres`` away from ``p`` in the given direction (degrees from north clockwise)."""
    r = math.radians(degrees)
    return (p[0] + metres * math.cos(r) / METRES_PER_DEGREE, p[1] + metres * math.sin(r) / metres_per_degree_lon(p[0]))
