"""Special points of a block (detail photos, panoramas, video): what they add to the route.

Until the special points are ported (a later step of the plugin) :func:`with_specials` leaves the route as it
is. :func:`ground_of_grid`, the ground under a point from the terrain grid a plan carries, is the planner's.
"""
from __future__ import annotations

from typing import Optional

from .geometry import LatLon

MIN_DISTANCE_FROM_GROUND = 5.0   # never lower than this above the ground by itself
MIN_SPECIAL_HEIGHT = 5.0         # and never below five metres above the takeoff


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


def with_specials(p, specials: list[dict], ground=None, ground_takeoff: Optional[float] = None):
    """The route with its special points inside, and what came of each: not ported yet, the route as it is."""
    return p, []
