"""Georeferencing of a grid surveyed in local metres: from the position of its corners (or of any
points of it) in the coordinate system of the project - typed in from the GPS survey of the grid
pegs, or taken from the markers placed on the orthomosaic - a similarity (shift, rotation and,
when asked, scale) is estimated by least squares; or the grid is placed by its origin and the
azimuth of its x axis."""
from __future__ import annotations

import numpy as np

from .grids import Grid


def similarity(local: np.ndarray, world: np.ndarray, fix_scale: bool = True) -> dict:
    """The 2D similarity ``world = s R local + t`` (Umeyama, least squares) from matching points;
    with ``fix_scale`` the scale is 1 (the grid was laid out in metres). Returns ``scale``,
    ``rotation`` (degrees counter-clockwise), ``translation`` and the residual of every point."""
    local = np.asarray(local, float).reshape(-1, 2)
    world = np.asarray(world, float).reshape(-1, 2)
    if len(local) < 2 or len(local) != len(world):
        raise ValueError("at least two matching points are needed")
    ml, mw = local.mean(axis=0), world.mean(axis=0)
    a, b = local - ml, world - mw
    h = a.T @ b / len(local)
    u, s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(u @ vt))
    dmat = np.diag([1.0, d])
    r = (u @ dmat @ vt).T
    var = (a ** 2).sum() / len(local)
    scale = 1.0 if fix_scale or var <= 0 else float(np.trace(np.diag(s) @ dmat) / var)
    t = mw - scale * (r @ ml)
    fitted = (scale * (r @ local.T)).T + t
    res = np.hypot(*(fitted - world).T)
    return {"scale": scale, "rotation": float(np.degrees(np.arctan2(r[1, 0], r[0, 0]))),
            "translation": [float(t[0]), float(t[1])], "residuals": [float(v) for v in res],
            "rmse": float(np.sqrt(np.mean(res ** 2)))}


def place(grid: Grid, fit: dict, crs: str) -> Grid:
    """The grid placed in ``crs`` by a similarity (its origin the top-left corner moved, its x
    axis turned, its cell scaled)."""
    s, rot = float(fit["scale"]), float(fit["rotation"])
    t = np.asarray(fit["translation"], float)
    a = np.radians(rot)
    r = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    o = s * (r @ np.asarray(grid.origin, float)) + t
    grid.origin = (float(o[0]), float(o[1]))
    grid.rotation = (grid.rotation + rot) % 360.0
    grid.cell = grid.cell * s
    grid.crs = crs
    return grid


def place_by_origin(grid: Grid, x: float, y: float, azimuth: float, crs: str, corner: str = "lower-left") -> Grid:
    """The grid placed by the world position of one corner (``lower-left`` of the survey: the
    origin of the local metres, or ``top-left``) and the azimuth of its x axis (degrees clockwise
    from the north, as a compass: 90 = the rows run east)."""
    rot = (90.0 - azimuth) % 360.0  # counter-clockwise from the east
    a = np.radians(rot)
    h, w = grid.shape
    if corner == "lower-left":  # the top-left corner is one grid height up along the y axis of the grid
        up = np.array([-np.sin(a), np.cos(a)]) * grid.cell * h
        origin = np.array([x, y]) + up
    else:
        origin = np.array([x, y], float)
    grid.origin = (float(origin[0]), float(origin[1]))
    grid.rotation = rot
    grid.crs = crs
    return grid


def corners_local(grid: Grid) -> dict[str, tuple[float, float]]:
    """The local coordinates of the four corners of a grid in survey metres (origin lower-left)."""
    h, w = grid.shape
    L, H = w * grid.cell, h * grid.cell
    return {"lower-left": (0.0, 0.0), "lower-right": (L, 0.0), "upper-right": (L, H), "upper-left": (0.0, H)}


def local_to_corner_frame(grid: Grid) -> Grid:
    """A grid read from a file with its own local coordinates (Surfer's x/y) brought to survey
    metres with the lower-left corner at (0, 0): what the pegs of a survey grid measure."""
    h, w = grid.shape
    grid.origin = (0.0, h * grid.cell)
    grid.rotation = 0.0
    grid.crs = "LOCAL"
    return grid
