"""The shape of a face: how the wall stands.

* :func:`face_shape` - the model's surface within the face's window (the cells of the elevation's relief: the surface
  the program found on the face's grid, within its bounds and depth band), the plane (or, on a cylinder, the cylinder)
  fitted to it by least squares with Huber's weight; the **bulge** raster - the distance of the surface from the fit,
  signed towards the viewer, in centimetres, on the elevation's grid (``<chunk data>/masonry/<wall>/<face>/bulge.tif``,
  float32, NaN where there is no surface); the **lean** of a plane - its tilt from the vertical in degrees and cm/m and
  the azimuth towards which its top goes; the axis of a cylinder - its inclination, direction, radius.
* :func:`sections` - horizontal sections of the model at chosen heights (the program's ``plan_slice``) around the face,
  stacked in one DXF (each at its height) and one SVG, with the offsets between them measured at chosen stations along
  the face (towards the viewer, from the lowest section).
* :func:`tower_axis` - the circles fitted to horizontal sections at several heights of a round tower: the axis through
  their centres (inclination, direction), the radius and the out-of-round at each height.
* :func:`compare_surveys` - two surveys of the same wall (two chunks of the project, in the same system): the surface of
  the other model on the face's grid, the difference as a raster (cm, positive where the other survey's surface stands
  towards the viewer), its median and 90th percentile.

The measures are said, not judged: a lean or a bulge is a number with its direction, never a verdict on the wall."""
from __future__ import annotations

import json
import time

import numpy as np

from . import walls
from .fit import robust_cylinder, robust_plane, tilt
from .texts import azimuth, cm, degrees, metres, num, t
from .walls import NotReady

MAX_FIT_POINTS = 200_000
BLOCK = 512
NOT_SURFACE_SHARE = 0.02  # a fit whose rms exceeds this share of the radius (cylinder) or height (plane)...
NOT_CYLINDER_MIN, NOT_PLANE_MIN = 0.10, 0.05  # ...and these: the surface is not the one fitted
TRACE_OFF = 0.15  # the face more than this from its trace at the stations: the trace is not on the face


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _frame(entry: dict):
    from pyarchinit_photogrammetry.engine.elevation import frame_from_dict

    return frame_from_dict(entry["frame"])


def read_relief(chunk, entry: dict):
    """``(depth (h, w) float64 with NaN, rasterio profile)`` of the elevation's relief, NaN outside the face's
    bounds (without them, its raster's extent)."""
    import rasterio

    from pyarchinit_photogrammetry.engine.elevation import elevation_paths

    _ortho, relief = elevation_paths(chunk, entry)
    if not relief.exists():
        raise NotReady(t("Il prospetto della faccia non ha il suo rilievo: rifallo.",
                         "The face's elevation has lost its relief: make it again."))
    with rasterio.open(relief) as src:
        d = src.read(1).astype(np.float64)
        if src.nodata is not None and np.isfinite(src.nodata):
            d[d == src.nodata] = np.nan
        profile = src.profile.copy()
    bounds = entry.get("bounds") or walls.raster_extent(chunk, entry)
    if bounds:  # the window of the face: a cell of the border lies half outside, where the model goes on otherwise
        rows, cols = np.indices(d.shape)
        u, v = cell_uv(profile["transform"], rows, cols)
        eps = 1e-9
        outside = (u < bounds[0] - eps) | (u > bounds[2] + eps) | (v < bounds[1] - eps) | (v > bounds[3] + eps)
        d[outside] = np.nan
    return d, profile


def cell_uv(transform, rows, cols):
    """``(u, v)`` of the centres of the cells ``rows``, ``cols`` of a raster in frame metres."""
    return transform.c + (cols + 0.5) * transform.a, transform.f + (rows + 0.5) * transform.e


def _write_raster(path, values: np.ndarray, profile: dict, tags: dict) -> None:
    import rasterio

    prof = {k: profile[k] for k in ("driver", "height", "width", "transform") if k in profile}
    prof.update(count=1, dtype="float32", nodata=float("nan"), compress="deflate", BIGTIFF="IF_SAFER")
    if values.shape[0] >= 256 and values.shape[1] >= 256:
        prof.update(tiled=True, blockxsize=256, blockysize=256)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(values.astype(np.float32)[None])
        dst.update_tags(**tags)


def _fit_sample(d: np.ndarray, max_points: int, seed: int = 0):
    rows, cols = np.nonzero(np.isfinite(d))
    if len(rows) > max_points:
        pick = np.random.default_rng(seed).choice(len(rows), max_points, replace=False)
        rows, cols = rows[pick], cols[pick]
    return rows, cols


def face_shape(chunk, wid: str, eid: str | None = None, max_points: int = MAX_FIT_POINTS, progress=None) -> dict:
    """The fit, the lean (or the axis) and the bulge raster of a face; kept with the wall and in ``wall.json``."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    w, entry = walls.need_face(chunk, wid, eid)
    frame = _frame(entry)
    progress(0.0, t("Lettura del rilievo della faccia", "Reading the relief of the face"))
    d, profile = read_relief(chunk, entry)
    tr = profile["transform"]
    warning = band_contact(d, entry)  # on the whole relief, before what lies behind is set aside
    behind = second_surface(d)
    if behind is not None:  # two structures within the band: the face is the larger, the other is set aside
        aside = (d < behind["threshold"]) if behind["side"] == "back" else (d >= behind["threshold"])
        d = np.where(np.isfinite(d) & aside, np.nan, d)
    rows, cols = _fit_sample(d, max_points)
    if len(rows) < 100:
        raise NotReady(t("La faccia ha troppo poca superficie del modello per misurarne la forma.",
                         "The face has too little of the model's surface to measure its shape."))
    u, v = cell_uv(tr, rows, cols)
    pts = frame.point(u, v, d[rows, cols])
    origin = pts.mean(axis=0)
    progress(0.2, t("Adattamento della superficie", "Fitting the surface"))
    result = {"surface": frame.kind, "points": int(np.isfinite(d).sum()), "fitted": int(len(rows)),
              "resolution": float(entry.get("resolution") or abs(tr.a)), "made": _now()}
    if frame.kind == "plane":
        f = robust_plane(pts - origin)
        n = f["normal"] if f["normal"] @ frame.normal > 0 else -f["normal"]
        centre = f["centre"] + origin
        nz = float(n[2])
        lean = float(np.degrees(np.arcsin(min(1.0, abs(nz)))))
        horizontal = n[:2] / max(np.linalg.norm(n[:2]), 1e-12)
        toward = "out" if nz < 0 else "in"
        direction = horizontal if toward == "out" else -horizontal
        result.update(lean_deg=lean, lean_cm_per_m=float(100.0 * np.tan(np.radians(lean))), lean_toward=toward,
                      lean_azimuth=float(np.degrees(np.arctan2(direction[0], direction[1])) % 360.0)
                      if lean > 1e-6 else None,
                      facing_azimuth=float(np.degrees(np.arctan2(horizontal[0], horizontal[1])) % 360.0),
                      plane={"centre": centre.tolist(), "normal": n.tolist()}, rms_cm=100.0 * f["rms"])
        _v0, _v1 = entry["bounds"][1], entry["bounds"][3]
        limit = max(NOT_PLANE_MIN, NOT_SURFACE_SHARE * (_v1 - _v0))
        result.update(lean_reliable=f["rms"] <= limit, surface_limit_cm=100.0 * limit)

        def distance(p):
            return (p - centre) @ n
    else:
        f = robust_cylinder(pts, frame.axis, frame.centre, frame.radius)
        axis, centre, radius = f["axis"], f["centre"], f["radius"]
        deg, az = tilt(axis)
        result.update(axis_deg=deg, axis_azimuth=az, lean_cm_per_m=float(100.0 * np.tan(np.radians(deg))),
                      radius=radius, inside=bool(frame.inside),
                      cylinder={"centre": centre.tolist(), "axis": axis.tolist(), "radius": radius},
                      rms_cm=100.0 * f["rms"])
        limit = max(NOT_CYLINDER_MIN, NOT_SURFACE_SHARE * radius)
        result.update(lean_reliable=f["rms"] <= limit, surface_limit_cm=100.0 * limit)
        if not result["lean_reliable"]:  # oval or battered: the radius by height, from the rings of the slices
            result["rings"] = _rings_by_height(chunk, frame, entry, centre, radius)
        sign = -1.0 if frame.inside else 1.0

        def distance(p):
            q = p - centre
            radial = q - (q @ axis)[:, None] * axis
            return sign * (np.linalg.norm(radial, axis=1) - radius)
    if warning is not None:
        result["band_warning"] = warning
    if behind is not None:
        result["behind"] = behind
    progress(0.5, t("Carta dello spanciamento", "Bulge map"))
    bulge = np.full(d.shape, np.nan, np.float32)
    for r0 in range(0, d.shape[0], BLOCK):
        block = d[r0:r0 + BLOCK]
        rr, cc = np.nonzero(np.isfinite(block))
        if not len(rr):
            continue
        uu, vv = cell_uv(tr, rr + r0, cc)
        bulge[rr + r0, cc] = 100.0 * distance(frame.point(uu, vv, block[rr, cc]))
    path = walls.face_folder(chunk, w["id"], entry["id"]) / "bulge.tif"
    _write_raster(path, bulge, profile, {"units": "cm", "ppg_masonry": "bulge", "surface": frame.kind,
                                         "sign": "positive towards the viewer"})
    finite = bulge[np.isfinite(bulge)]
    lo, hi = np.percentile(finite, [0.5, 99.5])
    result.update(bulge=str(path.relative_to(chunk.data_dir)), bulge_min_cm=float(lo), bulge_max_cm=float(hi),
                  bulge_p90_cm=float(np.percentile(np.abs(finite), 90)))
    walls.set_result(chunk, w["id"], entry["id"], "shape", result)
    progress(1.0, t("Forma della faccia misurata", "Shape of the face measured"))
    return result


BAND_CELLS = 2  # a surface within this many cells of the band's edge touches it


def band_contact(d: np.ndarray, entry: dict) -> dict | None:
    """Where the surface of the face touches the edge of its depth band (within ``BAND_CELLS`` cells of it): the part
    of the wall beyond the band is missing from the face, and the fit and the bulge do not see it. ``{"front": n,
    "back": n, "share": fraction of the surface}``, or None when it does not touch (or touches in fewer than 10 cells
    and 0.1 % of them)."""
    lo, hi = entry.get("depth") or (-np.inf, np.inf)
    res = float(entry.get("resolution") or 0.0)
    ok = np.isfinite(d)
    n = int(ok.sum())
    if not n or res <= 0:
        return None
    vals = d[ok]
    front = int((vals >= hi - BAND_CELLS * res).sum()) if np.isfinite(hi) else 0
    back = int((vals <= lo + BAND_CELLS * res).sum()) if np.isfinite(lo) else 0
    if front + back < max(10, 0.001 * n):
        return None
    return {"front": front, "back": back, "share": float((front + back) / n),
            "depth": [float(lo), float(hi)]}


SECOND_SURFACE_GAP = 0.10  # two surfaces of the face more than this apart in depth are two structures


def second_surface(d: np.ndarray, bin_m: float = 0.01) -> dict | None:
    """A second surface within the face's depth band: behind it (a set-back storey, the wall behind a parapet, a
    risega) or in front of it (a board, a bush, a scaffold): two modes of the depths more than ``SECOND_SURFACE_GAP``
    apart, each holding at least 10 % of the cells, with a valley between them below half the smaller one. The face
    is the mode with more cells (on a tie within 5 %, the one in front); the other is set aside. ``{"threshold": depth
    of the valley, "side": "back" or "front" (where the surface set aside lies), "cells", "share" (of the surface set
    aside), "gap": distance of the modes (m), "comparable": the two within 5 %}`` - or None."""
    vals = d[np.isfinite(d)]
    if vals.size < 100 or np.ptp(vals) < SECOND_SURFACE_GAP:
        return None
    edges = np.arange(vals.min(), vals.max() + bin_m, bin_m)
    if len(edges) < 4:
        return None
    hist, edges = np.histogram(vals, bins=edges)
    hist = np.convolve(hist, np.array([1, 2, 3, 2, 1]) / 9.0, mode="same")
    centres = (edges[:-1] + edges[1:]) / 2
    peaks = [i for i in range(len(hist)) if hist[i] > 0 and hist[i] >= hist[max(i - 1, 0)]
             and hist[i] >= hist[min(i + 1, len(hist) - 1)]]
    gap_bins = int(round(SECOND_SURFACE_GAP / bin_m))
    best = None
    for a in peaks:
        for b in peaks:
            if b - a <= gap_bins:
                continue
            valley = a + int(np.argmin(hist[a:b + 1]))
            if hist[valley] >= 0.5 * min(hist[a], hist[b]):
                continue
            back = float((vals < centres[valley]).sum())
            front = vals.size - back
            if min(back, front) < 0.1 * vals.size:
                continue
            score = min(hist[a], hist[b]) - hist[valley]
            if best is None or score > best[0]:
                best = (score, valley, back, centres[b] - centres[a])
    if best is None:
        return None
    _score, valley, back, gap = best
    front = vals.size - back
    comparable = abs(back - front) <= 0.05 * vals.size
    side = "back" if comparable or front > back else "front"
    cells = back if side == "back" else front
    return {"threshold": float(centres[valley]), "side": side, "cells": int(cells),
            "share": float(cells / vals.size), "gap": float(gap), "comparable": bool(comparable)}


def behind_text(b: dict) -> str:
    front = b.get("side") == "front"
    where_it = "davanti alla" if front else "dietro la"
    where_en = "in front of" if front else "behind"
    side_it = "più avanti" if front else "più indietro"
    side_en = "further forward" if front else "further back"
    tie = t(" Le due superfici hanno quasi le stesse celle: è stata tenuta quella davanti.",
            " The two surfaces have almost as many cells: the one in front was kept.") if b.get("comparable") else ""
    return t(f"Attenzione: {where_it} faccia, nella fascia di profondità, c'è una seconda superficie "
             f"{num(100 * b['gap'], 0)} cm {side_it} ({num(100 * b['share'], 0)} % delle celle): lasciata fuori "
             "dall'adattamento e dalla carta dello spanciamento. Se è parte della faccia, rifai la faccia con una "
             "fascia diversa." + tie,
             f"Warning: {where_en} the face, within the depth band, there is a second surface {num(100 * b['gap'], 0)}"
             f" cm {side_en} ({num(100 * b['share'], 0)} % of the cells): left out of the fit and of the bulge map. "
             "If it is part of the face, make the face again with another band." + tie)


def band_text(w: dict) -> str:
    sides = []
    if w.get("front"):
        sides.append(t("davanti", "in front"))
    if w.get("back"):
        sides.append(t("dietro", "behind"))
    where = t(" e ", " and ").join(sides)
    return t(f"Attenzione: la superficie tocca il limite della fascia di profondità ({where}) nel "
             f"{num(100 * w['share'], 1)} % delle celle: una parte del muro può restare fuori. Rifai la faccia con "
             "una fascia più larga.",
             f"Warning: the surface touches the edge of the depth band ({where}) in {num(100 * w['share'], 1)} % of "
             "the cells: part of the wall may be left out. Make the face again with a wider band.")


def _rings_by_height(chunk, frame, entry, centre, radius, levels: int = 5) -> list[dict]:
    """The circles of the model's slices at ``levels`` heights over a cylindrical face: radius, out-of-round, arc."""
    from .faces import slice_circle

    out = []
    for h in default_heights(frame, entry, levels):
        try:
            c = slice_circle(chunk, h, near=(np.asarray(centre)[:2], radius))
        except (NotReady, ValueError):
            continue
        out.append({k: c[k] for k in ("height", "radius", "out_of_round", "coverage", "pieces")})
    return out


def surface_text(r: dict) -> str:
    """What a fit that is not the surface says: oval or battered, the radius by height; or not a plane."""
    if r.get("lean_reliable", True):
        return ""
    if r.get("surface") == "plane":
        return t(f"La superficie non è piana (rms {num(r.get('rms_cm', 0), 1)} cm oltre "
                 f"{num(r.get('surface_limit_cm', 0), 1)} cm): lo strapiombo non è affidabile; la carta dello "
                 "spanciamento è la distanza dal piano migliore.",
                 f"The surface is not a plane (rms {num(r.get('rms_cm', 0), 1)} cm over "
                 f"{num(r.get('surface_limit_cm', 0), 1)} cm): the lean is not reliable; the bulge map is the distance "
                 "from the best plane.")
    rings = r.get("rings") or []
    by_h = "; ".join(f"{num(c['height'], 2)}: {metres(c['radius'])} ({t('fuori tondo', 'out-of-round')} "
                     f"{cm(c['out_of_round'])})" for c in rings)
    return t(f"La superficie non è un cilindro (ovale o a scarpa: rms {num(r.get('rms_cm', 0), 1)} cm oltre "
             f"{num(r.get('surface_limit_cm', 0), 1)} cm): l'inclinazione dell'asse non è affidabile; la carta dello "
             "spanciamento è la distanza dal cilindro migliore." + (f" Raggio per quota: {by_h}." if by_h else ""),
             f"The surface is not a cylinder (oval or battered: rms {num(r.get('rms_cm', 0), 1)} cm over "
             f"{num(r.get('surface_limit_cm', 0), 1)} cm): the axis' inclination is not reliable; the bulge map is the "
             "distance from the best cylinder." + (f" Radius by height: {by_h}." if by_h else ""))


def signed_cm(value) -> str:
    """``+7,0 cm`` / ``-0,3 cm``."""
    if value is None:
        return "-"
    x = float(value) * 100.0
    return f"{'-' if x < 0 and round(x, 1) != 0 else '+'}{num(abs(x), 1)} cm"


def summary_text(r: dict) -> str:
    """The shape of a face in words, for the pane and the log."""
    if not r:
        return t("Forma non ancora misurata.", "Shape not measured yet.")
    lines = []
    if r.get("surface") == "plane":
        deg = r.get("lean_deg") or 0.0
        if deg < 0.05:
            lines.append(t("A piombo (entro 0,05°).", "Plumb (within 0.05°)."))
        elif r.get("lean_toward") == "out":
            lines.append(t(f"Strapiombo {degrees(deg)} ({num(r['lean_cm_per_m'], 1)} cm/m) verso "
                           f"{azimuth(r.get('lean_azimuth'))} (verso chi guarda).",
                           f"Overhang {degrees(deg)} ({num(r['lean_cm_per_m'], 1)} cm/m) towards "
                           f"{azimuth(r.get('lean_azimuth'))} (towards the viewer)."))
        else:
            lines.append(t(f"Fuori piombo all'indietro {degrees(deg)} ({num(r['lean_cm_per_m'], 1)} cm/m) verso "
                           f"{azimuth(r.get('lean_azimuth'))} (lontano da chi guarda).",
                           f"Leaning back {degrees(deg)} ({num(r['lean_cm_per_m'], 1)} cm/m) towards "
                           f"{azimuth(r.get('lean_azimuth'))} (away from the viewer)."))
    else:
        lines.append(t(f"Asse del cilindro inclinato {degrees(r.get('axis_deg'))} ({num(r['lean_cm_per_m'], 1)} cm/m) "
                       f"verso {azimuth(r.get('axis_azimuth'))}; raggio {metres(r.get('radius'))}.",
                       f"Axis of the cylinder inclined {degrees(r.get('axis_deg'))} ({num(r['lean_cm_per_m'], 1)} "
                       f"cm/m) towards {azimuth(r.get('axis_azimuth'))}; radius {metres(r.get('radius'))}."))
    if not r.get("lean_reliable", True):
        lines[-1] = lines[-1].rstrip(".") + t(" (non affidabile).", " (not reliable).")
        lines.append(surface_text(r))
    if r.get("band_warning"):
        lines.append(band_text(r["band_warning"]))
    if r.get("behind"):
        lines.append(behind_text(r["behind"]))
    lo, hi = r.get("bulge_min_cm"), r.get("bulge_max_cm")
    if lo is not None:
        lines.append(t(f"Spanciamento (scarto dalla superficie adattata, verso chi guarda): da {signed_cm(lo / 100)} "
                       f"a {signed_cm(hi / 100)}, rms {num(r.get('rms_cm', 0), 1)} cm su {r.get('points', 0)} celle.",
                       f"Bulge (distance from the fitted surface, towards the viewer): from {signed_cm(lo / 100)} "
                       f"to {signed_cm(hi / 100)}, rms {num(r.get('rms_cm', 0), 1)} cm over {r.get('points', 0)} "
                       f"cells."))
    return "\n".join(lines)


# ------------------------------------------------------------------ sections
def _v_at_height(frame, u: float, h: float) -> float:
    """The ``v`` of the face at ``u`` whose surface point (``d`` = 0) is at the world height ``h``."""
    p0 = frame.point(np.array([u]), np.array([0.0]), np.array([0.0]))[0]
    p1 = frame.point(np.array([u]), np.array([1.0]), np.array([0.0]))[0]
    dz = p1[2] - p0[2]
    if abs(dz) < 1e-6:
        raise NotReady(t("La faccia è orizzontale: le sezioni orizzontali non la tagliano.",
                         "The face is horizontal: horizontal sections do not cut it."))
    return float((h - p0[2]) / dz)


def _station_ray(frame, u: float, h: float):
    """The horizontal line of the station ``u`` at the height ``h``: a point on the face's surface (``d`` = 0) and the
    horizontal direction towards the viewer."""
    v = _v_at_height(frame, u, h)
    p = frame.point(np.array([u]), np.array([v]), np.array([0.0]))[0]
    view = frame.view(np.array([u]), np.array([v]))[0]
    hor = np.array([view[0], view[1]])
    norm = np.linalg.norm(hor)
    if norm < 1e-9:
        raise NotReady(t("La faccia guarda in alto o in basso: niente stazioni orizzontali.",
                         "The face looks up or down: no horizontal stations."))
    return p[:2], hor / norm


def _crossings(lines: list[np.ndarray], p: np.ndarray, n: np.ndarray, reach: float) -> list[float]:
    """The distances along ``n`` from ``p`` where the line through them meets ``lines`` (within ``reach``)."""
    out = []
    for line in lines:
        a, b = line[:-1], line[1:]
        e = b - a
        den = n[0] * (-e[:, 1]) - n[1] * (-e[:, 0])
        ok = np.abs(den) > 1e-12
        if not ok.any():
            continue
        w = a - p
        s = (w[:, 0] * (-e[:, 1]) - w[:, 1] * (-e[:, 0])) / np.where(ok, den, 1.0)
        tt = (n[0] * w[:, 1] - n[1] * w[:, 0]) / np.where(ok, den, 1.0)
        hit = ok & (tt >= -1e-9) & (tt <= 1 + 1e-9) & (np.abs(s) <= reach)
        out += [float(x) for x in s[hit]]
    return out


def _crossing(lines, p, n, reach, surface: float | None = None, front: float = np.inf) -> float | None:
    """The crossing of the face's own surface at a station: the one nearest to the depth ``surface`` the face's relief
    has there (within 10 cm), else the one nearest to the viewer within the band (not beyond ``front``) - never a
    surface behind the face, such as the back of a thick wall traced on its axis."""
    cands = [c for c in _crossings(lines, p, n, reach) if c <= front + 0.02]
    if not cands:
        return None
    if surface is not None and np.isfinite(surface):
        best = min(cands, key=lambda c: abs(c - surface))
        if abs(best - surface) <= 0.10:
            return best
    return max(cands)


def _relief_at(d: np.ndarray, transform, u: float, v: float) -> float | None:
    col = int(np.floor((u - transform.c) / transform.a))
    row = int(np.floor((v - transform.f) / transform.e))
    if 0 <= row < d.shape[0] and 0 <= col < d.shape[1]:
        win = d[max(row - 1, 0):row + 2, max(col - 1, 0):col + 2]
        if np.isfinite(win).any():
            return float(np.nanmedian(win))
    return None


def _band(frame, entry: dict, h: float, reach: float):
    """The plan region of the face at the height ``h``: its ``u`` range, from ``reach`` behind its depth band to
    ``reach`` in front of it."""
    from shapely.geometry import Polygon

    u0, _v0, u1, _v1 = entry["bounds"]
    lo, hi = entry.get("depth") or (-0.5, 0.5)
    lo, hi = lo - reach, hi + reach
    if frame.kind == "cylinder" and frame.inside:
        hi = min(hi, frame.radius - 1e-3)
    us = np.linspace(u0, u1, 2 if frame.kind == "plane" else 90)
    vs = np.array([_v_at_height(frame, u, h) for u in us])
    front = frame.point(us, vs, np.full(len(us), hi))[:, :2]
    back = frame.point(us, vs, np.full(len(us), lo))[:, :2]
    return Polygon(np.vstack([front, back[::-1]])).buffer(0)


def _clip(lines, region) -> list[np.ndarray]:
    from shapely.geometry import LineString

    out = []
    for line in lines:
        if len(line) < 2:
            continue
        g = LineString(line).intersection(region)
        for part in getattr(g, "geoms", [g]):
            if not part.is_empty and part.geom_type == "LineString" and len(part.coords) > 1:
                out.append(np.asarray(part.coords)[:, :2])
    return out


def default_heights(frame, entry: dict, n: int = 4) -> list[float]:
    u0, v0, u1, v1 = entry["bounds"]
    um = (u0 + u1) / 2
    z = frame.point(np.array([um, um]), np.array([v0, v1]), np.zeros(2))[:, 2]
    lo, hi = float(z.min()), float(z.max())
    return [lo + f * (hi - lo) for f in np.linspace(0.1, 0.9, n)]


def default_stations(entry: dict, n: int = 3) -> list[float]:
    u0, _v0, u1, _v1 = entry["bounds"]
    return [u0 + (u1 - u0) * (2 * k + 1) / (2 * n) for k in range(n)]


COLOURS = ("#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#17becf", "#7f7f7f")


def sections(chunk, wid: str, eid: str | None = None, heights=None, stations=None, reach: float = 1.0,
             progress=None) -> dict:
    """Horizontal sections of the model at ``heights`` (world) around the face, stacked in ``sezioni.dxf`` (each line
    at its height, layer ``SEZIONE_<i>`` from the lowest; the stations on ``STAZIONI``, the offsets on ``TESTI``) and
    ``sezioni.svg`` (the plan), with the offsets between them at ``stations`` (``u`` along the face): the distance of
    each section from the lowest one along the horizontal towards the viewer (metres, positive towards the viewer)."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    from .faces import slice_rings

    progress = as_progress(progress)
    w, entry = walls.need_face(chunk, wid, eid)
    frame = _frame(entry)
    heights = sorted(float(h) for h in (heights if heights else default_heights(frame, entry)))
    stations = [float(s) for s in (stations if stations else default_stations(entry))]
    cuts = []
    for i, h in enumerate(heights):
        progress(0.8 * i / len(heights), t(f"Sezione a quota {num(h, 3)}", f"Section at height {num(h, 3)}"))
        cuts.append(_clip(slice_rings(chunk, h), _band(frame, entry, h, reach)))
    try:
        relief, profile = read_relief(chunk, entry)
        rtr = profile["transform"]
    except NotReady:
        relief = rtr = None
    front = float((entry.get("depth") or (-np.inf, np.inf))[1])
    out_stations, chosen = [], []
    for u in stations:
        found = []
        for h, lines in zip(heights, cuts):
            p, n = _station_ray(frame, u, h)
            surface = None if relief is None else _relief_at(relief, rtr, u, _v_at_height(frame, u, h))
            s_ = _crossing(lines, p, n, reach, surface, front)
            found.append((h, p, n, s_))
        ref = next(((f[0], f[3]) for f in found if f[3] is not None), (None, None))
        if ref[1] is not None:  # the face at the lowest section: where a trace on the face would be
            chosen.append(ref[1])
        offsets = []
        for h, p, n, s_ in found:
            pos = None if s_ is None else p + s_ * n
            offsets.append({"height": h, "d": s_, "offset": None if s_ is None or ref[1] is None else s_ - ref[1],
                            "point": None if pos is None else [float(pos[0]), float(pos[1]), h]})
        out_stations.append({"u": u, "offsets": offsets, "reference_height": ref[0]})
    warnings = []
    if chosen and float(np.median(np.abs(chosen))) > TRACE_OFF:
        warnings.append(t(f"La faccia sta in media {num(100 * float(np.median(chosen)), 0)} cm dalla traccia: la "
                          "traccia non è sulla faccia (forse sull'asse del muro). Gli scarti sono misurati sulla "
                          "superficie che la faccia vede.", f"The face lies {num(100 * float(np.median(chosen)), 0)} "
                          "cm from the trace on average: the trace is not on the face (perhaps on the wall's axis). "
                          "The "
                          "offsets are measured on the surface the face sees."))
    folder = walls.face_folder(chunk, w["id"], entry["id"])
    folder.mkdir(parents=True, exist_ok=True)
    progress(0.85, t("Scrittura di DXF e SVG", "Writing the DXF and the SVG"))
    dxf = _sections_dxf(folder / "sezioni.dxf", heights, cuts, out_stations, frame, reach)
    svg = _sections_svg(folder / "sezioni.svg", heights, cuts, out_stations, frame, reach, entry.get("name", ""))
    result = {"heights": heights, "stations": out_stations, "reach": reach, "made": _now(), "warnings": warnings,
              "dxf": str(dxf.relative_to(chunk.data_dir)), "svg": str(svg.relative_to(chunk.data_dir)),
              "lines": [len(c) for c in cuts]}
    (folder / "sezioni.json").write_text(json.dumps(result, indent=1))
    walls.set_result(chunk, w["id"], entry["id"], "sections", result)
    progress(1.0, t("Sezioni scritte", "Sections written"))
    return result


def _station_lines(frame, stations, h, reach):
    out = []
    for st in stations:
        p, n = _station_ray(frame, st["u"], h)
        out.append((st, p - reach * n, p + reach * n))
    return out


def _sections_dxf(path, heights, cuts, stations, frame, reach):
    import ezdxf

    doc = ezdxf.new("R2010")
    doc.units = ezdxf.units.M
    msp = doc.modelspace()
    for i, (h, lines) in enumerate(zip(heights, cuts)):
        name = f"SEZIONE_{i}"
        doc.layers.add(name, color=(i % 6) + 1).description = f"z = {h:.3f}"
        for line in lines:
            msp.add_polyline3d([(float(x), float(y), h) for x, y in line], dxfattribs={"layer": name})
    doc.layers.add("STAZIONI", color=7)
    doc.layers.add("TESTI", color=7)
    size = max(0.03, reach / 12)
    for st, a, b in _station_lines(frame, stations, heights[0], reach):
        msp.add_line((a[0], a[1], heights[0]), (b[0], b[1], heights[0]), dxfattribs={"layer": "STAZIONI"})
        label = ", ".join(f"{num(o['height'], 2)}: {signed_cm(o['offset'])}" for o in st["offsets"])
        msp.add_text(f"u {num(st['u'], 2)} - {label}", height=size,
                     dxfattribs={"layer": "TESTI"}).set_placement((b[0], b[1], heights[0]))
    doc.saveas(path)
    return path


def _nice(x: float) -> float:
    """The largest of 1, 2, 5 x 10^n not above ``x`` (1 at least)."""
    if x <= 1:
        return 1.0
    e = 10 ** np.floor(np.log10(x))
    return float(max(m * e for m in (1, 2, 5) if m * e <= x))


def _sections_svg(path, heights, cuts, stations, frame, reach, title):
    """The sections drawn along the face: ``u`` across the page, the distance towards the viewer upwards, exaggerated
    (the offsets of a wall are centimetres on metres); the stations as dashed lines with the offset of the highest
    section from the lowest."""
    from xml.sax.saxutils import escape

    drawn = []  # per section: polylines of (u, d)
    for h, lines in zip(heights, cuts):
        parts_h = []
        for line in lines:
            uvd = frame.coords(np.column_stack([line, np.full(len(line), h)]))
            ud = uvd[:, [0, 2]]
            if frame.kind == "cylinder":  # a line across the seam of the development is cut there
                jumps = np.nonzero(np.abs(np.diff(ud[:, 0])) > np.pi * frame.radius)[0]
                parts_h += [p for p in np.split(ud, jumps + 1) if len(p) > 1]
            else:
                parts_h.append(ud)
        drawn.append(parts_h)
    allp = np.vstack([p for d in drawn for p in d] + [np.array([[st["u"], 0.0] for st in stations])])
    u0, u1 = float(allp[:, 0].min()), float(allp[:, 0].max())
    d0, d1 = (np.percentile(allp[:, 1], [0.5, 99.5]) if len(allp) > 10 else (allp[:, 1].min(), allp[:, 1].max()))
    d0, d1 = float(min(d0, 0.0)), float(max(d1, 0.0))
    width = 1000.0
    k = (width - 80) / max(u1 - u0, 1e-3)
    exaggeration = _nice(0.3 * (u1 - u0) / max(d1 - d0, 1e-4))
    pad = 0.15 * (d1 - d0 + 1e-3)
    d0, d1 = d0 - pad, d1 + pad
    plot = (d1 - d0) * k * exaggeration
    height = plot + 120 + 22 * (len(heights) + 1)
    axes = t("lungo la faccia (u) e verso chi guarda (d), d esagerata", "along the face (u) and towards the viewer "
             "(d), d exaggerated")

    def x(u):
        return (u - u0) * k + 40

    def y(d):
        return (d1 - d) * k * exaggeration + 50

    parts = ['<?xml version="1.0" encoding="UTF-8"?>',
             f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
             f'viewBox="0 0 {width:.0f} {height:.0f}" font-family="sans-serif" font-size="12">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="40" y="24" font-size="14">{escape(t("Sezioni orizzontali", "Horizontal sections"))}'
             f' - {escape(title)}</text>',
             f'<text x="40" y="40">{escape(axes)} x{num(exaggeration, 0)}</text>']
    parts.append(f'<line x1="{x(u0):.2f}" y1="{y(0):.2f}" x2="{x(u1):.2f}" y2="{y(0):.2f}" stroke="#aaa"/>')
    for i, lines in enumerate(drawn):
        colour = COLOURS[i % len(COLOURS)]
        for line in lines:
            pts = " ".join(f"{x(p[0]):.2f},{y(p[1]):.2f}" for p in line)
            parts.append(f'<polyline fill="none" stroke="{colour}" stroke-width="1.6" points="{pts}"/>')
    for st in stations:
        parts.append(f'<line x1="{x(st["u"]):.2f}" y1="50" x2="{x(st["u"]):.2f}" y2="{plot + 50:.2f}" stroke="#444" '
                     'stroke-dasharray="4 3"/>')
        last = next((o for o in reversed(st["offsets"]) if o["offset"] is not None), None)
        if last is not None:
            parts.append(f'<text x="{x(st["u"]) + 4:.2f}" y="{plot + 66:.2f}">u {escape(num(st["u"], 2))}: '
                         f'{escape(signed_cm(last["offset"]))}</text>')
    yy = plot + 100
    for i, h in enumerate(heights):
        colour = COLOURS[i % len(COLOURS)]
        parts.append(f'<line x1="40" y1="{yy - 4:.0f}" x2="70" y2="{yy - 4:.0f}" stroke="{colour}" stroke-width="3"/>')
        parts.append(f'<text x="78" y="{yy:.0f}">{escape(t("quota", "height"))} {escape(num(h, 3))} m</text>')
        yy += 22
    note = t("Scarti alle stazioni: dalla sezione più bassa a quella più alta, verso chi guarda",
             "Offsets at the stations: from the lowest section to the highest, towards the viewer")
    parts.append(f'<text x="40" y="{yy:.0f}">{escape(note)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def sections_text(r: dict) -> str:
    lines = [t(f"Sezioni a {len(r['heights'])} quote: ", f"Sections at {len(r['heights'])} heights: ")
             + ", ".join(num(h, 2) for h in r["heights"])]
    for st in r["stations"]:
        ref = st.get("reference_height")
        offs = [o for o in st["offsets"] if o["height"] != ref]
        line = f"u {num(st['u'], 2)} m: " + ", ".join(f"{num(o['height'], 2)} {signed_cm(o['offset'])}" for o in offs)
        if ref is None:
            line = f"u {num(st['u'], 2)} m: " + t("nessuna sezione incontra la faccia",
                                                   "no section meets the face")
        elif ref != r["heights"][0]:
            line += t(f" (dalla sezione a quota {num(ref, 2)}: la più bassa non incontra la faccia)",
                      f" (from the section at {num(ref, 2)}: the lowest does not meet the face)")
        lines.append(line)
    lines += list(r.get("warnings") or [])
    return "\n".join(lines)


# ------------------------------------------------------------------ tower
def tower_axis(chunk, wid: str, heights=None, levels: int = 5, progress=None) -> dict:
    """The axis of a round tower from the circles fitted to horizontal sections at ``heights`` (world; default
    ``levels`` heights over the height of its cylindrical face): inclination from the vertical, direction of the
    top, cm/m; per height the centre, the radius, the out-of-round (spread of the radius, 0.5th to 99.5th
    percentile) and the rms of the fit. Kept with the wall."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    from .faces import slice_circle
    from .fit import line_through

    progress = as_progress(progress)
    w = walls.need(chunk, wid)
    near = None
    cyl = next((walls.with_bounds(chunk, e) for e in (walls.elevation(chunk, f) for f in w["faces"])
                if e is not None and (e.get("frame") or {}).get("kind") == "cylinder"), None)
    if cyl is not None:
        frame = _frame(cyl)
        near = (frame.centre[:2], frame.radius)
        if not heights:
            heights = default_heights(frame, cyl, levels)
    if not heights:
        raise NotReady(t("Dai le quote delle sezioni (il muro non ha una faccia cilindrica).",
                         "Give the heights of the sections (the wall has no cylindrical face)."))
    out, skipped = [], []
    for i, h in enumerate(sorted(float(x) for x in heights)):
        progress(i / len(heights), t(f"Cerchio a quota {num(h, 3)}", f"Circle at height {num(h, 3)}"))
        try:
            c = slice_circle(chunk, h, near=near)
        except NotReady:
            skipped.append(h)
            continue
        near = (np.asarray(c["centre"]), c["radius"])
        out.append(c)
    if len(out) < 2:
        raise NotReady(t("Servono almeno due quote con una sezione del modello per l'asse.",
                         "At least two heights with a section of the model are needed for the axis."))
    centres = np.array([[*c["centre"], c["height"]] for c in out])
    line = line_through(centres)
    deg, az = tilt(line["axis"])
    span = float(centres[:, 2].max() - centres[:, 2].min())
    result = {"levels": out, "skipped": skipped, "axis": line["axis"].tolist(), "point": line["point"].tolist(),
              "axis_deg": deg, "axis_azimuth": az, "lean_cm_per_m": float(100.0 * np.tan(np.radians(deg))),
              "offset": float(np.tan(np.radians(deg)) * span), "span": span,
              "centre_residuals": [float(x) for x in line["residuals"]], "made": _now()}
    walls.set_result(chunk, w["id"], None, "axis", result)
    progress(1.0, t("Asse della torre misurato", "Axis of the tower measured"))
    return result


def axis_text(a: dict) -> str:
    lines = [t(f"Asse da {len(a['levels'])} sezioni: inclinato {degrees(a['axis_deg'])} "
               f"({num(a['lean_cm_per_m'], 1)} cm/m) verso {azimuth(a['axis_azimuth'])}, "
               f"{cm(a['offset'])} su {metres(a['span'])} di altezza.",
               f"Axis from {len(a['levels'])} sections: inclined {degrees(a['axis_deg'])} "
               f"({num(a['lean_cm_per_m'], 1)} cm/m) towards {azimuth(a['axis_azimuth'])}, "
               f"{cm(a['offset'])} over {metres(a['span'])} of height.")]
    for c in a["levels"]:
        arc = c.get("coverage")
        cov_it = f", arco {num(arc, 0)}° in {c.get('pieces', 1)} pezzi" if arc is not None else ""
        cov_en = f", arc {num(arc, 0)}° in {c.get('pieces', 1)} pieces" if arc is not None else ""
        lines.append(t(f"quota {num(c['height'], 2)}: raggio {metres(c['radius'])}, fuori tondo "
                       f"{cm(c['out_of_round'])}{cov_it}",
                       f"height {num(c['height'], 2)}: radius {metres(c['radius'])}, out-of-round "
                       f"{cm(c['out_of_round'])}{cov_en}"))
    return "\n".join(lines)


# ------------------------------------------------------------------ two surveys
def _stem(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(text)) or "rilievo"


def compare_surveys(chunk, wid: str, eid: str | None, other, progress=None) -> dict:
    """The surface of the model of ``other`` (another chunk of the project, in the same system as ``chunk``) on the
    face's grid and its difference from this survey's: ``confronto_<other>.tif`` (cm, positive where the other
    surface stands towards the viewer), the median and the 90th percentile of the size of the difference."""
    import threading

    from pyarchinit_photogrammetry.core.progress import as_progress
    from pyarchinit_photogrammetry.engine.elevation import FrameSurface
    from pyarchinit_photogrammetry.engine.visibility import raycaster_for_chunk

    progress = as_progress(progress)
    if other is chunk:
        raise NotReady(t("Scegli un altro rilievo (un altro chunk).", "Choose another survey (another chunk)."))
    if (other.crs or "LOCAL") != (chunk.crs or "LOCAL"):
        raise NotReady(t(f"I due rilievi sono in sistemi diversi ({chunk.crs}, {other.crs}): allineali prima.",
                         f"The two surveys are in different systems ({chunk.crs}, {other.crs}): align them first."))
    if other.product_path("model") is None:
        raise NotReady(t(f"Il chunk «{other.label}» non ha un modello.", f"The chunk «{other.label}» has no model."))
    w, entry = walls.need_face(chunk, wid, eid)
    frame = _frame(entry)
    d, profile = read_relief(chunk, entry)
    res = float(entry["resolution"])
    progress(0.05, t(f"Superficie di «{other.label}» sulla faccia", f"Surface of «{other.label}» on the face"))
    tr = profile["transform"]
    bounds = entry["bounds"]
    if entry.get("_bounds_from_raster"):  # bounds that give FrameSurface the raster's own grid
        h0, w0 = d.shape
        bounds = (tr.c + 0.5 * res, tr.f - (h0 - 1.5) * res, tr.c + (w0 - 1.5) * res, tr.f - 0.5 * res)
    surf = FrameSurface(other, raycaster_for_chunk(other), frame, res, threading.Lock(),
                        tuple(entry.get("depth") or (-np.inf, np.inf)), bounds)
    if surf.shape != d.shape or abs(surf.origin[0] - tr.c) > res / 100 or abs(surf.origin[1] - tr.f) > res / 100:
        raise NotReady(t("La griglia del prospetto non corrisponde più alla faccia: rifai il prospetto.",
                         "The grid of the elevation no longer matches the face: make the elevation again."))
    other_d = np.full(d.shape, np.nan)
    h, wdt = d.shape
    tiles = [(r, c) for r in range(0, h, BLOCK) for c in range(0, wdt, BLOCK)]
    for i, (r0, c0) in enumerate(tiles):
        th, tw = min(BLOCK, h - r0), min(BLOCK, wdt - c0)
        other_d[r0:r0 + th, c0:c0 + tw] = surf.tile(r0, c0, th, tw)
        progress(0.05 + 0.85 * (i + 1) / len(tiles), t("Confronto", "Comparison"))
    diff = (other_d - d) * 100.0
    ok = np.isfinite(diff)
    if ok.sum() < 10:
        raise NotReady(t(f"Il modello di «{other.label}» non copre la faccia.",
                         f"The model of «{other.label}» does not cover the face."))
    path = walls.face_folder(chunk, w["id"], entry["id"]) / f"confronto_{_stem(other.label)}.tif"
    _write_raster(path, diff, profile, {"units": "cm", "ppg_masonry": "survey difference", "other": other.label,
                                        "sign": "positive where the other survey stands towards the viewer"})
    vals = diff[ok]
    result = {"other": other.label, "other_key": getattr(other, "key", None), "raster": str(path.relative_to(
        chunk.data_dir)), "median_cm": float(np.median(vals)), "p90_cm": float(np.percentile(np.abs(vals), 90)),
        "mean_cm": float(vals.mean()), "cells": int(ok.sum()), "coverage": float(ok.sum() / max(np.isfinite(d).sum(),
                                                                                                  1)), "made": _now()}
    surveys = walls.results(chunk, w["id"], entry["id"]).get("surveys") or {}
    surveys[other.label] = result
    walls.set_result(chunk, w["id"], entry["id"], "surveys", surveys)
    progress(1.0, t("Confronto fatto", "Comparison done"))
    return result


def compare_text(r: dict) -> str:
    return t(f"Confronto con «{r['other']}»: mediana {signed_cm(r['median_cm'] / 100)}, p90 della differenza "
             f"{cm(r['p90_cm'] / 100)} su {r['cells']} celle ({num(100 * r['coverage'], 0)} % della faccia); "
             f"positivo = l'altro rilievo sta verso chi guarda.",
             f"Comparison with «{r['other']}»: median {signed_cm(r['median_cm'] / 100)}, p90 of the difference "
             f"{cm(r['p90_cm'] / 100)} over {r['cells']} cells ({num(100 * r['coverage'], 0)} % of the face); "
             f"positive = the other survey stands towards the viewer.")
