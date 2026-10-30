"""The lines of the hull, cut from the model: cross-sections at the stations, longitudinals at given
y, horizontals at given z, in ship coordinates, with the program's cut (``trimesh.intersections.mesh_plane``
and the chaining of :mod:`pyarchinit_photogrammetry.archaeo.sections`) on one copy of the mesh turned
into the ship frame, so that a series of sections reads the model once. Stored in
``<chunk data>/shipwreck/sections.json``; nothing is faired, every vertex lies on the model.

Water: the bottom seen through water is apparent. For a nadir view the true depth is about the apparent
depth times the refractive index: a point under the level is drawn at ``level - (level - z) * n``.
A first approximation: oblique rays and a moving surface are not modelled. The raw and the corrected
line are both kept.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np

from pyarchinit_photogrammetry.archaeo.sections import _join_segments
from pyarchinit_photogrammetry.core import geometry
from pyarchinit_photogrammetry.core.progress import as_progress

from . import ship
from .texts import t

NDIGITS = 4  # 0.1 mm in the file
AXES = {"station": (0, ("y", "z")), "longitudinal": (1, ("x", "z")), "horizontal": (2, ("x", "y"))}
REFRACTION = 1.34


def ship_mesh(chunk, frame: ship.Frame):
    """The model in ship coordinates, in double precision."""
    import trimesh

    mesh = ship._mesh(chunk)
    vertices = geometry.apply(frame.matrix @ chunk.transform, np.asarray(mesh.vertices, float))
    return trimesh.Trimesh(vertices, mesh.faces, process=False)


def cut(mesh, axis: int, value: float, window: tuple | None = None) -> list[np.ndarray]:
    """The polylines of the plane ``axis = value`` of a mesh in ship coordinates, as ``(N, 2)`` in the two
    other coordinates. ``window``: ``(lo, hi)`` per kept coordinate, or None for each: segments outside
    are left out."""
    import trimesh

    normal = np.zeros(3)
    normal[axis] = 1.0
    segments = trimesh.intersections.mesh_plane(mesh, normal, normal * value)
    if not len(segments):
        return []
    flat = segments[..., [i for i in range(3) if i != axis]]
    if window is not None:
        keep = np.ones(len(flat), bool)
        for k, limits in enumerate(window):
            if limits is not None:
                keep &= ((flat[..., k] >= limits[0]) & (flat[..., k] <= limits[1])).all(axis=1)
        flat = flat[keep]
    if not len(flat):
        return []
    size = max(float(np.ptp(flat.reshape(-1, 2), axis=0).max()), 1e-3)
    return _join_segments(flat, tol=size * 1e-7)


def corrected(z, level: float, n: float) -> np.ndarray:
    """The heights with the water taken out: ``level - (level - z) * n`` under the level."""
    z = np.asarray(z, float)
    return np.where(z < level, level - (level - z) * n, z)


def fingerprint(chunk, frame: ship.Frame) -> dict:
    """What the sections were cut from: they are out of date when it changes."""
    path = chunk.product_path("model")
    return {"frame": [*frame.start, *frame.end, frame.bow, frame.z0, frame.x0],
            "model_size": path.stat().st_size if path is not None and path.exists() else None}


def _path(chunk):
    return ship.folder(chunk) / "sections.json"


def load(chunk) -> dict | None:
    """The contents of ``sections.json``, None when there is none."""
    path = _path(chunk)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def is_stale(chunk, data: dict | None = None) -> bool:
    """Sections made on another frame or another model."""
    data = data if data is not None else load(chunk)
    frame = ship.frame_of(chunk)
    return data is None or frame is None or data.get("source") != fingerprint(chunk, frame)


def _save(chunk, data: dict) -> None:
    path = _path(chunk)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def build(chunk, longitudinals=(), horizontals=(), half_width: float | None = None, margin: float | None = None,
          progress=None) -> dict:
    """Cut the stations of the chunk and the longitudinals (at y) and horizontals (at z) asked, inside
    ``half_width`` of the keel (default 30 % of its length) and ``margin`` beyond its ends (default
    5 %), and write ``sections.json``. Cancelled with the progress (:class:`Cancelled`): the old file stays."""
    progress = as_progress(progress)
    ship.check_metric(chunk)
    frame = ship.need_frame(chunk)
    stations = ship.stations_of(chunk)
    if not stations and not len(longitudinals) and not len(horizontals):
        raise ship.NotReady(t("Definisci prima le stazioni.", "Define the stations first."))
    half_width = float(half_width) if half_width else 0.3 * frame.length
    margin = float(margin) if margin is not None else 0.05 * frame.length
    x_lo, x_hi = sorted((frame.x0, frame.x0 + frame.length))
    along = (min([x_lo] + [s["x"] for s in stations]) - margin, max([x_hi] + [s["x"] for s in stations]) + margin)
    across = (-half_width, half_width)
    progress(0.02, t("Lettura del modello", "Reading the model"))
    mesh = ship_mesh(chunk, frame)
    todo = ([("station", s["name"], s["x"], (across, None)) for s in stations]
            + [("longitudinal", f"y = {y:g}", float(y), (along, None)) for y in longitudinals]
            + [("horizontal", f"z = {z:g}", float(z), (along, across)) for z in horizontals])
    sections = []
    for i, (kind, name, value, window) in enumerate(todo):
        progress(0.1 + 0.85 * i / len(todo), t(f"Sezione {name}", f"Section {name}"))
        axis, axes = AXES[kind]
        lines = cut(mesh, axis, value, window)
        sections.append({"kind": kind, "name": name, "value": value, "axes": axes,
                         "lines": [np.round(p, NDIGITS).tolist() for p in lines]})
    data = {"version": 1, "source": fingerprint(chunk, frame), "built": datetime.now().isoformat(timespec="seconds"),
            "half_width": half_width, "margin": margin, "faces": len(mesh.faces), "sections": sections}
    _with_water(data, water_of(chunk))
    _save(chunk, data)
    progress(1.0, t("Sezioni pronte", "Sections ready"))
    return data


def sections_of(chunk, kind: str | None = None) -> list[dict]:
    data = load(chunk)
    return [s for s in (data or {}).get("sections", []) if kind is None or s["kind"] == kind]


# ------------------------------------------------------------------------------------- water
def water_of(chunk) -> dict | None:
    """``{"level": ship z, "level_world", "n", "on"}``; None when no level was given."""
    water = chunk.meta.get(ship.KEY, {}).get("water")
    frame = ship.frame_of(chunk)
    if water is None or frame is None:
        return None
    return {**water, "level": water["level_world"] - frame.z0}


def set_water(chunk, level: float, n: float = REFRACTION, on: bool = False) -> dict:
    """The water level (ship z), the refractive index and whether the correction is on; the sections
    that exist get their corrected lines. The level is kept as a world height: it stays with a new frame."""
    frame = ship.need_frame(chunk)
    if not 1.0 <= float(n) <= 2.0:
        raise ship.NotReady(t("L'indice di rifrazione è fra 1 e 2 (acqua: 1,34).",
                              "The refractive index lies between 1 and 2 (water: 1.34)."))
    ship.state(chunk)["water"] = {"level_world": float(level) + frame.z0, "n": float(n), "on": bool(on)}
    data = load(chunk)
    if data is not None:
        _with_water(data, water_of(chunk))
        _save(chunk, data)
    return water_of(chunk)


def _with_water(data: dict, water: dict | None) -> None:
    """``corrected`` beside the raw ``lines`` of each section (the second coordinate is z, or, for
    a horizontal, ``corrected_value`` for its height), and the water they were made with."""
    data["water"] = water if water and water.get("on") else None
    for s in data["sections"]:
        s.pop("corrected", None)
        s.pop("corrected_value", None)
        if not data["water"]:
            continue
        level, n = water["level"], water["n"]
        if s["kind"] == "horizontal":
            s["corrected_value"] = float(corrected(s["value"], level, n))
        else:
            s["corrected"] = [np.round(np.column_stack([np.asarray(p)[:, 0], corrected(np.asarray(p)[:, 1], level, n)]),
                                       NDIGITS).tolist() for p in s["lines"]]


def shown(section: dict) -> list[np.ndarray]:
    """The polylines a drawing shows: the corrected ones when the water is on, else the raw ones."""
    return [np.asarray(p, float) for p in section.get("corrected", section["lines"])]


def note(chunk_or_data) -> str:
    """What the drawing says about the water: a drawing never hides what it shows."""
    water = chunk_or_data.get("water") if isinstance(chunk_or_data, dict) else water_of(chunk_or_data)
    if not water or not water.get("on"):
        return t("Quote come misurate sul modello; nessuna correzione della rifrazione: il fondo sott'acqua è apparente.",
                 "Heights as measured on the model; no refraction correction: the bottom under water is apparent.")
    from .texts import num

    return t(f"Sotto il livello {num(water['level'], 3)} m le quote sono corrette per la rifrazione (n = {num(water['n'], 2)}), "
             "prima approssimazione per vista nadirale.",
             f"Under the level {num(water['level'], 3)} m heights are corrected for refraction (n = {num(water['n'], 2)}), "
             "a first approximation for a nadir view.")
