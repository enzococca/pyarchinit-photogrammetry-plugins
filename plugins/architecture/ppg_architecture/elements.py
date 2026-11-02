"""The elements of a chunk: what the model says (the engine), the solid (the worker, CadQuery in its own process), how
far the model is from the solid (the fit), the files.

Everything of an element is in ``<chunk data>/architecture/<id>/`` (``solid/solid.step``, ``solid/solid.stl``,
``model.ply`` to show it in the 3D view, ``element.json`` with everything) and its entry in
``chunk.meta["architecture"]["elements"]``. The chunk's model, transform and products are never changed.

The frames. The solid is written in the chunk's *export frame*: world coordinates (the chunk's CRS, metres) minus a
round ``origin`` (the floor of the model's lowest corner, chosen once per chunk and kept in
``chunk.meta["architecture"]["origin"]``): every STEP of a chunk lines up with the others and with the mesh, with
small coordinates, and the IFC uses the same origin in its ``IfcMapConversion``. Each element has its own frame (the
``placement``, a rigid 4 x 4 matrix to the export frame): a column's z is its axis, an arch's xy is the section plane,
a moulding's path is in the export frame itself.

Every element records ``source`` (chunk, region, path or section, heights), ``params`` (every number, metres and
degrees), ``profile`` (the measured points, kept), ``geometry`` (what the solid is made of, in its frame), ``fit``
(median, 90th percentile and maximum distance of the model's points to the solid, and how many points), ``solid``
(volume, area, bounding box from OpenCascade), ``files``, ``made``, ``name``, ``kind``, ``notes``.
"""
from __future__ import annotations

import copy
import json
import shutil
import time
import uuid
from pathlib import Path

import numpy as np

from . import engine
from .texts import t

KEY = "architecture"
FOLDER = "architecture"
TOLERANCE = 0.0005  # of the STL: half a millimetre
COLOURS = ((0.85, 0.55, 0.25), (0.35, 0.6, 0.85), (0.55, 0.75, 0.35), (0.8, 0.4, 0.55), (0.6, 0.5, 0.8),
           (0.9, 0.75, 0.3))
WORKER = None  # (python, script) of another worker for the tests; None: the environment's


class NotReady(RuntimeError):
    """Something the element needs is missing: the message says what."""


# ====================================================================== the registry
def state(chunk) -> dict:
    return chunk.meta.setdefault(KEY, {})


def elements(chunk) -> list[dict]:
    return list((chunk.meta.get(KEY) or {}).get("elements") or [])


def entry(chunk, eid: str) -> dict | None:
    return next((e for e in elements(chunk) if e["id"] == eid or e.get("name") == eid), None)


def need(chunk, eid: str) -> dict:
    e = entry(chunk, eid)
    if e is None:
        raise NotReady(t(f"Nessun elemento «{eid}».", f"No element '{eid}'."))
    return e


def folder(chunk) -> Path:
    return chunk.data_dir / FOLDER


def element_dir(chunk, e: dict) -> Path:
    return folder(chunk) / e["id"]


def path_of(chunk, e: dict, what: str) -> Path | None:
    rel = (e.get("files") or {}).get(what)
    return None if rel is None else element_dir(chunk, e) / rel


def _save(chunk, e: dict) -> None:
    st = state(chunk)
    items = [x for x in st.get("elements") or [] if x["id"] != e["id"]]
    old = next((i for i, x in enumerate(st.get("elements") or []) if x["id"] == e["id"]), None)
    if old is None:
        items.append(e)
    else:
        items.insert(old, e)
    st["elements"] = items
    d = element_dir(chunk, e)
    d.mkdir(parents=True, exist_ok=True)
    record = dict(e, origin=origin(chunk).tolist(), crs=chunk.crs,
                  frames=t("Coordinate = coordinate del sistema del chunk meno «origin»; «placement» porta il "
                           "sistema dell'elemento in quello.", "Coordinates = the chunk's CRS coordinates minus "
                           "'origin'; 'placement' takes the element's frame to them."))
    (d / "element.json").write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")


def users_of(chunk, eid: str) -> list[dict]:
    """The openings whose top is the arch ``eid``."""
    return [x for x in elements(chunk) if x["kind"] == "opening" and (x.get("params") or {}).get("arch") == eid]


def remove(chunk, eid: str) -> bool:
    e = entry(chunk, eid)
    if e is None:
        return False
    users = users_of(chunk, e["id"])
    if users:
        names = ", ".join(f"«{x['name']}»" for x in users)
        raise NotReady(t(f"L'arco «{e['name']}» chiude in alto {names}: rimuovi prima l'apertura, o rifalla con "
                         "l'architrave piano.", f"The arch '{e['name']}' closes the top of {names}: remove the opening "
                         "first, or make it again with a flat lintel."))
    shutil.rmtree(element_dir(chunk, e), ignore_errors=True)
    state(chunk)["elements"] = [x for x in elements(chunk) if x["id"] != e["id"]]
    return True


def check_fresh(entries) -> None:
    """:class:`NotReady` when an element is waiting to be made again (an opening whose arch changed)."""
    stale = [e["name"] for e in entries if e.get("stale")]
    if stale:
        raise NotReady(t(f"Da ricostruire (l'arco che le chiude è cambiato): {', '.join(stale)}. Usa Ricostruisci.",
                         f"To be rebuilt (the arch that closes them changed): {', '.join(stale)}. Use Rebuild."))


def prune_orphans(chunk, older_than: float = 3600.0) -> list[str]:
    """The element folders without an entry in the chunk (left by a run that stopped before the project was saved)
    removed when they are older than ``older_than`` seconds (a solid being made has its folder before its entry),
    each one logged. Returns their names."""
    import logging

    base = folder(chunk)
    if not base.exists():
        return []
    known = {e["id"] for e in elements(chunk)}
    gone = []
    for d in base.iterdir():
        if not d.is_dir() or d.name.startswith(".") or d.name in known:
            continue
        if time.time() - d.stat().st_mtime < older_than:
            continue
        shutil.rmtree(d, ignore_errors=True)
        logging.getLogger(__name__).info("architecture: removed the folder %s, not an element of the chunk", d)
        gone.append(d.name)
    return gone


def _new_name(chunk, kind: str) -> str:
    from .texts import kind_title

    base = kind_title(kind)
    taken = {e.get("name") for e in elements(chunk)}
    k = 1
    while f"{base} {k}" in taken:
        k += 1
    return f"{base} {k}"


# ====================================================================== the model of the chunk
def origin(chunk) -> np.ndarray:
    """The chunk's export origin (see the module): chosen once, from the model, with the chunk's transform and system
    of that moment (:func:`check_frame`)."""
    st = state(chunk)
    if st.get("origin") is None or (not st.get("elements") and not frame_matches(chunk)):
        v = _world_vertices(chunk)
        st["origin"] = np.floor(v.min(axis=0)).tolist()
        st["frame"] = {"transform": np.asarray(chunk.transform, float).tolist(), "crs": chunk.crs}
    return np.asarray(st["origin"], np.float64)


def frame_matches(chunk) -> bool:
    """Whether the chunk is still in the transform and the system the elements were made in."""
    fr = state(chunk).get("frame")
    if fr is None:
        return True  # elements of a project older than the check: nothing to compare
    old = np.asarray(fr["transform"], float)
    new = np.asarray(chunk.transform, float)
    scale = max(1.0, float(np.abs(old[:3, 3]).max()))
    # absolute tolerances only: numpy's default relative one would let 7 m pass at UTM coordinates
    return fr.get("crs") == chunk.crs and old.shape == new.shape \
        and np.allclose(old[:3, :3], new[:3, :3], rtol=0.0, atol=1e-9) \
        and np.allclose(old[:3, 3], new[:3, 3], rtol=0.0, atol=1e-9 * scale)


def check_frame(chunk) -> None:
    """:class:`NotReady` when the chunk's transform or system changed after the elements were made (georeferenced,
    *Sistema e quote*): their solids and the IFC's origin would be in the old frame. :func:`rebuild` makes them
    again."""
    if state(chunk).get("elements") and not frame_matches(chunk):
        raise NotReady(t("Il sistema del chunk è cambiato dopo che gli elementi sono stati fatti: ricostruisci gli "
                         "elementi (Ricostruisci, «ppg plugin architecture rebuild»).",
                         "The chunk's system changed after the elements were made: rebuild the elements (Rebuild, "
                         "'ppg plugin architecture rebuild')."))


def _world_vertices(chunk) -> np.ndarray:
    from pyarchinit_photogrammetry.io.mesh_ply import read_mesh

    p = chunk.product_path("model")
    if p is None or not p.exists():
        raise NotReady(t("Il chunk non ha un modello: costruiscilo prima (gli elementi si misurano sul modello).",
                         "The chunk has no model: build it first (the elements are measured on the model)."))
    return chunk.to_world(np.asarray(read_mesh(p).xyz, np.float64))


def local_mesh(chunk) -> tuple[np.ndarray, np.ndarray]:
    """The chunk's model in the export frame: world coordinates (double precision) minus the origin, and its faces."""
    from pyarchinit_photogrammetry.io.mesh_ply import read_mesh

    p = chunk.product_path("model")
    if p is None or not p.exists():
        _world_vertices(chunk)  # raises
    m = read_mesh(p)
    v = chunk.to_world(np.asarray(m.xyz, np.float64)) - origin(chunk)
    return v, np.asarray(m.faces, np.int64)


def check_metric(chunk) -> None:
    """The elements are in metres: a chunk without a scale (no reference, no scale bar) is refused."""
    from pyarchinit_photogrammetry.core import crs as crs_mod

    if crs_mod.is_local(chunk.crs) and not (chunk.meta.get("metric") or chunk.scalebars
                                             or abs(chunk.scale - 1.0) > 1e-12):
        raise NotReady(t("Il chunk non ha una scala (né riferimenti né barre di scala): le misure non sarebbero in "
                         "metri.", "The chunk has no scale (no references, no scale bars): the measures would not "
                         "be in metres."))


# ====================================================================== the solid
def _worker():
    if WORKER is not None:
        return WORKER
    return None, None


def build_solid(chunk, e: dict, job_part: dict, progress=None) -> dict:
    """The solid of ``e`` with the worker (``job_part``: the kind's part of the job and its ``placement``): STEP and
    STL in ``<element>/solid`` (the old ones replaced only when the new ones are complete), ``model.ply`` for the 3D
    view, ``e["solid"]`` and ``e["files"]``. Nothing is left behind when the worker fails or is cancelled."""
    from . import runner

    d = element_dir(chunk, e)
    d.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d%H%M%S") + uuid.uuid4().hex[:4]
    out = d / f"solid-{stamp}"
    job = {"protocol": 1, "kind": e["kind"] if e["kind"] != "vault" else "arch", "output": str(out),
           "name": e.get("name", ""), "tolerance": TOLERANCE, **job_part}
    python, script = _worker()
    try:
        result = runner.run(job, d / "job", progress, python=python, script=script)
    finally:
        shutil.rmtree(d / "job", ignore_errors=True)
    final = d / "solid"
    old = d / f"solid-old-{stamp}"
    if final.exists():
        final.rename(old)
    Path(result["output"]).rename(final)
    shutil.rmtree(old, ignore_errors=True)
    e["solid"] = {k: result.get(k) for k in ("volume", "area", "bbox", "valid")}
    e["solid"]["versions"] = result.get("versions", {})
    e["files"] = {"step": "solid/solid.step", "stl": "solid/solid.stl", "json": "element.json"}
    _display(chunk, e)
    return e


def solid_mesh(chunk, e: dict):
    """The tessellated solid (export frame) as a trimesh."""
    import trimesh

    p = path_of(chunk, e, "stl")
    if p is None or not p.exists():
        raise NotReady(t("Il solido dell'elemento non c'è più: ricostruiscilo.",
                         "The solid of the element is gone: build it again."))
    m = trimesh.load(str(p), process=False, force="mesh")
    return trimesh.Trimesh(np.asarray(m.vertices, np.float64), np.asarray(m.faces), process=True)


def _display(chunk, e: dict) -> None:
    """``model.ply`` in the chunk's internal frame (as its own products), coloured: what the 3D view shows."""
    from pyarchinit_photogrammetry.io.mesh_ply import write_mesh_ply

    m = solid_mesh(chunk, e)
    v = chunk.to_internal(np.asarray(m.vertices, np.float64) + origin(chunk))
    k = [x["id"] for x in elements(chunk)].index(e["id"]) if entry(chunk, e["id"]) else len(elements(chunk))
    colour = (np.asarray(COLOURS[k % len(COLOURS)]) * 255).astype(np.uint8)
    write_mesh_ply(element_dir(chunk, e) / "model.ply", v, np.asarray(m.faces), np.tile(colour, (len(v), 1)))
    e.setdefault("files", {})["ply"] = "model.ply"


def measure_fit(chunk, e: dict, points) -> dict:
    """The distance of the model's ``points`` (export frame) to the solid: median, p90, max (metres), number."""
    m = solid_mesh(chunk, e)
    pts = np.asarray(points, np.float64)
    if len(pts) > 200000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 200000, replace=False)]
    if not len(pts):
        e["fit"] = engine.fit_report([])
        return e["fit"]
    d = engine.distances_to_mesh(pts, m.vertices, m.faces)
    e["fit"] = engine.fit_report(d)
    return e["fit"]


def _points_near_box(vertices, bbox, margin: float = 0.01) -> np.ndarray:
    lo = np.asarray(bbox[:3]) - margin
    hi = np.asarray(bbox[3:]) + margin
    v = np.asarray(vertices)
    return v[np.all((v >= lo) & (v <= hi), axis=1)]


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _new_element(chunk, e: dict, build):
    """Build a new element; on any failure (the worker, a cancel, the fit) its folder goes away with whatever the
    worker wrote, and no entry is left."""
    try:
        return build()
    except BaseException:
        if entry(chunk, e["id"]) is None:
            shutil.rmtree(element_dir(chunk, e), ignore_errors=True)
        raise


def _base_entry(chunk, kind: str, name: str | None, notes: str = "", eid: str | None = None) -> dict:
    return {"id": eid or uuid.uuid4().hex[:8], "kind": kind, "name": name or _new_name(chunk, kind), "notes": notes,
            "made": _now(), "chunk": chunk.label}


# ====================================================================== column
def column(chunk, polygon=None, zmin=None, zmax=None, box=None, knots: int = 12, as_measured: bool = False,
           same_axis: str | None = None, name: str | None = None, notes: str = "", progress=None,
           _rebuild: dict | None = None) -> dict:
    """A column (or a part of one: base, shaft, capital) from the model inside a plan ``polygon`` (world x, y)
    between the heights ``zmin`` and ``zmax`` (world), or inside a world ``box`` (xmin, ymin, zmin, xmax, ymax,
    zmax). ``knots``: of the spline (``as_measured``: every point of the measured profile); ``same_axis``: the
    id or the name of a column whose axis this part shares."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    check_metric(chunk)
    check_frame(chunk)
    progress(0.02, t("Lettura del modello", "Reading the model"))
    v, f = local_mesh(chunk)
    o = origin(chunk)
    poly = None if polygon is None else np.asarray(polygon, np.float64)[:, :2] - o[:2]
    lbox = None if box is None else np.r_[np.asarray(box[:3], float) - o, np.asarray(box[3:], float) - o]
    lo = None if zmin is None else zmin - o[2]
    hi = None if zmax is None else zmax - o[2]
    sv, sf = engine.select_faces(v, f, poly, lo, hi, lbox)
    if len(sf) < 50:
        raise NotReady(t("Nella zona scelta il modello ha troppo poche facce per una colonna.",
                         "The model has too few faces in the chosen region for a column."))
    progress(0.1, t("Asse della colonna", "Axis of the column"))
    initial = None
    if same_axis:
        other = need(chunk, same_axis)
        if other["kind"] != "column":
            raise NotReady(t("L'asse si prende da un'altra colonna.", "The axis is taken from another column."))
        initial = (np.asarray(other["params"]["axis_point"]) - o, other["params"]["axis_direction"])
        p, d = initial
        p = np.asarray(p, float)
        d = engine.unit(d)
        rms = float(other["params"].get("axis_rms") or 0.0)
    else:
        p, d, rms = engine.column_axis(sv, sf)
    progress(0.35, t("Profilo misurato", "Measured profile"))
    meas = engine.column_profile(sv, sf, p, d, rms=rms)
    e = _base_entry(chunk, "column", name, notes, (_rebuild or {}).get("id"))
    e["source"] = {"chunk": chunk.label, "polygon": None if polygon is None else np.asarray(polygon)[:, :2].tolist(),
                   "zmin": zmin, "zmax": zmax, "box": None if box is None else list(map(float, box)),
                   "same_axis": same_axis, "faces": int(len(sf))}
    e["profile"] = {"heights": meas.heights.tolist(), "radii": meas.radii.tolist(), "spread": meas.spread.tolist(),
                    "step": meas.step}
    e["axis"] = {"point": (meas.point).tolist(), "direction": meas.direction.tolist()}  # export frame, base point
    _column_params(e, knots, as_measured, meas.point + o, meas.direction, rms, meas.height)
    if _rebuild:
        if _rebuild.get("knot_radii") is not None and len(_rebuild["knot_radii"]) == len(e["params"]["knot_heights"]):
            _set_knot_radii(e, _rebuild["knot_radii"])
        _edited(e, *(_rebuild.get("edited") or []))
    return _new_element(chunk, e, lambda: _finish_column(chunk, e, sv, progress))


def _column_params(e, knots, as_measured, base_world, direction, rms, height) -> None:
    prof = e["profile"]
    z, r = np.asarray(prof["heights"]), np.asarray(prof["radii"])
    if as_measured:
        curve = np.column_stack([r, z])
        zk, rk = z, r
        method = "measured"
    else:
        zk, rk = engine.fit_knots(z, r, knots)
        curve = engine.profile_curve(zk, rk)
        method = "spline"
    tilt = float(np.degrees(np.arccos(np.clip(abs(direction[2]), -1, 1))))
    e["params"] = {"method": method, "knots": int(len(zk)), "knot_heights": list(map(float, zk)),
                   "knot_radii": list(map(float, rk)), "height": float(height), "radius_base": float(curve[0, 0]),
                   "radius_top": float(curve[-1, 0]), "radius_max": float(curve[:, 0].max()),
                   "entasis": engine.entasis(curve), "axis_point": list(map(float, base_world)),
                   "axis_direction": list(map(float, direction)), "tilt_deg": tilt, "axis_rms": float(rms)}
    e["geometry"] = {"profile": curve.tolist(), "smooth": method == "spline"}
    top = np.asarray(base_world) + float(height) * np.asarray(direction)
    e["params"]["world_top"] = list(map(float, top))
    e["warnings"] = column_warnings(z, r, direction)


def column_warnings(heights, radii, direction) -> list[str]:
    """What the user should know of a column's profile: a step in it (a base, an abacus, a capital in the region:
    parts are better made as separate elements on the same axis) and, for a shaft lying down, that which end is the
    base is arbitrary."""
    out = []
    r = np.asarray(radii, float)
    if len(r) > 2:
        jump = np.abs(np.diff(r))
        if jump.max() > max(0.01, 0.05 * float(np.median(r))):
            k = int(np.argmax(jump))
            out.append(t(f"Il profilo ha un gradino di {jump[k] * 1000:.0f} mm a {heights[k]:.2f} m: base, fusto e "
                         "capitello vanno fatti come elementi separati sullo stesso asse.",
                         f"The profile has a step of {jump[k] * 1000:.0f} mm at {heights[k]:.2f} m: base, shaft and "
                         "capital are better made as separate elements on the same axis."))
    if abs(float(np.asarray(direction)[2])) <= 0.5:
        out.append(t("Il fusto è coricato: quale capo è la base è arbitrario (raggio alla base e in cima possono "
                     "essere scambiati).", "The shaft lies down: which end is the base is arbitrary (the radius at "
                     "the base and at the top may be the other way round)."))
    return out


def _finish_column(chunk, e, points, progress) -> dict:
    p = np.asarray(e["axis"]["point"])
    d = np.asarray(e["axis"]["direction"])
    placement = engine.column_frame(p, d)
    e["geometry"]["placement"] = placement.tolist()
    progress(0.45, t("Solido (CadQuery)", "Solid (CadQuery)"))
    build_solid(chunk, e, {"placement": placement.tolist(),
                           "column": {"profile": e["geometry"]["profile"], "smooth": e["geometry"]["smooth"]}},
                progress.sub(0.45, 0.4))
    progress(0.88, t("Distanza del modello dal solido", "Distance of the model from the solid"))
    measure_fit(chunk, e, points)
    _save(chunk, e)
    progress(1.0, t("Colonna pronta", "Column ready"))
    return e


def edit_column(chunk, eid: str, knot_radii=None, knots: int | None = None, as_measured: bool | None = None,
                progress=None) -> dict:
    """The solid of a column again, with other radii at the knots (``knot_radii``), another number of knots or as
    measured: the measured profile and the axis stay as they are."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    check_frame(chunk)
    e = copy.deepcopy(need(chunk, eid))  # the project's entry changes only when the new solid is there
    p = e["params"]
    if knots is not None or as_measured is not None:
        measured = bool(as_measured) if as_measured is not None else p["method"] == "measured"
        _column_params(e, knots or p["knots"], measured,
                       np.asarray(p["axis_point"]), np.asarray(p["axis_direction"]), p["axis_rms"], p["height"])
        p = e["params"]
    if knot_radii is not None:
        _set_knot_radii(e, knot_radii)
    _edited(e, *[k for k, val in (("knot_radii", knot_radii), ("knots", knots), ("as_measured", as_measured))
                 if val is not None])
    e["made"] = _now()
    v, f = local_mesh(chunk)
    points = _column_points(chunk, e, v, f)
    return _finish_column(chunk, e, points, progress)


def _set_knot_radii(e, knot_radii) -> None:
    """Radii given at the knots of a column: the solid's profile through them (the measured profile unchanged)."""
    p = e["params"]
    rk = np.asarray(knot_radii, float)
    if len(rk) != len(p["knot_heights"]):
        raise NotReady(t(f"Servono {len(p['knot_heights'])} raggi, uno per nodo.",
                         f"{len(p['knot_heights'])} radii are needed, one per knot."))
    zk = np.asarray(p["knot_heights"])
    if np.any(rk <= 0):
        raise NotReady(t("Un raggio è zero o negativo.", "A radius is zero or negative."))
    try:
        curve = engine.profile_curve(zk, rk) if p["method"] == "spline" else np.column_stack([rk, zk])
    except ValueError as exc:
        raise NotReady(str(exc)) from exc
    p.update(knot_radii=list(map(float, rk)), radius_base=float(curve[0, 0]), radius_top=float(curve[-1, 0]),
             radius_max=float(curve[:, 0].max()), entasis=engine.entasis(curve))
    e["geometry"]["profile"] = curve.tolist()


def _edited(e, *what) -> None:
    """Remember which parameters were changed by hand: a rebuild keeps them."""
    if what:
        e["edited"] = sorted(set(e.get("edited") or []) | set(what))


def _column_points(chunk, e, v, f):
    src = e["source"]
    o = origin(chunk)
    poly = None if src.get("polygon") is None else np.asarray(src["polygon"], float) - o[:2]
    box = src.get("box")
    lbox = None if box is None else np.r_[np.asarray(box[:3]) - o, np.asarray(box[3:]) - o]
    sv, _sf = engine.select_faces(v, f, poly, None if src.get("zmin") is None else src["zmin"] - o[2],
                                  None if src.get("zmax") is None else src["zmax"] - o[2], lbox)
    return sv


# ====================================================================== moulding
def moulding(chunk, path, station: float | None = None, window=None, out: float | None = None,
             depth: float = 0.1, simplify: float = 0.002, section=None, height: float | None = None,
             name: str | None = None, notes: str = "", progress=None, _rebuild: dict | None = None) -> dict:
    """A moulding along ``path`` (world points, (N, 2) in plan with ``height``, or (N, 3)). The section: the model
    cut by the plane normal to the path at ``station`` metres from its start (the middle of the first segment when
    None), the line of the cut nearest the path's point (inside ``window`` ``(u0, v0, u1, v1)`` around it when given,
    u = up x tangent, v up), the stretch of it that stands out of the wall face (:func:`engine.wall_section`), closed
    against the wall ``depth`` metres behind the face, simplified (Douglas-Peucker, ``simplify`` metres); ``out`` +1 /
    -1: the side where it projects (default: the side of the photos); or ``section`` given as (u, v) points. The
    solid: the section swept along the path with mitred corners (an extrusion when straight)."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    check_metric(chunk)
    check_frame(chunk)
    o = origin(chunk)
    v, f = local_mesh(chunk)
    pth = engine.drop_repeated_path(_path_3d(np.asarray(path, np.float64), v, f, o, height))
    if len(pth) < 2:
        raise NotReady(t("Il percorso ha un solo punto.", "The path has a single point."))
    progress(0.1, t("Sezione della modanatura", "Section of the moulding"))
    if station is None:
        station = 0.5 * float(np.linalg.norm(pth[1] - pth[0]))
    point, tangent = engine.point_at(pth, station)
    so, u, vv, _n = engine.section_frame(tangent, point)
    if out is None:
        out = _photo_side(chunk, point + o, u)
    res = engine.mesh_resolution(v, f)
    e = _base_entry(chunk, "moulding", name, notes, (_rebuild or {}).get("id"))
    e["source"] = {"chunk": chunk.label, "path": (pth + o).tolist(), "station": float(station),
                   "window": None if window is None else list(window), "out": float(out), "depth": float(depth),
                   "simplify": float(simplify), "height": height, "section_given": section is not None,
                   "resolution": res}
    if section is None:
        try:
            cut = engine.wall_section(engine.mesh_section(v, f, so, u, vv), out, depth, simplify, res, window)
        except ValueError as exc:
            raise NotReady(t(f"La sezione non si legge: {exc}.", f"The section cannot be read: {exc}.")) from exc
    else:
        sec = np.asarray(section, float)
        if engine.polygon_area(sec) < 0:
            sec = sec[::-1]
        lo, hi = sec.min(axis=0), sec.max(axis=0)
        cut = {"section": sec, "wall_u": None, "v_range": [float(lo[1]), float(hi[1])], "projection": None,
               "width": float(hi[0] - lo[0]), "height": float(hi[1] - lo[1]), "window": None, "line": None,
               "closed": True, "front_area": abs(engine.polygon_area(sec))}
    e["profile"] = {"line": cut["line"], "station_point": (point + o).tolist()}
    _moulding_params(e, pth, cut)
    _edited(e, *((_rebuild or {}).get("edited") or []))
    return _new_element(chunk, e, lambda: _finish_moulding(chunk, e, v, progress, f))


NO_WALL_FACE = ("Nessuna faccia del muro nel tratto: sporgenza dal punto più arretrato.",
                "No wall face in the stretch: projection from the deepest point.")


def _moulding_params(e, pth, cut) -> None:
    sec = np.asarray(cut["section"], float)
    straight = engine.is_straight(pth)
    closed = bool(cut.get("closed"))
    e["params"] = {"method": "extrusion" if straight else "sweep", "corners": "mitre",
                   "length": engine.path_length(pth),
                   # the area of the moulding in front of the wall face; the solid's section adds the slab behind it
                   "section_area": float(cut.get("front_area") or abs(engine.polygon_area(sec))),
                   "section_area_with_slab": abs(engine.polygon_area(sec)),
                   "projection": None if closed or cut.get("projection") is None else float(cut["projection"]),
                   "width": float(cut["width"]) if closed else None, "section_height": float(cut["height"]),
                   "wall_u": cut["wall_u"], "v_range": list(cut["v_range"]), "window_used": cut["window"],
                   "vertices": int(len(sec)), "path_vertices": int(len(pth)), "closed_member": closed}
    e["warnings"] = [t(*NO_WALL_FACE)] if cut.get("warning") == "no-wall-face" else []
    e["geometry"] = {"path": np.asarray(pth).tolist(), "section": sec.tolist(), "placement": np.eye(4).tolist()}
    e["_world_path"] = True  # the world ends of the path are added by _finish_moulding (it knows the origin)


def _finish_moulding(chunk, e, v, progress, f=None) -> dict:
    progress(0.3, t("Solido (CadQuery)", "Solid (CadQuery)"))
    g = e["geometry"]
    e.pop("_world_path", None)
    pth = np.asarray(g["path"], float) + origin(chunk)
    e["params"].update(world_start=pth[0].tolist(), world_end=pth[-1].tolist())
    build_solid(chunk, e, {"placement": g["placement"],
                           "moulding": {"path": g["path"], "section": g["section"]}},
                progress.sub(0.3, 0.5))
    progress(0.85, t("Distanza del modello dal solido", "Distance of the model from the solid"))
    near = _points_near_box(v, e["solid"]["bbox"])
    # the wall is not the moulding: the model's points on the wall face (within the mesh's resolution of it) and
    # behind it stay out of the fit; they lie on the solid by construction and would make it look better
    p = e["params"]
    out = float(e["source"].get("out") or 1.0)
    sec = np.asarray(g["section"], float)
    face = p["wall_u"] if p.get("wall_u") is not None else float(sec[:, 0].max() if out < 0 else sec[:, 0].min())
    res = float(e["source"].get("resolution") or (engine.mesh_resolution(v, f) if f is not None else 0.0))
    margin = min(res, 0.25 * max(float(p.get("projection") or p.get("width") or 0.0), 1e-3))
    behind = engine.behind_back(near, g["path"], face, out, margin) if len(near) else np.zeros(0, bool)
    measure_fit(chunk, e, near[~behind])
    e["fit"]["excluded_behind"] = int(behind.sum())
    _save(chunk, e)
    progress(1.0, t("Modanatura pronta", "Moulding ready"))
    return e


def edit_moulding(chunk, eid: str, simplify: float | None = None, depth: float | None = None, progress=None) -> dict:
    """The solid of a moulding again with another simplification or wall depth, from the line of the cut kept in the
    element (the one that was used, not a new search)."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    check_frame(chunk)
    e = copy.deepcopy(need(chunk, eid))  # the project's entry changes only when the new solid is there
    src = e["source"]
    if simplify is not None:
        src["simplify"] = float(simplify)
    if depth is not None:
        src["depth"] = float(depth)
    pth = np.asarray(e["geometry"]["path"], float)
    v, f = local_mesh(chunk)
    if not src.get("section_given") and e["profile"].get("line"):
        line = np.asarray(e["profile"]["line"], float)
        res = float(src.get("resolution") or engine.mesh_resolution(v, f))
        try:
            cut = engine.wall_section([line], src["out"], src["depth"], src["simplify"], res)
        except ValueError as exc:
            raise NotReady(t(f"La sezione non si legge: {exc}.", f"The section cannot be read: {exc}.")) from exc
    else:
        p = e["params"]
        cut = {"section": np.asarray(e["geometry"]["section"], float), "wall_u": p.get("wall_u"),
               "v_range": p.get("v_range") or [0, 0], "projection": p.get("projection"), "width": p.get("width"),
               "height": p["section_height"], "window": p.get("window_used"), "line": None,
               "closed": p.get("closed_member", False), "front_area": p.get("section_area")}
    _moulding_params(e, pth, cut)
    _edited(e, *[k for k, val in (("simplify", simplify), ("depth", depth)) if val is not None])
    e["made"] = _now()
    return _finish_moulding(chunk, e, v, progress, f)


def marker_points(chunk, names) -> np.ndarray:
    """World positions of markers (labels or keys) that have one: the path of a straight moulding."""
    out = []
    for name in names:
        marker = next((m for m in chunk.markers if str(name) in (m.label, str(m.key))), None)
        if marker is None:
            raise NotReady(t(f"Nessun marker «{name}».", f"No marker '{name}'."))
        if marker.position is None:
            raise NotReady(t(f"Il marker {name} non ha una posizione.", f"Marker {name} has no position."))
        out.append(chunk.to_world(np.asarray(marker.position, float)[None])[0])
    return np.asarray(out)


def elevation_entries(chunk) -> list[dict]:
    """The elevations and sections of the project a path or a point can be drawn on (their frame and raster)."""
    from pyarchinit_photogrammetry.engine import elevation

    return [x for x in elevation.elevations(chunk) if x.get("frame") and x.get("orthophoto")]


def elevation_entry(chunk, eid: str) -> dict:
    found = next((x for x in elevation_entries(chunk) if eid in (x["id"], x.get("name"))), None)
    if found is None:
        raise NotReady(t(f"Nessun prospetto o sezione «{eid}» nel chunk.",
                         f"No elevation or section '{eid}' in the chunk."))
    return found


def elevation_raster(chunk, eid: str) -> Path:
    """The orthophoto of an elevation (what the 2D view shows: its coordinates are the frame's u, v)."""
    return chunk.data_dir / elevation_entry(chunk, eid)["orthophoto"]


def from_elevation(chunk, eid: str, uv) -> np.ndarray:
    """World points of (u, v) drawn on an elevation: through its frame, the depth from the elevation's relief
    (the surface of the model seen on it) or, where the relief has no value, from the model's vertices near the
    (u, v) (the one nearest the viewer)."""
    from pyarchinit_photogrammetry.engine import elevation

    entry = elevation_entry(chunk, eid)
    frame = elevation.frame_from_dict(entry["frame"])
    uv = np.asarray(uv, np.float64)[:, :2]
    d = np.full(len(uv), np.nan)
    relief = chunk.data_dir / entry["relief"] if entry.get("relief") else None
    if relief is not None and relief.exists():
        import rasterio

        with rasterio.open(relief) as src:
            d = np.array([float(x[0]) for x in src.sample([tuple(p) for p in uv])])
            if src.nodata is not None and np.isfinite(src.nodata):
                d[d == src.nodata] = np.nan
    if not np.all(np.isfinite(d)):
        v, f = local_mesh(chunk)
        missing = ~np.isfinite(d)
        d[missing] = engine.surface_depth(frame.coords(v + origin(chunk)), f, uv[missing])
        if not np.all(np.isfinite(d)):
            raise NotReady(t("Il percorso esce dal modello sul prospetto.",
                             "The path leaves the model on the elevation."))
    return frame.point(uv[:, 0], uv[:, 1], d)


def _path_3d(path, v, f, o, height) -> np.ndarray:
    """The path in the export frame: (N, 3) as given, (N, 2) at ``height`` (world) or on the top of the model."""
    if path.shape[1] >= 3:
        return path[:, :3] - o
    xy = path[:, :2] - o[:2]
    if height is not None:
        z = np.full(len(xy), float(height) - o[2])
    else:
        z = np.array([engine.model_height_at(v, f, p) for p in xy])
        if not np.all(np.isfinite(z)):
            raise NotReady(t("Il percorso esce dal modello: dai la quota della modanatura.",
                             "The path leaves the model: give the height of the moulding."))
        z = np.full(len(xy), float(np.median(z)))
    return np.column_stack([xy, z])


def _photo_side(chunk, point_world, u) -> float:
    """+1 when most aligned photos are on the +u side of the point, else -1 (+1 without photos)."""
    cams = chunk.aligned_cameras
    if not cams:
        return 1.0
    centres = chunk.to_world(np.array([c.center for c in cams]))
    return 1.0 if np.sum(np.sign((centres - point_world) @ u)) >= 0 else -1.0


# ====================================================================== arch and vault
def section_planes(chunk) -> list[dict]:
    """The sections of the project (elevations of kind "section" on a plane) an arch can be read on."""
    from pyarchinit_photogrammetry.engine import elevation

    return [x for x in elevation.elevations(chunk) if x.get("kind") == "section" and x["frame"].get("kind") == "plane"]


def plane_of(chunk, section: str | None = None, p1=None, p2=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """The plane of an arch: a section of the project (its id or name) or the vertical plane through two plan
    points (world). Returns ``(origin, u, v)`` in the export frame and what it was."""
    from pyarchinit_photogrammetry.engine import elevation

    o = origin(chunk)
    if section is not None:
        found = next((x for x in section_planes(chunk) if section in (x["id"], x.get("name"))), None)
        if found is None:
            raise NotReady(t(f"Nessuna sezione «{section}» nel chunk.", f"No section '{section}' in the chunk."))
        fr = elevation.frame_from_dict(found["frame"])
        return fr.origin - o, fr.u_axis, fr.v_axis, {"section": found["id"], "name": found.get("name")}
    if p1 is None or p2 is None:
        raise NotReady(t("Serve una sezione o due punti.", "A section or two points are needed."))
    fr = elevation.plane_from_line(chunk, p1, p2)
    return fr.origin - o, fr.u_axis, fr.v_axis, {"p1": list(map(float, p1[:2])), "p2": list(map(float, p2[:2]))}


def arch(chunk, pick, section: str | None = None, p1=None, p2=None, kind: str = "auto", thickness: float | None = None,
         depth: float | None = None, vault: bool = False, name: str | None = None, notes: str = "",
         progress=None, plane=None, _rebuild: dict | None = None) -> dict:
    """An arch (or, ``vault``, a barrel vault) on a section plane: ``pick`` (u, v of the plane) a point inside the
    opening, under the intrados. The intrados is what the point sees of the model's cut (the jambs below the
    springing line are part of the fit), the extrados where the rays leave the ring. ``kind``: "auto",
    "semicircular", "segmental", "pointed", "elliptical", "measured" (a type asked for is enforced). ``thickness``:
    of the ring (measured from the extrados only when it is consistent, else asked for: :class:`NotReady`);
    ``depth``: along the plane's normal (the extent of the model's points within the ring when None). ``plane``:
    ``(origin, u, v)`` in world coordinates instead of a section or two points (a rebuild)."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    check_metric(chunk)
    check_frame(chunk)
    v, f = local_mesh(chunk)
    if plane is not None:
        po = np.asarray(plane[0], float) - origin(chunk)
        u, vv, what = engine.unit(plane[1]), engine.unit(plane[2]), {"rebuilt": True}
    else:
        po, u, vv, what = plane_of(chunk, section, p1, p2)
    progress(0.1, t("Taglio del modello", "Cutting the model"))
    lines = engine.mesh_section(v, f, po, u, vv)
    if not lines:
        raise NotReady(t("Il piano non taglia il modello.", "The plane does not cut the model."))
    near, far = engine.visible_points(lines, pick)
    if len(near) < 8:
        raise NotReady(t("Dal punto scelto non si vede abbastanza intradosso: scegli un punto dentro l'apertura, sotto "
                         "l'arco.", "Too little intrados is seen from the chosen point: choose a point inside the "
                         "opening, under the arch."))
    progress(0.2, t("Adattamento dell'arco", "Fitting the arch"))
    try:
        fit = engine.fit_arch(near, kind)
    except ValueError as exc:
        raise NotReady(t(f"L'arco non si adatta: {exc}.", f"The arch does not fit: {exc}.")) from exc
    measured_thickness, why = engine.arch_thickness(fit, far)
    th = float(thickness) if thickness else measured_thickness
    if not th:
        raise NotReady(t(f"Lo spessore dell'arco non si misura ({why}): dallo tu (campo Spessore, --thickness).",
                         f"The thickness of the arch cannot be measured ({why}): give it (field Thickness, "
                         f"--thickness)."))
    e = _base_entry(chunk, "vault" if vault else "arch", name, notes, (_rebuild or {}).get("id"))
    _edited(e, *((_rebuild or {}).get("edited") or []))
    e["source"] = {"chunk": chunk.label, **what, "pick": list(map(float, pick)), "kind_asked": kind,
                   "plane": {"origin": (po + origin(chunk)).tolist(), "u": u.tolist(), "v": vv.tolist()},
                   "resolution": engine.mesh_resolution(v, f), "thickness_given": thickness is not None and
                   bool(thickness), "depth_given": bool(depth)}
    e["profile"] = {"intrados": near.tolist(), "extrados": far.tolist(), "fit": fit.to_dict(),
                    "thickness_measured": measured_thickness, "thickness_note": why}
    _arch_params(e, fit, th)
    outline = e["geometry"]["outline"]
    n = np.cross(u, vv)
    rel = v - po
    uvd = np.column_stack([rel @ u, rel @ vv, rel @ n])
    if depth:
        d0, d1 = -float(depth) / 2.0, float(depth) / 2.0
        depth_measured = None
    else:
        ext = engine.footprint_depth(uvd, _buffer(engine.outline_polygon(outline), 0.01))
        if ext is None:
            raise NotReady(t("La profondità non si misura: dalla tu.", "The depth cannot be measured: give it."))
        d0, d1 = ext
        depth_measured = d1 - d0
    e["params"].update(depth=float(d1 - d0), depth_measured=depth_measured, depth_from=float(d0))
    e["geometry"]["depth"] = float(d1 - d0)
    e["geometry"]["placement"] = engine.arch_frame(po + d0 * n, u, vv).tolist()
    return _new_element(chunk, e, lambda: _finish_arch(chunk, e, v, progress))


def _buffer(poly, d):
    from shapely.geometry import Polygon

    return np.asarray(Polygon(poly).buffer(d).exterior.coords)


def _arch_params(e, fit: engine.ArchFit, thickness: float) -> None:
    outline = engine.arch_outline(fit, thickness)
    poly = engine.outline_polygon(outline)
    p = {"type": fit.type, "family": fit.family, "span": fit.span, "rise": fit.rise, "springing": fit.springing,
         "centre_u": fit.centre_u, "thickness": float(thickness), "rms": fit.rms, "points": fit.points,
         "section_area": abs(engine.polygon_area(poly)),
         "candidates_rms": {k: float(c["rms"]) for k, c in (fit.candidates or {}).items()}}
    fp = fit.params
    if fit.family == "circle":
        p.update(radius=fp["R"], centre_v=fp["vc"])
    elif fit.family == "pointed":
        p.update(radius=fp["R"], centres_distance=2 * (fp["R"] - fp["s"]))
    elif fit.family == "ellipse":
        p.update(semi_axis_u=fp["a"], semi_axis_v=fp["b"], centre_v=fp["vc"])
    e["params"] = p
    e["geometry"] = {"outline": outline}
    e["arch_fit"] = fit.to_dict()  # the fit the solid is made of now: an edit starts from it


def _arch_world(chunk, e) -> None:
    """World coordinates (the chunk's system) beside the plane's (u, v): the springing points and the apex."""
    g, p = e["geometry"], e["params"]
    pl = np.asarray(g["placement"])
    half = p["span"] / 2.0
    local = np.array([[p["centre_u"] - half, p["springing"], 0.0], [p["centre_u"] + half, p["springing"], 0.0],
                      [p["centre_u"], p["springing"] + p["rise"], 0.0]])
    w = engine.apply(pl, local) + origin(chunk)
    p.update(world_springing_left=w[0].tolist(), world_springing_right=w[1].tolist(), world_apex=w[2].tolist())


def _finish_arch(chunk, e, v, progress) -> dict:
    progress(0.3, t("Solido (CadQuery)", "Solid (CadQuery)"))
    g = e["geometry"]
    _arch_world(chunk, e)
    build_solid(chunk, e, {"placement": g["placement"], "arch": {"outline": g["outline"], "depth": g["depth"]}},
                progress.sub(0.3, 0.5))
    progress(0.85, t("Distanza del modello dal solido", "Distance of the model from the solid"))
    # the fit on the ring's curved faces only: points within 3 mesh edges of the intrados or the extrados, not on the
    # two plane faces (a ring flush with a wall shares them with the wall: they would say nothing of the arch)
    pl = np.asarray(g["placement"])
    rel = engine.apply(np.linalg.inv(pl), v)
    res = float(e["source"].get("resolution") or 0.01)
    tol = 3.0 * res
    poly = _buffer(engine.outline_polygon(g["outline"]), tol)
    box = engine.points_in_polygon(rel[:, :2], poly) & (rel[:, 2] > -tol) & (rel[:, 2] < g["depth"] + tol)
    faces = box & ((rel[:, 2] < res) | (rel[:, 2] > g["depth"] - res))
    fit = engine.ArchFit(**{k: e["arch_fit"][k] for k in ("type", "family", "params", "rms", "springing", "span",
                                                          "rise", "centre_u", "points")},
                         measured=e["arch_fit"].get("measured"))
    ring = np.zeros(len(v), bool)
    idx = np.flatnonzero(box & ~faces)
    if len(idx):
        ring[idx] = engine.near_ring(rel[idx, :2], fit, e["params"]["thickness"], tol)
    measure_fit(chunk, e, v[ring])
    e["fit"]["excluded_faces"] = int(faces.sum())
    _save(chunk, e)
    progress(1.0, t("Arco pronto", "Arch ready"))
    return e


def edit_arch(chunk, eid: str, kind: str | None = None, thickness: float | None = None, depth: float | None = None,
              progress=None) -> dict:
    """The solid of an arch again with another type, thickness or depth: the intrados as measured stays, and so does
    the type of the solid unless another one is asked for."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    check_frame(chunk)
    e = copy.deepcopy(need(chunk, eid))  # the project's entry changes only when the new solid is there
    near = np.asarray(e["profile"]["intrados"], float)
    current = e.get("arch_fit") or e["profile"]["fit"]
    if kind and kind != e["params"]["type"]:
        try:
            fit = engine.fit_arch(near, kind)
        except ValueError as exc:
            raise NotReady(t(f"L'arco non si adatta: {exc}.", f"The arch does not fit: {exc}.")) from exc
    else:
        fit = _fit_from(current)
    th = float(thickness) if thickness else e["params"]["thickness"]
    keep = {k: e["params"].get(k) for k in ("depth", "depth_measured", "depth_from")}
    _arch_params(e, fit, th)
    e["params"].update(keep)
    if depth:
        e["params"]["depth"] = float(depth)
    e["geometry"]["depth"] = float(e["params"]["depth"])
    plane = e["source"]["plane"]
    o = origin(chunk)
    po = np.asarray(plane["origin"]) - o
    u, vv = np.asarray(plane["u"]), np.asarray(plane["v"])
    n = np.cross(u, vv)
    d0 = float(e["params"].get("depth_from") or 0.0)
    if depth:
        mid = d0 + (keep["depth"] or 0.0) / 2.0
        d0 = mid - float(depth) / 2.0
        e["params"]["depth_from"] = d0
    e["geometry"]["placement"] = engine.arch_frame(po + d0 * n, u, vv).tolist()
    _edited(e, *[k for k, val in (("type", kind), ("thickness", thickness), ("depth", depth)) if val])
    e["made"] = _now()
    v, _f = local_mesh(chunk)
    done = _finish_arch(chunk, e, v, progress)
    for x in users_of(chunk, e["id"]):  # their voids were made on the old intrados
        x["stale"] = True
        _save(chunk, x)
    return done


def _fit_from(d: dict) -> engine.ArchFit:
    return engine.ArchFit(d["type"], d["family"], d["params"], d["rms"], d["springing"], d["span"], d["rise"],
                          d["centre_u"], d["points"], d.get("candidates") or {}, d.get("measured"))


# ====================================================================== opening (door, window)
OPENING_TYPES = ("opening", "door", "window")
VOID_OVERLAP = 0.001  # the void under an arch reaches this far into the part below it: one solid, not two touching


OUTER_RULES = ("auto", "near", "far")


def opening(chunk, at, height: float | None = None, top: str | None = None, thickness: float | None = None,
            kind: str = "opening", sill: float | None = None, lintel: float | None = None, name: str | None = None,
            notes: str = "", outer: str = "auto", progress=None, _rebuild: dict | None = None) -> dict:
    """A door or a window: the void through the wall around the plan point ``at`` (world x, y, inside the opening).
    The plan of the void from the horizontal cut of the model at ``height`` (world; the middle of the opening when
    None): the wall faces first (the near one has a gap around the point: the opening), the reveals only between
    the faces from the gap's ends, each fitted by a line (a splayed reveal stays splayed); ``thickness`` given: the
    inner face that far behind the outer one (needed when only one face is on the model). ``outer``: which face is
    the outer one - "auto" (the face seen by more of the aligned photos that frame it; the face nearer the point when
    the photos do not tell), "near" or "far" (to the point). The sill and the lintel from the vertical section
    across the wall through the point, or given (``sill``, ``lintel``, world heights). ``top``: None or "flat" (a
    lintel), or the id or the name of an arch element whose intrados closes the opening above its springing line.
    ``kind``: "opening", "door" or "window" (how it goes to the IFC). The solid: the plan extruded from the sill to
    the lintel (or the springing line, the void under the arch above it), in world."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    check_metric(chunk)
    check_frame(chunk)
    if kind not in OPENING_TYPES:
        raise NotReady(t(f"Tipo «{kind}»: apertura, porta o finestra.", f"Kind '{kind}': opening, door or window."))
    progress(0.02, t("Lettura del modello", "Reading the model"))
    o = origin(chunk)
    v, f = local_mesh(chunk)
    c = np.asarray(at, np.float64)[:2] - o[:2]
    arch_e, top_limit = None, None
    if top and top != "flat":
        arch_e = need(chunk, top)
        if arch_e["kind"] not in ("arch", "vault"):
            raise NotReady(t("La parte alta di un'apertura si prende da un arco.",
                             "The top of an opening is taken from an arch."))
        top_limit = _arch_springing(chunk, arch_e)
    if outer not in OUTER_RULES:
        raise NotReady(t(f"Faccia esterna «{outer}»: auto, near o far.", f"Outer face '{outer}': auto, near or far."))
    progress(0.1, t("Pianta e altezze dell'apertura", "Plan and heights of the opening"))
    try:
        r = engine.measure_opening(v, f, c, None if height is None else float(height) - o[2], thickness or None,
                                   top_limit, None if sill is None else float(sill) - o[2],
                                   None if lintel is None else float(lintel) - o[2], apply_thickness=False)
    except ValueError as exc:
        raise NotReady(_opening_problem(str(exc), height is not None)) from exc
    plan = r["plan"]  # the outer face is the near one so far
    views = _face_views(chunk, v, plan, r["sill"], r["lintel"] if r["lintel"] is not None else top_limit)
    if outer == "near":
        flip, rule = False, "near"
    elif outer == "far":
        flip, rule = True, "far"
    elif views[0] != views[1]:
        flip, rule = views[1] > views[0], "photos"
    else:
        flip, rule = False, "near-fallback"
    widths = {k: r["widths"][k]["inner" if flip else "outer"] for k in ("sill", "lintel")}
    if flip:
        plan = plan.flipped()
        views = views[::-1]
    if thickness and plan.measured:
        plan = engine.with_thickness(plan, thickness)
    rb = _rebuild or {}
    if rb.get("width") is not None or rb.get("thickness") is not None:
        plan = _changed_plan(plan, rb.get("width"), rb.get("thickness"))
    e = _base_entry(chunk, "opening", name or _new_name(chunk, kind), notes, rb.get("id"))
    e["source"] = {"chunk": chunk.label, "at": list(map(float, np.asarray(at, float)[:2])),
                   "height": None if height is None else float(height), "cut_height": r["height"] + o[2],
                   "top": arch_e["id"] if arch_e else "flat", "thickness_given": bool(thickness),
                   "thickness": float(thickness) if thickness else None, "sill_given": sill is not None,
                   "lintel_given": lintel is not None, "kind": kind, "resolution": r["resolution"],
                   "outer": outer, "outer_rule": rule, "views": {"outer": int(views[0]), "inner": int(views[1])}}
    keep = 1.0 + max(plan.width_inner, plan.width_outer, plan.thickness)

    def near(lines):
        return [np.asarray(p).tolist() for p in lines if len(p) > 1 and np.min(np.linalg.norm(
            np.asarray(p)[:, :2] - c, axis=1)) < keep]

    e["profile"] = {**{k: np.asarray(x).tolist() for k, x in plan.points.items()},
                    "sill_measured": None if r["sill_measured"] is None else r["sill_measured"] + o[2],
                    "lintel_measured": None if r["lintel_measured"] is None else r["lintel_measured"] + o[2],
                    "sill_spread": r["sill_spread"], "lintel_spread": r["lintel_spread"],
                    # rise per metre going in (the section's u runs out of the near face)
                    "sill_slope": None if r["sill_slope"] is None else r["sill_slope"] * (1 if flip else -1),
                    "heights_found": None if r["heights_found"] is None else [z + o[2] for z in r["heights_found"]],
                    "cut": near(r["cut"]), "section": [np.asarray(p).tolist() for p in r["section"]
                                                       if np.min(np.abs(np.asarray(p)[:, 0])) < keep]}
    e["measured"] = {"width_at_sill": widths["sill"], "width_at_lintel": widths["lintel"]}
    _opening_params(chunk, e, plan, r["sill"], r["lintel"], arch_e)
    _edited(e, *(rb.get("edited") or []))
    return _new_element(chunk, e, lambda: _finish_opening(chunk, e, v, progress))


OPENING_PROBLEMS = (
    ("curved wall faces", ("Pareti curve non supportate in questa versione: le facce del muro intorno al punto non "
                           "stanno su una retta (più di 2 cm).", "Curved walls are not supported in this version: the "
                           "faces of the wall around the point are not on a straight line (more than 2 cm).")),
    ("the wall ends beside the point", ("Il muro finisce accanto al punto, o un pilastro accanto all'apertura è più "
                                        "corto di 15 cm: un'apertura si legge con almeno 15 cm di faccia del muro da "
                                        "entrambe le parti.", "The wall ends beside the point, or a pier beside the "
                                        "opening is shorter than 15 cm: an opening is read with at least 15 cm of wall "
                                        "face on both sides.")),
    ("thickness of the wall must be given", ("Sul modello c'è una sola faccia del muro: dai lo spessore del muro "
                                             "(campo Spessore del muro, --thickness).", "Only one face of the wall is "
                                             "on the model: give the thickness of the wall (field Wall thickness, "
                                             "--thickness).")),
    ("at any height", ("Il punto non è dentro un'apertura: a nessuna quota una faccia di muro ha un vano intorno a "
                       "esso. Clicca in pianta fra i due stipiti di una porta o di una finestra.", "The point is not "
                       "inside an opening: at no height does a wall face have a gap around it. Click in plan between "
                       "the two reveals of a door or a window.")),
    ("no gap in a wall face", ("A quella quota nessuna faccia di muro ha un vano intorno al punto: lascia la quota "
                               "automatica o danne una dentro l'apertura.", "At that height no wall face has a gap "
                               "around the point: leave the height automatic or give one inside the opening.")),
    ("no reveal at one end", ("Da una parte del vano non c'è uno stipite sul modello (un'apertura in fondo a un muro, "
                              "o uno stipite non rilevato): l'apertura non si chiude.", "On one side of the gap there "
                              "is no reveal on the model (an opening at the end of a wall, or a reveal not surveyed): "
                              "the opening cannot be closed.")),
    ("not across the wall", ("Uno stipite non attraversa il muro (più di 45 gradi dalla normale della faccia).",
                             "A reveal does not cross the wall (more than 45 degrees from the face's normal).")),
    ("give the sill", ("Sotto il punto non si incontra niente (la soglia o il pavimento): dai la quota della soglia.",
                       "Nothing is met under the point (the sill or the floor): give the height of the sill.")),
    ("give the lintel", ("Sopra il punto non si incontra niente (l'architrave): dai la sua quota.",
                         "Nothing is met over the point (the lintel): give its height.")),
    ("not around the point", ("Il modello non è intorno al punto.", "The model is not around the point.")),
)
THIN_WALL = ("Le facce del muro non si distinguono: per un muro sottile la mesh è troppo rada (lati di {res:.0f} mm). "
             "Dai lo spessore del muro o usa un modello più fitto.", "The faces of the wall cannot be told apart: for "
             "a thin wall the mesh is too coarse (edges of {res:.0f} mm). Give the thickness of the wall or use a "
             "denser model.")


def _opening_problem(text: str, height_given: bool = False) -> str:
    """What the engine said of an opening it cannot read, in the language of the interface."""
    import re

    if "thin wall" in text:
        m = re.search(r"edges of (\d+) mm", text)
        res = float(m.group(1)) if m else 0.0
        return t(THIN_WALL[0].format(res=res), THIN_WALL[1].format(res=res))
    for key, (it, en) in OPENING_PROBLEMS:
        if key in text:
            return t(it, en)
    return t(f"L'apertura non si legge: {text}.", f"The opening cannot be read: {text}.")


def _arch_springing(chunk, arch_e) -> float:
    """The springing line of an arch element in the export frame (its section plane must be vertical)."""
    plane = arch_e["source"]["plane"]
    V = np.asarray(plane["v"], float)
    if abs(V[2]) < 0.995:
        raise NotReady(t("L'arco non è su un piano verticale: non può chiudere un'apertura.",
                         "The arch is not on a vertical plane: it cannot close an opening."))
    return float(np.asarray(plane["origin"], float)[2] - origin(chunk)[2] + arch_e["arch_fit"]["springing"] * V[2])


def _face_views(chunk, v, plan: engine.OpeningPlan, bottom: float, top: float | None) -> tuple[int, int]:
    """How many aligned photos see each face of the wall around the opening: the model's points on the face (within 2
    cm of its line, along the opening and half a metre each side, between the sill and the top) in the frame of the
    photo (the program's ``project_valid``: at least half of them) from the side the face looks to. ``(outer,
    inner)``; a face with no points on the model is seen by none."""
    from pyarchinit_photogrammetry.engine.visibility import project_valid

    cams = chunk.aligned_cameras
    if not cams:
        return 0, 0
    o = origin(chunk)
    v = np.asarray(v, float)
    half = max(plan.width_outer, plan.width_inner) / 2 + 0.5
    z1 = float(top) if top is not None else float(bottom) + 2.0
    centres = chunk.to_world(np.array([c.center for c in cams])) - o
    out = []
    for line, facing in ((plan.outer, plan.axis), (plan.inner, -plan.axis)):
        p, d = np.asarray(line[0], float), np.asarray(line[1], float)
        n = np.array([-d[1], d[0]])
        m = (np.abs((v[:, :2] - p) @ n) < 0.02) & (np.abs((v[:, :2] - plan.centre) @ d) < half) & \
            (v[:, 2] >= float(bottom)) & (v[:, 2] <= z1)
        pts = v[m]
        if len(pts) < 10:
            out.append(0)
            continue
        if len(pts) > 200:
            pts = pts[np.random.default_rng(0).choice(len(pts), 200, replace=False)]
        internal = chunk.to_internal(pts + o)
        mid = pts[:, :2].mean(axis=0)
        k = 0
        for cam, centre in zip(cams, centres):
            if (centre[:2] - mid) @ facing <= 0:
                continue
            _px, _depth, ok = project_valid(chunk, cam, internal)
            if np.mean(ok) >= 0.5:
                k += 1
        out.append(k)
    return out[0], out[1]


OUTER_TEXTS = {"photos": ("la faccia vista da più foto", "the face seen by more photos"),
               "near": ("la faccia più vicina al clic, scelta", "the face nearer the click, chosen"),
               "far": ("l'altra faccia, scelta", "the other face, chosen"),
               "near-fallback": ("la faccia più vicina al clic (le foto non distinguono le facce)",
                                 "the face nearer the click (the photos do not tell the faces apart)")}


def _changed_plan(plan: engine.OpeningPlan, width: float | None = None, thickness: float | None = None):
    """The plan with another width at the outer face (both reveals moved by half the difference, their splay kept)
    or another thickness (the inner face moved, the outer face kept)."""
    a = plan.axis
    w = engine.left_of(a)
    left, right, outer, inner = plan.left, plan.right, plan.outer, plan.inner
    runs = {k: list(v) for k, v in (plan.runs or {}).items()}
    if width is not None:
        dw = (float(width) - plan.width_outer) / 2.0
        left = (np.asarray(left[0]) + dw * w, left[1])
        right = (np.asarray(right[0]) - dw * w, right[1])
        for side, sign in (("left", 1.0), ("right", -1.0)):  # every run of a stepped reveal moves with it
            runs[side] = [(np.asarray(r[0]) + sign * dw * w, r[1], np.asarray(r[2]) + sign * dw * w,
                           np.asarray(r[3]) + sign * dw * w) for r in runs.get(side) or []]
    out = engine.opening_plan_from_lines(plan.centre, a, left, right, outer, inner, plan.measured, plan.reveal_rms,
                                         plan.points, near=plan.near, runs=runs or None)
    return out if thickness is None else engine.with_thickness(out, thickness)


def _plan_from(d: dict) -> engine.OpeningPlan:
    return engine.opening_plan_from_lines(d["centre"], d["axis"], d["left"], d["right"], d["outer"], d["inner"],
                                          d["measured"], d.get("reveal_rms", 0.0), {}, near=d.get("near", True),
                                          runs=d.get("runs") or None)


def _opening_params(chunk, e, plan: engine.OpeningPlan, sill: float, lintel: float | None, arch_e) -> None:
    """Parameters and geometry of an opening (the export frame inside, world heights in ``params``)."""
    o = origin(chunk)
    bottom = float(sill)
    void = None
    springing = None
    if arch_e is not None:
        fit = _fit_from(arch_e["arch_fit"])
        pl = arch_e["source"]["plane"]
        po = np.asarray(pl["origin"], float) - o
        U, V = engine.unit(pl["u"]), engine.unit(pl["v"])
        n = np.cross(U, V)
        if abs(n[:2] @ plan.axis) < np.cos(np.radians(20)):
            raise NotReady(t("Il piano dell'arco non attraversa il muro dell'apertura.",
                             "The plane of the arch does not cross the wall of the opening."))
        crown = po + (fit.centre_u * U) + (fit.springing + fit.rise) * V
        w = engine.left_of(plan.axis)
        if abs((crown[:2] - plan.centre) @ w) > max(plan.width_outer, plan.width_inner) / 2:
            raise NotReady(t("L'arco non sta sopra questa apertura.", "The arch is not over this opening."))
        springing = float(po[2] + fit.springing * V[2])
        top = springing
        lintel = float(crown[2])
        corners = np.asarray(plan.polygon, float)
        d = (corners - po[:2]) @ n[:2]
        outline = engine.arch_void_outline(fit, below=VOID_OVERLAP)
        void = {"outline": outline, "placement": engine.arch_frame(po + float(d.min()) * n, U, V).tolist(),
                "depth": float(d.max() - d.min()), "arch": arch_e["id"]}
        void_area = abs(engine.polygon_area(engine.outline_polygon(outline, arc_points=400))) - \
            VOID_OVERLAP * fit.span
        warn = abs(fit.span - plan.width_outer) > max(0.05, 0.1 * plan.width_outer)
    else:
        if lintel is None:
            raise NotReady(t("L'architrave non si vede dal punto: dai la sua quota.",
                             "The lintel is not seen from the point: give its height."))
        top = float(lintel)
        void_area, warn = 0.0, False
    if top - bottom < 0.05:
        raise NotReady(t("Il sotto dell'apertura non è sotto il sopra.",
                         "The bottom of the opening is not below its top."))
    ground = _ground_at(chunk, plan, o)
    outer_mid = _outer_middle(plan)
    width = plan.width_outer
    meas = e.get("measured") or {}
    rebates = {}
    for side in ("left", "right"):
        steps = (plan.rebates or {}).get(side) or []
        rebates[side] = None if not steps else \
            dict(zip(("depth", "width"), map(float, max(steps, key=lambda x: x[1]))))
    p = {"opening_type": e["source"]["kind"], "top": "arch" if arch_e else "flat",
         "arch": arch_e["id"] if arch_e else None, "width": width, "width_outer": width,
         "clear_width": plan.clear_width, "overall_width": plan.overall_width,
         "rebate_left": rebates["left"], "rebate_right": rebates["right"],
         "width_inner": plan.width_inner, "width_at_sill": meas.get("width_at_sill"),
         "width_at_lintel": meas.get("width_at_lintel"), "sill": bottom + o[2], "lintel": float(lintel) + o[2],
         "springing": None if springing is None else springing + o[2], "height": float(lintel) - bottom,
         "ground": ground, "sill_above_ground": None if ground is None else bottom + o[2] - ground,
         "lintel_above_ground": None if ground is None else float(lintel) + o[2] - ground,
         "reveal_depth": plan.thickness, "thickness_measured": bool(plan.measured),
         "splay_left": plan.splay_left, "splay_right": plan.splay_right,
         "area": width * (top - bottom) + void_area, "plan_area": abs(engine.polygon_area(plan.polygon)),
         "cut_height": e["source"]["cut_height"], "reveal_rms": plan.reveal_rms,
         "world_threshold": [float(outer_mid[0] + o[0]), float(outer_mid[1] + o[1]), bottom + o[2]],
         "world_axis": [float(plan.axis[0]), float(plan.axis[1]), 0.0],
         "outer_rule": e["source"].get("outer_rule"), "outer_views": (e["source"].get("views") or {}).get("outer"),
         "inner_views": (e["source"].get("views") or {}).get("inner")}
    prof = e.get("profile") or {}
    slope = prof.get("sill_slope")
    if slope is not None:
        p["sill_slope_percent"] = 100.0 * float(slope)
        p["sill_drop"] = float(slope) * plan.thickness  # the rise from the outer face to the inner one
    e["params"] = p
    e["geometry"] = {"plan": np.asarray(plan.polygon).tolist(), "bottom": bottom, "top": top, "void": void,
                     "placement": np.eye(4).tolist(),
                     "lines": {k: x for k, x in plan.to_dict().items() if k != "points"}}
    e["warnings"] = []
    if warn:
        e["warnings"].append(t(f"La luce dell'arco ({fit.span:.3f} m) non è la larghezza dell'apertura "
                               f"({plan.width_outer:.3f} m).", f"The span of the arch ({fit.span:.3f} m) is not the "
                               f"width of the opening ({plan.width_outer:.3f} m)."))
    stepped = [s_ for s_ in ("left", "right") if len((plan.runs or {}).get(s_) or []) > 1]
    if stepped:
        e["warnings"].append(t(f"Stipiti a gradini (battuta): luce netta {plan.clear_width:.3f} m, larghezza massima "
                               f"{plan.overall_width:.3f} m; la strombatura è quella del tratto più lungo di ogni "
                               "stipite.", f"Stepped reveals (a rebate): clear width {plan.clear_width:.3f} m, overall "
                               f"width {plan.overall_width:.3f} m; the splay is that of each reveal's longest run."))
        p["reveal_stepped"] = ", ".join(stepped)
    if p.get("sill_drop") is not None and abs(p["sill_drop"]) > 0.01 and not e["source"].get("sill_given"):
        e["warnings"].append(t(f"La soglia pende di {abs(p['sill_drop']) * 1000:.0f} mm attraverso il muro "
                               f"({p['sill_slope_percent']:+.1f} % verso l'interno): la quota data è quella a metà "
                               "spessore.", f"The sill slopes by {abs(p['sill_drop']) * 1000:.0f} mm across the wall "
                               f"({p['sill_slope_percent']:+.1f} % going in): the height given is the one "
                               "at half the thickness."))
    seen = prof.get("lintel_measured")
    if arch_e is not None and seen is not None and seen - o[2] < springing - 0.03:
        e["warnings"].append(t(f"Sotto l'imposta dell'arco il modello ha un architrave a {seen:.3f} m: il vano "
                               "dell'arco entrerebbe nella muratura.", f"Below the arch's springing line the model has "
                               f"a lintel at {seen:.3f} m: the arch's void would run into the masonry."))
    if p["width_at_sill"] is not None and p["width_at_lintel"] is not None and \
            abs(p["width_at_sill"] - p["width_at_lintel"]) > max(0.01, 0.02 * width):
        e["warnings"].append(t(f"Gli stipiti non sono verticali: larghezza {p['width_at_sill']:.3f} m alla soglia, "
                               f"{p['width_at_lintel']:.3f} m all'architrave.", "The jambs are not upright: width "
                               f"{p['width_at_sill']:.3f} m at the sill, {p['width_at_lintel']:.3f} m at the lintel."))


def _outer_middle(plan: engine.OpeningPlan) -> np.ndarray:
    """The middle of the void on the outer face (plan)."""
    return 0.5 * (engine.intersect_lines(*plan.left, *plan.outer) + engine.intersect_lines(*plan.right, *plan.outer))


def _ground_at(chunk, plan: engine.OpeningPlan, o) -> float | None:
    """The height of the chunk's DEM in front of the threshold (outside, a cell beyond the outer face), None without
    a DEM or a value there."""
    path = chunk.product_path("dem")
    if path is None or not path.exists():
        return None
    import rasterio

    mid = _outer_middle(plan)
    try:
        with rasterio.open(path) as src:
            step = 2.0 * max(abs(src.transform.a), abs(src.transform.e))
            xy = mid + step * plan.axis + o[:2]
            value = float(next(src.sample([(float(xy[0]), float(xy[1]))]))[0])
            nodata = src.nodata
    except Exception:  # noqa: BLE001 - a DEM that cannot be read: no ground
        return None
    if not np.isfinite(value) or (nodata is not None and np.isfinite(nodata) and value == nodata):
        return None
    return value


def _finish_opening(chunk, e, v, progress) -> dict:
    progress(0.4, t("Solido (CadQuery)", "Solid (CadQuery)"))
    g = e["geometry"]
    spec = {"plan": g["plan"], "bottom": g["bottom"], "top": g["top"]}
    if g.get("void"):
        spec["void"] = {k: g["void"][k] for k in ("outline", "placement", "depth")}
    build_solid(chunk, e, {"placement": g["placement"], "opening": spec}, progress.sub(0.4, 0.4))
    progress(0.85, t("Distanza del modello dal solido", "Distance of the model from the solid"))
    # the fit on the reveals, the sill, the lintel (and the intrados): the model's points between the two wall faces,
    # not on them - the faces are the wall around the opening, not the opening
    plan = _plan_from(g["lines"])
    res = float(e["source"].get("resolution") or 0.01)
    near = _points_near_box(v, e["solid"]["bbox"], margin=max(3 * res, 0.02))
    if len(near):
        along = (near[:, :2] - plan.centre) @ plan.axis
        u_out = engine._along_axis(plan, plan.outer)
        u_in = engine._along_axis(plan, plan.inner)
        lo, hi = sorted((u_in, u_out))
        margin = max(res, 0.005)
        inside = (along > lo + margin) & (along < hi - margin)
    else:
        inside = np.zeros(0, bool)
    measure_fit(chunk, e, near[inside])
    e["fit"]["excluded_faces"] = int((~inside).sum())
    _save(chunk, e)
    progress(1.0, t("Apertura pronta", "Opening ready"))
    return e


def edit_opening(chunk, eid: str, sill: float | None = None, lintel: float | None = None, width: float | None = None,
                 thickness: float | None = None, progress=None) -> dict:
    """The solid of an opening again with another sill or lintel (world heights), width at the outer face (both
    reveals moved by half the difference, their splay kept) or wall thickness (the inner face moved): what was measured
    stays as it is."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    check_frame(chunk)
    e = copy.deepcopy(need(chunk, eid))  # the project's entry changes only when the new solid is there
    if e["kind"] != "opening":
        raise NotReady(t(f"«{e['name']}» non è un'apertura.", f"'{e['name']}' is not an opening."))
    p, o = e["params"], origin(chunk)
    arch_e = need(chunk, p["arch"]) if p.get("top") == "arch" else None
    if lintel is not None and arch_e is not None:
        raise NotReady(t("La parte alta viene dall'arco: si cambia modificando l'arco.",
                         "The top comes from the arch: it changes with the arch."))
    plan = _changed_plan(_plan_from(e["geometry"]["lines"]), width, thickness)
    new_sill = (float(sill) if sill is not None else p["sill"]) - o[2]
    new_lintel = None if arch_e is not None else (float(lintel) if lintel is not None else p["lintel"]) - o[2]
    _opening_params(chunk, e, plan, new_sill, new_lintel, arch_e)
    _edited(e, *[k for k, val in (("sill", sill), ("lintel", lintel), ("width", width), ("thickness", thickness))
                 if val is not None])
    e["made"] = _now()
    v, _f = local_mesh(chunk)
    return _finish_opening(chunk, e, v, progress)


# ====================================================================== show, export
def summary(e: dict) -> str:
    """The element in a line: kind, main numbers, fit."""
    from .texts import arch_title, kind_title, metres

    p, fit = e.get("params") or {}, e.get("fit") or {}
    if e["kind"] == "column":
        main = t(f"h {metres(p['height'])}, r {metres(p['radius_base'])} → {metres(p['radius_top'])}, entasi "
                 f"{metres(p['entasis'])}", f"h {metres(p['height'])}, r {metres(p['radius_base'])} → "
                 f"{metres(p['radius_top'])}, entasis {metres(p['entasis'])}")
    elif e["kind"] == "moulding" and p.get("closed_member"):
        main = t(f"lunghezza {metres(p['length'])}, larghezza {metres(p['width'])}, altezza "
                 f"{metres(p['section_height'])}", f"length {metres(p['length'])}, width {metres(p['width'])}, "
                 f"height {metres(p['section_height'])}")
    elif e["kind"] == "moulding":
        main = t(f"lunghezza {metres(p['length'])}, sporgenza {metres(p['projection'])}, altezza "
                 f"{metres(p['section_height'])}", f"length {metres(p['length'])}, projection "
                 f"{metres(p['projection'])}, height {metres(p['section_height'])}")
    elif e["kind"] == "opening":
        main = opening_text(p)
    else:
        main = t(f"{arch_title(p['type'])} ({candidates_text(p)}), luce {metres(p['span'])}, freccia "
                 f"{metres(p['rise'])}, spessore {metres(p['thickness'])}",
                 f"{arch_title(p['type'])} ({candidates_text(p)}), span {metres(p['span'])}, rise "
                 f"{metres(p['rise'])}, thickness {metres(p['thickness'])}")
    gap = t(f"scarto mediana {metres(fit.get('median'))}, p90 {metres(fit.get('p90'))}",
            f"fit median {metres(fit.get('median'))}, p90 {metres(fit.get('p90'))}") if fit.get("points") else \
        t("scarto non misurato", "fit not measured")
    warn = " " + " ".join(e.get("warnings") or []) if e.get("warnings") else ""
    if e.get("stale"):
        warn += " " + t("Da ricostruire: l'arco che la chiude è cambiato.", "To be rebuilt: the arch that closes it "
                        "changed.")
    if e.get("edited"):
        warn += " " + t("Parametri modificati a mano mantenuti: " + ", ".join(e["edited"]) + ".",
                        "Parameters edited by hand kept: " + ", ".join(e["edited"]) + ".")
    return f"{kind_title(e['kind'])}: {main}; {gap}.{warn}"


def opening_text(p: dict) -> str:
    """An opening's main numbers: its kind, width, height, sill, depth, splays."""
    from .texts import degrees, kind_title, metres

    top = t(", ad arco", ", arched") if p.get("top") == "arch" else ""
    splay = ""
    if abs(p.get("splay_left") or 0.0) >= 0.5 or abs(p.get("splay_right") or 0.0) >= 0.5:
        splay = t(f", strombatura {degrees(p['splay_left'])} / {degrees(p['splay_right'])}",
                  f", splay {degrees(p['splay_left'])} / {degrees(p['splay_right'])}")
    if p.get("rebate_left") or p.get("rebate_right"):
        parts = []
        for side, (it, en) in (("rebate_left", ("sinistra", "left")), ("rebate_right", ("destra", "right"))):
            r = p.get(side)
            if r:
                parts.append(t(f"{it} {metres(r['width'])} a {metres(r['depth'])}",
                               f"{en} {metres(r['width'])} at {metres(r['depth'])}"))
        splay += t(f", luce netta {metres(p['clear_width'])}, battuta " + "; ".join(parts),
                   f", clear width {metres(p['clear_width'])}, rebate " + "; ".join(parts))
    rule = OUTER_TEXTS.get(p.get("outer_rule") or "")
    outer = t(f", esterna: {rule[0]}", f", outer: {rule[1]}") if rule else ""
    return t(f"{kind_title(p.get('opening_type', 'opening')).lower()}{top} {metres(p['width'])} × "
             f"{metres(p['height'])}, soglia {metres(p['sill'], 0.0)}, profondità {metres(p['reveal_depth'])}{splay}"
             f"{outer}",
             f"{kind_title(p.get('opening_type', 'opening')).lower()}{top} {metres(p['width'])} × "
             f"{metres(p['height'])}, sill {metres(p['sill'], 0.0)}, depth {metres(p['reveal_depth'])}{splay}{outer}")


FAMILY_TITLES = {"circle": ("cerchio", "circle"), "pointed": ("acuto", "pointed"), "ellipse": ("ellisse", "ellipse")}


def candidates_text(p: dict) -> str:
    """The RMS of the fitted type and of the others tried: "rms 1,3 mm; cerchio 45,8, ellisse 9,9"."""
    from .texts import num

    rms = p.get("rms")
    head = f"rms {num((rms or 0.0) * 1000, 1)} mm"
    others = [f"{t(*FAMILY_TITLES[k])} {num(v * 1000, 1)}" for k, v in (p.get("candidates_rms") or {}).items()
              if k != p.get("family") and k in FAMILY_TITLES]
    return head + ("; " + ", ".join(others) if others else "")


def export_step(chunk, ids, target) -> list[Path]:
    """The STEP files of the elements ``ids`` copied to ``target`` (a folder, or a file for one element)."""
    check_frame(chunk)
    target = Path(target)
    out = []
    chosen = [need(chunk, i) for i in ids] if ids else elements(chunk)
    check_fresh(chosen)
    for e in chosen:
        src = path_of(chunk, e, "step")
        if src is None or not src.exists():
            raise NotReady(t(f"Lo STEP di «{e['name']}» non c'è: ricostruisci l'elemento.",
                             f"The STEP of '{e['name']}' is missing: build the element again."))
        if target.suffix.lower() in (".step", ".stp") and len(chosen) == 1:
            dst = target
        else:
            target.mkdir(parents=True, exist_ok=True)
            dst = target / f"{_safe(e['name'])}.step"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        # the STEP's numbers are metres from the chunk's origin: the origin goes with it, in a file beside it
        side = dst.with_name(dst.stem + ".origin.json")
        side.write_text(json.dumps({"element": e["name"], "units": "metres", "origin": origin(chunk).tolist(),
                                    "crs": chunk.crs, "placement": (e.get("geometry") or {}).get("placement"),
                                    "note": "world = STEP coordinates + origin (in the CRS)"}, indent=1),
                        encoding="utf-8")
        out.append(dst)
    return out


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_ " else "_" for c in name).strip() or "elemento"


# ====================================================================== rebuild after the chunk's system changed
def rebuild(chunk, progress=None) -> list[dict]:
    """Every element made again after the chunk's transform or system changed, in place (same ids, so "same axis as"
    holds): what each was made from (region, path, plane, point, knots, type, thickness, depth) and the parameters
    edited by hand (knot radii, type, thickness, depth, simplification) carried from the old frame to the new one
    through the model (old world -> the chunk's internal frame -> new world), then measured again on the model. All or
    nothing: on a failure the elements and their files stay as they were."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    st = state(chunk)
    old_entries = copy.deepcopy(st.get("elements") or [])
    if not old_entries:
        return []
    order = [e["id"] for e in old_entries]
    # an opening closed by an arch is made after the arch
    old_entries = [e for e in old_entries if e["kind"] != "opening"] + \
        [e for e in old_entries if e["kind"] == "opening"]
    fr = st.get("frame") or {"transform": np.asarray(chunk.transform, float).tolist(), "crs": chunk.crs}
    M = np.asarray(chunk.transform, float) @ np.linalg.inv(np.asarray(fr["transform"], float))
    scale = float(np.cbrt(abs(np.linalg.det(M[:3, :3]))))
    saved = {k: copy.deepcopy(st.get(k)) for k in ("elements", "origin", "frame")}
    backup = folder(chunk) / f".rebuild-{time.strftime('%Y%m%d%H%M%S')}"
    backup.mkdir(parents=True, exist_ok=True)
    for e in old_entries:  # the old files aside: the new ones take the same folders
        if element_dir(chunk, e).exists():
            element_dir(chunk, e).rename(backup / e["id"])
    st["elements"], st["origin"], st["frame"] = [], None, None
    made = []
    try:
        for k, e in enumerate(old_entries):
            sub = progress.sub(k / len(old_entries), 1.0 / len(old_entries))
            made.append(_rebuild_one(chunk, e, M, scale, sub))
    except BaseException:
        for e in old_entries:
            shutil.rmtree(element_dir(chunk, e), ignore_errors=True)
            if (backup / e["id"]).exists():
                (backup / e["id"]).rename(element_dir(chunk, e))
        shutil.rmtree(backup, ignore_errors=True)
        st.update(saved)
        raise
    shutil.rmtree(backup, ignore_errors=True)
    st["elements"] = sorted(st["elements"], key=lambda x: order.index(x["id"]) if x["id"] in order else len(order))
    return sorted(made, key=lambda x: order.index(x["id"]))


def rebuild_stale(chunk, progress=None) -> list[dict]:
    """The openings waiting to be made again (their arch changed), each in place from what it was made of: on a
    failure that opening keeps its old entry and files (still marked)."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    check_frame(chunk)
    stale = [e for e in elements(chunk) if e.get("stale")]
    made = []
    for k, e in enumerate(stale):
        sub = progress.sub(k / len(stale), 1.0 / len(stale))
        made.append(_rebuild_one(chunk, copy.deepcopy(e), np.eye(4), 1.0, sub))
    return made


def _rebuild_one(chunk, e: dict, M, scale: float, progress) -> dict:
    src, p = e["source"], e.get("params") or {}

    def world(points):
        return engine.apply(M, np.asarray(points, float))

    edited = list(e.get("edited") or [])
    common = {"name": e["name"], "notes": e.get("notes", ""), "progress": progress}
    carry = {"id": e["id"], "edited": edited}
    if "knot_radii" in edited and p.get("knot_radii") is not None:
        carry["knot_radii"] = [r * scale for r in p["knot_radii"]]
    if e["kind"] == "column":
        box = poly = zmin = zmax = None
        if src.get("box"):
            b = np.asarray(src["box"], float)
            corners = np.array([[x, y, z] for x in b[[0, 3]] for y in b[[1, 4]] for z in b[[2, 5]]])
            w = world(corners)
            box = [*w.min(axis=0), *w.max(axis=0)]
        else:
            zl = src.get("zmin") if src.get("zmin") is not None else p["axis_point"][2]
            zh = src.get("zmax") if src.get("zmax") is not None else p["world_top"][2] if p.get("world_top") else zl
            mid = 0.5 * (zl + zh)
            poly = world(np.column_stack([np.asarray(src["polygon"], float), np.full(len(src["polygon"]), mid)]))
            zc = world(np.array([[*np.mean(src["polygon"], axis=0), zl], [*np.mean(src["polygon"], axis=0), zh]]))[:, 2]
            zmin, zmax = (float(zc.min()), float(zc.max())) if src.get("zmin") is not None or \
                src.get("zmax") is not None else (None, None)
            poly = poly[:, :2]
        return column(chunk, polygon=poly, zmin=zmin, zmax=zmax, box=box, knots=p.get("knots", 12),
                      as_measured=p.get("method") == "measured", same_axis=src.get("same_axis"), _rebuild=carry,
                      **common)
    if e["kind"] == "moulding":
        win = src.get("window")
        return moulding(chunk, world(src["path"]), station=src["station"] * scale,
                        window=None if win is None else [x * scale for x in win], out=src["out"],
                        depth=src["depth"] * scale, simplify=src["simplify"] * scale,
                        section=(np.asarray(e["geometry"]["section"]) * scale) if src.get("section_given") else None,
                        _rebuild=carry, **common)
    if e["kind"] == "opening":
        at = world(np.array([[*src["at"], src["cut_height"]]]))[0]
        thr = np.asarray(p["world_threshold"], float)
        sill = world(np.array([[thr[0], thr[1], p["sill"]]]))[0][2] if "sill" in edited else None
        lintel = world(np.array([[thr[0], thr[1], p["lintel"]]]))[0][2] if "lintel" in edited else None
        if "width" in edited:
            carry["width"] = p["width"] * scale
        if "thickness" in edited:
            carry["thickness"] = p["reveal_depth"] * scale
        # the cut's height again only when the user gave it: else the opening's middle is found again
        return opening(chunk, at[:2], height=float(at[2]) if src.get("height") is not None else None,
                       top=None if src.get("top") == "flat" else src.get("top"),
                       thickness=src["thickness"] * scale if src.get("thickness") else None, kind=src["kind"],
                       sill=sill, lintel=lintel, outer=src.get("outer") or "auto", _rebuild=carry, **common)
    pl = src["plane"]
    o = np.asarray(pl["origin"], float)
    new_o = world(o[None])[0]
    new_u = world((o + np.asarray(pl["u"]))[None])[0] - new_o
    new_v = world((o + np.asarray(pl["v"]))[None])[0] - new_o
    return arch(chunk, [x * scale for x in src["pick"]], kind=p.get("type") or src.get("kind_asked") or "auto",
                thickness=p["thickness"] * scale, depth=p["depth"] * scale, vault=e["kind"] == "vault",
                plane=(new_o, new_u, new_v), _rebuild=carry, **common)

