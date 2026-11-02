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


def remove(chunk, eid: str) -> bool:
    e = entry(chunk, eid)
    if e is None:
        return False
    shutil.rmtree(element_dir(chunk, e), ignore_errors=True)
    state(chunk)["elements"] = [x for x in elements(chunk) if x["id"] != e["id"]]
    return True


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
    """The chunk's export origin (see the module): chosen once, from the model."""
    st = state(chunk)
    if st.get("origin") is None:
        v = _world_vertices(chunk)
        st["origin"] = np.floor(v.min(axis=0)).tolist()
    return np.asarray(st["origin"], np.float64)


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


def _base_entry(chunk, kind: str, name: str | None, notes: str = "") -> dict:
    return {"id": uuid.uuid4().hex[:8], "kind": kind, "name": name or _new_name(chunk, kind), "notes": notes,
            "made": _now(), "chunk": chunk.label}


# ====================================================================== column
def column(chunk, polygon=None, zmin=None, zmax=None, box=None, knots: int = 12, as_measured: bool = False,
           same_axis: str | None = None, name: str | None = None, notes: str = "", progress=None) -> dict:
    """A column (or a part of one: base, shaft, capital) from the model inside a plan ``polygon`` (world x, y)
    between the heights ``zmin`` and ``zmax`` (world), or inside a world ``box`` (xmin, ymin, zmin, xmax, ymax,
    zmax). ``knots``: of the spline (``as_measured``: every point of the measured profile); ``same_axis``: the
    id or the name of a column whose axis this part shares."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    check_metric(chunk)
    progress(0.02, t("Lettura del modello", "Reading the model"))
    v, f = local_mesh(chunk)
    o = origin(chunk)
    poly = None if polygon is None else np.asarray(polygon, np.float64)[:, :2] - o[:2]
    lbox = None if box is None else np.r_[np.asarray(box[:3], float) - o, np.asarray(box[3:], float) - o]
    sv, sf = engine.select_faces(v, f, poly, None if zmin is None else zmin - o[2], None if zmax is None else zmax - o[2],
                                 lbox)
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
    e = _base_entry(chunk, "column", name, notes)
    e["source"] = {"chunk": chunk.label, "polygon": None if polygon is None else np.asarray(polygon)[:, :2].tolist(),
                   "zmin": zmin, "zmax": zmax, "box": None if box is None else list(map(float, box)),
                   "same_axis": same_axis, "faces": int(len(sf))}
    e["profile"] = {"heights": meas.heights.tolist(), "radii": meas.radii.tolist(), "spread": meas.spread.tolist(),
                    "step": meas.step}
    e["axis"] = {"point": (meas.point).tolist(), "direction": meas.direction.tolist()}  # export frame, base point
    _column_params(e, knots, as_measured, meas.point + o, meas.direction, rms, meas.height)
    return _finish_column(chunk, e, sv, progress)


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
    e = need(chunk, eid)
    p = e["params"]
    if knots is not None or as_measured is not None:
        _column_params(e, knots or p["knots"], bool(as_measured) if as_measured is not None else p["method"] == "measured",
                       np.asarray(p["axis_point"]), np.asarray(p["axis_direction"]), p["axis_rms"], p["height"])
        p = e["params"]
    if knot_radii is not None:
        rk = np.asarray(knot_radii, float)
        if len(rk) != len(p["knot_heights"]):
            raise NotReady(t(f"Servono {len(p['knot_heights'])} raggi, uno per nodo.",
                             f"{len(p['knot_heights'])} radii are needed, one per knot."))
        zk = np.asarray(p["knot_heights"])
        curve = engine.profile_curve(zk, rk) if p["method"] == "spline" else np.column_stack([rk, zk])
        p.update(knot_radii=list(map(float, rk)), radius_base=float(curve[0, 0]), radius_top=float(curve[-1, 0]),
                 radius_max=float(curve[:, 0].max()), entasis=engine.entasis(curve))
        e["geometry"]["profile"] = curve.tolist()
    e["made"] = _now()
    v, f = local_mesh(chunk)
    points = _column_points(chunk, e, v, f)
    return _finish_column(chunk, e, points, progress)


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
def moulding(chunk, path, station: float | None = None, window=(-0.5, -0.5, 0.5, 0.5), out: float | None = None,
             depth: float = 0.1, simplify: float = 0.002, section=None, height: float | None = None,
             name: str | None = None, notes: str = "", progress=None) -> dict:
    """A moulding along ``path`` (world points, (N, 2) in plan with ``height``, or (N, 3)). The section: the model
    cut by the plane normal to the path at ``station`` metres from its start (the middle of the first segment when
    None), kept in ``window`` ``(u0, v0, u1, v1)`` around the path point (u = up x tangent, v up), the polyline
    nearest the path closed against the wall ``depth`` metres inside it, simplified (Douglas-Peucker, ``simplify``
    metres); ``out`` +1 / -1: the side where it projects (default: the side of the photos); or ``section`` given as
    (u, v) points. The solid: the section swept along the path with mitred corners (an extrusion when straight)."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    check_metric(chunk)
    o = origin(chunk)
    v, f = local_mesh(chunk)
    pth = _path_3d(np.asarray(path, np.float64), v, f, o, height)
    progress(0.1, t("Sezione della modanatura", "Section of the moulding"))
    if station is None:
        station = 0.5 * float(np.linalg.norm(pth[1] - pth[0]))
    point, tangent = engine.point_at(pth, station)
    so, u, vv, _n = engine.section_frame(tangent, point)
    if out is None:
        out = _photo_side(chunk, point + o, u)
    lines = []
    if section is None:
        lines = engine.mesh_section(v, f, so, u, vv)
        sec = engine.moulding_section(lines, window, out=out, depth=depth, simplify=simplify)
    else:
        sec = np.asarray(section, float)
        if engine.polygon_area(sec) < 0:
            sec = sec[::-1]
    e = _base_entry(chunk, "moulding", name, notes)
    straight = engine.is_straight(pth)
    e["source"] = {"chunk": chunk.label, "path": (pth + o).tolist(), "station": float(station), "window": list(window),
                   "out": float(out), "depth": float(depth), "simplify": float(simplify), "height": height,
                   "section_given": section is not None}
    e["profile"] = {"section": sec.tolist(), "cut": [ln.tolist() for ln in lines][:50],
                    "station_point": (point + o).tolist()}
    _moulding_params(e, pth, sec, straight)
    return _finish_moulding(chunk, e, v, progress)


def _moulding_params(e, pth, sec, straight) -> None:
    area = abs(engine.polygon_area(sec))
    e["params"] = {"method": "extrusion" if straight else "sweep", "corners": "mitre", "length": engine.path_length(pth),
                   "section_area": area, "projection": float(np.ptp(np.asarray(sec)[:, 0])),
                   "section_height": float(np.ptp(np.asarray(sec)[:, 1])), "vertices": int(len(sec)),
                   "path_vertices": int(len(pth))}
    e["geometry"] = {"path": np.asarray(pth).tolist(), "section": np.asarray(sec).tolist(),
                     "placement": np.eye(4).tolist()}


def _finish_moulding(chunk, e, v, progress) -> dict:
    progress(0.3, t("Solido (CadQuery)", "Solid (CadQuery)"))
    g = e["geometry"]
    build_solid(chunk, e, {"placement": g["placement"],
                           "moulding": {"path": g["path"], "section": g["section"]}},
                progress.sub(0.3, 0.5))
    progress(0.85, t("Distanza del modello dal solido", "Distance of the model from the solid"))
    measure_fit(chunk, e, _points_near_box(v, e["solid"]["bbox"]))
    _save(chunk, e)
    progress(1.0, t("Modanatura pronta", "Moulding ready"))
    return e


def edit_moulding(chunk, eid: str, simplify: float | None = None, depth: float | None = None, progress=None) -> dict:
    """The solid of a moulding again with another simplification or wall depth, from the cut kept in the element."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    e = need(chunk, eid)
    src = e["source"]
    if simplify is not None:
        src["simplify"] = float(simplify)
    if depth is not None:
        src["depth"] = float(depth)
    pth = np.asarray(e["geometry"]["path"], float)
    if not src.get("section_given") and e["profile"].get("cut"):
        lines = [np.asarray(x, float) for x in e["profile"]["cut"]]
        sec = engine.moulding_section(lines, src["window"], out=src["out"], depth=src["depth"], simplify=src["simplify"])
    else:
        sec = np.asarray(e["geometry"]["section"], float)
    _moulding_params(e, pth, sec, engine.is_straight(pth))
    e["made"] = _now()
    v, _f = local_mesh(chunk)
    return _finish_moulding(chunk, e, v, progress)


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
         progress=None) -> dict:
    """An arch (or, ``vault``, a barrel vault) on a section plane: ``pick`` (u, v of the plane) a point inside the
    opening, under the intrados. The intrados is what the point sees of the model's cut (the jambs below the
    springing line are part of the fit), the extrados what lies beyond it. ``kind``: "auto", "semicircular",
    "segmental", "pointed", "elliptical", "measured". ``thickness``: of the ring (the median distance to the extrados
    when None); ``depth``: along the plane's normal (the extent of the model's points within the ring when None)."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    check_metric(chunk)
    v, f = local_mesh(chunk)
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
    fit = engine.fit_arch(near, kind)
    measured_thickness = engine.arch_thickness(fit, far)
    th = float(thickness) if thickness else measured_thickness
    if not th:
        raise NotReady(t("Lo spessore dell'arco non si misura (nessun estradosso visibile): dallo tu.",
                         "The thickness of the arch cannot be measured (no extrados seen): give it."))
    e = _base_entry(chunk, "vault" if vault else "arch", name, notes)
    e["source"] = {"chunk": chunk.label, **what, "pick": list(map(float, pick)), "kind_asked": kind,
                   "plane": {"origin": (po + origin(chunk)).tolist(), "u": u.tolist(), "v": vv.tolist()}}
    e["profile"] = {"intrados": near.tolist(), "extrados": far.tolist(), "fit": fit.to_dict(),
                    "thickness_measured": measured_thickness}
    e["_plane"] = {"origin": po.tolist(), "u": u.tolist(), "v": vv.tolist()}
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
    e.pop("_plane")
    return _finish_arch(chunk, e, v, progress)


def _buffer(poly, d):
    from shapely.geometry import Polygon

    return np.asarray(Polygon(poly).buffer(d).exterior.coords)


def _arch_params(e, fit: engine.ArchFit, thickness: float) -> None:
    outline = engine.arch_outline(fit, thickness)
    poly = engine.outline_polygon(outline)
    p = {"type": fit.type, "family": fit.family, "span": fit.span, "rise": fit.rise, "springing": fit.springing,
         "centre_u": fit.centre_u, "thickness": float(thickness), "rms": fit.rms, "points": fit.points,
         "section_area": abs(engine.polygon_area(poly))}
    fp = fit.params
    if fit.family == "circle":
        p.update(radius=fp["R"], centre_v=fp["vc"])
    elif fit.family == "pointed":
        p.update(radius=fp["R"], centres_distance=2 * (fp["R"] - fp["s"]))
    elif fit.family == "ellipse":
        p.update(semi_axis_u=fp["a"], semi_axis_v=fp["b"], centre_v=fp["vc"])
    e["params"] = p
    e["geometry"] = {"outline": outline}


def _finish_arch(chunk, e, v, progress) -> dict:
    progress(0.3, t("Solido (CadQuery)", "Solid (CadQuery)"))
    g = e["geometry"]
    build_solid(chunk, e, {"placement": g["placement"], "arch": {"outline": g["outline"], "depth": g["depth"]}},
                progress.sub(0.3, 0.5))
    progress(0.85, t("Distanza del modello dal solido", "Distance of the model from the solid"))
    pl = np.asarray(g["placement"])
    rel = engine.apply(np.linalg.inv(pl), v)
    poly = _buffer(engine.outline_polygon(g["outline"]), 0.02)
    inside = engine.points_in_polygon(rel[:, :2], poly) & (rel[:, 2] > -0.02) & (rel[:, 2] < g["depth"] + 0.02)
    measure_fit(chunk, e, v[inside])
    _save(chunk, e)
    progress(1.0, t("Arco pronto", "Arch ready"))
    return e


def edit_arch(chunk, eid: str, kind: str | None = None, thickness: float | None = None, depth: float | None = None,
              progress=None) -> dict:
    """The solid of an arch again with another type, thickness or depth: the intrados as measured stays."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    e = need(chunk, eid)
    near = np.asarray(e["profile"]["intrados"], float)
    fit = engine.fit_arch(near, kind or e["params"]["type"]) if kind else _fit_from(e)
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
    e["made"] = _now()
    v, _f = local_mesh(chunk)
    return _finish_arch(chunk, e, v, progress)


def _fit_from(e) -> engine.ArchFit:
    d = dict(e["profile"]["fit"])
    return engine.ArchFit(d["type"], d["family"], d["params"], d["rms"], d["springing"], d["span"], d["rise"],
                          d["centre_u"], d["points"], d.get("candidates") or {}, d.get("measured"))


# ====================================================================== show, export
def summary(e: dict) -> str:
    """The element in a line: kind, main numbers, fit."""
    from .texts import arch_title, kind_title, metres

    p, fit = e.get("params") or {}, e.get("fit") or {}
    if e["kind"] == "column":
        main = t(f"h {metres(p['height'])}, r {metres(p['radius_base'])} → {metres(p['radius_top'])}, entasi "
                 f"{metres(p['entasis'])}", f"h {metres(p['height'])}, r {metres(p['radius_base'])} → "
                 f"{metres(p['radius_top'])}, entasis {metres(p['entasis'])}")
    elif e["kind"] == "moulding":
        main = t(f"lunghezza {metres(p['length'])}, sporgenza {metres(p['projection'])}",
                 f"length {metres(p['length'])}, projection {metres(p['projection'])}")
    else:
        main = t(f"{arch_title(p['type'])}, luce {metres(p['span'])}, freccia {metres(p['rise'])}, spessore "
                 f"{metres(p['thickness'])}", f"{arch_title(p['type'])}, span {metres(p['span'])}, rise "
                 f"{metres(p['rise'])}, thickness {metres(p['thickness'])}")
    gap = t(f"scarto mediana {metres(fit.get('median'))}, p90 {metres(fit.get('p90'))}",
            f"fit median {metres(fit.get('median'))}, p90 {metres(fit.get('p90'))}") if fit.get("points") else \
        t("scarto non misurato", "fit not measured")
    return f"{kind_title(e['kind'])}: {main}; {gap}"


def export_step(chunk, ids, target) -> list[Path]:
    """The STEP files of the elements ``ids`` copied to ``target`` (a folder, or a file for one element)."""
    target = Path(target)
    out = []
    chosen = [need(chunk, i) for i in ids] if ids else elements(chunk)
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
        out.append(dst)
    return out


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_ " else "_" for c in name).strip() or "elemento"
