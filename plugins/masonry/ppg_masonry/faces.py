"""The faces of a wall: elevations of the program (``engine.elevation.build_elevation``) on

* a vertical plane through two points of the plan (``plane_from_line``, ``v`` = the height): a wall face;
* a vertical cylinder - through three or more points of the plan, or the circle fitted to a horizontal slice of the
  model at a given height - seen from outside (a round tower) or from inside (a well, an apse);
* one plane per side of a polygonal tower: the outline of the horizontal slice at a given height simplified into a
  polygon, a face for every side longer than a minimum, each seen from outside.

A face of a stretch is limited to that stretch (``u`` between the two points, or along the side); its depth band
(``back`` behind the plane, ``front`` in front of it) keeps what stands in front of the wall out of it."""
from __future__ import annotations

import numpy as np

from .texts import t
from .walls import NotReady, add_face, need

DEFAULT_FRONT, DEFAULT_BACK = 0.5, 0.3
CYLINDER_BAND_SHARE, CYLINDER_BAND_MIN = 0.15, 0.5


def cylinder_band(radius: float) -> float:
    """The default depth band of a cylindrical face on each side: 15 % of the radius, at least half a metre."""
    return max(CYLINDER_BAND_SHARE * float(radius), CYLINDER_BAND_MIN)


def world_vertices(chunk) -> np.ndarray:
    from pyarchinit_photogrammetry.io.mesh_ply import read_mesh

    path = chunk.product_path("model")
    if path is None:
        raise NotReady(t("Il chunk non ha un modello: le facce si fanno sul modello.",
                         "The chunk has no model: the faces are made on the model."))
    return chunk.to_world(np.asarray(read_mesh(path).xyz, np.float64))


def _stretch_bounds(chunk, frame, length: float, depth: tuple[float, float], verts=None):
    """``(0, vmin, length, vmax)``: the model within the depth band and the stretch, or None when nothing is there."""
    verts = world_vertices(chunk) if verts is None else verts
    uvd = frame.coords(verts)
    keep = (uvd[:, 2] >= depth[0]) & (uvd[:, 2] <= depth[1]) & (uvd[:, 0] >= 0.0) & (uvd[:, 0] <= length)
    if keep.sum() < 3:
        return None
    return (0.0, float(uvd[keep, 1].min()), float(length), float(uvd[keep, 1].max()))


MAX_SIDE_PX = 20000  # a face larger than this on a side is refused unless forced
MIN_RESOLUTION = 0.001


class FaceTooLarge(NotReady):
    """A face that would exceed ``MAX_SIDE_PX`` pixels on a side: ``size`` (w, h) and the ``proposed`` resolution."""

    def __init__(self, size, resolution: float, proposed: float):
        self.size, self.resolution, self.proposed = size, resolution, proposed
        super().__init__(t(f"La faccia sarebbe di {size[0]} x {size[1]} pixel a {resolution * 1000:.1f} mm per pixel "
                           f"(più di {MAX_SIDE_PX} su un lato): usa {proposed * 1000:.1f} mm per pixel, o costruiscila "
                           "comunque (--force).", f"The face would be {size[0]} x {size[1]} pixels at "
                           f"{resolution * 1000:.1f} mm per pixel (over {MAX_SIDE_PX} on a side): use "
                           f"{proposed * 1000:.1f} mm per pixel, or build it anyway (--force)."))


def model_resolution(chunk, sample: int = 200_000) -> float:
    """The default resolution of a face: the median edge length of the model's triangles (world metres), at least a
    millimetre - the detail the model has."""
    from pyarchinit_photogrammetry.io.mesh_ply import read_mesh

    mesh = read_mesh(chunk.product_path("model"))
    faces_ = np.asarray(mesh.faces, np.int64)
    if len(faces_) > sample:
        faces_ = faces_[np.random.default_rng(0).choice(len(faces_), sample, replace=False)]
    used, inverse = np.unique(faces_, return_inverse=True)
    xyz = chunk.to_world(np.asarray(mesh.xyz, np.float64)[used])
    f = inverse.reshape(faces_.shape)
    edges = np.linalg.norm(xyz[f[:, [1, 2, 0]]] - xyz[f], axis=2).ravel()
    return float(max(np.median(edges), MIN_RESOLUTION))


def check_size(chunk, bounds, resolution: float | None, force: bool = False) -> float:
    """The resolution of a face (``None``: :func:`model_resolution`), refused with :class:`FaceTooLarge` when the
    face would exceed ``MAX_SIDE_PX`` on a side, unless ``force``."""
    res = float(resolution) if resolution else model_resolution(chunk)
    res = max(res, MIN_RESOLUTION)
    u0, v0, u1, v1 = bounds
    size = (int(np.ceil((u1 - u0) / res)) + 1, int(np.ceil((v1 - v0) / res)) + 1)
    if max(size) > MAX_SIDE_PX and not force:
        proposed = float(np.ceil(max(u1 - u0, v1 - v0) / (MAX_SIDE_PX - 2) * 10000.0) / 10000.0)
        raise FaceTooLarge(size, res, proposed)
    return res


def _band_bounds(chunk, frame, depth):
    """``(u0, v0, u1, v1)`` of the model within the depth band of ``frame``, or None."""
    uvd = frame.coords(world_vertices(chunk))
    keep = (uvd[:, 2] >= depth[0]) & (uvd[:, 2] <= depth[1])
    if keep.sum() < 3:
        return None
    return (float(uvd[keep, 0].min()), float(uvd[keep, 1].min()), float(uvd[keep, 0].max()),
            float(uvd[keep, 1].max()))


def _build(chunk, wid, frame, name, resolution, depth, bounds, progress):
    from pyarchinit_photogrammetry.engine.elevation import build_elevation

    entry = build_elevation(chunk, frame, resolution, name, depth, progress=progress, bounds=bounds)
    add_face(chunk, wid, entry["id"])
    return entry


def plane_face(chunk, wid: str, p1, p2, side=None, resolution: float | None = None, front: float = DEFAULT_FRONT,
               back: float = DEFAULT_BACK, name: str | None = None, progress=None, force: bool = False,
               dry_run: bool = False) -> dict:
    """The face of a wall on the vertical plane through ``p1``-``p2`` (plan points, chunk CRS) seen from ``side`` (a
    plan point on the viewer's side; default the side of the photos), between the two points; ``v`` is the height.
    Returns the program's elevation entry, listed as a face of the wall ``wid``. ``resolution`` None: the model's
    median edge (:func:`check_size`, which refuses a face over 20 000 pixels on a side unless ``force``);
    ``dry_run``: only ``{"resolution", "bounds"}``, nothing built."""
    from pyarchinit_photogrammetry.engine.elevation import plane_from_line, with_heights

    w = need(chunk, wid)
    a, b = np.asarray(p1, np.float64)[:2], np.asarray(p2, np.float64)[:2]
    length = float(np.linalg.norm(b - a))
    if length < 1e-3:
        raise NotReady(t("I due punti coincidono: traccia la faccia del muro in pianta.",
                         "The two points coincide: trace the face of the wall in plan."))
    frame = with_heights(plane_from_line(chunk, a, b, side))
    depth = (-float(back), float(front))
    bounds = _stretch_bounds(chunk, frame, length, depth)
    if bounds is None:
        raise NotReady(t("Nessuna parte del modello lungo la traccia entro la fascia di profondità: allarga la fascia "
                         "o traccia la faccia sul muro.", "No part of the model along the trace within the depth band: "
                         "widen the band or trace the face on the wall."))
    resolution = check_size(chunk, bounds, resolution, force)
    if dry_run:
        return {"resolution": resolution, "bounds": bounds}
    return _build(chunk, w["id"], frame, name or t(f"{w['name']} - faccia {len(w['faces']) + 1}",
                                                   f"{w['name']} - face {len(w['faces']) + 1}"),
                  resolution, depth, bounds, progress)


def slice_rings(chunk, height: float) -> list[np.ndarray]:
    """The lines (``(N, 2)`` plan coordinates) where the horizontal plane at ``height`` cuts the model."""
    from pyarchinit_photogrammetry.archaeo.sections import plan_slice

    return [np.asarray(p, np.float64)[:, :2] for p in plan_slice(chunk, float(height)).polylines if len(p) >= 2]


MIN_COVERAGE = 90.0  # degrees of arc a slice must cover to give a circle


def slice_circle(chunk, height: float, near=None, rings=None) -> dict:
    """The circle fitted to the horizontal slice of the model at ``height``: on ALL the slice's points, every piece of a
    broken ring (:func:`fit.ransac_circle`: RANSAC on three-point circles with a band of max(2 % of the radius, 5 cm),
    then least squares on the ring within max(15 % of the radius, 25 cm) of that circle, so that an oval ring gives its
    mean circle and not the curve of one end; :func:`shape.tower_axis` keeps out the points far from the previous
    ring, which guards against other structures near the tower). ``near``: ``(centre, radius)`` expected (a face's
    cylinder, the circle at the previous height): the points farther than 1.5 radii + 1 m from that centre are left
    out. ``centre``, ``radius``,
    ``rms``, ``out_of_round`` (the spread of the inliers' radius, 0.5th to 99.5th percentile), ``points``, ``pieces``
    (the lines of the slice holding inliers), ``coverage`` (degrees of arc). An arc under 90 degrees is refused."""
    from .fit import ransac_circle

    rings = slice_rings(chunk, height) if rings is None else rings
    rings = [np.asarray(r, np.float64)[:, :2] for r in rings if len(r) >= 2]
    if not rings:
        raise NotReady(t(f"Il modello non ha una sezione a quota {height:.3f}.",
                         f"The model has no slice at height {height:.3f}."))
    label = np.concatenate([np.full(len(r), k) for k, r in enumerate(rings)])
    pts = np.vstack(rings)
    if isinstance(near, tuple) and len(near) == 2 and near[1] is not None:  # (centre, radius) expected
        c, r = np.asarray(near[0], np.float64), float(near[1])
        keep = np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]) <= 1.5 * r + 1.0
        pts, label = pts[keep], label[keep]
    if len(pts) < 8:
        raise NotReady(t(f"Il modello non ha una sezione a quota {height:.3f}.",
                         f"The model has no slice at height {height:.3f}."))
    f = ransac_circle(pts)
    if f["coverage"] < MIN_COVERAGE:
        raise NotReady(t(f"Sezione a quota {height:.3f}: arco di {f['coverage']:.0f} gradi, troppo poco per un "
                         "cerchio.", f"Slice at height {height:.3f}: an arc of {f['coverage']:.0f} degrees, too little "
                         "for a circle."))
    lo, hi = np.percentile(f["residuals"], [0.5, 99.5])
    pieces = int(sum(1 for k in np.unique(label[f["inliers"]]) if (label[f["inliers"]] == k).sum() >= 5))
    return {"height": float(height), "centre": [float(x) for x in f["centre"]], "radius": f["radius"],
            "rms": f["rms"], "out_of_round": float(hi - lo), "points": int(f["inliers"].sum()), "pieces": pieces,
            "coverage": f["coverage"]}


def cylinder_face(chunk, wid: str, points=None, height: float | None = None, inside: bool = False,
                  resolution: float | None = None, front: float | None = None, back: float | None = None,
                  name: str | None = None, near=None, progress=None, force: bool = False,
                  dry_run: bool = False) -> dict:
    """The developed face of a round tower (``inside`` False) or of a well, an apse (``inside``): a vertical cylinder
    through ``points`` (three or more on the plan) or fitted to the horizontal slice of the model at ``height``. The
    depth band, ``front`` and ``back`` not given: 15 % of the radius each side, at least 0.5 m. ``resolution``,
    ``force``, ``dry_run``: as :func:`plane_face`."""
    from pyarchinit_photogrammetry.engine.elevation import Cylinder

    w = need(chunk, wid)
    if points is not None:  # the circle fitted with the points centred: map coordinates lose digits otherwise
        from .fit import robust_circle

        q = np.asarray(points, np.float64)[:, :2]
        if len(q) < 3:
            raise NotReady(t("Servono tre punti sulla pianta.", "Three points on the plan are needed."))
        c = robust_circle(q)
        frame = Cylinder([c["centre"][0], c["centre"][1], 0.0], [0.0, 0.0, 1.0], c["radius"], inside,
                         [q[0, 0] - c["centre"][0], q[0, 1] - c["centre"][1], 0.0])
    elif height is not None:
        c = slice_circle(chunk, height, near=near)
        frame = Cylinder([c["centre"][0], c["centre"][1], 0.0], [0.0, 0.0, 1.0], c["radius"], inside)
    else:
        raise NotReady(t("Servono tre punti sulla pianta o la quota della sezione.",
                         "Three points on the plan or the height of the slice are needed."))
    band = cylinder_band(frame.radius)  # a battered or oval tower leaves a narrow band: 15 % of the radius
    depth = (-float(back if back else band), float(front if front else band))
    if inside:
        depth = (depth[0], min(depth[1], frame.radius))
    bounds = _band_bounds(chunk, frame, depth)
    if bounds is None:
        raise NotReady(t("Nessuna parte del modello nella fascia di profondità del cilindro.",
                         "No part of the model within the cylinder's depth band."))
    resolution = check_size(chunk, bounds, resolution, force)
    if dry_run:
        return {"resolution": resolution, "bounds": bounds, "depth": depth, "radius": frame.radius}
    title = t("sviluppo interno", "inner development") if inside else t("sviluppo esterno", "outer development")
    return _build(chunk, w["id"], frame, name or f"{w['name']} - {title}", resolution, depth, bounds, progress)


def closed_rings(lines: list[np.ndarray], gap: float = 0.05) -> list[np.ndarray]:
    """The lines of a slice joined end to end where their ends are within ``gap`` metres (a mesh made of pieces, a
    small hole in it), kept when they close; each ring returned closed (first point repeated at the end)."""
    pool = [np.asarray(x, np.float64) for x in lines if len(x) >= 2]
    rings = []
    while pool:
        line = pool.pop(0)
        while True:
            if len(line) > 3 and np.linalg.norm(line[-1] - line[0]) <= gap:
                rings.append(np.vstack([line, line[:1]]) if not np.allclose(line[-1], line[0]) else line)
                break
            best, flip, dist = None, False, gap
            for i, other in enumerate(pool):
                for rev in (False, True):
                    end = other[-1] if rev else other[0]
                    d = float(np.linalg.norm(end - line[-1]))
                    if d <= dist:
                        best, flip, dist = i, rev, d
            if best is None:
                break
            other = pool.pop(best)
            line = np.vstack([line, other[::-1] if flip else other])
    return rings


def slice_polygon(chunk, height: float, tolerance: float | None = None, gap: float = 0.05):
    """The outline of the horizontal slice at ``height`` as a shapely polygon (the largest closed line, its pieces
    joined across gaps up to ``gap`` metres), simplified by ``tolerance`` metres (default 1 % of the square root of its
    area, at least 2 cm)."""
    from shapely.geometry import Polygon

    best = None
    for ring in closed_rings(slice_rings(chunk, height), gap):
        poly = Polygon(ring).buffer(0)
        if poly.is_empty:
            continue
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)
        if best is None or poly.area > best.area:
            best = poly
    if best is None:
        raise NotReady(t(f"Il modello non ha una sezione chiusa a quota {height:.3f}.",
                         f"The model has no closed slice at height {height:.3f}."))
    tol = max(0.02, 0.01 * float(np.sqrt(best.area))) if tolerance is None else float(tolerance)
    simple = best.simplify(tol, preserve_topology=False)
    if simple.is_empty or simple.geom_type != "Polygon":
        simple = best.simplify(tol, preserve_topology=True)
    return Polygon(_drop_straight(np.asarray(simple.exterior.coords)[:-1]))


def _drop_straight(ring: np.ndarray, degrees: float = 3.0) -> np.ndarray:
    """The corners of a ring where it turns by less than ``degrees`` taken out (a side split in two)."""
    ring = np.asarray(ring, np.float64)
    changed = True
    while changed and len(ring) > 3:
        changed = False
        for i in range(len(ring)):
            a, b, c = ring[i - 1], ring[i], ring[(i + 1) % len(ring)]
            d1, d2 = b - a, c - b
            n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
            if n1 < 1e-9 or n2 < 1e-9 or np.degrees(np.arccos(np.clip(d1 @ d2 / (n1 * n2), -1, 1))) < degrees:
                ring = np.delete(ring, i, axis=0)
                changed = True
                break
    return ring


def polygon_faces(chunk, wid: str, height: float, min_side: float = 0.5, tolerance: float | None = None,
                  resolution: float | None = None, front: float = DEFAULT_FRONT, back: float = DEFAULT_BACK,
                  progress=None, force: bool = False) -> list[dict]:
    """One face per side of a polygonal tower: the slice at ``height`` simplified into a polygon, a vertical plane
    for every side longer than ``min_side`` metres, seen from outside, limited to the side."""
    from pyarchinit_photogrammetry.core.progress import as_progress
    from pyarchinit_photogrammetry.engine.elevation import plane_from_line, with_heights
    from shapely.geometry import Point

    progress = as_progress(progress)
    w = need(chunk, wid)
    poly = slice_polygon(chunk, height, tolerance)
    ring = np.asarray(poly.exterior.coords)[:-1]
    sides = [(ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring))]
    sides = [(a, b) for a, b in sides if np.linalg.norm(b - a) >= min_side]
    if not sides:
        raise NotReady(t(f"Nessun lato più lungo di {min_side:g} m nella sezione.",
                         f"No side longer than {min_side:g} m in the slice."))
    verts = world_vertices(chunk)
    depth = (-float(back), float(front))
    resolution = resolution or model_resolution(chunk)
    import uuid

    slice_id = uuid.uuid4().hex[:8]  # the faces of one slice are the sides of one tower: never paired across it
    out = []
    for k, (a, b) in enumerate(sides):
        along = (b - a) / np.linalg.norm(b - a)
        normal = np.array([along[1], -along[0]])
        mid = (a + b) / 2
        if poly.contains(Point(*(mid + 0.01 * normal))):
            normal = -normal
        frame = with_heights(plane_from_line(chunk, a, b, side=mid + normal))
        bounds = _stretch_bounds(chunk, frame, float(np.linalg.norm(b - a)), depth, verts)
        if bounds is None:
            continue
        name = t(f"{w['name']} - lato {k + 1}", f"{w['name']} - side {k + 1}")
        res_k = check_size(chunk, bounds, resolution, force)
        entry = _build(chunk, w["id"], frame, name, res_k, depth, bounds,
                       progress.sub(k / len(sides), 1.0 / len(sides)))
        from .walls import _live

        _live(chunk, w["id"]).setdefault("slices", {})[entry["id"]] = slice_id
        out.append(entry)
    return out
