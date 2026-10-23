"""Point clouds through terrio: any format terrio reads brought into the project, the ground
classified with the PDAL-style filters, the dense cloud written as E57 or EPT.

Coordinates: terrio works in world coordinates (the CRS of the file or of the chunk); the chunk
keeps its cloud in the internal frame, ``chunk.transform`` away from the world. A new chunk made
from a file takes a round origin as its internal frame, as the laser scans of the program do.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from pyarchinit_photogrammetry.core.pointcloud import PointCloud
from pyarchinit_photogrammetry.core.progress import as_progress

from . import environment

METHODS = ("smrf", "csf", "pmf")
NOISE, GROUND, OTHER = 7, 2, 1
VEGETATION = ((0.5, 3), (2.0, 4), (np.inf, 5))  # height above the ground -> ASPRS low, medium, high vegetation
CHANGEABLE = (0, 1, 2)  # never classified, unclassified, ground: the classes the ground filters may change
VEGETATION_CLASSES = (3, 4, 5)  # changed too when the vegetation classes are asked for
# the defaults of the filters (terrio's, as PDAL's): the slope, and the height threshold - for PMF its
# initial distance, the first height step of its growing windows
DEFAULTS = {"smrf": {"slope": 0.15, "threshold": 0.5}, "csf": {"threshold": 0.5},
            "pmf": {"slope": 1.0, "threshold": 0.15}}
CLOUD_SUFFIXES = (".las", ".laz", ".ply", ".e57", ".ept", ".json")


def _arrays(points) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """``(xyz, rgb, classification)`` of a terrio point cloud: colours from the 16-bit LAS
    colours, else grey from the intensity; ``None`` where the file has none."""
    d = points.to_dict()
    xyz = np.column_stack([np.asarray(d["x"], np.float64), np.asarray(d["y"], np.float64),
                           np.asarray(d["z"], np.float64)])
    rgb = None
    if all(k in d for k in ("red", "green", "blue")):
        c = np.column_stack([d["red"], d["green"], d["blue"]]).astype(np.float64)
        if c.size and c.max() > 255:
            c = c / 257.0
        rgb = np.clip(np.round(c), 0, 255).astype(np.uint8)
    elif "intensity" in d:
        v = np.asarray(d["intensity"], np.float64)
        lo, hi = (float(v.min()), float(v.max())) if len(v) else (0.0, 1.0)
        grey = ((v - lo) / max(hi - lo, 1e-12) * 255).astype(np.uint8)
        rgb = np.repeat(grey[:, None], 3, axis=1)
    cls = np.asarray(d["classification"], np.uint8) if "classification" in d else None
    return xyz, rgb, cls


def read(source, max_points: int | None = None, classes=None, progress=None):
    """``(xyz, rgb, classification, crs)`` of a point cloud file or URL, in its own coordinates;
    ``max_points``: about that many (COPC and EPT read fewer octree levels, other files are
    thinned); ``classes``: only those classifications."""
    terrio = environment.load()
    progress = as_progress(progress)
    progress(0.0, f"Reading {Path(str(source)).name}")
    f = terrio.open_points(str(source))
    crs = f.crs or None
    kw = {}
    if max_points:
        kw["max_points"] = int(max_points)
    if classes:
        kw["classes"] = [int(c) for c in classes]
    xyz, rgb, cls = _arrays(f.read(**kw))
    if max_points and len(xyz) > int(max_points):  # COPC and EPT give whole octree levels: thin the rest here
        keep = np.sort(np.random.default_rng(0).choice(len(xyz), int(max_points), replace=False))
        xyz, rgb, cls = xyz[keep], (None if rgb is None else rgb[keep]), (None if cls is None else cls[keep])
    progress(1.0, f"{len(xyz):,} points")
    return xyz, rgb, cls, crs


def _same_crs(a: str, b: str) -> bool:
    if a == b:
        return True
    try:
        from pyarchinit_photogrammetry.core.crs import to_pyproj

        return to_pyproj(a) == to_pyproj(b)
    except Exception:
        return False


def import_cloud(doc, source, chunk=None, label: str | None = None, crs: str | None = None,
                 max_points: int | None = None, classes=None, progress=None):
    """The cloud of ``source`` as the dense cloud of a new chunk (its CRS the one of the file, or
    ``crs``, else local; internal frame from a round origin), or in place of the dense cloud of
    ``chunk``, which is kept as a version (the coordinates reprojected into the chunk CRS when the
    two are known and differ). Returns the chunk."""
    from pyarchinit_photogrammetry.core import crs as crs_mod

    progress = as_progress(progress)
    xyz, rgb, cls, file_crs = read(source, max_points, classes, progress.sub(0.0, 0.7))
    if not len(xyz):
        raise ValueError(f"{Path(str(source)).name}: no point")
    name = Path(str(source)).name
    source_crs = crs or file_crs
    if chunk is None:
        chunk = doc.add_chunk(label or name)
        chunk.crs = source_crs or crs_mod.LOCAL
        origin = np.floor(xyz.min(0))
        chunk.transform = np.eye(4)
        chunk.transform[:3, 3] = origin
        chunk.point_cloud = PointCloud(xyz - origin, rgb, classification=cls)
        chunk.meta["levelled"] = True
        chunk.meta["metric"] = True
    else:
        from pyarchinit_photogrammetry.core import versions

        if source_crs and crs_mod.is_local(chunk.crs):
            raise ValueError(f"{chunk.label} has no reference system and the file has one ({source_crs}): "
                             "import it into a new chunk")
        if source_crs and not _same_crs(source_crs, chunk.crs):
            progress(0.75, f"Reprojecting from {source_crs} to {chunk.crs}")
            xyz = crs_mod.transform_points(xyz, source_crs, chunk.crs)
        old = chunk.product_path("point_cloud")
        if old is not None and old.exists():
            versions.keep_active(chunk, "point_cloud", move=True)
        chunk.point_cloud = PointCloud(chunk.to_internal(xyz), rgb, classification=cls)
    chunk.meta["point_cloud_source"] = f"terrio: {name}"
    chunk.meta["terrio_import"] = {"source": str(source), "points": int(len(xyz)), "crs": file_crs or "",
                                   "max_points": int(max_points or 0), "classes": [int(c) for c in classes or []]}
    progress(1.0, f"{len(xyz):,} points in {chunk.label}")
    return chunk


def ground(chunk, method: str = "smrf", cell: float = 1.0, slope: float | None = None,
           threshold: float | None = None, window: float | None = None, resolution: float = 1.0, rigidness: int = 3,
           noise: bool = False, vegetation: bool = False, progress=None) -> dict:
    """Ground (class 2) and other points (class 1) of the dense cloud, with PDAL's SMRF, CSF or
    PMF filters run by terrio. Only the points in classes 0, 1 and 2 change (and 3, 4 and 5 with
    ``vegetation``): every other class - buildings, walls and the other archaeological classes
    (64 and up), noise, the classes of an imported lidar - stays as it is, and so do the points
    labelled by hand. The points already classed as noise (7, 18) are left out of the filters.
    ``slope`` and ``threshold`` (PMF: its initial distance) default to those of the filter
    (:data:`DEFAULTS`). ``noise``: low noise (class 7) found first with ELM and left out;
    ``vegetation``: the other points get the ASPRS vegetation classes 3, 4 and 5 by their height
    above the nearest ground points (below 0.5 m, below 2 m, above) - also walls, sections and
    spoil heaps that are not classified otherwise. Returns the classes counted."""
    from pyarchinit_photogrammetry.ai.classify import keep_manual_labels
    from pyarchinit_photogrammetry.core.pointcloud import NOISE_CLASSES

    terrio = environment.load()
    progress = as_progress(progress)
    if method not in METHODS:
        raise ValueError(f"method {method!r}: one of {', '.join(METHODS)}")
    pc = chunk.point_cloud
    if pc is None:
        raise RuntimeError("build the dense cloud first")
    defaults = DEFAULTS[method]
    slope = float(defaults.get("slope", 0.15) if slope is None else slope)
    threshold = float(defaults.get("threshold", 0.5) if threshold is None else threshold)
    before = pc.classification.copy()
    manual = pc.manual if pc.manual is not None else np.zeros(len(pc), bool)
    changeable = np.isin(before, CHANGEABLE + (VEGETATION_CLASSES if vegetation else ())) & ~manual
    used = np.nonzero(~np.isin(before, NOISE_CLASSES))[0]  # the noise found before is neither ground nor surface
    progress(0.05, f"{method.upper()}: {len(pc):,} points")
    codes = before.copy()
    if len(used):
        world = chunk.to_world(pc.xyz[used])
        cloud = terrio.PointCloud.from_dict({"x": world[:, 0], "y": world[:, 1], "z": world[:, 2]})
        ignore = None
        if noise:
            cloud = cloud.elm(cell=max(float(cell), 1.0), threshold=1.0, classification=NOISE)
            ignore = f"Classification[{NOISE}:{NOISE}]"
        if method == "smrf":
            kw = {"cell": float(cell), "slope": slope, "threshold": threshold}
            if window:
                kw["window"] = float(window)
            cloud = cloud.smrf(ignore=ignore, ground_class=GROUND, other_class=OTHER, **kw)
        elif method == "csf":
            cloud = cloud.csf(resolution=float(resolution), rigidness=int(rigidness), threshold=threshold,
                              ignore=ignore, ground_class=GROUND, other_class=OTHER)
        else:
            kw = {"cell_size": float(cell), "slope": slope, "initial_distance": threshold}
            if window:
                kw["max_window_size"] = float(window)
            cloud = cloud.pmf(ignore=ignore, ground_class=GROUND, other_class=OTHER, **kw)
        progress(0.7, "Classes")
        found = np.asarray(cloud.to_dict()["classification"], np.uint8)
        if len(found) != len(used):
            raise RuntimeError(f"terrio returned {len(found)} points for {len(used)}")
        proposed = before.copy()
        proposed[used] = found
        codes[changeable] = proposed[changeable]
        if vegetation:
            progress(0.8, "Height above the ground")
            # above the ground points of the result: those found now and those of other classes kept as ground
            ground_cloud = terrio.PointCloud.from_dict({"x": world[:, 0], "y": world[:, 1], "z": world[:, 2],
                                                        "classification": codes[used]})
            hag = np.asarray(ground_cloud.hag_nn(ground_class=GROUND).to_dict()["height_above_ground"], np.float64)
            veg = np.full(len(used), VEGETATION[-1][1], np.uint8)
            for limit, code in reversed(VEGETATION[:-1]):
                veg[hag < limit] = code
            other = changeable[used] & (codes[used] == OTHER)
            codes[used[other]] = veg[other]
    keep_manual_labels(pc, codes)
    pc.classification = codes
    chunk.point_cloud = pc
    counts = {int(c): int(n) for c, n in zip(*np.unique(codes, return_counts=True))}
    kept = int((~changeable).sum())
    chunk.meta["terrio_ground"] = {"method": method, "cell": cell, "slope": slope, "threshold": threshold,
                                   "window": window, "resolution": resolution, "rigidness": rigidness,
                                   "noise": noise, "vegetation": vegetation, "kept": kept,
                                   "counts": {str(k): v for k, v in counts.items()}}
    progress(1.0, f"Ground: {counts.get(GROUND, 0):,} of {len(codes):,} points")
    return {"method": method, "counts": counts, "points": int(len(codes)), "changed": int((before != codes).sum()),
            "kept": kept, "labels": dict(chunk.meta.get("classes") or {})}


def export_cloud(chunk, path, progress=None) -> Path:
    """The dense cloud in world coordinates written by terrio: E57 (``.e57``), an EPT dataset in
    the folder ``path`` (``ept.json`` inside) when ``path`` ends with ``.ept``, or LAS, LAZ, COPC
    (``.copc.laz``) and PLY. Colours and classes go with the points."""
    from pyarchinit_photogrammetry.core import crs as crs_mod

    terrio = environment.load()
    progress = as_progress(progress)
    pc = chunk.point_cloud
    if pc is None:
        raise RuntimeError("build the dense cloud first")
    progress(0.1, f"{len(pc):,} points")
    world = chunk.to_world(pc.xyz)
    arrays = {"x": world[:, 0], "y": world[:, 1], "z": world[:, 2], "classification": pc.classification}
    if pc.rgb is not None:
        colours = pc.rgb.astype(np.uint16) * 257
        arrays.update({"red": colours[:, 0], "green": colours[:, 1], "blue": colours[:, 2]})
    cloud = terrio.PointCloud.from_dict(arrays, crs=None if crs_mod.is_local(chunk.crs) else chunk.crs)
    path = Path(path)
    progress(0.4, f"Writing {path.name}")
    if path.suffix.lower() == ".ept" or path.name == "ept.json":
        folder = path if path.suffix.lower() == ".ept" else path.parent
        folder.mkdir(parents=True, exist_ok=True)
        cloud.write(str(folder / "ept.json"))
        progress(1.0, f"EPT in {folder}")
        return folder
    path.parent.mkdir(parents=True, exist_ok=True)
    cloud.write(str(path))
    progress(1.0, str(path))
    return path
