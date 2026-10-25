"""Prediction runs as reviewable GeoPackage proposals, with independent class rasters."""
from __future__ import annotations

import datetime as dt
import uuid
from pathlib import Path

import numpy as np


def records(chunk):
    return [] if chunk is None else (chunk.meta.get("geoai") or {}).get("runs", [])


def layer(run):
    return f"geoai: {run['id']}"


def path(chunk, run, kind):
    base = Path(chunk.data_dir).resolve()
    candidate = (base / run["files"][kind]).resolve()
    if not candidate.is_relative_to(base):
        raise ValueError("Invalid GeoAI result path")
    return candidate


def rows(chunk, run):
    from pyarchinit_photogrammetry.io.gpkg import read_table

    return read_table(path(chunk, run, "gpkg"), "geoai")


def predict(chunk, raster, model, min_area=0.0, device=None, progress=None, python=None):
    import rasterio
    from rasterio.features import shapes
    from shapely.geometry import MultiPolygon, shape
    from pyarchinit_photogrammetry.io.gpkg import write_table

    from .client import run as job
    from .dataset import IGNORE

    if min_area < 0:
        raise ValueError("min_area must be non-negative")
    rid = dt.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    folder = Path(chunk.data_dir) / "geoai"
    folder.mkdir(parents=True, exist_ok=True)
    mask = folder / f"{rid}_classes.tif"
    result = job({"op": "predict", "raster": str(raster), "model": str(model), "output": str(mask),
                  "device": device}, progress=progress, python=python)
    classes = {c["id"]: c for c in result["classes"]}
    features = []
    with rasterio.open(mask) as src:
        # Polygonise the complete class grid so components crossing tile seams remain one proposal.
        # A raster band lets GDAL read blocks without a whole-orthomosaic NumPy allocation.
        for geom_json, cid in shapes(rasterio.band(src, 1), transform=src.transform):
            if int(cid) in (0, IGNORE):
                continue
            geom = shape(geom_json)
            if geom.area < min_area:
                continue
            c = classes[int(cid)]
            features.append((MultiPolygon([geom]), {"n": len(features) + 1, "class_id": int(cid),
                              "class_name": c["name"], "tipo": c["tipo"], "area_m2": geom.area}))
    gpkg = folder / f"{rid}.gpkg"
    write_table(gpkg, "geoai", "MULTIPOLYGON", {"n": "INTEGER", "class_id": "INTEGER", "class_name": "TEXT",
                "tipo": "TEXT", "area_m2": "REAL"}, features, getattr(chunk, "crs", None), f"GeoAI {rid}")
    return {"id": rid, "model": str(Path(model).resolve()), "count": len(features), "visible": True,
            "backend": result["backend"], "classes": result["classes"], "min_area": min_area,
            "files": {"gpkg": gpkg.relative_to(chunk.data_dir).as_posix(),
                      "mask": mask.relative_to(chunk.data_dir).as_posix()}}


def add(chunk, run):
    chunk.meta.setdefault("geoai", {}).setdefault("runs", []).append(run)


def remove_proposals(chunk, run, numbers):
    """Delete reviewed proposals from the vector layer; the classified raster stays as model output."""
    from pyarchinit_photogrammetry.io.gpkg import write_table

    features = [(g, a) for g, a in rows(chunk, run) if a["n"] not in numbers]
    write_table(path(chunk, run, "gpkg"), "geoai", "MULTIPOLYGON",
                {"n": "INTEGER", "class_id": "INTEGER", "class_name": "TEXT", "tipo": "TEXT", "area_m2": "REAL"},
                features, getattr(chunk, "crs", None), f"GeoAI {run['id']}")
    run["count"] = len(features)


def unit_rows(chunk, run, selected=None):
    features = rows(chunk, run)
    if selected:
        features = [(g, a) for g, a in features if a["n"] in selected]
    return ([g for g, a in features],
            [{"tipo": a["tipo"], "geoai_class": a["class_name"], "geoai_model": run["model"],
              "geoai_run": run["id"], "geoai_proposal": a["n"]} for g, a in features])


def draw(view, chunk, run):
    if view is None:
        return
    name = layer(run)
    if name in view.layers:
        view.remove_layer(name)
    if view.tr is None or not run.get("visible", True):
        return
    with view.batch():
        for geom, attrs in rows(chunk, run):
            for part in getattr(geom, "geoms", [geom]):
                view.add_polygon(np.asarray(part.exterior.coords)[:, :2], label="", color="#00c8ff",
                                 fill_alpha=50, layer=name, attrs={**attrs, "geoai_run": run["id"]})
