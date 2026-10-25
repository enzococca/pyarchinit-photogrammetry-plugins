"""Reviewed unit labels and spatially independent RGB training tiles; no torch imports."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

IGNORE = 255


def unit_values(units: list[dict], field: str) -> list[str]:
    return sorted({str(value(u, field)) for u in units if value(u, field) not in (None, "")})


def value(unit: dict, field: str):
    return unit.get(field) if field in ("tipo", "us", "area", "sito") else (unit.get("attributes") or {}).get(field)


def class_table(mapping: dict, kinds: dict | None = None) -> list[dict]:
    """Mapping: attribute value -> class name. Background is reserved id 0."""
    if not mapping or any(not str(k).strip() or not str(v).strip() for k, v in mapping.items()):
        raise ValueError("Choose at least one non-empty attribute value and class name")
    names = sorted(set(str(v).strip() for v in mapping.values()))
    if "background" in names or len(names) > 254:
        raise ValueError("Use at most 254 classes; 'background' is reserved")
    return [{"id": 0, "name": "background", "tipo": ""}] + [
        {"id": i, "name": name, "tipo": (kinds or {}).get(name, "US")}
        for i, name in enumerate(names, 1)]


def check_raster(src) -> float:
    if src.count < 3 or any(d != "uint8" for d in src.dtypes[:3]):
        raise ValueError("GeoAI 0.1 requires an RGB uint8 orthomosaic (first three bands)")
    if src.crs is not None and src.crs.is_geographic:
        raise ValueError("Use a projected raster in metres, or a local metre grid")
    if src.crs is not None:
        try:
            if not math.isclose(src.crs.linear_units_factor[1], 1.0):
                raise ValueError("The raster CRS must use metres")
        except AttributeError:
            pass
    tr = src.transform
    if tr.b or tr.d or tr.a <= 0 or tr.e >= 0 or not math.isclose(tr.a, -tr.e, rel_tol=1e-5):
        raise ValueError("Use a north-up raster with square pixels")
    return float(tr.a)


def prepare(raster, units: list[dict], field: str, mapping: dict, area_wkt: str, output,
            tile_size: int = 512, validation_fraction: float = 0.2, kinds=None, progress=None) -> dict:
    """Write non-overlapping NPZ tiles. A whole tile column separates train and validation.

    Pixels outside the explicitly reviewed area, unmapped units, invalid source pixels and
    overlapping contradictory labels are ignored, never taught as background. A tile is assigned
    by its column, not randomly; the rightmost part is validation. Both splits need every class.
    Output must be new so a failed/repeated preparation cannot mix two datasets.
    """
    import rasterio
    from rasterio.features import rasterize
    from rasterio.windows import Window
    from shapely import wkt
    from shapely.geometry import box, shape

    if tile_size < 32 or tile_size > 2048 or tile_size % 32:
        raise ValueError("tile_size must be a multiple of 32 between 32 and 2048")
    if not 0.1 <= validation_fraction <= 0.5:
        raise ValueError("validation_fraction must be between 0.1 and 0.5")
    area = wkt.loads(area_wkt)
    if area.is_empty or not area.is_valid or area.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError("Draw a valid polygon enclosing the fully reviewed area")
    classes = class_table(mapping, kinds)
    ids = {c["name"]: c["id"] for c in classes}
    shapes = []
    for unit in units:
        raw_geometry = unit.get("geometry")
        if raw_geometry is not None:
            geom = shape(raw_geometry) if isinstance(raw_geometry, dict) else raw_geometry
        elif unit.get("wkt"):
            geom = wkt.loads(unit["wkt"])
        else:
            raise ValueError("A drawn unit has no polygon geometry")
        if not geom.is_valid or geom.is_empty or geom.geom_type not in ("Polygon", "MultiPolygon"):
            raise ValueError("Repair invalid unit geometries before creating the dataset")
        name = mapping.get(str(value(unit, field)))
        shapes.append((geom, ids[str(name).strip()] if name is not None else IGNORE))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    samples = []
    counts = {s: np.zeros(len(classes), dtype=np.int64) for s in ("train", "validation")}
    with rasterio.open(raster) as src:
        resolution = check_raster(src)
        extent = area.intersection(box(*src.bounds))
        if extent.is_empty:
            raise ValueError("Reviewed area does not intersect the orthomosaic")
        # Split across the reviewed area, not empty margins of the source raster.
        cols = sorted({c for c in range(math.ceil(src.width / tile_size))
                       if box(*rasterio.windows.bounds(Window(c * tile_size, 0, tile_size, src.height),
                                                       src.transform)).intersects(extent)})
        if len(cols) < 4:
            raise ValueError("Reviewed area needs at least four tile columns for a spatial split and buffer")
        nval = max(1, int(round(len(cols) * validation_fraction)))
        val_col = cols[-nval]
        gap_col = val_col - 1
        nrows = math.ceil(src.height / tile_size)
        for row in range(nrows):
            for col in cols:
                if col == gap_col:
                    continue
                win = Window(col * tile_size, row * tile_size, tile_size, tile_size)
                tr = src.window_transform(win)
                bounds = rasterio.windows.bounds(win, src.transform)
                if not box(*bounds).intersects(area):
                    continue
                rgb = src.read([1, 2, 3], window=win, boundless=True, fill_value=0)
                valid = (src.read_masks([1, 2, 3], window=win, boundless=True) > 0).all(axis=0)
                reviewed = rasterize([(area, 1)], out_shape=(tile_size, tile_size), transform=tr, dtype="uint8") > 0
                label = np.full((tile_size, tile_size), IGNORE, dtype=np.uint8)
                label[reviewed & valid] = 0
                occupied = np.zeros_like(label)
                conflict = np.zeros_like(label, dtype=bool)
                for geom, cid in shapes:
                    if not geom.intersects(box(*bounds)):
                        continue
                    mask = rasterize([(geom, 1)], out_shape=label.shape, transform=tr, dtype="uint8") > 0
                    conflict |= mask & (occupied != 0) & (occupied != cid)
                    occupied[mask] = cid
                    label[mask & reviewed & valid] = cid
                label[conflict] = IGNORE
                known = label != IGNORE
                if not known.any():
                    continue
                split = "validation" if col >= val_col else "train"
                name = f"{split}_{row:05d}_{col:05d}.npz"
                np.savez_compressed(output / name, image=rgb, label=label)
                counts[split] += np.bincount(label[known], minlength=len(classes))
                samples.append({"file": name, "split": split, "row": row, "col": col, "bounds": list(bounds)})
            if progress:
                progress((row + 1) / nrows, f"GeoAI dataset: {len(samples)} tiles")
        manifest = {"format": 1, "raster": str(Path(raster).resolve()), "field": field, "mapping": mapping,
                    "classes": classes, "bands": [1, 2, 3], "normalisation": "uint8/255", "resolution": resolution,
                    "crs": str(src.crs) if src.crs else None, "tile_size": tile_size, "ignore_index": IGNORE,
                    "reviewed_area": area.wkt, "validation_fraction": validation_fraction,
                    "split": {"method": "spatial-columns", "validation_first_column": val_col,
                              "excluded_buffer_column": gap_col}, "samples": samples,
                    "class_pixels": {s: c.tolist() for s, c in counts.items()}}
    for split, count in counts.items():
        missing = [classes[i]["name"] for i, n in enumerate(count) if n == 0]
        if missing:
            raise ValueError(f"{split} lacks labelled pixels for: {', '.join(missing)}. "
                             "Choose a larger reviewed area or change the class mapping/tile size. "
                             "This incomplete dataset cannot be trained; choose a new output folder.")
    (output / "dataset.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def read_manifest(folder, name="dataset.json") -> dict:
    folder = Path(folder)
    meta = json.loads((folder / name).read_text(encoding="utf-8"))
    if meta.get("format") != 1 or meta.get("normalisation") != "uint8/255":
        raise ValueError("Unsupported GeoAI format or normalisation")
    if meta.get("bands") != [1, 2, 3] or not 2 <= len(meta.get("classes", [])) <= 255:
        raise ValueError("Invalid GeoAI band or class metadata")
    if [c["id"] for c in meta["classes"]] != list(range(len(meta["classes"]))):
        raise ValueError("Invalid GeoAI class identifiers")
    for sample in meta.get("samples", []):
        path = (folder / sample["file"]).resolve()
        if not path.is_relative_to(folder.resolve()) or sample.get("split") not in ("train", "validation"):
            raise ValueError("Invalid dataset sample path or split")
    return meta
