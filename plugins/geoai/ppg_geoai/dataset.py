"""Reviewed labels and spatially independent RGB or physical raster tiles; no torch imports."""
from __future__ import annotations

import json
import math
from contextlib import ExitStack
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
    try:
        from .inputs import check_grid
    except ImportError:  # standalone worker script
        from inputs import check_grid
    return check_grid(src)


def prepare(raster, units: list[dict], field: str, mapping: dict, area_wkt: str, output,
            tile_size: int = 512, validation_fraction: float = 0.2, kinds=None, progress=None, inputs=None) -> dict:
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

    from .inputs import RasterStack

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
    stat_count, stat_mean, stat_m2 = 0, None, None
    with ExitStack() as opened:
        src = opened.enter_context(rasterio.open(raster))
        stack = opened.enter_context(RasterStack(raster, inputs)) if inputs is not None else None
        resolution = stack.resolution if stack is not None else check_raster(src)
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
                if stack is None:
                    image = src.read([1, 2, 3], window=win, boundless=True, fill_value=0)
                    valid = (src.read_masks([1, 2, 3], window=win, boundless=True) > 0).all(axis=0)
                else:
                    image, valid = stack.read(win)
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
                extra = {"valid": valid} if stack is not None else {}
                np.savez_compressed(output / name, image=image, label=label, **extra)
                counts[split] += np.bincount(label[known], minlength=len(classes))
                if stack is not None and split == "train":
                    pixels = image[:, known].astype(np.float64)
                    n = pixels.shape[1]
                    mean = pixels.mean(axis=1)
                    m2 = ((pixels - mean[:, None]) ** 2).sum(axis=1)
                    if stat_count == 0:
                        stat_mean, stat_m2 = mean, m2
                    else:
                        delta = mean - stat_mean
                        stat_m2 += m2 + delta ** 2 * stat_count * n / (stat_count + n)
                        stat_mean += delta * n / (stat_count + n)
                    stat_count += n
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
        if stack is not None:
            if not stat_count:
                raise ValueError("No valid labelled training pixels for normalisation")
            std = np.sqrt(np.maximum(stat_m2 / stat_count, 0))
            std[std < 1e-12] = 1
            manifest.update({"format": 2, "inputs": stack.schema, "channels": stack.channels,
                             "sources": [{**s, "path": str(Path(s["path"]).resolve())} for s in inputs],
                             "bands": [f"{s['name']}:{b}" for s in stack.schema for b in s["bands"]],
                             "normalisation": {"method": "standard", "mean": stat_mean.tolist(), "std": std.tolist()}})
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
    try:
        _validate_manifest(meta, folder, name == "dataset.json")
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError("Invalid GeoAI metadata") from exc
    return meta


def _validate_manifest(meta, folder, dataset):
    try:
        from .inputs import channel_count
    except ImportError:  # standalone worker script
        from inputs import channel_count

    if not isinstance(meta, dict) or type(meta.get("format")) is not int or meta["format"] not in (1, 2):
        raise ValueError("Unsupported GeoAI format or normalisation")
    channels = channel_count(meta)
    if meta["format"] == 1:
        if meta.get("normalisation") != "uint8/255" or meta.get("bands") != [1, 2, 3]:
            raise ValueError("Unsupported GeoAI bands or normalisation")
    else:
        bands = [f"{s['name']}:{b}" for s in meta["inputs"] for b in s["bands"]]
        norm = meta.get("normalisation")
        if not isinstance(norm, dict) or norm.get("method") != "standard" or meta.get("bands") != bands:
            raise ValueError("Invalid GeoAI bands or normalisation")
        for key in ("mean", "std"):
            values = np.asarray(norm.get(key), dtype=np.float64)
            if values.shape != (channels,) or not np.isfinite(values).all() or (key == "std" and (values <= 0).any()):
                raise ValueError("Invalid GeoAI channel normalisation statistics")
    classes = meta.get("classes")
    if not isinstance(classes, list) or not 2 <= len(classes) <= 255:
        raise ValueError("Invalid GeoAI class metadata")
    if (any(not isinstance(c, dict) or type(c.get("id")) is not int for c in classes)
            or [c["id"] for c in classes] != list(range(len(classes)))):
        raise ValueError("Invalid GeoAI class identifiers")
    names = [c.get("name") for c in classes]
    if any(not isinstance(n, str) or not n.strip() for n in names) or len(set(names)) != len(names):
        raise ValueError("Invalid GeoAI class names")
    if names[0] != "background":
        raise ValueError("GeoAI class zero must be background")
    if (not math.isfinite(meta["resolution"]) or meta["resolution"] <= 0
            or type(meta["tile_size"]) is not int or not 32 <= meta["tile_size"] <= 2048
            or meta["tile_size"] % 32 or meta.get("ignore_index") != IGNORE):
        raise ValueError("Invalid GeoAI resolution, tile size or ignore index")
    samples = meta.get("samples", [])
    if not isinstance(samples, list) or (dataset and not samples):
        raise ValueError("A prepared dataset needs training and validation samples")
    for sample in samples:
        if not isinstance(sample, dict) or not isinstance(sample.get("file"), str):
            raise ValueError("Invalid dataset sample path or split")
        path = (folder / sample["file"]).resolve()
        if not path.is_relative_to(folder.resolve()) or sample.get("split") not in ("train", "validation"):
            raise ValueError("Invalid dataset sample path or split")
    if dataset and {s["split"] for s in samples} != {"train", "validation"}:
        raise ValueError("Both spatial training and validation sets are required")
