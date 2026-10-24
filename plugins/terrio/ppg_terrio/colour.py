"""Points coloured from a raster, or given a value of it, with terrio's ``colorization`` (PDAL's
``filters.colorization``).

* **Colours** (``mode="rgb"``): the RGB of the orthomosaic of the chunk - or of another raster, a
  file or a COG on the web - for every point it covers: a lidar imported without colours, a
  cloud of another program, the colours of a newer orthophoto. The points outside the raster (or
  on its transparent or nodata pixels) keep the colours they had (grey when they had none). The
  cloud of before is kept as a version. Bands of 16 bits or of floats (reflectance) are stretched
  between their 2nd and 98th percentiles.
* **A value** (``mode="value"``): one band - the temperature of the thermal orthomosaic, an index
  (NDVI...), the DEM, a band of the reflectance - as a dimension of the points, a ``float``
  property of the dense cloud (see :mod:`.cloudfile`) named after it (``temperature``, ``ndvi``,
  ``height``), NaN where the raster has no value. The plugin's exports (LAS, LAZ, COPC, EPT, PLY)
  carry it as an extra dimension; :func:`preview_values` colours the points by it for the 3D
  view, the cloud's colours untouched. The program does not know the dimension: when it writes
  the cloud again (its cleaning, classes, cropping) the values go and must be taken again.

The raster is read tile by tile over the points (:data:`TILE` pixels a side) - a large
orthomosaic is never read whole - cut on its own grid when it is in the chunk's system, warped
into it otherwise (bilinear).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from pyarchinit_photogrammetry.core.progress import as_progress

from . import cloudfile, environment
from .texts import num, t

TILE = 4096  # pixels a side of the windows of the raster read at a time
PALETTES = ("turbo", "viridis", "inferno", "grey", "rdylgn", "coolwarm", "terrain")
_MPL = {"grey": "gray", "rdylgn": "RdYlGn"}
GREY = 128  # the colour of points that had none and the raster does not cover
OUTSIDE = float(np.float32(-1e30))  # the nodata of a window warped into the chunk system


def _feature() -> str:
    return t("I colori e i valori da un raster", "Colours and values from a raster")


def chunk_rasters(chunk) -> list[dict]:
    """The rasters of the chunk a cloud can take colours or values from: ``{"key", "label",
    "path", "mode", "band", "name"}`` - the orthomosaic and its versions (colours), the DEM and its
    versions, the temperature, the indices and the bands of the reflectance (values)."""
    from pyarchinit_photogrammetry.core import versions

    out = []

    def add(key, label, path, mode, band=1, name=None):
        if path is not None and Path(path).exists():
            out.append({"key": key, "label": label, "path": Path(path), "mode": mode, "band": band, "name": name})

    add("orthomosaic", t("Ortofoto (attiva)", "Orthomosaic (active)"), chunk.orthomosaic_path, "rgb")
    for e in versions.versions(chunk, "orthomosaic"):
        add(f"version:orthomosaic:{e['id']}", t("Ortofoto: {0}", "Orthomosaic: {0}").format(e["name"]),
            versions.version_file(chunk, "orthomosaic", e["id"]), "rgb")
    add("dem", t("DEM (attivo): quota", "DEM (active): height"), chunk.dem_path, "value", 1, "height")
    for e in versions.versions(chunk, "dem"):
        add(f"version:dem:{e['id']}", t("DEM: {0}", "DEM: {0}").format(e["name"]),
            versions.version_file(chunk, "dem", e["id"]), "value", 1, "height")
    thermal = (chunk.meta.get("thermal") or {}).get("file")
    if thermal:
        add("thermal", t("Temperatura (ortofoto termica, °C)", "Temperature (thermal orthomosaic, °C)"),
            chunk.data_dir / thermal, "value", 1, "temperature")
    ms = chunk.meta.get("multispectral") or {}
    for name, info in sorted((ms.get("indices") or {}).items()):
        add(f"index:{name}", t("Indice {0}", "Index {0}").format(name), chunk.data_dir / info["file"], "value", 1,
            cloudfile.dimension_name(name))
    if ms.get("reflectance"):
        path = chunk.data_dir / ms["reflectance"]
        try:
            import rasterio

            with rasterio.open(path) as s:
                names = [s.descriptions[i] or f"b{i + 1}" for i in range(s.count)]
        except Exception:  # not readable: left out
            names = []
        for i, band in enumerate(names):
            add(f"reflectance:{i + 1}", t("Riflettanza {0}", "Reflectance {0}").format(band), path, "value", i + 1,
                cloudfile.dimension_name(band))
    return out


def resolve(chunk, source) -> dict:
    """A key of :func:`chunk_rasters`, or a file or URL (``mode`` unknown then)."""
    text = str(source)
    for r in chunk_rasters(chunk):
        if r["key"] == text:
            return r
    if "://" not in text and not text.upper().startswith(("STAC", "/VSI")) and not Path(text).exists():
        raise FileNotFoundError(f"no raster {text}")
    return {"key": text, "label": Path(text).name or text, "path": text, "mode": None, "band": 1,
            "name": cloudfile.dimension_name(Path(text).stem.split(".")[0] or "value")}


def _same_crs(a, b) -> bool:
    from .clouds import _same_crs as same

    return same(str(a), str(b))


def _tiles(xy: np.ndarray, size: float, origin: np.ndarray):
    """Indices of the points of each square tile of ``size`` from ``origin``."""
    key = np.floor((xy - origin) / size).astype(np.int64)
    order = np.lexsort((key[:, 1], key[:, 0]))
    k = key[order]
    cut = np.nonzero(np.any(np.diff(k, axis=0) != 0, axis=1))[0] + 1
    for part in np.split(order, cut):
        if len(part):
            yield key[part[0]], part


def sample(chunk, path, bands, world: np.ndarray, resampling: str = "bilinear", progress=None) -> np.ndarray:
    """The values of ``bands`` of the raster ``path`` at the points ``world`` (chunk system):
    ``(N, len(bands))`` floats, NaN where the raster has no value (outside, nodata, alpha 0)."""
    from pyarchinit_photogrammetry.core import crs as crs_mod

    terrio = environment.load_features(_feature())
    progress = as_progress(progress)
    src = terrio.open_raster(str(path), lazy=True)
    count = int(src.count)
    bands = [int(b) for b in bands]
    if any(b < 1 or b > count for b in bands):
        raise ValueError(f"the raster has {count} bands")
    alpha = count == 4 and bands == [1, 2, 3] and src.dtype == "uint8"
    wanted = bands + ([4] if alpha else [])
    local = crs_mod.is_local(chunk.crs)
    same = local or src.crs is None or _same_crs(src.crs, chunk.crs)
    if local and src.crs is not None and Path(str(path)).exists() and not _inside(Path(str(path)), chunk.data_dir):
        raise RuntimeError("the chunk has no reference system: only its own rasters can colour it")
    if same:
        res = float(min(src.res))
    else:  # the size of a pixel of the source in the chunk system: its extent over its width
        b = src.bounds
        xs, ys = terrio.transform_points([b[0], b[2], b[0], b[2]], [b[1], b[1], b[3], b[3]], src.crs, chunk.crs)[:2]
        res = float(min((max(xs) - min(xs)) / src.width, (max(ys) - min(ys)) / src.height))
    out = np.full((len(world), len(bands)), np.nan)
    size = TILE * res
    origin = world[:, :2].min(0)
    tiles = list(_tiles(world[:, :2], size, origin))
    names = [f"tv{i}" for i in range(len(wanted))]
    for k, (key, idx) in enumerate(tiles):
        x0, y0 = origin + key * size
        pad = 2 * res
        x0, y0, x1, y1 = x0 - pad, y0 - pad, x0 + size + pad, y0 + size + pad
        try:
            if same:
                sb = src.bounds
                if x1 <= sb[0] or x0 >= sb[2] or y1 <= sb[1] or y0 >= sb[3]:
                    continue
                window = terrio.translate(str(path), projwin=(max(x0, sb[0]), min(y1, sb[3]), min(x1, sb[2]),
                                                              max(y0, sb[1])), bands=wanted)
            else:
                # floats with a nodata of their own: the cells the source does not reach are told apart
                window = terrio.warp(str(path), t_srs=chunk.crs, te=(x0, y0, x1, y1), tr=(res, res),
                                     resampling=resampling, ot="Float32", dstnodata=OUTSIDE)
                if window.count != len(wanted):
                    window = terrio.translate(window, bands=wanted)
        except Exception as exc:  # a window outside the raster: no value there
            if "outside" in str(exc).lower() or "no pixel" in str(exc).lower():
                continue
            raise
        pts = world[idx]
        arrays = {"x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2]} | {n: np.full(len(idx), np.nan) for n in names}
        cloud = terrio.PointCloud.from_dict(arrays)
        dims = ",".join(f"{n}:{i + 1}:1.0" for i, n in enumerate(names))
        got = cloud.colorization(window, dimensions=dims).to_dict()
        vals = np.column_stack([np.asarray(got[n], np.float64) for n in names])
        wnodata = window.nodata
        if wnodata is not None and np.isfinite(wnodata):
            vals[vals == wnodata] = np.nan
        if alpha:
            vals[vals[:, 3] == 0] = np.nan
            vals = vals[:, :3]
        out[idx] = vals
        progress((k + 1) / len(tiles), t("Tassello {0} di {1}", "Tile {0} of {1}").format(k + 1, len(tiles)))
    return out


def _inside(path: Path, folder: Path) -> bool:
    try:
        path.resolve().relative_to(folder.resolve())
        return True
    except ValueError:
        return False


def _to_bytes(values: np.ndarray, dtype: str) -> np.ndarray:
    """Colours of 8 bits from the values of the bands: as they are for bytes, else stretched
    between the 2nd and 98th percentiles of each band."""
    out = np.zeros(values.shape, np.uint8)
    for b in range(values.shape[1]):
        v = values[:, b]
        good = np.isfinite(v)
        if not good.any():
            continue
        if dtype == "uint8":
            out[good, b] = np.clip(np.round(v[good]), 0, 255)
        else:
            lo, hi = np.percentile(v[good], [2, 98])
            out[good, b] = np.clip(np.round((v[good] - lo) / max(hi - lo, 1e-12) * 255), 0, 255)
    return out


def colorize(chunk, source="orthomosaic", mode: str | None = None, band: int = 1, name: str | None = None,
             keep_version: bool = True, progress=None) -> dict:
    """Colours (``mode="rgb"``) or a value (``"value"``, of ``band``, as the dimension ``name``)
    of the dense cloud from ``source``: a key of :func:`chunk_rasters` (``orthomosaic``, ``dem``,
    ``thermal``, ``index:NDVI``...), a file or a URL. ``mode`` defaults to the raster's (colours
    for an orthomosaic, a value for the others; colours for a file of 3 or more bands). Returns
    the counts."""
    from pyarchinit_photogrammetry.core import versions

    terrio = environment.load_features(_feature())
    progress = as_progress(progress)
    raster = resolve(chunk, source)
    pc = chunk.point_cloud
    if pc is None:
        raise RuntimeError("build the dense cloud first")
    head = terrio.open_raster(str(raster["path"]), lazy=True)
    if mode is None:
        mode = raster["mode"] or ("rgb" if head.count >= 3 else "value")
    if mode not in ("rgb", "value"):
        raise ValueError("mode: rgb or value")
    path = chunk.product_path("point_cloud")
    extras = cloudfile.read_extras(path)
    world = chunk.to_world(pc.xyz)
    progress(0.02, t("{0} punti da {1}", "{0} points from {1}").format(num(len(pc)), raster["label"]))
    if mode == "rgb":
        if head.count < 3:
            raise ValueError(f"{raster['label']}: {head.count} band(s), colours need 3")
        values = sample(chunk, raster["path"], (1, 2, 3), world, progress=progress.sub(0.05, 0.8))
        covered = np.isfinite(values).all(axis=1)
        rgb = pc.rgb.copy() if pc.rgb is not None else np.full((len(pc), 3), GREY, np.uint8)
        rgb[covered] = _to_bytes(values[covered], head.dtype)
        kept = None
        if keep_version and covered.any():
            kept = versions.keep_active(chunk, "point_cloud", move=True)
        pc.rgb = rgb
        cloudfile.write(chunk, pc, extras)
        if kept is not None:
            versions.set_active_name(chunk, "point_cloud", t("Colori da {0} del {1}", "Colours from {0}, {1}")
                                     .format(raster["label"], datetime.now().strftime("%d/%m %H:%M"))[:100])
        chunk.meta["terrio_colours"] = {"source": str(raster["key"]), "points": int(covered.sum()),
                                        "created": datetime.now().isoformat(timespec="seconds")}
        out = {"mode": "rgb", "points": len(pc), "with_value": int(covered.sum()), "source": raster["label"],
               "version": None if kept is None else kept["id"], "version_name": None if kept is None else kept["name"]}
    else:
        band = int(band or raster.get("band") or 1)
        dim = cloudfile.dimension_name(name or raster.get("name") or "value")
        values = sample(chunk, raster["path"], (band,), world, progress=progress.sub(0.05, 0.8))[:, 0]
        good = np.isfinite(values)
        extras[dim] = values.astype(np.float32)
        progress(0.9, t("Scrittura della nuvola", "Writing the cloud"))
        cloudfile.add_properties(path, {dim: extras[dim]})
        stats = (float(np.nanmin(values)), float(np.nanmedian(values)), float(np.nanmax(values))) if good.any() else None
        chunk.meta.setdefault("terrio_values", {})[dim] = {
            "source": str(raster["key"]), "label": raster["label"], "band": band, "points": int(good.sum()),
            "range": list(stats) if stats else None, "created": datetime.now().isoformat(timespec="seconds")}
        out = {"mode": "value", "name": dim, "points": len(pc), "with_value": int(good.sum()), "source": raster["label"],
               "range": stats, "version": None, "version_name": None}
    progress(1.0, summary(out))
    return out


def values(chunk) -> dict[str, dict]:
    """The dimensions of the plugin the dense cloud has now, with what the project records of
    them; the records of dimensions the cloud lost (the program wrote it again) are dropped."""
    names = cloudfile.extra_names(chunk.product_path("point_cloud"))
    known = chunk.meta.get("terrio_values") or {}
    for gone in [k for k in known if k not in names]:
        known.pop(gone)
    return {n: dict(known.get(n) or {}) for n in names}


def remove_value(chunk, name: str) -> None:
    """Take the dimension ``name`` away from the dense cloud."""
    path = chunk.product_path("point_cloud")
    if name not in cloudfile.extra_names(path):
        raise KeyError(f"the dense cloud has no dimension {name!r}")
    extras = cloudfile.read_extras(path)
    extras.pop(name)
    pc = chunk.point_cloud
    cloudfile.write(chunk, pc, extras)
    (chunk.meta.get("terrio_values") or {}).pop(name, None)


def palette_colours(v: np.ndarray, palette: str = "turbo", limits=None) -> tuple[np.ndarray, tuple[float, float]]:
    """Colours of 8 bits for the values ``v`` (NaN: grey) along ``palette`` between ``limits``
    (by default the 2nd and 98th percentiles); returns the colours and the limits."""
    v = np.asarray(v, np.float64)
    good = np.isfinite(v)
    rgb = np.full((len(v), 3), 150, np.uint8)
    if not good.any():
        return rgb, (0.0, 0.0)
    lo, hi = (float(x) for x in (limits or np.percentile(v[good], [2, 98])))
    f = np.clip((v[good] - lo) / max(hi - lo, 1e-12), 0, 1)
    try:
        import matplotlib

        cmap = matplotlib.colormaps[_MPL.get(palette, palette)]
        rgb[good] = (np.asarray(cmap(f))[:, :3] * 255).round().astype(np.uint8)
    except Exception:  # no matplotlib: a grey ramp
        rgb[good] = np.repeat((f * 255).round().astype(np.uint8)[:, None], 3, axis=1)
    return rgb, (lo, hi)


def preview_values(chunk, name: str, path, palette: str = "turbo", limits=None) -> dict:
    """The dense cloud coloured by its dimension ``name`` along ``palette`` (grey where it has no
    value), written to ``path`` for the 3D view; the cloud itself is untouched."""
    from pyarchinit_photogrammetry.core.pointcloud import PointCloud

    v = cloudfile.read_extras(chunk.product_path("point_cloud"), [name]).get(name)
    if v is None:
        raise KeyError(f"the dense cloud has no dimension {name!r}")
    pc = chunk.point_cloud
    rgb, lims = palette_colours(v, palette, limits)
    cloudfile.save(PointCloud(pc.xyz, rgb, classification=pc.classification), path)
    return {"path": str(path), "limits": lims, "palette": palette, "name": name}


def summary(out: dict) -> str:
    if out["mode"] == "rgb":
        lines = [t("Colori da {0}: {1} punti su {2}", "Colours from {0}: {1} points of {2}")
                 .format(out["source"], num(out["with_value"]), num(out["points"]))]
        if out["with_value"] < out["points"]:
            lines.append(t("  {0} punti fuori dal raster hanno tenuto i loro colori",
                           "  {0} points outside the raster kept their colours").format(num(out["points"] - out["with_value"])))
        if out.get("version_name"):
            lines.append(t("La nuvola di prima e' fra le versioni: \"{0}\"",
                           "The cloud of before is among the versions: \"{0}\"").format(out["version_name"]))
        return "\n".join(lines)
    lines = [t("Valore \"{0}\" da {1}: {2} punti su {3}", "Value \"{0}\" from {1}: {2} points of {3}")
             .format(out["name"], out["source"], num(out["with_value"]), num(out["points"]))]
    if out.get("range"):
        lo, mid, hi = out["range"]
        lines.append(t("  da {0:.4g} a {1:.4g}, mediana {2:.4g}", "  from {0:.4g} to {1:.4g}, median {2:.4g}").format(lo, hi, mid))
    return "\n".join(lines)
