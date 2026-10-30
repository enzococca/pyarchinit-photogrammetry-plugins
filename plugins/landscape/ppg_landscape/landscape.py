"""The scenarios of a chunk on the program's side: the DEM read and resampled to the simulation
cell, the job handed to the worker (:mod:`.runner`), its arrays brought back onto the grid of the
DEM as GeoTIFFs, the registry of the runs, the final surface as a version of the DEM, the
comparison with the DEM of another chunk (a later phase), the exports.

Every file lives in ``<chunk data>/landscape/<run id>/`` (``final.tif``, ``dz.tif``, ``area.tif``,
``snap_<year>.tif``, ``compare_<chunk>.tif``, ``balance.json``), float32 with NaN as nodata, on the
grid, in the frame and with the CRS of the DEM the run started from. The registry is
``chunk.meta["landscape"]["runs"]``: id, name, preset, params, years, snapshots, cell, the DEM it
started from (``source``), files, balance, the shape of the simulation grid, the versions of
Landlab, created; ``dem_version`` once the final surface was added to the versions of the DEM,
``comparisons`` after :func:`compare`.

The simulation runs on a grid of its own (``cell``, by default the DEM's resolution coarsened so
that the longer side has at most 500 cells, :func:`ppg_landscape.engine.default_cell`): the DEM is
averaged onto it, and what the simulation CHANGED (dz, the snapshots minus the start) is
interpolated back onto the DEM and added to it, so the detail of the DEM the simulation could not
see stays in the final surface. Cells of the DEM out of reach of the simulation grid (its
outermost edge, where an interpolation would need cells without a value) are left unchanged: dz 0.
The drainage area is brought back by nearest neighbour (the channels stay one cell wide).

The active DEM is never replaced: :func:`add_as_dem_version` adds the final surface to the kept
versions of the DEM (``core.versions``); it becomes the active one only when the user activates it.
"""
from __future__ import annotations

import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from . import engine
from .texts import PRESET_TITLES, italian, t

FOLDER = "landscape"
STANDIN_CRS = "EPSG:3857"  # GDAL wants a CRS to warp; a local DEM keeps none in its files
RELIEF_SIDE = 512  # the relief for the estimate is read from a copy of the DEM at most this large
WORKER: tuple | None = None  # (python, script) of another worker: the tests run a fake one
PROFILE = {"driver": "GTiff", "dtype": "float32", "count": 1, "nodata": float("nan"), "compress": "deflate",
           "tiled": True, "blockxsize": 256, "blockysize": 256, "BIGTIFF": "IF_SAFER"}
LAYERS = ("final", "dz", "area")  # the rasters of every run (+ snapshots and comparisons)
_INFO: dict = {}  # path -> (mtime, info) of the DEMs read for the estimate


# ------------------------------------------------------------------ the DEM a run starts from
def folder(chunk) -> Path:
    return Path(chunk.data_dir) / FOLDER


def dem_sources(chunk) -> list[tuple[str, str]]:
    """``(key, name)`` of the DEMs a scenario can start from: ``"active"`` (the active DEM) and the
    ids of the kept versions of the DEM, with their names."""
    from pyarchinit_photogrammetry.core import versions

    out = []
    path = chunk.dem_path
    if path is not None and Path(path).exists():
        out.append(("active", versions.active_info(chunk, "dem")["name"]))
    for e in versions.versions(chunk, "dem"):
        try:
            if versions.version_file(chunk, "dem", e["id"]).exists():
                out.append((e["id"], e.get("name") or e["id"]))
        except KeyError:
            continue
    return out


def dem_path(chunk, source: str | None = None) -> Path:
    """The file of the active DEM (``source`` None or ``"active"``) or of a kept version
    (``RuntimeError`` when there is none)."""
    from pyarchinit_photogrammetry.core import versions

    if source in (None, "", "active"):
        path = chunk.dem_path
        if path is None or not Path(path).exists():
            raise RuntimeError("the chunk has no DEM: build it first")
        return Path(path)
    try:
        path = versions.version_file(chunk, "dem", source)
    except KeyError as exc:
        raise RuntimeError(f"the DEM has no version {source!r}") from exc
    if not path.exists():
        raise RuntimeError(f"the file of the version {source!r} of the DEM is missing")
    return path


def source_name(chunk, source: str | None = None) -> str:
    from pyarchinit_photogrammetry.core import versions

    if source in (None, "", "active"):
        return versions.active_info(chunk, "dem")["name"]
    return next((e.get("name") or e["id"] for e in versions.versions(chunk, "dem") if e["id"] == source), str(source))


def dem_info(path: Path) -> dict:
    """Grid, frame and relief of a DEM file (the relief from a copy at most :data:`RELIEF_SIDE`
    cells a side); remembered until the file changes."""
    import rasterio

    path = Path(path)
    mtime = path.stat().st_mtime
    known = _INFO.get(str(path))
    if known is not None and known[0] == mtime:
        return dict(known[1])
    with rasterio.open(path) as src:
        k = max(1, math.ceil(max(src.width, src.height) / RELIEF_SIDE))
        small = src.read(1, out_shape=(max(1, src.height // k), max(1, src.width // k)), masked=True)
        z = small.astype(np.float64).filled(np.nan)
        if src.nodata is not None and not math.isnan(src.nodata):
            z[z == src.nodata] = np.nan
        relief = float(np.nanmax(z) - np.nanmin(z)) if np.isfinite(z).any() else 0.0
        info = {"path": path, "transform": src.transform, "crs": src.crs, "width": src.width, "height": src.height,
                "cell": float(abs(src.transform.a)), "cell_y": float(abs(src.transform.e)), "nodata": src.nodata,
                "bounds": tuple(src.bounds), "relief": relief}
    _INFO[str(path)] = (mtime, info)
    return dict(info)


def read_dem(path: Path) -> tuple[np.ndarray, dict]:
    """The DEM as float64 with NaN where it has no value, and its :func:`dem_info`."""
    import rasterio

    info = dem_info(path)
    with rasterio.open(path) as src:
        z = src.read(1, masked=True).astype(np.float64).filled(np.nan)
    if info["nodata"] is not None and not math.isnan(info["nodata"]):
        z[z == info["nodata"]] = np.nan
    return z, info


def default_cell(info: dict) -> float:
    """The simulation cell the pane proposes: the DEM's resolution coarsened so the longer side has
    at most :data:`engine.MAX_SIDE` cells."""
    return engine.default_cell((info["height"], info["width"]), max(info["cell"], info["cell_y"]))


def simulation_grid(info: dict, cell: float):
    """``(transform, (rows, cols))`` of the simulation grid: square cells of ``cell`` from the
    top-left corner of the DEM, covering it whole."""
    from rasterio.transform import from_origin

    left, bottom, right, top = info["bounds"]
    rows = max(1, math.ceil((top - bottom) / cell - 1e-9))
    cols = max(1, math.ceil((right - left) / cell - 1e-9))
    return from_origin(left, top, cell, cell), (rows, cols)


def _crs(info: dict):
    import rasterio

    return info["crs"] if info["crs"] is not None else rasterio.crs.CRS.from_string(STANDIN_CRS)


def _warp(values: np.ndarray, src_transform, dst_transform, shape, crs, how: str) -> np.ndarray:
    from rasterio.warp import Resampling, reproject

    dst = np.full(shape, np.nan, np.float64)
    reproject(source=np.asarray(values, np.float64), destination=dst, src_transform=src_transform, src_crs=crs,
              dst_transform=dst_transform, dst_crs=crs, src_nodata=np.nan, dst_nodata=np.nan,
              resampling=getattr(Resampling, how))
    return dst


def to_simulation(z: np.ndarray, info: dict, cell: float):
    """The DEM on the simulation grid (averaged when the cell is larger than the DEM's, else
    bilinear): ``(elevation, transform)``."""
    transform, shape = simulation_grid(info, cell)
    coarser = cell > max(info["cell"], info["cell_y"]) * 1.0001
    return _warp(z, info["transform"], transform, shape, _crs(info), "average" if coarser else "bilinear"), transform


def to_dem(values: np.ndarray, sim_transform, info: dict, nearest: bool = False) -> np.ndarray:
    """An array of the simulation grid on the grid of the DEM (bilinear, or nearest neighbour)."""
    return _warp(values, sim_transform, info["transform"], (info["height"], info["width"]), _crs(info),
                 "nearest" if nearest else "bilinear")


def write_raster(path: Path, values: np.ndarray, info: dict, description: str = "") -> None:
    """``values`` as a float32 GeoTIFF with NaN as nodata on the grid of the DEM (BigTIFF when large)."""
    import rasterio

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.where(np.isfinite(values), values, np.nan).astype(np.float32)
    profile = dict(PROFILE, width=info["width"], height=info["height"], transform=info["transform"], crs=info["crs"])
    if info["width"] < 256 or info["height"] < 256:  # a tiny raster: no tiles larger than itself
        profile.update(tiled=False)
        profile.pop("blockxsize")
        profile.pop("blockysize")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
        if description:
            dst.set_band_description(1, description)


def read_raster(path: Path):
    """``(values with NaN, transform, crs)`` of a GeoTIFF of the plugin."""
    import rasterio

    with rasterio.open(path) as src:
        return src.read(1, masked=True).astype(np.float64).filled(np.nan), src.transform, src.crs


# ------------------------------------------------------------------ the parameters of a run
def job_params(params: dict) -> dict:
    """The parameters of a run with the defaults of their preset: ``preset`` (a key of
    :data:`engine.PRESETS` or ``custom``), ``years``, ``snapshots`` ([years]), ``cell`` (m; 0 or
    None: :func:`default_cell`), ``boundary`` ("open" | "closed"), ``dt`` (0 or None: automatic),
    ``diffusion`` ({"D"} or None: off), ``fluvial`` ({...} or None: off), ``router``, ``eroder``,
    ``source`` (the DEM: "active" or the id of a version)."""
    preset = str(params.get("preset") or "custom")
    base = engine.PRESETS.get(preset, {})
    out = {"preset": preset if preset in engine.PRESETS else "custom",
           "years": float(params.get("years") or base.get("years") or 100.0),
           "snapshots": sorted({float(s) for s in params.get("snapshots") or () if float(s) > 0}),
           "cell": float(params.get("cell") or 0.0),
           "boundary": str(params.get("boundary") or "open"),
           "dt": float(params["dt"]) if params.get("dt") else None,
           "router": str(params.get("router") or "auto"), "eroder": str(params.get("eroder") or "auto"),
           "source": str(params.get("source") or "active")}
    for name in ("diffusion", "fluvial"):
        if name in params:
            given = params[name]
            out[name] = None if given is None or given is False else {
                **engine.DEFAULTS[name], **base.get(name, {}), **({k: float(v) for k, v in given.items()}
                                                                    if isinstance(given, dict) else {})}
        else:
            out[name] = dict(base[name]) if name in base else dict(engine.DEFAULTS[name])
    if out["boundary"] not in engine.BOUNDARIES:
        raise ValueError(f"boundary: {', '.join(engine.BOUNDARIES)}, not {out['boundary']!r}")
    if not out["years"] > 0:
        raise ValueError("years must be positive")
    if out["diffusion"] is None and out["fluvial"] is None:
        raise ValueError("no process: soil creep and running water are both off")
    out["snapshots"] = [s for s in out["snapshots"] if s < out["years"]]
    return out


def estimate(chunk, params: dict, lang: str = "it") -> dict:
    """What a run of ``params`` will cost (:func:`engine.estimate`) on the DEM it starts from,
    with the simulation ``cell`` and ``shape`` used."""
    p = job_params(params)
    info = dem_info(dem_path(chunk, p["source"]))
    cell = p["cell"] or default_cell(info)
    _transform, shape = simulation_grid(info, cell)
    out = engine.estimate(shape, cell, p["years"], {"diffusion": p["diffusion"], "fluvial": p["fluvial"],
                                                    "dt": p["dt"], "relief": info["relief"]}, lang=lang)
    side = max(shape)
    out.update(cell=cell, shape=list(shape), large=side > engine.WARN_SIDE)
    return out


# ------------------------------------------------------------------ the registry
def runs(chunk) -> list[dict]:
    return list((chunk.meta.get(FOLDER) or {}).get("runs", []))


def entry(chunk, run_id: str) -> dict | None:
    return next((e for e in runs(chunk) if e.get("id") == run_id), None)


def _save_entry(chunk, e: dict) -> None:
    meta = chunk.meta.setdefault(FOLDER, {})
    rest = [x for x in meta.get("runs", []) if x.get("id") != e["id"]]
    rest.append(e)
    meta["runs"] = rest


def _new_id(chunk) -> str:
    base = datetime.now().strftime("r%Y%m%d-%H%M%S")
    taken = {e.get("id") for e in runs(chunk)}
    rid, k = base, 1
    while rid in taken or (folder(chunk) / rid).exists():
        k += 1
        rid = f"{base}-{k}"
    return rid


def default_name(params: dict) -> str:
    """The name of a run as stored (Italian, as the program's data): ``Tell in mattoni crudi, 1.000 anni``."""
    title = PRESET_TITLES.get(params.get("preset") or "custom", PRESET_TITLES["custom"])[0]
    years = float(params["years"])
    return f"{title}, {italian(years, 0 if years == int(years) else 2)} anni"


def path_of(chunk, e: dict, layer: str) -> Path | None:
    """The file of a layer of a run: ``final``, ``dz``, ``area``, ``snap:<year>``, ``compare:<chunk key>``."""
    files = e.get("files") or {}
    if layer.startswith("snap:"):
        rel = (files.get("snapshots") or {}).get(f"{float(layer[5:]):g}")
    elif layer.startswith("compare:"):
        rel = next((c.get("file") for c in e.get("comparisons") or [] if str(c.get("chunk_key")) == layer[8:]), None)
    else:
        rel = files.get(layer)
    return None if not rel else Path(chunk.data_dir) / rel


def layers(e: dict) -> list[str]:
    """The layers a run has: final, dz, area, then the snapshots and the comparisons."""
    out = [k for k in LAYERS if (e.get("files") or {}).get(k)]
    out += [f"snap:{y}" for y in sorted((e.get("files") or {}).get("snapshots") or {}, key=float)]
    out += [f"compare:{c['chunk_key']}" for c in e.get("comparisons") or []]
    return out


def kind_of(layer: str) -> str:
    """How a layer is coloured: ``elevation``, ``diverging`` (dz, differences), ``log`` (area)."""
    if layer in ("dz",) or layer.startswith("compare:"):
        return "diverging"
    if layer == "area":
        return "log"
    return "elevation"


def load(chunk, run_id: str, layer: str = "dz"):
    """``(values, transform, crs)`` of a layer of a run."""
    e = entry(chunk, run_id)
    if e is None:
        raise KeyError(run_id)
    path = path_of(chunk, e, layer)
    if path is None or not path.exists():
        raise KeyError(f"{run_id}: no layer {layer!r}")
    return read_raster(path)


# ------------------------------------------------------------------ a run
def run(chunk, params: dict, progress=None, name: str | None = None, python=None, script=None,
        force: bool = False) -> dict:
    """Simulate ``params`` (:func:`job_params`) on the DEM of ``chunk`` with the worker in a process
    of its own and register the run; its entry. ``force``: run also when :func:`estimate` refuses
    (more than two hours). ``python``/``script``: another worker (:data:`WORKER` by default)."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    from . import runner

    progress = as_progress(progress)
    p = job_params(params)
    src = dem_path(chunk, p["source"])
    cost = estimate(chunk, p)
    if cost["refuse"] and not force:
        raise ValueError(t("La simulazione durerebbe più di due ore: scegli una cella più grande o meno anni.",
                           "The simulation would take more than two hours: choose a larger cell or fewer years."))
    if python is None and script is None and WORKER is not None:
        python, script = WORKER
    progress(0.0, t("Lettura del DEM", "Reading the DEM"))
    z, info = read_dem(src)
    cell = p["cell"] or default_cell(info)
    elevation, sim_transform = to_simulation(z, info, cell)
    if np.isfinite(elevation).sum() < 9:
        raise ValueError("the DEM has fewer than 9 cells with a value on the simulation grid: choose a smaller cell")
    rid = _new_id(chunk)
    root = folder(chunk)
    work = root / "_work" / rid
    out_dir = root / rid
    try:
        work.mkdir(parents=True, exist_ok=True)
        np.save(work / "elevation.npy", elevation, allow_pickle=False)
        job = {"protocol": 1, "elevation": str((work / "elevation.npy").absolute()), "mask": None,
               "output": str((work / "out").absolute()), "cell": cell, "years": p["years"], "dt": p["dt"],
               "snapshots": p["snapshots"], "boundary": p["boundary"], "diffusion": p["diffusion"],
               "fluvial": p["fluvial"], "router": p["router"], "eroder": p["eroder"]}
        result = runner.run(job, work, progress.sub(0.05, 0.85), python=python, script=script)
        progress.check()
        progress(0.9, t("Risultati sulla griglia del DEM", "Results onto the grid of the DEM"))
        res = Path(result["output"])
        initial = np.load(res / "initial.npy")
        files = {}
        valid = np.isfinite(z)

        def change_on_dem(sim_change: np.ndarray) -> np.ndarray:
            back = to_dem(sim_change, sim_transform, info)
            return np.where(valid, np.where(np.isfinite(back), back, 0.0), np.nan)

        dz = change_on_dem(np.load(res / "dz.npy"))
        out_dir.mkdir(parents=True, exist_ok=False)
        final = np.where(valid, z + dz, np.nan)
        # dz written as the difference of the two float32 surfaces: final - initial = dz in the files too
        z32 = z.astype(np.float32)
        final32 = final.astype(np.float32)
        write_raster(out_dir / "final.tif", final32, info, "superficie finale (m)")
        write_raster(out_dir / "dz.tif", final32.astype(np.float64) - z32.astype(np.float64), info,
                     "erosione (-) e deposito (+), m")
        files["final"] = f"{FOLDER}/{rid}/final.tif"
        files["dz"] = f"{FOLDER}/{rid}/dz.tif"
        if (res / "area.npy").exists():
            area = to_dem(np.load(res / "area.npy"), sim_transform, info, nearest=True)
            write_raster(out_dir / "area.tif", np.where(valid, area, np.nan), info, "area drenata (m2)")
            files["area"] = f"{FOLDER}/{rid}/area.tif"
        snaps = {}
        for year, fname in sorted((result.get("files") or {}).get("snapshots", {}).items(), key=lambda kv: float(kv[0])):
            change = change_on_dem(np.load(res / fname) - initial)
            name_tif = f"snap_{float(year):g}.tif"
            write_raster(out_dir / name_tif, np.where(valid, z + change, np.nan), info,
                         f"superficie dopo {float(year):g} anni (m)")
            snaps[f"{float(year):g}"] = f"{FOLDER}/{rid}/{name_tif}"
        files["snapshots"] = snaps
        balance = dict(result.get("balance") or {})
        (out_dir / "balance.json").write_text(json.dumps(balance, indent=1), encoding="utf-8")
        files["balance"] = f"{FOLDER}/{rid}/balance.json"
        used = {k: p[k] for k in ("preset", "boundary", "dt", "diffusion", "fluvial", "router", "eroder")}
        e = {"id": rid, "name": name or default_name(p), "preset": p["preset"], "params": used,
             "years": p["years"], "snapshots": p["snapshots"], "cell": cell,
             "source": {"version": p["source"], "name": source_name(chunk, p["source"]),
                        "file": str(Path(src).relative_to(chunk.data_dir)) if _inside(src, chunk.data_dir) else str(src)},
             "files": files, "balance": balance, "grid": list(elevation.shape),
             "versions": dict(result.get("versions") or {}), "seconds": result.get("seconds"),
             "created": datetime.now().isoformat(timespec="seconds"), "stats": _stats(dz[np.isfinite(dz)])}
        _save_entry(chunk, e)
        progress(1.0, t("Fatto", "Done"))
        return e
    except BaseException:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(work, ignore_errors=True)
        try:
            (root / "_work").rmdir()
        except OSError:
            pass


def _inside(path: Path, base: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(base).resolve())
        return True
    except ValueError:
        return False


def _stats(v: np.ndarray) -> dict:
    """Median, NMAD (1.4826 x the median absolute deviation from the median), 95th percentile of the
    absolute values, mean, the 2nd and 98th percentiles, the cells."""
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"cells": 0}
    med = float(np.median(v))
    return {"cells": int(v.size), "median": med, "nmad": float(1.4826 * np.median(np.abs(v - med))),
            "p95": float(np.percentile(np.abs(v), 95)), "mean": float(v.mean()),
            "p2": float(np.percentile(v, 2)), "p98": float(np.percentile(v, 98)),
            "min": float(v.min()), "max": float(v.max())}


# ------------------------------------------------------------------ the run afterwards
def add_as_dem_version(chunk, run_id: str, name: str | None = None) -> dict:
    """The final surface of a run among the kept versions of the DEM (named ``Paesaggio dopo N anni
    (scenario)``); the active DEM does not change. Returns the version's entry (the same one when the
    run was added already and that version still exists)."""
    from pyarchinit_photogrammetry.core import versions

    e = entry(chunk, run_id)
    if e is None:
        raise KeyError(run_id)
    vid = e.get("dem_version")
    if vid:
        known = next((v for v in versions.versions(chunk, "dem") if v["id"] == vid), None)
        if known is not None:
            return known
    src = path_of(chunk, e, "final")
    if src is None or not src.exists():
        raise RuntimeError(f"{run_id}: the final surface is missing")
    vid = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    dst_dir = Path(chunk.data_dir) / "versions" / "dem" / vid
    dst_dir.mkdir(parents=True)
    dst = dst_dir / versions.FILES["dem"]
    shutil.copy2(src, dst)
    years = float(e["years"])
    label = name or f"Paesaggio dopo {italian(years, 0 if years == int(years) else 2)} anni ({e['name']})"
    version = {"id": vid, "name": label[:120], "created": datetime.now().isoformat(timespec="seconds"),
               "params": {"source": "landscape", "run": run_id, "years": years, "cell": e.get("cell"),
                          "preset": e.get("preset"), "from": (e.get("source") or {}).get("name")},
               "files": {"dem": str(dst.relative_to(chunk.data_dir))}}
    chunk.meta.setdefault("versions", {}).setdefault("dem", []).append(version)
    e["dem_version"] = vid
    _save_entry(chunk, e)
    return version


def compare(chunk, run_id: str, other, source: str | None = None) -> dict:
    """The final surface of a run minus the DEM of ``other`` (another chunk: a later phase), that DEM
    resampled onto the grid of the run (bilinear; reprojected when both have a CRS and they differ),
    written as ``compare_<chunk key>.tif``; the statistics of the difference and, as a baseline, those
    of the DEM the run started from minus the same DEM: the parameters get closer to the later surface
    when the difference shrinks. Stored in the run's ``comparisons`` and returned."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    e = entry(chunk, run_id)
    if e is None:
        raise KeyError(run_id)
    final, transform, crs = read_raster(path_of(chunk, e, "final"))
    other_path = dem_path(other, source)
    with rasterio.open(other_path) as src:
        later = src.read(1, masked=True).astype(np.float64).filled(np.nan)
        if src.nodata is not None and not math.isnan(src.nodata):
            later[later == src.nodata] = np.nan
        other_crs, other_transform = src.crs, src.transform
    if crs is not None and other_crs is not None:
        src_crs, dst_crs = other_crs, crs
    else:  # a local frame on one side at least: the same frame on both (as the program's chunks share it)
        src_crs = dst_crs = rasterio.crs.CRS.from_string(STANDIN_CRS)
    onto = np.full(final.shape, np.nan, np.float64)
    reproject(source=later, destination=onto, src_transform=other_transform, src_crs=src_crs,
              dst_transform=transform, dst_crs=dst_crs, src_nodata=np.nan, dst_nodata=np.nan,
              resampling=Resampling.bilinear)
    diff = final - onto
    dz = read_raster(path_of(chunk, e, "dz"))[0]
    start = final - dz  # the DEM the run started from, as it was then
    info = {"width": final.shape[1], "height": final.shape[0], "transform": transform, "crs": crs}
    key = str(getattr(other, "key", "x")) + ("" if source in (None, "", "active") else f"-{source}")
    out = folder(chunk) / run_id / f"compare_{key}.tif"
    write_raster(out, diff, info, f"superficie finale - DEM di {getattr(other, 'label', key)} (m)")
    ok = np.isfinite(diff)
    result = {"chunk_key": key, "chunk": getattr(other, "label", key), "source": source or "active",
              "source_name": source_name(other, source), "file": f"{FOLDER}/{run_id}/{out.name}",
              "stats": _stats(diff[ok]), "baseline": _stats((start - onto)[ok & np.isfinite(start)]),
              "created": datetime.now().isoformat(timespec="seconds")}
    rest = [c for c in e.get("comparisons") or [] if str(c.get("chunk_key")) != key]
    e["comparisons"] = rest + [result]
    _save_entry(chunk, e)
    return result


def remove(chunk, run_id: str) -> bool:
    """The run out of the chunk: its folder ``landscape/<run id>`` deleted (nothing else: a version of
    the DEM made from it stays, it belongs to the DEM now). False when there is no such run."""
    e = entry(chunk, run_id)
    meta = chunk.meta.setdefault(FOLDER, {})
    meta["runs"] = [x for x in meta.get("runs", []) if x.get("id") != run_id]
    target = folder(chunk) / run_id
    if run_id and "/" not in run_id and "\\" not in run_id and run_id not in (".", "..") and \
            _inside(target, folder(chunk)) and target.is_dir():
        shutil.rmtree(target)
    return e is not None


def export(chunk, run_id: str, path, layer: str = "all") -> Path:
    """A layer of a run as a GeoTIFF (a copy of its file), or ``all``: one GeoTIFF with a band per
    layer (final, dz, area, the snapshots), each band named."""
    import rasterio

    e = entry(chunk, run_id)
    if e is None:
        raise KeyError(run_id)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if layer != "all":
        src = path_of(chunk, e, layer)
        if src is None or not src.exists():
            raise KeyError(f"{run_id}: no layer {layer!r}")
        shutil.copy2(src, path)
        return path
    names = [k for k in layers(e) if not k.startswith("compare:")]
    first, transform, crs = read_raster(path_of(chunk, e, names[0]))
    profile = dict(PROFILE, count=len(names), width=first.shape[1], height=first.shape[0], transform=transform, crs=crs)
    if first.shape[0] < 256 or first.shape[1] < 256:
        profile.update(tiled=False)
        profile.pop("blockxsize")
        profile.pop("blockysize")
    with rasterio.open(path, "w", **profile) as dst:
        for i, k in enumerate(names, start=1):
            values = first if i == 1 else read_raster(path_of(chunk, e, k))[0]
            dst.write(values.astype(np.float32), i)
            dst.set_band_description(i, k.replace("snap:", "anni_"))
    return path


def latest(chunk) -> dict | None:
    rs = runs(chunk)
    return max(rs, key=lambda e: e.get("created", "")) if rs else None


def summary(chunk) -> str:
    """The runs of the chunk as JSON (``ppg plugin landscape list``)."""
    return json.dumps([{"id": e["id"], "name": e.get("name"), "years": e.get("years"), "cell": e.get("cell"),
                        "preset": e.get("preset"), "source": (e.get("source") or {}).get("name"),
                        "eroded_m3": (e.get("balance") or {}).get("eroded_m3"),
                        "deposited_m3": (e.get("balance") or {}).get("deposited_m3"),
                        "dem_version": e.get("dem_version"), "created": e.get("created")} for e in runs(chunk)],
                      indent=1, ensure_ascii=False)


__all__ = ["run", "runs", "entry", "estimate", "add_as_dem_version", "compare", "remove", "export", "load",
           "dem_sources", "job_params"]
