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

The DEM can be far larger than the simulation grid (10^8 cells on a large tell), so it is never held
whole in the program's process: it is averaged onto the simulation grid by GDAL reading it in
windows, and the change comes back block by block of rows (:data:`BLOCK_CELLS` cells, float32, the
coarse change interpolated per block from the rows of the simulation grid that cover it plus a
margin), each block written into ``final.tif``, ``dz.tif``, ``area.tif`` and the snapshots at once;
the statistics of dz (and of a comparison) come from a regular sample of at most
:data:`STATS_CELLS` cells (all of them on a smaller DEM).

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
from .texts import lang as ui_lang

FOLDER = "landscape"
STANDIN_CRS = "EPSG:3857"  # GDAL wants a CRS to warp; a local DEM keeps none in its files
RELIEF_SIDE = 512  # the relief for the estimate is read from a copy of the DEM at most this large
WORKER: tuple | None = None  # (python, script) of another worker: the tests run a fake one
PROFILE = {"driver": "GTiff", "dtype": "float32", "count": 1, "nodata": float("nan"), "compress": "deflate",
           "tiled": True, "blockxsize": 256, "blockysize": 256, "BIGTIFF": "IF_SAFER"}
LAYERS = ("final", "dz", "area")  # the rasters of every run (+ snapshots and comparisons)
BLOCK_CELLS = 2_000_000  # cells of the DEM brought back at a time (a few float32 arrays of this size)
STATS_CELLS = 1_000_000  # the statistics of a raster from a regular sample of at most this many cells
LARGE_DEM_CELLS = 50_000_000  # above this the estimate says that writing the results takes long
MARGIN = 2  # cells of the simulation grid around a block of the DEM (bilinear needs one)
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


def simulation_elevation(path: Path, info: dict, cell: float):
    """The DEM file on the simulation grid, read by GDAL in windows (averaged when the cell is larger
    than the DEM's, else bilinear): ``(elevation float64, transform)``."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    transform, shape = simulation_grid(info, cell)
    coarser = cell > max(info["cell"], info["cell_y"]) * 1.0001
    dst = np.full(shape, np.nan, np.float64)
    with rasterio.open(path) as src:
        reproject(source=rasterio.band(src, 1), destination=dst, src_transform=src.transform, src_crs=_crs(info),
                  dst_transform=transform, dst_crs=_crs(info), src_nodata=_src_nodata(src), dst_nodata=np.nan,
                  resampling=Resampling.average if coarser else Resampling.bilinear)
    return dst, transform


def read_raster(path: Path, max_side: int | None = None, nearest: bool = False):
    """``(values with NaN, transform, crs)`` of a GeoTIFF of the plugin; with ``max_side`` a copy at
    most that many cells a side (averaged, or by nearest neighbour), for the view."""
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as src:
        k = max(1, math.ceil(max(src.width, src.height) / max_side)) if max_side else 1
        if k == 1:
            return src.read(1, masked=True).astype(np.float64).filled(np.nan), src.transform, src.crs
        shape = (max(1, src.height // k), max(1, src.width // k))
        values = src.read(1, out_shape=shape, masked=True,
                          resampling=Resampling.nearest if nearest else Resampling.average)
        kx, ky = src.width / shape[1], src.height / shape[0]
        tr = src.transform
        transform = type(tr)(tr.a * kx, tr.b * ky, tr.c, tr.d * kx, tr.e * ky, tr.f)
        return values.astype(np.float64).filled(np.nan), transform, src.crs


def _profile(info: dict, count: int = 1) -> dict:
    profile = dict(PROFILE, count=count, width=info["width"], height=info["height"], transform=info["transform"],
                   crs=info["crs"])
    if info["width"] < 256 or info["height"] < 256:  # a tiny raster: no tiles larger than itself
        profile.update(tiled=False)
        profile.pop("blockxsize")
        profile.pop("blockysize")
    return profile


def _blocks(width: int, height: int, tiled: bool):
    """Windows of whole rows of about :data:`BLOCK_CELLS` cells (whole rows of tiles for a tiled file)."""
    from rasterio.windows import Window

    rows = max(1, int(BLOCK_CELLS) // max(1, int(width)))
    if tiled:
        rows = max(256, rows - rows % 256)
    for r0 in range(0, height, rows):
        yield Window(0, r0, width, min(rows, height - r0))


def _read_window(src, window) -> np.ndarray:
    """A window of a DEM as float32 with NaN where it has no value."""
    z = src.read(1, window=window, masked=True).astype(np.float32).filled(np.nan)
    if src.nodata is not None and not math.isnan(src.nodata):
        z[z == src.nodata] = np.nan
    return z


def _src_nodata(src):
    """The nodata to give GDAL for a source: its own, or NaN for floats without one."""
    if src.nodata is not None:
        return src.nodata
    return np.nan if np.dtype(src.dtypes[0]).kind == "f" else None


def _on_window(values: np.ndarray, sim_transform, window, dem_transform, crs, nearest: bool = False) -> np.ndarray:
    """An array of the simulation grid on a window of the DEM (float32, NaN out of reach): only the rows
    of the simulation grid that cover the window, with :data:`MARGIN` cells around, are interpolated."""
    from rasterio import windows
    from rasterio.warp import Resampling, reproject

    win_transform = windows.transform(window, dem_transform)
    left, bottom, right, top = windows.bounds(window, dem_transform)
    cell = abs(sim_transform.e)
    rows = values.shape[0]
    s0 = max(0, int(math.floor((sim_transform.f - top) / cell)) - MARGIN)
    s1 = min(rows, int(math.ceil((sim_transform.f - bottom) / cell)) + MARGIN)
    dst = np.full((int(window.height), int(window.width)), np.nan, np.float32)
    if s1 <= s0:
        return dst
    sub = np.ascontiguousarray(values[s0:s1], dtype=np.float32)
    reproject(source=sub, destination=dst, src_transform=windows.transform(windows.Window(0, s0, values.shape[1], s1 - s0), sim_transform),
              src_crs=crs, dst_transform=win_transform, dst_crs=crs, src_nodata=np.nan, dst_nodata=np.nan,
              resampling=Resampling.nearest if nearest else Resampling.bilinear)
    return dst


class _Sample:
    """A regular sample of the cells of a raster read block by block: every ``k``-th row and column,
    ``k`` such that at most :data:`STATS_CELLS` cells are kept."""

    def __init__(self, width: int, height: int):
        self.k = max(1, math.ceil(math.sqrt(width * height / max(1, STATS_CELLS))))
        self.parts: list[np.ndarray] = []

    def add(self, window, values: np.ndarray, keep: np.ndarray | None = None) -> None:
        r0 = int(window.row_off)
        first = (-r0) % self.k
        rows = values[first::self.k, ::self.k]
        ok = np.isfinite(rows) if keep is None else keep[first::self.k, ::self.k] & np.isfinite(rows)
        self.parts.append(rows[ok].astype(np.float32))

    def values(self) -> np.ndarray:
        return np.concatenate(self.parts) if self.parts else np.empty(0, np.float32)


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
    if not (out["cell"] >= 0 and math.isfinite(out["cell"])):
        raise ValueError(f"cell must be a positive length in metres (0: automatic), not {out['cell']:g}")
    if out["dt"] is not None and not (out["dt"] > 0 and math.isfinite(out["dt"])):
        raise ValueError(f"dt must be a positive number of years (0: automatic), not {out['dt']:g}")
    if out["diffusion"] is None and out["fluvial"] is None:
        raise ValueError("no process: soil creep and running water are both off")
    if out["fluvial"] is not None and out["fluvial"]["n_sp"] < 1:
        # below 1 the incision wave is fastest where the slope is least: on flat and lake cells the steps would be
        # hundreds of times shorter than any estimate
        raise ValueError(f"n_sp must be at least 1, not {out['fluvial']['n_sp']:g}")
    out["snapshots"] = [s for s in out["snapshots"] if s < out["years"]]
    return out


def estimate(chunk, params: dict, lang: str | None = None) -> dict:
    """What a run of ``params`` will cost (:func:`engine.estimate`) on the DEM it starts from,
    with the simulation ``cell`` and ``shape`` used, the cells of the DEM (``dem_cells``: above
    :data:`LARGE_DEM_CELLS` the warning says that writing the results takes long). ``lang``: "it"
    or "en", the interface's by default. A DEM in degrees is refused (``ValueError``)."""
    lang = lang or ui_lang()
    p = job_params(params)
    info = dem_info(dem_path(chunk, p["source"]))
    if info["crs"] is not None and info["crs"].is_geographic:
        raise ValueError("Il DEM deve essere in metri (un sistema proiettato o locale): questo è in gradi."
                         if lang == "it" else
                         "The DEM must be in metres (a projected or local system): this one is in degrees.")
    cell = p["cell"] or default_cell(info)
    _transform, shape = simulation_grid(info, cell)
    out = engine.estimate(shape, cell, p["years"], {"diffusion": p["diffusion"], "fluvial": p["fluvial"],
                                                    "dt": p["dt"], "relief": info["relief"]}, lang=lang)
    side = max(shape)
    dem_cells = int(info["width"]) * int(info["height"])
    out.update(cell=cell, shape=list(shape), large=side > engine.WARN_SIDE, dem_cells=dem_cells)
    if dem_cells > LARGE_DEM_CELLS:
        millions = f"{dem_cells / 1e6:.0f}"
        more = (f"Il DEM ha {millions} milioni di celle: riportare i risultati sulla sua griglia richiede alcuni minuti "
                f"e qualche GB su disco." if lang == "it" else
                f"The DEM has {millions} million cells: bringing the results back onto its grid takes a few minutes and "
                f"a few GB on disk.")
        out["warning"] = (out["warning"] + " " + more).strip()
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


def load(chunk, run_id: str, layer: str = "dz", max_side: int | None = None):
    """``(values, transform, crs)`` of a layer of a run; with ``max_side`` a copy at most that many
    cells a side (for the view: a DEM of 10^8 cells is never read whole in the window's thread)."""
    e = entry(chunk, run_id)
    if e is None:
        raise KeyError(run_id)
    path = path_of(chunk, e, layer)
    if path is None or not path.exists():
        raise KeyError(f"{run_id}: no layer {layer!r}")
    return read_raster(path, max_side, nearest=kind_of(layer) == "log")


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
    info = dem_info(src)
    cell = p["cell"] or default_cell(info)
    elevation, sim_transform = simulation_elevation(src, info, cell)
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
        out_dir.mkdir(parents=True, exist_ok=False)
        files, stats = _back_onto_dem(src, info, sim_transform, res, result, out_dir, rid,
                                      progress.sub(0.9, 0.1))
        files["final"], files["dz"] = f"{FOLDER}/{rid}/final.tif", f"{FOLDER}/{rid}/dz.tif"
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
             "created": datetime.now().isoformat(timespec="seconds"), "stats": stats}
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


def _back_onto_dem(src: Path, info: dict, sim_transform, res: Path, result: dict, out_dir: Path, rid: str,
                   progress=None) -> tuple[dict, dict]:
    """The arrays of the worker (``res``) onto the grid of the DEM ``src``, block by block of rows:
    ``final.tif`` = DEM + the change interpolated (0 where the interpolation does not reach),
    ``dz.tif`` = float32(final) - float32(DEM), ``area.tif`` by nearest neighbour, a
    ``snap_<year>.tif`` per snapshot = DEM + its change. Returns the files of ``area`` and of the
    snapshots (relative to the chunk's folder) and the statistics of dz (:func:`_stats` of a regular
    sample)."""
    import rasterio

    say = progress or (lambda *a, **k: None)
    crs = _crs(info)
    initial = np.load(res / "initial.npy")
    changes = [("dz", np.load(res / "dz.npy"))]
    area = np.load(res / "area.npy") if (res / "area.npy").exists() else None
    snaps = {}
    for year, fname in sorted((result.get("files") or {}).get("snapshots", {}).items(), key=lambda kv: float(kv[0])):
        snaps[f"{float(year):g}"] = f"snap_{float(year):g}.tif"
        changes.append((f"snap:{float(year):g}", np.load(res / fname) - initial))
    del initial
    profile = _profile(info)
    outputs = {"final": ("final.tif", "superficie finale (m)"), "dz": ("dz.tif", "erosione (-) e deposito (+), m")}
    if area is not None:
        outputs["area"] = ("area.tif", "area drenata (m2)")
    for year, name_tif in snaps.items():
        outputs[f"snap:{year}"] = (name_tif, f"superficie dopo {year} anni (m)")
    sample = _Sample(info["width"], info["height"])
    opened = {}
    try:
        for key, (fname, description) in outputs.items():
            opened[key] = rasterio.open(out_dir / fname, "w", **profile)
            opened[key].set_band_description(1, description)
        with rasterio.open(src) as dem:
            windows = list(_blocks(info["width"], info["height"], bool(profile.get("tiled"))))
            for i, window in enumerate(windows):
                z32 = _read_window(dem, window)
                valid = np.isfinite(z32)
                zd = z32.astype(np.float64)
                for key, sim_change in changes:
                    back = _on_window(sim_change, sim_transform, window, info["transform"], crs)
                    surface = np.where(valid, zd + np.where(np.isfinite(back), back, 0.0), np.nan).astype(np.float32)
                    del back
                    if key == "dz":
                        # dz written as the difference of the two float32 surfaces: final - initial = dz in the files
                        dz = (surface.astype(np.float64) - zd).astype(np.float32)
                        opened["final"].write(surface, 1, window=window)
                        opened["dz"].write(dz, 1, window=window)
                        sample.add(window, dz)
                        del dz
                    else:
                        opened[key].write(surface, 1, window=window)
                    del surface
                if area is not None:
                    back = _on_window(area, sim_transform, window, info["transform"], crs, nearest=True)
                    opened["area"].write(np.where(valid, back, np.nan).astype(np.float32), 1, window=window)
                    del back
                say((i + 1) / len(windows), t("Risultati sulla griglia del DEM", "Results onto the grid of the DEM"))
    finally:
        for dst in opened.values():
            dst.close()
    files = {"snapshots": {year: f"{FOLDER}/{rid}/{name_tif}" for year, name_tif in snaps.items()}}
    if area is not None:
        files["area"] = f"{FOLDER}/{rid}/area.tif"
    return files, _stats(sample.values())


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
    from rasterio import windows
    from rasterio.warp import Resampling, reproject

    e = entry(chunk, run_id)
    if e is None:
        raise KeyError(run_id)
    other_path = dem_path(other, source)
    key = str(getattr(other, "key", "x")) + ("" if source in (None, "", "active") else f"-{source}")
    out = folder(chunk) / run_id / f"compare_{key}.tif"
    with rasterio.open(path_of(chunk, e, "final")) as fin, rasterio.open(path_of(chunk, e, "dz")) as dzf, \
            rasterio.open(other_path) as later:
        transform, crs = fin.transform, fin.crs
        if crs is not None and later.crs is not None:
            src_crs, dst_crs = later.crs, crs
        else:  # a local frame on one side at least: the same frame on both (as the program's chunks share it)
            src_crs = dst_crs = rasterio.crs.CRS.from_string(STANDIN_CRS)
        info = {"width": fin.width, "height": fin.height, "transform": transform, "crs": crs}
        profile = _profile(info)
        diffs, bases = _Sample(fin.width, fin.height), _Sample(fin.width, fin.height)
        out.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out, "w", **profile) as dst:
            dst.set_band_description(1, f"superficie finale - DEM di {getattr(other, 'label', key)} (m)")
            for window in _blocks(fin.width, fin.height, bool(profile.get("tiled"))):
                final = fin.read(1, window=window, masked=True).astype(np.float64).filled(np.nan)
                dz = dzf.read(1, window=window, masked=True).astype(np.float64).filled(np.nan)
                onto = np.full(final.shape, np.nan, np.float64)
                reproject(source=rasterio.band(later, 1), destination=onto, src_transform=later.transform,
                          src_crs=src_crs, dst_transform=windows.transform(window, transform), dst_crs=dst_crs,
                          src_nodata=_src_nodata(later), dst_nodata=np.nan, resampling=Resampling.bilinear)
                if later.nodata is not None and not math.isnan(later.nodata):
                    onto[onto == later.nodata] = np.nan
                diff = final - onto
                dst.write(np.where(np.isfinite(diff), diff, np.nan).astype(np.float32), 1, window=window)
                ok = np.isfinite(diff)
                start = final - dz  # the DEM the run started from, as it was then
                diffs.add(window, diff, ok)
                bases.add(window, start - onto, ok & np.isfinite(start))
    result = {"chunk_key": key, "chunk": getattr(other, "label", key), "source": source or "active",
              "source_name": source_name(other, source), "file": f"{FOLDER}/{run_id}/{out.name}",
              "stats": _stats(diffs.values()), "baseline": _stats(bases.values()),
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
    with rasterio.open(path_of(chunk, e, names[0])) as first:
        info = {"width": first.width, "height": first.height, "transform": first.transform, "crs": first.crs}
    profile = _profile(info, count=len(names))
    with rasterio.open(path, "w", **profile) as dst:
        for i, k in enumerate(names, start=1):
            with rasterio.open(path_of(chunk, e, k)) as src:
                for window in _blocks(info["width"], info["height"], bool(profile.get("tiled"))):
                    dst.write(src.read(1, window=window).astype(np.float32), i, window=window)
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
