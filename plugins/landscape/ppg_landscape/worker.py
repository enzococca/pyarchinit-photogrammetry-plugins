"""A landscape scenario with Landlab, in a process of its own.

Landlab brings compiled libraries (richdem, netCDF, its Cython helpers) and a run can take minutes:
the plugin never imports it in the program's process, it runs this file with a Python that has
Landlab - the program's own when Landlab is installed there, else the environment the plugin
installs - and only the files and the lines below go between the two.

    python -u worker.py JOB.json

This file imports nothing of the program or of the plugin but :mod:`engine`, loaded by its path
from the same folder (the environment of the plugin has only Landlab and its dependencies). The
job::

    {"protocol": 1, "elevation": "/abs/elevation.npy", "mask": "/abs/mask.npy" | null,
     "output": "/abs/run", "cell": 0.5, "years": 1000, "dt": null, "snapshots": [10, 100],
     "boundary": "open", "diffusion": {"D": 0.01} | null,
     "fluvial": {"K_sed": ..., "K_br": ..., "F_f": ..., "phi": ..., "H_star": ..., "v_s": ...,
                 "m_sp": ..., "n_sp": ..., "runoff": ..., "soil_depth": ...} | null,
     "router": "auto", "eroder": "auto"}

``elevation``: a 2-D array in raster order (row 0 north) on the simulation grid, NaN without a
value; ``mask``: True for the cells to use. ``output`` must not exist: the files are written in
``<output>.partial`` and the folder is renamed to ``output`` only when every file is there, so a
run stopped half-way (the plugin kills the process to cancel) never leaves a result that looks
complete. The files: ``initial.npy``, ``final.npy``, ``dz.npy`` (final - initial), ``area.npy``
(drainage area of the final surface, m2), ``snap_<year>.npy``, ``balance.json``; float64, raster
order, NaN outside the data.

Every message is one line on stdout, the prefix ``@ppg-landscape`` and a JSON object:

* ``{"type": "hello", "protocol": 1, "backend": "landlab", "versions": {...}}``
* ``{"type": "progress", "fraction": 0.4, "phase": "run", "message": "...", "year": 120, "years": 1000}``
  (phases ``load``, ``grid``, ``run``, ``area``, ``save``)
* ``{"type": "result", "result": {"output": ..., "files": {...}, "balance": {...}, ...}}``
* ``{"type": "error", "error": "...", "phase": "...", "exception": "...", "traceback": "..."}``

Other lines on stdout and stderr are Landlab's own: the plugin keeps the last ones for the report
of a failure. The exit code is 0 with a result, 1 after an error.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

PROTOCOL = 1
PREFIX = "@ppg-landscape "
PHASE_TEXT = {"load": "reading the DEM", "grid": "building the grid", "area": "drainage area",
              "save": "writing the results", "done": "done"}


def emit(message_type: str, **data) -> None:
    """One message of the protocol, on a line of its own even after an unfinished line of Landlab."""
    sys.stdout.write("\n" + PREFIX + json.dumps({"type": message_type, **data}) + "\n")
    sys.stdout.flush()


def load_engine():
    """:mod:`engine`, from the file beside this one (the worker is not run as part of a package)."""
    name = "ppg_landscape_engine"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("engine.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _plain(value):
    """JSON from numpy scalars, arrays, dictionaries."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, float) and value != value:  # NaN is not JSON
        return None
    if isinstance(value, float) and value in (float("inf"), float("-inf")):
        return None
    return value


def _progress(fraction: float, phase: str, **info) -> None:
    if phase == "run" and "year" in info:
        message = f"year {info['year']:g} of {info['years']:g}"
    else:
        message = PHASE_TEXT.get(phase, phase)
    emit("progress", fraction=round(float(fraction), 4), phase=phase, message=message, **_plain(info))


def write_results(result, folder: Path) -> dict:
    """The arrays and the balance of an :class:`engine.Result` in ``folder``; the names written."""
    engine = load_engine()
    files = {}
    for key, array in (("initial", result.initial), ("final", result.final), ("dz", result.dz), ("area", result.area)):
        if array is None:
            continue
        np_save(folder / f"{key}.npy", array)
        files[key] = f"{key}.npy"
    snaps = {}
    for year in sorted(result.snapshots):
        name = engine.snapshot_name(year)
        np_save(folder / name, result.snapshots[year])
        snaps[f"{float(year):g}"] = name
    files["snapshots"] = snaps
    (folder / "balance.json").write_text(json.dumps(_plain(result.balance), indent=1), encoding="utf-8")
    files["balance"] = "balance.json"
    return files


def np_save(path: Path, array) -> None:
    import numpy as np

    np.save(path, np.asarray(array, dtype=np.float64), allow_pickle=False)


def run_job(job: dict, simulate=None, versions=None) -> dict:
    """Simulate ``job`` (``simulate``: :func:`engine.run` or another function with its signature),
    write its files, send the messages; the result (also the last message)."""
    import numpy as np

    engine = load_engine()
    simulate = simulate or engine.run
    t0 = time.monotonic()
    emit("hello", protocol=PROTOCOL, backend="landlab", versions=_plain((versions or engine.versions)()))
    where = {"phase": "load"}
    try:
        output = Path(job["output"]).absolute()
        if output.exists():
            raise FileExistsError(f"the output folder exists already: {output}")
        partial = output.with_name(output.name + ".partial")
        if partial.exists():  # what a stopped run left
            shutil.rmtree(partial)
        _progress(0.0, "load")
        elevation = np.load(job["elevation"], allow_pickle=False)
        valid = np.load(job["mask"], allow_pickle=False).astype(bool) if job.get("mask") else None
        if valid is not None and valid.shape != elevation.shape:
            raise ValueError(f"the mask is {valid.shape}, the elevation {elevation.shape}")

        def progress(fraction, phase, **info):
            where["phase"] = "run" if phase in ("run", "done") else phase
            _progress(0.02 + 0.93 * float(fraction), phase, **info)

        where["phase"] = "grid"
        result = simulate(job, elevation, valid, progress)
        where["phase"] = "save"
        _progress(0.96, "save")
        partial.mkdir(parents=True)
        files = write_results(result, partial)
        os.replace(partial, output)  # complete, or not there at all
    except Exception as exc:
        try:
            exc.ppg_where = dict(where)
        except AttributeError:
            pass
        raise
    out = {"output": str(output), "files": files, "balance": _plain(result.balance), "shape": list(elevation.shape),
           "seconds": round(time.monotonic() - t0, 3)}
    emit("result", result=out)
    return out


def main(argv=None, simulate=None, versions=None) -> int:
    """Run the job of the file named first in ``argv``; ``simulate``/``versions``: another
    implementation (the tests use one without Landlab)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if not argv:
            raise ValueError("usage: worker.py JOB.json")
        job = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
        if int(job.get("protocol", PROTOCOL)) != PROTOCOL:
            raise ValueError(f"protocol {job.get('protocol')}: this worker speaks {PROTOCOL}")
        run_job(job, simulate, versions)
        return 0
    except Exception as exc:  # noqa: BLE001 - every failure goes back to the plugin as a message
        where = getattr(exc, "ppg_where", None) or {"phase": "start"}
        emit("error", error=str(exc).strip() or type(exc).__name__, exception=type(exc).__name__,
             phase=where["phase"], traceback=traceback.format_exc())
        return 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
