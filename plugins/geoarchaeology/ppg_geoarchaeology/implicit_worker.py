"""The implicit model with LoopStructural, in a process of its own.

LoopStructural brings pandas 3 and a good part of the scientific stack: the plugin never imports it
in the program's process, it runs this file with a Python that has it - the program's own when
it is installed there, else the environment the plugin installs - and only the files and the lines
below go between the two.

    python -I -u implicit_worker.py JOB.json

This file imports nothing of the program or of the plugin but :mod:`implicit_engine`, loaded by
its path from the same folder. ``loop_common`` imports pyvista at the top of one module and uses it
only to draw: a stub takes its place in ``sys.modules`` before anything else, also in a Python that
has pyvista, so VTK is never loaded. The job::

    {"protocol": 1, "output": "/abs/run", "dem": "/abs/dem.npy",
     "grid": {"x0": ..., "y0": ..., "cell": ..., "width": ..., "height": ...},
     "zmin": ..., "zmax": ..., "dz": ...,
     "series": [{"name": "ground", "space": "depth" | "elevation", "units": ["US 2", ...]}, ...],
     "contacts": [[unit, x, y, z, weight, hole], ...],
     "orientations": [[unit, x, y, z, nx, ny, nz], ...],
     "column": [{"unit": "US 1", "kind": "layer" | "cut" | "fill"}, ...],
     "fills": {"US 7": "T1"},
     "cuts": {"T1": {"base": [[x, y, z], ...], "edge": [[x, y, z], ...],
                     "inside": [[x, y], ...], "outside": [[x, y], ...]}},
     "options": {"nelements": 20000, "vertical": null, "folds": null, "seed": 20261030}}

* ``grid`` and ``dem``: the raster, north up, ``(x0, y0)`` its top-left corner; ``dem.npy`` a 2-D array
  on it, NaN without a value. The voxels go from ``zmin`` to ``zmax`` by ``dz``.
* ``series``, oldest first: each is one scalar field. The order of the units comes from the ``column``.
* ``contacts``: the *top* of a unit at a point. ``weight`` and ``hole`` may be missing or null: without
  a hole, the rows at one position are one borehole for the leave-one-out. A point where a fill rests
  on a truncated layer belongs in the ``base`` of the cut, not among the contacts of that layer.
* ``orientations``: the normal of the top of a unit at a point, in the world, pointing up; of a
  ``cut``: of its surface, pointing into the cut.
* ``column``: the units top-down, the top one a unit with material. A cut has no material: it sits
  under its fills and above what it cuts. ``fills`` says which cut each fill fills.
* ``cuts``: ``base`` the bottom or the wall of the cut where a borehole or a section met it; ``edge``
  the closed outline on the ground in order, empty when nobody drew it (it is then estimated from
  ``inside`` and ``outside``, and a warning says so); ``inside`` and ``outside`` the positions of the
  holes in and out of the cut.
* ``options``: ``vertical`` null chooses the exaggeration from the thinnest unit; ``folds`` null leaves
  out the leave-one-out.

``output`` must not exist: the files are written in ``<output>.partial`` and the folder is renamed to
``output`` only when every file is there, so a run stopped half-way (the plugin kills the process
to cancel) never leaves a result that looks complete. The files, ``i`` an index of ``units``:

* ``labels.npy``: int16 (rows, cols, nz), ``-1`` air, else ``i``; level 0 is the lowest;
* ``top_<i>.npy``: float32 raster, the top of unit ``i`` and of those under it in the column
  (:func:`implicit_engine.top_of`), NaN outside the DEM; a cut has none;
* ``variance_<i>.npy``: float32 raster, the jackknife variance (m2) of the contact, with ``folds``;
* ``solid_<i>_vertices.npy`` (float64, world metres), ``solid_<i>_faces.npy`` (int32, outward);
* ``model.json``: what the result carries.

Every message is one line on stdout, the prefix ``@ppg-geoarch`` and a JSON object:

* ``{"type": "hello", "protocol": 1, "backend": "loopstructural", "versions": {...}}``
* ``{"type": "progress", "fraction": 0.4, "phase": "fit", "message": "..."}``
  (phases ``load``, ``fit``, ``paint``, ``tops``, ``solids``, ``loo``, ``save``)
* ``{"type": "result", "result": {"output": ..., "units": [...], "files": {...}, "volumes": {...},
  "loo": {unit: {"rmse", "mae", "n", "nan"}}, "info": {"vertical", "cuts", "warnings", ...}}}``
* ``{"type": "error", "error": "...", "phase": "...", "exception": "...", "traceback": "..."}``

``info["warnings"]`` are dictionaries with a ``code`` (``edge_estimated``, ``cut_without_data``,
``series_without_data``, ``unit_without_contact``): the plugin says them in the language of the
interface. Other lines on stdout and stderr are LoopStructural's own: the plugin keeps the last ones
for the report of a failure. The exit code is 0 with a result, 1 after an error.
"""
from __future__ import annotations

import sys
import types

sys.modules["pyvista"] = types.ModuleType("pyvista")  # before anything can import loop_common

import importlib.util  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

logging.disable(logging.INFO)  # LoopStructural says a lot at this level
PROTOCOL = 1
PREFIX = "@ppg-geoarch "
PHASE_TEXT = {"load": "reading the DEM", "fit": "fitting the fields", "paint": "painting the units",
              "tops": "the tops of the units", "solids": "the solids", "loo": "leave-one-out",
              "save": "writing the results", "done": "done"}


def emit(message_type: str, **data) -> None:
    """One message of the protocol, on a line of its own even after an unfinished line of the library."""
    sys.stdout.write("\n" + PREFIX + json.dumps({"type": message_type, **data}) + "\n")
    sys.stdout.flush()


def load_engine():
    """:mod:`implicit_engine`, from the file beside this one (the worker is not part of a package)."""
    name = "ppg_geoarchaeology_implicit_engine"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("implicit_engine.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def plain(value):
    """JSON from numpy scalars, arrays, dictionaries."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None  # not JSON
    return value


def _progress(fraction: float, phase: str, **info) -> None:
    emit("progress", fraction=round(float(fraction), 4), phase=phase, message=PHASE_TEXT.get(phase, phase),
         **plain(info))


def write_results(result, folder: Path) -> dict:
    """The arrays and the summary of a :class:`implicit_engine.Result` in ``folder``; the names written."""
    import numpy as np

    index = {unit: i for i, unit in enumerate(result.units)}
    files: dict = {"labels": "labels.npy", "tops": {}, "variance": {}, "solids": {}}
    np.save(folder / "labels.npy", result.labels, allow_pickle=False)
    for unit, array in result.tops.items():
        files["tops"][unit] = f"top_{index[unit]}.npy"
        np.save(folder / files["tops"][unit], array, allow_pickle=False)
    for unit, array in result.variance.items():
        files["variance"][unit] = f"variance_{index[unit]}.npy"
        np.save(folder / files["variance"][unit], array, allow_pickle=False)
    for unit, (vertices, faces) in result.solids.items():
        names = [f"solid_{index[unit]}_vertices.npy", f"solid_{index[unit]}_faces.npy"]
        np.save(folder / names[0], vertices, allow_pickle=False)
        np.save(folder / names[1], faces, allow_pickle=False)
        files["solids"][unit] = names
    return files


def run_job(job: dict, build=None, versions=None) -> dict:
    """Build ``job`` (``build``: :func:`implicit_engine.run` or another function with its
    signature), write its files, send the messages; the result (also the last message)."""
    import numpy as np

    t0 = time.monotonic()
    emit("hello", protocol=PROTOCOL, backend="loopstructural", versions=plain((versions or load_engine().versions)()))
    where = {"phase": "load"}
    try:
        output = Path(job["output"]).absolute()
        if output.exists():
            raise FileExistsError(f"the output folder exists already: {output}")
        partial = output.with_name(output.name + ".partial")
        if partial.exists():  # what a stopped run left
            shutil.rmtree(partial)
        _progress(0.0, "load")
        dem = np.load(job["dem"], allow_pickle=False)

        def progress(fraction, phase, **info):
            where["phase"] = phase
            _progress(0.02 + 0.93 * float(fraction), phase, **info)

        result = (build or load_engine().run)(job, dem, progress)
        where["phase"] = "save"
        _progress(0.96, "save")
        partial.mkdir(parents=True)
        files = write_results(result, partial)
        out = {"output": str(output), "units": list(result.units), "files": files, "volumes": plain(result.volumes),
               "loo": plain(result.loo), "info": plain(result.info), "seconds": round(time.monotonic() - t0, 3)}
        (partial / "model.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        os.replace(partial, output)  # complete, or not there at all
    except Exception as exc:
        try:
            exc.ppg_where = dict(where)
        except AttributeError:
            pass
        raise
    emit("result", result=out)
    return out


def main(argv=None, build=None, versions=None) -> int:
    """Run the job of the file named first in ``argv``; ``build``/``versions``: another
    implementation (the tests use one without LoopStructural)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if not argv:
            raise ValueError("usage: implicit_worker.py JOB.json")
        job = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
        if int(job.get("protocol", PROTOCOL)) != PROTOCOL:
            raise ValueError(f"protocol {job.get('protocol')}: this worker speaks {PROTOCOL}")
        run_job(job, build, versions)
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
