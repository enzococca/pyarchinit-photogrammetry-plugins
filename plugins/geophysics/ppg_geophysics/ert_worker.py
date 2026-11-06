"""Electrical resistivity tomography with pyGIMLi, in a process of its own.

pyGIMLi's core (``pgcore``) carries compiled libraries and its own OpenMP runtime: the plugin never
imports it in the program's process. It runs this file with a Python that has pyGIMLi - the
program's own when pyGIMLi is installed there, else the environment the plugin installs - and only
the files and the lines below go between the two.

    python -I -u ert_worker.py JOB.json

This file imports nothing of the program or of the plugin but :mod:`ert_engine`, loaded by its path
from the same folder. The job::

    {"protocol": 1, "task": "read" | "invert" | "simulate", "output": "/abs/result.npz",
     "source": "/abs/file.dat"            (read),
     "input": "/abs/profile.npz"          (invert: sensors, a b m n, rhoa, r, err, u, i, valid),
     "params": {...}                      (invert: see ert_engine.invert),
     "sensors": [[x, z], ...], "model": {...}, "scheme": "dd", "noise": 0.02   (simulate)}

The result goes to ``<output>.partial.npz`` and is renamed to ``output`` only when it is complete, so
a run stopped half-way (the plugin kills the process to cancel) never leaves a result that looks
whole. Every message is one line on stdout, the prefix ``@ppg-geophysics`` and a JSON object:

* ``{"type": "hello", "protocol": 1, "backend": "pygimli", "versions": {...}}``
* ``{"type": "progress", "fraction": 0.4, "phase": "invert"}`` (phases ``load``, ``data``,
  ``factors``, ``mesh``, ``invert``, ``doi``, ``forward``, ``save``, ``done``)
* ``{"type": "result", "result": {"output": ..., "fit": {...}, "present": [...], "seconds": ...}}``
* ``{"type": "error", "error": "...", "phase": "...", "exception": "...", "traceback": "..."}``

Other lines on stdout and stderr are pyGIMLi's own: the plugin keeps the last ones for the report of
a failure. The exit code is 0 with a result, 1 after an error.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

PROTOCOL = 1
PREFIX = "@ppg-geophysics "


def emit(message_type: str, **data) -> None:
    sys.stdout.write("\n" + PREFIX + json.dumps({"type": message_type, **data}) + "\n")
    sys.stdout.flush()


def load_engine():
    name = "ppg_geophysics_ert_engine"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("ert_engine.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    return value


def save_npz(path: Path, arrays: dict) -> None:
    import numpy as np

    partial = path.with_name(path.name + ".partial.npz")
    np.savez(partial, **{k: np.asarray(v) for k, v in arrays.items()})
    os.replace(partial, path)


def run_job(job: dict, engine=None, versions=None) -> dict:
    import numpy as np

    engine = engine or load_engine()
    t0 = time.monotonic()
    emit("hello", protocol=PROTOCOL, backend="pygimli", versions=_plain((versions or engine.versions)()))
    where = {"phase": "load"}

    def progress(fraction, phase, **info):
        where["phase"] = phase
        emit("progress", fraction=round(float(fraction), 4), phase=phase, **_plain(info))

    try:
        output = Path(job["output"]).absolute()
        task = job.get("task")
        progress(0.0, "load")
        if task == "read":
            arrays = engine.read(job["source"])
        elif task == "invert":
            with np.load(job["input"], allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
            arrays = engine.invert(arrays, dict(job.get("params") or {}), progress)
        elif task == "simulate":
            arrays = engine.simulate(job["sensors"], job.get("model") or {}, job.get("scheme", "dd"),
                                     float(job.get("noise", 0.02)), float(job.get("noise_u", 1e-6)),
                                     int(job.get("seed", 1337)), progress)
        else:
            raise ValueError(f"unknown task {task!r}")
        progress(0.99, "save")
        fit = arrays.pop("fit", None)
        present = arrays.pop("present", None)
        output.parent.mkdir(parents=True, exist_ok=True)
        save_npz(output, arrays)
    except Exception as exc:
        try:
            exc.ppg_where = dict(where)
        except AttributeError:
            pass
        raise
    out = {"output": str(output), "fit": _plain(fit or {}), "present": _plain(present or []),
           "seconds": round(time.monotonic() - t0, 3)}
    emit("result", result=out)
    return out


def main(argv=None, engine=None, versions=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if not argv:
            raise ValueError("usage: ert_worker.py JOB.json")
        job = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
        if int(job.get("protocol", PROTOCOL)) != PROTOCOL:
            raise ValueError(f"protocol {job.get('protocol')}: this worker speaks {PROTOCOL}")
        run_job(job, engine, versions)
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
