"""A worker that speaks the protocol of ``ppg_geoarchaeology/implicit_worker.py`` - the very code of it,
loaded from its file - with a model in numpy instead of LoopStructural, for the tests of the plugin
where LoopStructural is not installed. It runs as the real one does, in a process of its own:

    python -u implicit_fake_worker.py JOB.json

The model: one unit under the DEM, its solid a tetrahedron. ``PPG_GEOARCH_FAKE`` changes what it
does: ``fail`` (the model raises), ``memory`` (the allocation fails), ``crash`` (the process dies as on a
segmentation fault in the middle of the run), ``crash_save`` (it dies after writing the first file:
the partial folder stays behind), ``sleep`` (a phase that does not end: for cancelling), ``noise``
(the library's own lines among the messages).
"""
from __future__ import annotations

import importlib.util
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("ppg_geoarch_worker_fake", HERE.parent / "ppg_geoarchaeology" /
                                               "implicit_worker.py")
worker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(worker)
MODE = os.environ.get("PPG_GEOARCH_FAKE", "")


@dataclass
class Result:
    units: list
    labels: np.ndarray
    tops: dict = field(default_factory=dict)
    variance: dict = field(default_factory=dict)
    solids: dict = field(default_factory=dict)
    volumes: dict = field(default_factory=dict)
    loo: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)


def _die() -> None:
    sys.stdout.flush()
    if hasattr(signal, "SIGSEGV") and os.name != "nt":
        os.kill(os.getpid(), signal.SIGSEGV)
    os._exit(3)


def build(job, dem, progress):
    if MODE == "fail":
        raise ValueError("the column has no unit")
    if MODE == "memory":
        raise MemoryError("Unable to allocate 3.1 GiB")
    if MODE == "noise":
        sys.stdout.write("LoopStructural says something unfinished")  # no newline
        sys.stdout.flush()
        print("\nanother line of the library", flush=True)
    progress(0.1, "fit")
    if MODE == "crash":
        _die()
    if MODE == "sleep":
        while True:
            time.sleep(0.1)
    progress(0.5, "paint")
    nz = max(int((job["zmax"] - job["zmin"]) / job["dz"]), 1)
    labels = np.zeros(dem.shape + (nz,), dtype=np.int16)
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]], dtype=np.int32)
    progress(0.9, "solids")
    return Result(["A"], labels, {"A": dem.astype(np.float32)}, {"A": np.zeros(dem.shape, np.float32)},
                  {"A": (vertices, faces)}, {"A": 1.5}, {"A": {"rmse": 0.1, "mae": 0.05, "n": 4, "nan": 0}},
                  {"vertical": 8.0, "warnings": [{"code": "edge_estimated", "unit": "C"}], "nan": float("nan")})


if MODE == "crash_save":

    def write_results(result, folder):
        np.save(folder / "labels.npy", result.labels)
        _die()

    worker.write_results = write_results


if __name__ == "__main__":
    sys.exit(worker.main(sys.argv[1:], build=build, versions=lambda: {"loopstructural": "0.0-fake", "python": "3"}))
