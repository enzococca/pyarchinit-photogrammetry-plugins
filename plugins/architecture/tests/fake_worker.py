"""A worker that speaks the protocol of ``ppg_architecture/worker.py`` - the very code of it, loaded from its file -
with the solids made by the plugin's engine (trimesh meshes) instead of CadQuery, for the tests where CadQuery is not
installed. It runs as the real one does, in a process of its own:

    python -u fake_worker.py JOB.json

The STEP it writes is a text that says it is fake; the STL is the engine's mesh of the solid (a revolved polyline,
a mitred sweep, an extruded outline), placed as CadQuery places it. ``PPG_ARCHITECTURE_FAKE`` changes what it does:
``fail`` (the solid is "not valid"), ``crash`` (the process dies as on a segmentation fault after writing the STEP:
the partial folder stays behind), ``sleep`` (a long OpenCascade operation: for cancelling), ``noise`` (lines of
OpenCascade among the messages).
"""
from __future__ import annotations

import importlib.util
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("ppg_architecture_worker_fake", HERE.parent / "ppg_architecture" / "worker.py")
worker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(worker)
_spec = importlib.util.spec_from_file_location("ppg_architecture_engine_fake", HERE.parent / "ppg_architecture" / "engine.py")
engine = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = engine
_spec.loader.exec_module(engine)
MODE = os.environ.get("PPG_ARCHITECTURE_FAKE", "")


def _die() -> None:
    sys.stdout.flush()
    if hasattr(signal, "SIGSEGV") and os.name != "nt":
        os.kill(os.getpid(), signal.SIGSEGV)
    os._exit(3)


class Shape:
    """What the worker asks of a CadQuery shape, on a trimesh."""

    def __init__(self, mesh):
        self.mesh = mesh

    def isValid(self):
        return MODE != "fail" and self.mesh.is_watertight

    def Volume(self):
        return float(abs(self.mesh.volume)) if MODE != "fail" else 0.0

    def Area(self):
        return float(self.mesh.area)

    def BoundingBox(self):
        lo, hi = self.mesh.bounds

        class B:
            xmin, ymin, zmin = lo
            xmax, ymax, zmax = hi
        return B


def place(cq, shape, matrix):
    mesh = shape.mesh.copy()
    mesh.apply_transform(np.asarray(matrix, float))
    return Shape(mesh)


class Exporters:
    @staticmethod
    def export(shape, path, exportType=None, tolerance=None, angularTolerance=None):
        if exportType == "STEP":
            Path(path).write_text("ISO-10303-21; fake STEP of the tests\n")
            if MODE == "crash":
                _die()
        else:
            shape.mesh.export(path)


class FakeCadQuery:
    exporters = Exporters


def column(cq, spec):
    if MODE == "sleep":
        time.sleep(30)
    prof = np.asarray(spec["profile"], float)
    return Shape(engine.revolved_mesh(prof, segments=360))


def moulding(cq, spec):
    return Shape(engine.sweep_mesh(np.asarray(spec["path"], float), np.asarray(spec["section"], float)))


def arch(cq, spec):
    poly = engine.outline_polygon(spec["outline"], arc_points=181)
    return Shape(engine.prism_mesh(poly, float(spec["depth"])))


worker.BUILDERS = {"column": column, "moulding": moulding, "arch": arch}
worker.place = place
worker.versions = lambda: {"cadquery": "0.0-fake", "python": "3"}
sys.modules["cadquery"] = FakeCadQuery  # "import cadquery as cq" inside the worker

if __name__ == "__main__":
    if MODE == "noise":
        print("BRep_API: command not done (a line of OpenCascade)", flush=True)
    sys.exit(worker.main(sys.argv[1:]))
