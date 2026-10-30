"""A worker for the tests of the plugin side of the implicit model: it speaks the protocol of the real one,
with its very code, and answers any job with flat layers under the DEM, half a metre apart, and a tetrahedron
per unit. ``PPG_GEOARCH_FAKE=fail`` makes it raise after a while, as a model that does not solve.

    python -u implicit_plugin_fake_worker.py JOB.json
"""
from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("ppg_geoarch_worker_plugin_fake", HERE.parent / "ppg_geoarchaeology" /
                                               "implicit_worker.py")
worker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(worker)


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


def build(job, dem, progress):
    progress(0.1, "fit")
    if os.environ.get("PPG_GEOARCH_FAKE") == "fail":
        raise ValueError("the fake model does not solve")
    units = [e["unit"] for e in job["column"] if e["kind"] != "cut"]
    zmin, dz = job["zmin"], job["dz"]
    nz = max(int(np.ceil((job["zmax"] - zmin) / dz - 1e-9)), 1)
    zc = zmin + (np.arange(nz) + 0.5) * dz
    tops = {u: dem.astype(np.float32) - 0.5 * i for i, u in enumerate(units)}
    labels = np.full(dem.shape + (nz,), -1, dtype=np.int16)
    for i, u in enumerate(units):
        under = zc[None, None, :] < np.nan_to_num(tops[u], nan=-1e9)[:, :, None]
        labels[under] = i
    grid = job["grid"]
    x, y = grid["x0"] + grid["cell"], grid["y0"] - grid["cell"]
    solids = {}
    for i, u in enumerate(units):
        vertices = np.array([[x, y, 0], [x + 1, y, 0], [x, y + 1, 0], [x, y, 1]], dtype=np.float64) + [0, 0, zmin + i]
        faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]], dtype=np.int32)
        solids[u] = (vertices, faces)
    cuts = {}
    for cut, spec in (job.get("cuts") or {}).items():
        cuts[cut] = {"edge": "drawn" if spec.get("outline") else "estimated", "profile": spec.get("profile"), "field": False,
                     "plan_area": 50.0, "volume_range": {"flat": 100.0, "bowl": 80.0, "v": 50.0},
                     "cut_volume_range": {"flat": 120.0, "bowl": 100.0, "v": 70.0},
                     "base_miss": [0.0] * len(spec.get("base") or ()), "base_capped": [False] * len(spec.get("base") or ())}
    series = [u for s in job["series"] for u in s["units"]]
    return Result(units, labels, tops, {u: np.full(dem.shape, 0.01, np.float32) for u in series}, solids,
                  {u: float((labels == i).sum() * grid["cell"] ** 2 * dz) for i, u in enumerate(units)},
                  {u: {"rmse": 0.25, "mae": 0.2, "n": 4, "nan": 0} for u in series} if job["options"].get("folds") else {},
                  {"vertical": 8.0, "cuts": cuts, "warnings": [{"code": "few_holes", "series": "ground"}]})


if __name__ == "__main__":
    sys.exit(worker.main(sys.argv[1:], build=build, versions=lambda: {"loopstructural": "0.0-fake", "python": "3"}))
