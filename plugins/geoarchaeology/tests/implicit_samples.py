"""The plugin's implicit modules loaded by their path, and the synthetic grounds of their tests.

``implicit_engine`` and ``implicit_worker`` import nothing of the program: the engine's tests run in
a Python with LoopStructural only (no program, no VTK), as the worker does::

    <env>/bin/python -m pytest plugins/geoarchaeology/tests/test_implicit_engine.py -q
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parent.parent / "ppg_geoarchaeology"


def load(name: str):
    """``ppg_geoarchaeology/<name>.py`` as a module of its own (no package, no program)."""
    key = f"ppg_geoarchaeology_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, PACKAGE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


def has_loopstructural() -> bool:
    return importlib.util.find_spec("LoopStructural") is not None and importlib.util.find_spec("skimage") is not None


def stub_pyvista() -> None:
    """What the worker does before anything else: here the engine runs in the test's process."""
    sys.modules.setdefault("pyvista", types.ModuleType("pyvista"))


def loop_python() -> Path | None:
    """A Python with LoopStructural for the tests of the real worker: ``PPG_TEST_LOOP``, else this
    Python when it has LoopStructural, else None."""
    given = os.environ.get("PPG_TEST_LOOP")
    if given:
        return Path(given)
    return Path(sys.executable) if has_loopstructural() else None


class Ground:
    """A raster of ``n`` x ``n`` cells with its corner at (0, ``n * cell``)."""

    def __init__(self, n: int = 60, cell: float = 1.0):
        self.n, self.cell = n, cell
        self.xs = (np.arange(n) + 0.5) * cell
        self.ys = n * cell - (np.arange(n) + 0.5) * cell
        self.X, self.Y = np.meshgrid(self.xs, self.ys)

    @property
    def grid(self) -> dict:
        return {"x0": 0.0, "y0": self.n * self.cell, "cell": self.cell, "width": self.n, "height": self.n}

    def holes(self, per_side: int = 5, margin: float = 0.1, jitter: float = 0.0, seed: int = 3) -> np.ndarray:
        """Boreholes on a regular grid, moved at random by up to ``jitter`` metres."""
        size = self.n * self.cell
        line = np.linspace(margin * size, (1 - margin) * size, per_side)
        xy = np.array([(x, y) for y in line for x in line])
        return xy + np.random.default_rng(seed).uniform(-jitter, jitter, xy.shape)


def layers_job(ground: Ground, dem: np.ndarray, contacts: list, units: list, spaces: dict | None = None,
               zmin: float = 0.0, zmax: float = 12.0, dz: float = 0.25, **options) -> dict:
    """A job of plain layers, one series (or the ``spaces``: series name -> (space, units))."""
    series = ([{"name": k, "space": v[0], "units": v[1]} for k, v in spaces.items()] if spaces else
              [{"name": "layers", "space": "elevation", "units": units}])
    return {"grid": ground.grid, "zmin": zmin, "zmax": zmax, "dz": dz, "series": series, "contacts": contacts,
            "orientations": [], "column": [{"unit": u, "kind": "layer"} for u in units], "options": options}


def bowl(r: np.ndarray, radius: float, depth: float, power: float = 4.0) -> np.ndarray:
    """Depth (m) of a flat-bottomed pit at distance ``r`` from its middle, 0 at the rim and beyond."""
    return depth * np.clip(1 - (r / radius) ** power, 0.0, None)


def circle(centre, radius: float, z: float, n: int = 24) -> list:
    a = 2 * np.pi * np.arange(n) / n
    return [[float(centre[0] + radius * np.cos(t)), float(centre[1] + radius * np.sin(t)), float(z)] for t in a]
