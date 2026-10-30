"""Synthetic DEMs of the tests and the plugin's modules loaded by their path (``engine`` and
``worker`` import nothing of the program: the engine's tests run in a Python with Landlab only).

* :func:`tell`: a cone 4 m high and 10 m in radius on a plane at 10 m, 100 x 100 cells of 0.5 m;
* :func:`notched_slope`: a slope of 15 % (the north part) above a nearly flat apron (1 %, the 15 m
  to the south) with a notch 0.3 m deep and 1.5 m wide cut down the middle of the slope.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parent.parent / "ppg_landscape"
N, CELL = 100, 0.5


def load(name: str):
    """``ppg_landscape/<name>.py`` as a module of its own (no package, no program)."""
    key = f"ppg_landscape_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, PACKAGE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


def landlab_python() -> Path | None:
    """A Python with Landlab for the tests of the real worker: ``PPG_TEST_LANDLAB``, else this Python
    when Landlab is installed here, else None."""
    given = os.environ.get("PPG_TEST_LANDLAB")
    if given:
        return Path(given)
    if importlib.util.find_spec("landlab") is not None:
        return Path(sys.executable)
    return None


def coordinates(n: int = N, cell: float = CELL):
    """East and north (m) of the cell centres in raster order (row 0 north)."""
    rows, cols = np.mgrid[0:n, 0:n]
    return (cols + 0.5) * cell, (n - rows - 0.5) * cell


def tell(n: int = N, cell: float = CELL, height: float = 4.0, radius: float = 10.0, base: float = 10.0):
    """The cone and its radius of every cell (m from the summit)."""
    east, north = coordinates(n, cell)
    centre = n * cell / 2
    r = np.hypot(east - centre, north - centre)
    return base + np.maximum(0.0, height * (1 - r / radius)), r


def notched_slope(n: int = N, cell: float = CELL, seed: int = 1):
    """The slope, the notch mask and the north coordinate of every cell."""
    east, north = coordinates(n, cell)
    z = 5.0 + np.where(north > 15, 0.15 + 0.15 * (north - 15), 0.01 * north)
    z = z + np.random.default_rng(seed).random(z.shape) * 1e-3
    cols = np.arange(n)[None, :].repeat(n, 0)
    notch = (np.abs(cols - n // 2) <= 1) & (north > 15)
    return z - 0.3 * notch, notch, north, cols
