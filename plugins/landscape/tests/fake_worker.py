"""A worker that speaks the protocol of ``ppg_landscape/worker.py`` - the very code of it, loaded
from its file - with a simulation in numpy instead of Landlab, for the tests of the plugin where
Landlab is not installed. It runs as the real one does, in a process of its own:

    python -u fake_worker.py JOB.json

The simulation: explicit linear diffusion (``diffusion.D``, closed edges) for ``years`` in steps of
one year, a snapshot at each year asked; the drainage area is the cell area everywhere.
``PPG_LANDSCAPE_FAKE`` changes what it does: ``fail`` (the simulation raises Landlab's kind of
error), ``crash`` (the process dies as on a segmentation fault, in the middle of the run),
``crash_save`` (it dies after writing the first file: the partial folder stays behind), ``sleep``
(every year takes a second: for cancelling), ``noise`` (Landlab's own lines among the messages).
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
_spec = importlib.util.spec_from_file_location("ppg_landscape_worker_fake", HERE.parent / "ppg_landscape" / "worker.py")
worker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(worker)
engine = worker.load_engine()
MODE = os.environ.get("PPG_LANDSCAPE_FAKE", "")


def _die() -> None:
    sys.stdout.flush()
    if hasattr(signal, "SIGSEGV") and os.name != "nt":
        os.kill(os.getpid(), signal.SIGSEGV)
    os._exit(3)


def simulate(job, elevation, valid, progress):
    years = float(job["years"])
    D = float((job.get("diffusion") or {}).get("D", 0.01))
    dx = float(job["cell"])
    ok = np.isfinite(elevation) if valid is None else (np.isfinite(elevation) & valid)
    z = np.where(ok, elevation, np.nan).astype(np.float64)
    start = z.copy()
    if MODE == "fail":
        raise FloatingPointError("the surface became NaN after 3 years: a shorter step (dt) is needed")
    if MODE == "noise":
        sys.stdout.write("Landlab says something unfinished")  # no newline
        sys.stdout.flush()
        print("\nanother line of Landlab", flush=True)
    progress(0.0, "grid")
    snapshots = {}
    wanted = {float(s) for s in job.get("snapshots") or ()}
    n = int(round(years))
    for year in range(1, n + 1):
        filled = np.where(ok, z, np.nanmean(z))
        pad = np.pad(filled, 1, mode="edge")
        lap = pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:] - 4 * filled
        z = np.where(ok, z + min(D / dx ** 2, 0.2) * lap, np.nan)
        if year in wanted:
            snapshots[float(year)] = z.copy()
        progress(0.02 + 0.93 * year / n, "run", year=float(year), years=years, step=year)
        if MODE == "sleep":
            time.sleep(1.0)
        if MODE == "crash" and year == max(1, n // 2):
            _die()
    progress(0.96, "area")
    area = np.where(ok, dx * dx, np.nan)
    balance = {"years": years, "cell": dx, "eroded_m3": float(-np.nansum(np.minimum(z - start, 0)) * dx * dx),
               "deposited_m3": float(np.nansum(np.maximum(z - start, 0)) * dx * dx), "router": "fake", "eroder": None}
    return engine.Result(start, z, area, snapshots, balance)


if MODE == "crash_save":

    def write_results(result, folder):
        worker.np_save(folder / "initial.npy", result.initial)
        _die()

    worker.write_results = write_results


if __name__ == "__main__":
    sys.exit(worker.main(sys.argv[1:], simulate=simulate, versions=lambda: {"landlab": "0.0-fake", "python": "3"}))
