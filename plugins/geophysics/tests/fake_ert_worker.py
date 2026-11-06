"""A worker that speaks the protocol of ``ppg_geophysics/ert_worker.py`` - the very code of it,
loaded from its file - with an "inversion" in numpy instead of pyGIMLi, for the tests where pyGIMLi
is not installed. It runs as the real one does, in a process of its own:

    python -u fake_ert_worker.py JOB.json

The inversion: a mesh of triangles under the electrodes (following their heights) down to a quarter
of the line, every cell the log-mean apparent resistivity of the readings nearest to it in the
pseudosection (depth = pseudo-depth), the coverage falling with depth; chi2 1.0, RMS 3 %.
``read`` reads the unified data format only (a ``.dat`` file named so); ``simulate`` gives a layered
earth by a crude rule. ``PPG_ERT_FAKE`` changes what it does: ``fail`` (pyGIMLi's kind of error),
``crash`` (the process dies half-way), ``sleep`` (a long step: for cancelling).
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
_spec = importlib.util.spec_from_file_location("ppg_geophysics_ert_worker_fake", HERE.parent / "ppg_geophysics" /
                                               "ert_worker.py")
worker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(worker)
MODE = os.environ.get("PPG_ERT_FAKE", "")


class Engine:
    @staticmethod
    def versions():
        return {"pygimli": "0.0-fake", "numpy": np.__version__}

    @staticmethod
    def read(path):
        lines = [ln.split("#")[0].split() for ln in Path(path).read_text().splitlines()]
        lines = [ln for ln in lines if ln]
        n = int(lines[0][0])
        sens = np.array([[float(v) for v in ln[:2]] for ln in lines[1:1 + n]])
        k = 1 + n
        m = int(lines[k][0])
        rows = np.array([[float(v) for v in ln] for ln in lines[k + 1:k + 1 + m]])
        idx = rows[:, :4].astype(np.int64) - 1
        return {"sensors": sens, "a": idx[:, 0], "b": idx[:, 1], "m": idx[:, 2], "n": idx[:, 3], "rhoa": rows[:, 4],
                "present": ["rhoa"]}

    @staticmethod
    def invert(arrays, params, progress):
        if MODE == "fail":
            raise RuntimeError("Inversion did not converge: chi2 rose for 3 iterations")
        sensors = np.asarray(arrays["sensors"], float)
        a, b, m, n = (np.asarray(arrays[k], np.int64) for k in ("a", "b", "m", "n"))
        valid = np.asarray(arrays.get("valid", np.ones(len(a))), bool)
        rhoa = np.asarray(arrays["rhoa"], float)
        x = sensors[:, 0]
        pts = np.stack([a, b, m, n], 1)
        pos = np.where(pts >= 0, x[np.clip(pts, 0, None)], np.nan)
        mid = np.nanmean(pos, 1)
        pd = 0.17 * (np.nanmax(pos, 1) - np.nanmin(pos, 1))
        ok = valid & np.isfinite(rhoa) & (rhoa > 0)
        progress(0.1, "mesh")
        span = x.max() - x.min()
        s = np.median(np.diff(np.sort(x)))
        us = np.linspace(x.min() - 2 * s, x.max() + 2 * s, 60)
        depth_max = 0.25 * span
        ds = np.linspace(0, depth_max, 16)
        top = np.interp(us, x, sensors[:, 1])
        U, D = np.meshgrid(us, ds)
        nodes = np.column_stack([U.ravel(), (top[None, :] - D).ravel()])
        w = len(us)
        idx = np.arange(nodes.shape[0]).reshape(len(ds), w)
        q = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], -1).reshape(-1, 4)
        cells = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
        cen_u = nodes[cells, 0].mean(1)
        cen_d = (np.interp(cen_u, x, sensors[:, 1]) - nodes[cells, 1].mean(1))
        progress(0.5, "invert")
        if MODE == "sleep":
            time.sleep(30)
        if MODE == "crash":
            sys.stdout.flush()
            if hasattr(signal, "SIGSEGV") and os.name != "nt":
                os.kill(os.getpid(), signal.SIGSEGV)
            os._exit(3)
        lr = np.log10(rhoa[ok])
        dist = np.hypot((cen_u[:, None] - mid[ok][None]) / s, (cen_d[:, None] - pd[ok][None]) / s)
        near = np.argsort(dist, axis=1)[:, :6]
        model = 10 ** lr[near].mean(1)
        cover = 2.0 - 4.0 * cen_d / depth_max
        return {"nodes": nodes, "cells": cells, "cell_of": np.arange(len(cells)), "model": model, "coverage": cover,
                "response": rhoa[ok], "rhoa": rhoa[ok], "err": np.full(ok.sum(), 0.03), "k": np.ones(ok.sum()),
                "used": np.flatnonzero(ok), "markers": np.full(len(cells), 2),
                "fit": {"chi2": 1.0, "rrms": 3.0, "absrms": 1.0, "iterations": 4, "lam": float(params.get("lam", 20)),
                        "readings": int(ok.sum()), "cells": int(len(cells)), "spacing": float(s), "seconds": 0.01}}

    @staticmethod
    def simulate(sensors, model, scheme, noise, noise_u, seed, progress):
        sensors = np.asarray(sensors, float)
        k = len(sensors)
        a, b, m, n = [], [], [], []
        for sep in range(1, 7):
            for i in range(k - 2 - sep):
                a.append(i), b.append(i + 1), m.append(i + 1 + sep), n.append(i + 2 + sep)
        a, b, m, n = (np.array(v) for v in (a, b, m, n))
        x = sensors[:, 0]
        depth = 0.17 * (x[n] - x[a])
        bg = float(model.get("background", 100))
        layers = model.get("layers") or []
        rho = np.full(len(a), bg)
        for top, r in layers:
            rho = np.where(depth > top, np.sqrt(rho * r), rho)
        rng = np.random.default_rng(seed)
        rho = rho * (1 + noise * rng.standard_normal(len(rho)))
        return {"sensors": sensors, "a": a, "b": b, "m": m, "n": n, "rhoa": rho, "present": ["rhoa"]}


if __name__ == "__main__":
    sys.exit(worker.main(sys.argv[1:], engine=Engine()))
