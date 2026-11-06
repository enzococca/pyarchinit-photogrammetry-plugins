"""Electrical resistivity tomography with pyGIMLi: what the worker does in the Python that has it.

This file imports only numpy and, inside its functions, pyGIMLi: it runs in the worker process
(:mod:`.ert_worker`), loaded there by its path, never in the program's process (pyGIMLi's core
carries its own OpenMP runtime, which must not meet the program's).

The 2-D frame of a profile is pyGIMLi's: x = the distance along the line (metres), y = the
elevation of the electrodes (metres, up). The sensor indices of the readings count from 0, -1 for
an electrode at infinity (a pole).

* :func:`read` - a file of a resistivity meter through pyGIMLi's own readers (the unified data
  format, RES2DINV, ABEM and Syscal ASCII exports): the electrodes and the readings as arrays;
* :func:`invert` - the inversion of a profile (mesh with the topography, error model, smoothness
  or blocky constraints, robust data); the parameter mesh, the resistivity of its cells, the
  coverage and, on request, the depth-of-investigation index of Oldenburg and Li (1999) - two more
  inversions towards a reference model 10 times lower and higher, with damping towards it;
* :func:`simulate` - synthetic readings of a model (layers and rectangular blocks under the
  electrodes) with pyGIMLi's forward modelling, with noise: the tests and the checks.
"""
from __future__ import annotations

import importlib
import time

import numpy as np

TOKENS = ("rhoa", "r", "u", "i", "err", "k", "ip")


def versions() -> dict:
    out = {"numpy": np.__version__}
    try:
        import importlib.metadata as md

        for name in ("pygimli", "pgcore"):
            try:
                out[name] = md.version(name)
            except Exception:  # noqa: BLE001
                out[name] = ""
    except Exception:  # noqa: BLE001
        pass
    return out


def _gimli():
    """pyGIMLi with its cache switched off (it would write the results of the forward operator into the
    user's folder)."""
    import pygimli as pg

    try:
        importlib.import_module("pygimli.utils.cache").noCache(True)
    except Exception:  # noqa: BLE001
        pass
    try:
        pg.setVerbose(False)
    except Exception:  # noqa: BLE001
        pass
    return pg


# ------------------------------------------------------------------ the data container
def container(sensors, a, b, m, n, values: dict):
    """A ``DataContainerERT`` from arrays: ``sensors`` (N, 2) = distance along the line and
    elevation; ``a b m n`` 0-based (-1: at infinity); ``values`` token -> array (NaN = missing,
    the token is left out when every value misses)."""
    pg = _gimli()
    data = pg.DataContainerERT()
    for x, z in np.asarray(sensors, float)[:, :2]:
        data.createSensor([float(x), float(z)])
    count = len(a)
    data.resize(count)
    for name, idx in (("a", a), ("b", b), ("m", m), ("n", n)):
        data.set(name, np.asarray(idx, float))
    for name, vals in values.items():
        v = np.asarray(vals, float)
        if v.shape == (count,) and np.isfinite(v).all():
            data.set(name, v)
    data.set("valid", np.ones(count))
    return data


def _arrays(data) -> dict:
    """The electrodes and the readings of a container: ``sensors_xyz`` the positions as the file gave
    them (the plugin tells a 2-D profile from world coordinates), ``sensors`` the 2-D frame (x,
    elevation), the readings, ``valid`` when the container has the flag."""
    pos = np.array([[p[0], p[1], p[2]] for p in data.sensors()], float).reshape(-1, 3)
    out = {"sensors_xyz": pos.copy(), "sensors": pos[:, :2] if np.allclose(pos[:, 2], 0) else pos[:, [0, 2]]}
    for name in ("a", "b", "m", "n"):
        out[name] = np.asarray(data[name], np.int64)
    try:
        if data.haveData("valid"):
            out["valid"] = np.asarray(data["valid"], float)
    except Exception:  # noqa: BLE001
        pass
    present = []
    for name in TOKENS:
        try:
            if data.haveData(name):
                v = np.asarray(data[name], float)
                if np.any(v != 0):
                    out[name] = v
                    present.append(name)
        except Exception:  # noqa: BLE001
            pass
    out["present"] = present
    return out


def read(path: str) -> dict:
    """A file of a resistivity meter through pyGIMLi's readers (``pygimli.physics.ert.load``)."""
    _gimli()
    from pygimli.physics import ert

    data = ert.load(str(path))
    if data is None or data.size() == 0:
        raise ValueError(f"pyGIMLi read no readings from {path}")
    return _arrays(data)


# ------------------------------------------------------------------ geometric factors
def flat_k(sensors, a, b, m, n) -> np.ndarray:
    """The geometric factors of a flat half-space with the electrodes on a straight line at their
    distances (an electrode at infinity contributes nothing)."""
    x = np.asarray(sensors, float)[:, 0]

    def inv_dist(i, j):
        out = np.zeros(len(i))
        ok = (i >= 0) & (j >= 0)
        d = np.abs(x[i[ok]] - x[j[ok]])
        out[ok] = 1.0 / np.maximum(d, 1e-9)
        return out

    a, b, m, n = (np.asarray(v, np.int64) for v in (a, b, m, n))
    g = inv_dist(a, m) - inv_dist(b, m) - inv_dist(a, n) + inv_dist(b, n)
    with np.errstate(divide="ignore"):
        return np.where(np.abs(g) > 1e-12, 2 * np.pi / g, np.nan)


def _topographic(sensors) -> bool:
    z = np.asarray(sensors, float)[:, 1]
    x = np.asarray(sensors, float)[:, 0]
    span = max(float(x.max() - x.min()), 1e-9)
    return float(z.max() - z.min()) > 0.005 * span


# ------------------------------------------------------------------ inversion
def invert(arrays: dict, params: dict, progress=None) -> dict:
    """Invert the readings of ``arrays`` (``sensors``, ``a b m n`` and ``rhoa`` and/or ``r``,
    ``err``, ``u``, ``i``, ``valid``) with ``params``: ``lam`` (20), ``rel_error`` (0.03),
    ``abs_error_u`` (volts, None), ``use_file_errors`` (True), ``robust`` (False), ``blocky``
    (False), ``z_weight`` (1.0), ``max_iter`` (20), ``para_depth`` (None: pyGIMLi's), ``para_dx``
    (0.33 of the spacing), ``para_max_cell`` (None), ``quality`` (33.0), ``doi`` ("coverage" or
    "oldenburg"), ``topography_k`` (True: the geometric factors computed numerically on the
    topography). Returns the arrays of the result."""
    progress = progress or (lambda f, p, **k: None)
    _gimli()
    from pygimli.physics import ert

    t0 = time.monotonic()
    sensors = np.asarray(arrays["sensors"], float)[:, :2]
    a, b, m, n = (np.asarray(arrays[k], np.int64) for k in ("a", "b", "m", "n"))
    count = len(a)
    keep = np.asarray(arrays.get("valid", np.ones(count)), bool)
    rhoa = np.asarray(arrays.get("rhoa", np.full(count, np.nan)), float)
    r = np.asarray(arrays.get("r", np.full(count, np.nan)), float)
    file_err = np.asarray(arrays.get("err", np.full(count, np.nan)), float)
    cur = np.asarray(arrays.get("i", np.full(count, np.nan)), float)
    volt = np.asarray(arrays.get("u", np.full(count, np.nan)), float)
    progress(0.02, "data")
    topo = _topographic(sensors) and bool(params.get("topography_k", True))
    k0 = flat_k(sensors, a, b, m, n)
    if not np.isfinite(r).any():  # the resistances from the apparent resistivities (the factors of the instrument: flat)
        r = rhoa / k0
    elif not np.isfinite(rhoa).any():
        rhoa = r * k0
    keep &= np.isfinite(r) & np.isfinite(k0) & (r != 0)
    idx = np.flatnonzero(keep)
    if idx.size < 4:
        raise ValueError(f"only {idx.size} usable readings: an inversion needs more")
    values = {"r": r[idx]}
    if np.isfinite(cur[idx]).all():
        values["i"] = cur[idx]
    if np.isfinite(volt[idx]).all():
        values["u"] = volt[idx]
    data = container(sensors, a[idx], b[idx], m[idx], n[idx], values)
    if topo:
        progress(0.05, "factors")
        k = np.asarray(ert.createGeometricFactors(data, numerical=True, verbose=False), float)
    else:
        k = k0[idx]
    data.set("k", k)
    data.set("rhoa", r[idx] * k)
    bad = ~(np.asarray(data["rhoa"]) > 0)  # a reading of the wrong sign for this geometry
    if bad.any():
        data.set("valid", (~bad).astype(float))
        data.removeInvalid()
        idx = idx[~bad]
    if data.size() < 4:
        raise ValueError("fewer than 4 readings with a positive apparent resistivity")
    rel = float(params.get("rel_error", 0.03))
    abs_u = params.get("abs_error_u")
    if abs_u:
        err = np.asarray(ert.estimateError(data, relativeError=rel, absoluteUError=float(abs_u)), float)
    else:
        err = np.full(data.size(), rel)
    fe = file_err[idx]
    if params.get("use_file_errors", True):  # the error of the file, where a reading has one and it is larger
        has = np.isfinite(fe) & (fe > 0)
        err = np.where(has, np.maximum(err, np.where(has, fe, 0.0)), err)
    data.set("err", err)
    progress(0.08, "mesh")
    spacing = float(np.median(np.diff(np.sort(sensors[:, 0])))) if len(sensors) > 1 else 1.0
    mesh_kw = {"quality": float(params.get("quality", 33.0)),
               "paraDX": float(params.get("para_dx", 0.33))}
    if params.get("para_depth"):
        mesh_kw["paraDepth"] = float(params["para_depth"])
    if params.get("para_max_cell"):
        mesh_kw["paraMaxCellSize"] = float(params["para_max_cell"])
    inv_kw = {"lam": float(params.get("lam", 20.0)), "robustData": bool(params.get("robust", False)),
              "blockyModel": bool(params.get("blocky", False)), "zWeight": float(params.get("z_weight", 1.0)),
              "maxIter": int(params.get("max_iter", 20)), "verbose": False}
    mgr = ert.ERTManager(data, verbose=False)
    mesh = mgr.createMesh(data, **mesh_kw)
    progress(0.12, "invert")
    model = np.asarray(mgr.invert(mesh=mesh, **inv_kw), float)
    para = mgr.paraDomain
    nodes = np.array([[p.x(), p.y()] for p in para.nodes()], float)
    cells = _triangles(para)
    markers = np.asarray(para.cellMarkers(), np.int64)
    cover = np.asarray(mgr.coverage(), float)
    response = np.asarray(mgr.inv.response, float)
    out = {"nodes": nodes, "cells": cells[0], "cell_of": cells[1], "model": model, "coverage": cover,
           "response": response, "rhoa": np.asarray(data["rhoa"], float), "err": np.asarray(data["err"], float),
           "k": np.asarray(data["k"], float), "used": idx.astype(np.int64), "markers": markers,
           "chi2_history": _history(mgr)}
    fit = {"chi2": float(mgr.inv.chi2()), "rrms": float(mgr.inv.relrms()), "absrms": float(mgr.inv.absrms()),
           "iterations": int(_iterations(mgr)), "lam": float(inv_kw["lam"]), "readings": int(data.size()),
           "cells": int(para.cellCount()), "spacing": spacing, "topography_k": bool(topo)}
    if params.get("doi") == "oldenburg":
        progress(0.6, "doi")
        med = float(np.exp(np.median(np.log(model))))
        runs = []
        for k_ref, frac in ((0.1, 0.65), (10.0, 0.85)):
            progress(frac, "doi")
            mgr2 = ert.ERTManager(data, verbose=False)
            # with smoothness constraints alone a constant reference model leaves no trace (its roughness
            # is zero): the two runs add damping towards it (pyGIMLi's mixed constraints, cType 10)
            m2 = np.asarray(mgr2.invert(mesh=mesh, startModel=med * k_ref, isReference=True, cType=10,
                                        **{**inv_kw, "maxIter": min(int(inv_kw["maxIter"]), 10)}), float)
            runs.append(np.log(m2))
        doi = np.abs(runs[0] - runs[1]) / abs(np.log(0.1) - np.log(10.0))
        out["doi"] = doi
        fit["doi_reference"] = med
    fit["seconds"] = round(time.monotonic() - t0, 3)
    out["fit"] = fit
    progress(0.98, "done")
    return out


def _history(mgr) -> np.ndarray:
    """The chi2 of every iteration (a list or an array, whatever pyGIMLi gives)."""
    h = getattr(mgr.inv, "chi2History", None)
    try:
        return np.asarray(h if h is not None else [], float).ravel()
    except (TypeError, ValueError):
        return np.zeros(0)


def _iterations(mgr) -> int:
    for get in (lambda: mgr.inv.inv.iter(), lambda: mgr.inv.iter, lambda: len(mgr.inv.chi2History) - 1):
        try:
            v = get()
            v = v() if callable(v) else v
            return int(v)
        except Exception:  # noqa: BLE001
            continue
    return 0


def _triangles(mesh) -> tuple[np.ndarray, np.ndarray]:
    """The cells of a 2-D mesh as triangles (a quadrangle split in two) and the cell of each."""
    tris, owner = [], []
    for c in mesh.cells():
        ids = [c.node(i).id() for i in range(c.nodeCount())]
        if len(ids) == 3:
            tris.append(ids)
            owner.append(c.id())
        elif len(ids) >= 4:
            tris.append([ids[0], ids[1], ids[2]])
            tris.append([ids[0], ids[2], ids[3]])
            owner += [c.id(), c.id()]
    return np.asarray(tris, np.int64).reshape(-1, 3), np.asarray(owner, np.int64)


# ------------------------------------------------------------------ forward modelling
def scheme_readings(count: int, spacing: float, scheme: str):
    """The quadrupoles of a standard array on ``count`` electrodes (pyGIMLi's ``createData``)."""
    _gimli()
    from pygimli.physics import ert

    data = ert.createData(elecs=np.arange(count) * float(spacing), schemeName=scheme)
    return (np.asarray(data["a"], np.int64), np.asarray(data["b"], np.int64), np.asarray(data["m"], np.int64),
            np.asarray(data["n"], np.int64))


def simulate(sensors, model: dict, scheme: str = "dd", noise: float = 0.02, noise_u: float = 1e-6,
             seed: int = 1337, progress=None) -> dict:
    """Synthetic readings on the electrodes ``sensors`` (N, 2: distance, elevation) of ``model``:
    ``{"background": rho, "layers": [[depth below the surface, rho], ...], "blocks": [[x0, x1, depth0,
    depth1, rho], ...]}`` (depths below the ground under each point, metres). ``scheme``: "dd",
    "wa" (Wenner), "slm" (Schlumberger), "pd" (pole-dipole)... Relative ``noise`` and absolute
    ``noise_u`` (volts) added."""
    progress = progress or (lambda f, p, **k: None)
    _gimli()
    import pygimli.meshtools as mt
    from pygimli.physics import ert

    sensors = np.asarray(sensors, float)[:, :2]
    order = np.argsort(sensors[:, 0])
    sensors = sensors[order]
    spacing = float(np.median(np.diff(sensors[:, 0])))
    progress(0.05, "mesh")
    scheme_data = ert.createData(elecs=sensors[:, 0], schemeName=scheme)
    for i, (x, z) in enumerate(sensors):
        scheme_data.setSensorPosition(i, [float(x), float(z)])
    span = float(sensors[-1, 0] - sensors[0, 0])
    plc = mt.createParaMeshPLC(scheme_data, paraDX=0.2, paraMaxCellSize=max(spacing * spacing * 0.1, 0.01),
                               paraDepth=0.5 * span)
    mesh = mt.createMesh(plc, quality=33.0)
    centres = np.array([[c.center().x(), c.center().y()] for c in mesh.cells()], float)
    surface = np.interp(centres[:, 0], sensors[:, 0], sensors[:, 1])
    depth = surface - centres[:, 1]
    res = np.full(len(centres), float(model.get("background", 100.0)))
    for top, rho in sorted(model.get("layers") or [], key=lambda v: v[0]):
        res[depth >= float(top)] = float(rho)
    for x0, x1, d0, d1, rho in model.get("blocks") or []:
        inside = (centres[:, 0] >= x0) & (centres[:, 0] <= x1) & (depth >= d0) & (depth <= d1)
        res[inside] = float(rho)
    progress(0.3, "forward")
    data = ert.simulate(mesh, scheme=scheme_data, res=res, noiseLevel=float(noise), noiseAbs=float(noise_u),
                        seed=int(seed), verbose=False)
    data.remove(data["rhoa"] <= 0)
    out = _arrays(data)
    out["sensors"] = sensors
    progress(0.98, "done")
    return out
