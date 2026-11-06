"""The tomography with the real pyGIMLi, in its worker: a two-layer ground (50 ohm m over 200 ohm m
from 3 m) with a buried resistive block (2000 ohm m, 1.5-3.5 m deep, in the middle of the line),
readings of pyGIMLi's forward modelling with 2 % noise, inverted as the plugin inverts them; the
block must come back at its place and depth, the layers with their resistivity.

Skipped without a Python that has pyGIMLi: ``PPG_TEST_GIMLI`` names it (the environment the plugin
installs, ``<plugin data>/ert-env/bin/python``), else this Python when pyGIMLi is installed here.
The plugins' CI does not install pyGIMLi: these tests run on the author's computer."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

import pyarchinit_photogrammetry as ppg
from ppg_geophysics import ert, ert_ops, ert_section


def _gimli_python() -> Path | None:
    given = os.environ.get("PPG_TEST_GIMLI")
    if given and Path(given).exists():
        return Path(given)
    if importlib.util.find_spec("pygimli") is not None:
        return Path(sys.executable)
    return None


PYTHON = _gimli_python()
pytestmark = pytest.mark.skipif(PYTHON is None, reason="pyGIMLi is not installed (PPG_TEST_GIMLI names its Python)")
BLOCK = (0.42 * 47, 0.58 * 47, 1.5, 3.5)  # u from, u to, depth from, depth to: the default model of simulate_profile


@pytest.fixture
def gimli(monkeypatch, tmp_path):
    from ppg_geophysics import ert_environment

    monkeypatch.setattr(ert_ops, "WORKER", Path(ert_ops.__file__).with_name("ert_worker.py"))
    monkeypatch.setattr(ert_ops, "WORKER_PYTHON", PYTHON)
    monkeypatch.setattr(ert_environment, "_DATA", tmp_path / "plugin-data")
    return PYTHON


def _recovered(ch, pid):
    """Where the section is most resistive against the layers (log10 minus the median of its depth
    across the line), and the median of the two layers away from the block."""
    v, (u0, vtop), cell = ert_section.read_section(ch, pid)
    prof = ert.load(ch, pid)
    h, w = v.shape
    uu = u0 + (np.arange(w) + 0.5) * cell
    vv = vtop - (np.arange(h) + 0.5) * cell
    surface = np.interp(uu, prof.positions(), prof.heights())
    depth = surface[None, :] - vv[:, None]
    lg = np.log10(v)
    rows = np.isfinite(lg).any(axis=1, keepdims=True)
    anomaly = lg - np.nanmedian(np.where(rows, lg, 0.0), axis=1, keepdims=True)
    anomaly[(depth < 0.5) | ~np.isfinite(anomaly)] = -np.inf
    r, c = np.unravel_index(np.argmax(anomaly), anomaly.shape)
    away = (np.abs(uu - 0.5 * (BLOCK[0] + BLOCK[1])) > 12)[None, :] & np.isfinite(v)
    top = np.nanmedian(v[away & (depth > 0.3) & (depth < 1.5)])
    deep = np.nanmedian(v[away & (depth > 5) & (depth < 7)])
    return uu[c], depth[r, c], top, deep, 10 ** np.nanmax(lg[(depth > BLOCK[2]) & (depth < BLOCK[3]) &
                                                              (np.abs(uu - 0.5 * (BLOCK[0] + BLOCK[1])) < 3)[None, :]])


@pytest.mark.parametrize("slope", [0.0, 0.08])
def test_a_buried_block_comes_back(tmp_path, gimli, slope):
    doc = ppg.Document()
    ch = doc.add_chunk("ERT")
    doc.save(tmp_path / "ert.ppg")
    e = ert_ops.simulate_profile(ch, "sintetico", count=48, spacing=1.0, scheme="dd", slope=slope, noise=0.02)
    assert e["readings"] > 900 and e["array"] == "dipole-dipole"
    e = ert.place(ch, e["id"], [[0.0, 0.0], [47.0, 0.0]], heights="file" if slope else "flat", z0=0.0,
                  along="plan")
    e = ert_ops.invert_profile(ch, e["id"], {"lam": 20})
    fit = e["inversion"]["fit"]
    assert fit["chi2"] < 3.0 and fit["rrms"] < 5.0
    assert fit["topography_k"] == bool(slope)
    u, depth, top, deep, block = _recovered(ch, e["id"])
    print(f"slope {slope}: block at u {u:.2f} m, depth {depth:.2f} m, max {block:.0f} ohm m; top {top:.1f}, "
          f"deep {deep:.1f} ohm m; chi2 {fit['chi2']:.2f}, rms {fit['rrms']:.2f} %, {fit['seconds']} s")
    assert BLOCK[0] - 1.0 < u < BLOCK[1] + 1.0
    assert BLOCK[2] - 1.0 < depth < BLOCK[3] + 1.0
    assert 35 < top < 70  # 50
    assert 100 < deep < 400  # 200, smoothed
    assert block > 3 * deep  # 2000, smoothed


def test_depth_of_investigation_of_oldenburg_and_li(tmp_path, gimli):
    doc = ppg.Document()
    ch = doc.add_chunk("ERT")
    doc.save(tmp_path / "ert.ppg")
    e = ert_ops.simulate_profile(ch, "sintetico", count=32, spacing=1.0, scheme="dd", noise=0.02)
    ert.place(ch, e["id"], [[0.0, 0.0], [31.0, 0.0]], heights="flat", z0=0.0)
    e = ert_ops.invert_profile(ch, e["id"], {"lam": 20, "doi": "oldenburg"})
    res = ert_section.load_result(ch, e["id"])
    doi = res["doi"]
    assert doi.shape == res["model"].shape and e["inversion"]["display"]["blank"]["method"] == "doi"
    # resolved near the surface, not at the bottom of the mesh
    nodes, cells = res["nodes"], res["cells"]
    cen = nodes[cells].mean(axis=1)
    depth, middle = -cen[:, 1], np.abs(cen[:, 0] - 15.5) < 8
    owner = res["cell_of"]
    shallow, bottom = np.median(doi[owner[middle & (depth < 2)]]), np.median(doi[owner[depth > depth.max() * 0.8]])
    print(f"DOI index: {shallow:.2f} in the first 2 m under the line, {bottom:.2f} at the bottom of the mesh")
    assert shallow < ert_section.DEFAULT_BLANK["doi_max"] < bottom


def test_a_pole_dipole_line(tmp_path, gimli):
    """The remote electrode (b = -1) through the inversion."""
    doc = ppg.Document()
    ch = doc.add_chunk("ERT")
    doc.save(tmp_path / "ert.ppg")
    e = ert_ops.simulate_profile(ch, "polo-dipolo", count=32, spacing=1.0, scheme="pd", noise=0.02)
    prof = ert.load(ch, e["id"])
    assert e["array"] == "pole-dipole" and (prof.b == -1).all()
    ert.place(ch, e["id"], [[0.0, 0.0], [31.0, 0.0]], heights="flat", z0=0.0)
    e = ert_ops.invert_profile(ch, e["id"], {"lam": 20})
    fit = e["inversion"]["fit"]
    print(f"pole-dipole: chi2 {fit['chi2']:.2f}, rms {fit['rrms']:.2f} %, {fit['readings']} readings")
    assert fit["chi2"] < 3.0 and fit["readings"] > 0.9 * e["readings"]


def test_a_file_saved_by_pygimli_is_inverted(tmp_path, gimli):
    """pyGIMLi's own unified file (2-D with y the elevation, zero columns): read by the plugin, inverted
    on its topography."""
    doc = ppg.Document()
    ch = doc.add_chunk("ERT")
    doc.save(tmp_path / "ert.ppg")
    e = ert_ops.import_profile(ch, Path(__file__).resolve().parent / "data" / "gimli_topo.dat")
    e = ert.place(ch, e["id"], [[0.0, 0.0], [11.0, 0.0]], heights="file")
    assert e["placement"]["along"] == "plan" and e["placement"]["z_range"] == pytest.approx([100.0, 100.88])
    e = ert_ops.invert_profile(ch, e["id"], {"lam": 20})
    fit = e["inversion"]["fit"]
    assert fit["readings"] == 45 and fit["topography_k"] and np.isfinite(fit["chi2"])
