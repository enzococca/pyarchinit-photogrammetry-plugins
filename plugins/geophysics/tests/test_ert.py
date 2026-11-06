"""Electrical resistivity tomography in the geophysics plugin, without pyGIMLi: the readers, the
placement of the line, the filters, the inversion through a worker that speaks the protocol with a
numpy stand-in (``fake_ert_worker.py``), the section, its files and the 3D scene, the command line.
The tests with the real pyGIMLi are in ``test_ert_gimli.py``."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

import pyarchinit_photogrammetry as ppg
from ppg_geophysics import ert, ert_environment, ert_ops, ert_runner, ert_section

HERE = Path(__file__).resolve().parent
FAKE = HERE / "fake_ert_worker.py"
CRS = "EPSG:32633"


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(ert_ops, "WORKER", FAKE)
    monkeypatch.setattr(ert_ops, "WORKER_PYTHON", Path(sys.executable))
    monkeypatch.delenv("PPG_ERT_FAKE", raising=False)
    return FAKE


def dd_readings(count: int = 24, nmax: int = 6):
    """Dipole-dipole quadrupoles (0-based) and an apparent resistivity growing with depth, a
    resistive patch in the middle."""
    a, b, m, n = [], [], [], []
    for sep in range(1, nmax + 1):
        for i in range(count - 2 - sep):
            a.append(i), b.append(i + 1), m.append(i + 1 + sep), n.append(i + 2 + sep)
    a, b, m, n = (np.array(v) for v in (a, b, m, n))
    mid = (a + n) / 2.0
    depth = 0.17 * (n - a)
    rhoa = 50.0 * (1 + depth / 2.0) * (1 + 3.0 * np.exp(-((mid - count / 2) ** 2) / 8.0 - (depth - 1.0) ** 2))
    return a, b, m, n, rhoa


def write_csv(path: Path, positions: bool = False, err: bool = True, spacing: float = 2.0):
    a, b, m, n, rhoa = dd_readings()
    lines = ["A;B;M;N;Rho;Dev%" if err else "c1,c2,p1,p2,rhoa"]
    for i in range(len(a)):
        if positions:
            q = [f"{v * spacing:.1f}".replace(".", ",") for v in (a[i], b[i], m[i], n[i])]
        else:
            q = [str(v + 1) for v in (a[i], b[i], m[i], n[i])]
        if err:
            lines.append(";".join(q + [f"{rhoa[i]:.3f}".replace(".", ","), "1,5"]))
        else:
            lines.append(",".join(q + [f"{rhoa[i]:.3f}"]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_ohm(path: Path, spacing: float = 1.0, slope: float = 0.05):
    a, b, m, n, rhoa = dd_readings()
    out = ["24", "# x z"] + [f"{k * spacing:.2f} {slope * k * spacing:.3f}" for k in range(24)]
    out += [str(len(a)), "# a b m n rhoa err"]
    out += [f"{a[i] + 1} {b[i] + 1} {m[i] + 1} {n[i] + 1} {rhoa[i]:.4f} 0.02" for i in range(len(a))]
    out += ["0"]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path


def _project(tmp_path, dem: bool = True, crs: str = CRS):
    import rasterio
    from rasterio.transform import from_origin

    doc = ppg.Document()
    ch = doc.add_chunk("Saggio")
    doc.save(tmp_path / "ert.ppg")
    ch.crs = crs
    if dem:
        x0, y0, cell = 500000.0, 4500100.0, 0.5
        h, w = 200, 200
        cols, rows = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        z = (100.0 + 0.02 * cols * cell).astype(np.float32)  # rising to the east, 2 %
        p = ch.data_dir / "dem.tif"
        p.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(p, "w", driver="GTiff", width=w, height=h, count=1, dtype="float32", crs=crs,
                           transform=from_origin(x0, y0, cell, cell), nodata=-32767) as dst:
            dst.write(z, 1)
        ch._set_product("dem", p)
    return doc, ch


# ------------------------------------------------------------------ readers
def test_tables_of_readings_numbers_and_positions(tmp_path):
    p = ert.read_any(write_csv(tmp_path / "num.csv"), spacing=2.0)
    a, b, m, n, rhoa = dd_readings()
    assert p.electrodes == 24 and p.count == len(a) and p.numbered
    assert np.array_equal(p.a, a) and np.array_equal(p.n, n)
    assert np.allclose(p.file_u, np.arange(24) * 2.0)
    assert np.allclose(p.value("rhoa"), rhoa, rtol=1e-3) and np.allclose(p.value("err"), 0.015)
    assert ert.array_kind(p) == "dipole-dipole"
    q = ert.read_any(write_csv(tmp_path / "pos.csv", positions=True), spacing=None)
    assert not q.numbered and q.electrodes == 24 and np.allclose(q.file_u, np.arange(24) * 2.0)
    assert np.array_equal(q.a, a) and np.allclose(q.value("rhoa"), rhoa, rtol=1e-3)
    # a table pyGIMLi has to read: its message says what is missing
    bad = tmp_path / "other.txt"
    bad.write_text("foo bar\n1 2\n")
    with pytest.raises(ert.NeedsGimli):
        ert.read_any(bad)


def test_unified_format_resistance_and_poles(tmp_path):
    p = ert.read_any(write_ohm(tmp_path / "line.ohm"))
    assert p.fmt == "unified" and p.electrodes == 24 and np.allclose(p.file_z, 0.05 * np.arange(24))
    assert ert.array_kind(p) == "dipole-dipole" and np.allclose(p.value("err"), 0.02)
    # pole-dipole as voltage and current: rhoa from the flat geometric factors, b = -1
    lines = ["a,b,m,n,u,i", "1,,3,4,0.05,0.1", "2,,4,5,0.04,0.1", "3,,5,6,0.03,0.1", "4,,6,7,0.025,0.1"]
    f = tmp_path / "pd.csv"
    f.write_text("\n".join(lines) + "\n")
    q = ert.read_any(f, spacing=1.0)
    assert (q.b == -1).all() and ert.array_kind(q) == "pole-dipole"
    k = ert.flat_k(q.file_u, q.a, q.b, q.m, q.n)
    assert np.allclose(k, 2 * np.pi / (1 / 2 - 1 / 3))
    assert np.allclose(q.apparent(), np.array([0.5, 0.4, 0.3, 0.25]) * k)


def test_electrodes_table(tmp_path):
    csv = write_csv(tmp_path / "num.csv")
    el = tmp_path / "el.csv"
    el.write_text("n,x,y,z\n" + "\n".join(f"{k + 1},{500010 + 1.5 * k},{4500050 + 0.5 * k},{100 + 0.1 * k}"
                                          for k in range(24)) + "\n")
    p = ert.read_any(csv, electrodes=el)
    assert p.world and np.allclose(p.file_z, 100 + 0.1 * np.arange(24))
    assert np.allclose(np.diff(p.file_u), np.hypot(1.5, 0.5))


# ------------------------------------------------------------------ the project
def test_profile_in_the_project_placed_on_the_dem(tmp_path):
    doc, ch = _project(tmp_path)
    e = ert_ops.import_profile(ch, write_csv(tmp_path / "num.csv"), name="P1", spacing=2.0)
    assert e["name"] == "P1" and e["readings"] == len(dd_readings()[0]) and e["valid"] == e["readings"]
    line = [[500010.0, 4500050.0], [500090.0, 4500050.0]]
    e = ert.place(ch, e["id"], line, first=5.0)
    prof = ert.load(ch, e["id"])
    assert np.allclose(prof.elec["elec_u"], 5.0 + 2.0 * np.arange(24))
    assert np.allclose(prof.elec["elec_x"], 500015.0 + 2.0 * np.arange(24)) and np.allclose(prof.elec["elec_y"], 4500050.0)
    # the DEM rises 2 % to the east from 100 m at x 500000
    assert np.allclose(prof.elec["elec_z"], 100.0 + 0.02 * (prof.elec["elec_x"] - 500000.0), atol=0.02)
    assert e["placement"]["heights"] == "dem" and e["placement"]["length"] == pytest.approx(46.0)
    # flat at a height, another spacing: the electrodes follow
    e = ert.place(ch, e["id"], line, spacing=1.0, heights="flat", z0=99.0)
    prof = ert.load(ch, e["id"])
    assert np.allclose(prof.elec["elec_u"], np.arange(24)) and np.allclose(prof.elec["elec_z"], 99.0)
    # a bent line: the electrodes go round the corner
    bent = [[500010.0, 4500050.0], [500020.0, 4500050.0], [500020.0, 4500080.0]]
    ert.place(ch, e["id"], bent, spacing=1.0)
    prof = ert.load(ch, e["id"])
    assert np.allclose(prof.elec["elec_x"][15], 500020.0) and np.allclose(prof.elec["elec_y"][15], 4500055.0)
    doc.save()
    back = ppg.Document.open(tmp_path / "ert.ppg")
    assert ert.entry_of(back.chunk, "P1")["placement"]["method"] == "points"


def test_filters(tmp_path):
    doc, ch = _project(tmp_path)
    e = ert_ops.import_profile(ch, write_csv(tmp_path / "num.csv"), spacing=1.0)
    prof = ert.load(ch, e["id"])
    prof.values["rhoa"][5] = -3.0  # a reading of the wrong sign
    prof.values["rhoa"][40] *= 30.0  # a spike
    prof.values["err"][7] = 0.2  # a noisy reading
    ert.save(ch, prof)
    e = ert.set_filters(ch, e["id"], rules={"max_error": 10, "spikes": 6})
    c = e["filters"]["counts"]
    assert c["invalid"] == 1 and c["max_error"] == 1 and c["spikes"] >= 1
    valid = ert.valid_mask(ert.load(ch, e["id"]), e["filters"])
    assert not valid[5] and not valid[7] and not valid[40] and valid.sum() >= e["readings"] - 5
    e = ert.set_filters(ch, e["id"], remove_readings=[0, 1])
    assert e["filters"]["manual"] == [0, 1] and e["valid"] == int(valid.sum()) - 2
    e = ert.set_filters(ch, e["id"], restore=[1])
    assert e["filters"]["manual"] == [0]
    assert "picchi" in ert.filter_text(e)


def test_reciprocals(tmp_path):
    lines = ["a,b,m,n,r", "1,2,3,4,10", "3,4,1,2,10.2", "2,3,4,5,8", "4,5,2,3,12"]
    f = tmp_path / "rec.csv"
    f.write_text("\n".join(lines) + "\n")
    p = ert.read_any(f)
    rec = ert.reciprocal_errors(p)
    assert rec[0] == pytest.approx(0.2 / 10.1) and rec[2] == pytest.approx(4 / 10)
    r = ert.reasons(p, {"rules": {"reciprocal": 5}})
    assert r["reciprocal"].tolist() == [False, False, True, True]


# ------------------------------------------------------------------ the worker
def test_the_inversion_through_the_worker(tmp_path, fake):
    doc, ch = _project(tmp_path)
    e = ert_ops.import_profile(ch, write_ohm(tmp_path / "line.ohm"), name="P1")
    ert.place(ch, e["id"], [[500010.0, 4500050.0], [500040.0, 4500050.0]])
    seen = []
    e = ert_ops.invert_profile(ch, e["id"], {"lam": 10}, progress=lambda f, m="": seen.append((f, m)))
    inv = e["inversion"]
    assert inv["fit"]["chi2"] == 1.0 and inv["params"]["lam"] == 10 and inv["versions"]["pygimli"] == "0.0-fake"
    assert not inv["stale"] and any("pyGIMLi" in m for _f, m in seen)
    tif = ch.data_dir / inv["files"]["section"]
    import rasterio

    with rasterio.open(tif) as src:
        v = src.read(1)
        tr = src.transform
        assert src.crs is None and src.tags()["ppg_frame"] == "ert-section"
    # the frame: u along the line, v = elevation; the top of the section follows the electrodes
    prof = ert.load(ch, e["id"])
    assert tr.c <= prof.elec["elec_u"].min() and tr.f >= prof.elec["elec_z"].max() - 1e-6
    assert np.isfinite(v).any() and np.isnan(v).any()  # blanked below the coverage and above the ground
    # the resistive patch of the readings in the middle of the section
    ok = np.isfinite(v)
    cols = np.nonzero(ok)[1]
    centre = (tr.c + (cols + 0.5) * tr.a)
    assert np.nanmax(v) > 2.5 * np.nanmin(v)
    hot = centre[v[ok] > np.percentile(v[ok], 95)]
    assert abs(np.median(hot) - prof.elec["elec_u"].mean()) < 4.0
    # the record and the paradata
    para = json.loads((ert.folder(ch, e["id"]) / "inversion.json").read_text())
    assert para["inversion"]["fit"]["chi2"] == 1.0 and para["placement"]["heights"] == "dem"
    # a new placement or filter makes it stale
    e = ert.set_filters(ch, e["id"], rules={"max_rhoa": 1000})
    assert e["inversion"]["stale"]
    assert "da rifare" in ert_ops.summary_line(e)


def test_worker_failures_leave_nothing(tmp_path, fake, monkeypatch):
    doc, ch = _project(tmp_path)
    e = ert_ops.import_profile(ch, write_ohm(tmp_path / "line.ohm"))
    ert.place(ch, e["id"], [[500010.0, 4500050.0], [500040.0, 4500050.0]])
    for mode, text in (("fail", "did not converge"), ("crash", "inaspettatamente")):
        monkeypatch.setenv("PPG_ERT_FAKE", mode)
        with pytest.raises(ert_runner.WorkerError) as info:
            ert_ops.invert_profile(ch, e["id"])
        assert text in str(info.value)
        assert not (ert.folder(ch, e["id"]) / "inversion.npz").exists()
        assert not list((ert.folder(ch, e["id"]) / "work").glob("*.npz"))
        assert not ert.entry_of(ch, e["id"]).get("inversion")


def test_cancelling_stops_the_worker(tmp_path, fake, monkeypatch):
    import threading
    import time

    from pyarchinit_photogrammetry.core.progress import Cancelled, Progress

    doc, ch = _project(tmp_path)
    e = ert_ops.import_profile(ch, write_ohm(tmp_path / "line.ohm"))
    monkeypatch.setenv("PPG_ERT_FAKE", "sleep")
    prog = Progress()
    threading.Timer(1.5, prog.cancel).start()
    t0 = time.monotonic()
    with pytest.raises(Cancelled):
        ert_ops.invert_profile(ch, e["id"], progress=prog)
    assert time.monotonic() - t0 < 15
    assert not (ert.folder(ch, e["id"]) / "inversion.npz").exists()


def test_reading_through_the_worker(tmp_path, fake):
    doc, ch = _project(tmp_path)
    f = write_ohm(tmp_path / "line.ohm")
    other = tmp_path / "line.bin"  # not a format of the plugin: pyGIMLi (here the fake) reads it
    other.write_text("# a file of an instrument\n" + f.read_text().replace("24\n# x z", "24 electrodes\n# x z", 1))
    other.write_text(f.read_text())
    e = ert_ops.import_profile(ch, other, name="via worker")
    # the fake reads the unified format too: what matters is that it went through the worker when asked
    assert e["readings"] == len(dd_readings()[0])
    bad = tmp_path / "bad.bin"
    bad.write_text("not a format\n")
    with pytest.raises(Exception):
        ert_ops.import_profile(ch, bad)


# ------------------------------------------------------------------ section, files, 3D
def test_section_files_exports_and_scene(tmp_path, fake):
    doc, ch = _project(tmp_path)
    e = ert_ops.import_profile(ch, write_ohm(tmp_path / "line.ohm"), name="P 1")
    ert.place(ch, e["id"], [[500010.0, 4500050.0], [500040.0, 4500050.0]])
    ert_ops.invert_profile(ch, e["id"])
    out = tmp_path / "out"
    files = ert_ops.export_all(ch, e["id"], out)
    names = {p.name for p in files}
    assert {"P_1_sezione.tif", "P_1_celle.csv", "P_1_sezione.xyz", "P_1_sezione.png", "P_1_sezione.svg",
            "P_1_pseudosezione.png", "P_1_linea.gpkg", "P_1_paradati.json", "P_1_letture.ohm"} <= names
    assert (out / "P_1_sezione.frame.json").exists()
    back = ert.read_any(out / "P_1_letture.ohm")  # the readings kept, read again
    assert back.count == e["readings"] and np.allclose(back.file_z, ert.load(ch, e["id"]).elec["elec_z"], atol=1e-3)
    rows = (out / "P_1_celle.csv").read_text().splitlines()
    assert rows[0].startswith("cell,u,v,x,y,z,rho_ohm_m") and len(rows) > 100
    x = np.array([float(r.split(",")[3]) for r in rows[1:]])
    assert (x > 500005).all() and (x < 500045).all()
    import sqlite3

    con = sqlite3.connect(out / "P_1_linea.gpkg")
    assert con.execute("select count(*) from elettrodi_ert").fetchone()[0] == 24
    assert con.execute("select nome from profili_ert").fetchone()[0] == "P 1"
    con.close()
    # another palette and blanking: the files written again
    inv = ert_section.write(ch, e["id"], blank={"method": "none"}, palette="viridis")
    assert inv["display"]["palette"] == "viridis" and inv["display"]["blank"]["method"] == "none"
    v, _o, _c = ert_section.read_section(ch, e["id"])
    assert np.isfinite(v).sum() > 0
    # the 3D scene: the section under the line, the DEM cut away on one side
    path, what = ert_section.scene(ch)
    assert path is not None and what["sections"] == ["P 1"] and what["terrain"]
    from pyarchinit_photogrammetry.io.mesh_ply import read_mesh

    m = read_mesh(path)
    world = ch.to_world(np.asarray(m.xyz, float))
    sec = ert_section.section_mesh(ch, e["id"])
    assert len(m.faces) > len(sec[1])
    # the section stands under the line: its vertices at y 4500050, below the ground
    sv = sec[0]
    assert np.allclose(sv[:, 1], 4500050.0) and sv[:, 2].max() <= 100.0 + 0.02 * 90 + 0.5
    # the ground kept only on the left of the line (north of a line going east)
    ground = world[len(sv):]
    assert (ground[:, 1] >= 4500050.0 - 1.0).all()


def test_units_and_cores_of_geoarchaeology_in_the_scene(tmp_path, fake):
    doc, ch = _project(tmp_path)
    e = ert_ops.import_profile(ch, write_ohm(tmp_path / "line.ohm"), name="P1")
    ert.place(ch, e["id"], [[500010.0, 4500050.0], [500040.0, 4500050.0]])
    ert_ops.invert_profile(ch, e["id"])
    ch.meta["geoarchaeology"] = {
        "units": [{"code": "1", "colour": "#aa5500"}, {"code": "2", "colour": "#3366cc"}],
        "boreholes": [{"id": "S1", "x": 500020.0, "y": 4500052.0, "z": 100.4,
                       "intervals": [{"from": 0, "to": 1.2, "unit": "1"}, {"from": 1.2, "to": 3.0, "unit": "2"}]},
                      {"id": "far", "x": 500090.0, "y": 4500090.0, "z": 101.0,
                       "intervals": [{"from": 0, "to": 1, "unit": "1"}]}]}
    path, what = ert_section.scene(ch, terrain="none")
    assert what["cores"] == 2 and what["terrain"] is None  # the far borehole is left out
    from pyarchinit_photogrammetry.io.mesh_ply import read_mesh

    m = read_mesh(path)
    assert any((c[:3] == [0xaa, 0x55, 0x00]).all() for c in np.asarray(m.colors)[-16:])


def test_colours_and_ticks():
    v = np.array([[10.0, 100.0], [np.nan, 1000.0]])
    rgba = ert_section.colours(v, 10, 1000, "rainbow")
    assert rgba[1, 0, 3] == 0 and rgba[0, 0, 3] == 255
    assert tuple(rgba[0, 0, :3]) == (0, 0, 128) and tuple(rgba[1, 1, :3]) == (120, 0, 90)
    assert ert_section.ticks(10, 1000) == [10, 20, 50, 100, 200, 500, 1000]
    lo, hi = ert_section.limits(np.array([1.0, 10, 100, 1000]))
    assert 1 <= lo < 2 and 700 < hi <= 1000


# ------------------------------------------------------------------ environment
def test_where_pygimli_runs(tmp_path, monkeypatch):
    assert ert_environment.unsupported("win32", "ARM64") is not None
    assert ert_environment.unsupported("linux", "aarch64") is not None
    assert ert_environment.unsupported("darwin", "arm64", "15.4") is not None
    assert ert_environment.unsupported("darwin", "arm64", "26.1") is None
    assert ert_environment.python_for("darwin", "x86_64") == "3.13"
    assert ert_environment.python_for("win32", "AMD64") == "3.12"
    assert ert_environment.python_for("linux", "x86_64") == "3.12"
    monkeypatch.setenv(ert_environment.ENV_VAR, str(tmp_path / "env"))
    monkeypatch.setattr(ert_environment, "local_available", lambda: False)
    monkeypatch.setattr(ert_environment, "unsupported", lambda *a, **k: None)
    assert ert_environment.where() == (None, "")
    assert "Installa pyGIMLi" in ert_environment.status_text()
    with pytest.raises(ert_environment.GimliMissing):
        ert_environment.worker_python()


def test_the_blas_of_the_macos_wheel_is_linked(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    site = tmp_path / "env" / "lib" / "python3.12" / "site-packages" / "pgcore" / ".dylibs"
    site.mkdir(parents=True)
    (site / "libopenblas.0.dylib").write_bytes(b"\0")
    if os.name == "nt":
        pytest.skip("symbolic links need rights on Windows")
    made = ert_environment.repair(tmp_path / "env")
    assert made == [site / "libblas.3.dylib"] and os.readlink(site / "libblas.3.dylib") == "libopenblas.0.dylib"
    assert ert_environment.repair(tmp_path / "env") == []  # once


def test_the_worker_has_its_own_home(tmp_path, monkeypatch):
    monkeypatch.setattr(ert_environment, "_DATA", tmp_path / "data")
    env = ert_environment.worker_environment()
    assert env["HOME"] == str(tmp_path / "data" / "ert-home") and env["MPLBACKEND"] == "Agg"
    assert (tmp_path / "data" / "ert-home" / "Library" / "Caches").is_dir()
    assert not any(k.startswith("VIRTUAL_ENV") for k in env)
