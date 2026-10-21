"""The powerlines plugin on a synthetic survey: a sloping ground with trees, a building, six
poles, a line of three parallel conductors over three spans (one with a gap in the
reconstruction) and a lower line crossing it. Detection, catenaries, poles, clearance,
obstacles, classes, exports, the window layers, the command line, the plugin in the program."""
from __future__ import annotations

import contextlib
import json

import numpy as np
import pytest
from ppg_powerlines import detect, export

CRS = "EPSG:32637"
ORIGIN = np.array([300000.0, 3960000.0, 100.0])
HEIGHT = 10.0  # m: the poles of the main line
MAIN = [(20.0, 60.0), (80.0, 60.0), (140.0, 60.0), (190.0, 60.0)]  # x, y of its poles: spans of 60, 60 and 50 m
CROSS = [(100.0, 10.0), (100.0, 110.0)]  # the line crossing under it, 7 m high
OFFSETS = (-1.2, 0.0, 1.2)  # the three conductors of the main line, beside each other
TREE_BY_THE_LINE = (50.0, 64.5, 7.0, 2.0)  # x, y, height, radius: 1.3 m from the conductor at y = 61.2


def ground_z(x, y):
    return 0.01 * np.asarray(x, float) + 0.005 * np.asarray(y, float)


def _cable(p0, p1, sag, rng, step=0.15, keep=0.7, noise=0.03, gap=None):
    """Points along a cable hanging from p0 to p1 with the given sag (metres along the chord),
    a share of them kept, with noise; ``gap``: no points between these distances from p0."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    length = float(np.hypot(*(p1[:2] - p0[:2])))
    f = np.linspace(0.0, 1.0, int(length / step) + 1)
    pts = p0 + f[:, None] * (p1 - p0)
    a = length * length / (8.0 * sag)
    s = f * length
    pts[:, 2] -= sag - a * (np.cosh((s - length / 2) / a) - 1.0)
    sel = rng.random(len(pts)) < keep
    if gap is not None:
        sel &= ~((s >= gap[0]) & (s <= gap[1]))
    return pts[sel] + rng.normal(0.0, noise, (int(sel.sum()), 3))


def _scene(rng):
    """``(xyz, kinds, cables)`` in metres from the origin: kinds 0 ground, 1 tree, 2 building,
    3 cable, 4 pole; ``cables``: the true ends and sag of every conductor."""
    parts = []
    gx, gy = np.meshgrid(np.arange(0.25, 200.0, 0.5), np.arange(0.25, 120.0, 0.5))
    gx = gx.ravel() + rng.uniform(-0.2, 0.2, gx.size)
    gy = gy.ravel() + rng.uniform(-0.2, 0.2, gy.size)
    parts.append((np.column_stack([gx, gy, ground_z(gx, gy) + rng.normal(0, 0.02, gx.size)]), 0))
    for cx, cy, h, r in [TREE_BY_THE_LINE, (120, 30, 6.0, 2.5), (30, 20, 9.0, 2.5), (170, 100, 5.0, 2.0),
                         (90, 95, 8.0, 3.0), (150, 85, 7.0, 2.0)]:
        n = 1500
        d = rng.normal(size=(n, 3))
        d /= np.linalg.norm(d, axis=1)[:, None]
        canopy = np.array([cx, cy, ground_z(cx, cy) + h]) + d * (r * rng.uniform(0.85, 1.0, n))[:, None]
        trunk = np.column_stack([cx + rng.normal(0, 0.1, 60), cy + rng.normal(0, 0.1, 60),
                                 ground_z(cx, cy) + rng.uniform(0.0, h - r, 60)])
        parts.append((np.vstack([canopy, trunk]), 1))
    bx, by = np.meshgrid(np.arange(110.0, 130.0, 0.4), np.arange(10.0, 22.0, 0.4))
    roof = np.column_stack([bx.ravel(), by.ravel(), ground_z(bx, by).ravel() + 6.0]) + rng.normal(0, 0.02, (bx.size, 3))
    walls = []
    for x0, y0, x1, y1 in [(110, 10, 130, 10), (130, 10, 130, 22), (130, 22, 110, 22), (110, 22, 110, 10)]:
        f = np.linspace(0.0, 1.0, 60)
        xx, zz = np.meshgrid(x0 + f * (x1 - x0), np.arange(0.3, 6.0, 0.3))
        yy = np.meshgrid(y0 + f * (y1 - y0), np.arange(0.3, 6.0, 0.3))[0]
        walls.append(np.column_stack([xx.ravel(), yy.ravel(), ground_z(xx, yy).ravel() + zz.ravel()]))
    parts.append((np.vstack([roof, *walls]), 2))
    for px, py in MAIN + CROSS:
        zz = np.arange(0.2, HEIGHT + 0.6, 0.1)
        parts.append((np.column_stack([px + rng.normal(0, 0.08, len(zz)), py + rng.normal(0, 0.08, len(zz)),
                                       ground_z(px, py) + zz]), 4))
    cables = []
    for a, b in zip(MAIN[:-1], MAIN[1:]):
        for off in OFFSETS:
            p0 = (a[0], a[1] + off, float(ground_z(*a)) + HEIGHT)
            p1 = (b[0], b[1] + off, float(ground_z(*b)) + HEIGHT)
            gap = (25.0, 33.0) if a[0] == 80.0 and off == 0.0 else None
            parts.append((_cable(p0, p1, 1.5, rng, gap=gap), 3))
            cables.append({"p0": p0, "p1": p1, "sag": 1.5, "gap": gap is not None})
    p0 = (CROSS[0][0], CROSS[0][1], float(ground_z(*CROSS[0])) + 7.0)
    p1 = (CROSS[1][0], CROSS[1][1], float(ground_z(*CROSS[1])) + 7.0)
    parts.append((_cable(p0, p1, 2.0, rng), 3))
    cables.append({"p0": p0, "p1": p1, "sag": 2.0, "gap": False})
    xyz = np.vstack([p for p, _ in parts])
    kinds = np.concatenate([np.full(len(p), k) for p, k in parts])
    return xyz, kinds, cables


def _project(tmp_path):
    import pyarchinit_photogrammetry as ppg
    from pyarchinit_photogrammetry.core.pointcloud import PointCloud

    rng = np.random.default_rng(0)
    doc = ppg.Document()
    chunk = doc.add_chunk("rilievo")
    doc.chunk = chunk
    doc.save(tmp_path / "linee.ppg")
    chunk.crs = CRS
    chunk.transform = np.eye(4)
    chunk.transform[:3, 3] = ORIGIN
    xyz, kinds, cables = _scene(rng)
    chunk.point_cloud = PointCloud(xyz, rng.integers(0, 255, (len(xyz), 3)).astype(np.uint8))
    chunk.meta["kinds"] = kinds.tolist()
    chunk.meta["cables"] = cables
    doc.save()
    return doc


def _match(report, cables, along=2.5, beside=0.5):
    """The wire found for every true cable: both ends within ``along`` metres along the cable
    (the last points by a pole are not wire points) and ``beside`` metres beside it (the
    parallel conductors are 1.2 m apart), either way round."""

    def close(a, b, p0, p1):
        d = (p1 - p0) / np.linalg.norm(p1 - p0)
        for q, p in ((a, p0), (b, p1)):
            v = q - p
            if abs(v @ d) > along or np.linalg.norm(v - (v @ d) * d) > beside:
                return False
        return True

    found = {}
    for k, c in enumerate(cables):
        p0, p1 = np.array(c["p0"]) + ORIGIN, np.array(c["p1"]) + ORIGIN
        for w in report["wires"]:
            s, e = np.array(w["start"]), np.array(w["end"])
            if close(s, e, p0, p1) or close(e, s, p0, p1):
                found[k] = w
                break
    return found


def test_fit_recovers_a_catenary_and_a_straight_cable():
    rng = np.random.default_rng(1)
    a, s0, z0 = 250.0, 5.0, 100.0
    u = np.array([0.6, 0.8])
    s = np.linspace(-40.0, 20.0, 300)
    xy = np.array([1000.0, 2000.0]) + s[:, None] * u
    z = z0 + a * (np.cosh((s - s0) / a) - 1.0) + rng.normal(0, 0.02, len(s))
    cat, rms, lateral = detect.fit(np.column_stack([xy, z]))
    assert abs(cat.a - a) < 0.05 * a and abs(cat.s0 - (s0 + (cat.s_of(np.array([[1000.0, 2000.0]]))[0]))) < 0.5
    assert abs(cat.z0 - z0) < 0.05 and rms < 0.03 and lateral < 0.01
    assert abs(cat.length - a * (np.sinh(15 / a) + np.sinh(45 / a))) < 0.05
    straight = np.column_stack([xy, 50.0 + 0.1 * s + rng.normal(0, 0.02, len(s))])
    cat2, rms2, _ = detect.fit(straight)
    assert rms2 < 0.03 and cat2.sag < 0.05 and abs(cat2.length - 60.0 * np.hypot(1, 0.1)) < 0.1
    again = detect.Catenary.from_dict(cat.as_dict())
    assert np.allclose(again.z_at(s), cat.z_at(s), atol=0.002)


def test_wires_poles_clearance_obstacles_and_classes(tmp_path):
    doc = _project(tmp_path)
    chunk = doc.chunk
    kinds = np.array(chunk.meta["kinds"])
    cables = chunk.meta["cables"]
    seen = []
    report = detect.detect_chunk(chunk, progress=lambda f, m: seen.append(f))
    assert seen[-1] == 1.0 and report["ground"] == "morphology" and report["classified"]
    wires = report["wires"]
    assert len(wires) == 10, [(w["length_m"], w["start"]) for w in wires]
    found = _match(report, cables)
    assert len(found) == 10 and len({w["id"] for w in found.values()}) == 10
    for k, c in enumerate(cables):
        w = found[k]
        chord = float(np.hypot(*(np.array(c["p1"][:2]) - np.array(c["p0"][:2]))))
        assert 0.95 * chord <= w["length_m"] <= 1.02 * chord, (k, w["length_m"], chord)
        assert abs(w["sag_m"] - c["sag"]) < 0.3, (k, w["sag_m"])
        expected = (HEIGHT if c["sag"] == 1.5 else 7.0) - c["sag"]
        assert abs(w["clearance_m"] - expected) < 0.25, (k, w["clearance_m"], expected)
        assert w["rms_m"] < 0.1 and w["lateral_m"] < 0.1
    by_the_tree = found[2]  # first span, the conductor at y = 61.2
    assert 0.9 <= by_the_tree["obstacle_m"] <= 1.9, by_the_tree["obstacle_m"]
    assert found[1]["obstacle_m"] >= 2.0 and found[9]["obstacle_m"] >= 3.5  # the main line crosses 3.6 m above
    assert report["lines"] == 4  # three spans of three conductors, and the crossing line
    assert len({found[k]["line"] for k in (0, 1, 2)}) == 1 and found[9]["line"] != found[0]["line"]
    poles = report["poles"]
    assert len(poles) == 6 and all(p["detected"] for p in poles)
    for px, py in MAIN + CROSS:
        p = min(poles, key=lambda p: np.hypot(p["x"] - ORIGIN[0] - px, p["y"] - ORIGIN[1] - py))
        assert np.hypot(p["x"] - ORIGIN[0] - px, p["y"] - ORIGIN[1] - py) < 0.5
        assert abs(p["height_m"] - (HEIGHT + 0.5)) < 0.3, "the top of the pole's points, above the wires"
        assert abs(p["z_ground"] - ORIGIN[2] - float(ground_z(px, py))) < 0.2
        expected = {MAIN[0]: 3, MAIN[1]: 6, MAIN[2]: 6, MAIN[3]: 3, CROSS[0]: 1, CROSS[1]: 1}[(px, py)]
        assert len(p["wires"]) == expected, (px, py, p["wires"])
    codes = chunk.point_cloud.classification
    assert (codes[kinds == 3] == 14).mean() > 0.9, "the cables are class 14"
    assert (codes[np.isin(kinds, (0, 1, 2))] == 14).mean() < 0.002, "ground, trees and the building are not"
    assert (codes[kinds == 4] == 15).mean() > 0.8, "the poles are class 15"
    assert chunk.meta["classes"]["14"].startswith("Conduttore")
    json.dumps(chunk.meta["powerlines"])  # kept in the project as plain data
    assert "wire_index" not in chunk.meta["powerlines"]
    # the ground from the class 2 points when the cloud is classified
    pc = chunk.point_cloud  # read from its file at every access: change the object, then store it
    pc.classification[kinds == 0] = 2
    chunk.point_cloud = pc
    again = detect.detect_chunk(chunk, classify=False)
    assert again["ground"] == "classes" and len(again["wires"]) == 10 and not again["classified"]
    assert (chunk.point_cloud.classification[kinds == 3] == 14).mean() > 0.9, "classify=False leaves the classes"
    # a run with hand labels keeps them; forgetting resets the plugin's classes only
    from pyarchinit_photogrammetry.ai.classify import set_class

    first_tree = np.nonzero(kinds == 1)[0][:30]
    set_class(chunk, first_tree, 14, manual=True)
    detect.detect_chunk(chunk)
    assert (chunk.point_cloud.classification[first_tree] == 14).all()
    assert detect.clear_chunk(chunk) and "powerlines" not in chunk.meta
    codes = chunk.point_cloud.classification
    assert (codes[kinds == 3] == 1).mean() > 0.9 and (codes[first_tree] == 14).all() and "14" not in chunk.meta["classes"]
    assert not detect.clear_chunk(chunk)


def test_exports(tmp_path):
    import ezdxf

    from pyarchinit_photogrammetry.io import gpkg

    doc = _project(tmp_path)
    chunk = doc.chunk
    with pytest.raises(RuntimeError):
        export.export(chunk, tmp_path / "prima.gpkg")
    report = detect.detect_chunk(chunk)
    out = export.export(chunk, tmp_path / "out" / "linee.gpkg", progress=lambda f, m: None)
    rows = gpkg.read_table(out, "linee_elettriche")
    assert len(rows) == 10 and all(g.has_z for g, _ in rows)
    assert {r["id"] for _, r in rows} == set(range(1, 11)) and rows[0][1]["lunghezza_m"] > 40
    assert len(gpkg.read_table(out, "pali")) == 6
    import sqlite3

    con = sqlite3.connect(out)
    assert con.execute("SELECT z FROM gpkg_geometry_columns WHERE table_name = 'linee_elettriche'").fetchone()[0] == 1
    assert con.execute("SELECT srs_id FROM gpkg_contents WHERE table_name = 'pali'").fetchone()[0] == 32637
    con.close()
    dxf = export.export(chunk, tmp_path / "linee.dxf")
    msp = ezdxf.readfile(dxf).modelspace()
    assert len(msp.query("POLYLINE")) == 10 and len(msp.query('LINE[layer=="PALI"]')) == 6
    assert len(msp.query("TEXT")) == 16
    csv_path = export.export(chunk, tmp_path / "linee.csv")
    lines = csv_path.read_text().splitlines()
    assert len(lines) == 11 and lines[0].startswith("id,linea,lunghezza_m") and "x_inizio" in lines[0]
    assert len((tmp_path / "linee_pali.csv").read_text().splitlines()) == 7
    with pytest.raises(ValueError):
        export.export(chunk, tmp_path / "linee.txt")
    assert "10 conduttori" in export.table(report) and "L1" in export.table(report)
    assert "Franco minimo" in export.summary(report) and "classe 14" in export.summary(report)


class _View:
    def __init__(self):
        self.layers = {}
        self.tr = object()

    def remove_layer(self, name):
        self.layers.pop(name, None)

    def batch(self):
        return contextlib.nullcontext()


class _Window:
    def __init__(self):
        self.ortho_view = _View()
        self.line_layers = {}

    def _add_line_feature(self, layer, coords, color, label="", attrs=None, width=2.0):
        self.line_layers.setdefault(layer, []).append((np.asarray(coords, float), color, label, attrs, width))


def test_layers_of_the_window(tmp_path):
    doc = _project(tmp_path)
    report = detect.detect_chunk(doc.chunk, classify=False)
    window = _Window()
    export.draw(window, report)
    wires = window.line_layers[export.WIRES_LAYER]
    assert len(wires) == 10 and wires[0][2] == "L1" and wires[0][3]["lunghezza_m"] == report["wires"][0]["length_m"]
    assert all(c.shape[1] == 2 and len(c) > 40 for c, *_ in wires)
    assert len(window.line_layers[export.POLES_LAYER]) == 12
    export.draw(window, None)
    assert not window.line_layers


def test_the_plugin_in_the_program(tmp_path):
    from pyarchinit_photogrammetry.plugins import Manager

    manager = Manager(tmp_path / "plugins")
    manager.load()
    info = manager.get("powerlines")
    assert info is not None and info.active, info and info.error
    assert [a.title for a in manager.actions if a.plugin == "powerlines"] == [
        "Rileva linee elettriche...", "Mostra le linee elettriche", "Elenco delle linee elettriche...",
        "Dimentica le linee elettriche"]
    assert "powerlines" in manager.commands
    assert [e.suffix for e in manager.exporters if e.plugin == "powerlines"] == [".gpkg", ".dxf", ".csv"]


def test_command(tmp_path, capsys):
    from ppg_powerlines import PowerlinesPlugin

    doc = _project(tmp_path)
    plugin = PowerlinesPlugin()
    assert plugin.command("list", str(doc.path)) == 0 and "detect first" in capsys.readouterr().out
    assert plugin.command("detect", str(doc.path), "--export", str(tmp_path / "l.gpkg"), "--min-height", "3") == 0
    out = capsys.readouterr().out
    assert "10 conduttori" in out and "6 pali" in out and (tmp_path / "l.gpkg").exists()
    import pyarchinit_photogrammetry as ppg

    again = ppg.Document.open(doc.path)
    assert len(again.chunk.meta["powerlines"]["wires"]) == 10
    assert again.chunk.meta["powerlines"]["params"]["min_height"] == 3.0
    assert plugin.command("export", str(doc.path), str(tmp_path / "l.dxf")) == 0 and (tmp_path / "l.dxf").exists()
    assert plugin.command("export", str(doc.path), str(tmp_path / "l.txt")) == 1 and "error" in capsys.readouterr().out
    assert plugin.command("list", str(doc.path), "--chunk", "nessuno") == 2
    assert plugin.command("clear", str(doc.path)) == 0 and "forgotten" in capsys.readouterr().out
    assert "powerlines" not in ppg.Document.open(doc.path).chunk.meta
