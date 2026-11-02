"""Openings on the plugin's side, with the fake worker (the engine's meshes, no CadQuery) and once with the real one: a
project in UTM whose chunk has a rotated and scaled transform and a model with three walls - a door (both faces of the
wall on the model), a window splayed 10 degrees (the outer face only: the thickness given), a window with a
semicircular top closed by an arch element - the openings against the truth, the fit, the IFC with its
IfcOpeningElement, IfcDoor and IfcWindow read back by ifcopenshell.geom, the DXF, an edit, a failed edit, the rebuild
after a change of the chunk's system, the command line and the dialogs in the program's window."""
from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarchinit_photogrammetry", reason="the plugin side needs the program")
HAS_IFC = importlib.util.find_spec("ifcopenshell") is not None

import pyarchinit_photogrammetry as ppg  # noqa: E402
from arch_samples import WALL_T, cadquery_python, opening_wall  # noqa: E402
from ppg_architecture import elements, engine  # noqa: E402

FAKE = Path(__file__).with_name("fake_worker.py")
CRS = "EPSG:32633"
WORLD = np.array([712345.678, 4567890.123, 312.5])
WALLS = {"door": WORLD + [0.0, 8.0, 0.0], "splayed": WORLD + [6.0, 8.0, 0.0], "arched": WORLD + [-6.0, 8.0, 0.0]}
OUTWARD = np.array([0.0, -1.0])  # the photos are in front of the walls (y less than theirs)
FORBIDDEN = ("cadquery", "OCP", "vtkmodules", "casadi", "nlopt")
CADQUERY = cadquery_python()


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setenv("PPG_ARCHITECTURE_FAKE", "")
    monkeypatch.setattr(elements, "WORKER", (Path(sys.executable), FAKE))


def add_cameras(ch, poses):
    """Aligned photos at world poses ``[(eye, target), ...]`` (a pinhole 1600 x 1200, focal 1500 px), in the chunk's
    internal frame."""
    from pyarchinit_photogrammetry.core.camera import Camera, Sensor
    from pyarchinit_photogrammetry.core.geometry import decompose_similarity, look_at

    if not ch.sensors:
        ch.sensors[0] = Sensor(key=0, label="prova", width=1600, height=1200, model="PINHOLE",
                               params=[1500.0, 1500.0, 800.0, 600.0])
    _s, rot, _t = decompose_similarity(ch.transform)
    for eye, target in poses:
        pose = look_at(np.asarray(eye, float), np.asarray(target, float))
        internal = np.eye(4)
        internal[:3, :3] = rot.T @ pose[:3, :3]
        internal[:3, 3] = ch.to_internal(np.asarray(eye, float)[None])[0]
        key = max([c.key for c in ch.cameras], default=-1) + 1
        ch.cameras.append(Camera(key=key, label=f"F{key}", path=f"F{key}.jpg", sensor_key=0, transform=internal))


def front(at, n: int = 5, distance: float = 4.0):
    """Photos in front of a wall (y less than its outer face), framing its opening."""
    return [(at + [dx, -distance, 1.5], at + [0.0, WALL_T / 2, 1.2]) for dx in np.linspace(-1.5, 1.5, n)]


def _project(tmp_path: Path, cameras: bool = True):
    """The three walls in world coordinates (UTM), the model in the chunk's internal frame through a transform that
    rotates (30 degrees about z, 2 about x), scales (x 2.5) and moves it."""
    import trimesh

    from pyarchinit_photogrammetry.core.geometry import compose_similarity
    from pyarchinit_photogrammetry.io.mesh_ply import write_mesh_ply

    parts, truth = [], {}
    for kind, at in WALLS.items():
        mesh, tr = opening_wall(kind, both_faces=kind != "splayed")
        mesh.apply_translation(at)
        parts.append(mesh)
        truth[kind] = dict(tr, at=at)
    mesh = trimesh.util.concatenate(parts)
    a, b = np.radians(30.0), np.radians(2.0)
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    doc = ppg.Document()
    ch = doc.add_chunk("Facciata")
    doc.save(tmp_path / "porte.ppg")
    ch.crs = CRS
    ch.transform = compose_similarity(2.5, rz @ rx, WORLD)
    internal = ch.to_internal(np.asarray(mesh.vertices, np.float64))
    path = write_mesh_ply(ch.data_dir / "model.ply", internal, np.asarray(mesh.faces), double=True)
    ch._set_product("model", path)
    ch.meta["metric"] = True
    if cameras:
        add_cameras(ch, [p for at in WALLS.values() for p in front(at)])
    doc.save()
    return doc, ch, truth


def _inside(truth):
    return truth["at"][:2] + [0.0, WALL_T / 2]


def _volume(tr) -> float:
    plan = tr["plan"]
    if tr["top"] == "arch":
        r = tr["width"] / 2
        return engine.opening_volume(plan, tr["sill"], tr["springing"], np.pi * r * r / 2, WALL_T)
    return engine.opening_volume(plan, tr["sill"], tr["lintel"])


def _arch_over(ch, truth, **kwargs):
    """The arch of the arched window, on the vertical plane through the middle of the wall (the ring is flush with the
    wall: its thickness and depth are given)."""
    at = truth["at"]
    y = at[1] + WALL_T / 2
    p1, p2 = np.array([at[0] - 2.0, y]), np.array([at[0] + 2.0, y])
    po, u, vv, _w = elements.plane_of(ch, None, p1, p2)
    rel = np.array([at[0], y, at[2] + 1.4]) - elements.origin(ch) - po
    return elements.arch(ch, (float(rel @ u), float(rel @ vv)), p1=p1, p2=p2, kind="semicircular", thickness=0.2,
                         depth=WALL_T, name="Arco della finestra", **kwargs)


def _check(e, tr, tol=0.003):
    p = e["params"]
    assert abs(p["width"] - tr["width"]) < tol and abs(p["width_inner"] - tr["inner_width"]) < tol
    assert abs(p["reveal_depth"] - WALL_T) < tol
    assert abs(p["splay_left"] - tr["splay_left"]) < 0.3 and abs(p["splay_right"] - tr["splay_right"]) < 0.3
    assert p["outer_rule"] == "photos" and p["outer_views"] > p["inner_views"]
    assert abs(p["sill"] - (tr["at"][2] + tr["sill"])) < tol and abs(p["lintel"] - (tr["at"][2] + tr["lintel"])) < tol
    assert abs(p["height"] - (tr["lintel"] - tr["sill"])) < tol
    assert abs(e["solid"]["volume"] / _volume(tr) - 1) < 0.01
    assert e["fit"]["median"] < 0.001 and e["fit"]["points"] > 100 and e["fit"]["excluded_faces"] > 0
    threshold = np.asarray(p["world_threshold"])
    assert np.hypot(*(threshold[:2] - tr["at"][:2])) < tol  # the middle of the outer face at the sill
    assert np.asarray(p["world_axis"])[:2] @ OUTWARD > 0.999


def test_openings_from_a_model_in_utm_with_a_rotated_scaled_transform(tmp_path, fake):
    doc, ch, truth = _project(tmp_path)
    model = ch.product_path("model").read_bytes()
    door = elements.opening(ch, _inside(truth["door"]), kind="door", name="Porta")
    _check(door, truth["door"])
    assert door["params"]["thickness_measured"] and door["params"]["opening_type"] == "door"
    with pytest.raises(elements.NotReady, match="spessore"):  # one face only: the thickness is asked for
        elements.opening(ch, _inside(truth["splayed"]))
    window = elements.opening(ch, _inside(truth["splayed"]), kind="window", thickness=WALL_T, name="Finestra")
    _check(window, truth["splayed"])
    assert not window["params"]["thickness_measured"]
    arch = _arch_over(ch, truth["arched"])
    arched = elements.opening(ch, _inside(truth["arched"]), top=arch["id"], kind="window", name="Finestra ad arco")
    _check(arched, truth["arched"])
    assert arched["params"]["top"] == "arch" and arched["params"]["arch"] == arch["id"]
    assert abs(arched["params"]["springing"] - (truth["arched"]["at"][2] + 1.8)) < 0.003
    assert arched["geometry"]["void"]["arch"] == arch["id"]
    assert "finestra, ad arco 0,800 m × 1,200 m" in elements.summary(arched)  # the interface's numbers
    assert "strombatura 10,0° / 10,0°" in elements.summary(window)
    # nothing but the plugin's folder written, the model untouched
    assert ch.product_path("model").read_bytes() == model
    assert {q.relative_to(ch.data_dir).parts[0] for q in ch.data_dir.rglob("*") if q.is_file()} == \
        {"model.ply", "architecture"}
    doc.save()
    again = ppg.Document.open(doc.path).chunk
    assert [e["name"] for e in elements.elements(again)] == ["Porta", "Finestra", "Arco della finestra",
                                                             "Finestra ad arco"]
    assert not set(FORBIDDEN) & set(sys.modules)


def test_ifc_with_the_opening_elements_and_the_dxf(tmp_path, fake):
    pytest.importorskip("ifcopenshell")
    import ezdxf
    import ifcopenshell
    import ifcopenshell.util.element

    from ppg_architecture import drawings, ifc

    doc, ch, truth = _project(tmp_path)
    door = elements.opening(ch, _inside(truth["door"]), kind="door", name="Porta")
    window = elements.opening(ch, _inside(truth["splayed"]), kind="window", thickness=WALL_T, name="Finestra")
    arch = _arch_over(ch, truth["arched"])
    vano = elements.opening(ch, _inside(truth["arched"]), top=arch["id"], name="Vano")
    out = ifc.export(ch, tmp_path / "out" / "aperture.ifc")
    volumes = ifc.read_volumes(out)
    for e in (door, window, vano):
        for key in (f"IfcOpeningElement:{e['name']}", e["name"]):
            vol, bbox = volumes[key]
            assert abs(vol / e["solid"]["volume"] - 1) < 0.01, key
            np.testing.assert_allclose(bbox, e["solid"]["bbox"], atol=0.01, err_msg=key)
    assert abs(volumes["IfcDoor:Porta"][0] / door["solid"]["volume"] - 1) < 0.01
    assert abs(volumes["IfcWindow:Finestra"][0] / window["solid"]["volume"] - 1) < 0.01
    model = ifcopenshell.open(str(out))
    assert sorted(o.Name for o in model.by_type("IfcOpeningElement")) == ["Finestra", "Porta", "Vano"]
    d = model.by_type("IfcDoor")[0]
    assert d.PredefinedType == "DOOR" and d.OverallWidth == pytest.approx(door["params"]["width"])
    assert d.OverallHeight == pytest.approx(door["params"]["height"])
    assert model.by_type("IfcWindow")[0].PredefinedType == "WINDOW" and not model.by_type("IfcWindow")[1:]
    vano_e = [o for o in model.by_type("IfcOpeningElement") if o.Name == "Vano"][0]
    items = vano_e.Representation.Representations[0].Items
    assert len(items) == 2 and any(s.is_a("IfcArcIndex") for s in items[1].SweptArea.OuterCurve.Segments)
    pset = ifcopenshell.util.element.get_psets(d)["Pset_PyArchInit_Element"]
    for key in ("P_sill", "P_lintel", "P_reveal_depth", "P_splay_left", "P_splay_right"):
        assert key in pset, key
    assert pset["P_sill"] == pytest.approx(door["params"]["sill"]) and pset["Kind"] == "opening"
    dxf = drawings.write_dxf(window, tmp_path / "out" / "finestra.dxf")
    layers = {x.dxf.layer for x in ezdxf.readfile(dxf).modelspace()}
    assert {"SOLIDO", "MISURATO", "FACCE", "SOGLIA", "ARCHITRAVE", "ASSE"} <= layers
    plan = drawings.lines_of(window)["SOLIDO"][0][0]
    assert abs(np.ptp(plan[:, 1]) - WALL_T) < 0.003  # the plan across the wall, from the outer face
    assert abs(plan[:, 0].min() + truth["splayed"]["inner_width"] / 2) < 0.003
    assert "IMPOSTA" in {x.dxf.layer for x in ezdxf.readfile(drawings.write_dxf(vano, tmp_path / "v.dxf")).modelspace()}


def test_edits_a_failed_edit_and_the_rebuild(tmp_path, fake, monkeypatch):
    from pyarchinit_photogrammetry.core.geometry import compose_similarity

    from ppg_architecture import runner

    doc, ch, truth = _project(tmp_path)
    door = elements.opening(ch, _inside(truth["door"]), kind="door", name="Porta")
    wider = elements.edit_opening(ch, door["id"], width=1.0, sill=door["params"]["sill"] + 0.05)
    p = wider["params"]
    assert abs(p["width"] - 1.0) < 1e-9 and abs(p["width_inner"] - 1.0) < 1e-6
    assert abs(p["height"] - (door["params"]["height"] - 0.05)) < 1e-9 and wider["edited"] == ["sill", "width"]
    assert wider["fit"]["median"] > 0.01  # 5 cm wider each side than the model: the fit says it
    thinner = elements.edit_opening(ch, door["id"], thickness=0.3)
    assert abs(thinner["params"]["reveal_depth"] - 0.3) < 1e-9 and not thinner["params"]["thickness_measured"]
    arch = _arch_over(ch, truth["arched"])
    vano = elements.opening(ch, _inside(truth["arched"]), top=arch["id"], name="Vano")
    with pytest.raises(elements.NotReady, match="arco"):
        elements.edit_opening(ch, vano["id"], lintel=vano["params"]["lintel"] + 0.1)
    before = copy.deepcopy(elements.elements(ch))
    monkeypatch.setenv("PPG_ARCHITECTURE_FAKE", "fail")
    with pytest.raises(runner.WorkerError):
        elements.edit_opening(ch, door["id"], width=1.2)
    with pytest.raises(runner.WorkerError):
        elements.opening(ch, _inside(truth["door"]), name="Altra")
    assert elements.elements(ch) == before
    assert sorted(q.name for q in elements.folder(ch).iterdir()) == sorted(e["id"] for e in before)
    monkeypatch.setenv("PPG_ARCHITECTURE_FAKE", "")
    # the chunk turned 90 degrees about the vertical and moved: refused, then rebuilt with the edits
    volumes = {e["id"]: e["solid"]["volume"] for e in elements.elements(ch)}
    rz = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    move = compose_similarity(1.0, rz, [10.0, 0.0, 0.0])
    move[:3, 3] += WORLD - rz @ WORLD
    ch.transform = move @ ch.transform
    with pytest.raises(elements.NotReady, match="sistema del chunk è cambiato"):
        elements.edit_opening(ch, door["id"], width=0.9)
    made = elements.rebuild(ch)
    assert [e["name"] for e in made] == ["Porta", "Arco della finestra", "Vano"]
    for e in made:
        assert abs(e["solid"]["volume"] / volumes[e["id"]] - 1) < 0.002, e["name"]
    door2 = elements.entry(ch, "Porta")
    assert door2["edited"] == ["sill", "thickness", "width"]
    assert abs(door2["params"]["width"] - 1.0) < 1e-6 and abs(door2["params"]["reveal_depth"] - 0.3) < 1e-6
    np.testing.assert_allclose(door2["params"]["world_threshold"],
                               engine.apply(move, np.asarray(thinner["params"]["world_threshold"])[None])[0],
                               atol=0.003)
    assert elements.entry(ch, "Vano")["params"]["arch"] == arch["id"]


def _plugin_folder(root: Path) -> Path:
    folder = root / "plugins"
    dist = folder / "architecture" / "ppg_architecture-0.2.0.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: ppg-architecture\nVersion: 0.2.0\n")
    (dist / "entry_points.txt").write_text("[pyarchinit_photogrammetry.plugins]\narchitecture = "
                                           "ppg_architecture:ArchitecturePlugin\n")
    return folder


def test_command_line(tmp_path, fake, monkeypatch):
    from typer.testing import CliRunner

    from pyarchinit_photogrammetry.cli import app

    doc, ch, truth = _project(tmp_path)
    monkeypatch.setenv("PPG_PLUGINS_DIR", str(_plugin_folder(tmp_path)))
    cli = CliRunner()
    project = str(doc.path)
    x, y = _inside(truth["door"])
    r = cli.invoke(app, ["plugin", "architecture", "opening", project, "--at", f"{x},{y}", "--kind", "door",
                         "--outer", "auto", "--name", "P1"])
    assert r.exit_code == 0, r.output
    assert "Apertura: porta 0,900 m × 2,100 m" in r.output and "scarto mediana" in r.output
    x, y = _inside(truth["splayed"])
    r = cli.invoke(app, ["plugin", "architecture", "opening", project, "--at", f"{x},{y}", "--kind", "window"])
    assert r.exit_code == 1 and "spessore" in r.output  # one face only
    r = cli.invoke(app, ["plugin", "architecture", "opening", project, "--at", f"{x},{y}", "--kind", "window",
                         "--thickness", str(WALL_T), "--sill", str(truth["splayed"]["at"][2] + 1.1)])
    assert r.exit_code == 0, r.output
    win = elements.need(ppg.Document.open(project).chunk, "Finestra 1")
    assert abs(win["params"]["height"] - 0.9) < 0.003 and win["source"]["sill_given"]
    r = cli.invoke(app, ["plugin", "architecture", "opening", project, "--at", "-1,-1"])
    assert r.exit_code == 1 and "apertura" in r.output.lower()
    r = cli.invoke(app, ["plugin", "architecture", "export", project, "--out", str(tmp_path / "p.dxf"), "--ids", "P1"])
    assert r.exit_code == 0 and (tmp_path / "p.dxf").exists(), r.output


def test_the_dialogs_in_the_window(tmp_path, fake, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from ppg_architecture import ArchitecturePlugin
    from pyarchinit_photogrammetry.app import app
    from pyarchinit_photogrammetry.plugins import Context

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "settings"))
    QApplication.instance() or QApplication([])
    from pyarchinit_photogrammetry.gui import main_window as mwmod

    doc, ch, truth = _project(tmp_path)
    app.document = doc
    w = mwmod.MainWindow(doc, autoconnect=False, check_updates=False)
    w.runner.show_dialog = False
    plugin = ArchitecturePlugin()
    try:
        plugin.activate(Context("architecture", tmp_path / "plugin", window=w))
        assert "opening" in plugin.pane.buttons
        clicks = [_inside(truth["door"]), _inside(truth["arched"])]
        monkeypatch.setattr(plugin.context, "request_point", lambda done, hint="": done(np.array([clicks.pop(0)])))
        asked = []

        def answer(title, fields, intro=""):
            asked.append((title, [f.name for f in fields]))
            values = {f.name: f.default for f in fields if f.kind != "heading"}
            if title == "Porta / finestra":
                values["kind"] = "door"
                assert values["auto_height"] is True and values["outer"] == "auto"
            return values

        monkeypatch.setattr(plugin, "_ask", answer)
        plugin._do_opening()
        w.runner.wait()
        door = elements.elements(ch)[0]
        assert door["name"] == "Porta 1" and door["params"]["opening_type"] == "door"
        assert abs(door["params"]["width"] - 0.9) < 0.003
        assert plugin.pane.table.item(0, 2).text().startswith("Porta 0,900 m")
        _arch_over(ch, truth["arched"])
        plugin.refresh_pane()

        def answer_arch(title, fields, intro=""):
            values = answer(title, fields, intro)
            if title == "Porta / finestra":
                tops = [c for c in fields if c.name == "top"][0].choices
                values.update(kind="window", top=tops[1][0])
            return values

        monkeypatch.setattr(plugin, "_ask", answer_arch)
        plugin._do_opening()
        w.runner.wait()
        assert [e["params"].get("top") for e in elements.elements(ch)] == ["flat", None, "arch"]
        plugin.pane.table.clearSelection()
        plugin.pane.table.selectRow(0)

        def edit(title, fields, intro=""):
            values = {f.name: f.default for f in fields}
            values["width"] = 1.0
            return values

        monkeypatch.setattr(plugin, "_ask", edit)
        plugin._do_edit()
        w.runner.wait()
        assert abs(elements.elements(ch)[0]["params"]["width"] - 1.0) < 1e-6
        assert elements.elements(ch)[0]["edited"] == ["width"]
        plugin.pane.table.clearSelection()
        plugin.pane.table.selectRow(2)
        names = []
        monkeypatch.setattr(plugin, "_ask", lambda title, fields, intro="": names.append([f.name for f in fields]))
        plugin._do_edit()
        assert names == [["sill", "width", "thickness"]]  # the top of an arched opening is the arch's
        plugin.deactivate()
        plugin.context.release()
        assert not set(FORBIDDEN) & set(sys.modules)
    finally:
        w.setWindowModified(False)
        w.close()


@pytest.mark.skipif(CADQUERY is None, reason="no Python with CadQuery (PPG_TEST_CADQUERY)")
def test_openings_and_their_ifc_through_cadquery(tmp_path, monkeypatch):
    """The real worker: the solids of CadQuery (the arched one a single solid) against the truth, the IFC read back
    within 1 % of their volume."""
    pytest.importorskip("ifcopenshell")
    from ppg_architecture import ifc, runner

    monkeypatch.setattr(elements, "WORKER", (CADQUERY, Path(runner.__file__).with_name("worker.py")))
    doc, ch, truth = _project(tmp_path)
    window = elements.opening(ch, _inside(truth["splayed"]), kind="window", thickness=WALL_T, name="Finestra")
    arch = _arch_over(ch, truth["arched"])
    vano = elements.opening(ch, _inside(truth["arched"]), top=arch["id"], kind="door", name="Vano")
    for e, kind in ((window, "splayed"), (vano, "arched")):
        assert e["solid"]["valid"] and abs(e["solid"]["volume"] / _volume(truth[kind]) - 1) < 0.01, e["name"]
        assert e["fit"]["median"] < 0.001
    volumes = ifc.read_volumes(ifc.export(ch, tmp_path / "aperture.ifc", ids=[window["id"], vano["id"]]))
    for e in (window, vano):
        vol, bbox = volumes[f"IfcOpeningElement:{e['name']}"]
        assert abs(vol / e["solid"]["volume"] - 1) < 0.01, (e["name"], vol, e["solid"]["volume"])
        np.testing.assert_allclose(bbox, e["solid"]["bbox"], atol=0.01, err_msg=e["name"])
    assert abs(volumes["IfcDoor:Vano"][0] / vano["solid"]["volume"] - 1) < 0.01
    assert not set(FORBIDDEN) & set(sys.modules)


def test_the_outer_face_is_the_one_the_photos_see(tmp_path, fake):
    """More photos behind the wall than in front, but looking away from it: the outer face is the one seen by the
    photos that frame the opening (the mean direction of all the photos would say the other). Without photos the
    choice of the dialog decides, and the summary says which rule was used."""
    doc, ch, truth = _project(tmp_path, cameras=False)
    at = truth["splayed"]["at"]
    add_cameras(ch, front(at, n=3))
    add_cameras(ch, [(at + [dx, 6.0, 1.5], at + [dx, 20.0, 1.5]) for dx in np.linspace(-3, 3, 8)])
    w = elements.opening(ch, _inside(truth["splayed"]), thickness=WALL_T, name="Finestra")
    p = w["params"]
    assert p["outer_rule"] == "photos" and (p["outer_views"], p["inner_views"]) == (3, 0)
    assert abs(p["width"] - 0.8) < 0.003 and abs(p["splay_left"] - 10) < 0.3
    assert "la faccia vista da più foto" in elements.summary(w)
    doc2, ch2, truth2 = _project(tmp_path / "senza", cameras=False)
    near = elements.opening(ch2, truth2["door"]["at"][:2] + [0.0, 0.1], name="Vicina")  # nearer the outer face
    assert near["params"]["outer_rule"] == "near-fallback" and near["params"]["world_axis"][1] < -0.99
    assert "le foto non distinguono le facce" in elements.summary(near)
    far = elements.opening(ch2, truth2["door"]["at"][:2] + [0.0, 0.1], outer="far", name="Lontana")
    assert far["params"]["outer_rule"] == "far" and far["params"]["world_axis"][1] > 0.99


def test_the_ifc_is_valid_with_a_stand_in_wall(tmp_path, fake):
    pytest.importorskip("ifcopenshell")
    import logging

    import ifcopenshell
    import ifcopenshell.validate

    from ppg_architecture import ifc

    doc, ch, truth = _project(tmp_path)
    door = elements.opening(ch, _inside(truth["door"]), kind="door", name="Porta")
    window = elements.opening(ch, _inside(truth["splayed"]), kind="window", thickness=WALL_T, name="Finestra")
    # a second window in the door's wall, its stretch overlapping the door's: one host for both
    other = elements.opening(ch, _inside(truth["door"]), kind="window", name="Seconda")
    out = ifc.export(ch, tmp_path / "aperture.ifc")
    model = ifcopenshell.open(str(out))
    problems = []

    class Collect(logging.Handler):
        def emit(self, record):
            problems.append(record.getMessage())

    log = logging.getLogger("ppg-ifc-validate")
    log.addHandler(Collect())
    log.setLevel(logging.DEBUG)
    ifcopenshell.validate.validate(model, log, express_rules=False)
    assert problems == []
    walls = model.by_type("IfcWall")
    assert len(walls) == 2 and {x.Name for x in walls} == {"Muro (porzione misurata)"}
    hosts = {o.Name: o.VoidsElements[0].RelatingBuildingElement for o in model.by_type("IfcOpeningElement")}
    assert hosts["Porta"] == hosts["Seconda"] and hosts["Porta"] != hosts["Finestra"]
    door_e = model.by_type("IfcDoor")[0]
    assert door_e.FillsVoids[0].RelatingOpeningElement.Name == "Porta"
    import ifcopenshell.util.element

    pset = ifcopenshell.util.element.get_psets(walls[0])["Pset_PyArchInit_Element"]
    assert pset["Kind"] == "host" and "sostitutivo" in pset["Interpretation"]
    assert "sostitutivo" in ifcopenshell.util.element.get_psets(door_e)["Pset_PyArchInit_Element"]["Host"]
    assert not model.by_type("IfcRelContainedInSpatialStructure")[0].RelatedElements[0].is_a("IfcOpeningElement")
    for rel in model.by_type("IfcRelContainedInSpatialStructure"):
        assert not any(x.is_a("IfcOpeningElement") for x in rel.RelatedElements)
    volumes = ifc.read_volumes(out)
    for e in (door, window, other):
        assert abs(volumes[f"IfcOpeningElement:{e['name']}"][0] / e["solid"]["volume"] - 1) < 0.01


def test_an_opening_follows_its_arch(tmp_path, fake):
    """The arch of an arched opening cannot be removed while the opening uses it; editing the arch leaves the opening
    to be rebuilt (said in the pane's text, refused in the exports) until Rebuild makes it again on the new arch."""
    from ppg_architecture import ifc

    doc, ch, truth = _project(tmp_path)
    arch = _arch_over(ch, truth["arched"])
    vano = elements.opening(ch, _inside(truth["arched"]), top=arch["id"], name="Vano")
    with pytest.raises(elements.NotReady, match="chiude in alto «Vano»"):
        elements.remove(ch, arch["id"])
    elements.edit_arch(ch, arch["id"], thickness=0.3, depth=0.6)
    assert elements.entry(ch, vano["id"])["stale"] is True
    assert "Da ricostruire" in elements.summary(elements.entry(ch, vano["id"]))
    with pytest.raises(elements.NotReady, match="Da ricostruire"):
        elements.export_step(ch, None, tmp_path / "s")
    if HAS_IFC:
        with pytest.raises(elements.NotReady, match="Da ricostruire"):
            ifc.export(ch, tmp_path / "x.ifc")
    made = elements.rebuild_stale(ch)
    assert [e["name"] for e in made] == ["Vano"] and not elements.entry(ch, "Vano").get("stale")
    assert [e["name"] for e in elements.elements(ch)] == ["Arco della finestra", "Vano"]
    elements.export_step(ch, None, tmp_path / "s")
    elements.remove(ch, vano["id"])
    assert elements.remove(ch, arch["id"])


def test_rebuild_keeps_the_height_found_and_the_order(tmp_path, fake):
    from pyarchinit_photogrammetry.core.geometry import compose_similarity

    doc, ch, truth = _project(tmp_path)
    door = elements.opening(ch, _inside(truth["door"]), name="Porta")  # made before the arch it does not use
    arch = _arch_over(ch, truth["arched"])
    vano = elements.opening(ch, _inside(truth["arched"]), top=arch["id"], name="Vano")
    order = [e["id"] for e in elements.elements(ch)]
    # the arch's opening first in the registry: the rebuild must still make the arch before it
    st = elements.state(ch)
    st["elements"] = [st["elements"][2], st["elements"][0], st["elements"][1]]
    order = [order[2], order[0], order[1]]
    ch.transform = compose_similarity(1.0, np.eye(3), [5.0, 0.0, 0.0]) @ ch.transform
    made = elements.rebuild(ch)
    assert [e["id"] for e in made] == order and [e["id"] for e in elements.elements(ch)] == order
    assert elements.entry(ch, "Porta")["source"]["height"] is None  # found again, not frozen as given
    assert abs(elements.entry(ch, "Porta")["params"]["width"] - door["params"]["width"]) < 1e-6
    assert elements.entry(ch, "Vano")["params"]["arch"] == arch["id"] and vano


def test_warnings_of_an_arched_top_over_a_lintel(tmp_path, fake):
    doc, ch, truth = _project(tmp_path)
    arch = _arch_over(ch, truth["arched"])
    vano = elements.opening(ch, _inside(truth["arched"]), top=arch["id"], name="Vano")
    assert not vano["warnings"]
    entry = elements.entry(ch, vano["id"])
    entry["profile"]["lintel_measured"] = vano["params"]["springing"] - 0.2  # as if a flat lintel were under it
    again = elements.edit_opening(ch, vano["id"], sill=vano["params"]["sill"])
    assert any("entrerebbe nella muratura" in w for w in again["warnings"])


def test_messages_ask_for_what_helps(tmp_path, fake):
    doc, ch, truth = _project(tmp_path)
    with pytest.raises(elements.NotReady) as exc:
        elements.opening(ch, truth["door"]["at"][:2] + [1.5, WALL_T / 2])  # inside the masonry, beside the door
    assert "dai la quota" not in str(exc.value) and "fra i due stipiti" in str(exc.value)
    with pytest.raises(elements.NotReady, match="lascia la quota automatica"):
        elements.opening(ch, _inside(truth["door"]), height=truth["door"]["at"][2] + 2.6)
    assert "Dai lo spessore del muro o usa un modello più fitto" in elements._opening_problem(
        "the faces of a thin wall cannot be told apart at this resolution (edges of 30 mm)")


def test_a_door_with_a_rebate_in_the_project_and_its_ifc(tmp_path, fake):
    """A door 1.0 m clear with a 10 cm rebate on each side: the clear and the overall width, the rebates and the
    warning in the element, the summary and the Pset; the IFC's OverallWidth is the clear width, the void's plan has
    the steps."""
    from pyarchinit_photogrammetry.core.geometry import compose_similarity
    from pyarchinit_photogrammetry.io.mesh_ply import write_mesh_ply

    mesh, _truth = opening_wall("door", width=1.0, rebate=0.10, edge=0.02)
    mesh.apply_translation(WORLD)
    doc = ppg.Document()
    ch = doc.add_chunk("Porta")
    doc.save(tmp_path / "battuta.ppg")
    ch.crs = CRS
    ch.transform = compose_similarity(1.0, np.eye(3), WORLD)
    path = write_mesh_ply(ch.data_dir / "model.ply", ch.to_internal(np.asarray(mesh.vertices, np.float64)),
                          np.asarray(mesh.faces), double=True)
    ch._set_product("model", path)
    ch.meta["metric"] = True
    add_cameras(ch, front(WORLD))
    e = elements.opening(ch, WORLD[:2] + [0.0, WALL_T / 2], kind="door", name="Porta")
    p = e["params"]  # a coarser mesh than the engine's test (2 cm triangles, its own resolution): 5 mm
    assert abs(p["clear_width"] - 1.0) < 0.005 and abs(p["overall_width"] - 1.2) < 0.005
    for side in ("rebate_left", "rebate_right"):
        assert abs(p[side]["width"] - 0.10) < 0.005 and abs(p[side]["depth"] - WALL_T / 2) < 0.005
    assert abs(p["splay_left"]) < 0.5 and abs(p["splay_right"]) < 0.5 and p["reveal_stepped"] == "left, right"
    assert any("Stipiti a gradini" in w for w in e["warnings"])
    assert "luce netta " in elements.summary(e) and "battuta sinistra " in elements.summary(e)
    assert len(e["geometry"]["plan"]) == 8
    assert abs(e["solid"]["volume"] - (1.2 * 0.25 + 1.0 * 0.25) * 2.1) < 0.01
    if HAS_IFC:
        import ifcopenshell
        import ifcopenshell.util.element

        from ppg_architecture import ifc

        model = ifcopenshell.open(str(ifc.export(ch, tmp_path / "porta.ifc")))
        door = model.by_type("IfcDoor")[0]
        assert door.OverallWidth == pytest.approx(p["clear_width"])
        pset = ifcopenshell.util.element.get_psets(door)["Pset_PyArchInit_Element"]
        assert "P_rebate_left" in pset and "P_clear_width" in pset


def test_curved_walls_and_short_piers_are_said_in_italian():
    from ppg_architecture import engine

    assert "Pareti curve non supportate in questa versione" in elements._opening_problem(engine.CURVED_WALL)
    said = elements._opening_problem(engine.END_OR_PIER)
    assert "Il muro finisce accanto al punto, o un pilastro" in said and "15 cm" in said
