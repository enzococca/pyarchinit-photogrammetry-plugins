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
    monkeypatch.setattr(elements, "_photo_direction", lambda chunk, xy: OUTWARD)


def _project(tmp_path: Path):
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
    assert abs(p["splay_left"] - tr["splay"]) < 0.3 and abs(p["splay_right"] - tr["splay"]) < 0.3
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
    monkeypatch.setattr(elements, "_photo_direction", lambda chunk, xy: rz[:2, :2] @ OUTWARD)
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
                         "--name", "P1"])
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
    monkeypatch.setattr(elements, "_photo_direction", lambda chunk, xy: OUTWARD)
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
