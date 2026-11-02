"""The plugin's side, with the fake worker (the engine's meshes, no CadQuery): a project in UTM whose chunk has a
rotated and scaled transform and a model holding a column, an arch with its piers and a cornice along an L; the
elements made from it, their numbers against the truth, the fit, the files only in the chunk's folder, the model
untouched, the IFC read back by ifcopenshell.geom, the DXF, the command line through the program's ``ppg plugin``,
the pane in the program's window - and CadQuery never in the program's process."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import importlib.util
import pytest

pytest.importorskip("pyarchinit_photogrammetry", reason="the plugin side needs the program")
HAS_IFC = importlib.util.find_spec("ifcopenshell") is not None  # the program's bim extra

import pyarchinit_photogrammetry as ppg  # noqa: E402
from arch_samples import (HEIGHT, L_PATH, arch_mesh, cadquery_python, column_mesh, column_radius,  # noqa: E402
                          column_volume, cornice_mesh)
from ppg_architecture import elements, engine  # noqa: E402

FAKE = Path(__file__).with_name("fake_worker.py")
CRS = "EPSG:32633"
WORLD = np.array([712345.678, 4567890.123, 312.5])  # the middle of the scene
COLUMN_AT = WORLD + [0.0, 0.0, 0.0]
ARCH_AT = WORLD + [5.0, 0.0, 0.0]
CORNICE_AT = WORLD + [-8.0, -4.0, 0.0]
FORBIDDEN = ("cadquery", "OCP", "vtkmodules", "casadi", "nlopt")
CADQUERY = cadquery_python()


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setenv("PPG_ARCHITECTURE_FAKE", "")
    monkeypatch.setattr(elements, "WORKER", (Path(sys.executable), FAKE))


def _scene():
    """The scene in world coordinates and its truths."""
    import trimesh

    col, _b, _d = column_mesh(tilt_deg=0.0, origin=COLUMN_AT)
    arch, truth = arch_mesh("pointed")
    arch.apply_translation(ARCH_AT)
    cornice, plug = cornice_mesh()
    volume = abs(cornice.volume)  # before the move: trimesh's volume of a mesh at UTM coordinates is not exact
    v, f = trimesh.remesh.subdivide_to_size(cornice.vertices, cornice.faces, max_edge=0.05, max_iter=20)
    cornice = trimesh.Trimesh(v, f)  # vertices all along it, as a photogrammetric mesh has
    cornice.apply_translation(CORNICE_AT)
    mesh = trimesh.util.concatenate([col, arch, cornice])
    return mesh, truth, plug, volume


def _project(tmp_path: Path):
    """A project whose chunk's transform rotates (30 degrees about z, 2 about x), scales (x 2.5) and moves the model to
    UTM: the model.ply holds small internal coordinates."""
    from pyarchinit_photogrammetry.core.geometry import compose_similarity
    from pyarchinit_photogrammetry.io.mesh_ply import write_mesh_ply

    mesh, truth, plug, cornice_volume = _scene()
    a, b = np.radians(30.0), np.radians(2.0)
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    transform = compose_similarity(2.5, rz @ rx, WORLD)
    doc = ppg.Document()
    ch = doc.add_chunk("Facciata")
    doc.save(tmp_path / "chiesa.ppg")
    ch.crs = CRS
    ch.transform = transform
    internal = ch.to_internal(np.asarray(mesh.vertices, np.float64))
    path = write_mesh_ply(ch.data_dir / "model.ply", internal, np.asarray(mesh.faces), double=True)
    ch._set_product("model", path)
    ch.meta["metric"] = True
    doc.save()
    return doc, ch, {"arch": truth, "plug": plug, "cornice_volume": cornice_volume}


def _elevation(ch, eid: str, kind: str, origin, u_axis, v_axis, d=None):
    """An elevation (or section) of the project as the program lists it: its frame, an orthophoto and a relief in
    frame metres (``d``: the relief's value, NaN by default: the depth then comes from the model)."""
    import rasterio
    from rasterio.transform import from_origin

    from pyarchinit_photogrammetry.engine.elevation import Plane

    frame = Plane(origin, u_axis, v_axis)
    folder = ch.data_dir / "elevations" / eid
    folder.mkdir(parents=True, exist_ok=True)
    tr = from_origin(-5.0, 10.0, 0.05, 0.05)
    for name, count, dtype, value in (("elevation.tif", 4, "uint8", 128), ("relief.tif", 1, "float32",
                                                                              np.nan if d is None else d)):
        with rasterio.open(folder / name, "w", driver="GTiff", height=300, width=200, count=count, dtype=dtype,
                           transform=tr, nodata=float("nan") if dtype == "float32" else None) as dst:
            dst.write(np.full((count, 300, 200), value, dtype=dtype))
    entry = {"id": eid, "name": eid, "kind": kind, "frame": frame.to_dict(), "resolution": 0.05,
             "orthophoto": f"elevations/{eid}/elevation.tif", "relief": f"elevations/{eid}/relief.tif"}
    ch.meta.setdefault("elevations", []).append(entry)
    return frame


def _digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _square(centre, half):
    c = np.asarray(centre)[:2]
    return np.array([c + [-half, -half], c + [half, -half], c + [half, half], c + [-half, half]])


def test_elements_from_a_model_in_utm_with_a_rotated_scaled_transform(tmp_path, fake):
    doc, ch, truth = _project(tmp_path)
    model = ch.product_path("model")
    before, transform = _digest(model), ch.transform.copy()
    # the column: a square around it in plan, between two heights
    col = elements.column(ch, polygon=_square(COLUMN_AT, 0.6), zmin=COLUMN_AT[2] - 0.1, zmax=COLUMN_AT[2] + 3.1,
                          knots=12, name="Colonna A")
    p = col["params"]
    assert np.hypot(*(np.asarray(p["axis_point"])[:2] - COLUMN_AT[:2])) < 0.0005
    assert abs(p["axis_point"][2] - COLUMN_AT[2]) < 0.001 and p["tilt_deg"] < 0.1
    assert abs(p["height"] - HEIGHT) < 0.001 and abs(p["radius_base"] - column_radius(0.0)) < 0.001
    assert abs(col["solid"]["volume"] / column_volume() - 1) < 0.005
    assert col["fit"]["median"] < 0.0005 and col["fit"]["points"] > 1000
    stl = elements.solid_mesh(ch, col)
    centre = (stl.bounds[0] + stl.bounds[1]) / 2 + elements.origin(ch)
    assert np.hypot(*(centre[:2] - COLUMN_AT[:2])) < 0.001  # the STL in the export frame, the origin round
    assert np.allclose(elements.origin(ch), np.floor(elements.origin(ch)))
    # a part on the same axis
    top = elements.column(ch, polygon=_square(COLUMN_AT, 0.6), zmin=COLUMN_AT[2] + 2.0, zmax=COLUMN_AT[2] + 3.1,
                          same_axis="Colonna A", knots=6)
    assert top["params"]["axis_direction"] == p["axis_direction"] and abs(top["params"]["height"] - 1.0) < 0.05
    # edited radii: the measured profile stays, the solid changes
    radii = [r * 1.01 for r in col["params"]["knot_radii"]]
    measured, volume = list(col["profile"]["radii"]), col["solid"]["volume"]
    edited = elements.edit_column(ch, col["id"], knot_radii=radii)
    assert edited["profile"]["radii"] == measured and edited["solid"]["volume"] > volume * 1.015
    assert edited["fit"]["median"] > 0.002  # the solid is 1 % fatter than the model: the fit says it
    # the arch: on a vertical plane through two plan points, the pick guessed
    p1, p2 = ARCH_AT[:2] + [-2.0, -0.4], ARCH_AT[:2] + [2.0, -0.4]
    po, u, vv, _w = elements.plane_of(ch, None, p1, p2)
    v, f = elements.local_mesh(ch)
    pick = engine.guess_pick(engine.mesh_section(v, f, po, u, vv))
    a = elements.arch(ch, pick, p1=p1, p2=p2, name="Arco")
    ap, at = a["params"], truth["arch"]
    assert ap["type"] == "pointed" and abs(ap["span"] - at["span"]) < 0.002 and abs(ap["rise"] - at["rise"]) < 0.002
    assert abs(ap["thickness"] - at["thickness"]) < 0.002 and abs(ap["depth"] - at["depth"]) < 0.02
    assert abs(ap["springing"] - (ARCH_AT[2] + 1.5)) < 0.002  # v is the world height
    assert a["fit"]["median"] < 0.002
    # the cornice along the L at its height, projecting to the right of the path
    path = L_PATH[:, :2] + CORNICE_AT[:2]
    m = elements.moulding(ch, path, height=CORNICE_AT[2] + 2.0, station=1.5, window=(-0.6, -0.2, 0.6, 0.5), out=-1,
                          depth=0.1, simplify=0.0)
    assert m["params"]["method"] == "sweep"
    assert abs(m["params"]["section_area"] - abs(engine.polygon_area(truth["plug"]))) < 1e-5
    assert abs(m["solid"]["volume"] / truth["cornice_volume"] - 1) < 0.01
    assert m["fit"]["median"] < 0.001
    # the model, its transform and its products never touched; files only in <chunk>/architecture
    assert _digest(model) == before and np.array_equal(ch.transform, transform)
    made = {q.relative_to(ch.data_dir).parts[0] for q in ch.data_dir.rglob("*") if q.is_file()}
    assert made == {"model.ply", "architecture"}
    for e in (col, a, m):
        d = elements.element_dir(ch, e)
        assert {q.name for q in d.iterdir()} == {"solid", "model.ply", "element.json"}
        record = json.loads((d / "element.json").read_text())
        assert record["origin"] == elements.origin(ch).tolist() and record["fit"] == elements.entry(ch, e["id"])["fit"]
    doc.save()
    again = ppg.Document.open(doc.path).chunk
    assert [e["name"] for e in elements.elements(again)] == ["Colonna A", "Colonna 1", "Arco", "Modanatura 1"]
    assert not set(FORBIDDEN) & set(sys.modules)


def test_ifc_round_trip_and_dxf(tmp_path, fake):
    pytest.importorskip("ifcopenshell")
    from ppg_architecture import drawings, ifc

    doc, ch, _truth = _project(tmp_path)
    col = elements.column(ch, polygon=_square(COLUMN_AT, 0.6), knots=10, name="Colonna")
    p1, p2 = ARCH_AT[:2] + [-2.0, -0.4], ARCH_AT[:2] + [2.0, -0.4]
    po, u, vv, _w = elements.plane_of(ch, None, p1, p2)
    v, f = elements.local_mesh(ch)
    arch = elements.arch(ch, engine.guess_pick(engine.mesh_section(v, f, po, u, vv)), p1=p1, p2=p2, vault=True,
                         name="Volta")
    mould = elements.moulding(ch, L_PATH[:, :2] + CORNICE_AT[:2], height=CORNICE_AT[2] + 2.0, station=1.5,
                              window=(-0.6, -0.2, 0.6, 0.5), out=-1, name="Cornice")
    straight = elements.moulding(ch, L_PATH[:2, :2] + CORNICE_AT[:2], height=CORNICE_AT[2] + 2.0, station=1.5,
                                 window=(-0.6, -0.2, 0.6, 0.5), out=-1, name="Cornice dritta")
    import locale

    try:  # as a window leaves it: Qt sets the system's numbers (a decimal comma); the IFC must not care
        locale.setlocale(locale.LC_NUMERIC, "it_IT.UTF-8")
    except locale.Error:
        pass
    try:
        out = ifc.export(ch, tmp_path / "out" / "elementi.ifc", with_mesh=True)
        volumes = ifc.read_volumes(out)
    finally:
        locale.setlocale(locale.LC_NUMERIC, "C")
    for e in (col, arch, mould, straight):
        vol, bbox = volumes[e["name"]]
        assert abs(vol / e["solid"]["volume"] - 1) < 0.01, e["name"]
        np.testing.assert_allclose(bbox, e["solid"]["bbox"], atol=0.01, err_msg=e["name"])
    import ifcopenshell

    model = ifcopenshell.open(str(out))
    assert model.by_type("IfcColumn")[0].Representation.Representations[0].Items[0].is_a("IfcRevolvedAreaSolid")
    proxy = model.by_type("IfcBuildingElementProxy")[0]
    assert proxy.ObjectType == "Vault"
    item = proxy.Representation.Representations[0].Items[0]
    assert item.is_a("IfcExtrudedAreaSolid") and any(s.is_a("IfcArcIndex") for s in item.SweptArea.OuterCurve.Segments)
    members = {m.Name: m.Representation.Representations[0].Items[0] for m in model.by_type("IfcMember")}
    assert members["Cornice"].is_a("IfcFixedReferenceSweptAreaSolid")
    assert members["Cornice dritta"].is_a("IfcExtrudedAreaSolid")
    conv = model.by_type("IfcMapConversion")[0]
    assert (conv.Eastings, conv.Northings, conv.OrthogonalHeight) == tuple(elements.origin(ch))
    assert model.by_type("IfcProjectedCRS")[0].Name == "EPSG:32633"
    assert len(model.by_type("IfcGeographicElement")) == 1
    import ifcopenshell.util.element

    pset = ifcopenshell.util.element.get_psets(model.by_type("IfcColumn")[0])["Pset_PyArchInit_Element"]
    assert pset["Kind"] == "column" and pset["Fit_median_m"] == pytest.approx(col["fit"]["median"])
    assert pset["P_radius_base"] == pytest.approx(col["params"]["radius_base"])
    dxf = drawings.write_dxf(arch, tmp_path / "out" / "volta.dxf")
    import ezdxf

    layers = {e.dxf.layer for e in ezdxf.readfile(dxf).modelspace()}
    assert {"SOLIDO", "MISURATO", "IMPOSTA", "ASSE"} <= layers
    steps = elements.export_step(ch, None, tmp_path / "out" / "step")
    assert sorted(q.name for q in steps) == ["Colonna.step", "Cornice dritta.step", "Cornice.step", "Volta.step"]


def test_a_failed_solid_changes_nothing(tmp_path, fake, monkeypatch):
    import copy

    from ppg_architecture import runner

    doc, ch, _truth = _project(tmp_path)
    col = elements.column(ch, polygon=_square(COLUMN_AT, 0.6), name="Colonna")
    p1, p2 = ARCH_AT[:2] + [-2.0, -0.4], ARCH_AT[:2] + [2.0, -0.4]
    arch = elements.arch(ch, (2.0, ARCH_AT[2] + 0.9), p1=p1, p2=p2, name="Arco")
    files = sorted(q for q in elements.element_dir(ch, col).rglob("*"))
    before = copy.deepcopy(elements.elements(ch))
    monkeypatch.setenv("PPG_ARCHITECTURE_FAKE", "fail")
    with pytest.raises(runner.WorkerError):
        elements.edit_column(ch, col["id"], knot_radii=[r * 1.05 for r in col["params"]["knot_radii"]])
    with pytest.raises(runner.WorkerError):
        elements.edit_column(ch, col["id"], knots=6)
    with pytest.raises(runner.WorkerError):
        elements.edit_arch(ch, arch["id"], kind="elliptical")
    assert sorted(q for q in elements.element_dir(ch, col).rglob("*")) == files  # the old solid stays
    assert elements.elements(ch) == before  # and so do the entries: parameters, geometry, solid agree
    with pytest.raises(runner.WorkerError):
        elements.column(ch, polygon=_square(COLUMN_AT, 0.6), name="Altra")
    assert [e["name"] for e in elements.elements(ch)] == ["Colonna", "Arco"]
    assert sorted(q.name for q in elements.folder(ch).iterdir()) == sorted([col["id"], arch["id"]])  # no orphan
    monkeypatch.setenv("PPG_ARCHITECTURE_FAKE", "")
    # a path with a vertex repeated: no segment of zero length reaches the engine
    path = L_PATH[:, :2] + CORNICE_AT[:2]
    m = elements.moulding(ch, np.vstack([path[:2], path[1:]]), height=CORNICE_AT[2] + 2.0, station=1.5,
                          window=(-0.6, -0.2, 0.6, 0.5), out=-1, name="Ripetuto")
    assert m["params"]["path_vertices"] == 3


def test_an_edit_keeps_the_type_and_the_step_carries_its_origin(tmp_path, fake):
    doc, ch, _truth = _project(tmp_path)
    p1, p2 = ARCH_AT[:2] + [-2.0, -0.4], ARCH_AT[:2] + [2.0, -0.4]
    a = elements.arch(ch, (2.0, ARCH_AT[2] + 0.9), p1=p1, p2=p2, name="Arco")
    assert a["params"]["type"] == "pointed" and "ellipse" in a["params"]["candidates_rms"]
    assert "rms" in elements.summary(a) and "cerchio" in elements.summary(a)
    assert a["fit"]["excluded_faces"] > 0 and a["fit"]["points"] > 0  # the plane faces are not the ring
    semi = elements.edit_arch(ch, a["id"], kind="semicircular")
    assert semi["params"]["type"] == "semicircular"
    thicker = elements.edit_arch(ch, a["id"], thickness=0.5)  # only the thickness: the type stays
    assert thicker["params"]["type"] == "semicircular" and thicker["params"]["thickness"] == 0.5
    apex = a["params"]["world_apex"]
    assert len(apex) == 3 and abs(apex[2] - (ARCH_AT[2] + 1.5 + np.sqrt(3.0))) < 0.01
    out = elements.export_step(ch, [a["id"]], tmp_path / "out" / "arco.step")
    side = json.loads((tmp_path / "out" / "arco.origin.json").read_text())
    assert out[0].exists() and side["origin"] == elements.origin(ch).tolist() and side["units"] == "metres"


def test_a_changed_system_is_refused_until_the_elements_are_rebuilt(tmp_path, fake):
    from pyarchinit_photogrammetry.core.geometry import compose_similarity
    from ppg_architecture import ifc

    doc, ch, _truth = _project(tmp_path)
    col = elements.column(ch, polygon=_square(COLUMN_AT, 0.6), knots=8, name="Colonna")
    # a part on the same axis, referred to by id as the dialog does, and radii edited by hand
    top = elements.column(ch, polygon=_square(COLUMN_AT, 0.6), zmin=COLUMN_AT[2] + 2.0, same_axis=col["id"],
                          knots=6, name="Capitello")
    col = elements.edit_column(ch, col["id"], knot_radii=[r * 1.02 for r in col["params"]["knot_radii"]])
    p1, p2 = ARCH_AT[:2] + [-2.0, -0.4], ARCH_AT[:2] + [2.0, -0.4]
    elements.arch(ch, (2.0, ARCH_AT[2] + 0.9), p1=p1, p2=p2, name="Arco")
    old_axis = np.asarray(col["params"]["axis_point"])
    before = {e["id"]: e["solid"]["volume"] for e in elements.elements(ch)}
    # the chunk georeferenced again: everything moves 10 m east and turns 90 degrees about the vertical
    rz = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    move = compose_similarity(1.0, rz, [10.0, 0.0, 0.0])
    move[:3, 3] += WORLD - rz @ WORLD  # about the middle of the scene
    ch.transform = move @ ch.transform
    refused = [lambda: elements.export_step(ch, None, tmp_path / "s"), lambda: elements.edit_column(ch, col["id"], knots=6),
               lambda: elements.column(ch, polygon=_square(COLUMN_AT, 0.6))]
    if HAS_IFC:  # without ifcopenshell the IFC export says so before anything else
        refused.append(lambda: ifc.export(ch, tmp_path / "x.ifc"))
    for call in refused:
        with pytest.raises(elements.NotReady, match="sistema del chunk è cambiato"):
            call()
    made = elements.rebuild(ch)
    assert [e["name"] for e in made] == ["Colonna", "Capitello", "Arco"] and elements.frame_matches(ch)
    assert [e["id"] for e in made] == list(before)  # in place: the ids, and the same-axis reference, hold
    assert elements.entry(ch, "Capitello")["source"]["same_axis"] == col["id"]
    for e in made:  # the edited radii carried: the volumes are those before the change of system
        assert abs(e["solid"]["volume"] / before[e["id"]] - 1) < 0.001, e["name"]
    rebuilt = elements.entry(ch, col["id"])
    assert rebuilt["edited"] == ["knot_radii"] and "modificati a mano mantenuti" in elements.summary(rebuilt)
    del top
    new_axis = np.asarray(elements.entry(ch, "Colonna")["params"]["axis_point"])
    np.testing.assert_allclose(new_axis, engine.apply(move, old_axis[None])[0], atol=0.002)
    assert elements.entry(ch, "Arco")["params"]["type"] == "pointed"
    assert {q.name for q in elements.folder(ch).iterdir()} == {e["id"] for e in made}  # no backup left
    if HAS_IFC:
        ifc.export(ch, tmp_path / "x.ifc")  # allowed again


def test_a_tree_click_after_a_system_change_says_why(tmp_path, fake, monkeypatch):
    from ppg_architecture import ArchitecturePlugin
    from pyarchinit_photogrammetry.plugins import Context

    doc, ch, _truth = _project(tmp_path)
    elements.column(ch, polygon=_square(COLUMN_AT, 0.6), knots=6, name="Colonna")
    plugin = ArchitecturePlugin()
    plugin.activate(Context("architecture", tmp_path / "plugin"))
    monkeypatch.setattr(plugin, "_chunk", lambda: ch)
    said = []
    monkeypatch.setattr(plugin, "_message", lambda kind, text, details="": said.append(text))
    moved = ch.transform.copy()
    moved[0, 3] += 5.0
    ch.transform = moved
    plugin._tree_items(ch)[0]["children"][0]["activate"]()
    assert said and "sistema del chunk è cambiato" in said[0]


def _plugin_folder(root: Path) -> Path:
    folder = root / "plugins"
    dist = folder / "architecture" / "ppg_architecture-0.1.0.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: ppg-architecture\nVersion: 0.1.0\n")
    (dist / "entry_points.txt").write_text("[pyarchinit_photogrammetry.plugins]\narchitecture = "
                                           "ppg_architecture:ArchitecturePlugin\n")
    return folder


def test_command_line(tmp_path, fake, monkeypatch):
    from typer.testing import CliRunner

    from pyarchinit_photogrammetry.cli import app

    doc, ch, _truth = _project(tmp_path)
    monkeypatch.setenv("PPG_PLUGINS_DIR", str(_plugin_folder(tmp_path)))
    cli = CliRunner()
    project = str(doc.path)
    sq = _square(COLUMN_AT, 0.6)
    wkt = "POLYGON((" + ", ".join(f"{x} {y}" for x, y in np.vstack([sq, sq[:1]])) + "))"
    r = cli.invoke(app, ["plugin", "architecture", "column", project, "--polygon", wkt, "--knots", "8", "--name", "C1"])
    assert r.exit_code == 0, r.output
    assert "Colonna: h 3,000 m" in r.output and "scarto mediana" in r.output
    line = "LINESTRING(" + ", ".join(f"{x} {y}" for x, y in L_PATH[:, :2] + CORNICE_AT[:2]) + ")"
    r = cli.invoke(app, ["plugin", "architecture", "moulding", project, "--path", line, "--height",
                         str(CORNICE_AT[2] + 2.0), "--station", "1.5", "--window", "-0.6,-0.2,0.6,0.5",
                         "--out-side", "-1", "--name", "M1"])
    assert r.exit_code == 0, r.output
    x1, y1 = ARCH_AT[:2] + [-2.0, -0.4]
    x2, y2 = ARCH_AT[:2] + [2.0, -0.4]
    r = cli.invoke(app, ["plugin", "architecture", "arch", project, "--points", f"{x1},{y1},{x2},{y2}", "--name", "A1"])
    assert r.exit_code == 0, r.output
    assert "a sesto acuto" in r.output
    ch = ppg.Document.open(project).chunk
    a, b = (CORNICE_AT + [0.0, 0.0, 2.0], CORNICE_AT + [4.0, 0.0, 2.0])
    ch.add_marker("A", ch.to_internal(np.asarray([a]))[0])
    ch.add_marker("B", ch.to_internal(np.asarray([b]))[0])
    ch.document.save()
    r = cli.invoke(app, ["plugin", "architecture", "moulding", project, "--markers", "A,B", "--height",
                         str(CORNICE_AT[2] + 2.0), "--station", "1.5", "--window", "-0.6,-0.2,0.6,0.5",
                         "--out-side", "-1", "--name", "M2"])
    assert r.exit_code == 0, r.output
    m2 = elements.need(ppg.Document.open(project).chunk, "M2")
    assert m2["params"]["method"] == "extrusion" and abs(m2["params"]["length"] - 4.0) < 1e-6
    r = cli.invoke(app, ["plugin", "architecture", "remove", project, "M2"])
    r = cli.invoke(app, ["plugin", "architecture", "list", project])
    assert r.exit_code == 0 and r.output.count("\n") >= 3 and "C1" in r.output and "M1" in r.output
    r = cli.invoke(app, ["plugin", "architecture", "show", project, "A1"])
    assert json.loads(r.output[r.output.index("{"):])["params"]["type"] == "pointed"
    out = tmp_path / "e.ifc"
    r = cli.invoke(app, ["plugin", "architecture", "export", project, "--out", str(out)])
    if HAS_IFC:
        assert r.exit_code == 0 and out.exists(), r.output
    else:  # the program without its bim extra: the command says what to install
        assert r.exit_code != 0 and "ifcopenshell" in r.output, r.output
    r = cli.invoke(app, ["plugin", "architecture", "export", project, "--out", str(tmp_path / "a.dxf"), "--ids", "A1"])
    assert r.exit_code == 0 and (tmp_path / "a.dxf").exists(), r.output
    r = cli.invoke(app, ["plugin", "architecture", "remove", project, "M1"])
    assert r.exit_code == 0
    assert [e["name"] for e in elements.elements(ppg.Document.open(project).chunk)] == ["C1", "A1"]
    r = cli.invoke(app, ["plugin", "architecture", "status"])
    assert r.exit_code == 0 and "CadQuery" in r.output


def test_the_program_never_imports_cadquery(tmp_path):
    """In a process of its own: the plugin imported with everything it has, the pane built, a solid through the fake
    worker - and none of CadQuery, OpenCascade, VTK, casadi in ``sys.modules``."""
    code = f"""
import os, sys
sys.path[:0] = [{str(Path(__file__).resolve().parent.parent)!r}, {str(Path(__file__).resolve().parent)!r}]
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import numpy as np
import ppg_architecture, ppg_architecture.elements as el, ppg_architecture.ifc, ppg_architecture.drawings
import ppg_architecture.runner, ppg_architecture.environment, ppg_architecture.engine as engine
from PySide6.QtWidgets import QApplication
QApplication.instance() or QApplication([])
from ppg_architecture.pane import ArchitecturePane
ArchitecturePane()
from pyarchinit_photogrammetry.core.progress import Progress
job = {{"protocol": 1, "kind": "arch", "output": {str(tmp_path / 'solid')!r},
       "arch": {{"outline": {{"points": [[1, 0], [1.4, 0], [0, 1.4], [-1.4, 0], [-1, 0], [0, 1]],
                              "segments": [["L", 0, 1], ["A", 1, 2, 3], ["L", 3, 4], ["A", 4, 5, 0]]}}, "depth": 0.5}}}}
r = ppg_architecture.runner.run(job, {str(tmp_path / 'work')!r}, python=sys.executable, script={str(FAKE)!r})
assert abs(r["volume"] - np.pi / 2 * (1.96 - 1) * 0.5) < 0.002, r["volume"]
print(sorted(m for m in sys.modules if m.split(".")[0] in {FORBIDDEN!r}))
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-3000:]
    assert out.stdout.strip().splitlines()[-1] == "[]"


def test_the_pane_in_the_window(tmp_path, fake, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from ppg_architecture import ArchitecturePlugin
    from pyarchinit_photogrammetry.app import app
    from pyarchinit_photogrammetry.plugins import Context

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "settings"))
    QApplication.instance() or QApplication([])
    from pyarchinit_photogrammetry.gui import dialogs
    from pyarchinit_photogrammetry.gui import main_window as mwmod

    doc, ch, _truth = _project(tmp_path)
    app.document = doc
    w = mwmod.MainWindow(doc, autoconnect=False, check_updates=False)
    w.runner.show_dialog = False
    plugin = ArchitecturePlugin()
    try:
        plugin.activate(Context("architecture", tmp_path / "plugin", window=w))
        pane = plugin.pane
        dock = w.panes.groups["right"].dock(pane)
        assert dock is not None and dock.objectName().startswith("pane_plugin_architecture")
        assert pane.table.rowCount() == 0 and "Nessun elemento" in pane.l_state.text()
        assert pane.minimumSizeHint().width() <= 320
        plugin.column(ch, polygon=_square(COLUMN_AT, 0.6), knots=8, name="Colonna")
        w.runner.wait()
        assert pane.table.rowCount() == 1 and pane.table.item(0, 0).text() == "Colonna"
        assert pane.table.item(0, 3).text().endswith("mm") and w.isWindowModified()
        items = plugin._tree_items(ch)
        assert items[0]["label"] == "Elementi architettonici"
        assert items[0]["children"][0]["label"].startswith("Colonna (")
        # the dialogs, answered with their defaults: a column, a moulding, an arch on a plane through two points
        asked = []

        def answer(title, fields, intro=""):
            asked.append(title)
            return {f.name: f.default for f in fields if f.kind != "heading"}

        monkeypatch.setattr(plugin, "_ask", answer)
        plugin._column_dialog(_square(COLUMN_AT, 0.6))
        w.runner.wait()
        plugin._moulding_dialog(L_PATH[:, :2] + CORNICE_AT[:2])
        w.runner.wait()
        p1, p2 = ARCH_AT[:2] + [-2.0, -0.4], ARCH_AT[:2] + [2.0, -0.4]
        plugin._arch_dialog(ch, None, np.array([p1, p2]))
        w.runner.wait()
        kinds = [e["kind"] for e in elements.elements(ch)]
        assert kinds == ["column", "column", "moulding", "arch"], asked
        assert elements.elements(ch)[3]["params"]["type"] == "pointed" and pane.table.rowCount() == 4
        for row in range(1, 4):  # every kind's parameters, edited with the values they have
            pane.table.clearSelection()
            pane.table.selectRow(row)
            plugin._do_edit()
            w.runner.wait()
        assert asked[-3:] == ["Parametri della colonna", "Parametri della modanatura", "Parametri dell'arco"]
        for e in elements.elements(ch)[1:]:
            elements.remove(ch, e["id"])
        plugin.refresh_pane()
        pane.table.selectRow(0)
        plugin._do_show()
        monkeypatch.setattr(dialogs, "choice", lambda *a, **k: None)
        plugin._do_remove()
        assert len(elements.elements(ch)) == 1
        monkeypatch.setattr(dialogs, "choice", lambda *a, **k: 0)
        plugin._do_remove()
        assert elements.elements(ch) == [] and pane.table.rowCount() == 0 and plugin._tree_items(ch) == []
        plugin.deactivate()
        plugin.context.release()
        assert w.panes.dock(pane) is None
        assert not set(FORBIDDEN) & set(sys.modules)
    finally:
        w.setWindowModified(False)
        w.close()


@pytest.mark.skipif(CADQUERY is None, reason="no Python with CadQuery (PPG_TEST_CADQUERY)")
def test_elements_and_their_ifc_through_cadquery(tmp_path, monkeypatch):
    """The real worker: the solids of CadQuery against the truth, and the IFC read back within 1 % of their volume."""
    pytest.importorskip("ifcopenshell")
    from ppg_architecture import ifc, runner

    monkeypatch.setattr(elements, "WORKER", (CADQUERY, Path(runner.__file__).with_name("worker.py")))
    doc, ch, truth = _project(tmp_path)
    col = elements.column(ch, polygon=_square(COLUMN_AT, 0.6), knots=12, name="Colonna")
    assert abs(col["solid"]["volume"] / column_volume() - 1) < 0.005 and col["fit"]["median"] < 0.0005
    p1, p2 = ARCH_AT[:2] + [-2.0, -0.4], ARCH_AT[:2] + [2.0, -0.4]
    po, u, vv, _w = elements.plane_of(ch, None, p1, p2)
    v, f = elements.local_mesh(ch)
    arch = elements.arch(ch, engine.guess_pick(engine.mesh_section(v, f, po, u, vv)), p1=p1, p2=p2, name="Arco")
    assert arch["params"]["type"] == "pointed" and arch["fit"]["median"] < 0.002
    mould = elements.moulding(ch, L_PATH[:, :2] + CORNICE_AT[:2], height=CORNICE_AT[2] + 2.0, station=1.5,
                              window=(-0.6, -0.2, 0.6, 0.5), out=-1, name="Cornice")
    assert abs(mould["solid"]["volume"] / truth["cornice_volume"] - 1) < 0.01 and mould["fit"]["median"] < 0.001
    out = ifc.export(ch, tmp_path / "elementi.ifc")
    volumes = ifc.read_volumes(out)
    for e in (col, arch, mould):
        vol, bbox = volumes[e["name"]]
        assert abs(vol / e["solid"]["volume"] - 1) < 0.01, (e["name"], vol, e["solid"]["volume"])
        np.testing.assert_allclose(bbox, e["solid"]["bbox"], atol=0.01, err_msg=e["name"])
    assert not set(FORBIDDEN) & set(sys.modules)


def test_a_moulding_traced_on_an_elevation_and_an_arch_clicked_on_a_section(tmp_path, fake):
    doc, ch, truth = _project(tmp_path)
    # the cornice's first leg seen from -y: u along x from the corner of the L, v the height
    front = _elevation(ch, "Prospetto sud", "elevation", CORNICE_AT + [0.0, -2.0, -CORNICE_AT[2]], [1, 0, 0], [0, 0, 1])
    uv = np.array([[0.2, CORNICE_AT[2] + 2.1], [3.6, CORNICE_AT[2] + 2.1]])
    world = elements.from_elevation(ch, "Prospetto sud", uv)
    assert np.allclose(world[:, 0], CORNICE_AT[0] + uv[:, 0]) and np.allclose(world[:, 2], uv[:, 1])
    # no relief value: the depth from the model, the surface nearest the viewer (the cornice's front at that height)
    assert np.all(world[:, 1] < CORNICE_AT[1] - 0.05) and np.all(world[:, 1] > CORNICE_AT[1] - 0.2)
    m = elements.moulding(ch, world, station=1.0, window=(-0.6, -0.4, 0.6, 0.5), out=-1, name="Dal prospetto")
    assert abs(m["params"]["section_area"] - abs(engine.polygon_area(truth["plug"]))) < 1e-4
    assert m["params"]["method"] == "extrusion" and m["fit"]["median"] < 0.001
    assert m["fit"]["excluded_behind"] > 0  # the wall's back, behind the back line of the section, is not in the fit
    # a relief with values: the depth from the elevation itself
    _elevation(ch, "Con rilievo", "elevation", [0.0, 0.0, 0.0], [1, 0, 0], [0, 0, 1], d=0.5)
    np.testing.assert_allclose(elements.from_elevation(ch, "Con rilievo", [[1.0, 2.0]]), [[1.0, -0.5, 2.0]], atol=1e-9)
    assert front.normal @ [0, -1, 0] > 0.99
    # an arch read on a section of the project, the point clicked on it
    _elevation(ch, "Sezione arco", "section", ARCH_AT + [0.0, -0.4, -ARCH_AT[2]], [1, 0, 0], [0, 0, 1])
    assert [s["id"] for s in elements.section_planes(ch)] == ["Sezione arco"]
    a = elements.arch(ch, (0.0, ARCH_AT[2] + 0.9), section="Sezione arco", name="Arco cliccato")
    assert a["params"]["type"] == "pointed" and abs(a["params"]["span"] - truth["arch"]["span"]) < 0.002


def test_the_gui_draws_on_elevations(tmp_path, fake, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from ppg_architecture import ArchitecturePlugin
    from pyarchinit_photogrammetry.app import app
    from pyarchinit_photogrammetry.plugins import Context

    QApplication.instance() or QApplication([])
    from pyarchinit_photogrammetry.gui import main_window as mwmod

    doc, ch, truth = _project(tmp_path)
    _elevation(ch, "Prospetto sud", "elevation", CORNICE_AT + [0.0, -2.0, -CORNICE_AT[2]], [1, 0, 0], [0, 0, 1])
    _elevation(ch, "Sezione arco", "section", ARCH_AT + [0.0, -0.4, -ARCH_AT[2]], [1, 0, 0], [0, 0, 1])
    app.document = doc
    w = mwmod.MainWindow(doc, autoconnect=False, check_updates=False)
    w.runner.show_dialog = False
    plugin = ArchitecturePlugin()
    try:
        plugin.activate(Context("architecture", tmp_path / "plugin", window=w))
        shown = []
        monkeypatch.setattr(plugin.context, "show_raster",
                            lambda kind=None, chunk=None, path=None: shown.append(path) or True)
        monkeypatch.setattr(plugin.context, "request_line",
                            lambda done, hint="": done(np.array([[0.2, CORNICE_AT[2] + 2.1],
                                                                 [3.6, CORNICE_AT[2] + 2.1]])))
        monkeypatch.setattr(plugin.context, "request_point",
                            lambda done, hint="": done(np.array([[0.0, ARCH_AT[2] + 0.9]])))
        answers = {"Percorso della modanatura": {"how": "elevation", "view": "Prospetto sud"},
                   "Arco / volta": {"section": "Sezione arco", "pick": "click"}}
        seen = []

        def answer(title, fields, intro=""):
            values = {f.name: f.default for f in fields if f.kind != "heading"}
            if title in answers and title not in seen:
                seen.append(title)
                values.update(answers[title])
            if title == "Modanatura":
                assert "height" not in values  # a path drawn on an elevation keeps its heights
                values.update(side="right", below=0.4, reach=0.6)
            if title == "Arco / volta" and "pick_u" in values:
                assert (values["pick_u"], values["pick_v"]) == (0.0, ARCH_AT[2] + 0.9)  # the click, as typed values
            return values

        monkeypatch.setattr(plugin, "_ask", answer)
        plugin._do_moulding()
        w.runner.wait()
        plugin._do_arch()
        w.runner.wait()
        # after an elevation was shown, a column's region is asked on the chunk's own plan, not on the elevation
        asked = []
        monkeypatch.setattr(plugin.context, "request_polygon", lambda done, hint="": asked.append(list(shown)))
        plugin._do_column()
        assert asked and asked[0][-1] is None  # show_raster(None): the chunk's orthophoto or DEM
        kinds = {e["name"]: e for e in elements.elements(ch)}
        assert [p.name for p in shown[:2]] == ["elevation.tif", "elevation.tif"]
        m = kinds["Modanatura 1"]
        assert abs(m["params"]["section_area"] - abs(engine.polygon_area(truth["plug"]))) < 1e-4
        assert kinds["Arco 1"]["params"]["type"] == "pointed"
        assert "circa un minuto" in plugin.pane.l_env.text() or "non è installato" not in plugin.pane.l_env.text()
        plugin.deactivate()
        plugin.context.release()
    finally:
        w.setWindowModified(False)
        w.close()
