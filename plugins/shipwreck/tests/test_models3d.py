"""The 3D files of the traced timbers: OBJ (one object per timber, a material per kind, hypotheses apart), GLB (one
node per timber, a colour per kind, hypotheses translucent), IFC (one element per timber with
``Pset_PyArchInit_Timber`` and stable GlobalIds), the tables with volume, thickness and status; in the ship frame
(right-handed for both bows) or in the chunk's coordinates."""
from __future__ import annotations

import copy
import csv

import numpy as np
import pytest
import trimesh
from hulls import framed_project

from ppg_shipwreck import models3d, relief, ship, solids, timbers, tracing
from ppg_shipwreck import proposals as proposals_mod
from ppg_shipwreck.ship import NotReady

X_RANGE = (3.0, 9.0)


@pytest.fixture(scope="module", params=["start", "end"])
def project(request, tmp_path_factory):
    _doc, ch, ends = framed_project(tmp_path_factory.mktemp(f"models3d_{request.param}"), heading_deg=35,
                                    x_range=X_RANGE)
    ship.set_frame(ch, ship.Frame(ends[0], ends[1], request.param))
    relief.build(ch, cell=0.01)  # coarser than the solids' tests: the files, not the volumes
    proposals_mod.build(ch)
    tracing.accept(ch, [i["id"] for i in ship.state(ch)["proposals"]["items"]])
    solids.set_keel(ch, 0.12, 0.15)
    out = solids.build(ch)
    assert out["built"] and out["keel"] is not None
    return ch, copy.deepcopy(ch.meta[ship.KEY])


@pytest.fixture
def hull(project):
    ch, snapshot = project
    ch.meta[ship.KEY] = copy.deepcopy(snapshot)
    return ch


def _built(ch):
    return [r for r in timbers.timbers_of(ch) if r.get("solid")]


def _obj_objects(path):
    """{object name: (material, vertex count)} and all the vertices of an OBJ."""
    objects, current, verts = {}, None, []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("o "):
            current = line[2:].strip()
            objects[current] = [None, 0]
        elif line.startswith("usemtl "):
            objects[current][0] = line[7:].strip()
        elif line.startswith("v "):
            verts.append([float(v) for v in line.split()[1:4]])
            objects[current][1] += 1
    return {k: tuple(v) for k, v in objects.items()}, np.asarray(verts)


def _mtl(path):
    out, current = {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "newmtl":
            current = parts[1]
            out[current] = {}
        elif current and parts[0] in ("Kd", "d"):
            out[current][parts[0]] = [float(v) for v in parts[1:]]
    return out


def test_colours_and_lighter():
    assert set(models3d.KIND_COLOURS) == {"keel", "keelson", "frame", "plank", "ceiling", "stringer", "other"}
    assert models3d.KIND_COLOURS["plank"] == "#2e86c1"
    assert models3d.lighter("#000000") == "#808080"
    assert models3d.lighter("#ffffff") == "#ffffff"
    assert models3d.RIC == "_ric"


def test_items_right_handed_and_hypotheses_apart(hull):
    ch = hull
    for system in ("ship", "world"):
        items = models3d.items(ch, system)
        assert len(items) == len(_built(ch)) + 1  # + the typed keel
        names = [i["name"] for i in items]
        assert len(set(names)) == len(names)
        for i in items:
            v, f = i["vertices"] - i["vertices"].mean(axis=0), i["faces"]  # centred: world coordinates are large
            assert solids._signed_volume(v, f) > 0, (system, i["name"])  # faces outward in a right-handed frame
            assert trimesh.Trimesh(v, f, process=False).is_watertight
            assert i["name"].endswith(models3d.RIC) == (i["status"] != "measured")
        keel = next(i for i in items if i["id"] == "keel")
        assert keel["status"] == "reconstructed:given" and keel["name"].endswith("_ric")
        assert all(not i["name"].endswith("_ric") for i in items if i["id"] != "keel")


def test_ship_system_is_x_signed_y_z(hull):
    ch = hull
    frame = ship.frame_of(ch)
    xyz = np.array([[1.0, 0.5, -0.2], [2.0, -0.3, 0.1]])
    out = models3d.coords(ch, xyz, "ship")
    assert np.allclose(out, np.column_stack([xyz[:, 0], frame.sign * xyz[:, 1], xyz[:, 2]]))
    assert np.allclose(models3d.coords(ch, xyz, "world"), frame.to_world(xyz))
    # the right-handed ship frame is a rotation of the world's: distances and the vertical stay
    a, b = models3d.coords(ch, xyz, "ship"), models3d.coords(ch, xyz, "world")
    assert np.linalg.norm(a[0] - a[1]) == pytest.approx(np.linalg.norm(b[0] - b[1]))
    with pytest.raises(ValueError):
        models3d.coords(ch, xyz, "plan")


def test_obj_objects_materials_and_world(hull, tmp_path):
    from pyarchinit_photogrammetry.io.mesh_ply import PlyMesh

    ch = hull
    path = models3d.write_obj(ch, tmp_path / "legni.obj")
    objects, _v = _obj_objects(path)
    built = _built(ch)
    assert len(objects) == len(built) + 1
    assert any(n.endswith("_ric") and m == "keel_ric" for n, (m, _c) in objects.items())
    kinds = {r["kind"] for r in built}
    assert {m for m, _c in objects.values()} == kinds | {"keel_ric"}
    mtl = _mtl(tmp_path / "legni.mtl")
    assert mtl["keel_ric"]["d"] == [0.5]
    for k in kinds:
        assert mtl[k]["d"] == [1.0]
        assert mtl[k]["Kd"] == pytest.approx([int(models3d.KIND_COLOURS[k][i:i + 2], 16) / 255 for i in (1, 3, 5)],
                                             abs=1e-3)
    text = path.read_text(encoding="utf-8")
    assert "mtllib legni.mtl" in text and "# system ship" in text and "ship_frame" in text

    # the chunk's coordinates: a vertex back through the frame lands on the PLY's within 1 mm
    world = models3d.write_obj(ch, tmp_path / "mondo.obj", system="world")
    objects, verts = _obj_objects(world)
    first = built[0]
    mesh = PlyMesh(ch.data_dir / first["solid"])
    ply = mesh.xyz()
    del mesh
    wanted = next(i["name"] for i in models3d.items(ch, "world") if i["id"] == first["id"])
    start = 0
    for name, (_m, count) in objects.items():
        if name == wanted:
            assert count == len(ply)
            break
        start += count
    back = ship.frame_of(ch).to_ship(verts[start:start + len(ply)])
    assert np.abs(back - ply).max() < 1e-3


def test_glb_nodes_colours_translucent(hull, tmp_path):
    ch = hull
    path = models3d.write_glb(ch, tmp_path / "legni.glb")
    scene = trimesh.load(path, force="scene")
    items = models3d.items(ch, "ship")
    names = {i["name"] for i in items}
    assert set(scene.graph.nodes_geometry) == names
    for name in names:
        geom = scene.geometry[scene.graph[name][1]]
        mat = geom.visual.material
        colour = np.asarray(mat.baseColorFactor, float)
        colour = colour / 255.0 if colour.max() > 1.0 else colour
        if name.endswith("_ric"):
            assert mat.alphaMode == "BLEND" and colour[3] == pytest.approx(0.5, abs=0.01)
        else:
            assert mat.alphaMode in (None, "OPAQUE") and colour[3] == pytest.approx(1.0, abs=0.01)
        assert geom.volume > 0  # Y-up, still turned outward
    extras = scene.metadata.get("extras") or {}
    assert extras.get("system") == "ship" and "ship_frame" in extras

    world = models3d.write_glb(ch, tmp_path / "mondo.glb", system="world")
    scene = trimesh.load(world, force="scene")
    extras = scene.metadata.get("extras") or {}
    assert extras["system"] == "world" and extras["crs"] == ch.crs and len(extras["origin"]) == 3


def test_ifc_elements_psets_stable_ids(hull, tmp_path):
    ifcopenshell = pytest.importorskip("ifcopenshell")
    import ifcopenshell.util.element as element

    ch = hull
    out = models3d.write_ifc(ch, tmp_path / "legni.ifc")
    assert out["elements"] == len(_built(ch)) + 1 and out["map_conversion"]
    model = ifcopenshell.open(str(out["path"]))
    elements = model.by_type("IfcPlate") + model.by_type("IfcMember")
    assert len(elements) == out["elements"]
    by_id = {}
    for e in elements:
        pset = element.get_psets(e)["Pset_PyArchInit_Timber"]
        for key in ("Name", "Kind", "Side", "Chunk", "Status", "Length", "Width", "ZMin", "ZMax", "Volume",
                    "HiddenShare", "Sources"):
            assert key in pset, (e.Name, key)
        by_id[pset["Id"]] = (e, pset)
    for r in _built(ch):
        e, pset = by_id[r["id"]]
        assert e.is_a("IfcPlate" if r["kind"] in ("plank", "ceiling") else "IfcMember")
        assert pset["Volume"] == pytest.approx(r["volume"], rel=1e-6)
        assert pset["Status"] == "misurato" and pset["Chunk"] == str(ch.key)
        if r["kind"] == "plank":
            assert pset["Thickness"] == pytest.approx(r["thickness_mm"] / 1000.0)
            assert 0.0 < pset["HiddenShare"] < 1.0 or pset["HiddenShare"] == 0.0
        if e.is_a("IfcMember"):
            assert e.PredefinedType == "USERDEFINED" and e.ObjectType == timbers.KIND_TITLES[r["kind"]][1]
    keel, kpset = by_id["keel"]
    assert keel.Name.endswith("_ric") and kpset["Status"] == "ricostruito: dimensioni date"
    (pdef,) = [r.RelatingPropertyDefinition for r in keel.IsDefinedBy
               if r.is_a("IfcRelDefinesByProperties") and r.RelatingPropertyDefinition.Name == models3d.PSET]
    values = {p.Name: p.NominalValue for p in pdef.HasProperties}
    for key, measure in models3d.MEASURES.items():  # numbers with their unit (the keel has no thickness)
        if key != "Thickness":
            assert values[key].is_a(measure), key
    (conv,) = model.by_type("IfcMapConversion")
    # the ship system's map conversion lands every vertex of every element on the chunk's coordinates
    world = {i["name"]: i["vertices"] for i in models3d.items(ch, "world")}
    a, b = conv.XAxisAbscissa, conv.XAxisOrdinate
    for e in elements:
        local = np.asarray(e.Representation.Representations[0].Items[0].Coordinates.CoordList, float)
        mapped = np.column_stack([conv.Eastings + a * local[:, 0] - b * local[:, 1],
                                  conv.Northings + b * local[:, 0] + a * local[:, 1], conv.OrthogonalHeight + local[:, 2]])
        assert np.abs(mapped - world[e.Name]).max() < 1e-3, e.Name
    # stable GlobalIds: a second export (in the other system) gives the same ones
    again = models3d.write_ifc(ch, tmp_path / "mondo.ifc", system="world")
    other = ifcopenshell.open(str(again["path"]))
    first = sorted(e.GlobalId for e in elements)
    assert sorted(e.GlobalId for e in other.by_type("IfcPlate") + other.by_type("IfcMember")) == first
    assert len(set(first)) == len(first)
    assert first == sorted(models3d.stable_global_id(ch.key, "shipwreck", i) for i in by_id)


@pytest.mark.parametrize("crs_case", ["local", "geographic"])
def test_ifc_without_a_projected_system_has_no_map_conversion(hull, tmp_path, monkeypatch, crs_case):
    ifcopenshell = pytest.importorskip("ifcopenshell")
    from pyarchinit_photogrammetry.core import crs as crs_mod

    whole = models3d.items
    monkeypatch.setattr(models3d, "items", lambda ch, system="ship", reconstruction=True: whole(ch, system, reconstruction)[-2:])  # light: two timbers
    monkeypatch.setattr(crs_mod, "is_local" if crs_case == "local" else "is_geographic", lambda _crs: True)
    for system in ("ship", "world"):
        out = models3d.write_ifc(hull, tmp_path / f"{system}.ifc", system=system)
        assert not out["map_conversion"] and "IfcMapConversion" in out["note"]
        model = ifcopenshell.open(str(out["path"]))
        assert not model.by_type("IfcMapConversion")
        if system == "world":  # nowhere to keep an origin: the chunk's own coordinates
            assert "no origin taken off" in out["note"] or "senza origine sottratta" in out["note"]
            for e in model.by_type("IfcPlate") + model.by_type("IfcMember"):
                local = np.asarray(e.Representation.Representations[0].Items[0].Coordinates.CoordList, float)
                truth = next(i["vertices"] for i in whole(hull, "world") if i["name"] == e.Name)
                assert np.abs(local - truth).max() < 1e-6


def test_tables(hull, tmp_path):
    ch = hull
    out = models3d.write_tables(ch, tmp_path)
    assert out["csv"].exists() and out["xlsx"].exists()
    with open(out["csv"], encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_id = {r["id"]: r for r in rows}
    assert out["rows"] == len(rows) == len(timbers.timbers_of(ch)) + 1
    for r in _built(ch):
        assert float(by_id[r["id"]]["volume"]) == pytest.approx(r["volume"], rel=1e-3)
        assert by_id[r["id"]]["status"] == "measured"
    assert by_id["keel"]["status"] == "reconstructed:given"
    planks = [r for r in _built(ch) if r["kind"] == "plank"]
    assert float(by_id[planks[0]["id"]]["thickness_mm"]) == pytest.approx(planks[0]["thickness_mm"], abs=1e-3)
    from openpyxl import load_workbook

    sheet = load_workbook(out["xlsx"]).active
    header = [c.value for c in sheet[1]]
    assert {"volume", "thickness_mm", "status"} <= set(header)


def test_refused_without_solids_or_with_a_stale_frame(hull, tmp_path):
    ch = hull
    frame = ship.frame_of(ch)
    ship.state(ch).pop("solids")
    with pytest.raises(NotReady):
        models3d.write_obj(ch, tmp_path / "x.obj")
    ship.state(ch)["solids"] = {"count": 1, "frame_key": relief.frame_key(frame.flipped())}
    with pytest.raises(NotReady):
        models3d.items(ch, "ship")
    assert not (tmp_path / "x.obj").exists()


def test_records_and_chunk_untouched(hull, tmp_path):
    ch = hull
    before = copy.deepcopy(ch.meta[ship.KEY])
    transform = np.array(ch.transform)
    models3d.write_obj(ch, tmp_path / "a.obj")
    models3d.write_glb(ch, tmp_path / "a.glb", system="world")
    models3d.write_tables(ch, tmp_path)
    assert ch.meta[ship.KEY] == before
    assert np.array_equal(np.array(ch.transform), transform)


# ----------------------------------------------------------------------------- the reconstruction in the 3D files (0.4.0)
@pytest.fixture(scope="module")
def rebuilt(tmp_path_factory):
    """The project of the reconstruction's tests, built: mirrored, extended and spaced timbers and the faired hull."""
    from test_reconstruct import HOLE, _wreck_project

    from ppg_shipwreck import reconstruct

    ch = _wreck_project(tmp_path_factory.mktemp("rebuilt3d"), port_from=HOLE)
    reconstruct.build(ch)
    return ch


def test_the_reconstruction_translucent_in_the_3d_files(rebuilt, tmp_path):
    ifcopenshell = pytest.importorskip("ifcopenshell")
    import ifcopenshell.util.element as element

    from ppg_shipwreck import reconstruct

    ch = rebuilt
    measured = models3d.items(ch, "ship", reconstruction=False)
    whole = models3d.items(ch, "ship")
    built = [r for r in reconstruct.records_of(ch) if r.get("solid")]
    assert built
    assert len(whole) == len(measured) + len(built) + 1
    assert [e["name"] for e in whole[:len(measured)]] == [e["name"] for e in measured]  # the catalogue's names kept
    added = whole[len(measured):]
    assert [e["id"] for e in added] == [r["id"] for r in built] + ["carenatura"]
    assert all(e["name"].endswith("_ric") and e["status"].startswith("reconstructed:") for e in added)
    loft = added[-1]
    assert loft["name"] == "carenatura_ric" and loft["kind"] == "hull" and loft["status"] == "reconstructed:fairing"
    assert len({e["name"] for e in whole}) == len(whole)
    # GLB: a node per measured timber, reconstructed part and the faired hull; the reconstruction translucent
    scene = trimesh.load(models3d.write_glb(ch, tmp_path / "r.glb"), force="scene")
    assert set(scene.graph.nodes_geometry) == {e["name"] for e in whole}
    for e in added:
        mat = scene.geometry[scene.graph[e["name"]][1]].visual.material
        colour = np.asarray(mat.baseColorFactor, float)
        colour = colour / 255.0 if colour.max() > 1.0 else colour
        assert mat.alphaMode == "BLEND" and colour[3] == pytest.approx(0.5, abs=0.01), e["name"]
    off = trimesh.load(models3d.write_glb(ch, tmp_path / "m.glb", reconstruction=False), force="scene")
    assert set(off.graph.nodes_geometry) == {e["name"] for e in measured}
    # OBJ: the reconstructed objects with their translucent materials
    objects, _v = _obj_objects(models3d.write_obj(ch, tmp_path / "r.obj"))
    assert set(objects) == {e["name"] for e in whole}
    mtl = _mtl(tmp_path / "r.mtl")
    assert all(mtl[objects[e["name"]][0]]["d"] == [0.5] for e in added)
    objects, _v = _obj_objects(models3d.write_obj(ch, tmp_path / "m.obj", reconstruction=False))
    assert set(objects) == {e["name"] for e in measured}
    # IFC: one element per item, the status said, stable GlobalIds
    out = models3d.write_ifc(ch, tmp_path / "r.ifc")
    assert out["elements"] == len(whole)
    model = ifcopenshell.open(str(out["path"]))
    elements = model.by_type("IfcPlate") + model.by_type("IfcMember")
    assert len(elements) == len(whole)
    psets = {element.get_psets(e)[models3d.PSET]["Id"]: (e, element.get_psets(e)[models3d.PSET]) for e in elements}
    mirrored = next(r for r in built if r["status"] == "reconstructed:mirror")
    e, pset = psets[mirrored["id"]]
    assert pset["Status"] == "ricostruito: simmetria" and pset["StatusCode"] == "reconstructed:mirror"
    assert e.GlobalId == models3d.stable_global_id(ch.key, "shipwreck", mirrored["id"])
    e, pset = psets["carenatura"]
    assert e.is_a("IfcMember") and e.ObjectType == "carenatura (ipotesi)"
    assert pset["Status"] == "ricostruito: carenatura" and "Volume" not in pset and "Thickness" not in pset
    closed = {e.Name: e.Representation.Representations[0].Items[0].Closed for e in elements}
    assert closed["carenatura_ric"] is False  # an open surface, never a solid
    assert closed[next(i["name"] for i in added if i["id"] == mirrored["id"])] is True
    assert len({e.GlobalId for e in elements}) == len(elements)
    again = models3d.write_ifc(ch, tmp_path / "w.ifc", system="world")
    other = ifcopenshell.open(str(again["path"]))
    assert sorted(e.GlobalId for e in other.by_type("IfcPlate") + other.by_type("IfcMember")) == sorted(
        e.GlobalId for e in elements)
    assert models3d.write_ifc(ch, tmp_path / "m.ifc", reconstruction=False)["elements"] == len(measured)
