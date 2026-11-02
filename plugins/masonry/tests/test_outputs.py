"""The outputs of a face and of a wall: the plate at scale (SVG, PDF), the DXF with its layers, the whole export, the
IFC wall (one face and a thickness given, or the slab between two faces), validated and measured back."""
from __future__ import annotations

import importlib.util
import sqlite3

import numpy as np
import pytest

pytest.importorskip("pyarchinit_photogrammetry", reason="the faces are elevations of the program")

import masonry_samples as S  # noqa: E402
from ppg_masonry import faces, outputs, shape, stones, usm, walls  # noqa: E402

Z0 = S.WORLD[2]
HAS_IFC = importlib.util.find_spec("ifcopenshell") is not None


def _face(tmp_path, mesh=None):
    doc, ch = S.project(tmp_path, mesh or S.wall(lean=False, bulge=False))
    doc.meta["sito"] = "Saggio"
    w = walls.new_wall(ch, "Muro nord")
    entry = faces.plane_face(ch, w["id"], S.WORLD[:2], S.WORLD[:2] + [S.STONE_WALL, 0.0],
                             side=S.WORLD[:2] + [3.0, -5.0], resolution=0.02, name="Faccia sud")
    units, notch = S.usm_layout(Z0)
    polys = S.stone_pattern(Z0, hole=notch)[0]
    S.put_drawings(ch, entry["id"], polys, units)
    ch.meta["drawings"]["frames"][entry["id"]]["quote"] = [{"u": 1.0, "v": Z0 + 0.5, "quota_q": Z0 + 0.5}]
    shape.face_shape(ch, w["id"], entry["id"])
    shape.sections(ch, w["id"], entry["id"], heights=[Z0 + 0.5, Z0 + 1.5])
    stones.measure_face(ch, w["id"], entry["id"])
    return doc, ch, w, entry, polys


def test_plate_and_dxf(tmp_path):
    import ezdxf

    doc, ch, w, entry, polys = _face(tmp_path)
    assert outputs.choose_scale(6.0, 3.0) == 20 and outputs.choose_scale(30.0, 3.0) == 100
    out = outputs.plate(ch, w["id"], entry["id"])
    assert out["scale"] == 20 and out["svg"].exists() and out["pdf"].exists()
    svg = out["svg"].read_text(encoding="utf-8")
    assert "Faccia sud" in svg and "Scala 1:20" in svg and "Le misure suggeriscono" in svg
    assert out["pdf"].read_bytes()[:4] == b"%PDF"
    path = outputs.dxf(ch, w["id"], entry["id"], tmp_path / "f.dxf")
    msp = ezdxf.readfile(path).modelspace()
    count = {}
    for e in msp.query("LWPOLYLINE"):
        count[e.dxf.layer] = count.get(e.dxf.layer, 0) + 1
    assert count["PIETRE"] == len(polys) and count["USM"] == 4 and count["CORSI"] == S.COURSES
    assert count["SEZIONI"] == 2 and count["BORDO"] == 1
    assert len(msp.query("POINT[layer=='QUOTE']")) == 1
    texts = {e.dxf.text for e in msp.query("TEXT[layer=='USM']")}
    assert texts == {"USM 10", "USM 11", "USM 12", "USM 13"}


def test_the_whole_export_of_a_face(tmp_path):
    doc, ch, w, entry, polys = _face(tmp_path)
    out = outputs.export_face(ch, w["id"], entry["id"], tmp_path / "export")
    names = {p.name for p in (tmp_path / "export").iterdir()}
    for name in ("Faccia_sud_tavola.svg", "Faccia_sud_tavola.pdf", "Faccia_sud_muratura.dxf",
                 "Faccia_sud_disegni.gpkg", "Faccia_sud_disegni_mondo.gpkg", "Faccia_sud_pietre.xlsx",
                 "Faccia_sud_spanciamento.tif", "Faccia_sud_sezioni.dxf", "Faccia_sud_muro.json"):
        assert name in names, (name, sorted(names))
    con = sqlite3.connect(out["drawings_gpkg"])
    assert con.execute("SELECT COUNT(*) FROM pietre_misure").fetchone()[0] == len(polys)
    con.close()


@pytest.mark.skipif(not HAS_IFC, reason="the program's bim extra (ifcopenshell)")
def test_ifc_of_a_wall_with_one_face(tmp_path):
    import ifcopenshell
    import ifcopenshell.geom
    import ifcopenshell.util.element
    import ifcopenshell.util.shape
    import ifcopenshell.validate

    from ppg_masonry import ifc

    doc, ch, w, entry, _polys = _face(tmp_path)
    with pytest.raises(walls.NotReady, match="spessore"):
        ifc.export_wall(ch, w["id"], tmp_path / "w.ifc")
    found = usm.propose(ch, w["id"], entry["id"])
    accepted = [p for p in found if p["relation"] in ("Taglia", "Copre")]
    walls.set_result(ch, w["id"], entry["id"], "usm", {"accepted_full": accepted})
    r = ifc.export_wall(ch, w["id"], tmp_path / "w.ifc", thickness=0.6)
    assert r["parts"] == 4 and r["relations"] == 2 and abs(r["thickness"] - 0.6) < 1e-9
    model = ifcopenshell.open(r["path"])
    logger = ifcopenshell.validate.json_logger()
    ifcopenshell.validate.validate(model, logger)
    assert not logger.statements, logger.statements[:3]
    wall = model.by_type("IfcWall")[0]
    settings = ifcopenshell.geom.settings()
    shape_ = ifcopenshell.geom.create_shape(settings, wall)
    volume = ifcopenshell.util.shape.get_volume(shape_.geometry)
    assert abs(volume - r["area"] * 0.6) < 1e-3 * volume
    verts = np.asarray(shape_.geometry.verts).reshape(-1, 3)
    m = np.asarray(shape_.transformation.matrix).reshape(4, 4).T if len(shape_.transformation.matrix) == 16 else None
    if m is not None:
        verts = verts @ m[:3, :3].T + m[:3, 3]
    assert abs(np.ptp(verts[:, 1]) - 0.6) < 0.01  # through the wall: y is across it
    pset = ifcopenshell.util.element.get_psets(wall)["Pset_PyArchInit_Masonry"]
    assert pset["Filari"] == S.COURSES and pset["TipoSuggerito"].startswith("Le misure suggeriscono")
    assert pset["StrapiomboGradi"] < 0.1 and "SpanciamentoP90Cm" in pset
    parts = model.by_type("IfcBuildingElementPart")
    assert sorted(p.Name for p in parts) == ["USM 10", "USM 11", "USM 12", "USM 13"]
    assert {r.Name for r in model.by_type("IfcRelConnectsElements")} == {"Taglia", "Copre"}
    assert model.by_type("IfcMapConversion") and model.by_type("IfcRelAggregates")


@pytest.mark.skipif(not HAS_IFC, reason="the program's bim extra (ifcopenshell)")
def test_ifc_of_a_wall_between_two_faces(tmp_path):
    from ppg_masonry import ifc

    front = S.wall(lean=False, bulge=False)
    back = S.wall(base=S.WORLD + [0.0, 0.5, 0.0], lean=False, bulge=False)
    mesh = (np.vstack([front[0], back[0]]), np.vstack([front[1], back[1] + len(front[0])]))
    doc, ch = S.project(tmp_path, mesh)
    w = walls.new_wall(ch, "Muro")
    a = faces.plane_face(ch, w["id"], S.WORLD[:2], S.WORLD[:2] + [6.0, 0.0], side=S.WORLD[:2] + [3, -5],
                         resolution=0.04, back=0.2, front=0.2)
    b = faces.plane_face(ch, w["id"], S.WORLD[:2] + [0, 0.5], S.WORLD[:2] + [6.0, 0.5], side=S.WORLD[:2] + [3, 5],
                         resolution=0.04, back=0.2, front=0.2)
    for e in (a, b):
        shape.face_shape(ch, w["id"], e["id"])
    r = ifc.export_wall(ch, w["id"], tmp_path / "w.ifc")
    assert abs(r["thickness"] - 0.5) < 0.005 and "misurato fra le facce" in r["source"]
