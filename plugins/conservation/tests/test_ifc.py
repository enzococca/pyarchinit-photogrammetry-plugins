"""The IFC 4.3 of a campaign read back: the host wall (the plugin's stand-in behind a plane, a band of a cylinder, or
the masonry plugin's wall with the same GlobalId), the surface features adhering to it with their shells on the surface,
their property sets and quantities; a section refused; the GlobalIds shared with the masonry plugin."""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

pytest.importorskip("pyarchinit_photogrammetry", reason="the features live on the program's elevations")
pytest.importorskip("ifcopenshell", reason="the program's bim extra (ifcopenshell)")

import conservation_samples as S  # noqa: E402
from ppg_conservation import ifc43, records  # noqa: E402
from ppg_conservation.geometry import frame_of  # noqa: E402
from ppg_conservation.records import NotReady  # noqa: E402
from test_outputs import campaign  # noqa: E402

HAS_MASONRY = importlib.util.find_spec("ppg_masonry") is not None
Z0 = S.Z0


def _uv(chunk, entry, points):
    return np.asarray(frame_of(chunk, entry).coords(np.asarray(points, np.float64)))


def test_a_campaign_read_back(tmp_path):
    import ifcopenshell
    import ifcopenshell.validate

    doc, ch, entry, c, feats = campaign(tmp_path)
    eid = entry["id"]
    r = ifc43.export(ch, [c["id"]], tmp_path / "c.ifc", thickness=0.4)
    assert r["hosts"] == 1 and r["features"] == 7, r  # the six of the campaign and the note, not the proposal
    model = ifcopenshell.open(r["path"])
    assert model.schema_identifier == "IFC4X3_ADD2"
    logger = ifcopenshell.validate.json_logger()
    ifcopenshell.validate.validate(model, logger)
    assert not logger.statements, logger.statements[:3]
    assert not model.by_type("IfcTextureMap") and not model.by_type("IfcImageTexture")
    conv = model.by_type("IfcMapConversion")[0]
    rb = ifc43.read_back(r["path"])
    allv = np.vstack([h["vertices"] for h in rb["hosts"]] + [f["vertices"] for f in rb["features"]])
    assert np.allclose([conv.Eastings, conv.Northings, conv.OrthogonalHeight], np.floor(allv.min(axis=0)))

    # the host: the stand-in slab behind the plane, its GlobalId by the suite's rule
    (host,) = rb["hosts"]
    assert host["class"] == "IfcWall" and host["name"] == "Prospetto sud (parete di appoggio)"
    assert host["id"] == ifc43.stable_global_id(ch.key, "face", eid)
    d = _uv(ch, entry, host["vertices"])[:, 2]
    assert abs(d.max()) < 1e-3 and abs(d.min() + 0.4) < 1e-3  # from the plane to 40 cm behind it
    cond = host["psets"]["Pset_Condition"]
    assert "CC 2" in cond["AssessmentCondition"] and "UC 1" in cond["AssessmentCondition"]
    assert "RC 2" in cond["AssessmentDescription"] and cond["AssessmentType"] == "EN 16096"
    mine = host["psets"]["Pset_PyArchInit_Conservation"]
    assert mine["ConditionClass"] == 2 and mine["UrgencyClass"] == 1 and mine["RecommendationClass"] == 2
    assert mine["Surveyor"] == "E. Cocca" and mine["Date"] == c["date"] and mine["Campaign"] == c["name"]

    # the features: on the host, typed, quantified as the records say
    by_id = {f["id"]: f for f in rb["features"]}
    assert len(by_id) == 7 and all(f["host"] == host["id"] for f in by_id.values())
    assert rb["relations"] == [{"host": host["id"], "features": sorted(by_id)}]
    for key, f in feats.items():
        gid = ifc43.stable_global_id(ch.key, eid, f["id"])
        if key == "proposed":
            assert gid not in by_id
            continue
        got = by_id[gid]
        assert got["object_type"] == f["code"]
        assert got["predefined_type"] == ("TREATMENT" if key == "clean" else "DEFECT")
        assert got["triangles"] > 0
        cond = got["psets"]["Pset_Condition"]
        assert cond["AssessmentDate"] == c["date"] and cond["AssessmentType"] == "EN 16096"
        assert cond["AssessmentCondition"] == "CC 2"
        qto = got["psets"].get("Qto_SurfaceFeatureBaseQuantities") or {}
        if f["kind"] == "crack":
            assert abs(qto["Length"] - f["extent"]) < 0.01 * f["extent"]
        elif f["kind"] != "note":
            assert abs(qto["Area"] - f["extent"]) < 0.01 * f["extent"]
        # every vertex on the face: inside its (u, v), near the plane (the hole 3 cm deep)
        uvd = _uv(ch, entry, got["vertices"])
        assert np.abs(uvd[:, 2]).max() < 0.035
    crust = by_id[ifc43.stable_global_id(ch.key, eid, feats["crust"]["id"])]
    assert crust["name"].endswith(" 1") or crust["name"][-1].isdigit()
    mine = crust["psets"]["Pset_PyArchInit_Conservation"]
    assert mine["Code"] == "ISCS.CRUST.BLACK_CRUST" and mine["Standard"] == "ICOMOS-ISCS"
    assert mine["Intensity"] == "high" and mine["UrgencyClass"] == 2 and abs(mine["Extent_m2"] - 0.8) < 1e-6
    assert "Black crust" in mine["TermEN"] and mine["Kind"] == "decay"
    clean = by_id[ifc43.stable_global_id(ch.key, eid, feats["clean"]["id"])]["psets"]["Pset_PyArchInit_Conservation"]
    assert clean["Technique"] == "INT.CLEANING" and clean["Materials"] == "acqua deionizzata" and clean["Done"] is True

    # the hole lifted 3 cm into the wall through the relief
    hole = by_id[ifc43.stable_global_id(ch.key, eid, feats["hole"]["id"])]
    assert abs(_uv(ch, entry, hole["vertices"])[:, 2].min() + S.HOLE_DEPTH) < 0.002
    # the crack: a band of 5 mm (its 3 mm opening, below the narrowest band) along the groove
    crack = by_id[ifc43.stable_global_id(ch.key, eid, feats["crack"]["id"])]
    assert crack["psets"]["Pset_PyArchInit_Conservation"]["Width_mm"] == 3.0
    from shapely import distance, points

    off = distance(S.groove(), points(_uv(ch, entry, crack["vertices"])[:, :2]))
    assert off.max() < 0.0026 and off.max() > 0.0024
    # the note: a disc of 5 cm
    note = by_id[ifc43.stable_global_id(ch.key, eid, feats["note"]["id"])]
    uv = _uv(ch, entry, note["vertices"])[:, :2]
    assert abs(np.hypot(uv[:, 0] - 3.0, uv[:, 1] - Z0 - 1.5).max() - 0.05) < 1e-3


def test_unnamed_features_are_left_out_and_said(tmp_path):
    from ppg_conservation.vocabulary import UNNAMED

    doc, ch, entry, c, feats = campaign(tmp_path)
    f = records.add_feature(ch, c["id"], "decay", "ISCS.ALGA", S.box((5.4, Z0 + 0.1, 5.8, Z0 + 0.3)))
    f["code"] = UNNAMED  # as a proposal left unnamed would be, accepted by hand in the project
    r = ifc43.export(ch, [c["id"]], tmp_path / "c.ifc")
    assert r["features"] == 7
    assert any("da nominare" in w for w in r["warnings"]), r["warnings"]


def test_a_section_is_refused(tmp_path):
    doc, ch, entry, c, feats = campaign(tmp_path)
    section = dict(entry, id="sez00001", kind="section", name="Sezione A")
    ch.meta["elevations"].append(section)
    s = records.new_campaign(ch, section["id"])
    records.add_feature(ch, s["id"], "decay", "ISCS.CRUST.BLACK_CRUST", S.box(S.BLACK_CRUST))
    with pytest.raises(NotReady, match="una sezione è un taglio, non una faccia"):
        ifc43.export(ch, [s["id"]], tmp_path / "s.ifc")
    r = ifc43.export(ch, [c["id"], s["id"]], tmp_path / "both.ifc")  # the face exported, the section said
    assert r["hosts"] == 1 and any("una sezione è un taglio" in w for w in r["warnings"])


def test_a_cylinder_face(tmp_path):
    """Review focus: the host a band of the cylinder within 1 cm of it, the features' shells on the tower."""
    doc, ch, entry = S.tower(tmp_path)
    eid = entry["id"]
    c = records.new_campaign(ch, eid, surveyor="E. Cocca")
    f = records.add_feature(ch, c["id"], "decay", "ISCS.CRUST.BLACK_CRUST", S.box((0.8, Z0 + 1.5, 1.6, Z0 + 2.3)))
    r = ifc43.export(ch, [c["id"]], tmp_path / "t.ifc", thickness=0.6)
    assert r["hosts"] == 1 and r["features"] == 1
    rb = ifc43.read_back(r["path"])
    (host,) = rb["hosts"]
    assert host["id"] == ifc43.stable_global_id(ch.key, "face", eid) and host["triangles"] > 50
    frame = frame_of(ch, entry)
    uvd = _uv(ch, entry, host["vertices"])
    outer, inner = uvd[np.abs(uvd[:, 2]) < 0.01], uvd[np.abs(uvd[:, 2] + 0.6) < 0.01]
    assert len(outer) + len(inner) == len(uvd) and len(outer) and len(inner)  # the shell: the face and 60 cm in
    # the outer side's vertices on the cylinder, the band over the face's bounds
    across = host["vertices"] - frame.centre
    across -= (across @ frame.axis)[:, None] * frame.axis
    r_out = np.linalg.norm(across, axis=1)[np.abs(uvd[:, 2]) < 0.01]
    assert np.abs(r_out - frame.radius).max() < 0.01
    u0, v0, u1, v1 = entry["bounds"]
    assert abs(uvd[:, 0].min() - u0) < 0.01 and abs(uvd[:, 0].max() - u1) < 0.01
    assert abs(uvd[:, 1].min() - v0) < 0.01 and abs(uvd[:, 1].max() - v1) < 0.01
    (feat,) = rb["features"]
    assert feat["host"] == host["id"] and feat["id"] == ifc43.stable_global_id(ch.key, eid, f["id"])
    assert rb["relations"] == [{"host": host["id"], "features": [feat["id"]]}]
    # the feature's shell on the tower's surface (lifted through the relief), and over the drawn area
    assert np.abs(S.off_the_tower(feat["vertices"])).max() < 0.01
    fu = _uv(ch, entry, feat["vertices"])
    assert fu[:, 0].min() > 0.8 - 1e-3 and fu[:, 0].max() < 1.6 + 1e-3
    assert abs(feat["psets"]["Qto_SurfaceFeatureBaseQuantities"]["Area"] - f["extent"]) < 0.01 * f["extent"]


@pytest.mark.skipif(not HAS_MASONRY, reason="the masonry plugin")
def test_the_host_is_the_masonry_wall(tmp_path):
    from ppg_masonry import ifc as masonry_ifc
    from ppg_masonry import walls

    doc, ch, entry, c, feats = campaign(tmp_path)
    w = walls.new_wall(ch, "Muro sud")
    walls.add_face(ch, w["id"], entry["id"])
    r = ifc43.export(ch, [c["id"]], tmp_path / "c.ifc", thickness=0.4)
    rb = ifc43.read_back(r["path"])
    (host,) = rb["hosts"]
    assert host["id"] == masonry_ifc.stable_global_id(ch.key, w["id"], entry["id"])
    assert host["id"] == ifc43.stable_global_id(ch.key, w["id"], entry["id"]) and host["name"] == "Muro sud"
    assert all(f["host"] == host["id"] for f in rb["features"]) and len(rb["features"]) == 7
    # the features still on the face and lifted on the relief: the hole 3 cm into the wall
    by_id = {f["id"]: f for f in rb["features"]}
    for f in rb["features"]:
        assert np.abs(_uv(ch, entry, f["vertices"])[:, 2]).max() < 0.035
    hole = by_id[ifc43.stable_global_id(ch.key, entry["id"], feats["hole"]["id"])]
    assert abs(_uv(ch, entry, hole["vertices"])[:, 2].min() + S.HOLE_DEPTH) < 0.002
    # the masonry plugin's own file names the same wall, export after export
    ids = []
    for n in (1, 2):
        m = masonry_ifc.export_wall(ch, w["id"], tmp_path / f"m{n}.ifc", thickness=0.4)
        ids.append(sorted(h["id"] for h in ifc43.read_back(m["path"])["hosts"]))
    assert ids[0] == ids[1] == [host["id"]]
    # the two slabs coincide
    mb = ifc43.read_back(tmp_path / "m1.ifc")["hosts"][0]
    same = [np.unique(np.round(v, 3), axis=0) for v in (mb["vertices"], host["vertices"])]
    assert same[0].shape == same[1].shape and np.abs(same[0] - same[1]).max() < 2e-3


def test_ifc_under_an_italian_numeric_locale(tmp_path):
    """With LC_NUMERIC Italian (a decimal comma, as Qt sets it) the stand-in slab is still 0.45 m thick, read back
    from the file; the caller's locale is kept."""
    import locale

    doc, ch, entry, c, feats = campaign(tmp_path)
    old = locale.setlocale(locale.LC_NUMERIC)
    try:
        locale.setlocale(locale.LC_NUMERIC, "it_IT.UTF-8")
    except locale.Error:
        pytest.skip("no it_IT.UTF-8 locale here")
    try:
        r = ifc43.export(ch, [c["id"]], tmp_path / "c.ifc", thickness=0.45)
        assert locale.setlocale(locale.LC_NUMERIC) == "it_IT.UTF-8"
    finally:
        locale.setlocale(locale.LC_NUMERIC, old)
    (host,) = ifc43.read_back(r["path"])["hosts"]
    d = _uv(ch, entry, host["vertices"])[:, 2]
    assert abs(d.min() + 0.45) < 1e-3


def test_ifc_without_ifcopenshell_is_said(tmp_path, monkeypatch):
    import importlib.util as iu

    doc, ch, entry, c, feats = campaign(tmp_path)
    real = iu.find_spec
    monkeypatch.setattr(iu, "find_spec", lambda name, *a: None if name == "ifcopenshell" else real(name, *a))
    with pytest.raises(NotReady, match="extra bim"):
        ifc43.export(ch, [c["id"]], tmp_path / "c.ifc")


@pytest.mark.skipif(not HAS_MASONRY, reason="the masonry plugin")
def test_one_rule_for_the_suite():
    from ppg_masonry import ifc as masonry_ifc

    assert ifc43.SUITE_NAMESPACE == masonry_ifc.SUITE_NAMESPACE
    assert ifc43.stable_global_id(0, "w1", "e1") == masonry_ifc.stable_global_id(0, "w1", "e1")
    assert ifc43.stable_global_id(0, "w1", "e1") != ifc43.stable_global_id(0, "w1", "e2")
    assert len(ifc43.stable_global_id("x")) == 22


def test_the_ifc_from_the_pane(tmp_path, monkeypatch):
    """*IFC 4.3* in the pane: the dialog (the campaigns of the elevation, the thickness, the file), the export in the
    background, the line said; the exporter of *File > Esporta*."""
    pytest.importorskip("PySide6")
    from test_plugin import _window

    from ppg_conservation import ConservationPlugin
    from pyarchinit_photogrammetry.plugins import Context

    w, doc, ch, entry, errors, infos = _window(tmp_path, monkeypatch)
    _d, _c, _e, c, feats = campaign(tmp_path / "copy")  # the same records, put into the window's chunk
    ch.meta["conservation"] = _c.meta["conservation"]
    plugin = ConservationPlugin()
    try:
        plugin.activate(Context("conservation", tmp_path / "plugin", window=w))
        said, asked, answers = [], [], {}
        monkeypatch.setattr(plugin, "_message", lambda kind, text: said.append(text))

        def answer(title, fields, intro="", photos=None, **_k):
            asked.append((title, intro, {f.name: f for f in fields}))
            values = {f.name: f.default for f in fields if f.kind != "heading"}
            values.update(answers)
            return values

        monkeypatch.setattr(plugin, "_ask", answer)
        plugin.refresh_pane()
        assert plugin.pane.campaign_id() == c["id"]
        answers.update(path=str(tmp_path / "out" / "campagna"), thickness=0.3)
        plugin._requested("ifc")
        w.runner.wait()
        title, intro, fields = asked[-1]
        assert title == "IFC 4.3 della campagna" and "IfcSurfaceFeature" in intro
        assert fields["campaigns"].kind == "checklist" and fields["campaigns"].default == [c["id"]]
        assert [v for v, _l in fields["campaigns"].choices] == [c["id"]] and fields["thickness"].default == 0.5
        assert fields["path"].default.endswith(".ifc")
        path = tmp_path / "out" / "campagna.ifc"
        assert "IFC 4.3:" in said[-1] and str(path) in said[-1] and "1 parete con 7 elementi" in said[-1]
        rb = ifc43.read_back(path)
        d = _uv(ch, entry, rb["hosts"][0]["vertices"])[:, 2]
        assert abs(d.min() + 0.3) < 1e-3 and len(rb["features"]) == 7
        # cancelled: nothing more
        monkeypatch.setattr(plugin, "_ask", lambda *a, **k: None)
        n = len(said)
        plugin._requested("ifc")
        assert len(said) == n
        # File > Esporta > Conservazione (IFC 4.3)
        out = plugin._export_ifc(ch, tmp_path / "esportato.ifc")
        assert out == tmp_path / "esportato.ifc" and len(ifc43.read_back(out)["features"]) == 7
        assert not errors, errors
        plugin.deactivate()
        plugin.context.release()
    finally:
        w.setWindowModified(False)
        w.close()


def test_the_ifc_from_the_command_line(tmp_path, capsys):
    from ppg_conservation import cli

    doc, ch, entry, c, feats = campaign(tmp_path)
    out = tmp_path / "c.ifc"
    assert cli.run(["ifc", str(doc.path), "--campaign", c["name"], "--out", str(out), "--thickness", "0.25"]) == 0
    printed = capsys.readouterr().out
    assert f"{out}: 1 parete, 7 elementi" in printed
    rb = ifc43.read_back(out)
    assert abs(_uv(ch, entry, rb["hosts"][0]["vertices"])[:, 2].min() + 0.25) < 1e-3
    # by the elevation, every campaign of it
    assert cli.run(["ifc", str(doc.path), "--elevation", entry["name"], "--out", str(tmp_path / "e.ifc")]) == 0
    assert len(ifc43.read_back(tmp_path / "e.ifc")["features"]) == 7
    # a campaign that is not there: one line, code 1
    assert cli.run(["ifc", str(doc.path), "--campaign", "nessuna", "--out", str(tmp_path / "x.ifc")]) == 1
    assert "Nessuna campagna 'nessuna'" in capsys.readouterr().out
    assert "ifc" in cli.COMMANDS
