"""The decay mapping on the web site of a survey (program 1.10.0): the layer of the areas of decay with the project's
legend - on an upright face the stretch of the wall each covers, on a face that lies down the area itself -, the tabs
of the summary and of the campaigns (the EN 16096 classes, never the surveyor), an elevation of another system, the
site built with the plugin (moved into the local frame, no path, no system) and the plugin started by a program
without the hook."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarchinit_photogrammetry", reason="the site lives on the program's elevations")

import conservation_samples as S  # noqa: E402
from ppg_conservation import records  # noqa: E402
from ppg_conservation import site as wsite  # noqa: E402
from ppg_conservation.geometry import frame_of  # noqa: E402
from ppg_conservation.vocabulary import legend  # noqa: E402

Z0 = S.Z0
SURVEYOR = "E. Cocca"


def _campaign(tmp_path):
    doc, ch, entry = S.face(tmp_path)
    c = records.new_campaign(ch, entry["id"], surveyor=SURVEYOR, method="rilievo fotogrammetrico")
    records.set_campaign(ch, c["id"], classes={"cc": 2, "uc": 1, "reason": "lesione aperta"})
    records.set_assessment(ch, c["date"], 2, reason="riparazione", surveyor=SURVEYOR)
    cid = c["id"]
    records.add_feature(ch, cid, "decay", "ISCS.CRUST.BLACK_CRUST", S.box(S.BLACK_CRUST), intensity="high")
    records.add_feature(ch, cid, "decay", "UNI.PATINA_BIOLOGICA", S.box(S.GREEN_PATCH))
    records.add_feature(ch, cid, "decay", "ISCS.EFFLORESCENCE", S.box(S.EFFLORESCENCE), intensity="low")
    records.add_feature(ch, cid, "crack", "ISCS.CRACK", S.groove(), width_mm=3.0)
    records.add_feature(ch, cid, "treatment", "INT.CLEANING", S.box(S.BLACK_CRUST),
                        treatment={"technique": "INT.CLEANING", "done": True, "by": "Restauri srl"})
    records.add_feature(ch, cid, "note", "ISCS.CRUST", f"POINT (3 {Z0 + 1.5})", notes="campione")
    records.add_feature(ch, cid, "decay", "ISCS.ALGA", S.box((5.4, Z0 + 2.0, 5.8, Z0 + 2.4)), status="proposed",
                        source="proposed:colours")
    doc.save()
    return doc, ch, entry, c


def test_the_layer_on_an_upright_face_and_the_tabs(tmp_path):
    _doc, ch, entry, c = _campaign(tmp_path)
    layer, summary, campaigns = wsite.contributions(ch)
    assert (layer["id"], summary["id"], campaigns["id"]) == ("decay", "summary", "campaigns")
    feats = layer["features"]
    # the three accepted areas of decay, not the proposal; on a wall each is the stretch it covers, at its middle height
    assert len(feats) == 3 and {f["geometry"]["type"] for f in feats} == {"LineString"}
    frame = frame_of(ch, entry)
    crust = next(f for f in feats if f["properties"]["code"] == "ISCS.CRUST.BLACK_CRUST")
    u0, v0, u1, v1 = S.BLACK_CRUST
    expected = frame.point(np.array([u0, u1]), np.full(2, (v0 + v1) / 2), np.zeros(2))
    assert np.allclose(crust["geometry"]["coordinates"], expected, atol=1e-6)
    assert crust["properties"]["extent"] == pytest.approx((u1 - u0) * (v1 - v0), abs=1e-3)
    assert crust["properties"]["intensity"] == "alta" and crust["properties"]["date"] == c["date"]
    assert crust["properties"]["elevation"] == "Prospetto sud"
    # the legend: a tick per term present, the project's colours, both glossaries
    colours = legend(ch)
    items = {i["value"]: i for i in layer["legend"]["items"]}
    assert layer["legend"]["field"] == "code"
    assert set(items) == {"ISCS.CRUST.BLACK_CRUST", "UNI.PATINA_BIOLOGICA", "ISCS.EFFLORESCENCE"}
    assert items["ISCS.EFFLORESCENCE"]["colour"] == colours["ISCS.EFFLORESCENCE"]["colour"]
    assert set(items["ISCS.EFFLORESCENCE"]["title"]) == {"it", "en"}
    # the summary: per term, the extent and the share of the face; the crack in metres; the note left out
    rows = {(r["kind"], r["code"]): r for r in summary["rows"]}
    assert set(rows) == {("degrado", "ISCS.CRUST.BLACK_CRUST"), ("degrado", "UNI.PATINA_BIOLOGICA"),
                         ("degrado", "ISCS.EFFLORESCENCE"), ("lesione", "ISCS.CRACK"),
                         ("intervento", "INT.CLEANING")}
    face = records.face_area(ch, entry)
    assert rows[("degrado", "ISCS.CRUST.BLACK_CRUST")]["share"] == pytest.approx(
        100.0 * (u1 - u0) * (v1 - v0) / face, abs=0.01)
    assert rows[("lesione", "ISCS.CRACK")]["unit"] == "m" and "share" not in rows[("lesione", "ISCS.CRACK")]
    assert summary["rows"][0]["kind"] == "degrado"  # the areas first, the largest first
    # the campaigns: the classes the surveyor chose, the recommendation of the date; never the surveyor
    row = campaigns["rows"][0]
    assert (row["cc"], row["uc"], row["rc"], row["reason"]) == (2, 1, 2, "lesione aperta")
    assert row["method"] == "rilievo fotogrammetrico" and row["features"] == 6
    assert SURVEYOR not in json.dumps([layer, summary, campaigns], ensure_ascii=False)
    assert [c["key"] for c in campaigns["columns"]][:3] == ["elevation", "campaign", "date"]


def test_a_face_that_lies_down_keeps_its_areas(tmp_path):
    from pyarchinit_photogrammetry.engine.elevation import Plane

    _doc, ch, entry, _c = _campaign(tmp_path)
    floor = Plane(S.WORLD, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    entry["frame"] = floor.to_dict()
    layer = wsite.contributions(ch)[0]
    assert {f["geometry"]["type"] for f in layer["features"]} == {"Polygon"}
    crust = next(f for f in layer["features"] if f["properties"]["code"] == "ISCS.CRUST.BLACK_CRUST")
    ring = np.array(crust["geometry"]["coordinates"][0])
    u0, v0, u1, v1 = S.BLACK_CRUST
    assert ring[:, 0].min() == pytest.approx(S.WORLD[0] + u0) and ring[:, 1].max() == pytest.approx(S.WORLD[1] + v1)
    assert np.allclose(ring[:, 2], S.WORLD[2])
    # a cylinder: a stretch with many points along the arc
    from pyarchinit_photogrammetry.engine.elevation import Cylinder
    from shapely.geometry import box

    tower = Cylinder(S.WORLD, (0.0, 0.0, 1.0), 3.0, False, (1.0, 0.0, 0.0))
    line = wsite.place(tower, box(0.0, 1.0, 2.0, 1.5))[0]
    pts = np.array(line["coordinates"])
    assert line["type"] == "LineString" and len(pts) > 10
    assert np.allclose(np.hypot(pts[:, 0] - S.WORLD[0], pts[:, 1] - S.WORLD[1]), 3.0, atol=1e-6)


def test_an_elevation_of_another_system_and_a_chunk_without_campaigns(tmp_path):
    _doc, ch, entry, _c = _campaign(tmp_path)
    entry["crs"] = "EPSG:32632"
    given = wsite.contributions(ch)
    assert [g["id"] for g in given] == ["summary", "campaigns"]
    records.store(ch)["campaigns"] = []
    assert wsite.contributions(ch) == []


def test_the_decay_on_the_plan_of_a_site_with_the_first_form_of_the_hook(tmp_path, monkeypatch):
    """A program whose hook does not tell the frame of the site (``site_hook`` absent): the areas on the plan."""
    site_plugins = pytest.importorskip("pyarchinit_photogrammetry.io.site_plugins",
                                       reason="the web site of a survey: program 1.10.0")
    monkeypatch.setattr(site_plugins, "_providers", [])  # only the plugin of the test: none left by others
    from pyarchinit_photogrammetry.plugins.api import Context

    from ppg_conservation import ConservationPlugin

    monkeypatch.delattr(Context, "site_hook", raising=False)
    doc, ch, entry, _c = _campaign(tmp_path / "progetto")
    plugin = ConservationPlugin()
    context = Context("conservation", tmp_path / "plugin")
    before = len(site_plugins.providers())
    plugin.activate(context)
    try:
        assert len(site_plugins.providers()) == before + 1
        path = doc.export_site(tmp_path / "sito", products=["model", "sections"], model_faces=4000)
    finally:
        context.release()
    assert len(site_plugins.providers()) == before
    site = path.parent
    manifest = json.loads(path.read_text(encoding="utf-8"))
    chunk = manifest["chunks"][0]
    listed = {p["id"]: p for p in chunk["plugins"]}
    assert set(listed) == {"decay", "summary", "campaigns"}
    assert {p["plugin"] for p in listed.values()} == {"conservation"}
    layer = next(x for x in chunk["layers"] if x["id"] == "plugin:conservation:decay")
    assert layer["url"] == "data/plugins/conservation/decay.geojson" and layer["count"] == 3
    assert layer["group_by"] == "code" and {g["count"] for g in layer["groups"]} == {1}
    # moved into the local frame as the program's trace of the same elevation: one shift for both
    feats = json.loads((site / layer["url"]).read_text(encoding="utf-8"))["features"]
    crust = next(f for f in feats if f["properties"]["code"] == "ISCS.CRUST.BLACK_CRUST")
    frame = frame_of(ch, entry)
    u0, v0, u1, v1 = S.BLACK_CRUST
    world = frame.point(np.array([u0, u1]), np.full(2, (v0 + v1) / 2), np.zeros(2))
    shift = world - np.array(crust["geometry"]["coordinates"])
    assert np.allclose(shift, shift[0], atol=2e-3) and np.allclose(shift[0], np.round(shift[0]), atol=2e-3)
    traces = json.loads((site / "data/sections.geojson").read_text(encoding="utf-8"))["features"]
    from pyarchinit_photogrammetry.io.site_sections import trace
    from pyarchinit_photogrammetry.engine.elevation import bounds_of

    raw = np.array(trace(entry, bounds_of(ch, entry))["coordinates"])
    assert np.allclose(raw[0] - np.array(traces[0]["geometry"]["coordinates"][0]), shift[0][:2], atol=2e-3)
    # nothing in the files of the plugin tells the place, the computer or the surveyor
    private = [str(Path.home()), str(tmp_path), str(Path(doc.path).parent), SURVEYOR]
    for f in (site / "data/plugins/conservation").iterdir():
        text = f.read_text(encoding="utf-8")
        assert not any(p in text for p in private), f.name
        assert "EPSG" not in text and "/Users/" not in text
        assert all(abs(float(n)) < 1000 for n in re.findall(r"-?\d+\.\d+", text)), f.name


def _site(shift=(0.0, 0.0, 0.0), elevations=()):
    site_plugins = pytest.importorskip("pyarchinit_photogrammetry.io.site_plugins")
    if not hasattr(site_plugins, "Site"):
        pytest.skip("the frame of the site: the second form of the program's hook")
    from pyarchinit_photogrammetry.io.site import _Shift

    return site_plugins.Site(_Shift(*shift), "local", elevations=elevations)


def test_with_the_frame_the_areas_go_on_their_elevation(tmp_path):
    _doc, ch, entry, c = _campaign(tmp_path)
    on = _site((10.0, 20.0, 100.0), [{"elevation": entry["id"], "section": "1", "name": "Prospetto sud",
                                      "kind": "elevation", "heights": True}])
    layer, summary, _campaigns = wsite.contributions_with_site(ch, on)
    assert layer["elevation"] == entry["id"] and layer["id"] == "decay" and len(layer["features"]) == 3
    crust = next(f for f in layer["features"] if f["properties"]["code"] == "ISCS.CRUST.BLACK_CRUST")
    u0, v0, u1, v1 = S.BLACK_CRUST
    ring = np.asarray(crust["geometry"]["coordinates"][0])
    assert ring[:, 0].min() == pytest.approx(u0) and ring[:, 1].max() == pytest.approx(v1)  # (u, v): the site moves v
    p = crust["properties"]
    assert (p["z_min"], p["z_max"]) == (pytest.approx(v0 - 100.0), pytest.approx(v1 - 100.0))  # heights of the site
    assert p["term"]["it"] and p["term"]["en"] and p["intensity"] == {"it": "alta", "en": "high"}
    assert set(p["family"]) == {"it", "en"} and "term_it" not in p
    assert set(layer["fields"]) >= {"term", "z_min", "z_max"}
    rows = {(r["code"]): r for r in summary["rows"]}
    assert rows["ISCS.CRACK"]["kind"] == {"it": "lesione", "en": "crack"}
    assert [col["key"] for col in summary["columns"]][:6] == ["elevation", "campaign", "date", "kind", "code", "term"]
    # an elevation the site does not publish: its areas on the plan, as before, the words in both languages
    plan, *_ = wsite.contributions_with_site(ch, _site())
    assert "elevation" not in plan and len(plan["features"]) == 3
    assert {f["geometry"]["type"] for f in plan["features"]} == {"LineString"}
    assert "z_min" not in plan["features"][0]["properties"] and "term" in plan["features"][0]["properties"]


def test_the_decay_on_the_elevation_of_a_site(tmp_path, monkeypatch):
    site_plugins = pytest.importorskip("pyarchinit_photogrammetry.io.site_plugins",
                                       reason="the web site of a survey: program 1.10.0")
    monkeypatch.setattr(site_plugins, "_providers", [])  # only the plugin of the test: none left by others
    from pyarchinit_photogrammetry.plugins.api import Context

    from ppg_conservation import ConservationPlugin

    if getattr(Context, "site_hook", 1) < 2:
        pytest.skip("the frame of the site: the second form of the program's hook")
    doc, ch, entry, _c = _campaign(tmp_path / "progetto")
    plugin = ConservationPlugin()
    context = Context("conservation", tmp_path / "plugin")
    plugin.activate(context)
    try:
        path = doc.export_site(tmp_path / "sito", products=["model", "sections"], model_faces=4000)
    finally:
        context.release()
    site = path.parent
    manifest = json.loads(path.read_text(encoding="utf-8"))
    chunk = manifest["chunks"][0]
    assert not any("conservation" in n for n in manifest["build"]["notes"])
    assert not any(x["id"].startswith("plugin:conservation") for x in chunk["layers"])  # not on the plan
    (sec,) = chunk["sections"]
    (layer,) = sec["layers"]
    assert layer["id"] == "plugin:conservation:decay" and layer["count"] == 3 and layer["group_by"] == "code"
    feats = json.loads((site / layer["url"]).read_text(encoding="utf-8"))["features"]
    crust = next(f for f in feats if f["properties"]["code"] == "ISCS.CRUST.BLACK_CRUST")
    u0, v0, u1, v1 = S.BLACK_CRUST
    ring = np.asarray(crust["geometry"]["coordinates"][0])
    assert ring[:, 0].min() == pytest.approx(u0, abs=1e-4)
    dz = v1 - ring[:, 1].max()  # the base of the heights of the site: a whole metre, as the drawings of the section
    assert dz == pytest.approx(round(dz), abs=1e-3) and 0 <= ring[:, 1].min() < 50
    assert crust["properties"]["z_max"] == pytest.approx(ring[:, 1].max(), abs=1e-3)
    assert crust["properties"]["term"]["en"] and crust["properties"]["intensity"] == {"it": "alta", "en": "high"}
    private = [str(Path.home()), str(tmp_path), str(Path(doc.path).parent), SURVEYOR]
    for f in (site / "data/plugins/conservation").iterdir():
        text = f.read_text(encoding="utf-8")
        assert not any(p in text for p in private), f.name
        assert "EPSG" not in text and "/Users/" not in text
        assert all(abs(float(n)) < 1000 for n in re.findall(r"-?\d+\.\d+", text)), f.name


def test_a_program_without_the_web_site(tmp_path, monkeypatch):
    """The program before 1.10.0 has no ``add_site_contributions``: the plugin starts all the same."""
    from pyarchinit_photogrammetry.plugins.api import Context

    from ppg_conservation import ConservationPlugin

    monkeypatch.delattr(Context, "add_site_contributions", raising=False)
    context = Context("conservation", tmp_path / "plugin")
    plugin = ConservationPlugin()
    plugin.activate(context)
    context.release()
