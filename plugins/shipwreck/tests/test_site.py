"""The timber catalogue on the web site of a survey (program 1.10.0): the layer of the outlines with a legend per kind,
the tab of the catalogue with the measures in the ship frame, heights never absolute, a ship frame of another system,
the site built with the plugin (the outlines moved into the local frame, no path, no system), and the plugin started
by a program without the hook."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from hulls import project

import pyarchinit_photogrammetry as ppg  # noqa: F401
from ppg_shipwreck import ship, timbers
from ppg_shipwreck import site as wsite


def _world(frame, *points):
    return frame.to_world(np.array([[x, y, 0.0] for x, y in points]))[:, :2]


def _catalogue(tmp_path, z0=None):
    doc, ch, ends = project(tmp_path, heading_deg=37.0)
    ship.set_frame(ch, ship.Frame(ends[0], ends[1], "end", z0))
    frame = ship.frame_of(ch)
    timbers.add(ch, "keel", "centre", "K", timbers.geometry("line", _world(frame, (2.0, 0.0), (10.0, 0.0))))
    timbers.add(ch, "frame", "port", "F1", timbers.geometry("line", _world(frame, (6.0, 0.0), (6.0, 0.5))),
                station="S3", notes="ordinata intera")
    timbers.add(ch, "plank", "port", "P", timbers.geometry(
        "polygon", _world(frame, (3.0, 0.1), (5.0, 0.1), (5.0, 0.4), (3.0, 0.4))))
    timbers.add(ch, "fastening", "port", "N", timbers.geometry("point", _world(frame, (6.0, 0.5))), loose=True)
    timbers.add(ch, "frame", "starboard", "F1", timbers.geometry("line", _world(frame, (6.0, 0.0), (6.0, -0.5))),
                status="reconstructed:mirror")
    doc.save()
    return doc, ch, frame


def test_the_layer_and_the_catalogue(tmp_path):
    _doc, ch, frame = _catalogue(tmp_path)
    layer, table = wsite.contributions(ch)
    assert layer["kind"] == "layer" and layer["id"] == "timbers" and table["kind"] == "table"
    feats = layer["features"]
    assert [f["geometry"]["type"] for f in feats] == ["LineString", "LineString", "Polygon", "Point", "LineString"]
    # the outlines as drawn, in the coordinates of the chunk; a polygon closed
    keel = timbers.timbers_of(ch)[0]["geometry"]["xy"]
    assert np.allclose(feats[0]["geometry"]["coordinates"], keel)
    ring = feats[2]["geometry"]["coordinates"][0]
    assert len(ring) == 5 and ring[0] == ring[-1]
    # a tick per kind present, in the order of the kinds, with the plugin's colours
    legend = layer["legend"]
    assert legend["field"] == "kind"
    assert [i["value"] for i in legend["items"]] == ["chiglia", "ordinata", "corso di fasciame", "elemento di fissaggio"]
    assert legend["items"][1]["title"] == {"it": "ordinata", "en": "frame"} and legend["items"][1]["colour"] == "#ff8c00"
    assert {f["properties"]["kind"] for f in feats} == {i["value"] for i in legend["items"]}
    assert feats[3]["properties"]["loose"] == "sì" and "x" not in feats[0]["properties"]  # measures only in the tab
    assert feats[4]["properties"]["status"] == "ricostruito: simmetria"
    assert layer["fields"]["length"] == {"it": "Lunghezza (m)", "en": "Length (m)"}
    # the catalogue: a card per timber, the measures in the ship frame as the plugin's own catalogue gives them
    rows = {r["id"]: r for r in table["rows"]}
    measured = {r["id"]: r for r in timbers.measure(ch)}
    assert list(rows) == ["T1", "T2", "T3", "T4", "T5"]
    assert rows["T1"]["length"] == pytest.approx(measured["T1"]["length"], abs=1e-3)
    assert rows["T1"]["length"] == pytest.approx(8.0, abs=0.2)
    assert rows["T3"]["width"] == pytest.approx(0.3, abs=1e-3) and rows["T2"]["x"] == pytest.approx(6.0, abs=1e-3)
    assert rows["T2"]["z_max"] == pytest.approx(measured["T2"]["z_max"], abs=1e-3)
    assert rows["T2"]["station"] == "S3" and rows["T2"]["notes"] == "ordinata intera"
    keys = [c["key"] for c in table["columns"]]
    assert keys[:3] == ["id", "name", "kind"] and "z_min" in keys and "z_min_water" not in keys  # no water: no column
    assert all(abs(v) < 100 for r in table["rows"] for v in r.values() if isinstance(v, float))


def test_heights_never_absolute(tmp_path):
    """A ship frame with a datum of its own (here the world's zero): the heights on the site are still from the start
    of the keel line, as with the default datum."""
    _d, ch, _f = _catalogue(tmp_path / "a")
    _d2, ch2, _f2 = _catalogue(tmp_path / "b", z0=0.0)
    assert timbers.measure(ch2)[1]["z_max"] != pytest.approx(timbers.measure(ch)[1]["z_max"], abs=1.0)
    rows = {r["id"]: r for r in wsite.contributions(ch)[1]["rows"]}
    rows2 = {r["id"]: r for r in wsite.contributions(ch2)[1]["rows"]}
    for key in ("z_min", "z_max"):
        assert rows2["T2"][key] == pytest.approx(rows["T2"][key], abs=1e-6)


def test_a_frame_of_another_system_and_an_empty_catalogue(tmp_path):
    _doc, ch, _frame = _catalogue(tmp_path)
    ch.crs = "EPSG:32632"  # the chunk changed system after the frame was set: the outlines are in the old one
    given = wsite.contributions(ch)
    assert [g["kind"] for g in given] == ["table"]
    assert "length" not in given[0]["rows"][0] and given[0]["rows"][0]["kind"] == "chiglia"
    ship.state(ch)["timbers"] = []
    assert wsite.contributions(ch) == []


def test_the_catalogue_on_a_site_with_the_first_form_of_the_hook(tmp_path, monkeypatch):
    """A program whose hook does not tell the frame of the site (``site_hook`` absent): words in the interface's
    language, heights from the start of the keel line."""
    site_plugins = pytest.importorskip("pyarchinit_photogrammetry.io.site_plugins",
                                       reason="the web site of a survey: program 1.10.0")
    monkeypatch.setattr(site_plugins, "_providers", [])  # only the plugin of the test: none left by others
    from pyarchinit_photogrammetry.plugins.api import Context

    from ppg_shipwreck import ShipwreckPlugin

    monkeypatch.delattr(Context, "site_hook", raising=False)
    doc, ch, _frame = _catalogue(tmp_path / "progetto")
    plugin = ShipwreckPlugin()
    context = Context("shipwreck", tmp_path / "plugin")
    before = len(site_plugins.providers())
    plugin.activate(context)
    try:
        assert len(site_plugins.providers()) == before + 1
        path = doc.export_site(tmp_path / "sito", products=["model"], model_faces=4000)
    finally:
        context.release()
    assert len(site_plugins.providers()) == before
    site = path.parent
    manifest = json.loads(path.read_text(encoding="utf-8"))
    chunk = manifest["chunks"][0]
    listed = {p["id"]: p for p in chunk["plugins"]}
    assert set(listed) == {"timbers", "catalogue"} and {p["plugin"] for p in listed.values()} == {"shipwreck"}
    assert listed["catalogue"]["title"] == {"it": "Catalogo dei legni", "en": "Timber catalogue"}
    layer = next(x for x in chunk["layers"] if x["id"] == "plugin:shipwreck:timbers")
    assert layer["url"] == "data/plugins/shipwreck/timbers.geojson" and layer["count"] == 5
    assert layer["group_by"] == "kind" and [(g["name"], g["count"]) for g in layer["groups"]][:2] == \
        [("chiglia", 1), ("ordinata", 2)]
    # the outlines moved into the local frame: one shift of whole metres for every vertex, small numbers
    feats = json.loads((site / layer["url"]).read_text(encoding="utf-8"))["features"]
    keel = np.array(timbers.timbers_of(ch)[0]["geometry"]["xy"])
    published = np.array(feats[0]["geometry"]["coordinates"])
    shift = keel - published
    assert np.allclose(shift, shift[0], atol=2e-3) and np.allclose(shift[0], np.round(shift[0]), atol=2e-3)
    assert np.abs(published).max() < 100
    table = json.loads((site / listed["catalogue"]["url"]).read_text(encoding="utf-8"))
    assert len(table["rows"]) == 5 and table["rows"][0]["kind"] == "chiglia"
    # nothing in the files of the plugin tells the place or the computer
    private = [str(Path.home()), str(tmp_path), str(Path(doc.path).parent)]
    for f in (site / "data/plugins/shipwreck").iterdir():
        text = f.read_text(encoding="utf-8")
        assert not any(p in text for p in private), f.name
        assert "EPSG" not in text and "/Users/" not in text and "\\Users\\" not in text
        assert all(abs(float(n)) < 1000 for n in re.findall(r"-?\d+\.\d+", text)), f.name


def _site(shift):
    site_plugins = pytest.importorskip("pyarchinit_photogrammetry.io.site_plugins")
    if not hasattr(site_plugins, "Site"):
        pytest.skip("the frame of the site: the second form of the program's hook")
    from pyarchinit_photogrammetry.io.site import _Shift

    return site_plugins.Site(_Shift(*shift), "local")


def test_with_the_frame_the_heights_of_the_site_and_both_languages(tmp_path):
    """Told the frame of the site, the heights are those of the site (the world's less its base), whatever the datum
    of the ship frame; the words in both languages, the legend matching them."""
    _d, ch, frame = _catalogue(tmp_path / "a")
    _d2, ch2, frame2 = _catalogue(tmp_path / "b", z0=0.0)
    on = _site((1000.0, 2000.0, 3.0))
    layer, table = wsite.contributions_with_site(ch, on)
    rows = {r["id"]: r for r in table["rows"]}
    world = {r["id"]: r for r in timbers.measure(ch)}
    assert rows["T2"]["z_max"] == pytest.approx(world["T2"]["z_max"] + frame.z0 - 3.0, abs=1e-3)
    rows2 = {r["id"]: r for r in wsite.contributions_with_site(ch2, on)[1]["rows"]}
    for key in ("z_min", "z_max"):
        assert rows2["T2"][key] == pytest.approx(rows["T2"][key], abs=1e-3)  # the datum changes nothing
    assert rows["T1"]["kind"] == {"it": "chiglia", "en": "keel"} and rows["T2"]["side"] == {"it": "babordo",
                                                                                            "en": "port"}
    assert rows["T4"]["loose"] == {"it": "sì", "en": "yes"} and set(rows["T5"]["status"]) == {"it", "en"}
    assert [i["value"] for i in layer["legend"]["items"]][:2] == ["chiglia", "ordinata"]
    assert layer["features"][0]["properties"]["kind"] == {"it": "chiglia", "en": "keel"}


def test_the_catalogue_on_a_site_in_its_heights(tmp_path, monkeypatch):
    site_plugins = pytest.importorskip("pyarchinit_photogrammetry.io.site_plugins",
                                       reason="the web site of a survey: program 1.10.0")
    monkeypatch.setattr(site_plugins, "_providers", [])  # only the plugin of the test: none left by others
    from pyarchinit_photogrammetry.plugins.api import Context

    from ppg_shipwreck import ShipwreckPlugin

    if getattr(Context, "site_hook", 1) < 2:
        pytest.skip("the frame of the site: the second form of the program's hook")
    doc, ch, frame = _catalogue(tmp_path / "progetto")
    plugin = ShipwreckPlugin()
    context = Context("shipwreck", tmp_path / "plugin")
    plugin.activate(context)
    try:
        assert isinstance(site_plugins.providers()[-1][1], site_plugins._TakesSite)
        path = doc.export_site(tmp_path / "sito", products=["model"], model_faces=4000)
    finally:
        context.release()
    site = path.parent
    manifest = json.loads(path.read_text(encoding="utf-8"))
    chunk = manifest["chunks"][0]
    listed = {p["id"]: p for p in chunk["plugins"]}
    layer = next(x for x in chunk["layers"] if x["id"] == "plugin:shipwreck:timbers")
    assert [(g["name"], g["count"]) for g in layer["groups"]][:2] == [("chiglia", 1), ("ordinata", 2)]
    table = json.loads((site / listed["catalogue"]["url"]).read_text(encoding="utf-8"))
    rows = {r["id"]: r for r in table["rows"]}
    assert rows["T1"]["kind"] == {"it": "chiglia", "en": "keel"}
    # the heights less one whole metre, the base of the heights of the site: the shift of the outlines, in z
    world = {r["id"]: r for r in timbers.measure(ch)}
    measured = [k for k in rows if rows[k].get("z_max") is not None]
    assert measured
    dz = {round(world[k]["z_max"] + frame.z0 - rows[k]["z_max"], 3) for k in measured}
    assert len(dz) == 1 and float(next(iter(dz))).is_integer()
    for f in (site / "data/plugins/shipwreck").iterdir():
        assert all(abs(float(n)) < 1000 for n in re.findall(r"-?\d+\.\d+", f.read_text(encoding="utf-8"))), f.name


def test_a_program_without_the_web_site(tmp_path, monkeypatch):
    """The program before 1.10.0 has no ``add_site_contributions``: the plugin starts all the same."""
    from pyarchinit_photogrammetry.plugins.api import Context

    from ppg_shipwreck import ShipwreckPlugin

    monkeypatch.delattr(Context, "add_site_contributions", raising=False)
    context = Context("shipwreck", tmp_path / "plugin")
    plugin = ShipwreckPlugin()
    plugin.activate(context)
    context.release()
