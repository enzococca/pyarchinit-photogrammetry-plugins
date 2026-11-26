"""The epochs of the em.json as groups of the unit volumes on the web site of a survey (site hook 3 of the
program, 2.1.0): which units go into which group, and nothing of the Blender scene on the site."""
from __future__ import annotations

import json

from samples import emjson

from ppg_blender_em import emjson as em
from ppg_blender_em.site import unit_groups


def test_epochs_become_groups(tmp_path):
    path = tmp_path / "EM" / "Scavo.em.json"
    emjson(path, units=[("1", "US", "1"), ("1", "USM", "2"), ("1", "US", "4")])
    [c] = unit_groups(em.load(path))
    assert c["kind"] == "unit_groups" and c["id"] == "epochs"
    assert c["title"] == {"it": "Epoche", "en": "Epochs"}
    assert [g["name"] for g in c["groups"]] == ["Età romana", "Tardoantico"]  # by start time (-100, 300)
    assert [g["colour"] for g in c["groups"]] == ["#CC6633", "#3366CC"]
    units = {g["name"]: [(u["tipo"], u["us"], u["area"]) for u in g["units"]] for g in c["groups"]}
    assert units == {"Tardoantico": [("US", "1", "1")], "Età romana": [("USM", "2", "1"), ("US", "4", "1")]}


def test_no_matrix_or_no_epochs_gives_nothing(tmp_path):
    assert unit_groups(None) == []
    path = tmp_path / "e.em.json"
    emjson(path, epochs=False)
    assert unit_groups(em.load(path)) == []


def _edit(path, change):
    doc = json.loads(path.read_text(encoding="utf-8"))
    change(doc["graphs"]["Scavo"])
    path.write_text(json.dumps(doc), encoding="utf-8")


def test_a_colour_that_is_not_rrggbb_is_left_to_the_program(tmp_path):
    path = tmp_path / "e.em.json"
    emjson(path)

    def change(graph):
        for n in graph["nodes"]:
            if n["id"] == "ep-roman":
                n["data"]["color"] = "red"
            elif n["id"] == "ep-late":
                n["data"].pop("color")
    _edit(path, change)
    [c] = unit_groups(em.load(path))
    assert [g["name"] for g in c["groups"]] == ["Età romana", "Tardoantico"]
    assert all("colour" not in g for g in c["groups"])


def test_units_without_a_number_or_an_epoch_are_left_out(tmp_path):
    path = tmp_path / "e.em.json"
    ids = emjson(path, units=[("1", "US", "1"), ("1", "USM", "2"), ("", "USVs", "")])

    def change(graph):
        bare = next(n for n in graph["nodes"] if n["node_type"] == "USVs")
        bare["name"], bare["data"] = "USVs", {}
        first = next(iter(ids))
        graph["edges"] = [e for e in graph["edges"] if e["source"] != first or e["edge_type"] != "has_first_epoch"]
    _edit(path, change)
    [c] = unit_groups(em.load(path))
    assert [(g["name"], [u["us"] for u in g["units"]]) for g in c["groups"]] == [("Età romana", ["2"])]


def test_an_area_unknown_is_not_sent(tmp_path):
    path = tmp_path / "e.em.json"
    emjson(path, units=[("", "US", "7"), ("1", "USM", "2")])
    [c] = unit_groups(em.load(path))
    units = [u for g in c["groups"] for u in g["units"]]
    assert {"tipo": "US", "us": "7"} in units


def test_two_epochs_of_one_name_in_two_graphs_stay_two_groups(tmp_path):
    path = tmp_path / "e.em.json"
    emjson(path, units=[("1", "US", "1"), ("1", "USM", "2")])
    doc = json.loads(path.read_text(encoding="utf-8"))
    other = json.loads(json.dumps(doc["graphs"]["Scavo"]).replace("ep-roman", "ep-roman-2").replace(
        "ep-late", "ep-late-2"))
    other["graph_id"] = other["name"] = "Saggio 2"
    for n in other["nodes"]:
        if n["node_type"] == "US":
            n["id"] = n["data"]["node_uuid"] = n["id"] + "-2"
            n["data"]["area"] = "2"
    for e in other["edges"]:
        e["source"], e["target"] = (x if x.startswith("ep-") else x + "-2" for x in (e["source"], e["target"]))
    doc["graphs"]["Saggio 2"] = other
    path.write_text(json.dumps(doc), encoding="utf-8")
    [c] = unit_groups(em.load(path))
    names = [g["name"] for g in c["groups"]]
    assert len(names) == 4 and len(set(names)) == 4
    assert sum(n.startswith("Età romana") for n in names) == 2
    areas = sorted(u["area"] for g in c["groups"] for u in g["units"])
    assert areas == ["1", "1", "2", "2"]


# ------------------------------------------------------------------------------------------------ in the program
import pytest  # noqa: E402

from pyarchinit_photogrammetry.plugins import Context  # noqa: E402

needs_hook_3 = pytest.mark.skipif(getattr(Context, "site_hook", 0) < 3,
                                  reason="the program has no site hook 3 (unit groups, 2.1.0)")


def _ours():
    from pyarchinit_photogrammetry.io import site_plugins

    return [p for p in site_plugins.providers() if p[0] == "blender-em"]


@needs_hook_3
def test_registered_only_with_hook_3(tmp_path, monkeypatch):
    from ppg_blender_em import BlenderEmPlugin

    before = len(_ours())
    context = Context("blender-em", tmp_path / "plugin")
    BlenderEmPlugin().activate(context)
    try:
        assert len(_ours()) == before + 1
    finally:
        context.release()
    assert len(_ours()) == before  # the plugin disabled: nothing of it left on the site
    monkeypatch.setattr(Context, "site_hook", 2)
    context = Context("blender-em", tmp_path / "plugin2")
    BlenderEmPlugin().activate(context)
    try:
        assert len(_ours()) == before
    finally:
        context.release()


@needs_hook_3
def test_an_unreadable_em_json_gives_nothing(tmp_path):
    from samples import project

    from ppg_blender_em import BlenderEmPlugin

    doc = project(tmp_path, model=False)
    plugin = BlenderEmPlugin()
    context = Context("blender-em", tmp_path / "plugin")
    plugin.activate(context)
    try:
        assert plugin.site_contributions(doc.chunk, None) == []  # no em.json linked
        other = tmp_path / "other.em.json"
        emjson(other)
        plugin.snapshot = em.load(other)  # what EMtools sent live is not the em.json of the chunk
        assert plugin.site_contributions(doc.chunk, None) == []
        bad = tmp_path / "EM" / "Scavo.em.json"
        bad.parent.mkdir(parents=True)
        bad.write_text("{not json", encoding="utf-8")
        plugin._store(doc.chunk, emjson=bad)
        assert plugin.site_contributions(doc.chunk, None) == []
    finally:
        context.release()


@pytest.fixture
def built_site(tmp_path):
    from samples import project

    from ppg_blender_em import BlenderEmPlugin

    doc = project(tmp_path, model=False)
    em_path = tmp_path / "Scavo" / "EM" / "Scavo.em.json"
    emjson(em_path, units=[("1", "US", "1"), ("1", "USM", "2")], shift=(300000.0, 4500000.0, 0.0))
    plugin = BlenderEmPlugin()
    context = Context("blender-em", tmp_path / "plugin")
    plugin.activate(context)
    try:
        plugin._store(doc.chunk, emjson=em_path)
        doc.save()
        manifest = doc.export_site(tmp_path / "site", products=("proxies", "units"))
    finally:
        context.release()
    return manifest.parent, json.loads(manifest.read_text(encoding="utf-8"))


@needs_hook_3
def test_site_volumes_coloured_by_epoch(built_site):
    site, manifest = built_site
    data = json.loads((site / "data" / "proxies.json").read_text(encoding="utf-8"))
    us1 = next(v for v in data["volumes"] if v["us"] == "1")
    usm2 = next(v for v in data["volumes"] if v["us"] == "2")
    us3 = next(v for v in data["volumes"] if v["us"] == "3")
    assert (us1["group"], us1["colour"]) == ("Tardoantico", "#3366cc")
    assert (usm2["group"], usm2["colour"]) == ("Età romana", "#cc6633")
    assert us3["group"] is None  # not in the matrix: the program's colour, *Senza gruppo*
    assert [(g["name"], g["count"]) for g in data["groups"]] == [("Età romana", 1), ("Tardoantico", 1), ("", 1)]
    layer = next(x for c in manifest["chunks"] for x in c["layers"] if x["id"] == "volumes")
    assert layer["groups"] == data["groups"]


@needs_hook_3
def test_nothing_of_the_blender_scene_in_the_site(built_site):
    site, _manifest = built_site
    words = ("blender_em", "blender-em", "ppg_scene", "em.json", "Scavo/EM", "EPSG", "300000", "4500000")
    for f in site.rglob("*"):
        if "vendor" not in f.relative_to(site).parts and f.is_file() and f.suffix in (".json", ".js", ".html", ".geojson", ".txt", ".csv"):
            text = f.read_text(encoding="utf-8", errors="replace")
            found = [w for w in words if w in text]
            assert not found, (f.relative_to(site), found)
    assert not (site / "data" / "plugins").exists()
