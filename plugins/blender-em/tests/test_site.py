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
