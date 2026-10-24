"""The em.json read without s3dgraphy, and the proxies of the chunk paired with its units."""
from __future__ import annotations

import json

import pytest
from samples import UNITS, emjson

from ppg_blender_em import emjson as em


def test_the_container_pyarchinit_writes(tmp_path):
    ids = emjson(tmp_path / "EM" / "Scavo.em.json", shift=(300000, 4500000, 100))
    matrix = em.load(tmp_path / "EM" / "Scavo.em.json")
    assert matrix.graph_id == "Scavo" and matrix.name == "Scavo"
    assert [u.name for u in matrix.units] == ["1.US1", "1.USM2", "1.US3"]
    assert {u.id: (u.area, u.tipo, u.us) for u in matrix.units} == ids
    assert matrix.units[0].epoch.name == "Tardoantico" and matrix.units[1].color == "#CC6633"
    assert matrix.shift == (300000.0, 4500000.0, 100.0)
    assert sorted(e.name for e in matrix.epochs) == ["Età romana", "Tardoantico"]


def test_a_single_graph_and_a_snapshot_of_the_sync(tmp_path):
    emjson(tmp_path / "old.em.json", container=False, epochs=False)
    matrix = em.load(tmp_path / "old.em.json")
    assert len(matrix.units) == 3 and matrix.shift is None and matrix.units[0].epoch is None
    doc = json.loads((tmp_path / "old.em.json").read_text(encoding="utf-8"))
    assert [u.name for u in em.read({"doc": doc}).units] == [u.name for u in matrix.units]


def test_not_an_em_json(tmp_path):
    (tmp_path / "x.json").write_text('{"header": {"format": "other"}, "graph": {}}', encoding="utf-8")
    with pytest.raises(em.EmJsonError):
        em.load(tmp_path / "x.json")
    (tmp_path / "y.json").write_text("not json", encoding="utf-8")
    with pytest.raises(em.EmJsonError):
        em.load(tmp_path / "y.json")


def test_the_shelf_holds_no_units(tmp_path):
    emjson(tmp_path / "p.em.json")
    doc = json.loads((tmp_path / "p.em.json").read_text(encoding="utf-8"))
    doc["graphs"]["shelf"] = {"graph_id": "shelf", "data": {"em_collection": "ShelfGraph"},
                              "nodes": [{"id": "x", "node_type": "US", "name": "US99"}], "edges": []}
    assert [u.name for u in em.read(doc).units] == ["1.US1", "1.USM2", "1.US3"]


def _entries(units=UNITS):
    return [{"name": f"{t}{u}", "us": u, "area": a, "tipo": t} for a, t, u in units]


def test_match_by_the_fields_of_the_form(tmp_path):
    emjson(tmp_path / "p.em.json")
    matrix = em.load(tmp_path / "p.em.json")
    found = em.match(_entries(), matrix)
    assert [(m.name, m.how) for m in found] == [("1.US1", "form"), ("1.USM2", "form"), ("1.US3", "form")]
    # the US 2 of the matrix is a USM: a US 2 is not it
    wrong = em.match([{"name": "US2", "us": "2", "area": "1", "tipo": "US"}], matrix)[0]
    assert wrong.unit is None and wrong.name == "US2"


def test_match_by_node_uuid_first_and_by_name(tmp_path):
    ids = emjson(tmp_path / "p.em.json")
    matrix = em.load(tmp_path / "p.em.json")
    by_form = {v: k for k, v in ids.items()}
    found = em.match(_entries(), matrix, node_uuids=by_form)
    assert [m.how for m in found] == ["node_uuid"] * 3
    # a matrix of EMtools: bare names, no fields of the form
    doc = json.loads((tmp_path / "p.em.json").read_text(encoding="utf-8"))
    for n in doc["graphs"]["Scavo"]["nodes"]:
        if n.get("node_type") == "US":
            n["name"] = n["name"].split(".", 1)[1]
            n.pop("data")
    found = em.match(_entries(), em.read(doc))
    assert [(m.name, m.how) for m in found] == [("US1", "name"), ("USM2", "name"), ("US3", "name")]


def test_two_candidates_are_said_not_guessed(tmp_path):
    emjson(tmp_path / "p.em.json", units=[("1", "US", "1"), ("2", "US", "1")])
    matrix = em.load(tmp_path / "p.em.json")
    m = em.match([{"name": "US1", "us": "1", "area": "", "tipo": "US"}], matrix)[0]
    assert m.unit is None and m.problem == "1.US1, 2.US1"
    # with the area of the proxy there is one
    m = em.match([{"name": "US1", "us": "1", "area": "2", "tipo": "US"}], matrix)[0]
    assert m.name == "2.US1"
    # a unit goes to one proxy only
    both = em.match([{"name": "US1", "us": "1", "area": "1", "tipo": "US"},
                     {"name": "US1", "us": "1", "area": "1", "tipo": "US"}], matrix)
    assert [m.unit is not None for m in both] == [True, False]


def test_an_em_json_of_s3dgraphy(tmp_path):
    """The file s3dgraphy 1.6 writes (the library of PyArchInit, EMStudio and EMtools) reads the same."""
    pytest.importorskip("s3dgraphy.exporter.emjson_exporter")
    from s3dgraphy.exporter.emjson_exporter import export_emjson
    from s3dgraphy.graph import Graph
    from s3dgraphy.nodes.epoch_node import EpochNode
    from s3dgraphy.utils.utils import get_stratigraphic_node_class

    g = Graph(graph_id="Scavo")
    g.name = "Scavo"
    for nid, name, kind, data in (("a", "1.US1", "US", {"us": "1", "area": "1", "unita_tipo": "US"}),
                                  ("b", "1.USVs101", "USVs", {"us": "101", "area": "1", "unita_tipo": "USVs"})):
        node = get_stratigraphic_node_class(kind)(node_id=nid, name=name, description=name)
        node.attributes = dict(data, node_uuid=nid)
        g.add_node(node)
    g.add_node(EpochNode(node_id="ep", name="Età romana", start_time=-100, end_time=300))
    g.add_edge("e1", "a", "b", "is_after")
    g.add_edge("e2", "a", "ep", "has_first_epoch")
    export_emjson(g, str(tmp_path / "Scavo.em.json"))
    matrix = em.load(tmp_path / "Scavo.em.json")
    assert [(u.id, u.name, u.node_type, u.tipo, u.us) for u in matrix.units] == [
        ("a", "1.US1", "US", "US", "1"), ("b", "1.USVs101", "USVs", "USVs", "101")]
    assert matrix.units[0].epoch.name == "Età romana" and matrix.units[1].epoch is None
